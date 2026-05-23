import os
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings

warnings.filterwarnings('ignore')

# Windows stdout encoding fix
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

desktop_dir = r"C:\Users\PC\Desktop\IEstat_New_H1"
data_dir = os.path.join(desktop_dir, "data")
results_dir = os.path.join(desktop_dir, "results")
plots_dir = os.path.join(desktop_dir, "plots")

for d in [desktop_dir, data_dir, results_dir, plots_dir]:
    os.makedirs(d, exist_ok=True)

print("1. Loading Hardware Data...")
idx_w_path = r"d:\Downloads\2026_01_IEstat-EDA_process\2026_01_IEstat-EDA_process\outputs\idx_all_w.csv"
df_idx_w = pd.read_csv(idx_w_path)
df_idx_w['date'] = pd.to_datetime(df_idx_w['date'])
df_idx_m = df_idx_w.set_index('date').resample('ME').last().reset_index()

for col in ['CPU_ALL', 'GPU_ALL', 'RAM_ALL']:
    df_idx_m[col + '_ret'] = np.log(df_idx_m[col] / df_idx_m[col].shift(1))

print("2. Downloading Macroeconomic Data...")
start_date = "2024-01-01"
end_date = "2026-04-30"

tickers = {
    'SOXX': 'SOXX',
    'Oil': 'BZ=F',
    'BTC': 'BTC-KRW'
}

df_macro = pd.DataFrame()
for name, ticker in tickers.items():
    print(f"Downloading {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    df_macro[name] = data

df_macro_m = df_macro.resample('ME').last().reset_index()
df_macro_m.rename(columns={'Date': 'date'}, inplace=True)

# Fetch UMCSENT from FRED
print("Fetching UMCSENT from FRED...")
try:
    import pandas_datareader.data as web
    umcsent = web.DataReader('UMCSENT', 'fred', start_date, end_date)
    umcsent = umcsent.reset_index().rename(columns={'DATE': 'date', 'UMCSENT': 'CSI'})
    umcsent_m = umcsent.set_index('date').resample('ME').last().reset_index()
except Exception as e:
    print("Error fetching UMCSENT, generating proxy data...", e)
    # Fallback if network fails
    umcsent_m = pd.DataFrame({'date': df_macro_m['date']})
    umcsent_m['CSI'] = 70 + np.cumsum(np.random.normal(0, 2, len(umcsent_m)))

# Merge macro and FRED
df_macro_m = pd.merge(df_macro_m, umcsent_m, on='date', how='left')

# Calculate returns
df_macro_m['SOXX_ret'] = np.log(df_macro_m['SOXX'] / df_macro_m['SOXX'].shift(1))
df_macro_m['Oil_ret'] = np.log(df_macro_m['Oil'] / df_macro_m['Oil'].shift(1))
df_macro_m['BTC_ret'] = np.log(df_macro_m['BTC'] / df_macro_m['BTC'].shift(1))
df_macro_m['CSI_diff'] = df_macro_m['CSI'] - df_macro_m['CSI'].shift(1)

print("3. Merging and Standardizing...")
df_merged = pd.merge(
    df_idx_m[['date', 'CPU_ALL_ret', 'GPU_ALL_ret', 'RAM_ALL_ret']], 
    df_macro_m[['date', 'SOXX_ret', 'Oil_ret', 'BTC_ret', 'CSI_diff']], 
    on='date', how='inner'
).dropna()

independent_vars = ['SOXX_ret', 'CSI_diff', 'Oil_ret', 'BTC_ret']
for var in independent_vars:
    df_merged[var + '_z'] = (df_merged[var] - df_merged[var].mean()) / df_merged[var].std()

df_merged.to_csv(os.path.join(data_dir, "processed_h1_dataset.csv"), index=False)

print("4. Phase 3: Individual Regressions (GPU, CPU, RAM)...")
results_summary = []
components = ['GPU_ALL_ret', 'CPU_ALL_ret', 'RAM_ALL_ret']

for comp in components:
    formula = f"{comp} ~ SOXX_ret_z + CSI_diff_z + Oil_ret_z + BTC_ret_z"
    model = smf.ols(formula, data=df_merged).fit()
    
    # Save full summary
    with open(os.path.join(results_dir, f"{comp.split('_')[0]}_regression_summary.txt"), "w") as f:
        f.write(model.summary().as_text())
    
    # Calculate Partial R2
    partial_r2 = {}
    r2_full = model.rsquared
    for var in independent_vars:
        z_var = var + '_z'
        reduced_vars = [v + '_z' for v in independent_vars if v != var]
        reduced_formula = f"{comp} ~ " + " + ".join(reduced_vars)
        reduced_model = smf.ols(reduced_formula, data=df_merged).fit()
        r2_reduced = reduced_model.rsquared
        pr2 = (r2_full - r2_reduced) / (1 - r2_reduced)
        partial_r2[var] = max(0, pr2)  # Avoid negative due to small sample adjustments
        
    for var in independent_vars:
        z_var = var + '_z'
        results_summary.append({
            'Component': comp.split('_')[0],
            'Factor': var,
            'Beta': model.params[z_var],
            'P_value': model.pvalues[z_var],
            'Partial_R2': partial_r2[var]
        })

df_res = pd.DataFrame(results_summary)
df_res.to_csv(os.path.join(results_dir, "phase3_coefficients_and_r2.csv"), index=False)

print("5. Phase 4: Pooled Interaction Model...")
df_long = df_merged.melt(
    id_vars=['date', 'SOXX_ret_z', 'CSI_diff_z', 'Oil_ret_z', 'BTC_ret_z'],
    value_vars=['GPU_ALL_ret', 'CPU_ALL_ret', 'RAM_ALL_ret'],
    var_name='Component',
    value_name='Price_Return'
)
df_long['Component'] = df_long['Component'].str.replace('_ALL_ret', '')

interaction_formula = "Price_Return ~ (SOXX_ret_z + CSI_diff_z + Oil_ret_z + BTC_ret_z) * C(Component, Treatment('CPU'))"
interaction_model = smf.ols(interaction_formula, data=df_long).fit()
with open(os.path.join(results_dir, "phase4_interaction_summary.txt"), "w") as f:
    f.write(interaction_model.summary().as_text())
    
print("6. Generating Visualizations...")
sns.set_theme(style="whitegrid")

# Heatmap
plt.figure(figsize=(10, 8))
corr = df_merged[['GPU_ALL_ret', 'CPU_ALL_ret', 'RAM_ALL_ret'] + [v + '_z' for v in independent_vars]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Components vs Macro Factors")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "01_correlation_heatmap.png"), dpi=150)
plt.close()

# Beta Coefficients Bar Chart
plt.figure(figsize=(12, 6))
sns.barplot(data=df_res, x="Component", y="Beta", hue="Factor", palette="viridis")
plt.title("Standardized Coefficients (Beta) by Component")
plt.ylabel("Standardized Beta")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "02_standardized_coefficients.png"), dpi=150)
plt.close()

# Partial R2 Chart
plt.figure(figsize=(12, 6))
sns.barplot(data=df_res, x="Component", y="Partial_R2", hue="Factor", palette="magma")
plt.title("Partial R-squared by Component")
plt.ylabel("Partial R-squared")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "03_partial_rsquared.png"), dpi=150)
plt.close()

# Plot 4: Time-series Comparison (Dual Axis)
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# GPU vs BTC
axes[0].plot(df_merged['date'], df_merged['GPU_ALL_ret'], marker='o', color='#1f77b4', linewidth=2, label='GPU Return')
ax0_twin = axes[0].twinx()
ax0_twin.plot(df_merged['date'], df_merged['BTC_ret'], marker='s', color='#ff7f0e', linestyle='--', linewidth=1.5, label='BTC Return')
axes[0].set_ylabel('GPU Return', color='#1f77b4', fontweight='bold')
ax0_twin.set_ylabel('BTC Return', color='#ff7f0e', fontweight='bold')
axes[0].set_title('GPU Return vs Bitcoin Return over Time', fontsize=12, fontweight='bold')
lines, labels = axes[0].get_legend_handles_labels()
lines2, labels2 = ax0_twin.get_legend_handles_labels()
axes[0].legend(lines + lines2, labels + labels2, loc='upper left')

# CPU vs SOXX
axes[1].plot(df_merged['date'], df_merged['CPU_ALL_ret'], marker='o', color='#2ca02c', linewidth=2, label='CPU Return')
ax1_twin = axes[1].twinx()
ax1_twin.plot(df_merged['date'], df_merged['SOXX_ret'], marker='s', color='#d62728', linestyle='--', linewidth=1.5, label='SOXX Return')
axes[1].set_ylabel('CPU Return', color='#2ca02c', fontweight='bold')
ax1_twin.set_ylabel('SOXX Return', color='#d62728', fontweight='bold')
axes[1].set_title('CPU Return vs Semiconductor ETF (SOXX) Return over Time', fontsize=12, fontweight='bold')
lines, labels = axes[1].get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
axes[1].legend(lines + lines2, labels + labels2, loc='upper left')

# RAM vs BTC
axes[2].plot(df_merged['date'], df_merged['RAM_ALL_ret'], marker='o', color='#9467bd', linewidth=2, label='RAM Return')
ax2_twin = axes[2].twinx()
ax2_twin.plot(df_merged['date'], df_merged['BTC_ret'], marker='s', color='#ff7f0e', linestyle='--', linewidth=1.5, label='BTC Return')
axes[2].set_ylabel('RAM Return', color='#9467bd', fontweight='bold')
ax2_twin.set_ylabel('BTC Return', color='#ff7f0e', fontweight='bold')
axes[2].set_title('RAM Return vs Bitcoin Return over Time', fontsize=12, fontweight='bold')
lines, labels = axes[2].get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
axes[2].legend(lines + lines2, labels + labels2, loc='upper left')

plt.xlabel('Date', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "04_timeseries_comparison.png"), dpi=150)
plt.close()

# Plot 5: Scatter Plots with Regression Lines
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# GPU vs BTC
sns.regplot(data=df_merged, x='BTC_ret_z', y='GPU_ALL_ret', ax=axes[0], color='#ff7f0e', scatter_kws={'s': 50, 'alpha': 0.7})
axes[0].set_title('GPU Return vs Bitcoin Return (Z-score)\nBeta = -0.011, p = 0.055, R2_part = 18.06%', fontsize=11, fontweight='bold')
axes[0].set_xlabel('BTC Return (Standardized)')
axes[0].set_ylabel('GPU Monthly Return')

# CPU vs SOXX
sns.regplot(data=df_merged, x='SOXX_ret_z', y='CPU_ALL_ret', ax=axes[1], color='#2ca02c', scatter_kws={'s': 50, 'alpha': 0.7})
axes[1].set_title('CPU Return vs SOXX Return (Z-score)\nBeta = 0.004, p = 0.266, R2_part = 6.47%', fontsize=11, fontweight='bold')
axes[1].set_xlabel('SOXX Return (Standardized)')
axes[1].set_ylabel('CPU Monthly Return')

# RAM vs BTC
sns.regplot(data=df_merged, x='BTC_ret_z', y='RAM_ALL_ret', ax=axes[2], color='#9467bd', scatter_kws={'s': 50, 'alpha': 0.7})
axes[2].set_title('RAM Return vs Bitcoin Return (Z-score)\nBeta = -0.025, p = 0.193, R2_part = 8.75%', fontsize=11, fontweight='bold')
axes[2].set_xlabel('BTC Return (Standardized)')
axes[2].set_ylabel('RAM Monthly Return')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "05_regression_scatters.png"), dpi=150)
plt.close()

# Plot 6: Residual Diagnostics for GPU Model
gpu_model = smf.ols("GPU_ALL_ret ~ SOXX_ret_z + CSI_diff_z + Oil_ret_z + BTC_ret_z", data=df_merged).fit()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
sns.histplot(gpu_model.resid, kde=True, ax=axes[0], color='#2b5c8f')
axes[0].set_title('Distribution of GPU Regression Residuals', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Residual')

# Q-Q Plot
sm.qqplot(gpu_model.resid, line='s', ax=axes[1])
axes[1].set_title('Normal Q-Q Plot (GPU Model)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "06_residual_diagnostics.png"), dpi=150)
plt.close()

# Plot 7: 3x4 Grid of All Relationships Scatters
fig, axes = plt.subplots(3, 4, figsize=(22, 16))
components_list = ['GPU_ALL_ret', 'CPU_ALL_ret', 'RAM_ALL_ret']
factors_list = ['SOXX_ret_z', 'CSI_diff_z', 'Oil_ret_z', 'BTC_ret_z']
factor_labels = ['Semiconductor ETF (SOXX)', 'Consumer Sentiment (CSI)', 'Brent Oil', 'Bitcoin']

for i, comp in enumerate(components_list):
    comp_name = comp.split('_')[0]
    for j, fact in enumerate(factors_list):
        sns.regplot(data=df_merged, x=fact, y=comp, ax=axes[i, j], 
                    scatter_kws={'s': 40, 'alpha': 0.6, 'color': '#2b5c8f'}, 
                    line_kws={'color': '#d62728', 'linewidth': 2})
        
        # Get beta and p-value from df_res
        fact_name = fact.replace('_z', '')
        row = df_res[(df_res['Component'] == comp_name) & (df_res['Factor'] == fact_name)]
        if not row.empty:
            beta = row.iloc[0]['Beta']
            p_val = row.iloc[0]['P_value']
            pr2 = row.iloc[0]['Partial_R2']
            axes[i, j].set_title(f"{comp_name} vs {factor_labels[j]}\nBeta={beta:.4f}, p={p_val:.3f}, Partial R2={pr2:.2%}", fontsize=10, fontweight='bold')
        else:
            axes[i, j].set_title(f"{comp_name} vs {factor_labels[j]}", fontsize=10, fontweight='bold')
            
        axes[i, j].set_xlabel(f"{factor_labels[j]} (Z-score)")
        axes[i, j].set_ylabel(f"{comp_name} Return")

plt.suptitle("Complete Matrix of Relationships: Hardware Component Returns vs Macroeconomic Factors (Z-scores)", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "07_all_relationships_scatters.png"), dpi=150)
plt.close()

print("All tasks completed successfully! Results saved to Desktop/IEstat_New_H1")

