# New H1 Analysis: Comprehensive Report

## 1. Objective and Hypothesis
This analysis rigorously evaluates the **New Hypothesis 1 (H1)** proposed by the research team:
> *Do Financial and Macroeconomic Indicators (Semiconductor ETF, Consumer Sentiment Index, International Oil Price, and Bitcoin) affect component prices (GPU, CPU, and RAM) differently without considering time lag?*

The objective is to measure the contemporaneous impact of these four macro factors on three hardware components and formally test if their sensitivities significantly differ.

---

## 2. Data Preparation and Synchronization

### 2.1 Hardware Component Data
- **Source**: `idx_all_w.csv` (Chain Matched Jevons Index).
- **Processing**: We loaded the weekly price indices for GPU, CPU, and RAM. To align with macroeconomic data, the hardware indices were downsampled to a **monthly frequency** (taking the last observed value of each month). Log returns were calculated to ensure stationarity and capture month-over-month price elasticity.

### 2.2 Macroeconomic Data
We downloaded external macroeconomic data corresponding to the exact timeframe (March 2024 - April 2026):
1. **Semiconductor ETF (`SOXX`)**: Represents the broad semiconductor industry sentiment. Fetched via `yfinance`.
2. **Bitcoin (`BTC-KRW`)**: Represents crypto market sentiment and risk-on asset flow. Fetched via `yfinance`.
3. **International Oil Price (`BZ=F` - Brent Crude)**: Represents logistical costs and global macro shifts. Fetched via `yfinance`.
4. **Consumer Sentiment Index (`UMCSENT`)**: The University of Michigan Consumer Sentiment Index was fetched via the FRED API (using `pandas_datareader`) as a monthly global consumer sentiment proxy.

### 2.4 Time Period Verification
> [!NOTE]
> **Strict Time Synchronization:**
> The analysis timeframe is rigorously synchronized from **March 2024 to March 2026** (note that the first week of April 2026, `2026-04-05`, in the raw dataset `idx_all_w.csv` contains entirely missing values - NaNs, hence the clean series terminates on **March 31, 2026**).
> 
> Due to the monthly log difference return calculation, the first month (March 2024) is dropped. The OLS regression models are therefore estimated on a continuous series from **April 30, 2024 to March 31, 2026** (24 complete monthly observations for each component), which guarantees econometric consistency.

---

## 3. Modeling and Empirical Results

### Phase 3: Component-Specific Regressions
We ran three separate OLS models for GPU, CPU, and RAM log returns against the 4 macro factors.

#### Key Findings (Standardized Coefficients & Partial $R^2$):
* **GPU**: 
  * Bitcoin (`BTC_ret`) holds the strongest explanatory power (**Partial $R^2$ = 18.06%**), with a coefficient of -0.011 (p-value = 0.054). This approaches statistical significance, indicating a moderate inverse contemporaneous relationship during the sample period.
  * ETF, CSI, and Oil showed negligible explanatory power (< 2% Partial $R^2$).
* **CPU**:
  * Semiconductor ETF (`SOXX_ret`) is the strongest factor (**Partial $R^2$ = 6.47%**), though statistically insignificant (p=0.26).
* **RAM**:
  * Bitcoin and SOXX explain ~8.7% and 2.8% of the variance respectively, but neither coefficient is statistically significant.

**Intermediate Conclusion**: Visually and heuristically, Bitcoin heavily dominates GPU variance, while Semiconductor ETF has a stronger relative linkage to CPUs.

### Phase 4: Pooled Interaction Model
To formally test if these differences in impact are statistically robust, we pooled all components into a Long-format dataset (72 observations) and ran a Linear Mixed-Effects / Interaction Model with CPU as the baseline:
$$\text{Price Return} = (\text{Factors}) \times \text{Component}$$

#### Key Findings (Difference Testing):
* **BTC x GPU Interaction**: The coefficient is -0.0094 with a **p-value of 0.560**.
* **SOXX x GPU Interaction**: The coefficient is -0.0012 with a **p-value of 0.947**.
* **CSI and Oil Interactions**: All p-values > 0.60.

> [!WARNING] 
> **Final Conclusion on H1**
> The interaction terms completely failed to reject the null hypothesis. This means that while Phase 3 hints that Bitcoin explains more GPU variance than CPU variance, the sample size and noise level strictly prevent us from claiming that **"the components are impacted differently"** with statistical significance. The macroeconomic factors affect the components relatively symmetrically, or the variance is driven predominantly by unobserved component-specific shocks rather than these 4 macro factors.

---

## 4. Deliverables and Operation
All operations have been fully automated and outputs are neatly organized on your Desktop at `C:\Users\PC\Desktop\IEstat_New_H1`.

### Folder Structure Created
1. **`/data/`**: 
   - `processed_h1_dataset.csv`: The final, time-aligned, and standardized dataset ready for dashboarding or further ML models.
2. **`/results/`**: 
   - `phase3_coefficients_and_r2.csv`: A clean table of all betas, p-values, and Partial $R^2$ values.
   - `GPU_regression_summary.txt` (and CPU/RAM): Raw Statsmodels OLS outputs.
   - `phase4_interaction_summary.txt`: The formal interaction test output.
3. **`/plots/`**: 
   - `01_correlation_heatmap.png`: Showing raw variable correlations.
   - `02_standardized_coefficients.png`: Grouped bar chart comparing sensitivities.
   - `03_partial_rsquared.png`: Chart highlighting that BTC dominates GPU variance.
   - `04_timeseries_comparison.png` **[NEW]**: A beautiful dual-axis time-series chart showing co-movement of each component with its most prominent macro factor over time.
   - `05_regression_scatters.png` **[NEW]**: Scatter plots with OLS regression lines for key relationships (GPU vs. BTC, CPU vs. SOXX, RAM vs. BTC) showing the slope and data distribution.
   - `06_residual_diagnostics.png` **[NEW]**: Distribution and normal Q-Q plot of the OLS regression residuals for the GPU model to check econometric assumptions.
   - `07_all_relationships_scatters.png` **[NEW]**: A comprehensive 12-panel scatter matrix (3 components x 4 macro factors) showing all regression relationships with annotated Betas, p-values, and Partial $R^2$ values.

### Execution Details
- The process was executed via a unified Python script (`run_h1_analysis.py`).
- The script automatically handled all `yfinance` API rate limits, handled FRED API data synchronization, performed inner joins strictly on overlapping dates to prevent data leakage, and applied the correct degrees-of-freedom adjustments when calculating Partial $R^2$.

