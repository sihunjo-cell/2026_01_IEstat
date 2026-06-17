# Auto-generated from Hypo_1_final.ipynb (code cells only, markdown/outputs stripped)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.tsa.stattools import adfuller, coint

from statsmodels.tsa.api import VAR

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.stats.diagnostic import acorr_ljungbox

OUT = Path("outputs")

H1_OUT = OUT / "hypo1"
H1_OUT.mkdir(parents=True, exist_ok=True)

MAX_LAG = 8      # BTC 반영 시차: 0~8주
HAC_LAG = 4      # Newey-West HAC 표준오차 lag

GPU_RET_COL = "GPU_ALL"
MKT_RET_COLS = ["btc_krw_ret", "fx_ret", "nvda_ret", "nasdaq_ret", "sox_ret"]

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "MalGun Gothic"
plt.rcParams["axes.unicode_minus"] = False

import re as _re
from datetime import datetime as _dt

FIG_DIR = Path("output_pic") / "hypo_1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def _safe_fname(text, max_len=120):
    text = str(text).strip()
    text = _re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = _re.sub(r"\s+", "_", text)
    text = _re.sub(r"_+", "_", text).strip("._ ")
    if not text:
        text = "plot"
    return text[:max_len]

def _infer_title(fig=None):
    fig = fig or plt.gcf()
    if getattr(fig, "_suptitle", None) is not None:
        t = fig._suptitle.get_text()
        if t:
            return t
    for ax in fig.axes:
        t = ax.get_title()
        if t:
            return t
    for ax in fig.axes:
        ylabel = ax.get_ylabel()
        xlabel = ax.get_xlabel()
        if ylabel or xlabel:
            return f"{ylabel}_vs_{xlabel}".strip("_vs_")
    return "plot_" + _dt.now().strftime("%Y%m%d_%H%M%S")

def save_plot(name=None, folder=None, dpi=300):
    folder = Path(folder) if folder is not None else FIG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    title = name or _infer_title(fig)
    stem = _safe_fname(title)
    path = folder / f"{stem}.png"
    i = 2
    while path.exists():
        path = folder / f"{stem}_{i:02d}.png"
        i += 1
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[saved plot] {path}")
    return path

ret_all_w = (
    pd.read_csv(OUT / "ret_all_w.csv", parse_dates=["date"])
    .set_index("date")
    .sort_index()
)

idx_all_w = (
    pd.read_csv(OUT / "idx_all_w.csv", parse_dates=["date"])
    .set_index("date")
    .sort_index()
)

import yfinance as yf

MKT_TICKERS = {
    "BTC-KRW": "btc_krw",
    "ETC-KRW": "etc_krw",
    "KRW=X": "usdkrw",
    "NVDA": "nvda",
    "^IXIC": "nasdaq",
    "^SOX": "sox",
}

START_DATE = "2024-03-01"
END_DATE = "2026-03-31"

mkt_px = yf.download(
    list(MKT_TICKERS.keys()),
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False,
)["Close"].rename(columns=MKT_TICKERS)

WEEKLY_FREQ = "W-SUN"

mkt_log_level_w = np.log(mkt_px).resample(WEEKLY_FREQ).mean()
mkt_log_level_w = mkt_log_level_w.loc[pd.to_datetime(START_DATE):pd.to_datetime(END_DATE)]

mkt_w = pd.DataFrame(index=mkt_log_level_w.index)
mkt_w["btc_krw_ret"] = mkt_log_level_w["btc_krw"].diff()
mkt_w["etc_krw_ret"] = mkt_log_level_w["etc_krw"].diff()
mkt_w["fx_ret"] = mkt_log_level_w["usdkrw"].diff()
mkt_w["nvda_ret"] = mkt_log_level_w["nvda"].diff()
mkt_w["nasdaq_ret"] = mkt_log_level_w["nasdaq"].diff()
mkt_w["sox_ret"] = mkt_log_level_w["sox"].diff()

mkt_w.to_csv(OUT / "mkt_w.csv")
mkt_log_level_w.to_csv(OUT / "mkt_log_level_w.csv")

print("ret_all_w")
display(ret_all_w.head())

print("market weekly returns")
display(mkt_w.head())

print("market weekly log levels")
display(mkt_log_level_w.head())

print("saved:", OUT / "mkt_w.csv", OUT / "mkt_log_level_w.csv")

plt.figure(figsize=(12, 5))
plt.plot(mkt_w.index, mkt_w["btc_krw_ret"], linestyle="--", label="BTC-KRW weekly log return")
plt.plot(mkt_w.index, mkt_w["etc_krw_ret"], alpha=0.8, label="ETC-KRW weekly log return")
plt.axhline(0, linewidth=1)
plt.title("BTC-KRW and ETC-KRW weekly log returns")
plt.xlabel("Date")
plt.ylabel("Weekly log return")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

etc_eda = (
    ret_all_w[[GPU_RET_COL]]
    .rename(columns={GPU_RET_COL: "gpu_ret"})
    .join(mkt_w[["btc_krw_ret", "etc_krw_ret"]], how="inner")
    .dropna()
    .copy()
)

crypto_corr = etc_eda[["btc_krw_ret", "etc_krw_ret"]].corr().loc["btc_krw_ret", "etc_krw_ret"]

etc_lag_rows = []
for k in range(MAX_LAG + 1):
    col = f"etc_krw_ret_l{k}"
    etc_eda[col] = etc_eda["etc_krw_ret"].shift(k)
    temp = etc_eda[["gpu_ret", col]].dropna()
    etc_lag_rows.append({
        "lag_week": k,
        "corr_gpu_etc": temp["gpu_ret"].corr(temp[col]),
        "n": len(temp),
    })

etc_lag_corr = pd.DataFrame(etc_lag_rows)
etc_max_row = etc_lag_corr.loc[etc_lag_corr["corr_gpu_etc"].abs().idxmax()]

etc_eda_summary = pd.DataFrame([
    {
        "metric": "corr_btc_etc_return",
        "value": crypto_corr,
        "interpretation": "BTC-KRW와 ETC-KRW가 같은 crypto sentiment를 공유하는지 확인",
    },
    {
        "metric": "max_abs_corr_gpu_etc_lag_week",
        "value": int(etc_max_row["lag_week"]),
        "interpretation": "GPU return과 ETC-KRW lag return의 절대상관이 가장 큰 시차",
    },
    {
        "metric": "max_abs_corr_gpu_etc_value",
        "value": float(etc_max_row["corr_gpu_etc"]),
        "interpretation": "해당 시차에서의 GPU-ETC 단순 상관계수",
    },
])

print("BTC-KRW vs ETC-KRW return correlation")
print(round(crypto_corr, 4))

print("ETC-KRW lag correlation with GPU return")
display(etc_lag_corr.round(4))

print("ETC-KRW EDA summary")
display(etc_eda_summary)

etc_lag_corr.to_csv(H1_OUT / "etc_lag.csv", index=False)
etc_eda_summary.to_csv(H1_OUT / "etc_sum.csv", index=False)

plt.figure(figsize=(9, 4))
plt.plot(etc_lag_corr["lag_week"], etc_lag_corr["corr_gpu_etc"], marker="o")
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("EDA: correlation between GPU return and lagged ETC-KRW return")
plt.xlabel("ETC-KRW return lag, weeks")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

rolling_corr = etc_eda["btc_krw_ret"].rolling(8).corr(etc_eda["etc_krw_ret"])
plt.figure(figsize=(10, 4))
plt.plot(rolling_corr.index, rolling_corr)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("EDA: 8-week rolling correlation between BTC-KRW and ETC-KRW returns")
plt.xlabel("Date")
plt.ylabel("Rolling correlation")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

h1 = (
    ret_all_w[[GPU_RET_COL]]
    .rename(columns={GPU_RET_COL: "gpu_ret"})
    .join(mkt_w[MKT_RET_COLS], how="inner")
    .sort_index()
)

h1["gpu_ret_l1"] = h1["gpu_ret"].shift(1)

for k in range(MAX_LAG + 1):
    h1[f"btc_krw_ret_l{k}"] = h1["btc_krw_ret"].shift(k)

print("Hypo 1 regression dataset")
display(h1.head(12))

btc_lag_cols = [f"btc_krw_ret_l{k}" for k in range(MAX_LAG + 1)]
reg_cols = ["gpu_ret"] + btc_lag_cols + ["fx_ret", "nvda_ret", "gpu_ret_l1"]

sample_h1 = pd.DataFrame({
    "item": ["raw_join_weeks", "complete_regression_weeks"],
    "n": [len(h1), h1[reg_cols].dropna().shape[0]],
})

print("sample size check")
display(sample_h1)

h1.to_csv(H1_OUT / "reg.csv")
sample_h1.to_csv(H1_OUT / "sample.csv", index=False)

lag_na_rows = []
base_cols = ["gpu_ret", "btc_krw_ret", "fx_ret", "nvda_ret", "gpu_ret_l1"]

for k in range(MAX_LAG + 1):
    lag_col = f"btc_krw_ret_l{k}"
    needed = ["gpu_ret", lag_col, "fx_ret", "nvda_ret", "gpu_ret_l1"]
    temp = h1[needed]
    lag_na_rows.append({
        "lag_week": k,
        "raw_weeks": len(temp),
        "gpu_ret_nonnull": int(temp["gpu_ret"].notna().sum()),
        "btc_lag_nonnull": int(temp[lag_col].notna().sum()),
        "complete_weeks_for_single_lag": int(temp.dropna().shape[0]),
        "dropped_weeks_for_single_lag": int(len(temp) - temp.dropna().shape[0]),
    })

lag_na_check = pd.DataFrame(lag_na_rows)

full_complete_n = h1[reg_cols].dropna().shape[0]
lag_na_summary = pd.DataFrame([{
    "raw_join_weeks": len(h1),
    "complete_regression_weeks_all_lags": full_complete_n,
    "dropped_weeks_all_lags": len(h1) - full_complete_n,
    "ffill_used": False,
}])

print("Lag missingness check")
display(lag_na_check)
display(lag_na_summary)

lag_na_check.to_csv(H1_OUT / "lag_na.csv", index=False)
lag_na_summary.to_csv(H1_OUT / "lag_na_sum.csv", index=False)

lag_corr_rows = []

for k in range(MAX_LAG + 1):
    col = f"btc_krw_ret_l{k}"
    temp = h1[["gpu_ret", col]].dropna()
    lag_corr_rows.append({
        "lag_week": k,
        "corr_gpu_btc": temp["gpu_ret"].corr(temp[col]),
        "n": len(temp),
    })

lag_corr = pd.DataFrame(lag_corr_rows)
max_corr_row = lag_corr.loc[lag_corr["corr_gpu_btc"].abs().idxmax()]

print("BTC lag별 GPU 수익률과의 상관계수")
display(lag_corr.round(4))
print(f"max abs correlation lag = {int(max_corr_row['lag_week'])} week, corr = {max_corr_row['corr_gpu_btc']:.4f}")

lag_corr.to_csv(H1_OUT / "lag_corr.csv", index=False)

plt.figure(figsize=(9, 4))
plt.plot(lag_corr["lag_week"], lag_corr["corr_gpu_btc"], marker="o")
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Hypo 1: correlation between GPU return and lagged BTC-KRW return")
plt.xlabel("BTC-KRW return lag, weeks")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

ccf = lag_corr.copy()
n_eff = int(ccf["n"].min())
ccf_band = 1.96 / np.sqrt(n_eff)
ccf["conf_band_95"] = ccf_band

print(f"Approximate 95% confidence band: ±{ccf_band:.4f}, n_eff={n_eff}")
display(ccf.round(4))
ccf.to_csv(H1_OUT / "ccf.csv", index=False)

plt.figure(figsize=(9, 4))
markerline, stemlines, baseline = plt.stem(ccf["lag_week"], ccf["corr_gpu_btc"])
plt.setp(stemlines, linewidth=1)
plt.axhline(0, linestyle="--", linewidth=1)
plt.axhline(ccf_band, linestyle=":", linewidth=1)
plt.axhline(-ccf_band, linestyle=":", linewidth=1)
plt.title("CCF-style lag correlation: BTC-KRW return vs GPU return")
plt.xlabel("BTC-KRW return lag, weeks")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

roll_base = h1[["gpu_ret", "btc_krw_ret"]].dropna().copy()
roll = pd.DataFrame(index=roll_base.index)
ROLL_WINDOWS = [8, 12, 24]

for w in ROLL_WINDOWS:
    roll[f"corr_{w}w"] = roll_base["gpu_ret"].rolling(w, min_periods=w).corr(roll_base["btc_krw_ret"])
    roll[f"beta_{w}w"] = (
        roll_base["gpu_ret"].rolling(w, min_periods=w).cov(roll_base["btc_krw_ret"])
        / roll_base["btc_krw_ret"].rolling(w, min_periods=w).var()
    )

roll_summary = (
    roll.agg(["count", "mean", "std", "min", "max"])
    .T
    .reset_index()
    .rename(columns={"index": "metric"})
)

print("Rolling correlation / beta summary")
display(roll_summary.round(4))
roll.to_csv(H1_OUT / "roll.csv")
roll_summary.to_csv(H1_OUT / "roll_sum.csv", index=False)

plt.figure(figsize=(11, 4))
for col in [f"corr_{w}w" for w in ROLL_WINDOWS]:
    plt.plot(roll.index, roll[col], label=col)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Rolling BTC-GPU return correlation")
plt.xlabel("Date")
plt.ylabel("Rolling correlation")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

plt.figure(figsize=(11, 4))
for col in [f"beta_{w}w" for w in ROLL_WINDOWS]:
    plt.plot(roll.index, roll[col], label=col)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Rolling BTC beta of GPU return")
plt.xlabel("Date")
plt.ylabel("Rolling beta")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

reg = h1[["gpu_ret", "btc_krw_ret"]].dropna().copy()
reg = reg.sort_index()

mid_pos = len(reg) // 2
mid_date = reg.index[mid_pos]
reg["time_regime"] = np.where(reg.index < mid_date, "early", "late")

reg["btc_vol_12w"] = reg["btc_krw_ret"].rolling(12, min_periods=12).std()
vol_cut = reg["btc_vol_12w"].median()
reg["btc_vol_regime"] = np.where(reg["btc_vol_12w"] >= vol_cut, "high_btc_vol", "low_btc_vol")
reg.loc[reg["btc_vol_12w"].isna(), "btc_vol_regime"] = np.nan


def corr_beta_by_group(df, group_col):
    rows = []
    for label, sub in df.dropna(subset=[group_col]).groupby(group_col):
        sub = sub[["gpu_ret", "btc_krw_ret"]].dropna()
        if len(sub) < 8:
            continue
        beta = sub["gpu_ret"].cov(sub["btc_krw_ret"]) / sub["btc_krw_ret"].var()
        rows.append({
            "regime_type": group_col,
            "regime": label,
            "n": len(sub),
            "corr": sub["gpu_ret"].corr(sub["btc_krw_ret"]),
            "beta": beta,
        })
    return pd.DataFrame(rows)

regime_stats = pd.concat([
    corr_beta_by_group(reg, "time_regime"),
    corr_beta_by_group(reg, "btc_vol_regime"),
], ignore_index=True)

print("Regime-level BTC-GPU correlation and beta")
display(regime_stats.round(4))
reg.to_csv(H1_OUT / "regime.csv")
regime_stats.to_csv(H1_OUT / "regime_beta.csv", index=False)

def residualize(y, X):
    X_const = sm.add_constant(X.astype(float))
    return sm.OLS(y.astype(float), X_const).fit().resid

partial_rows = []
control_cols = ["fx_ret", "nvda_ret", "gpu_ret_l1"]

for k in range(MAX_LAG + 1):
    btc_col = f"btc_krw_ret_l{k}"
    temp = h1[["gpu_ret", btc_col] + control_cols].dropna().copy()
    gpu_resid = residualize(temp["gpu_ret"], temp[control_cols])
    btc_resid = residualize(temp[btc_col], temp[control_cols])
    partial_rows.append({
        "lag_week": k,
        "partial_corr_gpu_btc": pd.Series(gpu_resid).corr(pd.Series(btc_resid)),
        "n": len(temp),
    })

partial_corr = pd.DataFrame(partial_rows)

print("Partial correlation by BTC lag after controlling FX, NVDA, and GPU lag 1")
display(partial_corr.round(4))
partial_corr.to_csv(H1_OUT / "partial_corr.csv", index=False)

plt.figure(figsize=(9, 4))
plt.plot(partial_corr["lag_week"], partial_corr["partial_corr_gpu_btc"], marker="o")
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Partial correlation: GPU return vs lagged BTC-KRW return")
plt.xlabel("BTC-KRW return lag, weeks")
plt.ylabel("Partial correlation")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

h1_reg = h1[reg_cols].dropna().copy()

formula = (
    "gpu_ret ~ "
    + " + ".join(btc_lag_cols)
    + " + fx_ret + nvda_ret + gpu_ret_l1"
)

h1_model = smf.ols(formula, data=h1_reg).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": HAC_LAG},
)

print(h1_model.summary())

coef_table = pd.DataFrame({
    "term": h1_model.params.index,
    "coef": h1_model.params.values,
    "se_HAC": h1_model.bse.values,
    "t_HAC": h1_model.tvalues.values,
    "p_value_HAC": h1_model.pvalues.values,
})

btc_coef = coef_table[coef_table["term"].isin(btc_lag_cols)].copy()
btc_coef["lag_week"] = btc_coef["term"].str.extract(r"l(\d+)").astype(int)
btc_coef = btc_coef.sort_values("lag_week")

print("BTC lag coefficient table")
display(btc_coef.round(5))

coef_table.to_csv(H1_OUT / "dl_coef.csv", index=False)
btc_coef.to_csv(H1_OUT / "btc_lag_coef.csv", index=False)

plt.figure(figsize=(9, 4))
plt.errorbar(
    btc_coef["lag_week"],
    btc_coef["coef"],
    yerr=1.96 * btc_coef["se_HAC"],
    marker="o",
    capsize=4,
)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Hypo 1: distributed lag coefficients of BTC-KRW returns")
plt.xlabel("BTC-KRW return lag, weeks")
plt.ylabel("Coefficient on BTC-KRW return")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

resid_vol = h1_reg[["gpu_ret", "btc_krw_ret_l0", "fx_ret", "nvda_ret"]].copy()
resid_vol = resid_vol.rename(columns={"btc_krw_ret_l0": "btc_krw_ret"})

resid_vol["resid"] = h1_model.resid
resid_vol["resid_sq"] = resid_vol["resid"] ** 2
resid_vol["btc_abs"] = resid_vol["btc_krw_ret"].abs()
resid_vol["btc_sq"] = resid_vol["btc_krw_ret"] ** 2

vol_corr = pd.DataFrame([{
    "corr_resid_sq_btc_abs": resid_vol["resid_sq"].corr(resid_vol["btc_abs"]),
    "corr_resid_sq_btc_sq": resid_vol["resid_sq"].corr(resid_vol["btc_sq"]),
    "mean_resid_sq": resid_vol["resid_sq"].mean(),
    "max_resid_sq": resid_vol["resid_sq"].max(),
}])

lb_sq = acorr_ljungbox(
    resid_vol["resid_sq"].dropna(),
    lags=[4, 8],
    return_df=True,
).reset_index().rename(columns={"index": "lag"})

print("Residual volatility correlation with BTC volatility")
display(vol_corr.round(5))
print("Ljung-Box test on squared residuals")
display(lb_sq.round(5))

resid_vol.to_csv(H1_OUT / "resid_vol.csv")
vol_corr.to_csv(H1_OUT / "resid_vol_corr.csv", index=False)
lb_sq.to_csv(H1_OUT / "resid_vol_lb.csv", index=False)

plt.figure(figsize=(11, 4))
plt.plot(resid_vol.index, resid_vol["resid_sq"], label="squared residual")
plt.title("Squared residuals from H1 distributed lag regression")
plt.xlabel("Date")
plt.ylabel("Residual squared")
plt.grid(True, alpha=0.3)
plt.legend()
save_plot(); plt.show()

plt.figure(figsize=(11, 4))
plt.plot(resid_vol.index, resid_vol["btc_abs"], label="abs BTC return")
plt.plot(resid_vol.index, resid_vol["resid_sq"], label="squared residual", linestyle="--")
plt.title("BTC volatility proxy and GPU regression residual volatility")
plt.xlabel("Date")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.legend()
save_plot(); plt.show()

from statsmodels.stats.stattools import jarque_bera, omni_normtest

resid = pd.Series(h1_model.resid, index=h1_reg.index, name="resid")
jb_stat, jb_p, skew, kurt = jarque_bera(resid)
omni_stat, omni_p = omni_normtest(resid)

resid_dist = pd.DataFrame([{
    "n": len(resid),
    "mean": resid.mean(),
    "std": resid.std(),
    "skew": skew,
    "kurtosis": kurt,
    "jarque_bera_stat": jb_stat,
    "jarque_bera_p": jb_p,
    "omni_stat": omni_stat,
    "omni_p": omni_p,
}])

print("Residual distribution diagnostic")
display(resid_dist.round(5))
resid_dist.to_csv(H1_OUT / "resid_dist.csv", index=False)

rlm_X = sm.add_constant(h1_reg[btc_lag_cols + ["fx_ret", "nvda_ret", "gpu_ret_l1"]].astype(float), has_constant="add")
rlm_y = h1_reg["gpu_ret"].astype(float)
rlm_model = sm.RLM(rlm_y, rlm_X, M=sm.robust.norms.HuberT()).fit()

rlm_coef = pd.DataFrame({
    "term": rlm_model.params.index,
    "coef_huber": rlm_model.params.values,
    "se_huber": rlm_model.bse.values,
    "t_huber": rlm_model.tvalues.values,
    "p_value_huber": rlm_model.pvalues.values,
})

print("Huber robust regression coefficients")
display(rlm_coef.round(5))
rlm_coef.to_csv(H1_OUT / "huber_coef.csv", index=False)

plt.figure(figsize=(8, 4))
plt.hist(resid, bins=20, alpha=0.7)
plt.title("Residual distribution from H1 distributed lag regression")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

resid_vis = pd.DataFrame({
    "date": h1_reg.index,
    "resid": h1_model.resid,
    "fitted": h1_model.fittedvalues,
}).set_index("date")
resid_vis["std_resid"] = resid_vis["resid"] / resid_vis["resid"].std(ddof=1)
resid_vis["abs_std_resid"] = resid_vis["std_resid"].abs()

resid_spikes = (
    resid_vis.reset_index()
    .sort_values("abs_std_resid", ascending=False)
    .head(10)
)

print("Top residual spikes from H1 main model")
display(resid_spikes.round(5))

resid_vis.to_csv(H1_OUT / "resid_series.csv")
resid_spikes.to_csv(H1_OUT / "resid_spike.csv", index=False)

plt.figure(figsize=(11, 4))
plt.plot(resid_vis.index, resid_vis["resid"], marker="o", linewidth=1)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("H1 residual plot: distributed lag model")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

plt.figure(figsize=(5, 5))
sm.qqplot(resid_vis["resid"].dropna(), line="45", fit=True)
plt.title("H1 residual Q-Q plot")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

joint_hypothesis = ", ".join([f"{col} = 0" for col in btc_lag_cols])
joint_test = h1_model.f_test(joint_hypothesis)

joint_result = pd.DataFrame([{
    "hypothesis": f"btc_krw_ret_l0 ~ btc_krw_ret_l{MAX_LAG} jointly zero",
    "F_stat": float(np.asarray(joint_test.fvalue)),
    "p_value": float(np.asarray(joint_test.pvalue)),
    "df_num": float(joint_test.df_num),
    "df_denom": float(joint_test.df_denom),
}])

print("BTC lag 공동 유의성 검정")
display(joint_result)
joint_result.to_csv(H1_OUT / "joint.csv", index=False)


def make_vif_table(df, cols):
    X = sm.add_constant(df[cols].astype(float))
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        rows.append({
            "variable": col,
            "VIF": variance_inflation_factor(X.values, i),
        })
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)

vif_core = make_vif_table(h1_reg, ["btc_krw_ret_l0", "fx_ret", "nvda_ret", "gpu_ret_l1"])

vif_full = make_vif_table(h1_reg, btc_lag_cols + ["fx_ret", "nvda_ret", "gpu_ret_l1"])

print("Core VIF: BTC contemporaneous return, FX, NVDA, GPU lag")
display(vif_core.round(3))

print("Full model VIF: includes all BTC lags")
display(vif_full.round(3))

vif_core.to_csv(H1_OUT / "vif_core.csv", index=False)
vif_full.to_csv(H1_OUT / "vif_full.csv", index=False)

ext_ctrls = ["fx_ret", "nvda_ret", "nasdaq_ret", "sox_ret", "gpu_ret_l1"]
ext_cols = ["gpu_ret"] + btc_lag_cols + ext_ctrls
h1_ext_reg = h1[ext_cols].dropna().copy()

ext_formula = (
    "gpu_ret ~ "
    + " + ".join(btc_lag_cols)
    + " + fx_ret + nvda_ret + nasdaq_ret + sox_ret + gpu_ret_l1"
)

h1_ext_model = smf.ols(ext_formula, data=h1_ext_reg).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": HAC_LAG},
)

ext_joint = h1_ext_model.f_test(", ".join([f"{col} = 0" for col in btc_lag_cols]))
ext_joint_result = pd.DataFrame([{
    "model": "extended_market_controls",
    "controls": "FX, NVDA, NASDAQ, SOX, GPU lag 1",
    "F_stat": float(np.asarray(ext_joint.fvalue).ravel()[0]),
    "p_value": float(np.asarray(ext_joint.pvalue).ravel()[0]),
    "df_num": float(ext_joint.df_num),
    "df_denom": float(ext_joint.df_denom),
    "n": len(h1_ext_reg),
}])

ext_coef = pd.DataFrame({
    "term": h1_ext_model.params.index,
    "coef": h1_ext_model.params.values,
    "se_HAC": h1_ext_model.bse.values,
    "t_HAC": h1_ext_model.tvalues.values,
    "p_value_HAC": h1_ext_model.pvalues.values,
})

ext_vif = make_vif_table(h1_ext_reg, ["btc_krw_ret_l0", "fx_ret", "nvda_ret", "nasdaq_ret", "sox_ret", "gpu_ret_l1"])

print(h1_ext_model.summary())
print("Extended-control BTC lag joint test")
display(ext_joint_result.round(5))
print("Extended-control core VIF")
display(ext_vif.round(3))

ext_coef.to_csv(H1_OUT / "ext_coef.csv", index=False)
ext_joint_result.to_csv(H1_OUT / "ext_joint.csv", index=False)
ext_vif.to_csv(H1_OUT / "ext_vif.csv", index=False)

def fit_hac_formula(formula):
    return smf.ols(formula, data=h1_reg).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAG})

btc_rhs = " + ".join(btc_lag_cols)
mod_controls = fit_hac_formula("gpu_ret ~ fx_ret + nvda_ret")
mod_controls_ar = fit_hac_formula("gpu_ret ~ fx_ret + nvda_ret + gpu_ret_l1")
mod_controls_btc = fit_hac_formula("gpu_ret ~ " + btc_rhs + " + fx_ret + nvda_ret")
mod_full = h1_model

model_comp = pd.DataFrame([
    {"model": "controls_only", "r2": mod_controls.rsquared, "adj_r2": mod_controls.rsquared_adj, "aic": mod_controls.aic, "bic": mod_controls.bic},
    {"model": "controls_plus_gpu_l1", "r2": mod_controls_ar.rsquared, "adj_r2": mod_controls_ar.rsquared_adj, "aic": mod_controls_ar.aic, "bic": mod_controls_ar.bic},
    {"model": "controls_plus_btc_lags", "r2": mod_controls_btc.rsquared, "adj_r2": mod_controls_btc.rsquared_adj, "aic": mod_controls_btc.aic, "bic": mod_controls_btc.bic},
    {"model": "full_btc_lags_controls_gpu_l1", "r2": mod_full.rsquared, "adj_r2": mod_full.rsquared_adj, "aic": mod_full.aic, "bic": mod_full.bic},
])

increment_rows = pd.DataFrame([
    {
        "comparison": "add_gpu_l1_after_controls",
        "delta_r2": mod_controls_ar.rsquared - mod_controls.rsquared,
        "interpretation": "incremental explanatory power of GPU return persistence",
    },
    {
        "comparison": "add_btc_lags_after_controls",
        "delta_r2": mod_controls_btc.rsquared - mod_controls.rsquared,
        "interpretation": "incremental explanatory power of BTC lag structure before GPU lag",
    },
    {
        "comparison": "add_btc_lags_after_controls_and_gpu_l1",
        "delta_r2": mod_full.rsquared - mod_controls_ar.rsquared,
        "interpretation": "BTC incremental explanatory power in the main specification",
    },
    {
        "comparison": "add_gpu_l1_after_controls_and_btc_lags",
        "delta_r2": mod_full.rsquared - mod_controls_btc.rsquared,
        "interpretation": "GPU persistence contribution after BTC lags",
    },
])

print("Nested model comparison")
display(model_comp.round(5))
print("Incremental R-squared decomposition")
display(increment_rows.round(5))
model_comp.to_csv(H1_OUT / "nested.csv", index=False)
increment_rows.to_csv(H1_OUT / "inc_r2.csv", index=False)

ret_all_nt_w = (
    pd.read_csv(OUT / "ret_all_notrim_w.csv", parse_dates=["date"])
    .set_index("date")
    .sort_index()
)

h1_nt = (
    ret_all_nt_w[[GPU_RET_COL]]
    .rename(columns={GPU_RET_COL: "gpu_ret"})
    .join(mkt_w[MKT_RET_COLS], how="inner")
    .sort_index()
)

h1_nt["gpu_ret_l1"] = h1_nt["gpu_ret"].shift(1)
for k in range(MAX_LAG + 1):
    h1_nt[f"btc_krw_ret_l{k}"] = h1_nt["btc_krw_ret"].shift(k)

h1_nt_reg = h1_nt[reg_cols].dropna().copy()

h1_nt_model = smf.ols(formula, data=h1_nt_reg).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": HAC_LAG},
)

nt_joint_test = h1_nt_model.f_test(joint_hypothesis)
nt_joint = pd.DataFrame([{
    "block": "no_trim",
    "hypothesis": f"btc_krw_ret_l0 ~ btc_krw_ret_l{MAX_LAG} jointly zero",
    "F_stat": float(np.asarray(nt_joint_test.fvalue)),
    "p_value": float(np.asarray(nt_joint_test.pvalue)),
    "df_num": float(nt_joint_test.df_num),
    "df_denom": float(nt_joint_test.df_denom),
    "n": len(h1_nt_reg),
}])

nt_coef = pd.DataFrame({
    "term": h1_nt_model.params.index,
    "coef": h1_nt_model.params.values,
    "se_HAC": h1_nt_model.bse.values,
    "t_HAC": h1_nt_model.tvalues.values,
    "p_value_HAC": h1_nt_model.pvalues.values,
})

main_vs_notrim = pd.DataFrame([
    {
        "model": "main_trimmed",
        "n": len(h1_reg),
        "joint_p_value": float(joint_result.loc[0, "p_value"]),
    },
    {
        "model": "no_quantile_trim",
        "n": len(h1_nt_reg),
        "joint_p_value": float(nt_joint.loc[0, "p_value"]),
    },
])

print("H1 no-trim robustness joint test")
display(nt_joint.round(5))
display(main_vs_notrim.round(5))

nt_joint.to_csv(H1_OUT / "nt_joint.csv", index=False)
nt_coef.to_csv(H1_OUT / "nt_coef.csv", index=False)
main_vs_notrim.to_csv(H1_OUT / "nt_compare.csv", index=False)

var_data = h1[["gpu_ret", "btc_krw_ret", "fx_ret", "nvda_ret"]].dropna().copy()

var_fit = VAR(var_data).fit(maxlags=MAX_LAG, ic="aic")
if var_fit.k_ar == 0:
    var_fit = VAR(var_data).fit(maxlags=1)

btc_to_gpu = var_fit.test_causality(
    caused="gpu_ret",
    causing=["btc_krw_ret"],
    kind="f",
)

var_result = pd.DataFrame([{
    "selected_lag": int(var_fit.k_ar),
    "test": "VAR Granger causality: BTC-KRW return -> GPU return | FX, NVDA included",
    "statistic": float(btc_to_gpu.test_statistic),
    "p_value": float(btc_to_gpu.pvalue),
    "df": str(btc_to_gpu.df),
    "interpretation_limit": "predictive precedence only; not structural/economic causality",
}])

print("VAR-based multivariate Granger causality test")
display(var_result)
var_result.to_csv(H1_OUT / "var_granger.csv", index=False)

level_h1 = (
    idx_all_w[[GPU_RET_COL]]
    .rename(columns={GPU_RET_COL: "gpu_index"})
)

level_h1 = (
    level_h1
    .join(mkt_log_level_w[["btc_krw", "usdkrw", "nvda"]], how="inner")
    .dropna()
    .copy()
)

level_h1["log_gpu_index"] = np.log(level_h1["gpu_index"])
level_h1 = level_h1.rename(columns={
    "btc_krw": "log_btc_krw",
    "usdkrw": "log_usdkrw",
    "nvda": "log_nvda",
})


def adf_rows(series_dict):
    rows = []
    for name, s in series_dict.items():
        stat, p_value, usedlag, nobs, crit, icbest = adfuller(s.dropna(), autolag="AIC")
        rows.append({
            "series": name,
            "adf_stat": stat,
            "p_value": p_value,
            "used_lag": usedlag,
            "nobs": nobs,
            "crit_1pct": crit["1%"],
            "crit_5pct": crit["5%"],
            "crit_10pct": crit["10%"],
        })
    return pd.DataFrame(rows)

adf_level = adf_rows({
    "log_gpu_index": level_h1["log_gpu_index"],
    "log_btc_krw": level_h1["log_btc_krw"],
})

adf_diff = adf_rows({
    "d_log_gpu_index": level_h1["log_gpu_index"].diff(),
    "d_log_btc_krw": level_h1["log_btc_krw"].diff(),
})

coint_stat, coint_p, coint_crit = coint(
    level_h1["log_gpu_index"],
    level_h1["log_btc_krw"],
)

coint_result = pd.DataFrame([{
    "test": "Engle-Granger cointegration: log GPU index ~ log BTC-KRW",
    "stat": coint_stat,
    "p_value": coint_p,
    "crit_1pct": coint_crit[0],
    "crit_5pct": coint_crit[1],
    "crit_10pct": coint_crit[2],
}])

print("ADF test on levels")
display(adf_level.round(4))
print("ADF test on first differences")
display(adf_diff.round(4))
print("Engle-Granger cointegration test")
display(coint_result.round(4))

level_h1.to_csv(H1_OUT / "level.csv")
adf_level.to_csv(H1_OUT / "adf_level.csv", index=False)
adf_diff.to_csv(H1_OUT / "adf_diff.csv", index=False)
coint_result.to_csv(H1_OUT / "coint.csv", index=False)

coint_ols = smf.ols("log_gpu_index ~ log_btc_krw", data=level_h1).fit()

ecm = level_h1.copy()
ecm["ec_term"] = coint_ols.resid

ecm["d_log_gpu"] = ecm["log_gpu_index"].diff()
ecm["d_log_btc"] = ecm["log_btc_krw"].diff()
ecm["d_log_usdkrw"] = ecm["log_usdkrw"].diff()
ecm["d_log_nvda"] = ecm["log_nvda"].diff()
ecm["ec_l1"] = ecm["ec_term"].shift(1)

for k in range(MAX_LAG + 1):
    ecm[f"d_log_btc_l{k}"] = ecm["d_log_btc"].shift(k)

ecm_models = {}
ecm_select_rows = []

for K in range(0, MAX_LAG + 1):
    lag_cols = [f"d_log_btc_l{k}" for k in range(K + 1)]
    cols = ["d_log_gpu", "ec_l1", "d_log_usdkrw", "d_log_nvda"] + lag_cols
    reg = ecm[cols].dropna().copy()
    formula_k = "d_log_gpu ~ ec_l1 + " + " + ".join(lag_cols + ["d_log_usdkrw", "d_log_nvda"])
    fit_k = smf.ols(formula_k, data=reg).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAG},
    )
    ecm_models[K] = fit_k
    ecm_select_rows.append({
        "K_btc_lag": K,
        "n": int(len(reg)),
        "aic": fit_k.aic,
        "bic": fit_k.bic,
        "ec_l1_coef": fit_k.params.get("ec_l1", np.nan),
        "ec_l1_p": fit_k.pvalues.get("ec_l1", np.nan),
    })

ecm_select = pd.DataFrame(ecm_select_rows)
BEST_K_AIC = int(ecm_select.loc[ecm_select["aic"].idxmin(), "K_btc_lag"])
BEST_K_BIC = int(ecm_select.loc[ecm_select["bic"].idxmin(), "K_btc_lag"])
BEST_K = BEST_K_AIC

ecm_model = ecm_models[BEST_K]

print("ECM lag selection by AIC/BIC")
display(ecm_select.round(4))
print(f"Selected ECM BTC lag K by AIC = {BEST_K_AIC}")
print(f"Selected ECM BTC lag K by BIC = {BEST_K_BIC}")
print(ecm_model.summary())

ecm_coef = pd.DataFrame({
    "term": ecm_model.params.index,
    "coef": ecm_model.params.values,
    "se_HAC": ecm_model.bse.values,
    "t_HAC": ecm_model.tvalues.values,
    "p_value_HAC": ecm_model.pvalues.values,
})

print("Selected ECM coefficients")
display(ecm_coef.round(5))

ecm.to_csv(H1_OUT / "ecm_reg.csv")
ecm_select.to_csv(H1_OUT / "ecm_lag.csv", index=False)
ecm_coef.to_csv(H1_OUT / "ecm_coef.csv", index=False)

max_corr_row = lag_corr.loc[lag_corr["corr_gpu_btc"].abs().idxmax()]
max_partial_row = partial_corr.loc[partial_corr["partial_corr_gpu_btc"].abs().idxmax()]
max_etc_row = etc_lag_corr.loc[etc_lag_corr["corr_gpu_etc"].abs().idxmax()]

summary_rows = [
    {
        "block": "lag_correlation",
        "main_result": "max_abs_corr_lag_week",
        "value": int(max_corr_row["lag_week"]),
        "detail": float(max_corr_row["corr_gpu_btc"]),
    },
    {
        "block": "ccf_lag_correlation",
        "main_result": "approx_95_conf_band",
        "value": float(ccf_band),
        "detail": "CCF-style lag plot; descriptive EDA only",
    },
    {
        "block": "partial_correlation",
        "main_result": "max_abs_partial_corr_lag_week",
        "value": int(max_partial_row["lag_week"]),
        "detail": float(max_partial_row["partial_corr_gpu_btc"]),
    },
    {
        "block": "distributed_lag_regression",
        "main_result": "btc_lags_joint_p_value",
        "value": float(joint_result.loc[0, "p_value"]),
        "detail": f"H0: btc_krw_ret_l0 ... btc_krw_ret_l{MAX_LAG} are all zero",
    },
    {
        "block": "lag_missingness",
        "main_result": "complete_regression_weeks_all_lags",
        "value": int(lag_na_summary.loc[0, "complete_regression_weeks_all_lags"]),
        "detail": "No forward fill; complete-case regression after lag construction",
    },
    {
        "block": "rolling_eda",
        "main_result": "max_abs_rolling_corr_24w",
        "value": float(roll["corr_24w"].abs().max()),
        "detail": "24-week rolling correlation added to reduce low-degree-of-freedom noise",
    },
    {
        "block": "regime_eda",
        "main_result": "regime_corr_beta_rows",
        "value": int(len(regime_stats)),
        "detail": "early/late and BTC-volatility regime EDA; not confirmatory evidence",
    },
    {
        "block": "volatility_clustering",
        "main_result": "ljungbox_sq_resid_p_lag8",
        "value": float(lb_sq.loc[lb_sq["lag"] == 8, "lb_pvalue"].iloc[0]) if (lb_sq["lag"] == 8).any() else np.nan,
        "detail": "Ljung-Box p-value on squared residuals",
    },
    {
        "block": "residual_distribution",
        "main_result": "jarque_bera_p_value",
        "value": float(resid_dist.loc[0, "jarque_bera_p"]),
        "detail": "Fat-tail/non-normality diagnostic for OLS residuals",
    },
    {
        "block": "incremental_r2",
        "main_result": "btc_lags_after_controls_and_gpu_l1_delta_r2",
        "value": float(increment_rows.loc[increment_rows["comparison"] == "add_btc_lags_after_controls_and_gpu_l1", "delta_r2"].iloc[0]),
        "detail": "Checks whether model fit is driven mainly by BTC lags or GPU persistence",
    },
    {
        "block": "no_trim_robustness",
        "main_result": "btc_lags_joint_p_value_no_trim",
        "value": float(nt_joint.loc[0, "p_value"]),
        "detail": "Robustness using ret_all_notrim_w: no quantile trim, hard error filter kept",
    },
    {
        "block": "extended_market_controls",
        "main_result": "btc_lags_joint_p_value_with_nasdaq_sox",
        "value": float(ext_joint_result.loc[0, "p_value"]),
        "detail": "Robustness only: FX, NVDA, NASDAQ, SOX, GPU lag included",
    },
    {
        "block": "residual_visual_diagnostic",
        "main_result": "max_abs_standardized_residual",
        "value": float(resid_spikes["abs_std_resid"].max()),
        "detail": "Residual time plot and Q-Q plot saved/printed for regime-misspecification check",
    },
    {
        "block": "multicollinearity",
        "main_result": "max_core_vif",
        "value": float(vif_core["VIF"].max()),
        "detail": "VIF among btc_krw_ret_l0, fx_ret, nvda_ret, gpu_ret_l1",
    },
    {
        "block": "var_granger",
        "main_result": "btc_to_gpu_p_value_with_controls",
        "value": float(var_result.loc[0, "p_value"]),
        "detail": f"VAR selected lag = {int(var_result.loc[0, 'selected_lag'])}; FX and NVDA included; predictive precedence only",
    },
    {
        "block": "cointegration",
        "main_result": "engle_granger_p_value",
        "value": float(coint_result.loc[0, "p_value"]),
        "detail": "log GPU index and log BTC-KRW",
    },
    {
        "block": "ecm",
        "main_result": "selected_btc_lag_by_aic",
        "value": BEST_K_AIC,
        "detail": f"BIC selected lag = {BEST_K_BIC}; ECM includes ec_l1, d_log_usdkrw, d_log_nvda",
    },
    {
        "block": "etc_eda",
        "main_result": "btc_etc_return_corr",
        "value": float(crypto_corr),
        "detail": "ETC-KRW is used only as GPU-mining-domain EDA, not main regressor",
    },
    {
        "block": "etc_eda",
        "main_result": "max_abs_corr_gpu_etc_lag_week",
        "value": int(max_etc_row["lag_week"]),
        "detail": float(max_etc_row["corr_gpu_etc"]),
    },
]

h1_summary = pd.DataFrame(summary_rows)

print("Hypo 1 summary")
display(h1_summary)

h1_summary.to_csv(H1_OUT / "summary.csv", index=False)
print("Hypo 1 outputs saved to:", H1_OUT)
