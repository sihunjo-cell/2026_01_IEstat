# Auto-generated from Hypo_2_final.ipynb (code cells only, markdown/outputs stripped)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

OUT = Path("outputs")
H1_DIR = OUT / "hypo1"
H2_DIR = OUT / "hypo2"
H2_DIR.mkdir(parents=True, exist_ok=True)

L = 8
GROUPS = ["high", "mid", "low"]
BASE = "mid"  # 메인 기준항: 표본이 충분한 mid 그룹

MKT_FILE = H1_DIR / "reg.csv"

plt.rcParams["font.family"] = 'MalGun Gothic'
plt.rcParams["axes.unicode_minus"] = False

import re as _re
from datetime import datetime as _dt

FIG_DIR = Path("output_pic") / "hypo_2"
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

ret_chip_w = (
    pd.read_csv(OUT / "ret_chip_w.csv", parse_dates=["date"])
      .set_index("date")
      .sort_index()
)
idx_chip_w = (
    pd.read_csv(OUT / "idx_chip_w.csv", parse_dates=["date"])
      .set_index("date")
      .sort_index()
)
ret_vram_w = (
    pd.read_csv(OUT / "ret_vram_w.csv", parse_dates=["date"])
      .set_index("date")
      .sort_index()
)
idx_vram_w = (
    pd.read_csv(OUT / "idx_vram_w.csv", parse_dates=["date"])
      .set_index("date")
      .sort_index()
)

gpu_meta = pd.read_csv(OUT / "gpu_meta.csv")
panel_gpu = pd.read_csv(OUT / "panel_gpu.csv", parse_dates=["date"])
diag_chip = pd.read_csv(OUT / "diag_chip.csv")
diag_vram = pd.read_csv(OUT / "diag_vram.csv")
gap_week = pd.read_csv(OUT / "gap_week.csv", parse_dates=["date"])

h1_mkt = pd.read_csv(MKT_FILE, index_col=0, parse_dates=True).sort_index()
base_mkt_cols = ["btc_krw_ret", "fx_ret", "nvda_ret"]
extra_mkt_cols = [c for c in ["nasdaq_ret", "sox_ret"] if c in h1_mkt.columns]
mkt_cols = base_mkt_cols + extra_mkt_cols + [f"btc_krw_ret_l{k}" for k in range(L + 1)]
mkt = h1_mkt[mkt_cols].copy()

print("ret_chip_w, main H2 input")
display(ret_chip_w.head())
print("ret_vram_w, VRAM proxy robustness input")
display(ret_vram_w.head())
print("market variables from Hypo 1")
display(mkt.head())

chip_sample = (
    gpu_meta[gpu_meta["chip_perf_group"].isin(GROUPS)]
    .groupby("chip_perf_group")
    .agg(
        product_n=("Id", "nunique"),
        median_vram_gb=("vram_gb", "median"),
        min_vram_gb=("vram_gb", "min"),
        max_vram_gb=("vram_gb", "max"),
    )
    .reindex(GROUPS)
    .reset_index()
)

vram_sample = (
    gpu_meta[gpu_meta["h2_perf_group"].isin(GROUPS)]
    .groupby("h2_perf_group")
    .agg(
        product_n=("Id", "nunique"),
        median_vram_gb=("vram_gb", "median"),
        min_vram_gb=("vram_gb", "min"),
        max_vram_gb=("vram_gb", "max"),
    )
    .reindex(GROUPS)
    .reset_index()
)

print("chip_perf_group sample summary")
display(chip_sample)
print("h2_perf_group / VRAM proxy sample summary")
display(vram_sample)
print("chip index diagnostics")
display(diag_chip)
print("VRAM proxy index diagnostics")
display(diag_vram)

chip_sample.to_csv(H2_DIR / "chip_sample.csv", index=False)
vram_sample.to_csv(H2_DIR / "vram_sample.csv", index=False)
diag_chip.to_csv(H2_DIR / "diag_chip.csv", index=False)
diag_vram.to_csv(H2_DIR / "diag_vram.csv", index=False)

meta_g = gpu_meta[
    gpu_meta["chip_perf_group"].isin(GROUPS)
    & gpu_meta["h2_perf_group"].isin(GROUPS)
].copy()

vram_chip_n = (
    pd.crosstab(meta_g["chip_perf_group"], meta_g["h2_perf_group"])
      .reindex(index=GROUPS, columns=GROUPS, fill_value=0)
)
vram_chip_pct = vram_chip_n.div(vram_chip_n.sum(axis=1).replace(0, np.nan), axis=0)

vram_mis = pd.DataFrame([{
    "product_n": len(meta_g),
    "matched_chip_vram_group_n": int((meta_g["chip_perf_group"] == meta_g["h2_perf_group"]).sum()),
    "mismatched_chip_vram_group_n": int((meta_g["chip_perf_group"] != meta_g["h2_perf_group"]).sum()),
    "mismatch_rate": float((meta_g["chip_perf_group"] != meta_g["h2_perf_group"]).mean()),
}])

print("chip performance group × VRAM proxy group, count")
display(vram_chip_n)
print("chip performance group × VRAM proxy group, row percentage")
display(vram_chip_pct.round(3))
print("VRAM proxy mismatch summary")
display(vram_mis.round(4))

comp_panel = panel_gpu[panel_gpu["chip_perf_group"].isin(GROUPS)].copy()

first_w = (
    comp_panel.groupby(["row_id", "chip_perf_group"], as_index=False)["date"]
              .min()
              .rename(columns={"date": "date"})
)

active = (
    comp_panel.groupby(["date", "chip_perf_group"], as_index=False)["row_id"]
              .nunique()
              .rename(columns={"row_id": "active_product_n"})
)

new_prod = (
    first_w.groupby(["date", "chip_perf_group"], as_index=False)["row_id"]
              .nunique()
              .rename(columns={"row_id": "new_product_n"})
)

comp_diag = active.merge(new_prod, on=["date", "chip_perf_group"], how="left")
comp_diag["new_product_n"] = comp_diag["new_product_n"].fillna(0)
comp_diag["new_product_share"] = comp_diag["new_product_n"] / comp_diag["active_product_n"].replace(0, np.nan)
comp_diag["log_active_product_n"] = np.log1p(comp_diag["active_product_n"])

gap_c = gap_week.rename(columns={"group": "chip_perf_group"}).copy()
gap_c["bridge_ratio"] = gap_c["bridge_n"] / gap_c["adj_n"].replace(0, np.nan)

comp_diag = comp_diag.merge(
    gap_c[["date", "chip_perf_group", "adj_n", "bridge_n", "bridge_ratio"]],
    on=["date", "chip_perf_group"],
    how="left",
)

comp_sum = (
    comp_diag.groupby("chip_perf_group")
    .agg(
        mean_active_product_n=("active_product_n", "mean"),
        median_active_product_n=("active_product_n", "median"),
        max_new_product_share=("new_product_share", "max"),
        mean_new_product_share=("new_product_share", "mean"),
        max_bridge_ratio=("bridge_ratio", "max"),
        mean_bridge_ratio=("bridge_ratio", "mean"),
    )
    .reset_index()
)

print("product-composition diagnostic summary")
display(comp_sum.round(4))

plt.figure(figsize=(11, 4))
for g in GROUPS:
    sub = comp_diag[comp_diag["chip_perf_group"] == g]
    plt.plot(sub["date"], sub["active_product_n"], label=g)
plt.title("Hypo 2: active product count by chip performance group")
plt.xlabel("Date")
plt.ylabel("Active products in return panel")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

plt.figure(figsize=(11, 4))
for g in GROUPS:
    sub = comp_diag[comp_diag["chip_perf_group"] == g]
    plt.plot(sub["date"], sub["new_product_share"], label=g)
plt.title("Hypo 2: new product share proxy by chip performance group")
plt.xlabel("Date")
plt.ylabel("New product share in return panel")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

vram_chip_n.to_csv(H2_DIR / "vram_chip_n.csv")
vram_chip_pct.to_csv(H2_DIR / "vram_chip_pct.csv")
vram_mis.to_csv(H2_DIR / "vram_mismatch.csv", index=False)
comp_diag.to_csv(H2_DIR / "comp_diag.csv", index=False)
comp_sum.to_csv(H2_DIR / "comp_sum.csv", index=False)

chip_cols = [g for g in GROUPS if g in ret_chip_w.columns]
vram_cols = [g for g in GROUPS if g in ret_vram_w.columns]

plt.figure(figsize=(11, 4))
for g in chip_cols:
    plt.plot(idx_chip_w.index, idx_chip_w[g], label=f"chip {g}")
plt.axhline(100, linestyle="--", linewidth=1)
plt.title("Hypo 2: chip performance group price index")
plt.xlabel("Date")
plt.ylabel("Index, base=100")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

plt.figure(figsize=(11, 4))
for g in chip_cols:
    plt.plot(ret_chip_w.index, ret_chip_w[g], label=f"chip {g}", alpha=0.8)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Hypo 2: chip performance group weekly log returns")
plt.xlabel("Date")
plt.ylabel("Weekly log return")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

chip_ret = (
    ret_chip_w[chip_cols]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .T
    .reset_index()
    .rename(columns={"index": "group"})
)

print("chip group return summary")
display(chip_ret.round(5))
chip_ret.to_csv(H2_DIR / "chip_ret.csv", index=False)

def run_anova(ret_w, cols, block_name):
    samples = [ret_w[g].dropna().to_numpy(float) for g in cols]
    labels = cols

    levene_stat, levene_p = stats.levene(*samples, center="median")
    f_stat, f_p = stats.f_oneway(*samples)
    welch = anova_oneway(samples, use_var="unequal")

    out = pd.DataFrame([
        {"block": block_name, "test": "Levene_median", "stat": levene_stat, "p_value": levene_p, "note": "H0: equal variances"},
        {"block": block_name, "test": "One_way_ANOVA", "stat": f_stat, "p_value": f_p, "note": "H0: equal group means"},
        {"block": block_name, "test": "Welch_ANOVA", "stat": welch.statistic, "p_value": welch.pvalue, "note": "H0: equal group means, unequal variance allowed"},
    ])

    long = (
        ret_w[cols]
        .reset_index()
        .melt(id_vars="date", value_vars=cols, var_name="group", value_name="gpu_ret")
        .dropna()
    )

    return out, long

anova_chip, chip_long_anova = run_anova(ret_chip_w, chip_cols, "chip_main")
anova_vram, vram_long_anova = run_anova(ret_vram_w, vram_cols, "vram_proxy")
anova_results = pd.concat([anova_chip, anova_vram], ignore_index=True)

print("ANOVA / Welch ANOVA results")
display(anova_results.round(5))
anova_results.to_csv(H2_DIR / "anova.csv", index=False)

plt.figure(figsize=(8, 4))
chip_long_anova.boxplot(column="gpu_ret", by="group", flierprops={
        "marker": "o",
        "markersize": 2,
        "alpha": 0.5,
    })
plt.title("Hypo 2 Phase 1: chip group weekly returns")
plt.suptitle("")
plt.xlabel("Chip performance group")
plt.ylabel("Weekly log return")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

def make_reg_df(ret_w, block_name, cols):
    long = (
        ret_w[cols]
        .reset_index()
        .melt(id_vars="date", value_vars=cols, var_name="group", value_name="gpu_ret")
        .dropna(subset=["gpu_ret"])
        .sort_values(["group", "date"])
        .copy()
    )

    dat = long.merge(mkt.reset_index().rename(columns={mkt.index.name or "index": "date"}), on="date", how="inner")
    dat = dat.sort_values(["group", "date"]).copy()

    dat["gpu_ret_l1"] = dat.groupby("group")["gpu_ret"].shift(1)

    dat["is_high"] = (dat["group"] == "high").astype(int)
    dat["is_low"] = (dat["group"] == "low").astype(int)

    btc_lags = [f"btc_krw_ret_l{k}" for k in range(L + 1)]
    for col in btc_lags:
        lag = col.replace("btc_krw_ret_l", "")
        dat[f"high_btc_l{lag}"] = dat["is_high"] * dat[col]
        dat[f"low_btc_l{lag}"] = dat["is_low"] * dat[col]

    dat["high_nvda_ret"] = dat["is_high"] * dat["nvda_ret"]
    dat["low_nvda_ret"] = dat["is_low"] * dat["nvda_ret"]
    dat["high_fx_ret"] = dat["is_high"] * dat["fx_ret"]
    dat["low_fx_ret"] = dat["is_low"] * dat["fx_ret"]

    dat["block"] = block_name
    return dat

h2_chip = make_reg_df(ret_chip_w, "chip_main", chip_cols)
h2_vram = make_reg_df(ret_vram_w, "vram_proxy", vram_cols)

btc_lags = [f"btc_krw_ret_l{k}" for k in range(L + 1)]
hi_btc = [f"high_btc_l{k}" for k in range(L + 1)]
low_btc = [f"low_btc_l{k}" for k in range(L + 1)]

model_cols = (
    ["gpu_ret", "date", "group", "is_high", "is_low"]
    + btc_lags
    + hi_btc
    + low_btc
    + ["fx_ret", "nvda_ret", "high_fx_ret", "low_fx_ret", "high_nvda_ret", "low_nvda_ret", "gpu_ret_l1"]
)

chip_reg = h2_chip[model_cols].dropna().copy()
vram_reg = h2_vram[model_cols].dropna().copy()

print("H2 chip regression dataset, mid as base group")
display(chip_reg.head())
print("H2 VRAM proxy regression dataset, mid as base group")
display(vram_reg.head())

sample_check = pd.DataFrame([
    {"block": "chip_main", "base_group": BASE, "raw_rows": len(h2_chip), "complete_reg_rows": len(chip_reg), "week_n": chip_reg["date"].nunique()},
    {"block": "vram_proxy", "base_group": BASE, "raw_rows": len(h2_vram), "complete_reg_rows": len(vram_reg), "week_n": vram_reg["date"].nunique()},
])

print("H2 sample check")
display(sample_check)

h2_chip.to_csv(H2_DIR / "chip_raw.csv", index=False)
chip_reg.to_csv(H2_DIR / "chip_reg.csv", index=False)
h2_vram.to_csv(H2_DIR / "vram_raw.csv", index=False)
vram_reg.to_csv(H2_DIR / "vram_reg.csv", index=False)
sample_check.to_csv(H2_DIR / "sample.csv", index=False)

scatter_rows = []
scatter_df = chip_reg[["date", "group", "gpu_ret", "btc_krw_ret_l0"]].dropna().copy()

plt.figure(figsize=(8, 5))
for g in GROUPS:
    sub = scatter_df[scatter_df["group"] == g].copy()
    if sub.empty:
        continue
    x = sub["btc_krw_ret_l0"].to_numpy(float)
    y = sub["gpu_ret"].to_numpy(float)
    plt.scatter(x, y, s=16, alpha=0.5, label=g)

    if len(sub) >= 8 and np.nanvar(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        y_line = intercept + slope * x_line
        plt.plot(x_line, y_line, linewidth=1.5)
        corr = sub["gpu_ret"].corr(sub["btc_krw_ret_l0"])
        simple_mod = smf.ols("gpu_ret ~ btc_krw_ret_l0", data=sub).fit()
        scatter_rows.append({
            "group": g,
            "n": len(sub),
            "corr_l0": corr,
            "simple_beta_l0": simple_mod.params.get("btc_krw_ret_l0", np.nan),
            "simple_beta_p": simple_mod.pvalues.get("btc_krw_ret_l0", np.nan),
            "intercept": simple_mod.params.get("Intercept", np.nan),
        })

plt.axhline(0, linestyle="--", linewidth=1)
plt.axvline(0, linestyle="--", linewidth=1)
plt.title("BTC-KRW return vs GPU return by chip performance group")
plt.xlabel("BTC-KRW weekly return, lag 0")
plt.ylabel("GPU group weekly return")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

scatter_summary = pd.DataFrame(scatter_rows)
print("Group-wise scatter trend summary")
display(scatter_summary.round(5))
scatter_summary.to_csv(H2_DIR / "scatter_beta.csv", index=False)

ccf_rows = []
for g in GROUPS:
    sub = chip_reg[chip_reg["group"] == g].copy()
    for k in range(L + 1):
        col = f"btc_krw_ret_l{k}"
        tmp = sub[["gpu_ret", col]].dropna()
        if len(tmp) == 0:
            corr = np.nan
            band = np.nan
        else:
            corr = tmp["gpu_ret"].corr(tmp[col])
            band = 1.96 / np.sqrt(len(tmp))
        ccf_rows.append({
            "group": g,
            "lag_week": k,
            "corr": corr,
            "n": len(tmp),
            "approx_95_band": band,
            "outside_band": bool(abs(corr) > band) if pd.notna(corr) and pd.notna(band) else False,
        })

ccf_group = pd.DataFrame(ccf_rows)
print("Group-specific CCF table")
display(ccf_group.round(4))
ccf_group.to_csv(H2_DIR / "ccf_group.csv", index=False)

fig, axes = plt.subplots(1, len(GROUPS), figsize=(14, 3.8), sharey=True)
for ax, g in zip(axes, GROUPS):
    sub = ccf_group[ccf_group["group"] == g].sort_values("lag_week")
    ax.stem(sub["lag_week"], sub["corr"])
    band = sub["approx_95_band"].median()
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axhline(band, linestyle=":", linewidth=1)
    ax.axhline(-band, linestyle=":", linewidth=1)
    ax.set_title(f"{g}: GPU ret vs BTC lag")
    ax.set_xlabel("BTC lag, weeks")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Correlation")
plt.suptitle("Hypo 2 EDA: group-specific CCF with approximate 95% band")
plt.tight_layout()
save_plot(); plt.show()

vol_df = chip_reg[["date", "group", "gpu_ret", "btc_krw_ret_l0"]].dropna().copy().sort_values(["group", "date"])
vol_df["gpu_ret_sq"] = vol_df["gpu_ret"] ** 2
vol_df["btc_ret_abs"] = vol_df["btc_krw_ret_l0"].abs()
vol_df["gpu_roll_std_12w"] = vol_df.groupby("group")["gpu_ret"].transform(lambda s: s.rolling(12, min_periods=8).std())
vol_df["gpu_roll_std_24w"] = vol_df.groupby("group")["gpu_ret"].transform(lambda s: s.rolling(24, min_periods=16).std())

vol_rows = []
for g, sub in vol_df.groupby("group"):
    vol_rows.append({
        "group": g,
        "n": len(sub),
        "mean_sq_ret": sub["gpu_ret_sq"].mean(),
        "max_sq_ret": sub["gpu_ret_sq"].max(),
        "corr_sqret_absbtc": sub["gpu_ret_sq"].corr(sub["btc_ret_abs"]),
        "corr_rollstd12_absbtc": sub["gpu_roll_std_12w"].corr(sub["btc_ret_abs"]),
        "corr_rollstd24_absbtc": sub["gpu_roll_std_24w"].corr(sub["btc_ret_abs"]),
    })
vol_summary = pd.DataFrame(vol_rows)
print("Group volatility clustering summary")
display(vol_summary.round(5))
vol_df.to_csv(H2_DIR / "vol_group.csv", index=False)
vol_summary.to_csv(H2_DIR / "vol_sum.csv", index=False)

plt.figure(figsize=(11, 4))
for g in GROUPS:
    sub = vol_df[vol_df["group"] == g]
    plt.plot(sub["date"], sub["gpu_roll_std_24w"], label=f"{g} 24w rolling std")
plt.title("Hypo 2 EDA: chip group rolling volatility")
plt.xlabel("Date")
plt.ylabel("24-week rolling std of GPU return")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

reg2 = chip_reg[["date", "group", "gpu_ret", "btc_krw_ret_l0"]].dropna().copy().sort_values("date")
unique_dates = sorted(reg2["date"].unique())
mid_date = unique_dates[len(unique_dates)//2]
reg2["time_regime"] = np.where(reg2["date"] < mid_date, "early", "late")

btc_date = (
    chip_reg[["date", "btc_krw_ret_l0"]]
    .drop_duplicates("date")
    .sort_values("date")
    .set_index("date")
)
btc_date["btc_vol_12w"] = btc_date["btc_krw_ret_l0"].rolling(12, min_periods=12).std()
vol_cut = btc_date["btc_vol_12w"].median()
btc_date["btc_vol_regime"] = np.where(btc_date["btc_vol_12w"] >= vol_cut, "high_btc_vol", "low_btc_vol")
btc_date.loc[btc_date["btc_vol_12w"].isna(), "btc_vol_regime"] = np.nan
reg2 = reg2.merge(btc_date[["btc_vol_regime"]].reset_index(), on="date", how="left")


def group_regime_beta(df, regime_col):
    rows = []
    for (regime, group), sub in df.dropna(subset=[regime_col]).groupby([regime_col, "group"]):
        if len(sub) < 8 or sub["btc_krw_ret_l0"].var() == 0:
            continue
        beta = sub["gpu_ret"].cov(sub["btc_krw_ret_l0"]) / sub["btc_krw_ret_l0"].var()
        rows.append({
            "regime_type": regime_col,
            "regime": regime,
            "group": group,
            "n": len(sub),
            "corr_l0": sub["gpu_ret"].corr(sub["btc_krw_ret_l0"]),
            "simple_beta_l0": beta,
        })
    return pd.DataFrame(rows)

h2_regime_beta = pd.concat([
    group_regime_beta(reg2, "time_regime"),
    group_regime_beta(reg2, "btc_vol_regime"),
], ignore_index=True)

print("H2 group beta by regime")
display(h2_regime_beta.round(5))
reg2.to_csv(H2_DIR / "regime.csv", index=False)
h2_regime_beta.to_csv(H2_DIR / "regime_beta.csv", index=False)

def h2_lag_audit(raw_df, reg_df, block):
    rows = []
    for group, sub in raw_df.groupby("group"):
        rows.append({
            "block": block,
            "group": group,
            "raw_rows": len(sub),
            "gpu_ret_nonnull": int(sub["gpu_ret"].notna().sum()),
            "btc_lag0_nonnull": int(sub["btc_krw_ret_l0"].notna().sum()),
            f"btc_lag{L}_nonnull": int(sub[f"btc_krw_ret_l{L}"].notna().sum()),
            "gpu_ret_l1_nonnull": int(sub["gpu_ret_l1"].notna().sum()),
            "complete_reg_rows": int(reg_df[reg_df["group"] == group].shape[0]),
            "dropped_rows": int(len(sub) - reg_df[reg_df["group"] == group].shape[0]),
            "ffill_used": False,
        })
    return pd.DataFrame(rows)


lag_audit = pd.concat(
    [
        h2_lag_audit(h2_chip, chip_reg, "chip_main"),
        h2_lag_audit(h2_vram, vram_reg, "vram_proxy"),
    ],
    ignore_index=True,
)

print("H2 lag missingness audit")
display(lag_audit)

lag_audit.to_csv(H2_DIR / "lag_na.csv", index=False)

x_cols = (
    ["is_high", "is_low"]
    + btc_lags
    + hi_btc
    + low_btc
    + ["fx_ret", "nvda_ret", "high_fx_ret", "low_fx_ret", "high_nvda_ret", "low_nvda_ret", "gpu_ret_l1"]
)


def fit_int(dat, block_name):
    y = dat["gpu_ret"].astype(float)
    X = sm.add_constant(dat[x_cols].astype(float), has_constant="add")

    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": dat["date"]}
    )

    coef = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "se": model.bse.values,
        "t": model.tvalues.values,
        "p_value": model.pvalues.values,
        "block": block_name,
        "base_group": BASE,
    })
    return model, coef


chip_mod, chip_coef = fit_int(chip_reg, "chip_main")
vram_mod, vram_coef = fit_int(vram_reg, "vram_proxy")

print(chip_mod.summary())
print("chip main coefficient table")
display(chip_coef.round(5))

chip_coef.to_csv(H2_DIR / "chip_coef.csv", index=False)
vram_coef.to_csv(H2_DIR / "vram_coef.csv", index=False)

h2_resid = chip_reg[["date", "group", "gpu_ret"]].copy()
h2_resid["fitted"] = chip_mod.fittedvalues
h2_resid["resid"] = chip_mod.resid
h2_resid["std_resid"] = h2_resid["resid"] / h2_resid["resid"].std(ddof=1)
h2_resid["abs_std_resid"] = h2_resid["std_resid"].abs()

h2_resid_by_date = (
    h2_resid.groupby("date", as_index=False)
    .agg(mean_resid=("resid", "mean"), max_abs_std_resid=("abs_std_resid", "max"))
)
h2_resid_spikes = h2_resid.sort_values("abs_std_resid", ascending=False).head(10)

jb = stats.jarque_bera(h2_resid["resid"].dropna())
h2_resid_diag = pd.DataFrame([{
    "n": len(h2_resid),
    "resid_mean": h2_resid["resid"].mean(),
    "resid_std": h2_resid["resid"].std(ddof=1),
    "resid_skew": stats.skew(h2_resid["resid"].dropna()),
    "resid_kurtosis": stats.kurtosis(h2_resid["resid"].dropna(), fisher=False),
    "jarque_bera_stat": float(jb.statistic),
    "jarque_bera_p": float(jb.pvalue),
}])

print("H2 main residual distribution diagnostic")
display(h2_resid_diag.round(5))
print("Top H2 residual spikes")
display(h2_resid_spikes.round(5))

h2_resid.to_csv(H2_DIR / "chip_resid.csv", index=False)
h2_resid_by_date.to_csv(H2_DIR / "chip_resid_date.csv", index=False)
h2_resid_spikes.to_csv(H2_DIR / "chip_resid_top.csv", index=False)
h2_resid_diag.to_csv(H2_DIR / "chip_resid_diag.csv", index=False)

plt.figure(figsize=(11, 4))
plt.plot(h2_resid_by_date["date"], h2_resid_by_date["mean_resid"], marker="o", linewidth=1)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("H2 chip main model: date-level mean residual")
plt.xlabel("Date")
plt.ylabel("Mean residual")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

plt.figure(figsize=(5, 5))
sm.qqplot(h2_resid["resid"].dropna(), line="45", fit=True)
plt.title("H2 chip main residual Q-Q plot")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

def joint_test(model, terms, label, block):
    names = list(model.params.index)
    R = np.zeros((len(terms), len(names)))
    for i, term in enumerate(terms):
        R[i, names.index(term)] = 1.0
    test = model.f_test(R)
    return {
        "block": block,
        "test": label,
        "F_stat": float(np.asarray(test.fvalue).ravel()[0]),
        "p_value": float(np.asarray(test.pvalue).ravel()[0]),
        "df_num": float(test.df_num),
        "df_denom": float(test.df_denom),
    }


def joint_diff_test(model, pos_terms, neg_terms, label, block):
    names = list(model.params.index)
    R = np.zeros((len(pos_terms), len(names)))
    for i, (pos, neg) in enumerate(zip(pos_terms, neg_terms)):
        R[i, names.index(pos)] = 1.0
        R[i, names.index(neg)] = -1.0
    test = model.f_test(R)
    return {
        "block": block,
        "test": label,
        "F_stat": float(np.asarray(test.fvalue).ravel()[0]),
        "p_value": float(np.asarray(test.pvalue).ravel()[0]),
        "df_num": float(test.df_num),
        "df_denom": float(test.df_denom),
    }


def lin_combo_test(model, weights, label, block):
    names = list(model.params.index)
    R = np.zeros((1, len(names)))
    for term, weight in weights.items():
        R[0, names.index(term)] += weight
    test = model.t_test(R)
    return {
        "block": block,
        "test": label,
        "estimate": float(np.asarray(test.effect).ravel()[0]),
        "se": float(np.asarray(test.sd).ravel()[0]),
        "t": float(np.asarray(test.tvalue).ravel()[0]),
        "p_value": float(np.asarray(test.pvalue).ravel()[0]),
    }


def cum_test(model, terms, label, block):
    return lin_combo_test(model, {term: 1.0 for term in terms}, label, block)


def cum_diff_test(model, pos_terms, neg_terms, label, block):
    weights = {term: 1.0 for term in pos_terms}
    for term in neg_terms:
        weights[term] = weights.get(term, 0.0) - 1.0
    return lin_combo_test(model, weights, label, block)


def h2_test_block(model, block):
    joint_rows = [
        joint_test(model, hi_btc, "high_vs_mid_BTC_lags_joint_zero", block),
        joint_test(model, low_btc, "low_vs_mid_BTC_lags_joint_zero", block),
        joint_diff_test(model, hi_btc, low_btc, "high_vs_low_BTC_lags_joint_zero", block),
        joint_test(model, hi_btc + low_btc, "all_group_BTC_interactions_joint_zero", block),
    ]

    cumulative_rows = [
        cum_test(model, btc_lags, "mid_cumulative_BTC_effect", block),
        cum_test(model, btc_lags + hi_btc, "high_cumulative_BTC_effect", block),
        cum_test(model, btc_lags + low_btc, "low_cumulative_BTC_effect", block),
        cum_test(model, hi_btc, "high_minus_mid_cumulative_BTC_effect", block),
        cum_test(model, low_btc, "low_minus_mid_cumulative_BTC_effect", block),
        cum_diff_test(model, hi_btc, low_btc, "high_minus_low_cumulative_BTC_effect", block),
    ]
    return pd.DataFrame(joint_rows), pd.DataFrame(cumulative_rows)

joint_chip, cumulative_chip = h2_test_block(chip_mod, "chip_main")
joint_vram, cumulative_vram = h2_test_block(vram_mod, "vram_proxy")

main_joint = pd.concat([joint_chip, joint_vram], ignore_index=True)
main_cum = pd.concat([cumulative_chip, cumulative_vram], ignore_index=True)

print("H2 BTC interaction joint tests, mid as base")
display(main_joint.round(5))
print("H2 cumulative BTC effects, mid as base")
display(main_cum.round(5))

main_joint.to_csv(H2_DIR / "joint.csv", index=False)
main_cum.to_csv(H2_DIR / "cum.csv", index=False)

ret_chip_nt_w = (
    pd.read_csv(OUT / "ret_chip_notrim_w.csv", parse_dates=["date"])
    .set_index("date")
    .sort_index()
)

h2_chip_nt = make_reg_df(ret_chip_nt_w, "chip_main_no_trim", chip_cols)
chip_nt_reg = h2_chip_nt[model_cols].dropna().copy()

chip_nt_mod, chip_nt_coef = fit_int(chip_nt_reg, "chip_main_no_trim")
nt_joint, nt_cum = h2_test_block(chip_nt_mod, "chip_main_no_trim")

nt_sample = pd.DataFrame([{
    "block": "chip_main_no_trim",
    "raw_rows": len(h2_chip_nt),
    "complete_reg_rows": len(chip_nt_reg),
    "week_n": chip_nt_reg["date"].nunique(),
}])

print("no-trim H2 sample")
display(nt_sample)
print("no-trim H2 joint tests")
display(nt_joint.round(5))
print("no-trim H2 cumulative effects")
display(nt_cum.round(5))

h2_chip_nt.to_csv(H2_DIR / "chip_notrim_raw.csv", index=False)
chip_nt_reg.to_csv(H2_DIR / "chip_notrim_reg.csv", index=False)
chip_nt_coef.to_csv(H2_DIR / "chip_notrim_coef.csv", index=False)
nt_joint.to_csv(H2_DIR / "chip_notrim_joint.csv", index=False)
nt_cum.to_csv(H2_DIR / "chip_notrim_cum.csv", index=False)
nt_sample.to_csv(H2_DIR / "chip_notrim_sample.csv", index=False)

comp_ctrls = ["log_active_product_n", "new_product_share", "bridge_ratio"]

comp_src = (
    comp_diag.rename(columns={"chip_perf_group": "group"})
    [["date", "group"] + comp_ctrls]
    .copy()
)

comp_reg = chip_reg.merge(comp_src, on=["date", "group"], how="left")

comp_reg = comp_reg.dropna(subset=["gpu_ret"] + x_cols + comp_ctrls).copy()

comp_x = x_cols + comp_ctrls

def fit_custom(dat, rhs, block_name):
    y = dat["gpu_ret"].astype(float)
    X = sm.add_constant(dat[rhs].astype(float), has_constant="add")
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": dat["date"]}
    )
    coef = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "se": model.bse.values,
        "t": model.tvalues.values,
        "p_value": model.pvalues.values,
        "block": block_name,
    })
    return model, coef

h2_comp_model, h2_comp_coef = fit_custom(
    comp_reg,
    comp_x,
    "chip_main_composition_control",
)

comp_joint, comp_cum = h2_test_block(
    h2_comp_model,
    "chip_main_composition_control",
)

composition_sample = pd.DataFrame([{
    "block": "chip_main_composition_control",
    "obs_n": len(comp_reg),
    "week_n": comp_reg["date"].nunique(),
    "group_n": comp_reg["group"].nunique(),
}])

print("composition-control model summary")
print(h2_comp_model.summary())
print("composition-control sample")
display(composition_sample)
print("composition-control joint tests")
display(comp_joint.round(5))
print("composition-control cumulative effects")
display(comp_cum.round(5))

comp_reg.to_csv(H2_DIR / "comp_reg.csv", index=False)
h2_comp_coef.to_csv(H2_DIR / "comp_coef.csv", index=False)
comp_joint.to_csv(H2_DIR / "comp_joint.csv", index=False)
comp_cum.to_csv(H2_DIR / "comp_cum.csv", index=False)
composition_sample.to_csv(H2_DIR / "comp_sample.csv", index=False)

LAG_WIN = {
    "w0": [0],
    "w1_2": [1, 2],
    "w3_5": [3, 4, 5],
    "w6_8": [6, 7, 8],
}

win_df = chip_reg.copy()

btc_win = []
hi_win = []
low_win = []

for name, lags in LAG_WIN.items():
    base_col = f"btc_{name}"
    high_col = f"high_btc_{name}"
    low_col = f"low_btc_{name}"

    win_df[base_col] = win_df[[f"btc_krw_ret_l{k}" for k in lags]].sum(axis=1)
    win_df[high_col] = win_df["is_high"] * win_df[base_col]
    win_df[low_col] = win_df["is_low"] * win_df[base_col]

    btc_win.append(base_col)
    hi_win.append(high_col)
    low_win.append(low_col)

win_x = (
    ["is_high", "is_low"]
    + btc_win
    + hi_win
    + low_win
    + ["fx_ret", "nvda_ret", "high_fx_ret", "low_fx_ret", "high_nvda_ret", "low_nvda_ret", "gpu_ret_l1"]
)

h2_window_reg = win_df[["gpu_ret", "date", "group"] + win_x].dropna().copy()

h2_window_model, h2_window_coef = fit_custom(
    h2_window_reg,
    win_x,
    "chip_main_lag_window",
)

win_joint = pd.DataFrame([
    joint_test(h2_window_model, hi_win, "window_high_vs_mid_BTC_joint_zero", "chip_main_lag_window"),
    joint_test(h2_window_model, low_win, "window_low_vs_mid_BTC_joint_zero", "chip_main_lag_window"),
    joint_diff_test(h2_window_model, hi_win, low_win, "window_high_vs_low_BTC_joint_zero", "chip_main_lag_window"),
    joint_test(h2_window_model, hi_win + low_win, "window_all_group_BTC_interactions_joint_zero", "chip_main_lag_window"),
])

win_cum = pd.DataFrame([
    cum_test(h2_window_model, btc_win, "window_mid_cumulative_BTC_effect", "chip_main_lag_window"),
    cum_test(h2_window_model, btc_win + hi_win, "window_high_cumulative_BTC_effect", "chip_main_lag_window"),
    cum_test(h2_window_model, btc_win + low_win, "window_low_cumulative_BTC_effect", "chip_main_lag_window"),
    cum_test(h2_window_model, hi_win, "window_high_minus_mid_cumulative_BTC_effect", "chip_main_lag_window"),
    cum_test(h2_window_model, low_win, "window_low_minus_mid_cumulative_BTC_effect", "chip_main_lag_window"),
    cum_diff_test(h2_window_model, hi_win, low_win, "window_high_minus_low_cumulative_BTC_effect", "chip_main_lag_window"),
])

window_sample = pd.DataFrame([{
    "block": "chip_main_lag_window",
    "obs_n": len(h2_window_reg),
    "week_n": h2_window_reg["date"].nunique(),
    "group_n": h2_window_reg["group"].nunique(),
    "base_group": BASE,
    "lag_window_n": len(LAG_WIN),
}])

print("lag-window compact model summary")
print(h2_window_model.summary())
print("lag-window sample")
display(window_sample)
print("lag-window joint tests")
display(win_joint.round(5))
print("lag-window cumulative effects")
display(win_cum.round(5))

h2_window_reg.to_csv(H2_DIR / "win_reg.csv", index=False)
h2_window_coef.to_csv(H2_DIR / "win_coef.csv", index=False)
win_joint.to_csv(H2_DIR / "win_joint.csv", index=False)
win_cum.to_csv(H2_DIR / "win_cum.csv", index=False)
window_sample.to_csv(H2_DIR / "win_sample.csv", index=False)

ALMON_DEGREE = 2
almon_df = chip_reg.copy()
lag_idx = np.arange(L + 1)
lag_power_sum = {d: float(np.sum(lag_idx ** d)) for d in range(ALMON_DEGREE + 1)}

almon_base = []
almon_high = []
almon_low = []

for d in range(ALMON_DEGREE + 1):
    base_col = f"btc_poly_d{d}"
    high_col = f"high_btc_poly_d{d}"
    low_col = f"low_btc_poly_d{d}"

    almon_df[base_col] = 0.0
    for k in range(L + 1):
        almon_df[base_col] += (k ** d) * almon_df[f"btc_krw_ret_l{k}"]

    almon_df[high_col] = almon_df["is_high"] * almon_df[base_col]
    almon_df[low_col] = almon_df["is_low"] * almon_df[base_col]

    almon_base.append(base_col)
    almon_high.append(high_col)
    almon_low.append(low_col)

almon_x = (
    ["is_high", "is_low"]
    + almon_base
    + almon_high
    + almon_low
    + ["fx_ret", "nvda_ret", "high_fx_ret", "low_fx_ret", "high_nvda_ret", "low_nvda_ret", "gpu_ret_l1"]
)

almon_reg = almon_df[["gpu_ret", "date", "group"] + almon_x].dropna().copy()
almon_mod, almon_coef = fit_custom(almon_reg, almon_x, "chip_main_almon")

almon_joint = pd.DataFrame([
    joint_test(almon_mod, almon_high, "almon_high_vs_mid_BTC_joint_zero", "chip_main_almon"),
    joint_test(almon_mod, almon_low, "almon_low_vs_mid_BTC_joint_zero", "chip_main_almon"),
    joint_diff_test(almon_mod, almon_high, almon_low, "almon_high_vs_low_BTC_joint_zero", "chip_main_almon"),
    joint_test(almon_mod, almon_high + almon_low, "almon_all_group_BTC_interactions_joint_zero", "chip_main_almon"),
])

mid_weights = {col: lag_power_sum[d] for d, col in enumerate(almon_base)}
high_weights = mid_weights.copy()
high_weights.update({col: lag_power_sum[d] for d, col in enumerate(almon_high)})
low_weights = mid_weights.copy()
low_weights.update({col: lag_power_sum[d] for d, col in enumerate(almon_low)})
high_minus_mid_weights = {col: lag_power_sum[d] for d, col in enumerate(almon_high)}
low_minus_mid_weights = {col: lag_power_sum[d] for d, col in enumerate(almon_low)}
high_minus_low_weights = {col: lag_power_sum[d] for d, col in enumerate(almon_high)}
for d, col in enumerate(almon_low):
    high_minus_low_weights[col] = -lag_power_sum[d]

almon_cum = pd.DataFrame([
    lin_combo_test(almon_mod, mid_weights, "almon_mid_cumulative_BTC_effect", "chip_main_almon"),
    lin_combo_test(almon_mod, high_weights, "almon_high_cumulative_BTC_effect", "chip_main_almon"),
    lin_combo_test(almon_mod, low_weights, "almon_low_cumulative_BTC_effect", "chip_main_almon"),
    lin_combo_test(almon_mod, high_minus_mid_weights, "almon_high_minus_mid_cumulative_BTC_effect", "chip_main_almon"),
    lin_combo_test(almon_mod, low_minus_mid_weights, "almon_low_minus_mid_cumulative_BTC_effect", "chip_main_almon"),
    lin_combo_test(almon_mod, high_minus_low_weights, "almon_high_minus_low_cumulative_BTC_effect", "chip_main_almon"),
])

almon_sample = pd.DataFrame([{
    "block": "chip_main_almon",
    "obs_n": len(almon_reg),
    "week_n": almon_reg["date"].nunique(),
    "group_n": almon_reg["group"].nunique(),
    "base_group": BASE,
    "almon_degree": ALMON_DEGREE,
    "parameter_note": "BTC lag terms compressed into polynomial basis",
}])

print("Almon distributed lag model summary")
print(almon_mod.summary())
print("Almon sample")
display(almon_sample)
print("Almon joint tests")
display(almon_joint.round(5))
print("Almon cumulative effects")
display(almon_cum.round(5))

almon_reg.to_csv(H2_DIR / "almon_reg.csv", index=False)
almon_coef.to_csv(H2_DIR / "almon_coef.csv", index=False)
almon_joint.to_csv(H2_DIR / "almon_joint.csv", index=False)
almon_cum.to_csv(H2_DIR / "almon_cum.csv", index=False)
almon_sample.to_csv(H2_DIR / "almon_sample.csv", index=False)

def lin_est(model, terms, label, group, lag, block):
    names = list(model.params.index)
    R = np.zeros((1, len(names)))
    for term in terms:
        R[0, names.index(term)] += 1.0
    test = model.t_test(R)
    return {
        "block": block,
        "base_group": BASE,
        "group": group,
        "lag_week": lag,
        "term_label": label,
        "coef": float(np.asarray(test.effect).ravel()[0]),
        "se": float(np.asarray(test.sd).ravel()[0]),
        "p_value": float(np.asarray(test.pvalue).ravel()[0]),
    }


def lag_coef_table(model, block):
    rows = []
    for k in range(L + 1):
        base = f"btc_krw_ret_l{k}"
        rows.append(lin_est(model, [base], f"mid_l{k}", "mid", k, block))
        rows.append(lin_est(model, [base, f"high_btc_l{k}"], f"high_l{k}", "high", k, block))
        rows.append(lin_est(model, [base, f"low_btc_l{k}"], f"low_l{k}", "low", k, block))
    return pd.DataFrame(rows)

chip_lag_coef = lag_coef_table(chip_mod, "chip_main")
vram_lag_coef = lag_coef_table(vram_mod, "vram_proxy")

print("chip group-specific BTC lag coefficients")
display(chip_lag_coef.round(5))
chip_lag_coef.to_csv(H2_DIR / "chip_lag.csv", index=False)
vram_lag_coef.to_csv(H2_DIR / "vram_lag.csv", index=False)

plt.figure(figsize=(9, 4))
for group in GROUPS:
    sub = chip_lag_coef[chip_lag_coef["group"] == group].sort_values("lag_week")
    plt.errorbar(
        sub["lag_week"],
        sub["coef"],
        yerr=1.96 * sub["se"],
        marker="o",
        capsize=3,
        label=group,
    )
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Hypo 2: group-specific BTC lag coefficients, chip main")
plt.xlabel("BTC-KRW return lag, weeks")
plt.ylabel("Coefficient on BTC-KRW return")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

def get_vif(df, cols, block):
    X = df[cols].dropna().astype(float).copy()
    X = sm.add_constant(X)
    rows = []
    for i, col in enumerate(X.columns):
        rows.append({
            "block": block,
            "variable": col,
            "VIF": variance_inflation_factor(X.values, i),
        })
    return pd.DataFrame(rows)

vif_core = get_vif(chip_reg, ["btc_krw_ret_l0", "fx_ret", "nvda_ret", "gpu_ret_l1"], "chip_core")
vif_full = get_vif(chip_reg, btc_lags + ["fx_ret", "nvda_ret", "gpu_ret_l1"], "chip_lag_full")

print("VIF core")
display(vif_core.round(3))
print("VIF with BTC lags")
display(vif_full.round(3))

vif_core.to_csv(H2_DIR / "vif_core.csv", index=False)
vif_full.to_csv(H2_DIR / "vif_full.csv", index=False)

print("VRAM proxy model summary")
print(vram_mod.summary())

print("VRAM proxy joint tests")
display(main_joint[main_joint["block"] == "vram_proxy"].round(5))

print("VRAM proxy cumulative effects")
display(main_cum[main_cum["block"] == "vram_proxy"].round(5))

plt.figure(figsize=(11, 4))
for g in vram_cols:
    plt.plot(idx_vram_w.index, idx_vram_w[g], label=f"VRAM {g}")
plt.axhline(100, linestyle="--", linewidth=1)
plt.title("Hypo 2 robustness: VRAM proxy group price index")
plt.xlabel("Date")
plt.ylabel("Index, base=100")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

plt.figure(figsize=(9, 4))
for group in GROUPS:
    sub = vram_lag_coef[vram_lag_coef["group"] == group].sort_values("lag_week")
    plt.errorbar(
        sub["lag_week"],
        sub["coef"],
        yerr=1.96 * sub["se"],
        marker="o",
        capsize=3,
        label=group,
    )
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Hypo 2: group-specific BTC lag coefficients, VRAM proxy")
plt.xlabel("BTC-KRW return lag, weeks")
plt.ylabel("Coefficient on BTC-KRW return")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

panel = panel_gpu[panel_gpu["chip_perf_group"].isin(GROUPS)].copy()
panel = panel.merge(mkt.reset_index().rename(columns={mkt.index.name or "index": "date"}), on="date", how="inner")
panel = panel.sort_values(["row_id", "date"]).copy()

panel["is_high"] = (panel["chip_perf_group"] == "high").astype(int)
panel["is_low"] = (panel["chip_perf_group"] == "low").astype(int)

for col in btc_lags:
    lag = col.replace("btc_krw_ret_l", "")
    panel[f"high_btc_l{lag}"] = panel["is_high"] * panel[col]
    panel[f"low_btc_l{lag}"] = panel["is_low"] * panel[col]

panel["high_nvda_ret"] = panel["is_high"] * panel["nvda_ret"]
panel["low_nvda_ret"] = panel["is_low"] * panel["nvda_ret"]
panel["high_fx_ret"] = panel["is_high"] * panel["fx_ret"]
panel["low_fx_ret"] = panel["is_low"] * panel["fx_ret"]

prod_x = (
    btc_lags
    + hi_btc
    + low_btc
    + ["fx_ret", "nvda_ret", "high_fx_ret", "low_fx_ret", "high_nvda_ret", "low_nvda_ret"]
)
prod_reg = panel[["row_id", "date", "gpu_log_return", "chip_perf_group"] + prod_x].dropna().copy()

within_cols = ["gpu_log_return"] + prod_x
prod_dm = prod_reg[within_cols] - prod_reg.groupby("row_id")[within_cols].transform("mean")

y_fe = prod_dm["gpu_log_return"].astype(float)
X_fe = prod_dm[prod_x].astype(float)

prod_mod = sm.OLS(y_fe, X_fe).fit(
    cov_type="cluster",
    cov_kwds={"groups": prod_reg["date"]}
)

prod_coef = pd.DataFrame({
    "term": prod_mod.params.index,
    "coef": prod_mod.params.values,
    "se": prod_mod.bse.values,
    "t": prod_mod.tvalues.values,
    "p_value": prod_mod.pvalues.values,
    "base_group": BASE,
})

prod_sample = pd.DataFrame([{
    "obs_n": len(prod_reg),
    "product_n": prod_reg["row_id"].nunique(),
    "week_n": prod_reg["date"].nunique(),
    "base_group": BASE,
}])

print(prod_mod.summary())
print("product FE panel sample")
display(prod_sample)
print("product FE coefficient table")
display(prod_coef.round(5))

prod_joint, prod_cum = h2_test_block(prod_mod, "product_FE")

print("product FE joint tests")
display(prod_joint.round(5))
print("product FE cumulative effects")
display(prod_cum.round(5))

prod_sample.to_csv(H2_DIR / "prod_fe_sample.csv", index=False)
prod_coef.to_csv(H2_DIR / "prod_fe_coef.csv", index=False)
prod_joint.to_csv(H2_DIR / "prod_fe_joint.csv", index=False)
prod_cum.to_csv(H2_DIR / "prod_fe_cum.csv", index=False)

sum_rows = []

for _, row in anova_results.iterrows():
    if row["test"] in ["One_way_ANOVA", "Welch_ANOVA"]:
        sum_rows.append({
            "block": row["block"],
            "metric": row["test"],
            "estimate": row["stat"],
            "p_value": row["p_value"],
            "interpretation_target": "group mean return difference, exploratory only",
        })

joint_list = [main_joint, nt_joint, comp_joint, win_joint, prod_joint]
cum_list = [main_cum, nt_cum, comp_cum, win_cum, prod_cum]

if "almon_joint" in globals():
    joint_list.append(almon_joint)
if "almon_cum" in globals():
    cum_list.append(almon_cum)

for table in joint_list:
    for _, row in table.iterrows():
        sum_rows.append({
            "block": row["block"],
            "metric": row["test"],
            "estimate": row["F_stat"],
            "p_value": row["p_value"],
            "interpretation_target": "joint significance of BTC sensitivity difference",
        })

for table in cum_list:
    for _, row in table.iterrows():
        sum_rows.append({
            "block": row["block"],
            "metric": row["test"],
            "estimate": row["estimate"],
            "p_value": row["p_value"],
            "interpretation_target": "cumulative BTC response over lag structure",
        })

if "scatter_summary" in globals() and len(scatter_summary) > 0:
    for _, row in scatter_summary.iterrows():
        sum_rows.append({
            "block": "chip_scatter_eda",
            "metric": f"{row['group']}_simple_beta_l0",
            "estimate": row["simple_beta_l0"],
            "p_value": row.get("simple_beta_p", np.nan),
            "interpretation_target": "uncontrolled visual slope; main conclusion relies on interaction regression",
        })

if "ccf_group" in globals() and len(ccf_group) > 0:
    for _, row in ccf_group.iterrows():
        if bool(row.get("outside_band", False)):
            sum_rows.append({
                "block": "chip_group_ccf_eda",
                "metric": f"{row['group']}_lag{int(row['lag_week'])}_outside_band",
                "estimate": row["corr"],
                "p_value": np.nan,
                "interpretation_target": "CCF visual signal; not confirmatory evidence",
            })

if "vol_summary" in globals() and len(vol_summary) > 0:
    for _, row in vol_summary.iterrows():
        sum_rows.append({
            "block": "chip_volatility_eda",
            "metric": f"{row['group']}_corr_sqret_absbtc",
            "estimate": row["corr_sqret_absbtc"],
            "p_value": np.nan,
            "interpretation_target": "volatility clustering diagnostic for robust standard errors",
        })

if "h2_resid_diag" in globals() and len(h2_resid_diag) > 0:
    sum_rows.append({
        "block": "chip_residual_diagnostic",
        "metric": "jarque_bera_p",
        "estimate": float(h2_resid_diag.loc[0, "jarque_bera_stat"]),
        "p_value": float(h2_resid_diag.loc[0, "jarque_bera_p"]),
        "interpretation_target": "residual fat-tail diagnostic; visualized with residual plot and Q-Q plot",
    })

if "h2_regime_beta" in globals() and len(h2_regime_beta) > 0:
    for _, row in h2_regime_beta.iterrows():
        sum_rows.append({
            "block": "chip_regime_eda",
            "metric": f"{row['regime_type']}:{row['regime']}:{row['group']}_beta_l0",
            "estimate": row["simple_beta_l0"],
            "p_value": np.nan,
            "interpretation_target": "regime-specific descriptive beta; not confirmatory evidence",
        })

h2_sum = pd.DataFrame(sum_rows)

print("Hypo 2 summary")
display(h2_sum.round(5))
h2_sum.to_csv(H2_DIR / "summary.csv", index=False)

print("saved files to:", H2_DIR)
print("main files:")
for name in [
    "vram_chip_n.csv",
    "vram_mismatch.csv",
    "comp_diag.csv",
    "comp_sum.csv",
    "anova.csv",
    "lag_na.csv",
    "scatter_beta.csv",
    "ccf_group.csv",
    "vol_sum.csv",
    "regime_beta.csv",
    "chip_coef.csv",
    "joint.csv",
    "cum.csv",
    "chip_lag.csv",
    "vif_core.csv",
    "vif_full.csv",
    "chip_notrim_joint.csv",
    "chip_notrim_cum.csv",
    "comp_coef.csv",
    "comp_joint.csv",
    "comp_cum.csv",
    "win_coef.csv",
    "win_joint.csv",
    "win_cum.csv",
    "almon_coef.csv",
    "almon_joint.csv",
    "almon_cum.csv",
    "prod_fe_coef.csv",
    "prod_fe_joint.csv",
    "prod_fe_cum.csv",
    "summary.csv",
]:
    print("-", H2_DIR / name)
