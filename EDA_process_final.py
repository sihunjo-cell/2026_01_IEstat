# Auto-generated from EDA_process_final.ipynb (code cells only, markdown/outputs stripped)

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


START_DATE = "2024-03-01"
END_DATE = "2026-03-31"

BASE_INDEX = 100.0
WEEKLY_FREQ = "W-SUN"
WEEKLY_AGG = "median"
AGG_CHECK = ["median", "last"]

MIN_DAYS = 3

RET_Q = 0.05
RET_MIN = 20
RET_MAX = np.log(1.8)

MIN_ALL = 10
MIN_GRP = 3

ROOT = Path(".")
OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SPECS = {
    "CPU": {
        "folder": ROOT / "CPU_PREPARE",
        "pattern": "processed_*_CPU.csv",
        "merge_cols": ["Id", "Name", "cpu_model", "cpu_codename", "sale_info"],
    },
    "GPU": {
        "folder": ROOT / "GPU_PREPARE",
        "pattern": "processed_*_GPU.csv",
        "merge_cols": ["Id", "Name", "gpu_model", "capacity", "tail_paren", "name_rest"],
    },
    "RAM": {
        "folder": ROOT / "RAM_PREPARE",
        "pattern": "processed_*_RAM.csv",
        "merge_cols": ["Id", "Name", "ram_model", "memory_gen", "memory_speed", "cl_value", "capacity", "capacity_gb"],
    },
}


def load_files(folder, pattern, merge_cols):
    files = sorted(Path(folder).glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"processed 파일을 찾지 못했습니다: {folder / pattern}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True, sort=False)

    meta = pd.DataFrame({
        "file_n": [len(files)],
        "row_n": [len(raw)],
        "col_n": [raw.shape[1]],
    })
    return raw, meta


def load_report(df, meta_cols):
    date_cols = [c for c in df.columns if DATE_RE.fullmatch(str(c).strip())]
    valid_counts = df[date_cols].notna().sum(axis=1) if date_cols else pd.Series(0, index=df.index)
    return pd.DataFrame({
        "row_n": [len(df)],
        "date_n": [len(date_cols)],
        "max_obs": [valid_counts.max()],
        "full_row_n": [(valid_counts == valid_counts.max()).sum()],
    })


raw_C, meta_C = load_files(**SPECS["CPU"])
raw_G, meta_G = load_files(**SPECS["GPU"])
raw_R, meta_R = load_files(**SPECS["RAM"])

for name, raw, meta in [("CPU", raw_C, meta_C), ("GPU", raw_G, meta_G), ("RAM", raw_R, meta_R)]:
    print(f"[{name}] files: {meta.loc[0, 'file_n']} / rows: {len(raw)} / cols: {raw.shape[1]}")
    display(load_report(raw, SPECS[name]["merge_cols"]))
    display(raw.head(3))

import re as _re
from datetime import datetime as _dt

FIG_DIR = ROOT / "output_pic"
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

def get_dates(df, start=START_DATE, end=END_DATE):
    date_cols = sorted(c for c in df.columns if DATE_RE.fullmatch(str(c)))
    return [c for c in date_cols if start <= c <= end]


def price_mat(df, date_cols):
    X = df.loc[:, date_cols].replace(",", "", regex=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.where(X > 0).astype(float)


g_dates = get_dates(raw_G)
c_dates = get_dates(raw_C)
r_dates = get_dates(raw_R)

for name, dates in [("GPU", g_dates), ("CPU", c_dates), ("RAM", r_dates)]:
    print(f"{name}: {len(dates)} date columns ({dates[0]} ~ {dates[-1]})")

price_check = pd.DataFrame({
    "dataset": ["GPU", "CPU", "RAM"],
    "rows": [len(raw_G), len(raw_C), len(raw_R)],
    "date_cols": [len(g_dates), len(c_dates), len(r_dates)],
    "non_missing_price_cells": [
        price_mat(raw_G, g_dates).notna().sum().sum(),
        price_mat(raw_C, c_dates).notna().sum().sum(),
        price_mat(raw_R, r_dates).notna().sum().sum(),
    ],
})
display(price_check)

def date_gap(date_cols, name):
    observed = pd.to_datetime(date_cols)
    full = pd.date_range(observed.min(), observed.max(), freq="D")
    missing = full.difference(observed)

    report = pd.DataFrame({
        "dataset": [name],
        "observed_days": [len(observed)],
        "full_calendar_days": [len(full)],
        "missing_calendar_days": [len(missing)],
        "missing_ratio": [len(missing) / len(full)],
    })

    missing_weekday = (
        pd.Series(missing.day_name()).value_counts().rename_axis("weekday").reset_index(name="missing_cnt")
        if len(missing) else pd.DataFrame(columns=["weekday", "missing_cnt"])
    )
    return report, missing_weekday


for name, dates in [("GPU", g_dates), ("CPU", c_dates), ("RAM", r_dates)]:
    report, missing_weekday = date_gap(dates, name)
    display(report)
    display(missing_weekday)

def cap_gb(capacity):
    text = str(capacity).upper().replace(" ", "")
    if text in {"", "0", "NAN", "NONE"}:
        return np.nan

    gb = re.search(r"(\d+(?:\.\d+)?)GB", text)
    if gb:
        return float(gb.group(1))

    mb = re.search(r"(\d+(?:\.\d+)?)MB", text)
    if mb:
        return float(mb.group(1)) / 1024

    g_only = re.search(r"(\d+(?:\.\d+)?)G$", text)
    if g_only:
        return float(g_only.group(1))

    return np.nan


def get_chip(model):
    text = re.sub(r"\s+", " ", str(model).upper()).strip()
    text = text.replace("GEFORCE", "").replace("지포스", "")
    text = text.replace("RADEON", "").replace("라데온", "")
    text = re.sub(r"\s+", " ", text).strip()

    patterns = [
        (r"\bRTX\s*A\s?(\d{4})\b", lambda m: f"RTX A{m.group(1)}"),
        (r"\bPRO\s*W\s?(\d{4})\b", lambda m: f"PRO W{m.group(1)}"),
        (r"\bRTX\s*(\d{4})\s*(TI)?\s*(SUPER)?", lambda m: f"RTX {m.group(1)} {' '.join(x for x in [m.group(2), m.group(3)] if x)}".strip()),
        (r"\bGTX\s*(\d{3,4})\s*(TI)?\s*(SUPER)?", lambda m: f"GTX {m.group(1)} {' '.join(x for x in [m.group(2), m.group(3)] if x)}".strip()),
        (r"\bGT\s*(\d{3,4})\b", lambda m: f"GT {m.group(1)}"),
        (r"\bRX\s*(\d{3,4})\s*(XTX|XT|GRE)?", lambda m: f"RX {m.group(1)} {m.group(2) or ''}".strip()),
        (r"\bARC\s*A\s?(\d{3,4})\b", lambda m: f"ARC A{m.group(1)}"),
    ]

    for pattern, formatter in patterns:
        match = re.search(pattern, text)
        if match:
            return formatter(match)
    return np.nan


def chip_group(chip):
    chip = str(chip).upper().strip()
    if chip in {"", "NAN", "NONE"}:
        return "unknown"

    if any(x in chip for x in ["RTX A", "QUADRO", "TESLA", "PRO W"]):
        return "workstation"

    rtx = re.search(r"\bRTX\s*(\d{4})\b", chip)
    if rtx:
        tier = int(rtx.group(1)) % 100
        if tier >= 80:
            return "high"
        if tier == 70:
            return "high" if ("TI" in chip or "SUPER" in chip) else "mid"
        return "mid" if tier == 60 else "low"

    gtx = re.search(r"\bGTX\s*(\d{3,4})\b", chip)
    if gtx:
        return "mid" if int(gtx.group(1)) >= 1660 else "low"

    if re.search(r"\bGT\s*(\d{3,4})\b", chip):
        return "low"

    rx = re.search(r"\bRX\s*(\d{3,4})\b", chip)
    if rx:
        num = int(rx.group(1))
        tier = num % 100 if num >= 9000 else ((num // 100) % 10) * 10
        if tier >= 80:
            return "high"
        if tier == 70:
            return "high" if any(x in chip for x in ["XTX", "XT", "GRE"]) else "mid"
        return "mid" if tier == 60 else "low"

    arc = re.search(r"\bARC\s*A\s?(\d{3,4})\b", chip)
    if arc:
        return "mid" if int(arc.group(1)) in {770, 750, 580} else "low"

    return "unknown"


def vram_group(vram_gb):
    if pd.isna(vram_gb) or vram_gb <= 0:
        return "unknown"
    if vram_gb >= 12:
        return "high"
    return "mid" if vram_gb > 6 else "low"


EX_GPU = r"노트북|모바일|LAPTOP|MOBILE|MAX-Q|NOTEBOOK"

def prep_gpu(gpu_df):
    g = gpu_df.copy()

    text_cols = [c for c in ["Name", "gpu_model", "name_rest"] if c in g.columns]
    txt = g[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    g = g[~txt.str.contains(EX_GPU, case=False, regex=True, na=False)].copy()

    g["vram_gb"] = g["capacity"].apply(cap_gb)
    g["gpu_chip_clean"] = g["gpu_model"].apply(get_chip)
    g["chip_perf_group"] = g["gpu_chip_clean"].apply(chip_group)
    g["h2_perf_group"] = g["vram_gb"].apply(vram_group)
    return g


gpu_meta = prep_gpu(raw_G)
gpu_cards = gpu_meta[gpu_meta["chip_perf_group"].isin(["low", "mid", "high"]) & gpu_meta["vram_gb"].notna()].copy()

print("GPU rows after 1st preprocessing:", len(raw_G))
print("GPU rows used in main analysis:", len(gpu_cards))

print("chip_perf_group, main H2 group")
display(gpu_cards["chip_perf_group"].value_counts(dropna=False))

print("h2_perf_group, VRAM proxy")
display(gpu_cards["h2_perf_group"].value_counts(dropna=False))

print("chip group vs VRAM proxy")
display(pd.crosstab(gpu_cards["chip_perf_group"], gpu_cards["h2_perf_group"]))

audit_cols = ["Id", "Name", "gpu_model", "capacity", "vram_gb", "gpu_chip_clean", "chip_perf_group", "h2_perf_group"]
display(gpu_cards[audit_cols].head(20))

KEYS = {
    "GPU": ["Id"],
    "CPU": ["Id"],
    "RAM": ["Id", "capacity"],
}


def dup_spread(df, date_cols, key_cols, top=20):
    X = price_mat(df, date_cols)
    keys = df[key_cols].reset_index(drop=True).copy()
    parts = []

    for col in date_cols:
        temp = keys.copy()
        temp["price"] = X[col].to_numpy()
        temp = temp.dropna(subset=["price"])
        if temp.empty:
            continue

        spread = (
            temp.groupby(key_cols, dropna=False)["price"]
                .agg(["count", "min", "median", "max"])
                .reset_index()
        )
        spread = spread[spread["count"] >= 2].copy()
        if len(spread) == 0:
            continue

        spread["date"] = col
        spread["max_min_ratio"] = spread["max"] / spread["min"]
        parts.append(spread)

    cols = key_cols + ["date", "count", "min", "median", "max", "max_min_ratio"]
    if len(parts) == 0:
        return pd.DataFrame(columns=cols)

    return (
        pd.concat(parts, ignore_index=True)[cols]
        .sort_values("max_min_ratio", ascending=False)
        .head(top)
    )


def merge_dups(df, date_cols, key_cols):
    X = price_mat(df, date_cols)

    price_agg = (
        pd.concat([df[key_cols].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        .groupby(key_cols, dropna=False)[date_cols]
        .median()
        .reset_index()
    )

    meta_cols = [c for c in df.columns if c not in date_cols]
    extra_cols = [c for c in meta_cols if c not in key_cols]
    meta_agg = df[key_cols + extra_cols].groupby(key_cols, dropna=False).first().reset_index()
    return meta_agg.merge(price_agg, on=key_cols, how="right")


def filter_cpu(df):
    out = df.copy()
    name = out["Name"].astype(str)
    out = out[~name.str.contains("중고|XEON|EPYC|제온|옵테론", case=False, regex=True, na=False)].copy()
    out = out[~out["sale_info"].astype(str).str.contains("중고", na=False)].copy()
    return out


def filter_ram(df):
    out = df.copy()
    out["capacity_gb"] = pd.to_numeric(out["capacity_gb"], errors="coerce")
    out = out[out["memory_gen"].astype(str).str.upper().isin(["DDR4", "DDR5"])].copy()
    out = out[out["capacity_gb"].notna() & (out["capacity_gb"] > 0)].copy()
    return out


cpu_main = filter_cpu(raw_C)
ram_main = filter_ram(raw_R)

print("duplicate merge keys")
print("GPU:", KEYS["GPU"])
print("CPU:", KEYS["CPU"])
print("RAM:", KEYS["RAM"])

print("GPU duplicate price spread")
display(dup_spread(gpu_cards, g_dates, KEYS["GPU"]))
print("CPU duplicate price spread")
display(dup_spread(cpu_main, c_dates, KEYS["CPU"]))
print("RAM duplicate price spread")
display(dup_spread(ram_main, r_dates, KEYS["RAM"]))

gpu_dedup = merge_dups(gpu_cards, g_dates, KEYS["GPU"])
cpu_dedup = merge_dups(cpu_main, c_dates, KEYS["CPU"])
ram_dedup = merge_dups(ram_main, r_dates, KEYS["RAM"])

print("GPU before/after dedup:", len(gpu_cards), len(gpu_dedup))
print("CPU raw/main/after dedup:", len(raw_C), len(cpu_main), len(cpu_dedup))
print("RAM raw/main/after dedup:", len(raw_R), len(ram_main), len(ram_dedup))

print("GPU chip_perf_group after dedup")
display(gpu_dedup["chip_perf_group"].value_counts(dropna=False))

def cov_table(df, date_cols, group_col=None):
    X = price_mat(df, date_cols)
    out = df.copy()
    out["obs_cnt"] = X.notna().sum(axis=1)
    out["coverage"] = X.notna().mean(axis=1)
    out["first_price_date"] = X.apply(lambda r: r.dropna().index[0] if r.notna().any() else np.nan, axis=1)
    out["last_price_date"] = X.apply(lambda r: r.dropna().index[-1] if r.notna().any() else np.nan, axis=1)

    if group_col is None:
        summary = pd.DataFrame({
            "product_n": [len(out)],
            "median_coverage": [out["coverage"].median()],
            "min_coverage": [out["coverage"].min()],
            "max_coverage": [out["coverage"].max()],
            "median_obs_cnt": [out["obs_cnt"].median()],
        })
    else:
        summary = (
            out.groupby(group_col, dropna=False)
               .agg(
                   product_n=("obs_cnt", "size"),
                   median_coverage=("coverage", "median"),
                   min_coverage=("coverage", "min"),
                   max_coverage=("coverage", "max"),
                   median_obs_cnt=("obs_cnt", "median"),
               )
               .reset_index()
        )
    return out, summary


gpu_cov_detail, gpu_chip_cov_summary = cov_table(gpu_dedup, g_dates, group_col="chip_perf_group")
_, gpu_vram_cov_summary = cov_table(gpu_dedup, g_dates, group_col="h2_perf_group")

print("H2 main: chip group coverage")
display(gpu_chip_cov_summary)
print("H2 auxiliary: VRAM proxy coverage")
display(gpu_vram_cov_summary)

def weekly_mat(df, date_cols, freq=WEEKLY_FREQ, agg=WEEKLY_AGG, min_days=MIN_DAYS):
    X = price_mat(df, date_cols).T
    X.index = pd.to_datetime(date_cols)

    obs_n = X.resample(freq).count()
    if agg == "last":
        weekly = X.resample(freq).agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    else:
        weekly = getattr(X.resample(freq), agg)()

    weekly = weekly.where(obs_n >= min_days)
    weekly = weekly.T
    weekly.columns = [d.strftime("%Y-%m-%d") for d in weekly.columns]
    return weekly, list(weekly.columns)


def log_price(X):
    X = X.astype(float).where(X > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(X)
    return out.replace([np.inf, -np.inf], np.nan)


def lr_mean(r, q=RET_Q, trim_min=RET_MIN, abs_max=RET_MAX):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    n0 = len(r)
    if n0 == 0:
        return np.nan, 0, 0

    if abs_max is not None:
        r = r[r.abs() <= abs_max]

    if len(r) == 0:
        return np.nan, 0, int(n0)

    if q > 0 and len(r) >= trim_min:
        lo, hi = r.quantile([q, 1 - q])
        r2 = r[(r >= lo) & (r <= hi)]
    else:
        r2 = r

    return float(r2.mean()), int(len(r2)), int(n0 - len(r2))


def chain_idx(X, date_cols, group_name="ALL", min_matched=5, base=BASE_INDEX, q=RET_Q):
    logX = log_price(X)
    dates = pd.to_datetime(date_cols)

    idx = float(base)
    anchor = 0

    rows = [{
        "date": dates[0],
        "group": group_name,
        "from_date": pd.NaT,
        "index_ffill": idx,
        "index_valid": idx,
        "log_return": np.nan,
        "ret_lvl": np.nan,
        "ret_w": np.nan,
        "n_matched": 0,
        "n_used": 0,
        "n_trim": 0,
        "n_adj": 0,
        "n_adj_used": 0,
        "n_adj_trim": 0,
        "step_n": 0,
        "interval_days": np.nan,
        "bridge": False,
        "valid_step": True,
        "ret_ok": False,
    }]

    for j in range(1, len(date_cols)):
        adj = logX.iloc[:, j] - logX.iloc[:, j - 1]
        n_adj = int(adj.dropna().shape[0])
        adj_lr, n_adj_used, n_adj_trim = lr_mean(adj, q=q)
        ret_ok = n_adj >= min_matched and n_adj_used >= min_matched
        ret_w = adj_lr if ret_ok else np.nan

        ret = logX.iloc[:, j] - logX.iloc[:, anchor]
        n_matched = int(ret.dropna().shape[0])
        lr, n_used, n_trim = lr_mean(ret, q=q)

        step_n = j - anchor
        interval_days = int((dates[j] - dates[anchor]).days)
        from_date = dates[anchor]

        if n_matched >= min_matched and n_used >= min_matched:
            idx *= np.exp(lr)
            index_valid = idx
            ret_lvl = lr / step_n
            bridge = step_n > 1
            valid_step = True
            anchor = j
        else:
            lr = np.nan
            index_valid = np.nan
            ret_lvl = np.nan
            n_used = 0
            n_trim = 0
            bridge = False
            valid_step = False

        rows.append({
            "date": dates[j],
            "group": group_name,
            "from_date": from_date,
            "index_ffill": float(idx),
            "index_valid": index_valid,
            "log_return": lr,
            "ret_lvl": ret_lvl,
            "ret_w": ret_w,
            "n_matched": int(n_matched),
            "n_used": int(n_used),
            "n_trim": int(n_trim),
            "n_adj": int(n_adj),
            "n_adj_used": int(n_adj_used),
            "n_adj_trim": int(n_adj_trim),
            "step_n": int(step_n),
            "interval_days": interval_days,
            "bridge": bool(bridge),
            "valid_step": bool(valid_step),
            "ret_ok": bool(ret_ok),
        })

    return pd.DataFrame(rows)


def build_idx(df, date_cols, group_name, min_matched, agg=WEEKLY_AGG, q=RET_Q):
    X_weekly, weekly_dates = weekly_mat(df, date_cols, agg=agg)
    return chain_idx(X_weekly, weekly_dates, group_name, min_matched, q=q)


def build_group_idx(df, date_cols, group_col, min_matched, keep_groups, agg=WEEKLY_AGG, q=RET_Q):
    parts = []
    for group, sub in df.groupby(group_col, dropna=False):
        group = "unknown" if pd.isna(group) else str(group)
        if group in keep_groups:
            parts.append(build_idx(sub, date_cols, group, min_matched, agg=agg, q=q))
    return pd.concat(parts, ignore_index=True)

idx_gpu = build_idx(gpu_dedup, g_dates, "GPU_ALL", MIN_ALL)
idx_cpu = build_idx(cpu_dedup, c_dates, "CPU_ALL", MIN_ALL)
idx_ram = build_idx(ram_dedup, r_dates, "RAM_ALL", MIN_ALL)

idx_all = pd.concat([idx_gpu, idx_cpu, idx_ram], ignore_index=True)

idx_chip = build_group_idx(
    gpu_dedup,
    g_dates,
    group_col="chip_perf_group",
    min_matched=MIN_GRP,
    keep_groups=["low", "mid", "high"],
)

idx_vram = build_group_idx(
    gpu_dedup,
    g_dates,
    group_col="h2_perf_group",
    min_matched=MIN_GRP,
    keep_groups=["low", "mid", "high"],
)

idx_chip_copy = idx_chip.copy()

print("overall weekly index")
display(idx_all.head())
print("H2 chip main index")
display(idx_chip.head())
print("H2 VRAM proxy auxiliary index")
display(idx_vram.head())

def idx_diag(index_df):
    temp = index_df[index_df["date"] != index_df["date"].min()].copy()
    return (
        temp.groupby("group")
            .agg(
                total_steps=("valid_step", "size"),
                valid_steps=("valid_step", "sum"),
                valid_ratio=("valid_step", "mean"),
                ret_steps=("ret_ok", "sum"),
                ret_valid_ratio=("ret_ok", "mean"),
                bridge_steps=("bridge", "sum"),
                median_n_matched=("n_matched", "median"),
                median_n_used=("n_used", "median"),
                total_n_trim=("n_trim", "sum"),
                median_n_adj=("n_adj", "median"),
                median_n_adj_used=("n_adj_used", "median"),
                total_n_adj_trim=("n_adj_trim", "sum"),
                min_n_adj=("n_adj", "min"),
                max_n_adj=("n_adj", "max"),
                max_step_n=("step_n", "max"),
                max_interval_days=("interval_days", "max"),
                max_abs_ret_w=("ret_w", lambda x: np.nanmax(np.abs(x)) if x.notna().any() else np.nan),
                max_abs_ret_lvl=("ret_lvl", lambda x: np.nanmax(np.abs(x)) if x.notna().any() else np.nan),
                max_abs_log_return=("log_return", lambda x: np.nanmax(np.abs(x)) if x.notna().any() else np.nan),
            )
            .reset_index()
    )


def gap_report(df, date_cols, group_col, agg=WEEKLY_AGG):
    X_weekly, weekly_dates = weekly_mat(df, date_cols, agg=agg)
    groups = df[group_col].reset_index(drop=True)

    rows = []
    for j in range(1, len(weekly_dates)):
        cur = X_weekly.iloc[:, j].notna()
        prev1 = X_weekly.iloc[:, j - 1].notna()
        prev2 = X_weekly.iloc[:, j - 2].notna() if j >= 2 else pd.Series(False, index=X_weekly.index)

        temp = pd.DataFrame({
            "group": groups.values,
            "adj_n": (cur & prev1).values,
            "bridge_n": (cur & ~prev1 & prev2).values,
        })
        summary = (
            temp.groupby("group")
                .agg(adj_n=("adj_n", "sum"), bridge_n=("bridge_n", "sum"))
                .reset_index()
        )
        summary["date"] = pd.to_datetime(weekly_dates[j])
        rows.append(summary)

    by_week = pd.concat(rows, ignore_index=True)
    summary = (
        by_week.groupby("group")
            .agg(
                median_adj_n=("adj_n", "median"),
                median_bridge_n=("bridge_n", "median"),
                max_bridge_n=("bridge_n", "max"),
            )
            .reset_index()
    )
    summary["bridge_adj_ratio"] = summary["median_bridge_n"] / summary["median_adj_n"].replace(0, np.nan)
    return summary, by_week


diag_all = idx_diag(idx_all)
diag_chip = idx_diag(idx_chip)
diag_vram = idx_diag(idx_vram)
diag_chip_copy = diag_chip.copy()

gap_sum, gap_week = gap_report(gpu_dedup, g_dates, group_col="chip_perf_group", agg=WEEKLY_AGG)

print("overall index diagnostics")
display(diag_all)
print("H2 chip diagnostics")
display(diag_chip)
print("H2 VRAM proxy diagnostics")
display(diag_vram)
print("H2 chip gap-loss")
display(gap_sum)
print("H2 chip invalid level weeks")
display(idx_chip[~idx_chip["valid_step"]])
print("H2 chip bridge level weeks")
display(idx_chip[idx_chip["bridge"]])

CHIP_LABEL = {
    "low": "Low chip-tier",
    "mid": "Mid chip-tier",
    "high": "High chip-tier",
}

VRAM_LABEL = {
    "low": "Low VRAM proxy (≤6GB)",
    "mid": "Mid VRAM proxy (6GB<VRAM<12GB)",
    "high": "High VRAM proxy (≥12GB)",
}


def plot_idx(index_df, title, value_col="index_valid", label_map=None, order=None):
    plt.figure(figsize=(13, 5))
    groups = order if order is not None else list(index_df["group"].dropna().unique())

    for group in groups:
        sub = index_df[index_df["group"] == group].sort_values("date")
        if len(sub):
            label = label_map.get(group, group) if label_map else group
            plt.plot(sub["date"], sub[value_col], label=label)

    plt.axhline(100, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Chain matched Jevons index, base=100")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(); plt.show()


plot_idx(idx_all, "Weekly GPU / CPU / RAM Overall Chain Matched Jevons Index")
plot_idx(idx_chip, "Weekly GPU Chip-based Performance Group Chain Matched Jevons Index", label_map=CHIP_LABEL, order=["high", "mid", "low"])
plot_idx(idx_vram, "Weekly GPU VRAM Proxy Chain Matched Jevons Index, Auxiliary", label_map=VRAM_LABEL, order=["high", "mid", "low"])

def idx_summary(index_df, eps=1e-4, jump_threshold=0.08):
    temp = index_df.copy()
    temp["is_valid_return"] = temp["ret_ok"] & temp["ret_w"].notna()
    temp["is_flat"] = temp["is_valid_return"] & (temp["ret_w"].abs() <= eps)
    temp["is_big_jump"] = temp["is_valid_return"] & (temp["ret_w"].abs() >= jump_threshold)
    temp["is_trimmed"] = temp["n_adj_trim"].fillna(0) > 0

    return (
        temp.groupby("group")
            .agg(
                total_rows=("date", "size"),
                valid_return_weeks=("is_valid_return", "sum"),
                invalid_return_weeks=("ret_ok", lambda x: (~x).sum()),
                bridge_weeks=("bridge", "sum"),
                flat_weeks=("is_flat", "sum"),
                big_jump_weeks=("is_big_jump", "sum"),
                trimmed_weeks=("is_trimmed", "sum"),
                median_n_adj=("n_adj", "median"),
                max_abs_ret_w=("ret_w", lambda x: np.nanmax(np.abs(x)) if x.notna().any() else np.nan),
            )
            .reset_index()
    )


def top_moves(index_df, group=None, top=10, value_col="ret_w"):
    temp = index_df.dropna(subset=[value_col]).copy()
    if group is not None:
        temp = temp[temp["group"] == group].copy()
    temp[f"abs_{value_col}"] = temp[value_col].abs()
    return temp.sort_values(f"abs_{value_col}", ascending=False).head(top)


def week_moves(df, date_cols, group_col, group_value, target_week, idx_df=None, top=20):
    sub = df[df[group_col] == group_value].copy()
    X_weekly, weekly_dates = weekly_mat(sub, date_cols, agg=WEEKLY_AGG)

    target_week = pd.to_datetime(target_week).strftime("%Y-%m-%d")
    j = weekly_dates.index(target_week)

    if idx_df is not None:
        row = idx_df[(idx_df["group"] == str(group_value)) & (pd.to_datetime(idx_df["date"]) == pd.to_datetime(target_week))]
        if len(row) and pd.notna(row["from_date"].iloc[0]):
            prev_week = pd.to_datetime(row["from_date"].iloc[0]).strftime("%Y-%m-%d")
        else:
            prev_week = weekly_dates[j - 1]
    else:
        prev_week = weekly_dates[j - 1]

    step_n = max(1, weekly_dates.index(target_week) - weekly_dates.index(prev_week))
    logW = log_price(X_weekly[[prev_week, target_week]])
    r = (logW[target_week] - logW[prev_week]).dropna()
    meta_cols = ["Id", "Name", "vram_gb", "gpu_chip_clean", "chip_perf_group", "h2_perf_group"]

    out = sub.loc[r.index, meta_cols].copy()
    out["prev_week"] = prev_week
    out["target_week"] = target_week
    out["step_n"] = step_n
    out["price_prev"] = X_weekly.loc[r.index, prev_week]
    out["price_target"] = X_weekly.loc[r.index, target_week]
    out["log_return"] = r
    out["ret_lvl"] = r / step_n
    out["ret_w"] = r if step_n == 1 else np.nan
    out["price_ratio"] = np.exp(r)
    out["abs_ret_lvl"] = out["ret_lvl"].abs()
    return out.sort_values("abs_ret_lvl", ascending=False).head(top)


idx_behavior = idx_summary(idx_chip)
print("H2 chip return behavior")
display(idx_behavior)

print("overall largest observed weekly moves")
display(top_moves(idx_all, top=10))
print("H2 chip largest observed weekly moves")
display(top_moves(idx_chip, top=15))

print("H2 chip largest bridge-level moves")
display(top_moves(idx_chip, top=10, value_col="ret_lvl"))

top_high = top_moves(idx_chip, group="high", top=1)
if len(top_high):
    top_high_week = top_high["date"].iloc[0]
    print("high chip group largest observed move week:", top_high_week)
    display(week_moves(gpu_dedup, g_dates, "chip_perf_group", "high", top_high_week, idx_df=idx_chip, top=20))

agg_check = pd.concat(
    [
        build_group_idx(
            gpu_dedup,
            g_dates,
            group_col="chip_perf_group",
            min_matched=MIN_GRP,
            keep_groups=["low", "mid", "high"],
            agg=agg_name,
        ).assign(weekly_agg=agg_name)
        for agg_name in AGG_CHECK
    ],
    ignore_index=True,
)


def plot_agg(index_df, group):
    plt.figure(figsize=(13, 5))
    for agg_name, sub in index_df[index_df["group"] == group].groupby("weekly_agg"):
        sub = sub.sort_values("date")
        plt.plot(sub["date"], sub["index_valid"], label=f"{CHIP_LABEL.get(group, group)} - {agg_name}")

    plt.axhline(100, linestyle="--", linewidth=1)
    plt.title(f"H2 {CHIP_LABEL.get(group, group)}: weekly median vs last")
    plt.xlabel("Date")
    plt.ylabel("Chain matched Jevons index, base=100")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(); plt.show()


for group in ["high", "mid", "low"]:
    plot_agg(agg_check, group)

agg_last = (
    agg_check.dropna(subset=["index_valid"])
    .sort_values("date")
    .groupby(["group", "weekly_agg"])
    .tail(1)
    [["group", "weekly_agg", "date", "index_valid"]]
    .sort_values(["group", "weekly_agg"])
)

display(agg_last)


def fixed_idx(df, date_cols, group_name="ALL", min_coverage=0.70, agg=WEEKLY_AGG, base=BASE_INDEX):
    X_weekly, weekly_dates = weekly_mat(df, date_cols, agg=agg)
    basket = X_weekly.loc[X_weekly.notna().mean(axis=1) >= min_coverage].copy()

    first_valid = basket.apply(lambda r: r.dropna().iloc[0] if r.notna().any() else np.nan, axis=1)
    idx = basket.div(first_valid, axis=0).mul(base).median(axis=0, skipna=True)

    return pd.DataFrame({
        "date": pd.to_datetime(weekly_dates),
        "group": group_name,
        "fixed_basket_index": idx.values,
        "basket_n": len(basket),
    })


def build_fixed_idx(df, date_cols, group_col, keep_groups, min_coverage=0.70, agg=WEEKLY_AGG):
    parts = []
    for group, sub in df.groupby(group_col, dropna=False):
        group = "unknown" if pd.isna(group) else str(group)
        if group in keep_groups:
            parts.append(fixed_idx(sub, date_cols, group, min_coverage, agg, BASE_INDEX))
    return pd.concat(parts, ignore_index=True)


def plot_fixed(chain_df, fixed_df, group):
    plt.figure(figsize=(13, 5))
    chain = chain_df[chain_df["group"] == group].sort_values("date")
    fixed = fixed_df[fixed_df["group"] == group].sort_values("date")

    plt.plot(chain["date"], chain["index_valid"], label=f"{CHIP_LABEL.get(group, group)} - chain Jevons")
    plt.plot(fixed["date"], fixed["fixed_basket_index"], label=f"{CHIP_LABEL.get(group, group)} - fixed basket", linestyle="--")
    plt.axhline(100, linestyle="--", linewidth=1)
    plt.title(f"Chain drift check: {CHIP_LABEL.get(group, group)}")
    plt.xlabel("Date")
    plt.ylabel("Index, base=100")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(); plt.show()


idx_fixed = build_fixed_idx(
    gpu_dedup,
    g_dates,
    group_col="chip_perf_group",
    keep_groups=["low", "mid", "high"],
    min_coverage=0.70,
    agg=WEEKLY_AGG,
)

for group in ["high", "mid", "low"]:
    plot_fixed(idx_chip, idx_fixed, group)

print("fixed basket product count")
display(idx_fixed.groupby("group")["basket_n"].max().reset_index())

idx_all_w = idx_all.pivot(index="date", columns="group", values="index_valid")
ret_all_w = idx_all.pivot(index="date", columns="group", values="ret_w")
ret_all_lvl_w = idx_all.pivot(index="date", columns="group", values="ret_lvl")
ret_all_int_w = idx_all.pivot(index="date", columns="group", values="log_return")

idx_chip_w = idx_chip.pivot(index="date", columns="group", values="index_valid")
ret_chip_w = idx_chip.pivot(index="date", columns="group", values="ret_w")
ret_chip_lvl_w = idx_chip.pivot(index="date", columns="group", values="ret_lvl")
ret_chip_int_w = idx_chip.pivot(index="date", columns="group", values="log_return")

idx_vram_w = idx_vram.pivot(index="date", columns="group", values="index_valid")
ret_vram_w = idx_vram.pivot(index="date", columns="group", values="ret_w")
ret_vram_lvl_w = idx_vram.pivot(index="date", columns="group", values="ret_lvl")
ret_vram_int_w = idx_vram.pivot(index="date", columns="group", values="log_return")

idx_chip_w_copy = idx_chip_w.copy()
ret_chip_w_copy = ret_chip_w.copy()

print("idx_all_w")
display(idx_all_w.head())
print("ret_all_w, observed adjacent weekly return for tests")
display(ret_all_w.head())
print("ret_all_lvl_w, level-link return for index bridge check")
display(ret_all_lvl_w.head())
print("idx_chip_w, chip main")
display(idx_chip_w.head())
print("idx_vram_w, VRAM proxy")
display(idx_vram_w.head())


def make_panel(df, date_cols, freq=WEEKLY_FREQ, agg=WEEKLY_AGG):
    df0 = df.reset_index(drop=True).copy()
    X_weekly, weekly_cols = weekly_mat(df0, date_cols, freq=freq, agg=agg)
    R = log_price(X_weekly).diff(axis=1)

    vals = R.to_numpy()
    row_i, col_i = np.where(np.isfinite(vals))
    panel = pd.DataFrame({
        "row_id": row_i.astype(int),
        "date": pd.to_datetime(np.array(R.columns)[col_i]),
        "gpu_log_return": vals[row_i, col_i],
    })

    meta_cols = ["Id", "Name", "vram_gb", "gpu_chip_clean", "chip_perf_group", "h2_perf_group"]
    meta = df0[meta_cols].copy()
    meta["row_id"] = np.arange(len(meta))

    return panel.merge(meta, on="row_id", how="left")


panel_gpu = make_panel(gpu_dedup, g_dates)

print("product-week return panel")
display(panel_gpu.head())

print("product-week return count by chip group")
display(
    panel_gpu.groupby("chip_perf_group")
    .agg(
        obs_n=("gpu_log_return", "count"),
        product_n=("row_id", "nunique"),
        mean_return=("gpu_log_return", "mean"),
        std_return=("gpu_log_return", "std"),
    )
    .reset_index()
)

idx_gpu_notrim = build_idx(gpu_dedup, g_dates, "GPU_ALL", MIN_ALL, q=0.0)
idx_cpu_notrim = build_idx(cpu_dedup, c_dates, "CPU_ALL", MIN_ALL, q=0.0)
idx_ram_notrim = build_idx(ram_dedup, r_dates, "RAM_ALL", MIN_ALL, q=0.0)
idx_all_notrim = pd.concat([idx_gpu_notrim, idx_cpu_notrim, idx_ram_notrim], ignore_index=True)

idx_chip_notrim = build_group_idx(
    gpu_dedup,
    g_dates,
    group_col="chip_perf_group",
    min_matched=MIN_GRP,
    keep_groups=["low", "mid", "high"],
    q=0.0,
)

idx_vram_notrim = build_group_idx(
    gpu_dedup,
    g_dates,
    group_col="h2_perf_group",
    min_matched=MIN_GRP,
    keep_groups=["low", "mid", "high"],
    q=0.0,
)

idx_all_notrim_w = idx_all_notrim.pivot(index="date", columns="group", values="index_valid")
ret_all_notrim_w = idx_all_notrim.pivot(index="date", columns="group", values="ret_w")
idx_chip_notrim_w = idx_chip_notrim.pivot(index="date", columns="group", values="index_valid")
ret_chip_notrim_w = idx_chip_notrim.pivot(index="date", columns="group", values="ret_w")
idx_vram_notrim_w = idx_vram_notrim.pivot(index="date", columns="group", values="index_valid")
ret_vram_notrim_w = idx_vram_notrim.pivot(index="date", columns="group", values="ret_w")


def compare_trim(main_w, notrim_w, cols, block):
    rows = []
    for col in cols:
        temp = (
            main_w[[col]]
            .rename(columns={col: "main_ret"})
            .join(notrim_w[[col]].rename(columns={col: "notrim_ret"}), how="inner")
            .dropna()
        )
        if len(temp) == 0:
            continue
        diff = temp["notrim_ret"] - temp["main_ret"]
        rows.append({
            "block": block,
            "group": col,
            "n": len(temp),
            "corr_main_notrim": temp["main_ret"].corr(temp["notrim_ret"]),
            "mean_abs_diff": diff.abs().mean(),
            "max_abs_diff": diff.abs().max(),
            "main_std": temp["main_ret"].std(),
            "notrim_std": temp["notrim_ret"].std(),
        })
    return pd.DataFrame(rows)


trim_robust = pd.concat(
    [
        compare_trim(ret_all_w, ret_all_notrim_w, ["GPU_ALL", "CPU_ALL", "RAM_ALL"], "all"),
        compare_trim(ret_chip_w, ret_chip_notrim_w, ["high", "mid", "low"], "chip"),
        compare_trim(ret_vram_w, ret_vram_notrim_w, ["high", "mid", "low"], "vram"),
    ],
    ignore_index=True,
)

print("main vs no-trim return robustness")
display(trim_robust.round(5))


def plot_trim_compare(main_idx_w, notrim_idx_w, col, title):
    temp = (
        main_idx_w[[col]].rename(columns={col: "main"})
        .join(notrim_idx_w[[col]].rename(columns={col: "notrim"}), how="inner")
    )
    plt.figure(figsize=(12, 4))
    plt.plot(temp.index, temp["main"], label="main: 5% trimmed")
    plt.plot(temp.index, temp["notrim"], label="robustness: no quantile trim", linestyle="--")
    plt.axhline(100, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Index, base=100")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(); plt.show()


plot_trim_compare(idx_all_w, idx_all_notrim_w, "GPU_ALL", "Trim sensitivity: GPU overall index")
for group in ["high", "mid", "low"]:
    if group in idx_chip_w.columns:
        plot_trim_compare(idx_chip_w, idx_chip_notrim_w, group, f"Trim sensitivity: chip {group} index")

save_files = {
    "idx_all.csv": idx_all,
    "idx_chip.csv": idx_chip,
    "idx_vram.csv": idx_vram,
    "panel_gpu.csv": panel_gpu,
    "gpu_meta.csv": gpu_dedup,
    "cpu_meta.csv": cpu_dedup,
    "ram_meta.csv": ram_dedup,
    "diag_all.csv": diag_all,
    "diag_chip.csv": diag_chip,
    "diag_vram.csv": diag_vram,
    "diag_ret.csv": idx_behavior,
    "agg_check.csv": agg_last,
    "gap_sum.csv": gap_sum,
    "gap_week.csv": gap_week,
    "fixed_idx.csv": idx_fixed,
    "idx_all_notrim.csv": idx_all_notrim,
    "idx_chip_notrim.csv": idx_chip_notrim,
    "idx_vram_notrim.csv": idx_vram_notrim,
    "trim_robust.csv": trim_robust,
}

save_wide = {
    "idx_all_w.csv": idx_all_w,
    "ret_all_w.csv": ret_all_w,
    "ret_all_lvl_w.csv": ret_all_lvl_w,
    "ret_all_int_w.csv": ret_all_int_w,
    "idx_chip_w.csv": idx_chip_w,
    "ret_chip_w.csv": ret_chip_w,
    "ret_chip_lvl_w.csv": ret_chip_lvl_w,
    "ret_chip_int_w.csv": ret_chip_int_w,
    "idx_vram_w.csv": idx_vram_w,
    "ret_vram_w.csv": ret_vram_w,
    "ret_vram_lvl_w.csv": ret_vram_lvl_w,
    "ret_vram_int_w.csv": ret_vram_int_w,
    "idx_all_notrim_w.csv": idx_all_notrim_w,
    "ret_all_notrim_w.csv": ret_all_notrim_w,
    "idx_chip_notrim_w.csv": idx_chip_notrim_w,
    "ret_chip_notrim_w.csv": ret_chip_notrim_w,
    "idx_vram_notrim_w.csv": idx_vram_notrim_w,
    "ret_vram_notrim_w.csv": ret_vram_notrim_w,
}

for filename, df in save_files.items():
    df.to_csv(OUT / filename, index=False)
for filename, df in save_wide.items():
    df.to_csv(OUT / filename)

print("saved to:", OUT)

mkt_w = pd.read_csv('./outputs/mkt_w.csv')
mkt_w["Date"] = pd.to_datetime(mkt_w["Date"])
mkt_w = mkt_w.set_index("Date").sort_index()
mkt_w.index = mkt_w.index.normalize()

MKT_COLS = ["btc_krw_ret", "fx_ret"]
CHIP_ORDER = [g for g in ["high", "mid", "low"] if g in ret_chip_w.columns]


def common_sample(ret_df, cols, name):
    dat = ret_df[cols].join(mkt_w[MKT_COLS], how="inner")
    return {
        "sample": name,
        "week_n": len(dat),
        "complete_n": dat.dropna().shape[0],
        "hw_nonnull_min": int(dat[cols].notna().sum().min()),
        "mkt_nonnull_min": int(dat[MKT_COLS].notna().sum().min()),
    }


sample_check = pd.DataFrame([
    common_sample(ret_all_w, ["GPU_ALL"], "H1_GPU_vs_BTC"),
    common_sample(ret_chip_w, CHIP_ORDER, "H2_chip_groups"),
    common_sample(ret_all_w, ["GPU_ALL", "CPU_ALL", "RAM_ALL"], "H3_parts"),
])

print("가설별 공통 표본 수")
display(sample_check)


def corr_with_mkt(ret_df, cols):
    dat = ret_df[cols].join(mkt_w[MKT_COLS], how="inner").dropna()
    return dat[cols + MKT_COLS].corr()


corr_all = corr_with_mkt(ret_all_w, ["GPU_ALL", "CPU_ALL", "RAM_ALL"])
corr_chip = corr_with_mkt(ret_chip_w, CHIP_ORDER)

print("H1/H3 return correlation")
display(corr_all.round(3))
print("H2 chip return correlation")
display(corr_chip.round(3))


def ols_eda(ret_df, cols):
    rows = []
    for col in cols:
        dat = ret_df[[col]].join(mkt_w[MKT_COLS], how="inner").dropna()
        if len(dat) < 10:
            continue
        y = dat[col].to_numpy(float)
        X = np.column_stack([
            np.ones(len(dat)),
            dat["btc_krw_ret"].to_numpy(float),
            dat["fx_ret"].to_numpy(float),
        ])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        yhat = X @ b
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        rows.append({
            "target": col,
            "n": len(dat),
            "beta_btc_krw": b[1],
            "beta_fx": b[2],
            "r2": np.nan if ss_tot == 0 else 1 - ss_res / ss_tot,
        })
    return pd.DataFrame(rows)


beta_all = ols_eda(ret_all_w, ["GPU_ALL", "CPU_ALL", "RAM_ALL"])
beta_chip = ols_eda(ret_chip_w, CHIP_ORDER)
beta_eda = pd.concat([beta_all.assign(block="H1_H3"), beta_chip.assign(block="H2")], ignore_index=True)

print("환율 통제 포함 단순 사전 회귀계수")
display(beta_eda.round(4))


def idx_health(index_df, name):
    temp = index_df.copy()
    temp["date"] = pd.to_datetime(temp["date"])
    temp = temp[temp["date"] != temp["date"].min()]

    rows = []
    for group, sub in temp.groupby("group"):
        valid = sub[sub["ret_ok"]]
        idx_nonnull = sub["index_valid"].dropna()
        rows.append({
            "group": group,
            "valid_ret_n": int(valid.shape[0]),
            "bridge_n": int(sub["bridge"].sum()),
            "gap_n": int((~sub["ret_ok"] & ~sub["bridge"]).sum()),
            "median_valid_n_used": valid["n_used"].median(),
            "min_valid_n_used": valid["n_used"].min(),
            "max_abs_ret": valid["ret_w"].dropna().abs().max(),
            "final_index": idx_nonnull.iloc[-1] if len(idx_nonnull) else np.nan,
            "block": name,
        })
    return pd.DataFrame(rows)


health = pd.concat([
    idx_health(idx_all, "all"),
    idx_health(idx_chip, "chip"),
    idx_health(idx_vram, "vram"),
], ignore_index=True)

print("지수 산출 안정성 요약")
display(health.round(3))


def q(x, p):
    return x.quantile(p)


panel_eda = (
    panel_gpu.groupby("chip_perf_group")
    .agg(
        obs_n=("gpu_log_return", "count"),
        product_n=("row_id", "nunique"),
        zero_share=("gpu_log_return", lambda x: np.mean(np.isclose(x.dropna(), 0))),
        mean_ret=("gpu_log_return", "mean"),
        std_ret=("gpu_log_return", "std"),
        p05=("gpu_log_return", lambda x: q(x, 0.05)),
        p25=("gpu_log_return", lambda x: q(x, 0.25)),
        p50=("gpu_log_return", "median"),
        p75=("gpu_log_return", lambda x: q(x, 0.75)),
        p95=("gpu_log_return", lambda x: q(x, 0.95)),
    )
    .reset_index()
)

print("GPU 제품 단위 주간 수익률 분포: 표로만 진단")
display(panel_eda.round(4))

chip_ret_long = (
    ret_chip_w[CHIP_ORDER]
    .melt(var_name="group", value_name="ret")
    .dropna()
)

chain_eda = (
    chip_ret_long.groupby("group")
    .agg(
        week_n=("ret", "count"),
        mean_ret=("ret", "mean"),
        std_ret=("ret", "std"),
        p05=("ret", lambda x: q(x, 0.05)),
        p50=("ret", "median"),
        p95=("ret", lambda x: q(x, 0.95)),
    )
    .reset_index()
)

print("H2 검정용 chain-level 주간 수익률 분포")
display(chain_eda.round(4))

plt.figure(figsize=(13, 5))
for group, sub in idx_chip.groupby("group"):
    sub = sub.sort_values("date").copy()
    sub["date"] = pd.to_datetime(sub["date"])
    y = sub["n_used"].where(sub["ret_ok"])
    plt.plot(sub["date"], y, label=f"{CHIP_LABEL.get(group, group)}")
plt.axhline(MIN_GRP, linestyle="--", linewidth=1, label="min matched")
plt.title("H2 chip groups: valid adjacent matched products")
plt.xlabel("Date")
plt.ylabel("n_used, ret_ok weeks only")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

vals = [ret_chip_w[g].dropna().to_numpy(float) for g in CHIP_ORDER]
labels = [CHIP_LABEL.get(g, g) for g in CHIP_ORDER]

plt.figure(figsize=(8, 5))
plt.boxplot(vals, tick_labels=labels, whis=(5, 95), showfliers=False)
plt.axhline(0, linewidth=1)
plt.title("H2 chain-level weekly log return distribution")
plt.xlabel("Chip performance group")
plt.ylabel("Weekly log return, whis=5%-95%, fliers hidden")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

sample_check.to_csv(OUT / "eda_sample.csv", index=False)
corr_all.to_csv(OUT / "eda_corr_all.csv")
corr_chip.to_csv(OUT / "eda_corr_chip.csv")
beta_eda.to_csv(OUT / "eda_beta.csv", index=False)
health.to_csv(OUT / "eda_health.csv", index=False)
panel_eda.to_csv(OUT / "eda_panel_gpu.csv", index=False)
chain_eda.to_csv(OUT / "eda_chain_chip.csv", index=False)

print("extra EDA saved to:", OUT)

def season_summary(ret_df, cols, block):
    long = (
        ret_df[cols]
        .reset_index()
        .melt(id_vars="date", value_vars=cols, var_name="group", value_name="ret")
        .dropna()
    )
    long["date"] = pd.to_datetime(long["date"])
    long["month"] = long["date"].dt.month
    long["quarter"] = long["date"].dt.quarter

    month = (
        long.groupby(["group", "month"])
        .agg(
            n=("ret", "count"),
            mean_ret=("ret", "mean"),
            std_ret=("ret", "std"),
            median_ret=("ret", "median"),
        )
        .reset_index()
        .assign(block=block)
    )

    quarter = (
        long.groupby(["group", "quarter"])
        .agg(
            n=("ret", "count"),
            mean_ret=("ret", "mean"),
            std_ret=("ret", "std"),
            median_ret=("ret", "median"),
        )
        .reset_index()
        .assign(block=block)
    )

    return long, month, quarter


season_all_long, season_all_month, season_all_quarter = season_summary(
    ret_all_w, ["GPU_ALL", "CPU_ALL", "RAM_ALL"], "all"
)
season_chip_long, season_chip_month, season_chip_quarter = season_summary(
    ret_chip_w, CHIP_ORDER, "chip"
)

print("monthly return summary, all hardware")
display(season_all_month.round(4))
print("quarterly return summary, chip groups")
display(season_chip_quarter.round(4))

season_all_month.to_csv(OUT / "season_all_month.csv", index=False)
season_all_quarter.to_csv(OUT / "season_all_quarter.csv", index=False)
season_chip_month.to_csv(OUT / "season_chip_month.csv", index=False)
season_chip_quarter.to_csv(OUT / "season_chip_quarter.csv", index=False)

plt.figure(figsize=(10, 4))
season_all_long[season_all_long["group"] == "GPU_ALL"].boxplot(
    column="ret",
    by="month",
    flierprops={"marker": "o", "markersize": 2, "alpha": 0.5},
)
plt.title("Monthly distribution of GPU weekly returns")
plt.suptitle("")
plt.xlabel("Month")
plt.ylabel("Weekly log return")
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

plt.figure(figsize=(10, 4))
season_chip_long.boxplot(
    column="ret",
    by=["quarter", "group"],
    flierprops={"marker": "o", "markersize": 2, "alpha": 0.5},
)
plt.title("Quarterly distribution of chip-group weekly returns")
plt.suptitle("")
plt.xlabel("Quarter, group")
plt.ylabel("Weekly log return")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
save_plot(); plt.show()

print("[Hypo 1] GPU 전체 가격과 BTC-KRW 가격의 공변동")
print("시각화용 index: idx_all_w['GPU_ALL']")
print("검정용 hardware return: ret_all_w['GPU_ALL']  # bridge 제외, 인접 주 기준")
print("기본 BTC 변수: mkt_w['btc_krw_ret']")
print("환율 통제 변수: mkt_w['fx_ret']")
print()

print("[Hypo 2] GPU 성능군별 BTC-KRW 민감도 차이")
print("본분석 index: idx_chip_w[['high', 'mid', 'low']]  # chip 기준")
print("본분석 return: ret_chip_w[['high', 'mid', 'low']]  # bridge 제외, 인접 주 기준")
print("보조분석 index: idx_vram_w[['high', 'mid', 'low']]  # VRAM proxy")
print("제품 단위 panel: panel_gpu")
print()

print("[Hypo 3] GPU / CPU / RAM 간 BTC-KRW 영향 크기 비교")
print("시각화용 index: idx_all_w[['GPU_ALL', 'CPU_ALL', 'RAM_ALL']]")
print("검정용 return: ret_all_w[['GPU_ALL', 'CPU_ALL', 'RAM_ALL']]  # bridge 제외, 인접 주 기준")


# ── C-2) 기본 EDA: 부품 전체 주간 로그수익률 분포 요약(평균·분산·CV) ──────────
# 이미 메모리에 있는 ret_all_w(인접 주 기준 수익률)에서 GPU/CPU/RAM 전체 그룹의
# 기술통계를 한 표로 정리한다. NaN(첫 주)은 pandas 기본 skipna 로 자동 제외.
_hw_cols = ["GPU_ALL", "CPU_ALL", "RAM_ALL"]
eda_hw_desc = ret_all_w[_hw_cols].agg(
    ["count", "mean", "std", "skew", "min", "median", "max"]
).T
eda_hw_desc["cv"] = eda_hw_desc["std"] / eda_hw_desc["mean"].abs()  # 변동계수
eda_hw_desc = eda_hw_desc.round(5)
eda_hw_desc.to_csv(OUT / "eda_hw_desc.csv", encoding="utf-8-sig")
print()
print("eda_hw_desc.csv ->")
print(eda_hw_desc.to_string())
