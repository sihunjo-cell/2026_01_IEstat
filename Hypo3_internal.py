# Auto-generated from Hypo3_internal.ipynb (code cells only, markdown/outputs stripped)


from pathlib import Path
import re, zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

START = "2024-03-01"
END = "2026-03-31"
FREQ = "W-SUN"
MIN_D = 3
RET_Q = 0.05
RET_MIN = 20
RET_MAX = np.log(1.8)

ROOT = Path(".")
OUT = ROOT / "outputs" / "hypo3_internal"
PIC = ROOT / "output_pic" / "hypo3_internal"
OUT.mkdir(parents=True, exist_ok=True)
PIC.mkdir(parents=True, exist_ok=True)

GPU_Z = ROOT / "GPU_PREPARE.zip"

Z = {
    "GPU": GPU_Z,
    "CPU": ROOT / "CPU_PREPARE.zip",
    "RAM": ROOT / "RAM_PREPARE.zip",
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def dates(df):
    cols = sorted(c for c in df.columns if DATE_RE.fullmatch(str(c)))
    return [c for c in cols if START <= c <= END]


def px(df, cols):
    x = df[cols].replace(",", "", regex=True).apply(pd.to_numeric, errors="coerce")
    return x.where(x > 0).astype(float)


def cap(x):
    s = str(x).upper().replace(" ", "")
    m = re.search(r"(\d+(?:\.\d+)?)GB", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)MB", s)
    if m:
        return float(m.group(1)) / 1024
    m = re.search(r"(\d+(?:\.\d+)?)G$", s)
    if m:
        return float(m.group(1))
    return np.nan


def chip(x):
    s = re.sub(r"\s+", " ", str(x).upper()).strip()
    s = s.replace("GEFORCE", "").replace("지포스", "")
    s = s.replace("RADEON", "").replace("라데온", "")
    s = re.sub(r"\s+", " ", s).strip()

    pats = [
        (r"\bRTX\s*A\s?(\d{4})\b", lambda m: f"RTX A{m.group(1)}"),
        (r"\bPRO\s*W\s?(\d{4})\b", lambda m: f"PRO W{m.group(1)}"),
        (r"\bRTX\s*(\d{4})\s*(TI)?\s*(SUPER)?", lambda m: f"RTX {m.group(1)} {' '.join(v for v in [m.group(2), m.group(3)] if v)}".strip()),
        (r"\bGTX\s*(\d{3,4})\s*(TI)?\s*(SUPER)?", lambda m: f"GTX {m.group(1)} {' '.join(v for v in [m.group(2), m.group(3)] if v)}".strip()),
        (r"\bGT\s*(\d{3,4})\b", lambda m: f"GT {m.group(1)}"),
        (r"\bRX\s*(\d{3,4})\s*(XTX|XT|GRE)?", lambda m: f"RX {m.group(1)} {m.group(2) or ''}".strip()),
        (r"\bARC\s*A\s?(\d{3,4})\b", lambda m: f"ARC A{m.group(1)}"),
    ]
    for p, f in pats:
        m = re.search(p, s)
        if m:
            return f(m)
    return np.nan


def tier(x):
    s = str(x).upper().strip()
    if s in {"", "NAN", "NONE"}:
        return "unknown"
    if any(v in s for v in ["RTX A", "QUADRO", "TESLA", "PRO W"]):
        return "workstation"

    m = re.search(r"\bRTX\s*(\d{4})\b", s)
    if m:
        t = int(m.group(1)) % 100
        if t >= 80:
            return "high"
        if t == 70:
            return "high" if ("TI" in s or "SUPER" in s) else "mid"
        return "mid" if t == 60 else "low"

    m = re.search(r"\bGTX\s*(\d{3,4})\b", s)
    if m:
        return "mid" if int(m.group(1)) >= 1660 else "low"
    if re.search(r"\bGT\s*(\d{3,4})\b", s):
        return "low"

    m = re.search(r"\bRX\s*(\d{3,4})\b", s)
    if m:
        n = int(m.group(1))
        t = n % 100 if n >= 9000 else ((n // 100) % 10) * 10
        if t >= 80:
            return "high"
        if t == 70:
            return "high" if any(v in s for v in ["XTX", "XT", "GRE"]) else "mid"
        return "mid" if t == 60 else "low"

    m = re.search(r"\bARC\s*A\s?(\d{3,4})\b", s)
    if m:
        return "mid" if int(m.group(1)) in {770, 750, 580} else "low"
    return "unknown"


def prep_gpu(df):
    d = df.copy()
    txt = d[[c for c in ["Name", "gpu_model", "name_rest"] if c in d.columns]].fillna("").astype(str).agg(" ".join, axis=1)
    d = d[~txt.str.contains(r"노트북|모바일|LAPTOP|MOBILE|MAX-Q|NOTEBOOK", case=False, regex=True, na=False)].copy()
    d["vram_gb"] = d["capacity"].apply(cap)
    d["chip"] = d["gpu_model"].apply(chip)
    d["tier"] = d["chip"].apply(tier)
    return d[d["tier"].isin(["low", "mid", "high"]) & d["vram_gb"].notna()].copy()


def prep_cpu(df):
    d = df.copy()
    d = d[~d["Name"].astype(str).str.contains("중고|XEON|EPYC|제온|옵테론", case=False, regex=True, na=False)].copy()
    d = d[~d["sale_info"].astype(str).str.contains("중고", na=False)].copy()
    d["maker"] = np.select(
        [
            d["cpu_model"].astype(str).str.contains("AMD", case=False, na=False),
            d["cpu_model"].astype(str).str.contains("인텔|INTEL", case=False, regex=True, na=False),
        ],
        ["AMD", "Intel"],
        default="Other",
    )
    return d


def prep_ram(df):
    d = df.copy()
    d["capacity_gb"] = pd.to_numeric(d["capacity_gb"], errors="coerce")
    d["memory_speed"] = pd.to_numeric(d["memory_speed"], errors="coerce")
    d = d[d["memory_gen"].astype(str).str.upper().isin(["DDR4", "DDR5"])].copy()
    return d[d["capacity_gb"].notna() & (d["capacity_gb"] > 0)].copy()


PREP = {"GPU": prep_gpu, "CPU": prep_cpu, "RAM": prep_ram}
KEY = {"GPU": ["Id"], "CPU": ["Id"], "RAM": ["Id", "capacity"]}


def load(zp, tag):
    frames = []
    with zipfile.ZipFile(zp) as z:
        names = sorted(n for n in z.namelist() if n.endswith(f"_{tag}.csv"))
        for n in names:
            d = pd.read_csv(z.open(n))
            d.columns = [str(c).strip().replace("\ufeff", "") for c in d.columns]
            d = d.loc[:, ~d.columns.astype(str).str.match(r"^Unnamed")]
            cols = dates(d)
            d = PREP[tag](d)
            meta = [c for c in d.columns if c not in cols]

            x = pd.concat([d[KEY[tag]].reset_index(drop=True), px(d, cols).reset_index(drop=True)], axis=1)
            x = x.groupby(KEY[tag], dropna=False)[cols].median().reset_index()
            m = d[meta].groupby(KEY[tag], dropna=False).first().reset_index()
            frames.append(m.merge(x, on=KEY[tag], how="right"))

    d = pd.concat(frames, ignore_index=True, sort=False)
    cols = sorted(c for c in d.columns if DATE_RE.fullmatch(str(c)))
    meta = [c for c in d.columns if c not in cols]
    extra = [c for c in meta if c not in KEY[tag]]

    x = d[KEY[tag] + cols].groupby(KEY[tag], dropna=False)[cols].median().reset_index()
    m = d[KEY[tag] + extra].groupby(KEY[tag], dropna=False).first().reset_index()
    return m.merge(x, on=KEY[tag], how="right").reset_index(drop=True), cols


def wmat(df, cols):
    x = df[cols].T
    x.index = pd.to_datetime(cols)
    n = x.resample(FREQ).count()
    w = x.resample(FREQ).median().where(n >= MIN_D).T
    w.columns = [c.strftime("%Y-%m-%d") for c in w.columns]
    return w


def ret_avg(r):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return np.nan, 0
    r = r[r.abs() <= RET_MAX]
    if RET_Q > 0 and len(r) >= RET_MIN:
        lo, hi = r.quantile([RET_Q, 1 - RET_Q])
        r = r[(r >= lo) & (r <= hi)]
    return (float(r.mean()) if len(r) else np.nan), len(r)


def stat(w):
    a = w.notna()
    lw = np.log(w.where(w > 0))
    rr = lw.diff(axis=1)
    rows = []
    cols = list(w.columns)

    for i, c in enumerate(cols):
        now = a[c]
        prev = a[cols[i - 1]] if i else pd.Series(False, index=a.index)
        act = int(now.sum())
        prv = int(prev.sum())
        inn = int((now & ~prev).sum()) if i else act
        out = int((~now & prev).sum()) if i else 0
        mat = int((now & prev).sum()) if i else 0

        p = w.loc[now, c]
        lp = np.log(p)
        r = rr[c].dropna()
        ret, n = ret_avg(r)

        rows.append({
            "date": pd.to_datetime(c),
            "ret": ret,
            "ret_n": n,
            "act_n": act,
            "in_n": inn,
            "out_n": out,
            "match_n": mat,
            "in_sh": inn / act if act else np.nan,
            "out_sh": out / prv if prv else np.nan,
            "match_sh": mat / act if act else np.nan,
            "med_px": p.median(),
            "logp_iqr": lp.quantile(.75) - lp.quantile(.25) if len(lp) else np.nan,
            "px_cv": p.std() / p.mean() if len(p) and p.mean() else np.nan,
            "ret_iqr": r.quantile(.75) - r.quantile(.25) if len(r) else np.nan,
            "ret_abs": r.abs().median() if len(r) else np.nan,
        })
    return pd.DataFrame(rows)


def share(w, meta, funcs):
    out = pd.DataFrame({"date": pd.to_datetime(w.columns)})
    for name, fn in funcs.items():
        out[name] = [fn(meta.loc[w[c].notna()]) if w[c].notna().any() else np.nan for c in w.columns]
    return out


def pref(df, p):
    return df.rename(columns={c: f"{p}_{c}" for c in df.columns if c != "date"})


def part(tag, funcs):
    d, cols = load(Z[tag], tag)
    w = wmat(d, cols)
    out = stat(w).merge(share(w, d, funcs), on="date")
    info = {
        "part": tag.lower(),
        "prod_n": len(d),
        "week_n": w.shape[1],
        "med_act": out["act_n"].median(),
        "med_match_sh": out["match_sh"].median(),
    }
    return pref(out, tag.lower()), info


gpu, sg = part("GPU", {
    "high_sh": lambda x: (x["tier"] == "high").mean(),
    "mid_sh": lambda x: (x["tier"] == "mid").mean(),
    "low_sh": lambda x: (x["tier"] == "low").mean(),
    "vram_avg": lambda x: x["vram_gb"].mean(),
    "vram_hi_sh": lambda x: (x["vram_gb"] >= 12).mean(),
})

cpu, sc = part("CPU", {
    "amd_sh": lambda x: (x["maker"] == "AMD").mean(),
    "intel_sh": lambda x: (x["maker"] == "Intel").mean(),
    "box_sh": lambda x: x["sale_info"].astype(str).str.contains("정품", na=False).mean(),
    "bulk_sh": lambda x: x["sale_info"].astype(str).str.contains("벌크", na=False).mean(),
    "imp_sh": lambda x: x["sale_info"].astype(str).str.contains("해외구매", na=False).mean(),
})

ram, sr = part("RAM", {
    "ddr5_sh": lambda x: (x["memory_gen"].astype(str).str.upper() == "DDR5").mean(),
    "cap_avg": lambda x: x["capacity_gb"].mean(),
    "spd_avg": lambda x: x["memory_speed"].replace(0, np.nan).mean(),
    "cap_hi_sh": lambda x: (x["capacity_gb"] >= 32).mean(),
})

int_w = gpu.merge(cpu, on="date", how="outer").merge(ram, on="date", how="outer").sort_values("date")
int_sum = pd.DataFrame([sg, sc, sr])

int_w.to_csv(OUT / "int_w.csv", index=False)
int_sum.to_csv(OUT / "int_sum.csv", index=False)

print("saved:", OUT / "int_w.csv")
print("saved:", OUT / "int_sum.csv")
display(int_w.head())
display(int_sum)


def line(cols, name, y):
    plt.figure(figsize=(11, 4))
    for c in cols:
        plt.plot(int_w["date"], int_w[c], label=c.replace("_", " "))
    plt.title(name)
    plt.ylabel(y)
    plt.xlabel("date")
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PIC / f"{name}.png", dpi=180)
    plt.close()


line(["gpu_act_n", "cpu_act_n", "ram_act_n"], "01_act", "active products")
line(["gpu_in_sh", "cpu_in_sh", "ram_in_sh"], "02_in", "entry share")
line(["gpu_ret_iqr", "cpu_ret_iqr", "ram_ret_iqr"], "03_ret_iqr", "product return IQR")
line(["gpu_high_sh", "cpu_amd_sh", "ram_ddr5_sh"], "04_mix", "share")

print("plot saved to:", PIC)
