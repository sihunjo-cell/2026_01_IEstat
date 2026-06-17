# Auto-generated from Hypo3_time_lag.ipynb (code cells only, markdown/outputs stripped)

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt

out = Path("outputs/hypo3_time_lag")
pic = Path("output_pic/hypo3_time_lag")
out.mkdir(parents=True, exist_ok=True)
pic.mkdir(parents=True, exist_ok=True)

p = Path("outputs/hypo3_relation/data.csv")
if not p.exists():
    p = Path("data.csv")

lags = range(1, 9)

plt.rcParams["axes.unicode_minus"] = False


def save(name):
    fp = pic / name
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    print("saved:", fp)


def vif(d, xs):
    X = sm.add_constant(d[xs])
    v = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    return max(v)


def bh(pv):
    p = np.asarray(pv, dtype=float)
    n = len(p)
    q = np.empty(n)
    order = np.argsort(p)
    best = 1.0
    for i in range(n - 1, -1, -1):
        j = order[i]
        best = min(best, p[j] * n / (i + 1))
        q[j] = min(best, 1.0)
    return q

df = pd.read_csv(p)
df = df.rename(columns={df.columns[0]: "date"})
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

need = [
    "gpu_ret", "ram_ret",
    "btc_ret_z", "soxx_ret_z", "fx_ret_z",
]
df = df[need].copy()

display(df.head())
print("loaded:", p)
print("shape:", df.shape)

spec = {
    "BTC-GPU": {
        "y": "gpu_ret",
        "x": "btc_ret_z",
        "ctrl": ["fx_ret_z", "soxx_ret_z"],
    },
    "SOXX-RAM": {
        "y": "ram_ret",
        "x": "soxx_ret_z",
        "ctrl": ["fx_ret_z", "btc_ret_z"],
    },
}

pd.DataFrame([
    {"pair": k, "y": v["y"], "x": v["x"], "ctrl": ", ".join(v["ctrl"])}
    for k, v in spec.items()
])

def part(d, y, x, cs):
    yr = sm.OLS(d[y], sm.add_constant(d[cs])).fit().resid
    xr = sm.OLS(d[x], sm.add_constant(d[cs])).fit().resid
    return xr, yr


def run(pair, k):
    s = spec[pair]
    y, x, cs = s["y"], s["x"], s["ctrl"]

    d = pd.DataFrame({y: df[y]})
    xs = []
    for v in [x] + cs:
        nm = v.replace("_ret_z", "").replace("_z", "") + f"_l{k}"
        d[nm] = df[v].shift(k)
        xs.append(nm)

    xk = xs[0]
    ck = xs[1:]
    d = d.dropna()

    X = sm.add_constant(d[xs])
    m = sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    r = sm.OLS(d[y], sm.add_constant(d[ck])).fit()
    pr2 = (m.rsquared - r.rsquared) / (1 - r.rsquared)

    xr, yr = part(d, y, xk, ck)
    pred = np.sign(m.params[xk] * xr)
    real = np.sign(yr)
    ok = (pred != 0) & (real != 0)
    acc = float((pred[ok] == real[ok]).mean())

    lb = acorr_ljungbox(m.resid, lags=[4], return_df=True)["lb_pvalue"].iloc[0]
    bp = het_breuschpagan(m.resid, m.model.exog)[1]

    return {
        "pair": pair,
        "lag": k,
        "n": int(m.nobs),
        "coef": m.params[xk],
        "se": m.bse[xk],
        "t": m.tvalues[xk],
        "p": m.pvalues[xk],
        "r2": m.rsquared,
        "pr2": pr2,
        "max_vif": vif(d, xs),
        "dw": durbin_watson(m.resid),
        "lb_p": lb,
        "bp_p": bp,
        "dir_acc": acc,
        "dir": "up" if m.params[xk] > 0 else "down",
        "model": y + " ~ " + " + ".join(xs),
    }

res = pd.DataFrame([run(pair, k) for pair in spec for k in lags])
res["q"] = bh(res["p"])
res["keep"] = (res["q"] < 0.10) & (res["pr2"] >= 0.01) & (res["max_vif"] < 5)

res = res[[
    "pair", "lag", "n", "coef", "se", "t", "p", "q", "r2", "pr2",
    "max_vif", "dw", "lb_p", "bp_p", "dir_acc", "dir", "keep", "model"
]]

top = res[res["keep"]].copy()
if len(top) == 0:
    top = res.sort_values(["pair", "p", "pr2"], ascending=[True, True, False]).groupby("pair").head(2).copy()
else:
    top = top.sort_values(["pair", "q", "p", "pr2"], ascending=[True, True, True, False])

res.to_csv(out / "lag_res.csv", index=False)
top.to_csv(out / "lag_top.csv", index=False)

display(res.round(5))
print("saved:", out / "lag_res.csv", out / "lag_top.csv")

# 1) coefficient path with HAC CI
fig0, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
for i, pair in enumerate(spec):
    d = res[res["pair"] == pair].copy()
    lo = d["coef"] - 1.96 * d["se"]
    hi = d["coef"] + 1.96 * d["se"]
    ax[i].plot(d["lag"], d["coef"], marker="o")
    ax[i].fill_between(d["lag"], lo, hi, alpha=0.2)
    ax[i].axhline(0, linewidth=1)
    ax[i].set_title(pair)
    ax[i].set_xlabel("lag week")
    ax[i].set_ylabel("coef with 95% CI")
    ax[i].set_xticks(list(lags))
plt.tight_layout()
save("01_coef.png")
plt.show()

# 2) partial R2 path
fig0, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
for i, pair in enumerate(spec):
    d = res[res["pair"] == pair].copy()
    ax[i].bar(d["lag"], d["pr2"])
    ax[i].set_title(pair)
    ax[i].set_xlabel("lag week")
    ax[i].set_ylabel("partial R²")
    ax[i].set_xticks(list(lags))
plt.tight_layout()
save("02_pr2.png")
plt.show()

# 3) residualized scatter for selected lags
best = top.sort_values(["pair", "p", "pr2"], ascending=[True, True, False]).groupby("pair").head(1)
fig0, ax = plt.subplots(1, len(best), figsize=(6 * len(best), 4))
if len(best) == 1:
    ax = [ax]

for a, (_, row) in zip(ax, best.iterrows()):
    pair = row["pair"]
    k = int(row["lag"])
    s = spec[pair]
    y, x, cs = s["y"], s["x"], s["ctrl"]

    d = pd.DataFrame({y: df[y]})
    xs = []
    for v in [x] + cs:
        nm = v.replace("_ret_z", "").replace("_z", "") + f"_l{k}"
        d[nm] = df[v].shift(k)
        xs.append(nm)
    d = d.dropna()

    xr, yr = part(d, y, xs[0], xs[1:])
    a.scatter(xr, yr, s=32, alpha=0.65)
    b1, b0 = np.polyfit(xr, yr, 1)
    xx = np.linspace(xr.min(), xr.max(), 50)
    a.plot(xx, b1 * xx + b0, linewidth=1.5)
    a.axhline(0, linewidth=0.8)
    a.axvline(0, linewidth=0.8)
    a.set_title(f"{pair}, lag {k}\ncoef={row.coef:.4f}, p={row.p:.3f}, q={row.q:.3f}")
    a.set_xlabel("factor residual")
    a.set_ylabel("hardware residual")

plt.tight_layout()
save("03_top.png")
plt.show()

print("csv:", sorted(p.name for p in out.glob("*.csv")))
print("png:", sorted(p.name for p in pic.glob("*.png")))
print("pairs:", list(spec.keys()))
print("lags:", list(lags))
print("SOXX is used as broad semiconductor/AI proxy; single-stock proxy is not used")
print("no quarter dummy, no hardware autoregressive regressor, no pooled model")
