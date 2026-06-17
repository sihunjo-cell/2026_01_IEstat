# Auto-generated from Hypo3_relation.ipynb (code cells only, markdown/outputs stripped)

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

out = Path("outputs/hypo3_relation")
pic = Path("output_pic/hypo3_relation")
out.mkdir(parents=True, exist_ok=True)
pic.mkdir(parents=True, exist_ok=True)

ret_path = Path("outputs/ret_all_w.csv")
if not ret_path.exists():
    ret_path = Path("ret_all_w.csv")

week = "W-SUN"
hws = ["GPU", "CPU", "RAM"]

plt.rcParams["axes.unicode_minus"] = False


def z(s):
    return (s - s.mean()) / s.std()


def fig(name):
    p = pic / name
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print("saved:", p)

ret = pd.read_csv(ret_path, parse_dates=["date"]).set_index("date").sort_index()
ret = ret[["GPU_ALL", "CPU_ALL", "RAM_ALL"]].rename(columns={
    "GPU_ALL": "gpu_ret",
    "CPU_ALL": "cpu_ret",
    "RAM_ALL": "ram_ret",
})

print("loaded:", ret_path)
display(ret.head())
print(ret.shape)

st = ret.index.min() - pd.Timedelta(days=14)
ed = ret.index.max() + pd.Timedelta(days=7)

tick = {
    "BTC-KRW": "btc",
    "KRW=X": "fx",
    "SOXX": "soxx",
    "BZ=F": "oil",
}

px = yf.download(list(tick.keys()), start=st, end=ed, auto_adjust=True, progress=False)["Close"]
px = px.rename(columns=tick)
wk = px.resample(week).last()

mkt = np.log(wk).diff().rename(columns={
    "btc": "btc_ret",
    "fx": "fx_ret",
    "soxx": "soxx_ret",
    "oil": "oil_ret",
})

# CSI = Korea consumer confidence (OECD Composite Consumer Confidence for Korea, FRED, monthly).
# Was UMCSENT (US consumer sentiment), which is the wrong country for a Korean demand proxy -> replaced.
# Source is monthly, so weekly ffill+diff only captures one real change per month
# (most weekly csi_chg values are 0; interpret csi accordingly).
csi = web.DataReader("CSCICP02KRM066S", "fred", st, ed).rename(columns={"CSCICP02KRM066S": "csi"})
mkt = mkt.join(csi.resample(week).ffill(), how="left")
mkt["csi_chg"] = mkt["csi"].diff()

display(mkt.head())

df = ret.join(mkt, how="inner").sort_index()
raw = ["btc_ret", "fx_ret", "soxx_ret", "oil_ret", "csi_chg"]
for c in raw:
    df[c + "_z"] = z(df[c])

df.to_csv(out / "data.csv")

na = round(df[raw].isna().sum() / len(df) * 100, 2)
print("missing rate (%)")
display(na)
print("saved:", out / "data.csv")

# CSI(월간 지수)는 주간 회귀에 넣으면 csi_chg_z 가 한 달에 한 주만 ≠0 인 희소
# 월간 더미가 되어 주간 요인으로 해석 불가 → 주간 회귀(spec)에서는 제외하고,
# CSI 는 본래 월간 주기에서 별도 검정한다(아래 csi_monthly_robust.csv).
spec = {
    "BTC":  ["btc_ret_z",  ["fx_ret_z", "soxx_ret_z"]],
    "SOXX": ["soxx_ret_z", ["fx_ret_z", "btc_ret_z"]],
    "Oil":  ["oil_ret_z",  ["fx_ret_z", "btc_ret_z", "soxx_ret_z"]],
}

pd.DataFrame([
    {"factor": k, "x": v[0], "ctrl": ", ".join(v[1])}
    for k, v in spec.items()
])

def cut(d, cols):
    st0 = max(d[c].first_valid_index() for c in cols)
    body = d.loc[st0:].copy()
    mc = body[body[cols].isna().any(axis=1)]
    use = body.dropna(subset=cols)
    return use, st0, len(d.loc[:st0].iloc[:-1]), len(mc)


def vmax(d, xs):
    X = sm.add_constant(d[xs])
    v = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    return max(v)


def fit(y, fac):
    x, ctrl = spec[fac]
    xs = [x] + ctrl
    cols = [y] + xs
    d, st0, n0, n1 = cut(df[cols], cols)

    form = y + " ~ " + " + ".join(xs)
    m = smf.ols(form, data=d).fit(cov_type="HC3")
    r = smf.ols(y + " ~ " + " + ".join(ctrl), data=d).fit(cov_type="HC3")
    pr2 = (m.rsquared - r.rsquared) / (1 - r.rsquared)

    return {
        "hw": y[:3].upper(),
        "factor": fac,
        "x": x,
        "ctrl": ", ".join(ctrl),
        "n": int(m.nobs),
        "coef": m.params[x],
        "se": m.bse[x],
        "t": m.tvalues[x],
        "p": m.pvalues[x],
        "r2": m.rsquared,
        "pr2": pr2,
        "max_vif": vmax(d, xs),
        "start": st0,
        "struct_na": n0,
        "mcar_na": n1,
        "model": form,
    }

ys = ["gpu_ret", "cpu_ret", "ram_ret"]
rows = [fit(y, f) for y in ys for f in spec]
res = pd.DataFrame(rows)
miss = res[["hw", "factor", "start", "struct_na", "mcar_na", "n"]].copy()

res.to_csv(out / "res.csv", index=False)
miss.to_csv(out / "miss.csv", index=False)

display(res.round(5))
print("saved:", out / "res.csv", out / "miss.csv")

# --- CSI 월간 주기 robustness (CSI 본래 주기에서 재검정) ----------------------
# CSI(CSCICP02KRM066S)는 월간 지수라 주간 지수에 ffill+diff 하면 한 달에 한 주만
# csi_chg≠0(나머지 0)인 인공적 시계열이 된다. 주파수 불일치가 결론을 만들거나
# 숨기지 않았는지, CSI 본래 월간 주기에서 동일 검정을 재수행한다.
# (표본이 ~24개월로 작아 검정력은 낮음 — 한계로 함께 기재할 것.)
ret_m = ret[["gpu_ret", "cpu_ret", "ram_ret"]].resample("MS").sum()  # 월간 로그수익(주간 합)
wk_n = ret["gpu_ret"].resample("MS").count()
ret_m = ret_m[wk_n >= 3]                                             # 주 부족한 양 끝 달 제외
csi_m = csi["csi"].resample("MS").last().to_frame()                 # 이미 받아둔 월간 CSI 재사용
csi_m["csi_chg"] = csi_m["csi"].diff()
dm = ret_m.join(csi_m["csi_chg"], how="inner").dropna()
dm["csi_chg_z"] = z(dm["csi_chg"])
rows_m = []
for y in ["gpu_ret", "cpu_ret", "ram_ret"]:
    m = smf.ols(f"{y} ~ csi_chg_z", data=dm).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    rows_m.append({"hw": y[:3].upper(), "n_months": int(m.nobs),
                   "coef": m.params["csi_chg_z"], "p": m.pvalues["csi_chg_z"], "r2": m.rsquared})
csi_month = pd.DataFrame(rows_m)
csi_month.to_csv(out / "csi_monthly_robust.csv", index=False)
display(csi_month.round(4))
print("saved:", out / "csi_monthly_robust.csv")

# 1) correlation matrix
cc = ["gpu_ret", "cpu_ret", "ram_ret"] + [spec[k][0] for k in spec]
C = df[cc].corr()

plt.figure(figsize=(8, 6))
plt.imshow(C, aspect="auto")
plt.colorbar(label="corr")
plt.xticks(range(len(C.columns)), C.columns, rotation=45, ha="right")
plt.yticks(range(len(C.index)), C.index)
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        plt.text(j, i, f"{C.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Current correlation")
plt.tight_layout()
fig("01_corr.png")
plt.show()

# 2) controlled coefficient
pv = res.pivot(index="factor", columns="hw", values="coef").loc[list(spec.keys())]
ax = pv.plot(kind="bar", figsize=(9, 5))
plt.axhline(0, linewidth=1)
plt.title("Controlled coefficient")
plt.ylabel("coefficient on z-scored factor")
plt.xlabel("factor")
plt.tight_layout()
fig("02_coef.png")
plt.show()

# 3) partial R2
pv = res.pivot(index="factor", columns="hw", values="pr2").loc[list(spec.keys())]
ax = pv.plot(kind="bar", figsize=(9, 5))
plt.axhline(0, linewidth=1)
plt.title("Partial R-squared")
plt.ylabel("partial R²")
plt.xlabel("factor")
plt.tight_layout()
fig("03_pr2.png")
plt.show()

# 4) scatter matrix; title reports controlled regression p-value
fig0, ax = plt.subplots(3, 4, figsize=(18, 11))
ys = ["gpu_ret", "cpu_ret", "ram_ret"]
ylab = ["GPU", "CPU", "RAM"]
facs = list(spec.keys())

for i, y in enumerate(ys):
    for j, fac in enumerate(facs):
        a = ax[i, j]
        x = spec[fac][0]
        tmp = df[[y, x]].dropna()
        a.scatter(tmp[x], tmp[y], s=28, alpha=0.65)
        if len(tmp) >= 3:
            b1, b0 = np.polyfit(tmp[x], tmp[y], 1)
            xx = np.linspace(tmp[x].min(), tmp[x].max(), 50)
            a.plot(xx, b1 * xx + b0, linewidth=1.5)
        rr = res[(res["hw"] == ylab[i]) & (res["factor"] == fac)].iloc[0]
        a.set_title(f"{ylab[i]} vs {fac}\ncoef={rr.coef:.4f}, p={rr.p:.3f}", fontsize=10)
        a.set_xlabel(x)
        a.set_ylabel(y)
        a.axhline(0, linewidth=0.8)
        a.axvline(0, linewidth=0.8)

plt.suptitle("Current macro factor vs hardware return", y=1.02, fontsize=15)
plt.tight_layout()
fig("04_scatter.png")
plt.show()

print("saved csv:", sorted(p.name for p in out.glob("*.csv")))
print("saved png:", sorted(p.name for p in pic.glob("*.png")))
print("current-period regressions only")
print("SOXX is used as broad semiconductor/AI proxy; single-stock proxy is not used")
print("no quarter dummy or hardware autoregressive regressor is used")
