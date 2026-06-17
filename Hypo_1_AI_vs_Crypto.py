# Auto-generated from Hypo_1_AI_vs_Crypto.ipynb (code cells only, markdown/outputs stripped)

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display

OUT = Path("outputs")
RES = OUT / "h1_final_v7"
PIC = Path("output_pic") / "h1_final_v7"
RES.mkdir(parents=True, exist_ok=True)
PIC.mkdir(parents=True, exist_ok=True)

RET_PATH = OUT / "ret_all_w.csv"
HW = ["GPU_ALL", "CPU_ALL", "RAM_ALL"]
NAME = {"GPU_ALL": "GPU", "CPU_ALL": "CPU", "RAM_ALL": "RAM"}
ORDER = ["GPU", "CPU", "RAM"]

FREQ = "W-SUN"
HAC_LAG = 8
Q = 0.90
B = 5000
SEED = 42
MIN_TRAIN = 52
EPS = 1e-8

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def std(s):
    return (s - s.mean()) / s.std(ddof=0)


def nm(s):
    s = re.sub(r'[\\/:*?"<>|]+', "_", str(s))
    s = re.sub(r"\s+", "_", s).strip("._ ")
    return s[:70]


def save(name):
    p = PIC / f"{nm(name)}.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    print("saved:", p)


def fit(formula, df, hac=True):
    m = smf.ols(formula, data=df)
    if hac:
        return m.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAG})
    return m.fit()


def wald(mod, terms):
    h = ", ".join(f"{t}=0" for t in terms)
    out = mod.f_test(h)
    return float(np.asarray(out.fvalue).ravel()[0]), float(np.asarray(out.pvalue).ravel()[0])


def pr2(full, red):
    return (full.rsquared - red.rsquared) / (1 - red.rsquared)


def lagwin(s, lags):
    x = pd.concat([s.shift(k) for k in lags], axis=1)
    return x.mean(axis=1).where(x.notna().all(axis=1))


def win(a, b, eps=EPS):
    if a > b + eps:
        return "AI"
    if b > a + eps:
        return "Crypto"
    return "Mixed"


def gain_win(ai_gain, cr_gain, eps=EPS):
    if ai_gain > cr_gain + eps and ai_gain > eps:
        return "AI"
    if cr_gain > ai_gain + eps and cr_gain > eps:
        return "Crypto"
    return "Mixed"


def loss_p(diff):
    x = np.ones(len(diff))
    m = sm.OLS(diff, x).fit(cov_type="HAC", cov_kwds={"maxlags": min(HAC_LAG, max(1, len(diff)//4))})
    return float(m.params[0]), float(m.pvalues[0])

ret_w = pd.read_csv(RET_PATH, parse_dates=["date"]).set_index("date").sort_index()

hw = ret_w[HW].rename(columns=NAME)
ret = (
    hw.reset_index()
      .melt(id_vars="date", var_name="cat", value_name="ret")
      .dropna()
      .sort_values(["cat", "date"])
)

sample = (
    ret.groupby("cat")
       .agg(n=("ret", "size"), mean=("ret", "mean"), sd=("ret", "std"), min=("ret", "min"), max=("ret", "max"))
       .reset_index()
)

display(sample.round(5))
ret.to_csv(RES / "ret_long.csv", index=False)
sample.to_csv(RES / "sample.csv", index=False)

def prices(tickers, start, end):
    out = {}
    for t, n in tickers.items():
        s = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        out[n] = s
    return pd.DataFrame(out)

start = (ret["date"].min() - pd.Timedelta(days=21)).strftime("%Y-%m-%d")
end = (ret["date"].max() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

TICK = {
    "BTC-KRW": "btc",
    "ETC-KRW": "etc",
    "KRW=X": "fx",
    "NVDA": "nvda",
    "^IXIC": "nasdaq",
    "^SOX": "sox",
}

px = prices(TICK, start, end)
log_w = np.log(px).resample(FREQ).last()
mkt = log_w.diff().rename(columns={
    "btc": "btc_ret",
    "etc": "etc_ret",
    "fx": "fx_ret",
    "nvda": "nvda_ret",
    "nasdaq": "nasdaq_ret",
    "sox": "sox_ret",
})

mkt["ai"] = std(mkt["nvda_ret"])
mkt["crypto"] = std(mkt["btc_ret"])
mkt["crypto_fxadj"] = std(mkt["btc_ret"] - mkt["fx_ret"])

r0 = mkt[["nvda_ret", "nasdaq_ret", "sox_ret"]].dropna()
rfit = sm.OLS(r0["nvda_ret"], sm.add_constant(r0[["nasdaq_ret", "sox_ret"]])).fit()
mkt["ai_resid"] = np.nan
mkt.loc[r0.index, "ai_resid"] = rfit.resid
mkt["ai_resid_z"] = std(mkt["ai_resid"])

pc0 = mkt[["btc_ret", "etc_ret"]].dropna().apply(std)
U, S, VT = np.linalg.svd(pc0, full_matrices=False)
pc1 = pd.Series(U[:, 0] * S[0], index=pc0.index)
if pc1.corr(mkt.loc[pc0.index, "btc_ret"]) < 0:
    pc1 = -pc1
mkt["crypto_pc1"] = np.nan
mkt.loc[pc0.index, "crypto_pc1"] = std(pc1)

cols = ["btc_ret", "etc_ret", "fx_ret", "nvda_ret", "nasdaq_ret", "sox_ret", "ai", "ai_resid_z", "crypto", "crypto_fxadj", "crypto_pc1"]
display(mkt[cols].dropna(how="all").head())
mkt[cols].to_csv(RES / "market.csv")

def make_feat(ai_col="ai", cr_col="crypto"):
    f = mkt[[ai_col, cr_col, "fx_ret"]].rename(columns={ai_col: "ai", cr_col: "crypto"}).copy()
    for v in ["ai", "crypto"]:
        f[f"{v}_0"] = f[v]
        f[f"{v}_12"] = lagwin(f[v], [1, 2])
        f[f"{v}_38"] = lagwin(f[v], list(range(3, 9)))
    for c in ["ai_0", "ai_12", "ai_38", "crypto_0", "crypto_12", "crypto_38"]:
        f[c] = std(f[c])
    return f

feat = make_feat()

AI = ["ai_0", "ai_12", "ai_38"]
CR = ["crypto_0", "crypto_12", "crypto_38"]
QTR = ["q2", "q3", "q4"]

BASE = ["ret_l1", "fx_ret"] + QTR
TERMS = AI + CR
FULL = BASE + TERMS

model = ret.merge(feat[["fx_ret"] + TERMS], left_on="date", right_index=True, how="inner")
model = model.sort_values(["cat", "date"]).copy()

model["ret_l1"] = model.groupby("cat")["ret"].shift(1)

# Quarter fixed effects: q1 is the baseline
model["q"] = model["date"].dt.quarter
for k in [2, 3, 4]:
    model[f"q{k}"] = (model["q"] == k).astype(int)

n = model.dropna(subset=["ret"] + FULL).groupby("cat").size().reset_index(name="n")
display(n)

model.to_csv(RES / "model_data.csv", index=False)

def vif(df, terms):
    x = sm.add_constant(df[terms].dropna())
    return pd.DataFrame({"term": terms, "vif": [variance_inflation_factor(x.values, i + 1) for i in range(len(terms))]})


def run(cat, data=model):
    d = data[data["cat"] == cat].dropna(subset=["ret"] + FULL).copy()
    fb = "ret ~ " + " + ".join(BASE)
    fa = "ret ~ " + " + ".join(BASE + AI)
    fc = "ret ~ " + " + ".join(BASE + CR)
    ff = "ret ~ " + " + ".join(FULL)

    mb, ma, mc, mf = fit(fb, d), fit(fa, d), fit(fc, d), fit(ff, d)
    ai_F, ai_p = wald(mf, AI)
    cr_F, cr_p = wald(mf, CR)
    ai_coef = mf.params[AI]
    cr_coef = mf.params[CR]

    row = {
        "cat": cat,
        "n": len(d),
        "base_r2": mb.rsquared,
        "full_r2": mf.rsquared,
        "ai_F": ai_F,
        "ai_p": ai_p,
        "crypto_F": cr_F,
        "crypto_p": cr_p,
        "ai_inc_r2": mf.rsquared - mc.rsquared,
        "crypto_inc_r2": mf.rsquared - ma.rsquared,
        "ai_pr2": pr2(mf, mc),
        "crypto_pr2": pr2(mf, ma),
        "ai_sens": float(np.sqrt((ai_coef ** 2).sum())),
        "crypto_sens": float(np.sqrt((cr_coef ** 2).sum())),
        "ai_cum": float(ai_coef.sum()),
        "crypto_cum": float(cr_coef.sum()),
    }

    comp = pd.DataFrame([
        {"cat": cat, "model": "base", "r2": mb.rsquared, "adj_r2": mb.rsquared_adj, "aic": mb.aic, "bic": mb.bic},
        {"cat": cat, "model": "ai_only", "r2": ma.rsquared, "adj_r2": ma.rsquared_adj, "aic": ma.aic, "bic": ma.bic},
        {"cat": cat, "model": "crypto_only", "r2": mc.rsquared, "adj_r2": mc.rsquared_adj, "aic": mc.aic, "bic": mc.bic},
        {"cat": cat, "model": "full", "r2": mf.rsquared, "adj_r2": mf.rsquared_adj, "aic": mf.aic, "bic": mf.bic},
    ])

    vt = vif(d, FULL)
    vt["cat"] = cat
    return row, comp, vt, mf

rows, comps, vifs, mods = [], [], [], {}
for cat in ORDER:
    row, comp, vt, mod = run(cat)
    rows.append(row)
    comps.append(comp)
    vifs.append(vt)
    mods[cat] = mod

main = pd.DataFrame(rows)
main["r2_win"] = [win(a, c) for a, c in zip(main["ai_pr2"], main["crypto_pr2"])]
main["sens_win"] = [win(a, c) for a, c in zip(main["ai_sens"], main["crypto_sens"])]

compare = pd.concat(comps, ignore_index=True)
vif_out = pd.concat(vifs, ignore_index=True)

display(main.round(5))
display(compare.round(5))
display(vif_out.round(3))

main.to_csv(RES / "main.csv", index=False)
compare.to_csv(RES / "compare.csv", index=False)
vif_out.to_csv(RES / "vif.csv", index=False)

for cat, mod in mods.items():
    print("\n" + "=" * 80)
    print(cat)
    print(mod.summary())

import warnings
warnings.filterwarnings("ignore")

FORM = {
    "base": "ret ~ " + " + ".join(BASE),
    "ai": "ret ~ " + " + ".join(BASE + AI),
    "crypto": "ret ~ " + " + ".join(BASE + CR),
    "full": "ret ~ " + " + ".join(FULL),
}

def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))

def forecast(cat):
    d = model[model["cat"] == cat].dropna(subset=["ret"] + FULL).sort_values("date").reset_index(drop=True)
    rows = []
    for i in range(MIN_TRAIN, len(d)):
        tr = d.iloc[:i]
        te = d.iloc[[i]]
        for lab, form in FORM.items():
            m = fit(form, tr, hac=False)
            rows.append({
                "cat": cat,
                "date": te["date"].iloc[0],
                "model": lab,
                "y": te["ret"].iloc[0],
                "pred": m.predict(te).iloc[0],
            })
    return pd.DataFrame(rows)

pred = pd.concat([forecast(c) for c in ORDER], ignore_index=True)

fc = (
    pred.groupby(["cat", "model"])
        .apply(lambda x: rmse(x["y"], x["pred"]))
        .reset_index(name="rmse")
)

base_rmse = fc[fc["model"] == "base"][["cat", "rmse"]].rename(columns={"rmse": "base_rmse"})
fc = fc.merge(base_rmse, on="cat", how="left")
fc["gain"] = fc["base_rmse"] - fc["rmse"]

wide_gain = fc.pivot(index="cat", columns="model", values="gain").reset_index()
wide_gain = wide_gain.rename(columns={
    "ai": "ai_rmse_gain",
    "crypto": "crypto_rmse_gain",
    "full": "full_rmse_gain",
})
wide_gain["rmse_win"] = [
    gain_win(a, c)
    for a, c in zip(wide_gain["ai_rmse_gain"], wide_gain["crypto_rmse_gain"])
]

loss_rows = []
for cat in ORDER:
    p = pred[pred["cat"] == cat].pivot(index="date", columns="model", values="pred")
    y = pred[pred["cat"] == cat].drop_duplicates("date").set_index("date")["y"]
    d = ((y - p["ai"]) ** 2 - (y - p["crypto"]) ** 2).dropna()
    mean_diff, pval = loss_p(d)
    loss_rows.append({
        "cat": cat,
        "ai_minus_crypto_loss": mean_diff,
        "loss_p": pval,
    })

loss = pd.DataFrame(loss_rows)

add_cols = [
    "ai_rmse_gain",
    "crypto_rmse_gain",
    "full_rmse_gain",
    "rmse_win",
    "ai_minus_crypto_loss",
    "loss_p",
]

drop_cols = [
    c for c in main.columns
    if c in add_cols or any(c.startswith(x + "_") for x in add_cols)
]

main = main.drop(columns=drop_cols)
main = main.merge(
    wide_gain[["cat", "ai_rmse_gain", "crypto_rmse_gain", "full_rmse_gain", "rmse_win"]],
    on="cat",
    how="left",
)
main = main.merge(loss, on="cat", how="left")

pred.to_csv(RES / "forecast_pred.csv", index=False)
fc.to_csv(RES / "forecast_rmse.csv", index=False)
loss.to_csv(RES / "forecast_loss_test.csv", index=False)
main.to_csv(RES / "main.csv", index=False)

show_cols = [
    "cat",
    "ai_pr2",
    "crypto_pr2",
    "ai_sens",
    "crypto_sens",
    "ai_rmse_gain",
    "crypto_rmse_gain",
    "rmse_win",
    "ai_minus_crypto_loss",
    "loss_p",
]

display(fc.round(6))
display(loss.round(6))
display(main[show_cols].round(6))

plot1 = main.set_index("cat")[["ai_pr2", "crypto_pr2"]].rename(columns={"ai_pr2": "AI", "crypto_pr2": "Crypto"}).loc[ORDER]
plt.figure(figsize=(8, 4))
plot1.plot(kind="bar", ax=plt.gca())
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("AI vs Crypto partial R2")
plt.xlabel("Hardware")
plt.ylabel("Partial R2")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
save("pr2_bar")
plt.show()

plot2 = main.set_index("cat")[["ai_sens", "crypto_sens"]].rename(columns={"ai_sens": "AI", "crypto_sens": "Crypto"}).loc[ORDER]
plt.figure(figsize=(8, 4))
plot2.plot(kind="bar", ax=plt.gca())
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("AI vs Crypto standardized sensitivity")
plt.xlabel("Hardware")
plt.ylabel("L2 norm of standardized coefficients")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
save("sens_bar")
plt.show()

plot3 = main.set_index("cat")[["ai_rmse_gain", "crypto_rmse_gain"]].rename(columns={"ai_rmse_gain": "AI", "crypto_rmse_gain": "Crypto"}).loc[ORDER]
plt.figure(figsize=(8, 4))
plot3.plot(kind="bar", ax=plt.gca())
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("AI vs Crypto rolling forecast RMSE gain")
plt.xlabel("Hardware")
plt.ylabel("Base RMSE - channel RMSE")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
save("rmse_gain_bar")
plt.show()

coef_rows = []
for cat, mod in mods.items():
    for t in TERMS:
        coef_rows.append({
            "cat": cat,
            "term": t,
            "coef": mod.params[t],
            "se": mod.bse[t],
            "p": mod.pvalues[t],
            "channel": "AI" if t in AI else "Crypto",
            "window": t.split("_", 1)[1],
        })

coef = pd.DataFrame(coef_rows)
coef.to_csv(RES / "coef.csv", index=False)
display(coef.round(5))

xmap = {"0": 0, "12": 1, "38": 2}
xlbl = {"0": "t", "12": "t-1:t-2", "38": "t-3:t-8"}

for cat in ORDER:
    d = coef[coef["cat"] == cat].copy()
    d["x"] = d["window"].map(xmap)
    plt.figure(figsize=(7, 4))
    for ch in ["AI", "Crypto"]:
        s = d[d["channel"] == ch].sort_values("x")
        plt.errorbar(s["x"], s["coef"], yerr=1.96 * s["se"], marker="o", capsize=4, label=ch)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks([0, 1, 2], [xlbl[k] for k in ["0", "12", "38"]])
    plt.title(f"{cat}: AI vs Crypto lag-window coefficients")
    plt.xlabel("Lag window")
    plt.ylabel("Coefficient")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save(f"coef_{cat.lower()}")
    plt.show()

events = mkt[["ai", "crypto"]].dropna().copy()
ai_cut = events["ai"].quantile(Q)
cr_cut = events["crypto"].quantile(Q)

events["ai_raw"] = events["ai"] >= ai_cut
events["crypto_raw"] = events["crypto"] >= cr_cut
events["overlap"] = events["ai_raw"] & events["crypto_raw"]
events["ai_event"] = events["ai_raw"] & ~events["overlap"]
events["crypto_event"] = events["crypto_raw"] & ~events["overlap"]

ecnt = pd.DataFrame([{
    "ai_raw": int(events["ai_raw"].sum()),
    "crypto_raw": int(events["crypto_raw"].sum()),
    "overlap": int(events["overlap"].sum()),
    "ai_event": int(events["ai_event"].sum()),
    "crypto_event": int(events["crypto_event"].sum()),
}])

display(ecnt)
events.to_csv(RES / "events.csv")

plt.figure(figsize=(10, 4))
plt.plot(events.index, events["ai"], label="AI shock")
plt.plot(events.index, events["crypto"], label="Crypto shock", alpha=0.8)
plt.scatter(events.index[events["ai_event"]], events.loc[events["ai_event"], "ai"], marker="^", label="AI event")
plt.scatter(events.index[events["crypto_event"]], events.loc[events["crypto_event"], "crypto"], marker="v", label="Crypto event")
plt.axhline(ai_cut, linestyle="--", linewidth=1)
plt.axhline(cr_cut, linestyle="--", linewidth=1)
plt.title("AI and Crypto shock event weeks")
plt.xlabel("Date")
plt.ylabel("z-score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save("event_timeline")
plt.show()

def abn(cat):
    d = model[model["cat"] == cat].dropna(subset=["ret"] + BASE).copy()
    mod = fit("ret ~ " + " + ".join(BASE), d)
    out = d[["date", "cat", "ret"] + BASE].copy()
    out["abn"] = mod.resid
    return out

ab = pd.concat([abn(c) for c in ORDER], ignore_index=True)
ab = ab.merge(events[["ai_event", "crypto_event"]], left_on="date", right_index=True, how="left")
ab[["ai_event", "crypto_event"]] = ab[["ai_event", "crypto_event"]].fillna(False)


def car_one(cat, max_h=8):
    d = ab[ab["cat"] == cat].sort_values("date").reset_index(drop=True)
    rows = []
    for col, lab in [("ai_event", "AI"), ("crypto_event", "Crypto")]:
        for i in d.index[d[col]]:
            for h in range(max_h + 1):
                j = i + h
                if j < len(d):
                    rows.append({"cat": cat, "event": lab, "event_date": d.loc[i, "date"], "h": h, "car": d.loc[i:j, "abn"].sum()})
    return pd.DataFrame(rows)


def perm(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.r_[a, b]
    rng = np.random.default_rng(SEED)
    vals = np.empty(B)
    for i in range(B):
        p = rng.permutation(pool)
        vals[i] = p[:len(a)].mean() - p[len(a):].mean()
    return obs, np.mean(np.abs(vals) >= abs(obs))

car = pd.concat([car_one(c) for c in ORDER], ignore_index=True)
rows = []
for cat in ORDER:
    for h in [4, 8]:
        d = car[(car["cat"] == cat) & (car["h"] == h)]
        a = d[d["event"] == "AI"]["car"]
        c = d[d["event"] == "Crypto"]["car"]
        if len(a) and len(c):
            diff, p = perm(a, c)
            rows.append({"cat": cat, "h": h, "n_ai": len(a), "n_crypto": len(c), "ai_car": a.mean(), "crypto_car": c.mean(), "ai_minus_crypto": diff, "perm_p": p})

car_sum = pd.DataFrame(rows)
display(car_sum.round(5))
car.to_csv(RES / "car_long.csv", index=False)
car_sum.to_csv(RES / "car.csv", index=False)

path = car.groupby(["cat", "event", "h"])["car"].mean().reset_index()
for cat in ORDER:
    plt.figure(figsize=(7, 4))
    for ev in ["AI", "Crypto"]:
        s = path[(path["cat"] == cat) & (path["event"] == ev)]
        plt.plot(s["h"], s["car"], marker="o", label=ev)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{cat}: mean CAR after AI vs Crypto events")
    plt.xlabel("Weeks after event")
    plt.ylabel("Cumulative abnormal return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save(f"car_{cat.lower()}")
    plt.show()

def run_alt(ai_col, cr_col, label):
    f = make_feat(ai_col, cr_col)

    d0 = ret.merge(f[["fx_ret"] + TERMS], left_on="date", right_index=True, how="inner")
    d0 = d0.sort_values(["cat", "date"]).copy()

    d0["ret_l1"] = d0.groupby("cat")["ret"].shift(1)

    # Quarter fixed effects: q1 is the baseline
    d0["q"] = d0["date"].dt.quarter
    for k in [2, 3, 4]:
        d0[f"q{k}"] = (d0["q"] == k).astype(int)

    rows = []
    for cat in ORDER:
        d = d0[d0["cat"] == cat].dropna(subset=["ret"] + FULL).copy()

        ff = "ret ~ " + " + ".join(FULL)
        fna = "ret ~ " + " + ".join(BASE + CR)
        fnc = "ret ~ " + " + ".join(BASE + AI)

        mf = fit(ff, d)
        mna = fit(fna, d)
        mnc = fit(fnc, d)

        ai_F, ai_p = wald(mf, AI)
        cr_F, cr_p = wald(mf, CR)

        ai_coef = mf.params[AI]
        cr_coef = mf.params[CR]

        rows.append({
            "test": label,
            "cat": cat,
            "n": len(d),
            "ai_p": ai_p,
            "crypto_p": cr_p,
            "ai_pr2": pr2(mf, mna),
            "crypto_pr2": pr2(mf, mnc),
            "ai_sens": float(np.sqrt((ai_coef ** 2).sum())),
            "crypto_sens": float(np.sqrt((cr_coef ** 2).sum())),
        })

    return pd.DataFrame(rows)

rob = pd.concat([
    run_alt("ai_resid_z", "crypto", "resid_nvda_vs_btc"),
    run_alt("ai", "crypto_pc1", "nvda_vs_crypto_pc1"),
    run_alt("ai", "crypto_fxadj", "nvda_vs_btc_fxadj"),
    run_alt("ai_resid_z", "crypto_pc1", "resid_nvda_vs_crypto_pc1"),
], ignore_index=True)

rob["r2_win"] = [win(a, c) for a, c in zip(rob["ai_pr2"], rob["crypto_pr2"])]
rob["sens_win"] = [win(a, c) for a, c in zip(rob["ai_sens"], rob["crypto_sens"])]
rob["judgment"] = np.where(rob["r2_win"] == rob["sens_win"], rob["r2_win"], "Mixed")

display(rob.round(5))
rob.to_csv(RES / "robustness.csv", index=False)

core_cols = ["r2_win", "sens_win"]
all_cols = ["r2_win", "sens_win", "rmse_win"]


def vote(row):
    if row["r2_win"] == row["sens_win"] and row["r2_win"] != "Mixed":
        return row["r2_win"]
    return "Mixed"


def support(row):
    a = sum(row[c] == "AI" for c in all_cols)
    c = sum(row[c] == "Crypto" for c in all_cols)
    if row["judgment"] == "Mixed":
        return "Mixed"
    if row["judgment"] == "AI" and a >= 2:
        return "Core+Diagnostic"
    if row["judgment"] == "Crypto" and c >= 2:
        return "Core+Diagnostic"
    return "Core only"

main["judgment"] = main.apply(vote, axis=1)
main["support_level"] = main.apply(support, axis=1)

final = main[[
    "cat", "n", "ai_p", "crypto_p", "ai_pr2", "crypto_pr2",
    "ai_sens", "crypto_sens", "ai_rmse_gain", "crypto_rmse_gain",
    "ai_minus_crypto_loss", "loss_p", "ai_cum", "crypto_cum",
    "r2_win", "sens_win", "rmse_win", "judgment", "support_level"
]].copy()

final["sentence"] = final["judgment"].map({
    "AI": "AI channel has stronger in-sample explanatory power and standardized sensitivity.",
    "Crypto": "Crypto channel has stronger in-sample explanatory power and standardized sensitivity.",
    "Mixed": "Evidence is mixed; do not claim a single dominant channel.",
})

display(final.round(5))
final.to_csv(RES / "final.csv", index=False)

print("saved tables:", RES)
print("saved figures:", PIC)
