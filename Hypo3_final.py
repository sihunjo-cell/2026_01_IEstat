# Auto-generated from Hypo3_final.ipynb (code cells only, markdown/outputs stripped)


import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

OUT = Path("outputs/hypo3_final")
PIC = Path("output_pic/hypo3_final")
OUT.mkdir(parents=True, exist_ok=True)
PIC.mkdir(parents=True, exist_ok=True)

MAC = Path("outputs/hypo3_relation/data.csv")
INT = Path("outputs/hypo3_internal_reg/int_reg.csv")
assert MAC.exists(), "Run Hypo3_relation.ipynb first."
assert INT.exists(), "Run Hypo3_internal_reg.ipynb first."


mac = pd.read_csv(MAC)
if "Unnamed: 0" in mac.columns:
    mac = mac.rename(columns={"Unnamed: 0": "date"})
mac["date"] = pd.to_datetime(mac["date"])

it = pd.read_csv(INT, parse_dates=["date"])

mx = ["btc_ret_z", "fx_ret_z", "soxx_ret_z", "oil_ret_z", "csi_chg_z"]
spec = {
    "GPU": {
        "y": "gpu_ret",
        "ix": ["gpu_n_chg", "gpu_new_sh", "gpu_riq_l1", "gpu_high_chg", "gpu_vram_chg"],
    },
    "CPU": {
        "y": "cpu_ret",
        "ix": ["cpu_n_chg", "cpu_new_sh", "cpu_riq_l1", "cpu_amd_chg"],
    },
    "RAM": {
        "y": "ram_ret",
        "ix": ["ram_n_chg", "ram_new_sh", "ram_riq_l1", "ram_ddr5_chg", "ram_cap_chg"],
    },
}

df = it.merge(mac[["date"] + mx], on="date", how="inner").sort_values("date")

xall = mx + sorted({x for v in spec.values() for x in v["ix"]})
for c in xall:
    df[c] = (df[c] - df[c].mean()) / df[c].std(ddof=0)

need = ["date"] + mx
for v in spec.values():
    need += [v["y"]] + v["ix"]
reg = df[need].dropna().reset_index(drop=True)
reg.to_csv(OUT / "reg.csv", index=False)
print(reg.shape)
reg.head()


def fit(d, y, x):
    X = sm.add_constant(d[x], has_constant="add")
    return sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})


def vmax(d, x):
    X = sm.add_constant(d[x], has_constant="add")
    vals = []
    for i, c in enumerate(X.columns):
        if c != "const":
            vals.append(variance_inflation_factor(X.values, i))
    return max(vals)


def diag(m):
    lb = acorr_ljungbox(m.resid, lags=[4], return_df=True)["lb_pvalue"].iloc[0]
    bp = het_breuschpagan(m.resid, m.model.exog)[1]
    return float(lb), float(bp)


def wald(m, cols):
    names = list(m.params.index)
    R = []
    for c in cols:
        r = [0] * len(names)
        r[names.index(c)] = 1
        R.append(r)
    return float(m.wald_test(np.asarray(R), scalar=True).pvalue)


def info(hw, name, m, d, y, x):
    lb, bp = diag(m)
    pred = m.fittedvalues
    return {
        "hw": hw,
        "model": name,
        "n": int(m.nobs),
        "k": len(x),
        "r2": m.rsquared,
        "adj_r2": m.rsquared_adj,
        "aic": m.aic,
        "bic": m.bic,
        "mae": np.mean(np.abs(d[y] - pred)),
        "dir_acc": (np.sign(d[y]) == np.sign(pred)).mean(),
        "max_vif": vmax(d, x),
        "lb_p": lb,
        "bp_p": bp,
    }


model_rows = []
coef_rows = []
test_rows = []
fits = {}

for hw, sp in spec.items():
    y = sp["y"]
    ix = sp["ix"]
    cols = ["date", y] + mx + ix
    d = reg[cols].dropna().copy()

    xs = {
        "macro": mx,
        "internal": ix,
        "combined": mx + ix,
    }
    fm = {name: fit(d, y, x) for name, x in xs.items()}
    fits[hw] = fm

    for name, m in fm.items():
        model_rows.append(info(hw, name, m, d, y, xs[name]))
        for c in xs[name]:
            coef_rows.append({
                "hw": hw,
                "model": name,
                "var": c,
                "coef": m.params[c],
                "se": m.bse[c],
                "p": m.pvalues[c],
            })

    test_rows.append({
        "hw": hw,
        "test": "internal_adds_to_macro",
        "p": wald(fm["combined"], ix),
        "adj_gain": fm["combined"].rsquared_adj - fm["macro"].rsquared_adj,
        "aic_gain": fm["macro"].aic - fm["combined"].aic,
        "mae_gain": model_rows[-3]["mae"] - model_rows[-1]["mae"],
    })
    test_rows.append({
        "hw": hw,
        "test": "macro_adds_to_internal",
        "p": wald(fm["combined"], mx),
        "adj_gain": fm["combined"].rsquared_adj - fm["internal"].rsquared_adj,
        "aic_gain": fm["internal"].aic - fm["combined"].aic,
        "mae_gain": model_rows[-2]["mae"] - model_rows[-1]["mae"],
    })

model = pd.DataFrame(model_rows)
coef = pd.DataFrame(coef_rows)
test = pd.DataFrame(test_rows)

model.to_csv(OUT / "model.csv", index=False)
coef.to_csv(OUT / "coef.csv", index=False)
test.to_csv(OUT / "test.csv", index=False)

model.round(4)


# 1. Model fit comparison
fig, ax = plt.subplots(figsize=(8, 4.5))
piv = model.pivot(index="hw", columns="model", values="adj_r2").loc[["GPU", "CPU", "RAM"], ["macro", "internal", "combined"]]
piv.plot(kind="bar", ax=ax)
ax.set_title("Model comparison: adjusted R²")
ax.set_xlabel("hardware")
ax.set_ylabel("adjusted R²")
ax.axhline(0, linewidth=1)
plt.tight_layout()
plt.savefig(PIC / "01_fit.png", dpi=160)
plt.close()

# 2. Added explanatory value
fig, ax = plt.subplots(figsize=(8, 4.5))
g = test.pivot(index="hw", columns="test", values="adj_gain").loc[["GPU", "CPU", "RAM"]]
g.plot(kind="bar", ax=ax)
ax.set_title("Incremental adjusted R²")
ax.set_xlabel("hardware")
ax.set_ylabel("gain over base model")
ax.axhline(0, linewidth=1)
plt.tight_layout()
plt.savefig(PIC / "02_gain.png", dpi=160)
plt.close()

# 3. Combined-model coefficients
cm = coef[coef["model"] == "combined"].copy()
for hw in ["GPU", "CPU", "RAM"]:
    d = cm[cm["hw"] == hw].copy()
    d = d.reindex(d["coef"].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(d["var"], d["coef"])
    ax.axvline(0, linewidth=1)
    ax.set_title(f"{hw}: combined model coefficients")
    ax.set_xlabel("coef on standardized predictor")
    plt.tight_layout()
    plt.savefig(PIC / f"03_coef_{hw.lower()}.png", dpi=160)
    plt.close()

print("csv:", sorted(p.name for p in OUT.glob("*.csv")))
print("plots:", sorted(p.name for p in PIC.glob("*.png")))


# Minimal validation for generated files
for p in [OUT / "reg.csv", OUT / "model.csv", OUT / "coef.csv", OUT / "test.csv"]:
    assert p.exists(), p
for p in [PIC / "01_fit.png", PIC / "02_gain.png", PIC / "03_coef_gpu.png", PIC / "03_coef_cpu.png", PIC / "03_coef_ram.png"]:
    assert p.exists(), p

print("reg rows:", len(reg))
print("missing in reg:", int(reg.isna().sum().sum()))
print("model rows:", len(model), "coef rows:", len(coef), "test rows:", len(test))
test.round(4)


# ── C-1) 가정검토(assumption) 요약 표 — combined 모형 기준 ────────────────────
# model.csv 에 이미 산출된 진단치(max_vif·lb_p·bp_p)를 슬라이드용으로 정리만 한다.
#   max_vif : 다중공선성 (VIF<5 양호)
#   lb_p    : Ljung-Box 자기상관 검정 p값 (<0.05 자기상관 존재)
#   bp_p    : Breusch-Pagan 이분산 검정 p값 (<0.05 이분산 존재)
# 자기상관/이분산이 잡히는 모형은 이미 HAC(Newey-West) 표준오차로 보정되어 있음.
asm = model[model["model"] == "combined"][
    ["hw", "n", "k", "max_vif", "lb_p", "bp_p"]
].copy()
asm["VIF(<5)"] = np.where(asm["max_vif"] < 5, "양호", "주의")
asm["자기상관(LB)"] = np.where(asm["lb_p"] < 0.05, "존재→HAC 보정", "없음")
asm["이분산(BP)"] = np.where(asm["bp_p"] < 0.05, "존재→HAC 보정", "없음")
asm["max_vif"] = asm["max_vif"].round(2)  # p값은 매우 작아 반올림 시 0이 되므로 원값 유지
asm.to_csv(OUT / "assumption_summary.csv", index=False, encoding="utf-8-sig")
print("assumption_summary.csv ->")
print(asm.to_string(index=False))
