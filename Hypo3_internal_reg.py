# Auto-generated from Hypo3_internal_reg.ipynb (code cells only, markdown/outputs stripped)


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN = Path("outputs/hypo3_internal/int_w.csv")
OUT = Path("outputs/hypo3_internal_reg")
PIC = Path("output_pic/hypo3_internal_reg")

OUT.mkdir(parents=True, exist_ok=True)
PIC.mkdir(parents=True, exist_ok=True)

assert IN.exists(), "Run Hypo3_internal.ipynb first: outputs/hypo3_internal/int_w.csv not found."

w = pd.read_csv(IN, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
w.head()


P = ["gpu", "cpu", "ram"]

reg = pd.DataFrame({"date": w["date"]})


def pos(p):
    return w[f"{p}_act_n"] > 0


def ok(p):
    return pos(p) & pos(p).shift(1, fill_value=False)


def lchg(s, m):
    x = s.where(s > 0)
    y = np.log(x).diff()
    return y.where(m)


def dchg(s, m):
    return s.diff().where(m)


def nsh(p):
    return w[f"{p}_in_sh"].where(ok(p))


def lag1(s, m):
    return s.shift(1).where(m)

for p in P:
    m = ok(p)
    reg[f"{p}_ret"] = w[f"{p}_ret"].where(pos(p))
    reg[f"{p}_n_chg"] = lchg(w[f"{p}_act_n"], m)
    reg[f"{p}_new_sh"] = nsh(p)
    reg[f"{p}_riq_l1"] = lag1(w[f"{p}_ret_iqr"], m)

reg["gpu_high_chg"] = dchg(w["gpu_high_sh"], ok("gpu"))
reg["gpu_vram_chg"] = dchg(w["gpu_vram_avg"], ok("gpu"))
reg["cpu_amd_chg"] = dchg(w["cpu_amd_sh"], ok("cpu"))
reg["ram_ddr5_chg"] = dchg(w["ram_ddr5_sh"], ok("ram"))
reg["ram_cap_chg"] = dchg(w["ram_cap_avg"], ok("ram"))

cols = [c for c in reg.columns if c != "date"]
raw_n = len(reg)
reg.head()


why = {
    "ret": "first/gap return unavailable",
    "n_chg": "log-difference requires current and previous positive active count",
    "new_sh": "entry share requires previous positive active count; post-gap value is artifact",
    "riq_l1": "lagged internal-volatility variable is undefined after first/gap week",
    "mix": "first-difference requires current and previous valid mix share",
}

rows = []
for c in cols:
    if c.endswith("_ret"):
        typ = "ret"
    elif c.endswith("_n_chg"):
        typ = "n_chg"
    elif c.endswith("_new_sh"):
        typ = "new_sh"
    elif c.endswith("_riq_l1"):
        typ = "riq_l1"
    else:
        typ = "mix"
    rows.append({
        "var": c,
        "na": int(reg[c].isna().sum()),
        "na_pct": round(reg[c].isna().mean() * 100, 2),
        "mech": "structural",
        "method": "drop",
        "reason": why[typ],
    })

miss = pd.DataFrame(rows)
reg2 = reg.dropna(subset=cols).reset_index(drop=True)

miss.loc[len(miss)] = {
    "var": "final_rows",
    "na": raw_n - len(reg2),
    "na_pct": round((raw_n - len(reg2)) / raw_n * 100, 2),
    "mech": "structural_complete_case",
    "method": "drop",
    "reason": f"kept {len(reg2)} of {raw_n} weeks after structural-cleaning",
}

reg2.to_csv(OUT / "int_reg.csv", index=False)
miss.to_csv(OUT / "miss.csv", index=False)

print("saved:", OUT / "int_reg.csv")
print("saved:", OUT / "miss.csv")
print("shape:", reg2.shape)
miss


corr = reg2[cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=90)
ax.set_yticklabels(cols)
fig.colorbar(im, ax=ax, label="corr")
ax.set_title("internal regression variables")
plt.tight_layout()
plt.savefig(PIC / "01_corr.png", dpi=160)
plt.close()

print("saved:", PIC / "01_corr.png")
