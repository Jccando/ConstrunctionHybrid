"""补充实验: (1) 不确定性分析(NGBoost 区间覆盖率/宽度)
(2) SHAP dependence plot (Duration 的边际效应)
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.datasets import load_uci, load_nyc_sca_raw, ROOT

TAB = ROOT / "04_实验结果" / "tables"
FIG = ROOT / "04_实验结果" / "figures"

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "figure.dpi": 130, "savefig.dpi": 250})
SURF, INK2, MUTED, GRID = "#fcfcfb", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
Z90 = 1.645
SCOPE_KW = ["boiler","roof","window","electr","ventil","floor","ceil","abate","parapet","mason",
            "heat","cool","fire","alarm","bath","cafet","gym","scienc","lab","air","tile","plaster","wall","play"]


def style(ax):
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8); ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_facecolor(SURF)


def uncertainty_analysis():
    """NGBoost 90% 区间: 覆盖率 + 区间宽 + 校准"""
    results = []

    # UCI
    d = load_uci(); X, y = d["X"], d["y_cost"]
    for ds_name in ["UCI #437"]:
        for seed in range(3):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
            sc = StandardScaler().fit(Xtr)
            m = NGBRegressor(n_estimators=200, learning_rate=0.02,
                             Base=DecisionTreeRegressor(criterion="friedman_mse", max_depth=3),
                             random_state=seed, verbose=False).fit(sc.transform(Xtr), ytr)
            dist = m.pred_dist(sc.transform(Xte))
            loc = dist.loc.ravel(); scale = dist.scale.ravel()
            lo, hi = loc - Z90*scale, loc + Z90*scale
            cov = np.mean((yte >= lo) & (yte <= hi))
            wid = np.mean(hi - lo)
            results.append({"Dataset": ds_name, "Coverage_90": cov, "MPIW": wid, "Median_Cost": np.median(yte)})
    # NYC
    df = load_nyc_sca_raw(); dd = df[df["cost_spend"] >= 10000].copy().reset_index(drop=True)
    dd["y"] = np.log1p(dd["cost_spend"].astype(float))
    desc = dd["Project Description"].astype(str).str.lower()
    for kw in SCOPE_KW: dd[f"scope_{kw}"] = desc.str.contains(kw).astype(int)
    for c in ["start_year","planned_dur_days","n_phases_bldg"]: dd[c] = pd.to_numeric(dd[c], errors="coerce").fillna(0)
    feats = ["program_type","phase","district","status","start_year","planned_dur_days","n_phases_bldg"] + [f"scope_{k}" for k in SCOPE_KW]
    X2 = pd.get_dummies(dd[feats], columns=["program_type","phase","district","status"], dummy_na=False).astype(float).values
    y2 = dd["y"].values
    for seed in range(3):
        Xtr, Xte, ytr, yte = train_test_split(X2, y2, test_size=0.2, random_state=seed)
        sc = StandardScaler().fit(Xtr)
        m = NGBRegressor(n_estimators=150, learning_rate=0.04,
                         Base=DecisionTreeRegressor(criterion="friedman_mse", max_depth=4),
                         random_state=seed, verbose=False).fit(sc.transform(Xtr), ytr)
        dist = m.pred_dist(sc.transform(Xte))
        loc = dist.loc.ravel(); scale = dist.scale.ravel()
        lo, hi = loc - Z90*scale, loc + Z90*scale
        cov = np.mean((yte >= lo) & (yte <= hi))
        wid = np.mean(np.expm1(hi) - np.expm1(lo))
        results.append({"Dataset": "NYC SCA", "Coverage_90": cov, "MPIW": wid, "Median_Cost": np.median(np.expm1(yte))})

    dfr = pd.DataFrame(results)
    agg = dfr.groupby("Dataset").agg(Coverage=("Coverage_90","mean"), MPIW=("MPIW","mean"), MedianCost=("Median_Cost","mean")).reset_index()
    agg["Relative_Width_pct"] = (agg["MPIW"] / agg["MedianCost"] * 100).round(1)
    agg.to_csv(TAB / "uncertainty_analysis.csv", index=False)
    print("=== Uncertainty Analysis (NGBoost 90% interval) ===")
    print(agg.to_string(index=False))
    return agg


def shap_dependence():
    """SHAP dependence plot: Duration vs SHAP value (NYC)"""
    import shap
    from xgboost import XGBRegressor
    df = load_nyc_sca_raw(); dd = df[df["cost_spend"] >= 10000].copy().reset_index(drop=True)
    dd["y"] = np.log1p(dd["cost_spend"].astype(float))
    desc = dd["Project Description"].astype(str).str.lower()
    for kw in SCOPE_KW: dd[f"scope_{kw}"] = desc.str.contains(kw).astype(int)
    for c in ["start_year","planned_dur_days","n_phases_bldg"]: dd[c] = pd.to_numeric(dd[c], errors="coerce").fillna(0)
    feats = ["program_type","phase","district","status","start_year","planned_dur_days","n_phases_bldg"] + [f"scope_{k}" for k in SCOPE_KW]
    X = pd.get_dummies(dd[feats], columns=["program_type","phase","district","status"], dummy_na=False).astype(float)
    y = dd["y"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    m = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, random_state=0, n_jobs=1, verbosity=0).fit(Xtr, ytr)
    sv = np.asarray(shap.TreeExplainer(m).shap_values(Xte))
    if sv.ndim == 3: sv = sv[..., 0]

    # Duration dependence
    dur_idx = list(X.columns).index("planned_dur_days")
    dur_vals = Xte["planned_dur_days"].values
    dur_shap = sv[:, dur_idx]
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=SURF)
    ax.scatter(dur_vals / 30, dur_shap, s=12, alpha=0.5, c=BLUE, edgecolors="none", zorder=3)
    z = np.polyfit(dur_vals / 30, dur_shap, 2)
    x_s = np.linspace(0, (dur_vals / 30).max(), 100)
    ax.plot(x_s, np.polyval(z, x_s), color=ORANGE, lw=2, zorder=4)
    ax.set_xlabel("Planned duration (months)", color=INK2)
    ax.set_ylabel("SHAP value (log-cost impact)", color=INK2)
    ax.axhline(0, color="#c3c2b7", ls="--", lw=0.8)
    style(ax)
    plt.tight_layout()
    plt.savefig(FIG / "shap_dependence_duration.png", dpi=250, bbox_inches="tight", facecolor=SURF)
    plt.close()
    print(f"SHAP dependence: Duration range {dur_vals.min()/30:.0f}--{dur_vals.max()/30:.0f} months")
    print(f"  SHAP range: [{dur_shap.min():.3f}, {dur_shap.max():.3f}]")

    # Roof scope dependence
    roof_idx = list(X.columns).index("scope_roof")
    roof_shap_mean = np.mean(sv[Xte["scope_roof"].values == 1, roof_idx])
    noroof_shap_mean = np.mean(sv[Xte["scope_roof"].values == 0, roof_idx])
    print(f"\n  scope_roof: with={roof_shap_mean:.3f}, without={noroof_shap_mean:.3f}, delta={roof_shap_mean-noroof_shap_mean:.3f}")
    # Masonry
    mas_idx = list(X.columns).index("scope_mason")
    mas_shap = np.mean(sv[Xte["scope_mason"].values == 1, mas_idx])
    nomas_shap = np.mean(sv[Xte["scope_mason"].values == 0, mas_idx])
    print(f"  scope_mason: with={mas_shap:.3f}, without={nomas_shap:.3f}, delta={mas_shap-nomas_shap:.3f}")
    # Phase Construction
    ph_idx = list(X.columns).index("phase_Construction")
    ph_shap = np.mean(sv[Xte["phase_Construction"].values == 1, ph_idx])
    noph_shap = np.mean(sv[Xte["phase_Construction"].values == 0, ph_idx])
    print(f"  phase_Construction: with={ph_shap:.3f}, without={noph_shap:.3f}")


def main():
    print("--- (1) Uncertainty Analysis ---", flush=True)
    uncertainty_analysis()
    print("\n--- (2) SHAP Dependence ---", flush=True)
    shap_dependence()
    print("\nDone. Outputs: uncertainty_analysis.csv, shap_dependence_duration.png")


if __name__ == "__main__":
    main()
