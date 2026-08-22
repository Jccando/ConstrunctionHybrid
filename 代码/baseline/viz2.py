"""顶刊级成图(无 title, 编号): Fig4 预测-实际 parity / Fig7 多种子稳定性小提琴。
需训练(proposed Stacking / 各模型)。多面板 (a)(b)(c) 角标, 无 title。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from utils.datasets import load_uci, load_nyc_sca_raw, load_comstock_public, ROOT
from utils.metrics import mape, r2

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans", "figure.dpi": 130, "savefig.dpi": 300, "axes.unicode_minus": False})
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#008300"
FIG = ROOT / "04_实验结果" / "figures"


def style(ax, ygrid=True):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9, length=3); ax.set_axisbelow(True)
    if ygrid: ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_facecolor(SURF)


def panel(ax, label, x=-0.02, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, va="top")


def _stack(seed):
    def xgb(s): return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=1, verbosity=0)
    def lgb(s): return LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, random_state=s, verbose=-1)
    def cat(s):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=600, learning_rate=0.05, depth=7, random_seed=s, verbose=0)
    def ann(s): return TransformedTargetRegressor(regressor=Pipeline([("sc", StandardScaler()), ("m", MLPRegressor(hidden_layer_sizes=(128,64), alpha=1e-2, learning_rate_init=1e-3, max_iter=400, early_stopping=True, n_iter_no_change=12, random_state=s))]), transformer=StandardScaler())
    return StackingRegressor(estimators=[("xgb", xgb(seed)), ("lgb", lgb(seed)), ("cat", cat(seed)), ("ann", ann(seed))], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=1)


def _nyc_xy():
    df = load_nyc_sca_raw(); d = df[df["cost_spend"] >= 10000].copy()
    d["y"] = np.log1p(d["cost_spend"].astype(float)); desc = d["Project Description"].astype(str).str.lower()
    for kw in ["boiler","roof","window","electr","ventil","floor","ceil","abate","parapet","mason","heat","cool","fire","alarm","bath","cafet","gym","scienc","lab","air","tile","plaster","wall","play"]:
        d[f"scope_{kw}"] = desc.str.contains(kw).astype(int)
    for c in ["start_year","planned_dur_days","n_phases_bldg"]: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    feats = ["program_type","phase","district","status","start_year","planned_dur_days","n_phases_bldg"] + [c for c in d.columns if c.startswith("scope_")]
    X = pd.get_dummies(d[feats], columns=["program_type","phase","district","status"], dummy_na=False).astype(float)
    return X.values, d["y"].values


def _comstock_xy():
    cs = load_comstock_public(); cs = cs[cs["in.comstock_building_type"].isin(["PrimarySchool","SecondarySchool","SmallOffice","MediumOffice","LargeOffice","Outpatient","Hospital","RetailStandalone"])].sample(8000, random_state=42)
    cs["log_sqft"] = np.log1p(pd.to_numeric(cs["in.sqft"], errors="coerce")); cs["stories"] = pd.to_numeric(cs["in.number_of_stories"], errors="coerce").fillna(2)
    feats = ["log_sqft","stories","in.comstock_building_type","in.vintage","in.hvac_system_type","in.ashrae_iecc_climate_zone_2006"]
    X = pd.get_dummies(cs[feats], columns=["in.comstock_building_type","in.vintage","in.hvac_system_type","in.ashrae_iecc_climate_zone_2006"], dummy_na=False).astype(float)
    BASE = {"Hospital":550,"Outpatient":380,"LargeOffice":300,"MediumOffice":250,"SmallOffice":220,"PrimarySchool":270,"SecondarySchool":260,"RetailStandalone":180}
    rng = np.random.default_rng(42)
    cost = pd.to_numeric(cs["in.sqft"],errors="coerce") * cs["in.comstock_building_type"].astype(str).map(BASE) * rng.lognormal(0,0.28,len(cs))
    return X.values, np.log1p(cost.values)


def fig4_parity():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), facecolor=SURF)
    panels = [("(a)", lambda: (load_uci()["X"], load_uci()["y_cost"])),
              ("(b)", _nyc_xy), ("(c)", _comstock_xy)]
    for ax, (lab, loader) in zip(axes, panels):
        X, y = loader()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
        m = _stack(0).fit(Xtr, ytr); pred = m.predict(Xte); rr = r2(yte, pred)
        ax.hexbin(yte, pred, gridsize=35, cmap="Blues", mincnt=1, zorder=2)
        lo = min(yte.min(), pred.min()); hi = max(yte.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color=ORANGE, lw=1.8, zorder=3)
        ax.set_xlabel("actual (log space)", color=INK2)
        if ax is axes[0]: ax.set_ylabel("predicted (log space)", color=INK2)
        ax.text(0.97, 0.04, f"$R^2$ = {rr:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color=INK,
                bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.3", alpha=0.9))
        style(ax, ygrid=False); panel(ax, lab, x=-0.02, y=1.04)
    plt.tight_layout(); plt.savefig(FIG / "Fig4_parity.png", dpi=300, bbox_inches="tight", facecolor=SURF); plt.close()


def fig7_violin():
    d = load_uci(); X, y = d["X"], d["y_cost"]
    def xgb(s): return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, random_state=s, n_jobs=1, verbosity=0)
    def lgb(s): return LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, random_state=s, verbose=-1)
    def cat(s):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=600, learning_rate=0.05, depth=7, random_seed=s, verbose=0)
    def ann(s): return Pipeline([("sc", StandardScaler()), ("m", MLPRegressor(hidden_layer_sizes=(128,64), alpha=1e-2, learning_rate_init=1e-3, max_iter=400, early_stopping=True, n_iter_no_change=12, random_state=s))])
    models = [("Stacking", lambda s: _stack(s), BLUE), ("CatBoost", cat, ORANGE), ("LightGBM", lgb, AQUA), ("ANN", ann, GREEN)]
    data, names, colors = [], [], []
    for name, mb, col in models:
        vals = []
        for seed in range(8):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
            m = mb(seed).fit(Xtr, ytr); vals.append(mape(yte, m.predict(Xte)))
        data.append(vals); names.append(name); colors.append(col)
    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=SURF)
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i]); pc.set_edgecolor("none"); pc.set_alpha(0.55)
    for i, vals in enumerate(data):
        ax.scatter(np.full_like(vals, i+1)+np.random.default_rng(i).normal(0,0.04,len(vals)), vals, color=colors[i], s=22, zorder=3, edgecolors=SURF, linewidths=1)
        ax.scatter([i+1], [np.median(vals)], color=colors[i], s=60, marker="_", zorder=4)
    ax.set_xticks(range(1, len(names)+1)); ax.set_xticklabels([n+(" (proposed)" if n=="Stacking" else "") for n in names], color=INK2)
    ax.set_ylabel("MAPE (%) across 8 seeds", color=INK2); style(ax)
    plt.tight_layout(); plt.savefig(FIG / "Fig7_stability_violin.png", dpi=300, bbox_inches="tight", facecolor=SURF); plt.close()


def main():
    print("生成 Fig4 / Fig7 (无 title, 编号)...", flush=True)
    fig4_parity(); print("  Fig4_parity.png", flush=True)
    fig7_violin(); print("  Fig7_stability_violin.png", flush=True)


if __name__ == "__main__":
    main()
