"""E4b - 公共建筑(学校)真实造价驱动分析(SHAP), 在 NYC SCA 阶段级真实造价上。
特征=元数据+工作范围+DDC。SHAP 揭示哪些工类/阶段/学区驱动真实造价(非循环, 真实Y)。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.datasets import load_nyc_sca_raw, ROOT
TAB = ROOT / "04_实验结果" / "tables"; FIG = ROOT / "04_实验结果" / "figures"
SCOPE_KW = ["boiler", "roof", "window", "electr", "ventil", "floor", "ceil", "abate",
            "parapet", "mason", "heat", "cool", "fire", "alarm", "bath", "cafet",
            "gym", "scienc", "lab", "air", "tile", "plaster", "wall", "play"]


def main():
    df = load_nyc_sca_raw()
    d = df[df["cost_spend"] >= 10000].copy()
    d["y"] = np.log1p(d["cost_spend"].astype(float))
    desc = d["Project Description"].astype(str).str.lower()
    for kw in SCOPE_KW:
        d[f"scope_{kw}"] = desc.str.contains(kw).astype(int)
    for c in ["start_year", "planned_dur_days", "n_phases_bldg"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    feats = ["program_type", "phase", "district", "status", "start_year", "planned_dur_days", "n_phases_bldg"] + [f"scope_{k}" for k in SCOPE_KW]
    X = pd.get_dummies(d[feats], columns=["program_type", "phase", "district", "status"], dummy_na=False)
    y = d["y"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    m = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=1, verbosity=0).fit(Xtr, ytr)
    print(f"E4b | NYC SCA SHAP  n={len(y)}  特征={X.shape[1]}  R2log={m.score(Xte,yte):.3f}", flush=True)
    sv = np.asarray(shap.TreeExplainer(m).shap_values(Xte))
    if sv.ndim == 3: sv = sv[..., 0]
    imp = np.mean(np.abs(sv), axis=0)
    order = np.argsort(imp)[::-1]
    cols = list(X.columns)
    dfi = pd.DataFrame([(cols[i], float(imp[i])) for i in order], columns=["feature", "mean_abs_shap"]).head(25)
    dfi.to_csv(TAB / "e4b_nyc_shap.csv", index=False, encoding="utf-8-sig")
    print("\nTop 20 真实造价驱动 (NYC 公共学校):", flush=True)
    print(dfi.head(20).to_string(index=False), flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    top = dfi.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#16a085", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("mean |SHAP value| (log-cost 空间)")
    ax.set_title("E4 · Public School Cost Drivers (NYC SCA, real cost)\nTop features by SHAP — scope/phase/district drive real cost")
    plt.tight_layout(); plt.savefig(FIG / "e4b_nyc_shap_bar.png", dpi=300)
    print("\n已保存: tables/e4b_nyc_shap.csv, figures/e4b_nyc_shap_bar.png", flush=True)


if __name__ == "__main__":
    main()
