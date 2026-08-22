"""E4 - 设计阶段成本驱动分析(SHAP)，在真实造价+真实特征上(非循环)。
UCI #437: 真实建造成本 V-10 + 真实面积/工期/经济变量 -> SHAP 全局重要性 + 依赖图。
产出: figures/e4_shap_bar.png, e4_shap_dep_*.png, tables/e4_shap_importance.csv
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
from utils.datasets import load_uci

ROOT = Path(__file__).resolve().parents[2]  # 项目根
TAB = ROOT / "04_实验结果" / "tables"; FIG = ROOT / "04_实验结果" / "figures"

# 可读特征名映射（UCI 物理与财务变量）
RENAME = {"V1": "Locality", "V2": "FloorArea_m2", "V3": "LotArea_m2",
          "V4": "PrelimCost_A", "V5": "PrelimCost_B", "V6": "EquivPrelimCost",
          "V7": "Duration", "V8": "V8",
          "StartYear": "StartYear", "StartQuarter": "StartQuarter",
          "CompletionYear": "CompletionYear", "CompletionQuarter": "CompletionQuarter"}
# 经济变量含义（V11-V29），用于注释顶级经济特征
ECON_MEANING = {"V11": "#BuildingPermits", "V12": "BSI", "V13": "WPI_materials",
                "V14": "PermitFloorArea", "V16": "PrivateInvestment",
                "V21": "AvgCost_m2_complete", "V22": "AvgCost_m2_start",
                "V23": "LandPrice", "V29": "CPI"}


def readable(name):
    if name in RENAME:
        return RENAME[name]
    # 经济变量 V{num}_L{lag}
    if "_L" in name:
        base, lag = name.split("_L")
        mean = ECON_MEANING.get(base, base)
        return f"{mean}_lag{lag}"
    return name


def main():
    d = load_uci(); X = pd.DataFrame(d["X"], columns=d["feature_names"]); y = d["y_cost"]
    Xc = X.rename(columns={c: readable(c) for c in X.columns})
    Xtr, Xte, ytr, yte = train_test_split(Xc, y, test_size=0.2, random_state=0)
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, random_state=0, n_jobs=1, verbosity=0).fit(Xtr, ytr)
    print(f"E4 | UCI SHAP  X={Xc.shape}  test R2={model.score(Xte,yte):.3f}", flush=True)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 0]

    imp = np.mean(np.abs(sv), axis=0)
    order = np.argsort(imp)[::-1]
    cols = list(Xte.columns)
    top = [(cols[i], float(imp[i])) for i in order]
    df_imp = pd.DataFrame(top, columns=["feature", "mean_abs_shap"]).head(25)
    df_imp.to_csv(TAB / "e4_shap_importance.csv", index=False, encoding="utf-8-sig")
    print("\nTop 15 成本驱动特征 (mean|SHAP|):", flush=True)
    print(df_imp.head(15).to_string(index=False), flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # 全局重要性条形图(top 20)
    top20 = df_imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top20["feature"], top20["mean_abs_shap"], color="#2980b9", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("mean |SHAP value|  (平均边际贡献)")
    ax.set_title("E4 · Construction Cost Drivers (UCI #437, real cost)\nTop 20 features by SHAP importance")
    plt.tight_layout(); plt.savefig(FIG / "e4_shap_bar.png", dpi=300); plt.close()

    # 依赖图：前 3 个特征
    for k in range(3):
        fname = df_imp["feature"].iloc[k]
        try:
            idx = list(Xte.columns).index(fname)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(Xte[fname], sv[:, idx], c=yte, cmap="viridis", s=18, alpha=0.8)
            ax.set_xlabel(fname); ax.set_ylabel(f"SHAP value for {fname}")
            ax.set_title(f"E4 · Cost driver: {fname} (rank {k+1})\ncolor = actual construction cost")
            plt.tight_layout(); plt.savefig(FIG / f"e4_shap_dep_{k+1}_{fname}.png", dpi=300); plt.close()
        except Exception as e:
            print(f"  dep plot {fname} skipped: {e}", flush=True)
    print("\n已保存: figures/e4_shap_bar.png, e4_shap_dep_*.png, tables/e4_shap_importance.csv", flush=True)


if __name__ == "__main__":
    main()
