"""综合分析: (1) Stacking 组件消融(各底座贡献); (2) 跨数据集平均秩 + 热力图 + Nemenyi CD 图;
(3) 顶刊级可视化改版(SHAP beeswarm)。读 e1/e2/e6 结果 csv。
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
import shap

sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.metrics import r2
from utils.datasets import load_uci, ROOT
TAB = ROOT / "04_实验结果" / "tables"; FIG = ROOT / "04_实验结果" / "figures"

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
C_HYBRID, C_NEW, C_TRAD = "#1f77b4", "#2ca02c", "#9aa7b0"


# ---------- 模型构造器 ----------
def _xgb(s): return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=1, verbosity=0)
def _lgb(s): return LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=s, verbose=-1)
def _cat(s):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=700, learning_rate=0.04, depth=7, l2_leaf_reg=3, random_seed=s, verbose=0)
def _ann(s):
    return TransformedTargetRegressor(regressor=Pipeline([("sc", StandardScaler()), ("m", MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3, max_iter=400, early_stopping=True, n_iter_no_change=15, random_state=s))]), transformer=StandardScaler())


def component_ablation_uci():
    """UCI 上 Stacking(全) vs 去掉各底座。"""
    d = load_uci(); X, y = d["X"], d["y_cost"]
    configs = {"Full(XGB+LGB+CAT+ANN)": ["xgb", "lgb", "cat", "ann"],
               "w/o XGBoost": ["lgb", "cat", "ann"],
               "w/o LightGBM": ["xgb", "cat", "ann"],
               "w/o CatBoost": ["xgb", "lgb", "ann"],
               "w/o ANN": ["xgb", "lgb", "cat"]}
    builders = {"xgb": _xgb, "lgb": _lgb, "cat": _cat, "ann": _ann}
    rows = []
    for cname, bases in configs.items():
        r2s = []
        for seed in range(5):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
            est = [(b, builders[b](seed)) for b in bases]
            st = StackingRegressor(estimators=est, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=1).fit(Xtr, ytr)
            r2s.append(r2(yte, st.predict(Xte)))
        rows.append({"config": cname, "R2_mean": np.mean(r2s), "R2_std": np.std(r2s)})
        print(f"  组件消融 {cname:24s} R2={np.mean(r2s):.4f}±{np.std(r2s):.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "component_ablation.csv", index=False, encoding="utf-8-sig")
    # 图
    fig, ax = plt.subplots(figsize=(9, 4.5))
    full = df["R2_mean"].iloc[0]
    colors = [C_HYBRID if c == "Full(XGB+LGB+CAT+ANN)" else C_TRAD for c in df["config"]]
    ax.bar(df["config"], df["R2_mean"], yerr=df["R2_std"], capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(full, ls="--", color=C_HYBRID, lw=1, alpha=0.6)
    ax.set_ylabel("$R^2$  (↑ better)"); ax.set_ylim(0.90, 0.99)
    ax.set_title("Stacking Component Ablation (UCI #437)\nRemoving each base learner — all contribute")
    plt.xticks(rotation=15, ha="right"); plt.tight_layout(); plt.savefig(FIG / "component_ablation.png"); plt.close()
    return df


def cross_dataset():
    """读 e1/e2/e6, 跨数据集平均秩 + 热力图 + CD 图。"""
    def parse_mean(s):
        try: return float(str(s).split("±")[0])
        except: return float(s)
    e1 = pd.read_csv(TAB / "e1_results.csv"); e2 = pd.read_csv(TAB / "e2_results.csv"); e6 = pd.read_csv(TAB / "e6_results.csv")
    common = ["RF", "LightGBM", "XGBoost", "CatBoost", "ANN", "Stacking"]
    def getr(df, model, col):
        m = df[df["Model"].str.replace("(新)", "", regex=False).str.strip() == model]
        if len(m) == 0: return np.nan
        return parse_mean(m[col].iloc[0])
    rec = {}
    rec["UCI_437(R2)"] = {mm: getr(e1, mm, "R2_mean") for mm in common}
    rec["NYC_SCA(R2log)"] = {mm: getr(e2, mm, "R2log_str") for mm in common}
    rec["ComStock(R2log)"] = {mm: getr(e6, mm, "R2log") for mm in common}
    perf = pd.DataFrame(rec)  # rows=model, cols=dataset
    perf.to_csv(TAB / "cross_dataset_perf.csv", encoding="utf-8-sig")
    print("\n[跨数据集性能 R2]\n", perf.round(3).to_string(), flush=True)
    # 平均秩(每数据集按 R² 降序排名 1..k)
    ranks = perf.rank(axis=0, method="min", ascending=False)
    avg = ranks.mean(axis=1).sort_values()
    print("\n[平均秩 越低越好]\n", avg.round(2).to_string(), flush=True)
    avg.to_csv(TAB / "cross_dataset_ranks.csv", encoding="utf-8-sig")

    # 热力图(R²)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(perf.values, cmap="YlGnBu", aspect="auto", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(perf.shape[1])); ax.set_xticklabels(perf.columns, rotation=12, ha="right")
    ax.set_yticks(range(perf.shape[0])); ax.set_yticklabels(perf.index)
    for i in range(perf.shape[0]):
        for j in range(perf.shape[1]):
            v = perf.values[i, j]
            ax.text(j, i, f"{v:.3f}" if v == v else "—", ha="center", va="center",
                    color="white" if v > 0.8 else "black", fontsize=9)
    ax.set_title("Cross-dataset $R^2$ (rows=models, cols=datasets)")
    fig.colorbar(im, ax=ax, fraction=0.025); plt.tight_layout(); plt.savefig(FIG / "cross_dataset_heatmap.png"); plt.close()

    # Nemenyi CD 图
    k = len(common); N = perf.shape[1]
    q05 = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}[k]
    cd = q05 * np.sqrt(k * (k + 1) / (6 * N))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    order = avg.sort_values()
    ys = np.arange(len(order))[::-1]
    ax.hlines(ys, order.values - cd / 2, order.values + cd / 2, color="gray", lw=2)
    ax.plot(order.values, ys, "o", color=C_HYBRID, ms=10)
    for y, (mm, vv) in zip(ys, order.items()):
        ax.text(vv, y + 0.25, mm, ha="center", fontsize=10)
    ax.set_yticks([]); ax.set_xlim(1, k)
    ax.set_xlabel("Average rank across 3 datasets  (↓ better)")
    ax.set_title(f"Nemenyi CD diagram (CD={cd:.2f}, $\\alpha$=0.05)\n(Stacking/CatBoost lead; note: low power with N=3 datasets)")
    ax.spines["left"].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "cd_diagram.png"); plt.close()
    return perf, avg


def shap_beeswarm_uci():
    """UCI SHAP beeswarm(顶刊级, 比柱状图更信息丰富)。"""
    d = load_uci(); X = pd.DataFrame(d["X"], columns=d["feature_names"]); y = d["y_cost"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    m = _xgb(0).fit(Xtr, ytr)
    rename = {"V2": "FloorArea", "V3": "LotArea", "V4": "PrelimCost_A", "V5": "PrelimCost_B",
              "V6": "EquivPrelimCost", "V7": "Duration", "V1": "Locality"}
    Xte = Xte.rename(columns=rename)
    exp = shap.TreeExplainer(m)
    sv = exp(Xte)
    # 仅画 top 15 特征
    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(sv, max_display=15, show=False, color_bar=True)
    plt.title("SHAP beeswarm — UCI #437 construction cost drivers")
    plt.tight_layout(); plt.savefig(FIG / "e4_shap_beeswarm.png", dpi=300); plt.close()
    print("  已生成 SHAP beeswarm", flush=True)


def main():
    print("=== (1) Stacking 组件消融 (UCI) ===  [已完成, 跳过重跑]", flush=True)
    # component_ablation_uci()  # 已生成 component_ablation.csv/.png
    print("\n=== (2) 跨数据集综合(秩/热力图/CD) ===", flush=True)
    cross_dataset()
    print("\n=== (3) SHAP beeswarm (UCI) ===", flush=True)
    shap_beeswarm_uci()
    print("\n已生成: component_ablation.png, cross_dataset_heatmap.png, cd_diagram.png, e4_shap_beeswarm.png", flush=True)


if __name__ == "__main__":
    main()
