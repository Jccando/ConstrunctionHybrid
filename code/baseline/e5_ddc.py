"""E5 - 成本知识增强消融阶梯（诚实版）。
NYC SCA 阶段级真实造价上, 逐层注入知识:
  M0 元数据(program/phase/district/status/year/duration/n_phases)
  M1 + 工作范围知识(DDC 分类法指导的 24 个工类关键词, 从描述抽取)
  M2 + DDC 成本知识库单价(命中工类的中位单价 -> 成本强度特征)
模型: CatBoost / Stacking(+ANN) 混合。证明: 注入施工领域知识(工作范围)大幅提升造价估算。
"""
import sys, time, warnings
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

sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.metrics import mape, r2, wilcoxon_p
from utils.datasets import load_nyc_sca_raw, ROOT
from utils.metaheuristics import ga

TAB = ROOT / "04_实验结果" / "tables"; FIG = ROOT / "04_实验结果" / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
N_SEEDS = 5; TEST_SIZE = 0.2; MIN_SPEND = 10000
SCOPE_KW = ["boiler", "roof", "window", "electr", "ventil", "floor", "ceil", "abate",
            "parapet", "mason", "heat", "cool", "fire", "alarm", "bath", "cafet",
            "gym", "scienc", "lab", "air", "tile", "plaster", "wall", "play"]
SYN = {"electr": "electric", "ventil": "ventil", "ceil": "ceil", "abate": "abate|demolit|asbestos",
       "parapet": "parapet|roof", "mason": "mason|concret|brick|block", "cool": "cool|air condition|hvac",
       "heat": "heat|boiler|hvac", "fire": "fire|sprinkler", "alarm": "alarm", "bath": "bath|toilet|sanitar",
       "cafet": "cafet|kitchen", "gym": "gym|sport", "scienc": "scienc|lab", "lab": "lab",
       "air": "air|hvac", "tile": "tile", "plaster": "plaster", "wall": "wall", "play": "play",
       "floor": "floor", "window": "window|glass", "roof": "roof", "boiler": "boiler"}
META = ["program_type", "phase", "district", "status", "start_year", "planned_dur_days", "n_phases_bldg"]


def ddc_prices():
    ddc = pd.read_csv(ROOT / "02_数据" / "raw" / "ddc_cwicr_zh" / "DDC_CWICR_ZH_CHINA_Catalog.csv")
    ddc.columns = [c.strip() for c in ddc.columns]
    ddc["price_median"] = pd.to_numeric(ddc["price_median"], errors="coerce")
    txt = (ddc["parent_department"].astype(str) + " " + ddc["parent_section"].astype(str) + " " +
           ddc["name"].astype(str) + " " + ddc["category"].astype(str)).str.lower()
    pr = {}
    for kw in SCOPE_KW:
        m = ddc[txt.str.contains(SYN.get(kw, kw), na=False)]
        pr[kw] = float(m["price_median"].median()) if len(m) and m["price_median"].notna().any() else 0.0
    return pr


def build(df, prices, level):
    d = df[df["cost_spend"] >= MIN_SPEND].copy()
    d["y"] = np.log1p(d["cost_spend"].astype(float))
    desc = d["Project Description"].astype(str).str.lower()
    feats = META[:]
    if level >= 1:
        for kw in SCOPE_KW:
            d[f"scope_{kw}"] = desc.str.contains(kw).astype(int)
            feats.append(f"scope_{kw}")
    if level >= 2:
        idx = np.zeros(len(d)); mx = np.zeros(len(d)); cnt = np.zeros(len(d))
        for kw in SCOPE_KW:
            p = prices.get(kw, 0.0); pres = d[f"scope_{kw}"].values
            idx += pres * p; mx = np.maximum(mx, pres * p); cnt += pres * (1 if p > 0 else 0)
        d["ddc_cost_index"] = idx; d["ddc_price_max"] = mx; d["ddc_n_cat"] = cnt
        feats += ["ddc_cost_index", "ddc_price_max", "ddc_n_cat"]
    for c in ["start_year", "planned_dur_days", "n_phases_bldg"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    X = pd.get_dummies(d[feats], columns=["program_type", "phase", "district", "status"], dummy_na=False)
    return X.astype(float).values, d["y"].values


def met(y_log, pred_log):
    y = np.expm1(y_log); p = np.expm1(pred_log)
    return {"MAPE": mape(y, p), "R2log": r2(y_log, pred_log)}, np.abs(y - p)


def m_cat(s):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=700, learning_rate=0.04, depth=7, l2_leaf_reg=3, random_seed=s, verbose=0)
def m_xgb(s): return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=1, verbosity=0)
def m_ann(s):
    return TransformedTargetRegressor(
        regressor=Pipeline([("sc", StandardScaler()),
                            ("m", MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3,
                                               max_iter=500, early_stopping=True, n_iter_no_change=15, random_state=s))]),
        transformer=StandardScaler())
def m_stack(s):
    est = [("cat", m_cat(s)), ("xgb", m_xgb(s)), ("lgb", LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, random_state=s, verbose=-1)), ("ann", m_ann(s))]
    return StackingRegressor(estimators=est, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=1)


def main():
    df = load_nyc_sca_raw(); prices = ddc_prices()
    print("E5 | 成本知识增强消融阶梯 (DDC 单价 CNY):", {k: round(v,1) for k,v in prices.items() if v>0}, flush=True)
    DATA = {lvl: build(df, prices, lvl) for lvl in [0, 1, 2]}
    print(f"  n={len(DATA[0][1])}  特征数 M0={DATA[0][0].shape[1]} M1={DATA[1][0].shape[1]} M2={DATA[2][0].shape[1]}", flush=True)
    models = [("CatBoost", m_cat), ("Stacking(+ANN) hybrid", m_stack)]
    rows = []
    for mname, mbuild in models:
        rec = {}
        aes = {}
        for lvl in [0, 1, 2]:
            X, y = DATA[lvl]; ms = []; a = []
            for seed in range(N_SEEDS):
                itr, ite = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, random_state=seed)
                mdl = mbuild(seed).fit(X[itr], y[itr]); pred = mdl.predict(X[ite])
                mm, ae = met(y[ite], pred); ms.append(mm); a.append(ae)
            rec[lvl] = (np.mean([m["MAPE"] for m in ms]), np.mean([m["R2log"] for m in ms]))
            aes[lvl] = np.concatenate(a)
        # 显著性: 知识注入 M0->M1, M1->M2
        p01 = wilcoxon_p(aes[1], aes[0]); p12 = wilcoxon_p(aes[2], aes[1])
        rows.append({"Model": mname,
                     "MAPE_M0": rec[0][0], "MAPE_M1": rec[1][0], "MAPE_M2": rec[2][0],
                     "R2_M0": rec[0][1], "R2_M1": rec[1][1], "R2_M2": rec[2][1],
                     "p_M0vsM1": p01, "p_M1vsM2": p12})
        print(f"  {mname:22s} MAPE: M0={rec[0][0]:.1f} -> M1={rec[1][0]:.1f} -> M2={rec[2][0]:.1f} | "
              f"R2log: {rec[0][1]:.3f}->{rec[1][1]:.3f}->{rec[2][1]:.3f} | p(0vs1)={p01:.1e} p(1vs2)={p12:.1e}", flush=True)
    dfr = pd.DataFrame(rows)
    dfr.to_csv(TAB / "e5_ablation.csv", index=False, encoding="utf-8-sig")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(dfr)); w = 0.27; labels = ["M0 元数据", "M1 +工作范围知识", "M2 +DDC单价"]
    for ax, metric, title in [(axes[0], "MAPE", "MAPE (%)"), (axes[1], "R2", "R² (log-space)")]:
        for i, lv in enumerate(["M0", "M1", "M2"]):
            vals = dfr[f"{metric}_{lv}"]
            ax.bar(x + (i-1)*w, vals, w, label=labels[i])
        ax.set_xticks(x); ax.set_xticklabels(dfr["Model"]); ax.set_ylabel(title)
    axes[0].set_title("E5 · Knowledge Ablation — MAPE (↓ better)")
    axes[1].set_title("E5 · Knowledge Ablation — R² (↑ better)")
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "e5_ddc_ablation.png", dpi=300)
    print("\n已保存: tables/e5_ablation.csv, figures/e5_ddc_ablation.png", flush=True)


if __name__ == "__main__":
    main()
