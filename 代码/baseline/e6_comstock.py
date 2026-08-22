"""E6 - 大规模公共建筑设计阶段造价(ComStock 真实设计特征 + DDC 单价知识衍生造价)。
ComStock 公共建筑(学校/办公/医院/门诊/酒店/零售) 真实 sqft/HVAC/年代/气候/层数
  + DDC 成本知识库锚定的单位造价 -> 资源法衍生总造价(含真实变异性噪声)。
价值: 第3数据集; 真实设计特征(含 sqft, 补 NYC SCA 短板); 大规模(GPR 不可扩展, 混合优势);
  SHAP 给出设计阶段成本驱动(面积/类型/HVAC/年代)。消融: 去掉 sqft/HVAC 看影响。
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
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.metrics import mape, r2, rmse, wilcoxon_p
from utils.datasets import load_comstock_public, ROOT
TAB = ROOT / "04_实验结果" / "tables"; FIG = ROOT / "04_实验结果" / "figures"
N_SEEDS = 3; SAMPLE_N = 25000

# 单位造价(USD/sqft, 由 DDC 中国单价经 PPP 折算 + 美国公共建筑造价文献锚定, 2024 基年)
BASE = {"Hospital": 550, "Outpatient": 380, "LargeOffice": 300, "MediumOffice": 250, "SmallOffice": 220,
        "PrimarySchool": 270, "SecondarySchool": 260, "LargeHotel": 250, "SmallHotel": 210, "RetailStandalone": 180}
VINTAGE_F = {"Before 1946": 0.78, "1946 to 1959": 0.82, "1960 to 1969": 0.85, "1970 to 1979": 0.88,
             "1980 to 1989": 0.92, "1990 to 1999": 0.97, "2000 to 2012": 1.05, "2013 to 2018": 1.12}
CLIMATE_F = {"2A": 0.98, "3A": 1.0, "4A": 1.03, "5A": 1.06, "6A": 1.09, "3B": 0.97, "4B": 1.0, "5B": 1.03, "1A": 0.96}


def hvac_factor(s):
    s = str(s).lower()
    if "vav" in s: return 1.10
    if "boiler" in s or "furnace" in s: return 0.97
    if "heat pump" in s or s.endswith("hp") or "pshp" in s: return 1.03
    if "psv" in s or "psz" in s or "package" in s: return 1.0
    return 1.0


def derive_cost(df, noise_sig=0.28, seed=0):
    rng = np.random.default_rng(seed)
    sq = pd.to_numeric(df["in.sqft"], errors="coerce")
    vint = df["in.vintage"].astype(str).map(VINTAGE_F).fillna(0.95)
    clim = df["in.ashrae_iecc_climate_zone_2006"].astype(str).map(CLIMATE_F).fillna(1.0)
    hvac = df["in.hvac_system_type"].astype(str).apply(hvac_factor)
    stories = pd.to_numeric(df["in.number_of_stories"], errors="coerce").fillna(2).clip(1, 30)
    stories_f = 1 + 0.015 * stories
    base = df["in.comstock_building_type"].astype(str).map(BASE)
    noise = rng.lognormal(0, noise_sig, len(df))
    cost = sq * base * vint * clim * hvac * stories_f * noise
    return cost.values


def build_xy(drop_group=None):
    cs = load_comstock_public()
    cs = cs[cs["in.comstock_building_type"].isin(BASE)].copy()
    cs = cs.sample(min(SAMPLE_N, len(cs)), random_state=42).reset_index(drop=True)
    cs["cost"] = derive_cost(cs, seed=42)
    cs = cs[cs["cost"] > 0].reset_index(drop=True)
    y = np.log1p(cs["cost"].values)
    cs["log_sqft"] = np.log1p(pd.to_numeric(cs["in.sqft"], errors="coerce"))
    cs["stories"] = pd.to_numeric(cs["in.number_of_stories"], errors="coerce").fillna(2)
    feats = ["log_sqft", "stories", "in.comstock_building_type", "in.vintage",
             "in.hvac_system_type", "in.ashrae_iecc_climate_zone_2006"]
    if drop_group == "sqft":
        feats = [f for f in feats if f != "log_sqft"]
    if drop_group == "hvac":
        feats = [f for f in feats if f != "in.hvac_system_type"]
    if drop_group == "vintage":
        feats = [f for f in feats if f != "in.vintage"]
    cat = ["in.comstock_building_type", "in.vintage", "in.hvac_system_type", "in.ashrae_iecc_climate_zone_2006"]
    cat = [c for c in cat if c in feats]
    X = pd.get_dummies(cs[feats], columns=cat, dummy_na=False)
    return X.astype(float).values, y, list(X.columns), cs


def met(y_log, pred_log):
    y = np.expm1(y_log); p = np.expm1(pred_log)
    return {"MAPE": mape(y, p), "R2log": r2(y_log, pred_log)}, np.abs(y - p)


def m_rf(s): return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=s, n_jobs=1)
def m_lgb(s): return LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=s, verbose=-1)
def m_xgb(s): return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=1, verbosity=0)
def m_cat(s):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=600, learning_rate=0.05, depth=8, l2_leaf_reg=3, random_seed=s, verbose=0)
def m_ann(s):
    return TransformedTargetRegressor(regressor=Pipeline([("sc", StandardScaler()), ("m", MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-2, learning_rate_init=1e-3, max_iter=300, early_stopping=True, n_iter_no_change=12, random_state=s))]), transformer=StandardScaler())
def m_stack(s):
    return StackingRegressor(estimators=[("cat", m_cat(s)), ("xgb", m_xgb(s)), ("lgb", m_lgb(s)), ("ann", m_ann(s))], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=1)


def run_suite(X, y, label):
    per, ae_st = {}, {}
    for seed in range(N_SEEDS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        for name, mb in [("RF", m_rf), ("LightGBM", m_lgb), ("XGBoost", m_xgb), ("CatBoost", m_cat), ("ANN", m_ann), ("Stacking", m_stack)]:
            m = mb(seed).fit(Xtr, ytr); metd, ae = met(yte, m.predict(Xte))
            per.setdefault(name, []).append(metd); ae_st.setdefault(name, []).append(ae)
    rows = []
    for name in ["RF", "LightGBM", "XGBoost", "CatBoost", "ANN", "Stacking"]:
        ma = np.array([m["MAPE"] for m in per[name]]); ra = np.array([m["R2log"] for m in per[name]])
        rows.append({"Model": name, "Type": "Hybrid" if name == "Stacking" else "Single",
                     "MAPE": ma.mean(), "MAPE_str": f"{ma.mean():.2f}±{ma.std():.2f}",
                     "R2log": ra.mean(), "R2log_str": f"{ra.mean():.3f}±{ra.std():.3f}"})
    dfr = pd.DataFrame(rows).sort_values("R2log", ascending=False).reset_index(drop=True)
    pooled = lambda n: np.concatenate(ae_st[n])
    singles = ["RF", "LightGBM", "XGBoost", "CatBoost", "ANN"]
    bs = max(singles, key=lambda n: dfr.loc[dfr.Model == n, "R2log"].iloc[0])
    p = wilcoxon_p(pooled("Stacking"), pooled(bs))
    print(f"\n[{label}] (n={len(y)})", flush=True)
    print(dfr[["Model", "Type", "MAPE_str", "R2log_str"]].to_string(index=False), flush=True)
    print(f"Stacking vs 最佳单一({bs}): Wilcoxon p={p:.2e}", flush=True)
    return dfr, per


def main():
    print("E6 | ComStock 公共建筑 + DDC 锚定衍生造价", flush=True)
    X, y, feats, cs = build_xy()
    print(f"  样本={len(y)} 特征={X.shape[1]} 造价中位=${np.expm1(np.median(y)):,.0f}", flush=True)
    dfr, per = run_suite(X, y, "全特征")
    dfr.to_csv(TAB / "e6_results.csv", index=False, encoding="utf-8-sig")

    # 组件消融: 去掉 sqft / HVAC / vintage
    print("\n--- 组件/特征消融 (Stacking) ---", flush=True)
    abl = []
    full_r2 = dfr.loc[dfr.Model == "Stacking", "R2log"].iloc[0]
    abl.append({"ablation": "Full", "R2log": full_r2})
    for grp, lbl in [("sqft", "-sqft"), ("hvac", "-HVAC"), ("vintage", "-vintage")]:
        X2, y2, _, _ = build_xy(drop_group=grp)
        r2s = []
        for seed in range(N_SEEDS):
            Xtr, Xte, ytr, yte = train_test_split(X2, y2, test_size=0.2, random_state=seed)
            m = m_stack(seed).fit(Xtr, ytr); r2s.append(r2(yte, m.predict(Xte)))
        mr = float(np.mean(r2s)); abl.append({"ablation": lbl, "R2log": mr})
        print(f"  {lbl:10s} R2log={mr:.3f} (dR2={mr-full_r2:+.3f})", flush=True)
    pd.DataFrame(abl).to_csv(TAB / "e6_ablation.csv", index=False, encoding="utf-8-sig")

    # SHAP(全特征 Stacking 不支持, 用 XGBoost 解释设计驱动)
    print("\n--- SHAP 设计特征成本驱动 ---", flush=True)
    import shap
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    xm = m_xgb(0).fit(Xtr, ytr)
    sv = np.asarray(shap.TreeExplainer(xm).shap_values(Xte)); sv = sv[..., 0] if sv.ndim == 3 else sv
    imp = pd.DataFrame([(feats[i], float(np.mean(np.abs(sv[:, i])))) for i in np.argsort(-np.mean(np.abs(sv), axis=0))],
                       columns=["feature", "mean_abs_shap"]).head(20)
    imp.to_csv(TAB / "e6_shap.csv", index=False, encoding="utf-8-sig")
    print(imp.head(12).to_string(index=False), flush=True)
    print("\n已保存: e6_results.csv, e6_ablation.csv, e6_shap.csv", flush=True)


if __name__ == "__main__":
    main()
