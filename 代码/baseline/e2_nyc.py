"""E2 - NYC SCA 公共建筑(学校)真实造价（阶段级）。
Y=阶段实际花费 Total Phase Actual Spending(真实USD, 过滤>=$10k 消除近零行)。
特征: 项目类型/阶段/学区/状态/年份/工期/阶段数 + 工作范围关键词。
单一(RF/LightGBM/XGBoost/CatBoost/ANN/NGBoost) vs 混合(Stacking/GA-ANN)。
ANN 用 TransformedTargetRegressor 标准化 y + early_stopping 防发散。
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
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.metrics import mape, r2, rmse, wilcoxon_p
from utils.datasets import load_nyc_sca_raw
from utils.metaheuristics import ga

ROOT = Path(__file__).resolve().parents[2]  # 项目根
TAB = ROOT / "04_实验结果" / "tables"; FIG = ROOT / "04_实验结果" / "figures"
N_SEEDS = 5; TEST_SIZE = 0.2; MIN_SPEND = 10000
SCOPE_KW = ["boiler", "roof", "window", "electr", "ventil", "floor", "ceil", "abate",
            "parapet", "mason", "heat", "cool", "fire", "alarm", "bath", "cafet",
            "gym", "scienc", "lab", "air", "tile", "plaster", "wall", "play"]


def build_xy(df):
    d = df[df["cost_spend"] >= MIN_SPEND].copy()
    d["y"] = np.log1p(d["cost_spend"].astype(float))
    desc = d["Project Description"].astype(str).str.lower()
    for kw in SCOPE_KW:
        d[f"scope_{kw}"] = desc.str.contains(kw).astype(int)
    for c in ["start_year", "planned_dur_days", "n_phases_bldg"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    feats = ["program_type", "phase", "district", "status", "start_year", "planned_dur_days", "n_phases_bldg"] + [f"scope_{k}" for k in SCOPE_KW]
    X = pd.get_dummies(d[feats], columns=["program_type", "phase", "district", "status"], dummy_na=False)
    return X.astype(float).values, d["y"].values, list(X.columns)


def met_all(y_log, pred_log):
    y = np.expm1(y_log); p = np.expm1(pred_log)
    return {"MAPE": mape(y, p), "R2": r2(y, p), "R2_log": r2(y_log, pred_log), "RMSE": rmse(y, p)}, np.abs(y - p)


def m_rf(s): return RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=s, n_jobs=1)
def m_lgb(s): return LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=s, verbose=-1)
def m_xgb(s): return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=1, verbosity=0)
def m_cat(s):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=700, learning_rate=0.04, depth=7, l2_leaf_reg=3, random_seed=s, verbose=0)
def m_ann(s):
    # 自带 X 标准化 + y 标准化(TransformedTargetRegressor) + early stopping, 防发散
    return TransformedTargetRegressor(
        regressor=Pipeline([("sc", StandardScaler()),
                            ("m", MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3,
                                               max_iter=500, early_stopping=True, n_iter_no_change=15, random_state=s))]),
        transformer=StandardScaler())


def run_single(build, Xtr, Xte, ytr, yte, seed):
    m = build(seed).fit(Xtr, ytr); pred = m.predict(Xte)
    return met_all(yte, pred) + (m,)


def run_ngboost(Xtr, Xte, ytr, yte, seed):
    from ngboost import NGBRegressor
    sc = StandardScaler().fit(Xtr)
    m = NGBRegressor(n_estimators=120, learning_rate=0.04, Base=DecisionTreeRegressor(criterion="friedman_mse", max_depth=4),
                     random_state=seed, verbose=False)
    m.fit(sc.transform(Xtr), ytr); pred = m.predict(sc.transform(Xte))
    return met_all(yte, pred) + (m,)


def run_stacking(Xtr, Xte, ytr, yte, seed):
    est = [("cat", m_cat(seed)), ("lgb", m_lgb(seed)), ("xgb", m_xgb(seed)),
           ("ann", m_ann(seed))]
    st = StackingRegressor(estimators=est, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=1)
    st.fit(Xtr, ytr); pred = st.predict(Xte)
    return met_all(yte, pred) + (st,)


def run_ga_ann(Xtr, Xte, ytr, yte, seed):
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    ym, ys = ytr.mean(), ytr.std(); ytr_s = (ytr - ym) / ys
    idx = np.random.RandomState(seed).choice(len(Xtr), min(1500, len(Xtr)), replace=False)
    mask = np.ones(len(Xtr), bool); mask[idx] = False
    Xa, ya = Xtr_s[idx], ytr_s[idx]; Xb_s = Xtr_s[mask][:400]; yb = ytr[mask][:400]
    def obj(p):
        h = int(np.clip(round(10 ** p[0]), 32, 256))
        mm = MLPRegressor(hidden_layer_sizes=(h,), alpha=10 ** p[1], learning_rate_init=10 ** p[2],
                          max_iter=200, early_stopping=True, n_iter_no_change=12, random_state=seed)
        mm.fit(Xa, ya); return mape(np.expm1(yb), np.expm1(mm.predict(Xb_s) * ys + ym))
    best, _ = ga(obj, [(1.5, 2.4), (-3, -1), (-4, -2)], pop=10, gens=7, seed=seed)
    h = int(np.clip(round(10 ** best[0]), 32, 256))
    m = MLPRegressor(hidden_layer_sizes=(h,), alpha=10 ** best[1], learning_rate_init=10 ** best[2],
                     max_iter=450, early_stopping=True, n_iter_no_change=15, random_state=seed)
    m.fit(Xtr_s, ytr_s); pred = m.predict(Xte_s) * ys + ym
    return met_all(yte, pred) + (m,)


def main():
    df = load_nyc_sca_raw()
    X, y, feat = build_xy(df)
    print(f"E2(v3) | NYC SCA 阶段级真实造价(spend>=$10k)  n={len(y)}  特征={X.shape[1]}  seeds={N_SEEDS}", flush=True)
    print(f"    Y(阶段实际花费) 中位=${np.expm1(np.median(y)):,.0f}", flush=True)
    per, ae_st = {}, {}
    for seed in range(N_SEEDS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
        t0 = time.time()
        runs = [("RF", lambda: run_single(m_rf, Xtr, Xte, ytr, yte, seed)),
                ("LightGBM", lambda: run_single(m_lgb, Xtr, Xte, ytr, yte, seed)),
                ("XGBoost", lambda: run_single(m_xgb, Xtr, Xte, ytr, yte, seed)),
                ("CatBoost", lambda: run_single(m_cat, Xtr, Xte, ytr, yte, seed)),
                ("ANN", lambda: run_single(m_ann, Xtr, Xte, ytr, yte, seed)),
                ("NGBoost", lambda: run_ngboost(Xtr, Xte, ytr, yte, seed)),
                ("Stacking", lambda: run_stacking(Xtr, Xte, ytr, yte, seed)),
                ("GA-ANN", lambda: run_ga_ann(Xtr, Xte, ytr, yte, seed))]
        for name, fn in runs:
            met, ae, _ = fn(); per.setdefault(name, []).append(met); ae_st.setdefault(name, []).append(ae)
        bs = min(["RF","LightGBM","XGBoost","CatBoost","ANN","NGBoost"], key=lambda n: per[n][-1]["MAPE"])
        bh = min(["Stacking","GA-ANN"], key=lambda n: per[n][-1]["MAPE"])
        print(f"  seed {seed} ({time.time()-t0:.1f}s) | best single={bs} {per[bs][-1]['MAPE']:.1f}% (R2log={per[bs][-1]['R2_log']:.2f}) | "
              f"best hybrid={bh} {per[bh][-1]['MAPE']:.1f}% (R2log={per[bh][-1]['R2_log']:.2f})", flush=True)

    SINGLES = {"RF","LightGBM","XGBoost","CatBoost","ANN","NGBoost"}; HYBRIDS = {"Stacking","GA-ANN"}; NEWER={"CatBoost","NGBoost"}
    rows = []
    for name in ["RF","LightGBM","XGBoost","CatBoost","ANN","NGBoost","Stacking","GA-ANN"]:
        ma=np.array([m["MAPE"] for m in per[name]]); ra=np.array([m["R2_log"] for m in per[name]])
        tag="(新)" if name in NEWER else ""
        rows.append({"Model":name+tag,"Type":"Hybrid" if name in HYBRIDS else "Single",
                     "MAPE_mean":ma.mean(),"MAPE_str":f"{ma.mean():.2f}±{ma.std():.2f}",
                     "R2log_str":f"{ra.mean():.3f}±{ra.std():.3f}"})
    dfr=pd.DataFrame(rows).sort_values("MAPE_mean").reset_index(drop=True)
    dfr.to_csv(TAB/"e2_results.csv",index=False,encoding="utf-8-sig")
    pooled=lambda n: np.concatenate(ae_st[n])
    bs=min(SINGLES,key=lambda n:dfr.loc[dfr.Model.str.startswith(n),"MAPE_mean"].iloc[0])
    bh=min(HYBRIDS,key=lambda n:dfr.loc[dfr.Model.str.startswith(n),"MAPE_mean"].iloc[0])
    p_pool=wilcoxon_p(pooled(bh),pooled(bs))
    print("\n===== E2(v3) 结果（按 MAPE 升序）=====",flush=True)
    print(dfr[["Model","Type","MAPE_str","R2log_str"]].to_string(index=False),flush=True)
    print(f"\n最佳混合={bh} vs 最佳单一={bs}: pooled Wilcoxon p={p_pool:.2e} (n={len(pooled(bh))})",flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order=dfr.sort_values("MAPE_mean")
    colors=["#2980b9" if t=="Hybrid" else ("#16a085" if "(新)" in m else "#95a5a6") for m,t in zip(order["Model"],order["Type"])]
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.bar(order["Model"],order["MAPE_mean"],color=colors,edgecolor="black",linewidth=0.6)
    ax.set_ylabel("MAPE (%)  ↓ better"); ax.set_ylim(0,order["MAPE_mean"].max()*1.2)
    ax.set_title("E2 · NYC SCA Public School Construction Cost (real USD, phase-level)\nGrey=traditional · Green=newer · Blue=hybrid learning")
    for i,v in enumerate(order["MAPE_mean"]): ax.text(i,v+1,f"{v:.0f}",ha="center",fontsize=9)
    plt.xticks(rotation=20,ha="right"); plt.tight_layout(); plt.savefig(FIG/"e2_mape.png",dpi=300)
    print("\n已保存: tables/e2_results.csv, figures/e2_mape.png",flush=True)


if __name__=="__main__":
    main()
