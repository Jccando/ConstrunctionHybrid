"""E1 - UCI #437 方法有效性验证（v3，扩充方法集）。
单一(网格SVR/RF/LightGBM/XGBoost/CatBoost/ANN/NGBoost/GPR) vs 混合(Stacking/PSO-SVR/GA-ANN)。
目标 V-10 实际建造成本。5 种子 -> mean±std + pooled Wilcoxon + 概率区间覆盖(NGBoost/GPR)。
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
# catboost / ngboost 改为函数内懒加载：避免 joblib worker 子进程 re-import 时触发 gmpy2 DLL 崩溃

sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.metrics import eval_metrics, mape, wilcoxon_p
from utils.datasets import load_uci
from utils.metaheuristics import pso, ga

ROOT = Path(__file__).resolve().parents[2]  # 项目根(03_代码/baseline -> parents[2])
TAB = ROOT / "04_实验结果" / "tables"; FIG = ROOT / "04_实验结果" / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
N_SEEDS = 5; TEST_SIZE = 0.2; Z90 = 1.645


def _met(yte, pred):
    return eval_metrics(yte, pred), np.abs(yte - pred)


def run_svr_grid(Xtr, Xte, ytr, yte, seed):
    pipe = Pipeline([("sc", StandardScaler()), ("m", SVR())])
    gs = GridSearchCV(pipe, {"m__C": [0.1, 1, 10, 100], "m__gamma": ["scale", 1e-3, 1e-2, 1e-1],
                             "m__epsilon": [0.01, 0.1]}, cv=3,
                      scoring="neg_mean_absolute_percentage_error", n_jobs=1).fit(Xtr, ytr)
    pred = gs.predict(Xte); m, ae = _met(yte, pred); return pred, m, ae, {}


def run_rf(Xtr, Xte, ytr, yte, seed):
    m = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=1).fit(Xtr, ytr)
    pred = m.predict(Xte); met, ae = _met(yte, pred); return pred, met, ae, {}


def run_lgb(Xtr, Xte, ytr, yte, seed):
    m = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8,
                      colsample_bytree=0.8, random_state=seed, verbose=-1).fit(Xtr, ytr)
    pred = m.predict(Xte); met, ae = _met(yte, pred); return pred, met, ae, {}


def run_xgb(Xtr, Xte, ytr, yte, seed):
    m = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                     colsample_bytree=0.8, random_state=seed, n_jobs=1, verbosity=0).fit(Xtr, ytr)
    pred = m.predict(Xte); met, ae = _met(yte, pred); return pred, met, ae, {}


def run_cat(Xtr, Xte, ytr, yte, seed):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                          random_seed=seed, verbose=0).fit(Xtr, ytr)
    pred = m.predict(Xte); met, ae = _met(yte, pred); return pred, met, ae, {}


def run_ann(Xtr, Xte, ytr, yte, seed):
    m = Pipeline([("sc", StandardScaler()),
                  ("m", MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, learning_rate_init=1e-3,
                                     max_iter=600, random_state=seed))]).fit(Xtr, ytr)
    pred = m.predict(Xte); met, ae = _met(yte, pred); return pred, met, ae, {}


def run_ngboost(Xtr, Xte, ytr, yte, seed):
    from ngboost import NGBRegressor
    m = NGBRegressor(n_estimators=300, learning_rate=0.02,
                     Base=DecisionTreeRegressor(criterion="friedman_mse", max_depth=3),
                     random_state=seed, verbose=False).fit(Xtr, ytr)
    pred = m.predict(Xte); met, ae = _met(yte, pred)
    cov = {}
    try:
        dist = m.pred_dist(Xte); loc = np.asarray(dist.loc).ravel(); sc = np.asarray(dist.scale).ravel()
        lo, hi = loc - Z90 * sc, loc + Z90 * sc
        cov["cov90"] = float(np.mean((yte >= lo) & (yte <= hi))); cov["mpiw"] = float(np.mean(hi - lo))
    except Exception:
        pass
    return pred, met, ae, cov


def run_gpr(Xtr, Xte, ytr, yte, seed):
    kernel = ConstantKernel(1.0) * RBF() + WhiteKernel()
    m = Pipeline([("sc", StandardScaler()),
                  ("g", GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True,
                                                  n_restarts_optimizer=2, random_state=seed))]).fit(Xtr, ytr)
    pred, std = m.predict(Xte, return_std=True); met, ae = _met(yte, pred)
    lo, hi = pred - Z90 * std, pred + Z90 * std
    cov = {"cov90": float(np.mean((yte >= lo) & (yte <= hi))), "mpiw": float(np.mean(hi - lo))}
    return pred, met, ae, cov


def run_stacking(Xtr, Xte, ytr, yte, seed):
    est = [("xgb", XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                                colsample_bytree=0.8, random_state=seed, n_jobs=1, verbosity=0)),
           ("lgb", LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, random_state=seed, verbose=-1)),
           ("ann", Pipeline([("sc", StandardScaler()),
                             ("m", MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=seed))])),
           ("svr", Pipeline([("sc", StandardScaler()), ("m", SVR(C=10, gamma="scale"))]))]
    m = StackingRegressor(estimators=est, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=1).fit(Xtr, ytr)
    pred = m.predict(Xte); met, ae = _met(yte, pred); return pred, met, ae, {}


def run_pso_svr(Xtr, Xte, ytr, yte, seed):
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    Xa, Xb, ya, yb = train_test_split(Xtr_s, ytr, test_size=0.25, random_state=seed)
    obj = lambda p: mape(yb, SVR(C=10 ** p[0], gamma=10 ** p[1], epsilon=10 ** p[2]).fit(Xa, ya).predict(Xb))
    best, _ = pso(obj, [(0, 3), (-5, 0), (-2, 0)], n_particles=10, iters=16, seed=seed)
    pred = SVR(C=10 ** best[0], gamma=10 ** best[1], epsilon=10 ** best[2]).fit(Xtr_s, ytr).predict(Xte_s)
    met, ae = _met(yte, pred); return pred, met, ae, {}


def run_ga_ann(Xtr, Xte, ytr, yte, seed):
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    Xa, Xb, ya, yb = train_test_split(Xtr_s, ytr, test_size=0.25, random_state=seed)
    def obj(p):
        h = int(np.clip(round(10 ** p[0]), 16, 256))
        return mape(yb, MLPRegressor(hidden_layer_sizes=(h,), alpha=10 ** p[1], learning_rate_init=10 ** p[2],
                                     max_iter=200, random_state=seed).fit(Xa, ya).predict(Xb))
    best, _ = ga(obj, [(1.2, 2.4), (-5, -1), (-4, -2)], pop=10, gens=8, seed=seed)
    h = int(np.clip(round(10 ** best[0]), 16, 256))
    pred = MLPRegressor(hidden_layer_sizes=(h,), alpha=10 ** best[1], learning_rate_init=10 ** best[2],
                        max_iter=450, random_state=seed).fit(Xtr_s, ytr).predict(Xte_s)
    met, ae = _met(yte, pred); return pred, met, ae, {}


MODELS = [
    ("SVR(grid)", run_svr_grid, "Single"), ("RF", run_rf, "Single"),
    ("LightGBM", run_lgb, "Single"), ("XGBoost", run_xgb, "Single"),
    ("CatBoost", run_cat, "Single"), ("ANN", run_ann, "Single"),
    ("NGBoost", run_ngboost, "Single"), ("GPR", run_gpr, "Single"),
    ("Stacking", run_stacking, "Hybrid"), ("PSO-SVR", run_pso_svr, "Hybrid"),
    ("GA-ANN", run_ga_ann, "Hybrid"),
]
SINGLES = {n for n, _, t in MODELS if t == "Single"}
HYBRIDS = {n for n, _, t in MODELS if t == "Hybrid"}
NEWER = {"CatBoost", "NGBoost", "GPR"}  # 较新方法标注


def main():
    d = load_uci(); X, y = d["X"], d["y_cost"]
    print(f"E1(v3) | UCI #437  n={len(y)}  目标=V-10 建造成本  seeds={N_SEEDS}  模型数={len(MODELS)}", flush=True)
    per_model = {n: [] for n, _, _ in MODELS}
    ae_store = {n: [] for n, _, _ in MODELS}
    cov_store = {n: [] for n, _, _ in MODELS}

    for seed in range(N_SEEDS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
        t0 = time.time()
        for name, fn, _ in MODELS:
            _, met, ae, cov = fn(Xtr, Xte, ytr, yte, seed)
            per_model[name].append(met); ae_store[name].append(ae); cov_store[name].append(cov)
        bs = min(SINGLES, key=lambda n: per_model[n][-1]["MAPE"])
        bh = min(HYBRIDS, key=lambda n: per_model[n][-1]["MAPE"])
        print(f"  seed {seed} ({time.time()-t0:.1f}s) | best single={bs} {per_model[bs][-1]['MAPE']:.2f} | "
              f"best hybrid={bh} {per_model[bh][-1]['MAPE']:.2f}", flush=True)

    rows = []
    for name, _, typ in MODELS:
        ma = np.array([m["MAPE"] for m in per_model[name]])
        ra = np.array([m["R2"] for m in per_model[name]])
        rm = np.array([m["RMSE"] for m in per_model[name]])
        tag = ("(新)" if name in NEWER else "")
        rows.append({"Model": name + tag, "Type": typ, "MAPE_mean": ma.mean(), "MAPE_std": ma.std(),
                     "R2_mean": ra.mean(), "RMSE_mean": rm.mean(),
                     "MAPE_str": f"{ma.mean():.2f}±{ma.std():.2f}",
                     "R2_str": f"{ra.mean():.3f}±{ra.std():.3f}",
                     "RMSE_str": f"{rm.mean():.2f}±{rm.std():.2f}"})
    df = pd.DataFrame(rows).sort_values("MAPE_mean").reset_index(drop=True)
    df.to_csv(TAB / "e1_results.csv", index=False, encoding="utf-8-sig")

    pooled = lambda n: np.concatenate(ae_store[n])
    best_single = min(SINGLES, key=lambda n: df.loc[df.Model.str.startswith(n), "MAPE_mean"].iloc[0])
    best_hybrid = min(HYBRIDS, key=lambda n: df.loc[df.Model.str.startswith(n), "MAPE_mean"].iloc[0])
    p_pool = wilcoxon_p(pooled(best_hybrid), pooled(best_single))
    ps_seed = [wilcoxon_p(ae_store[best_hybrid][s], ae_store[best_single][s]) for s in range(N_SEEDS)]
    sig = sum(1 for p in ps_seed if p < 0.05)

    print("\n===== E1(v3) 结果（按 MAPE 升序）=====", flush=True)
    print(df[["Model", "Type", "MAPE_str", "R2_str", "RMSE_str"]].to_string(index=False), flush=True)
    print(f"\n最佳混合={best_hybrid} vs 最佳单一={best_single}: pooled Wilcoxon p={p_pool:.2e} "
          f"(n={len(pooled(best_hybrid))}); 逐种子显著 {sig}/{N_SEEDS} (中位 p={np.median(ps_seed):.2e})", flush=True)
    # 概率区间覆盖（NGBoost/GPR）
    for nm in ["NGBoost", "GPR"]:
        covs = [c for c in cov_store[nm] if c]
        if covs:
            mc = np.mean([c.get("cov90", np.nan) for c in covs]); mw = np.mean([c.get("mpiw", np.nan) for c in covs])
            print(f"  {nm}: 90% 区间覆盖率={mc*100:.1f}%  平均区间宽={mw:.1f}", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order = df.sort_values("MAPE_mean")
    colors = ["#2980b9" if t == "Hybrid" else ("#16a085" if "(新)" in m else "#95a5a6")
              for m, t in zip(order["Model"], order["Type"])]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(order["Model"], order["MAPE_mean"], yerr=order["MAPE_std"], capsize=4,
           color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("MAPE (%)  ↓ better"); ax.set_ylim(0, (order["MAPE_mean"] + order["MAPE_std"]).max() * 1.25)
    ax.set_title("E1 · UCI #437 Construction Cost — MAPE (mean±std, 5 seeds)\nGrey=traditional · Green=newer · Blue=hybrid learning")
    for i, v in enumerate(order["MAPE_mean"]):
        ax.text(i, v + order["MAPE_std"].iloc[i] + 0.15, f"{v:.2f}", ha="center", fontsize=8.5)
    plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(FIG / "e1_mape.png", dpi=300)
    print("\n已保存: tables/e1_results.csv, figures/e1_mape.png", flush=True)


if __name__ == "__main__":
    main()
