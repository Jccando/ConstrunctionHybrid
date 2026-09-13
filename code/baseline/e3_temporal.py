"""E3 - temporal generalization validation (NYC SCA): train on earlier projects -> test on recent projects,
verifying the model's temporal robustness.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.metrics import mape, r2
from utils.datasets import load_nyc_sca_raw, ROOT
TAB = ROOT / "code" / "experimentresult" / "tables"; FIG = ROOT / "code" / "experimentresult" / "figures"
SCOPE_KW = ["boiler", "roof", "window", "electr", "ventil", "floor", "ceil", "abate",
            "parapet", "mason", "heat", "cool", "fire", "alarm", "bath", "cafet",
            "gym", "scienc", "lab", "air", "tile", "plaster", "wall", "play"]


def main():
    df = load_nyc_sca_raw()
    d = df[df["cost_spend"] >= 10000].copy()
    d["y"] = np.log1p(d["cost_spend"].astype(float))
    d["start_year"] = pd.to_numeric(d["start_year"], errors="coerce")
    d = d[d["start_year"].notna()].copy()
    desc = d["Project Description"].astype(str).str.lower()
    for kw in SCOPE_KW:
        d[f"scope_{kw}"] = desc.str.contains(kw).astype(int)
    for c in ["start_year", "planned_dur_days", "n_phases_bldg"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    feats = ["program_type", "phase", "district", "status", "start_year", "planned_dur_days", "n_phases_bldg"] + [f"scope_{k}" for k in SCOPE_KW]
    X = pd.get_dummies(d[feats], columns=["program_type", "phase", "district", "status"], dummy_na=False).astype(float)
    y = d["y"].values
    yr = d["start_year"].values
    cut = np.quantile(yr, 0.75)  # train on first 75% of years, test on last 25% (recent)
    tr = yr <= cut; te = yr > cut
    # align columns
    Xtr, Xte = X.loc[tr].values, X.loc[te].values
    # ensure the test column space is consistent (same DataFrame, columns already aligned)
    print(f"E3 | temporal generalization  train<= {cut:.0f}yr n={tr.sum()} | test>{cut:.0f}yr n={te.sum()}", flush=True)

    def m_xgb(s): return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=1, verbosity=0)
    def m_cat(s):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=700, learning_rate=0.04, depth=7, l2_leaf_reg=3, random_seed=s, verbose=0)
    def m_ann(s):
        return TransformedTargetRegressor(regressor=Pipeline([("sc", StandardScaler()), ("m", MLPRegressor(hidden_layer_sizes=(128,64), alpha=1e-2, learning_rate_init=1e-3, max_iter=500, early_stopping=True, n_iter_no_change=15, random_state=s))]), transformer=StandardScaler())
    def m_stack(s):
        return StackingRegressor(estimators=[("cat", m_cat(s)), ("xgb", m_xgb(s)), ("lgb", LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, random_state=s, verbose=-1)), ("ann", m_ann(s))], final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=1)

    ytr, yte = y[tr], y[te]
    rows = []
    for name, mb in [("CatBoost", m_cat), ("Stacking hybrid", m_stack)]:
        m = mb(0).fit(Xtr, ytr); pred = m.predict(Xte)
        ma = mape(np.expm1(yte), np.expm1(pred)); rr = r2(yte, pred)
        rows.append({"Model": name, "MAPE_future": ma, "R2log_future": rr})
        print(f"  {name:14s} recent-year test: MAPE={ma:.2f}%  R2log={rr:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(TAB / "e3_temporal.csv", index=False, encoding="utf-8-sig")
    print("\nsaved: tables/e3_temporal.csv", flush=True)


if __name__ == "__main__":
    main()
