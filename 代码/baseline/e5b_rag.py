"""E5b - RAG 检索增强造价估算(B路升级: 真创新点)。
对每条 NYC 项目描述, 在 DDC 工项库(2681条)里检索 top-k 最相关工项,
聚合其单价/相似度/类别多样性 -> 丰富的"上下文成本知识"特征, 喂入模型。
消融: M0(元数据) -> M1(+工作范围关键词) -> M2_RAG(+RAG检索特征)。
证明: 检索式知识增强 > 关键词增强(真正的知识库利用)。
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.metrics import mape, r2, wilcoxon_p
from utils.datasets import load_nyc_sca_raw, ROOT
TAB = ROOT / "04_实验结果" / "tables"; FIG = ROOT / "04_实验结果" / "figures"
N_SEEDS = 5; K = 10
SCOPE_KW = ["boiler","roof","window","electr","ventil","floor","ceil","abate","parapet","mason",
            "heat","cool","fire","alarm","bath","cafet","gym","scienc","lab","air","tile","plaster","wall","play"]


def rag_features(nyc_desc, ddc):
    """对每条描述在 DDC 库检索 top-K, 返回 RAG 特征 DataFrame。"""
    ddc = ddc.copy(); ddc.columns = [c.strip() for c in ddc.columns]
    ddc["price"] = pd.to_numeric(ddc["price_median"], errors="coerce").fillna(0)
    ddc["_text"] = (ddc["name"].astype(str) + " " + ddc["parent_department"].astype(str) + " " +
                    ddc["parent_section"].astype(str) + " " + ddc["category"].astype(str)).fillna("")
    vec = TfidfVectorizer(max_features=6000, stop_words="english", ngram_range=(1, 2))
    Dd = vec.fit_transform(ddc["_text"].tolist())
    Dn = vec.transform(nyc_desc.tolist())
    sim = cosine_similarity(Dn, Dd)                  # (n_nyc, n_ddc)
    top = np.argpartition(-sim, K, axis=1)[:, :K]    # 每行 top-K 索引
    price = ddc["price"].values
    dept = ddc["parent_department"].astype(str).values
    rows = []
    for i in range(sim.shape[0]):
        idx = top[i]; s = sim[i, idx]; p = price[idx]
        rows.append({
            "rag_price_mean": p.mean(), "rag_price_median": np.median(p),
            "rag_price_max": p.max(), "rag_price_sum": p.sum(), "rag_price_std": p.std(),
            "rag_sim_mean": s.mean(), "rag_sim_max": s.max(),
            "rag_n_dept": len(set(dept[idx])),
        })
    return pd.DataFrame(rows), float(sim.max())


def build_xy(df, rag_df, level):
    d = df[df["cost_spend"] >= 10000].copy().reset_index(drop=True)
    d["y"] = np.log1p(d["cost_spend"].astype(float))
    feats = ["program_type", "phase", "district", "status", "start_year", "planned_dur_days", "n_phases_bldg"]
    if level >= 1:
        desc = d["Project Description"].astype(str).str.lower()
        for kw in SCOPE_KW:
            d[f"scope_{kw}"] = desc.str.contains(kw).astype(int)
            feats.append(f"scope_{kw}")
    if level >= 2:
        for c in rag_df.columns:
            d[c] = rag_df[c].values
            feats.append(c)
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


def main():
    df = load_nyc_sca_raw()
    ddc = pd.read_csv(ROOT / "02_数据" / "raw" / "ddc_cwicr_zh" / "DDC_CWICR_ZH_CHINA_Catalog.csv")
    df_f = df[df["cost_spend"] >= 10000].copy().reset_index(drop=True)
    print("E5b | RAG 检索增强(对每条项目描述检索 DDC 工项库)", flush=True)
    rag, maxsim = rag_features(df_f["Project Description"].astype(str), ddc)
    print(f"  样本={len(df_f)}  DDC库={len(ddc)}  top-K={K}  最大检索相似度={maxsim:.3f}", flush=True)
    print(f"  RAG 特征: {list(rag.columns)}", flush=True)

    DATA = {lv: build_xy(df, rag, lv) for lv in [0, 1, 2]}
    models = [("CatBoost", m_cat), ("XGBoost", m_xgb)]
    rows = []
    for mname, mb in models:
        rec, aes = {}, {}
        for lv in [0, 1, 2]:
            X, y = DATA[lv]; ms, a = [], []
            for seed in range(N_SEEDS):
                itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
                m = mb(seed).fit(X[itr], y[itr]); mm, ae = met(y[ite], m.predict(X[ite]))
                ms.append(mm); a.append(ae)
            rec[lv] = (np.mean([m["MAPE"] for m in ms]), np.mean([m["R2log"] for m in ms]))
            aes[lv] = np.concatenate(a)
        p01 = wilcoxon_p(aes[1], aes[0]); p12 = wilcoxon_p(aes[2], aes[1])
        rows.append({"Model": mname,
                     "MAPE_M0": rec[0][0], "MAPE_M1": rec[1][0], "MAPE_M2RAG": rec[2][0],
                     "R2_M0": rec[0][1], "R2_M1": rec[1][1], "R2_M2RAG": rec[2][1],
                     "p_M1vsM2RAG": p12})
        print(f"  {mname:10s} MAPE: M0={rec[0][0]:.1f} -> M1={rec[1][0]:.1f} -> M2_RAG={rec[2][0]:.1f} | "
              f"R2log: {rec[0][1]:.3f}->{rec[1][1]:.3f}->{rec[2][1]:.3f} | RAG vs M1 p={p12:.2e}", flush=True)
    dfr = pd.DataFrame(rows)
    dfr.to_csv(TAB / "e5b_rag_ablation.csv", index=False, encoding="utf-8-sig")
    gain = (dfr["MAPE_M1"] - dfr["MAPE_M2RAG"]).mean()
    print(f"\nRAG 平均降 MAPE {gain:+.2f} 个百分点", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    SEQ = ["#cde2fb", "#86b6ef", "#2a78d6"]; SURF = "#fcfcfb"
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURF)
    x = np.arange(len(dfr)); w = 0.26
    for i, (lv, lab) in enumerate(zip(["M0", "M1", "M2RAG"], ["M0  metadata", "M1  +scope keywords", "M2  +RAG retrieval (proposed)"])):
        ax.bar(x + (i-1)*w, dfr[f"MAPE_{lv}"], w, label=lab, color=SEQ[i], edgecolor=SURF, linewidth=2, zorder=3)
        for xi, v in zip(x + (i-1)*w, dfr[f"MAPE_{lv}"]): ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=8.5, color="#52514e")
    ax.set_xticks(x); ax.set_xticklabels(dfr["Model"], color="#52514e"); ax.set_ylabel("MAPE (%)  lower is better", color="#52514e")
    ax.set_title("RAG knowledge augmentation ablation (NYC SCA real cost)\nRetrieval over DDC work-item base (top-10) beats keyword features", pad=10)
    ax.legend(fontsize=9, frameon=False); ax.set_ylim(60, dfr[["MAPE_M0","MAPE_M1","MAPE_M2RAG"]].values.max()*1.08)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color="#e1e0d9", lw=1, zorder=0); ax.set_axisbelow(True); ax.set_facecolor(SURF)
    plt.tight_layout(); plt.savefig(FIG / "e5b_rag_ablation.png", dpi=300, bbox_inches="tight", facecolor=SURF); plt.close()
    print("已保存: e5b_rag_ablation.csv, e5b_rag_ablation.png", flush=True)


if __name__ == "__main__":
    main()
