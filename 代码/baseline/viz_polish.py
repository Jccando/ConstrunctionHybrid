"""可视化改版(顶刊审美): 读 e1/e2/e6 结果 CSV, 重绘 MAPE 对比图。
- 统一配色(despine, 无 chartjunk, 误差棒, 数值标签, 类型着色)
- 3 数据集组合面板(论文主图候选)
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from utils.datasets import ROOT
TAB = ROOT / "04_实验结果" / "tables"; FIG = ROOT / "04_实验结果" / "figures"

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans",
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--"})
C_HYBRID, C_NEW, C_TRAD = "#1f6fb4", "#2ca02c", "#b0b7bf"


def load(df, mape_col):
    df = df.copy()
    df["Model"] = df["Model"].astype(str).str.replace("(新)", "", regex=False).str.strip()
    df["mape"] = pd.to_numeric(df[mape_col], errors="coerce")
    # 解析 std
    if "MAPE_str" in df.columns:
        df["std"] = df["MAPE_str"].astype(str).str.split("±").str[1]
        df["std"] = pd.to_numeric(df["std"], errors="coerce").fillna(0)
    else:
        df["std"] = 0
    df["Type"] = df["Type"].fillna("Single")
    return df.sort_values("mape").reset_index(drop=True)


def color_for(row):
    if row["Type"] == "Hybrid": return C_HYBRID
    if "新" in str(row.get("_orig", "")): return C_NEW
    return C_TRAD


def draw(ax, df, title, ylabel=True):
    colors = [C_HYBRID if t == "Hybrid" else C_NEW if "(新)" in str(m) else C_TRAD
              for t, m in zip(df["Type"], df["_orig"])]
    bars = ax.bar(df["Model"], df["mape"], yerr=df["std"], capsize=3, color=colors,
                  edgecolor="black", linewidth=0.5, error_kw={"lw": 1})
    ax.set_title(title)
    if ylabel: ax.set_ylabel("MAPE (%)  ↓ better")
    ax.set_ylim(0, (df["mape"] + df["std"]).max() * 1.22)
    for i, (v, s) in enumerate(zip(df["mape"], df["std"])):
        ax.text(i, v + s + (df["mape"].max() * 0.01), f"{v:.1f}", ha="center", fontsize=8.5)
    ax.tick_params(axis="x", rotation=22)
    for l in ax.get_xticklabels(): l.set_ha("right")


def main():
    e1 = load(pd.read_csv(TAB / "e1_results.csv"), "MAPE_mean"); e1["_orig"] = pd.read_csv(TAB / "e1_results.csv")["Model"]
    e2 = load(pd.read_csv(TAB / "e2_results.csv"), "MAPE_mean"); e2["_orig"] = pd.read_csv(TAB / "e2_results.csv")["Model"]
    e6 = load(pd.read_csv(TAB / "e6_results.csv"), "MAPE"); e6["_orig"] = pd.read_csv(TAB / "e6_results.csv")["Model"]

    # 组合 3 面板主图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    draw(axes[0], e1, "(a) UCI #437 — residential\n(method validation)")
    draw(axes[1], e2, "(b) NYC SCA — public school\n(real cost, phase-level)")
    draw(axes[2], e6, "(c) ComStock — public buildings\n(design-stage, with sqft)")
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=C_TRAD, edgecolor="black", label="Traditional single"),
           Patch(facecolor=C_NEW, edgecolor="black", label="Newer single (CatBoost/NGBoost/GPR)"),
           Patch(facecolor=C_HYBRID, edgecolor="black", label="Hybrid learning (proposed)")]
    axes[0].legend(handles=leg, loc="upper left", fontsize=8.5, frameon=False)
    fig.suptitle("Cost estimation accuracy across three datasets — hybrid learning vs baselines", fontsize=14, y=1.02)
    plt.tight_layout(); plt.savefig(FIG / "main_comparison_3panel.png", dpi=300, bbox_inches="tight"); plt.close()

    # 单独精修版
    for df, name, ttl in [(e1, "e1", "E1 · UCI #437 construction cost (MAPE)"),
                          (e2, "e2", "E2 · NYC SCA public school cost (MAPE)"),
                          (e6, "e6", "E6 · ComStock public building cost (MAPE)")]:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        draw(ax, df, ttl); plt.tight_layout(); plt.savefig(FIG / f"{name}_mape_pub.png", dpi=300, bbox_inches="tight"); plt.close()
    print("已生成: main_comparison_3panel.png, e1/e2/e6_mape_pub.png", flush=True)


if __name__ == "__main__":
    main()
