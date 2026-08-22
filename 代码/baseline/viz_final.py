"""顶刊级可视化(遵循 dataviz 技能: 验证配色 + mark 规范 + 突出故事系列)。
重做主图: 3数据集MAPE主图 / 跨数据集热力图 / CD图 / 知识消融 / 组件消融 / SHAP驱动。
配色: hybrid=blue(故事) · newer=orange · traditional=gray(recessive) · 顺序蓝 ramp · blue<->red发散。
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
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans", "axes.titleweight": "bold",
                     "figure.dpi": 130, "savefig.dpi": 300, "axes.unicode_minus": False})

# ---- 验证过的配色(dataviz skill) ----
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c8c6be"   # hybrid / newer / traditional(recessive)
SEQ3 = ["#cde2fb", "#86b6ef", "#2a78d6"]               # M0->M2 顺序(知识递增)
BLUE_RAMP = ["#e8f1fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#104281"]


def style(ax, ygrid=True):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.set_axisbelow(True)
    if ygrid: ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_facecolor(SURF)


def classify(model, typecol):
    if typecol == "Hybrid" or model in {"Stacking", "GA-ANN", "PSO-SVR"}: return BLUE
    if "(新)" in model or model in {"CatBoost", "NGBoost", "GPR"}: return ORANGE
    return GRAY


def load_bar(csv, mape_col):
    df = pd.read_csv(TAB / csv)
    df["Model"] = df["Model"].astype(str)
    df["mape"] = pd.to_numeric(df[mape_col], errors="coerce")
    df["color"] = [classify(str(m).replace("(新)", "").strip(), t) for m, t in zip(df["Model"], df.get("Type", "Single"))]
    df["model_short"] = df["Model"].str.replace("(新)", "*", regex=False).str.strip()
    return df.sort_values("mape").reset_index(drop=True)


# ============ Fig 1: 3-panel 主对比图 ============
def fig_main():
    dsets = [("e1_results.csv", "MAPE_mean", "(a) UCI #437 · residential\n(method validation, n=372)"),
             ("e2_results.csv", "MAPE_mean", "(b) NYC SCA · public school\n(real cost, phase-level, n=6,783)"),
             ("e6_results.csv", "MAPE", "(c) ComStock · public buildings\n(design-stage w/ sqft, n=25,000)")]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), facecolor=SURF)
    for ax, (csv, col, title) in zip(axes, dsets):
        df = load_bar(csv, col)
        bars = ax.bar(df["model_short"], df["mape"], color=df["color"], edgecolor=SURF, linewidth=2.2, width=0.72, zorder=3)
        ax.set_title(title, color=INK, pad=10)
        ax.set_ylim(0, (df["mape"].max()) * 1.18)
        style(ax)
        ax.set_ylabel("MAPE (%)  lower is better", color=INK2) if ax is axes[0] else ax.set_ylabel("")
        # 仅标注最低(hybrid 故事)与最高
        i_best = df["mape"].idxmin()
        ax.text(i_best, df["mape"].iloc[i_best] + df["mape"].max() * 0.02, f"{df['mape'].iloc[i_best]:.1f}",
                ha="center", fontsize=9, color=BLUE, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        for l in ax.get_xticklabels(): l.set_ha("right"); l.set_color(INK2)
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=BLUE, edgecolor=SURF, label="Hybrid learning (proposed)"),
           Patch(facecolor=ORANGE, edgecolor=SURF, label="Newer single (CatBoost/NGBoost/GPR)"),
           Patch(facecolor=GRAY, edgecolor=SURF, label="Traditional single")]
    axes[0].legend(handles=leg, loc="upper left", fontsize=8.5, frameon=False)
    fig.suptitle("Cost-estimation accuracy across three datasets — hybrid learning vs baselines",
                 fontsize=13.5, color=INK, y=1.01)
    plt.tight_layout(); plt.savefig(FIG / "main_comparison_3panel.png", dpi=300, bbox_inches="tight", facecolor=SURF); plt.close()


# ============ Fig 2: 跨数据集热力图 ============
def fig_heatmap():
    perf = pd.read_csv(TAB / "cross_dataset_perf.csv", index_col=0)
    perf = perf.reindex(["Stacking", "CatBoost", "LightGBM", "XGBoost", "ANN", "RF"])
    cmap = LinearSegmentedColormap.from_list("blu", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURF)
    im = ax.imshow(perf.values, cmap=cmap, aspect="auto", vmin=0.55, vmax=1.0)
    ax.set_xticks(range(perf.shape[1])); ax.set_xticklabels(perf.columns, color=INK2, rotation=8)
    ax.set_yticks(range(perf.shape[0])); ax.set_yticklabels(perf.index, color=INK2)
    for i in range(perf.shape[0]):
        for j in range(perf.shape[1]):
            v = perf.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 0.85 else INK)
    ax.set_title("Cross-dataset $R^2$ — models (rows) × datasets (cols)", color=INK, pad=10)
    for s in ["top", "right", "left", "bottom"]: ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    plt.tight_layout(); plt.savefig(FIG / "cross_dataset_heatmap.png", dpi=300, bbox_inches="tight", facecolor=SURF); plt.close()


# ============ Fig 3: CD 图 ============
def fig_cd():
    ranks = pd.read_csv(TAB / "cross_dataset_ranks.csv", index_col=0).iloc[:, 0].sort_values()
    k, N = 6, 3; q05 = 2.850; cd = q05 * np.sqrt(k * (k + 1) / (6 * N))
    fig, ax = plt.subplots(figsize=(9.5, 3.6), facecolor=SURF)
    ys = np.arange(len(ranks))[::-1]
    colors = [BLUE if m == "Stacking" else GRAY for m in ranks.index]
    ax.hlines(ys, ranks.values - cd / 2, ranks.values + cd / 2, color=GRID, lw=3, zorder=1)
    ax.scatter(ranks.values, ys, s=140, c=colors, edgecolors=SURF, linewidths=2, zorder=3)
    for y, (m, v) in zip(ys, ranks.items()):
        ax.text(v, y + 0.32, m, ha="center", fontsize=10, color=(BLUE if m == "Stacking" else INK2),
                fontweight=("bold" if m == "Stacking" else "normal"))
    ax.set_yticks([]); ax.set_xlim(1, k); ax.set_xlabel("Average rank across 3 datasets  (lower is better)", color=INK2)
    ax.set_title(f"Nemenyi critical-distance diagram  (CD = {cd:.2f}, $\\alpha$=0.05)\n"
                 f"Stacking (proposed) ranks #1; N=3 datasets limits test power", color=INK, pad=8)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS); ax.tick_params(colors=MUTED)
    plt.tight_layout(); plt.savefig(FIG / "cd_diagram.png", dpi=300, bbox_inches="tight", facecolor=SURF); plt.close()


# ============ Fig 4: 知识消融(M0/M1/M2) ============
def fig_e5():
    df = pd.read_csv(TAB / "e5_ablation.csv")
    models = df["Model"].tolist()
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURF)
    x = np.arange(len(models)); w = 0.26
    for i, (lv, lab) in enumerate(zip(["M0", "M1", "M2"], ["M0  metadata only", "M1  +work-scope knowledge", "M2  +DDC unit-price"])):
        ax.bar(x + (i - 1) * w, df[f"MAPE_{lv}"], w, label=lab, color=SEQ3[i], edgecolor=SURF, linewidth=2, zorder=3)
        for xi, v in zip(x + (i - 1) * w, df[f"MAPE_{lv}"]):
            ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([m.replace(" hybrid", "\n(hybrid)") for m in models], color=INK2)
    ax.set_ylabel("MAPE (%)  lower is better", color=INK2)
    ax.set_title("Knowledge-augmentation ablation (NYC SCA real cost)\nInjecting construction work-knowledge (M1) then DDC cost-knowledge (M2)", color=INK, pad=10)
    ax.set_ylim(70, df[["MAPE_M0", "MAPE_M1", "MAPE_M2"]].values.max() * 1.05)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    style(ax)
    plt.tight_layout(); plt.savefig(FIG / "e5_ddc_ablation.png", dpi=300, bbox_inches="tight", facecolor=SURF); plt.close()


# ============ Fig 5: 组件消融 ============
def fig_component():
    df = pd.read_csv(TAB / "component_ablation.csv")
    colors = [BLUE if c.startswith("Full") else GRAY for c in df["config"]]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURF)
    ax.bar(df["config"], df["R2_mean"], yerr=df["R2_std"], capsize=4, color=colors, edgecolor=SURF,
           linewidth=2, width=0.62, error_kw={"ecolor": MUTED, "lw": 1}, zorder=3)
    full = df["R2_mean"].iloc[0]; ax.axhline(full, ls=":", color=BLUE, lw=1.2, alpha=0.7)
    ax.set_ylim(0.93, 0.975); ax.set_ylabel("$R^2$ (UCI #437)  higher is better", color=INK2)
    ax.set_title("Stacking component ablation — each base learner contributes", color=INK, pad=10)
    for i, v in enumerate(df["R2_mean"]): ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8.5, color=INK2)
    ax.tick_params(axis="x", rotation=14)
    for l in ax.get_xticklabels(): l.set_ha("right"); l.set_color(INK2)
    style(ax)
    plt.tight_layout(); plt.savefig(FIG / "component_ablation.png", dpi=300, bbox_inches="tight", facecolor=SURF); plt.close()


# ============ Fig 6: SHAP 驱动(UCI + NYC, 双面板) ============
def fig_shap():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), facecolor=SURF)
    for ax, (csv, title) in zip(axes, [("e4_shap_importance.csv", "(a) UCI #437 — residential cost drivers"),
                                        ("e4b_nyc_shap.csv", "(b) NYC SCA — public-school cost drivers")]):
        df = pd.read_csv(TAB / csv).head(14).iloc[::-1]
        cols = [BLUE if i == len(df) - 1 else "#9ec5f4" for i in range(len(df))]  # 顶(最重要)=深蓝
        ax.barh(df["feature"], df["mean_abs_shap"], color=cols, edgecolor=SURF, linewidth=1.5, height=0.72, zorder=3)
        ax.set_title(title, color=INK, pad=8); ax.set_xlabel("mean |SHAP value|", color=INK2)
        ax.tick_params(axis="y", labelsize=8.5)
        for l in ax.get_yticklabels(): l.set_color(INK2)
        style(ax, ygrid=False); ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    fig.suptitle("Design-stage cost drivers (SHAP, real cost data)", fontsize=13, color=INK, y=1.01)
    plt.tight_layout(); plt.savefig(FIG / "shap_drivers_2panel.png", dpi=300, bbox_inches="tight", facecolor=SURF); plt.close()


def main():
    fig_main(); print("  main_comparison_3panel.png", flush=True)
    fig_heatmap(); print("  cross_dataset_heatmap.png", flush=True)
    fig_cd(); print("  cd_diagram.png", flush=True)
    fig_e5(); print("  e5_ddc_ablation.png", flush=True)
    fig_component(); print("  component_ablation.png", flush=True)
    fig_shap(); print("  shap_drivers_2panel.png", flush=True)
    print("顶刊级图表已生成", flush=True)


if __name__ == "__main__":
    main()
