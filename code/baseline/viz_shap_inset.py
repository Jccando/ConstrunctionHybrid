"""Standalone SHAP feature-importance figure (no title, publication style), for the SHAP output box of the method diagram (Fig. 2).
Data: SHAP of NYC SCA real public-school cost (e4b_nyc_shap.csv).
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from utils.datasets import ROOT
TAB = ROOT / "code" / "experimentresult" / "tables"; FIG = ROOT / "code" / "experimentresult" / "figures"

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "figure.dpi": 130, "savefig.dpi": 300, "axes.unicode_minus": False})
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# light-to-dark blue ramp (most important = darkest)
RAMP = ["#104281", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6"]


def pretty(name):
    s = str(name)
    s = s.replace("scope_", "").replace("phase_", "").replace("program_type_", "").replace("status_", "").replace("district_", "District ")
    s = s.replace("_", " ").strip()
    return s


def main():
    df = pd.read_csv(TAB / "e4b_nyc_shap.csv").head(10).iloc[::-1].reset_index(drop=True)  # most important on top
    df["label"] = df["feature"].map(pretty)
    n = len(df)
    colors = [RAMP[n - 1 - i] for i in range(n)]  # top (most important) = darkest

    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURF)
    ax.barh(df["label"], df["mean_abs_shap"], color=colors, edgecolor=SURF, linewidth=1.6, height=0.72, zorder=3)
    xmax = df["mean_abs_shap"].max()
    for i, v in enumerate(df["mean_abs_shap"]):
        ax.text(v + xmax * 0.015, i, f"{v:.2f}", va="center", fontsize=9.5, color=INK2)
    ax.set_xlabel("mean |SHAP value|  (cost-driver importance)", color=INK2)
    ax.set_xlim(0, xmax * 1.18)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=10, length=3)
    ax.set_axisbelow(True); ax.xaxis.grid(True, color=GRID, lw=1, zorder=0); ax.set_facecolor(SURF)
    plt.tight_layout()
    out = FIG / "Fig_SHAP_importance_inset.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURF); plt.close()
    print(f"generated: {out}", flush=True)
    print("Top drivers:", list(df["label"])[::-1], flush=True)


if __name__ == "__main__":
    main()
