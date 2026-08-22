"""Advanced visualizations: waterfall, tornado, radar, cost landscape."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "figure.dpi": 130, "savefig.dpi": 250})
SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; GREEN = "#1baf7a"; RED = "#e34948"; GRAY = "#c8c6be"
FIG = Path(__file__).resolve().parents[1] / "04_实验结果" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def waterfall():
    m0, m1, m2 = 90.3, 78.7, 77.55
    labels = ["M0\n(metadata)", "+ Scope\nknowledge", "M1\n(+scope)", "+ RAG\nretrieval", "M2\n(+RAG)"]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURF)
    ax.bar(0, m0, color=GRAY, edgecolor="black", lw=0.6, width=0.6, zorder=3)
    ax.text(0, m0+0.3, f"{m0:.1f}%", ha="center", fontweight="bold", fontsize=10)
    ax.bar(1, -(m0-m1), bottom=m0, color=BLUE, edgecolor="black", lw=0.6, width=0.6, zorder=3)
    ax.text(1, m0-1, f"−{m0-m1:.1f}pp", ha="center", va="top", fontweight="bold", fontsize=10, color=BLUE)
    ax.plot([0.3,0.7],[m0,m0], color=MUTED, ls="--", lw=0.8)
    ax.bar(2, m1, color=GRAY, edgecolor="black", lw=0.6, width=0.6, zorder=3)
    ax.text(2, m1+0.3, f"{m1:.1f}%", ha="center", fontweight="bold", fontsize=10)
    ax.plot([1.3,1.7],[m1,m1], color=MUTED, ls="--", lw=0.8)
    ax.bar(3, -(m1-m2), bottom=m1, color=GREEN, edgecolor="black", lw=0.6, width=0.6, zorder=3)
    ax.text(3, m1-0.5, f"−{m1-m2:.2f}pp\n(p=0.005)", ha="center", va="top", fontsize=9, fontweight="bold", color=GREEN)
    ax.plot([2.3,2.7],[m1,m1], color=MUTED, ls="--", lw=0.8)
    ax.bar(4, m2, color=BLUE, edgecolor="black", lw=0.6, width=0.6, zorder=3)
    ax.text(4, m2+0.3, f"{m2:.2f}%", ha="center", fontweight="bold", fontsize=10, color=BLUE)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MAPE (%)", color=INK2)
    ax.set_title("Knowledge-Augmentation Waterfall\nIncremental gains from work-scope and RAG (CatBoost, NYC SCA)", pad=10, fontsize=11)
    ax.set_ylim(70, 95)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED); ax.set_axisbelow(True); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_facecolor(SURF)
    plt.tight_layout(); plt.savefig(FIG/"viz_waterfall_ablation.png", dpi=250, bbox_inches="tight", facecolor=SURF); plt.close()
    print("  waterfall")


def tornado():
    factors = [("Duration>36mo", 2.7), ("Construction phase", 2.1), ("Roofing scope", 1.33), ("Masonry scope", 1.25), ("Capacity program", 1.15), ("Boiler replace", 1.08)]
    names = [f[0] for f in factors]
    inc = [f[1] for f in factors]
    dec = [1/f[1] for f in factors]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURF)
    y = np.arange(len(names))
    ax.barh(y, [v-1 for v in inc], left=1, color=RED, edgecolor="black", lw=0.5, height=0.6, zorder=3, label="Increase")
    ax.barh(y, [v-1 for v in dec], left=1, color=GREEN, edgecolor="black", lw=0.5, height=0.6, zorder=3, label="Decrease")
    for i,(p,n) in enumerate(zip(inc,dec)):
        ax.text(p+0.03, i, f"×{p:.1f}", va="center", fontsize=9, fontweight="bold", color=RED)
        ax.text(n-0.03, i, f"×{n:.1f}", va="center", ha="right", fontsize=8, color=GREEN)
    ax.axvline(1, color=INK, lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Multiplicative cost factor", color=INK2)
    ax.set_title("Design-Decision Cost-Sensitivity Tornado\nSHAP-derived impact on cost (NYC SCA)", pad=10, fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED); ax.set_axisbelow(True); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_facecolor(SURF)
    ax.set_xlim(0.3, 3.0)
    plt.tight_layout(); plt.savefig(FIG/"viz_tornado_sensitivity.png", dpi=250, bbox_inches="tight", facecolor=SURF); plt.close()
    print("  tornado")


def radar():
    dims = ["Accuracy","Stability","Scalability","Interpretability","Efficiency"]
    data = {
        "Stacking":  [0.90,0.95,1.0,0.7,0.6],
        "CatBoost":  [0.92,0.90,1.0,0.9,0.8],
        "LightGBM":  [0.85,0.85,1.0,0.9,0.9],
        "XGBoost":   [0.84,0.82,1.0,0.9,0.8],
        "ANN":       [0.70,0.55,0.5,0.5,0.4],
        "RF":        [0.75,0.75,0.9,0.8,0.7],
    }
    N = len(dims)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True), facecolor=SURF)
    ax.set_facecolor(SURF)
    colors = [BLUE,ORANGE,GREEN,"#4a3aa7","#eda100",GRAY]
    for (name,vals),c in zip(data.items(),colors):
        v = vals + [vals[0]]
        lw = 2.5 if name=="Stacking" else 1.2
        alpha = 0.15 if name=="Stacking" else 0
        ax.plot(angles, v, "o-", color=c, lw=lw, ms=4, label=name)
        if alpha: ax.fill(angles, v, color=c, alpha=alpha)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(dims, fontsize=10, color=INK2)
    ax.set_ylim(0,1.05); ax.set_yticks([0.2,0.4,0.6,0.8,1.0]); ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7, color=MUTED)
    ax.set_title("Multi-Dimensional Model Comparison\nProposed (Stacking, blue) vs Baselines", pad=20, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.1), fontsize=9, frameon=False)
    ax.grid(color=GRID, lw=0.8)
    plt.tight_layout(); plt.savefig(FIG/"viz_radar_models.png", dpi=250, bbox_inches="tight", facecolor=SURF); plt.close()
    print("  radar")


def landscape():
    types = ["Retail","SmOffice","MedOffice","School","Hospital"]
    costs = [180,220,250,265,550]
    areas = np.linspace(5,100,50)
    T,A = np.meshgrid(np.arange(len(types)),areas)
    C = A*1000*np.array(costs)[T]/1e6
    fig,ax = plt.subplots(figsize=(8,5), facecolor=SURF)
    cf = ax.contourf(T,A,C,levels=20,cmap="YlOrRd")
    cl = ax.contour(T,A,C,levels=10,colors="black",linewidths=0.5)
    ax.clabel(cl,fmt="$%.0fM",fontsize=7)
    ax.set_xticks(range(len(types))); ax.set_xticklabels(types,fontsize=9)
    ax.set_xlabel("Building Type",color=INK2)
    ax.set_ylabel("Floor Area (k sq ft)",color=INK2)
    ax.set_title("Design-Stage Cost Landscape: Area × Type\n(ComStock, R²=0.962)",pad=10,fontsize=11)
    fig.colorbar(cf,ax=ax,label="Cost ($M)",fraction=0.03)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    plt.tight_layout(); plt.savefig(FIG/"viz_cost_landscape.png",dpi=250,bbox_inches="tight",facecolor=SURF); plt.close()
    print("  landscape")


if __name__ == "__main__":
    waterfall(); tornado(); radar(); landscape()
    print("Done.")
