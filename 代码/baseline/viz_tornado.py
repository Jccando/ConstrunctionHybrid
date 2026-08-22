"""Cost-sensitivity tornado chart — clean version, no overlaps."""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "figure.dpi": 130, "savefig.dpi": 250})
SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"
BLUE = "#2a78d6"; GREEN = "#1baf7a"; RED = "#e34948"
FIG = Path(__file__).resolve().parents[1] / "04_实验结果" / "figures"


def tornado():
    factors = [
        ("Duration > 36 months", 2.70),
        ("Construction phase",    2.10),
        ("Roofing scope",         1.33),
        ("Masonry scope",         1.25),
        ("Capacity program",      1.15),
        ("Boiler replacement",    1.08),
    ]
    names = [f[0] for f in factors]
    inc   = [f[1] for f in factors]
    dec   = [1 / f[1] for f in factors]

    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURF)

    n = len(names)
    y = np.arange(n) * 1.4  # extra vertical spacing between bars
    bar_h = 0.6             # bar height (gap = 1.4 - 0.6 = 0.8)

    # Draw bars
    ax.barh(y, [v - 1 for v in inc], left=1, color=RED, edgecolor="black",
            lw=0.6, height=bar_h, zorder=3)
    ax.barh(y, [v - 1 for v in dec], left=1, color=GREEN, edgecolor="black",
            lw=0.6, height=bar_h, zorder=3)

    # Value labels — placed OUTSIDE the bar ends with generous offset
    for i, (p, n_) in enumerate(zip(inc, dec)):
        # Right side (increase): label to the RIGHT of the red bar end
        ax.text(p + 0.08, y[i], f"×{p:.1f}", va="center", ha="left",
                fontsize=10, fontweight="bold", color=RED, zorder=4)
        # Left side (decrease): label to the LEFT of the green bar end
        ax.text(n_ - 0.08, y[i], f"×{n_:.1f}", va="center", ha="right",
                fontsize=9, color=GREEN, zorder=4)

    # Baseline at ×1.0
    ax.axvline(1, color=INK, lw=1.3, zorder=2)

    # Y-axis labels
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, color=INK2)
    ax.invert_yaxis()  # Largest factor at top

    # X-axis
    ax.set_xlabel("Multiplicative cost factor", color=INK2, fontsize=10)
    ax.set_xlim(-0.05, 3.3)  # extra space on both sides for labels
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    # Legend — ABOVE the plot, centered, completely outside
    legend_handles = [
        Patch(facecolor=RED, edgecolor="black", lw=0.5, label="Cost increase"),
        Patch(facecolor=GREEN, edgecolor="black", lw=0.5, label="Cost decrease"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              ncol=2, fontsize=9, frameon=False, borderaxespad=0)

    # Clean spines
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_facecolor(SURF)

    plt.tight_layout()
    plt.savefig(FIG / "viz_tornado_sensitivity.png", dpi=250,
                bbox_inches="tight", facecolor=SURF, pad_inches=0.15)
    plt.close()
    print("  tornado v3 (clean, no overlaps)")


if __name__ == "__main__":
    tornado()
