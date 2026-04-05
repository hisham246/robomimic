import matplotlib.pyplot as plt
from pathlib import Path

# Optional save directory
# out_dir = Path("/mnt/data")
# out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Data
# -----------------------------
method_labels = ["BC-RNN", "BC-RNN-CaMI (Ours)"]

square_labels = ["25%", "50%", "100%"]
square_bc = [54, 64, 82]
square_cami = [60, 80, 88]

square_full = [82, 88]

tool_labels = ["50%", "100%"]
tool_bc = [16, 40]
tool_cami = [48, 64]

tool_full = [40, 64]

# -----------------------------
# Global aesthetic settings
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 26,
    "axes.titlesize": 28,
    "axes.labelsize": 30,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 26,
})

BG_COLOR = "#ffffff"
GRID_COLOR = "#b0b0b0"
BC_COLOR = "#d94b73"      # pink/red
CAMI_COLOR = "#eda055"    # orange
BAR_BC_COLOR = "#d94b73"
BAR_CAMI_COLOR = "#eda055"

def apply_common_style(ax):
    ax.set_facecolor(BG_COLOR)
    ax.figure.patch.set_facecolor(BG_COLOR)

    ax.grid(axis="y", linestyle="--", linewidth=1.0, color=GRID_COLOR)
    ax.set_axisbelow(True)

    # Remove top/right spines and soften left/bottom
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis='both', length=6, width=1.2)

def make_data_reduction_line_plot(labels, baseline, cami, title, filename=None):
    x = [0, 2.2, 4.4][:len(labels)]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    apply_common_style(ax)

    ax.plot(
        x, baseline,
        marker="o",
        markersize=12,
        linewidth=6.0,
        color=BC_COLOR,
        label="BC-RNN"
    )

    ax.plot(
        x, cami,
        marker="o",
        markersize=12,
        linewidth=6.0,
        color=CAMI_COLOR,
        label="BC-RNN-CaMI (Ours)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(min(x) - 0.6, max(x) + 0.6)
    ax.set_ylim(-2, 100)
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel("Training Data Fraction")
    ax.set_title(title, pad=14)

    for xi, yi in zip(x, baseline):
        ax.text(
            xi,
            yi - 5,
            f"{yi}%",
            ha="center",
            va="top",
            fontsize=24
        )

    for xi, yi in zip(x, cami):
        ax.text(
            xi,
            yi + 2,
            f"{yi}%",
            ha="center",
            va="bottom",
            fontsize=24
        )

    ax.legend(frameon=False, loc="best")

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.show()

def make_full_data_bar_plot(labels, values, title, filename=None):
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    apply_common_style(ax)

    colors = [BAR_BC_COLOR, BAR_CAMI_COLOR]
    bars = ax.bar(labels, values, width=0.55, color=colors)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(title, pad=14)

    # Optional value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=24
        )

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.show()

# -----------------------------
# Plots
# -----------------------------
make_data_reduction_line_plot(
    square_labels,
    square_bc,
    square_cami,
    "Effect of Training Data Reduction"
)

make_data_reduction_line_plot(
    tool_labels,
    tool_bc,
    tool_cami,
    "Effect of Training Data Reduction"
)

make_full_data_bar_plot(
    method_labels,
    square_full,
    "Rollout Performance"
)

make_full_data_bar_plot(
    method_labels,
    tool_full,
    "Rollout Performance"
)