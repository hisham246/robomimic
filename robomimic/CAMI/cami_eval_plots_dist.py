import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# Input files
# =========================================================
square_csv = Path("/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/cami_eval_square/trial_results_square.csv")
tool_csv = Path("/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/cami_eval_tool_hang/trial_results_tool_hang.csv")

# Optional save dir
# out_dir = Path("/mnt/data/plots")
# out_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# Global style
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
})

BG_COLOR = "#ffffff"
GRID_COLOR = "#b0b0b0"
BC_COLOR = "#d94b73"
CAMI_COLOR = "#eda055"

def apply_common_style(ax):
    ax.set_facecolor(BG_COLOR)
    ax.figure.patch.set_facecolor(BG_COLOR)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, color=GRID_COLOR, alpha=0.8)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis="both", length=6, width=1.2)

# =========================================================
# Helpers
# =========================================================
def preprocess(df, task_name):
    df = df.copy()

    # success_rate_mean is in [0,1], convert to percentage
    df["success_percent"] = df["success_rate_mean"] * 100

    # Parse method
    df["method"] = np.where(df["model"].str.contains("baseline"), "BC-RNN", "BC-RNN-CaMI (Ours)")

    # Parse fraction
    if task_name == "square":
        df["fraction"] = df["model"].str.extract(r"square_(\d+)_percent").astype(int)
    elif task_name == "tool_hang":
        df["fraction"] = df["model"].str.extract(r"tool_hang_(\d+)_percent").astype(int)
    else:
        raise ValueError("Unknown task_name")

    return df


def compute_summary(df):
    summary = (
        df.groupby(["method", "fraction"])["success_percent"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return summary

def make_data_reduction_line_plot(summary_df, task_title, filename=None):
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    apply_common_style(ax)

    method_order = ["BC-RNN", "BC-RNN-CaMI (Ours)"]
    colors = {
        "BC-RNN": BC_COLOR,
        "BC-RNN-CaMI (Ours)": CAMI_COLOR
    }

    # only use the fractions that actually exist, with equal spacing
    fractions = sorted(summary_df["fraction"].unique())
    x_positions = np.arange(len(fractions))
    frac_to_pos = {f: i for i, f in enumerate(fractions)}

    for method in method_order:
        sub = summary_df[summary_df["method"] == method].sort_values("fraction")

        x = np.array([frac_to_pos[f] for f in sub["fraction"]])
        y = sub["mean"].to_numpy()
        s = sub["std"].to_numpy()

        ax.errorbar(
            x, y, yerr=s,
            fmt='o-',
            markersize=10,
            linewidth=4,
            elinewidth=2.2,
            capsize=8,
            capthick=2.2,
            color=colors[method],
            label=method
        )
        
        for xi, yi, si in zip(x, y, s):
            if method == "BC-RNN-CaMI (Ours)":
                ax.text(
                    xi,
                    yi + si + 2.9,
                    f"{yi:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    color=colors[method],
                    fontweight="regular"
                )
            else:
                ax.text(
                    xi,
                    yi - si - 2.5,
                    f"{yi:.1f}",
                    ha="center",
                    va="top",
                    fontsize=18,
                    color=colors[method],
                    fontweight="regular"
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(f) for f in fractions])
    ax.set_xlim(-0.25, len(fractions) - 1 + 0.25)

    ax.set_xlabel("Training Data Fraction (%)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(task_title, pad=12)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="best")

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.show()


def make_full_data_box_plot(df, task_title, filename=None):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    apply_common_style(ax)

    full_df = df[df["fraction"] == 100].copy()

    bc_vals = full_df[full_df["method"] == "BC-RNN"]["success_percent"].to_numpy()
    cami_vals = full_df[full_df["method"] == "BC-RNN-CaMI (Ours)"]["success_percent"].to_numpy()

    box = ax.boxplot(
        [bc_vals, cami_vals],
        labels=["BC-RNN", "BC-RNN-CaMI\n(Ours)"],
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2)
    )

    box["boxes"][0].set_facecolor(BC_COLOR)
    box["boxes"][1].set_facecolor(CAMI_COLOR)
    box["boxes"][0].set_alpha(0.75)
    box["boxes"][1].set_alpha(0.75)

    # overlay trial points
    x1 = np.random.normal(1, 0.04, size=len(bc_vals))
    x2 = np.random.normal(2, 0.04, size=len(cami_vals))
    ax.scatter(x1, bc_vals, s=45, color="black", alpha=0.7, zorder=3)
    ax.scatter(x2, cami_vals, s=45, color="black", alpha=0.7, zorder=3)

    ax.set_ylabel("Success Rate (%)")
    ax.set_title(task_title, pad=12)

    # adaptive y-axis
    all_vals = np.concatenate([bc_vals, cami_vals])
    ymin = np.floor(all_vals.min() - 5)
    ymax = np.ceil(all_vals.max() + 5)
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.show()

# =========================================================
# Load data
# =========================================================
square_df = preprocess(pd.read_csv(square_csv), task_name="square")
tool_df = preprocess(pd.read_csv(tool_csv), task_name="tool_hang")

square_summary = compute_summary(square_df)
tool_summary = compute_summary(tool_df)

print("Square summary:")
print(square_summary.to_string(index=False))
print()
print("Tool-hang summary:")
print(tool_summary.to_string(index=False))

# =========================================================
# Plots
# =========================================================
make_data_reduction_line_plot(
    square_summary,
    "Square Task: Effect of Training Data Reduction"
)

make_data_reduction_line_plot(
    tool_summary,
    "Tool-Hang Task: Effect of Training Data Reduction"
)

make_full_data_box_plot(
    square_df,
    "Square Task: Rollout Performance at 100% Data"
)

make_full_data_box_plot(
    tool_df,
    "Tool-Hang Task: Rollout Performance at 100% Data"
)