#!/usr/bin/env python3
"""Context invariance plot: distribution of predicted lengths across canvas sizes.

3 rows (models) × 4 cols (tasks). Each cell shows KDE distributions of
predicted lengths for C=512, 1024, 2048.

Usage:
    python scripts/plot_context_invariance.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

TASKS        = ["gsm8k", "math500", "humaneval", "mbpp"]
TASK_LABELS  = {"gsm8k": "GSM8K", "math500": "MATH-500",
                "humaneval": "HumanEval", "mbpp": "MBPP"}

MODELS = [
    ("llada_8b_instruct",    "LLaDA-8B"),
    ("llada_1_5",            "LLaDA-1.5"),
    ("dream_v0_instruct_7b", "DREAM-7B"),
]

CONTEXT_SIZES = [512, 1024, 2048, 4096]
COLORS        = ["#AABFD4", "#7090C0", "#3060A8", "#0B2A6B"]


def kde_plot(ax, lens, color, label):
    lens = np.array(lens, dtype=float)
    kde  = gaussian_kde(lens, bw_method=0.25)
    xs   = np.linspace(0, max(lens) * 1.05, 400)
    ys   = kde(xs)
    ax.plot(xs, ys, color=color, lw=1.8, label=label)
    ax.fill_between(xs, ys, alpha=0.15, color=color)


def main():
    out = Path("figures/context_invariance.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(MODELS), len(TASKS),
        figsize=(14, 3.0 * len(MODELS)),
        sharey=False,
    )
    fig.subplots_adjust(wspace=0.28, hspace=0.45)

    for row, (model_dir, model_label) in enumerate(MODELS):
        base = Path(f"results/context_invariance/{model_dir}")

        for col, task in enumerate(TASKS):
            ax   = axes[row][col]
            recs = json.loads((base / f"{task}.json").read_text())

            for C, color in zip(CONTEXT_SIZES, COLORS):
                lens = [
                    r["preds"][str(C)]["pred_new"]
                    for r in recs
                    if r["preds"].get(str(C)) is not None
                ]
                if lens:
                    kde_plot(ax, lens, color, f"C={C}")

            if row == 0:
                ax.set_title(TASK_LABELS[task], fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{model_label}\nDensity", fontsize=10)
            if row == len(MODELS) - 1:
                ax.set_xlabel("Predicted length (tokens)", fontsize=9)

            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # single legend top-right
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9,
               frameon=False, bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(
        "Context invariance — predicted length distributions at q=0.90",
        fontsize=12, y=1.02,
    )

    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
