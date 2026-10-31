#!/usr/bin/env python3
"""Main CDF figure: 3 panels (one per model), mean CDF(EOS) per benchmark
with ±1-std shading.

Usage:
    python scripts/plot_eos_cdf_examples.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS       = ["gsm8k", "math500", "humaneval", "mbpp"]
TASK_LABELS = {"gsm8k": "GSM8K", "math500": "MATH-500",
               "humaneval": "HumanEval", "mbpp": "MBPP"}
TASK_COLORS = {
    "gsm8k":     "#1A6DB5",
    "math500":   "#E05A00",
    "humaneval": "#1D9E3A",
    "mbpp":      "#9B59B6",
}

MODELS = [
    ("llada_8b_instruct",    "LLaDA-8B"),
    ("llada_1_5",            "LLaDA-1.5"),
    ("dream_v0_instruct_7b", "DREAM-7B"),
]

C_REF = 1024
Q     = 0.90


def load_peos_matrix(model_dir: str, task: str) -> np.ndarray:
    """Return float32 [N, C_REF] matrix of raw per-position p(EOS), padded to C_REF.

    No smoothing: each row is the raw p_eos array for one sample, padded with the
    last observed value (typically near 1) to length C_REF.
    """
    base      = Path(f"results/context_invariance/{model_dir}")
    probs_dir = base / "probs" / task
    files     = sorted(probs_dir.glob(f"*_C{C_REF}.npz"))
    if not files:
        return np.empty((0, C_REF), dtype=np.float32)

    curves = []
    for f in files:
        p_eos = np.load(f)["probs"][:, 0].astype(np.float32)
        if len(p_eos) < C_REF:
            p_eos = np.concatenate(
                [p_eos, np.ones(C_REF - len(p_eos), dtype=np.float32) * p_eos[-1]]
            )
        curves.append(p_eos[:C_REF])

    return np.stack(curves)   # [N, C_REF]


def last_hit_one(mat: np.ndarray, threshold: float = 0.995) -> int:
    """Position where the mean curve first permanently stays above threshold."""
    mean = mat.mean(axis=0)
    hits = np.where(mean >= threshold)[0]
    return int(hits[0]) + 1 if len(hits) else mat.shape[1]


def round_up_50(x: int) -> int:
    return int(np.ceil(x / 50) * 50)


def add_axis_break(ax, color="#888888"):
    """Draw a short dotted extension of the x-axis spine to signal truncation."""
    # plot a dotted segment in axes-fraction coords just past the right edge
    xlim = ax.get_xlim()
    span = xlim[1] - xlim[0]
    extra = span * 0.10          # extend by 10% of the current range
    ax.plot([xlim[1], xlim[1] + extra], [0, 0],
            transform=ax.get_xaxis_transform(),
            color="#555555", lw=1.1, ls=":", clip_on=False)


def main():
    out = Path("figures/eos_cdf_examples.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    xs = np.arange(1, C_REF + 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    legend_handles = []

    for ax_idx, (ax, (model_dir, model_label)) in enumerate(zip(axes, MODELS)):
        # collect all mats to determine x limit for this panel
        mats = {}
        for task in TASKS:
            mat = load_peos_matrix(model_dir, task)
            if mat.shape[0] > 0:
                mats[task] = mat

        # x limit: rightmost last-hit-1 across tasks, rounded to 50
        xlim = round_up_50(max(last_hit_one(m) for m in mats.values())) if mats else C_REF

        for task in TASKS:
            if task not in mats:
                continue
            color = TASK_COLORS[task]
            mat   = mats[task]
            mean  = mat.mean(axis=0)
            std   = mat.std(axis=0)

            ax.fill_between(xs, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            alpha=0.07, color=color, step="mid")
            # Stepped (binned) plot of mean raw per-position p(EOS) — no smoothing.
            h, = ax.step(xs, mean, where="mid", color=color, lw=1.0,
                         label=TASK_LABELS[task] if ax_idx == 0 else "_")
            if ax_idx == 0:
                legend_handles.append(h)

        ax.set_title(model_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Response position $i$", fontsize=10)
        ax.set_xlim(0, xlim)
        ax.set_ylim(-0.02, 1.05)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_idx > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)

        # break mark only when canvas is truncated
        if xlim < C_REF:
            add_axis_break(ax)

    axes[0].set_ylabel(r"$p(\mathrm{EOS})$ at $t=1$", fontsize=10)

    fig.legend(
        legend_handles,
        [TASK_LABELS[t] for t in TASKS],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=4,
        fontsize=9,
        frameon=False,
    )

    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
