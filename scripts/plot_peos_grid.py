#!/usr/bin/env python3
"""Grid of per-example p(EOS) CDF curves at t=1 for a single model.

Generates a grid matching the style of peos_dream_all.png, reusable for any
model. Each cell shows one sample's CDF with a red dot at the q=0.90 crossing.

Usage:
    python scripts/plot_peos_grid.py --model llada_8b_instruct \
        --out figures/peos_llada8_all.png
    python scripts/plot_peos_grid.py --model dream_v0_instruct_7b \
        --out figures/peos_dream_all.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS       = ["gsm8k", "math500", "humaneval", "mbpp"]
TASK_LABELS = {"gsm8k": "GSM8K", "math500": "MATH-500",
               "humaneval": "HumanEval", "mbpp": "MBPP"}
LINE_COLOR  = "#08306B"  # dark blue
DOT_COLOR   = "#CC0000"  # red marker for q=0.90 crossing
C_REF  = 1024
Q      = 0.90
NCOLS  = 5


def load_all_examples(model_dir: str, n_per_task: int = 5):
    """Return interleaved list of (p_eos, task), n_per_task from each task."""
    base = Path(f"results/context_invariance/{model_dir}")
    per_task = {t: [] for t in TASKS}
    for task in TASKS:
        json_path = base / f"{task}.json"
        if not json_path.exists():
            continue
        recs = json.loads(json_path.read_text())
        for r in recs:
            entry = r["preds"].get(str(C_REF))
            if entry is None:
                continue
            npz_path = base / "probs" / task / f"{r['npz_key']}_C{C_REF}.npz"
            if not npz_path.exists():
                continue
            arr   = np.load(npz_path)["probs"].astype(np.float32)
            p_eos = arr[:, 0]
            per_task[task].append((p_eos, task))
            if len(per_task[task]) >= n_per_task:
                break

    # interleave: task0[0], task1[0], task2[0], task3[0], task0[1], ...
    examples = []
    for i in range(n_per_task):
        for task in TASKS:
            if i < len(per_task[task]):
                examples.append(per_task[task][i])
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llada_8b_instruct")
    parser.add_argument("--out",   type=Path, default=Path("figures/peos_llada8_all.png"))
    parser.add_argument("--nmax",  type=int,  default=20,
                        help="max examples to show")
    args = parser.parse_args()

    n_per_task = args.nmax // len(TASKS)
    examples = load_all_examples(args.model, n_per_task=n_per_task)
    nrows = (len(examples) + NCOLS - 1) // NCOLS

    fig, axes = plt.subplots(
        nrows, NCOLS,
        figsize=(NCOLS * 2.6, nrows * 2.2),
    )
    axes = axes.flatten()

    for i, (p_eos, task) in enumerate(examples):
        ax = axes[i]
        xs = np.arange(1, len(p_eos) + 1)
        # Stepped (binned) plot of raw per-position p(EOS) — no smoothing.
        ax.step(xs, p_eos, where="mid", color=LINE_COLOR, lw=0.8)

        crossed = p_eos >= Q
        cross = int(np.argmax(crossed))
        has_cross = bool(crossed.any())
        if has_cross:
            ax.plot(xs[cross], p_eos[cross], "o",
                    color=DOT_COLOR, ms=5, zorder=5)

        ax.axhline(Q, color="#cccccc", lw=0.7, ls="--")
        ax.set_xlim(0, len(p_eos))
        ax.set_ylim(-0.02, 1.05)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"{TASK_LABELS[task]}, pos={xs[cross] if has_cross else 'none'}",
                     fontsize=7)

    # hide unused cells
    for j in range(len(examples), len(axes)):
        axes[j].set_visible(False)

    model_label = args.model.replace("_", " ")
    fig.suptitle(
        f"p(EOS) vs position — {model_label}, C={C_REF}, q={Q}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
