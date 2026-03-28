#!/usr/bin/env python3
"""Plot survival function results from run_gsm8k_step_analysis.py."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Path to step_analysis_XXXXX directory")
    parser.add_argument("--canvas", type=int, default=512)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out) if args.out else results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas = args.canvas
    npz_path = results_dir / f"survival_arrays_canvas{canvas}.npz"
    gen_df   = pd.read_csv(results_dir / "generation_results.csv")
    gen_df   = gen_df[gen_df["canvas_size"] == canvas]

    if not npz_path.exists():
        print(f"No survival arrays found at {npz_path}. Re-run with updated script.")
        return

    data = np.load(npz_path)
    # Parse keys: ex{i}_step{k}
    parsed = {}
    for key in data.files:
        parts = key.split("_step")
        ex_id = int(parts[0].replace("ex", ""))
        step  = int(parts[1])
        parsed.setdefault(ex_id, {})[step] = data[key]

    example_ids = sorted(parsed.keys())
    n_examples  = len(example_ids)
    positions   = np.arange(canvas)

    # ------------------------------------------------------------------ #
    # Fig 1: Survival function S(t) at different denoising steps k
    #         one subplot per example
    # ------------------------------------------------------------------ #
    ncols = min(n_examples, 3)
    nrows = (n_examples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, ex_id in enumerate(example_ids):
        ax = axes[idx // ncols][idx % ncols]
        steps = sorted(parsed[ex_id].keys())
        cmap  = cm.plasma(np.linspace(0.1, 0.9, len(steps)))

        for step, color in zip(steps, cmap):
            s_t = parsed[ex_id][step]
            ax.plot(positions, s_t, color=color, lw=0.8, alpha=0.7)

        # Mark q=0.9 threshold
        ax.axhline(0.1, color="red", ls="--", lw=1, label="S(t)=0.1  (q=0.9 cutoff)")

        # Colorbar proxy
        sm = plt.cm.ScalarMappable(cmap="plasma",
                                   norm=plt.Normalize(vmin=min(steps), vmax=max(steps)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Denoising step k", pad=0.02)

        ex_gen = gen_df[gen_df["example_id"] == ex_id]
        ever_correct = ex_gen["correct"].max() if len(ex_gen) else 0
        ax.set_title(f"Example {ex_id}  ({'correct' if ever_correct else 'wrong'})",
                     fontsize=10)
        ax.set_xlabel("Token position t")
        ax.set_ylabel("S(t) = P(stop > t)")
        ax.set_xlim(0, canvas)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

    # Hide unused axes
    for idx in range(n_examples, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Survival function S(t) evolution — canvas={canvas}\n"
                 f"(color = denoising step k, earlier=dark, later=light)",
                 fontsize=12)
    fig.tight_layout()
    p = out_dir / f"survival_function_canvas{canvas}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

    # ------------------------------------------------------------------ #
    # Fig 2: Mean S(t) across examples at representative steps
    # ------------------------------------------------------------------ #
    all_steps = sorted({s for ex in parsed.values() for s in ex})
    rep_steps = all_steps[::max(1, len(all_steps) // 8)]  # ~8 representative steps

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = cm.plasma(np.linspace(0.1, 0.9, len(rep_steps)))

    for step, color in zip(rep_steps, cmap):
        arrs = [parsed[ex][step] for ex in example_ids if step in parsed[ex]]
        if not arrs:
            continue
        mean_s = np.mean(arrs, axis=0)
        ax.plot(positions, mean_s, color=color, lw=1.5,
                label=f"k={step} ({step/canvas:.0%})")

    ax.axhline(0.1, color="red", ls="--", lw=1.2, label="S(t)=0.1  (q=0.9 cutoff)")
    ax.set_xlabel("Token position t")
    ax.set_ylabel("Mean S(t) across examples")
    ax.set_title(f"Mean survival function at representative steps — canvas={canvas}")
    ax.set_xlim(0, canvas)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"mean_survival_canvas{canvas}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

    # ------------------------------------------------------------------ #
    # Fig 3: Accuracy at eval checkpoints
    # ------------------------------------------------------------------ #
    acc = gen_df.groupby("checkpoint_k")["correct"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(acc["checkpoint_k"].astype(str), acc["correct"], color="steelblue", alpha=0.8)
    for _, row in acc.iterrows():
        ax.text(str(int(row["checkpoint_k"])), row["correct"] + 0.02,
                f"{row['correct']:.2f}", ha="center", fontsize=9)
    ax.set_xlabel("Eval checkpoint k")
    ax.set_ylabel("Accuracy (flexible-match)")
    ax.set_title(f"Accuracy at eval checkpoints — canvas={canvas}")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"accuracy_canvas{canvas}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
