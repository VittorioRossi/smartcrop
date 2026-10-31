#!/usr/bin/env python3
"""Compute–accuracy Pareto plot for DREAM (2×2, one panel per benchmark).

x-axis : N_token (canvas tokens allocated)
y-axis : Accuracy / Pass@1 (%)

Curves:
  fixed_length  — gray dashed line, N ∈ available GL values
  cdf_eos       — single colored marker (q=0.90, eos_free=False)
  cdf_eos_free  — single colored marker (q=0.90, eos_free=True)

Usage:
    python scripts/plot_pareto_dream.py --out figures/pareto_dream.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# MBPP GL=1024 incomplete (43/374 records) — omitted
# `n_samples` is the number of evaluation examples per benchmark, used for
# Wilson 95% confidence intervals on accuracy / Pass@1.

DATA = {
    "GSM8K": {
        "n_samples": 1319,
        "fixed_length": {
            "n":   [64,   128,   256,   512,   1024],
            "acc": [39.0, 65.6,  79.8,  83.9,  84.4],
        },
        "cdf_eos":      {"n": 203.7, "acc": 79.2},
        "cdf_eos_free": {"n": 203.7, "acc": 76.0},
        "task_mean":    {"n": 204.0, "acc": 74.2},
    },
    "MATH500": {
        "n_samples": 500,
        "fixed_length": {
            "n":   [64,   128,   256,   512,  1024],
            "acc": [25.0, 35.0,  41.8,  44.0, 45.0],
        },
        "cdf_eos":      {"n": 284.6, "acc": 43.8},
        "cdf_eos_free": {"n": 284.6, "acc": 44.0},
        "task_mean":    {"n": 285.0, "acc": 46.0},
    },
    "HumanEval": {
        "n_samples": 164,
        "fixed_length": {
            "n":   [64,  128,   256,   512,   1024],
            "acc": [6.1, 29.3,  53.0,  53.0,  54.9],
        },
        "cdf_eos":      {"n": 301.0, "acc": 47.6},
        "cdf_eos_free": {"n": 301.0, "acc": 51.8},
        "task_mean":    {"n": 301.0, "acc": 53.0},
    },
    "MBPP": {
        "n_samples": 374,
        "fixed_length": {
            "n":   [64,   128,   256,   512],
            "acc": [14.4, 42.8,  48.7,  50.3],
        },
        "cdf_eos":      {"n": 267.0, "acc": 46.3},
        "cdf_eos_free": {"n": 267.0, "acc": 47.9},
        "task_mean":    {"n": 267.0, "acc": 43.9},
    },
}

BENCHMARKS = ["GSM8K", "MATH500", "HumanEval", "MBPP"]


# ── Stats ─────────────────────────────────────────────────────────────────────

def wilson_halfwidth(p_pct: float, n: int, z: float = 1.96) -> float:
    """Wilson 95% CI half-width on accuracy (in percentage points).

    Symmetric half-width = z * sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n).
    Returns the half-width in the same scale as `p_pct` (percent).
    """
    p = p_pct / 100.0
    denom = 1.0 + z * z / n
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return half * 100.0


def wilson_array(acc, n):
    return np.array([wilson_halfwidth(a, n) for a in acc])

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE = "#2166AC"
RED  = "#D6604D"
GRAY = "#888888"

RC = {
    "font.family":     "serif",
    "font.size":       9,
    "axes.labelsize":  9,
    "axes.titlesize":  10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
    "figure.dpi":      150,
}


def plot_panel(ax, benchmark: str):
    d = DATA[benchmark]
    n_samples = d["n_samples"]

    fl = d["fixed_length"]
    fl_err = wilson_array(fl["acc"], n_samples)
    ax.errorbar(
        fl["n"], fl["acc"], yerr=fl_err,
        color=GRAY, lw=1.2, ls="--",
        marker="s", ms=4, zorder=2,
        ecolor=GRAY, elinewidth=0.9, capsize=2.5, capthick=0.9,
        label="Fixed-length",
    )

    cdf = d["cdf_eos"]
    cdf_err = wilson_halfwidth(cdf["acc"], n_samples)
    ax.errorbar(
        cdf["n"], cdf["acc"], yerr=cdf_err,
        marker="o", ms=7, color=BLUE, mec=BLUE, mfc=BLUE, ls="none",
        ecolor=BLUE, elinewidth=1.0, capsize=2.5, capthick=1.0,
        zorder=5, label=r"CDF (q=0.90)",
    )

    ef = d["cdf_eos_free"]
    ef_err = wilson_halfwidth(ef["acc"], n_samples)
    ax.errorbar(
        ef["n"], ef["acc"], yerr=ef_err,
        marker="^", ms=7, color=RED, mec=RED, mfc=RED, ls="none",
        ecolor=RED, elinewidth=1.0, capsize=2.5, capthick=1.0,
        zorder=5, label=r"CDF EOS-free (q=0.90)",
    )

    x_right = max(max(fl["n"]), cdf["n"], ef["n"]) * 1.15
    ax.set_xlim(left=0, right=x_right)

    ax.set_title(f"{benchmark} (n={n_samples})")
    ax.set_xlabel(r"$N_{\rm token}$")
    ax.set_ylabel("Accuracy / Pass@1 (%)")
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures/pareto_dream.pdf"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    for ax, bm in zip(axes.flat, BENCHMARKS):
        plot_panel(ax, bm)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle("DREAM-v0-Instruct-7B — Compute–Accuracy Pareto", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
