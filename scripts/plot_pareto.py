#!/usr/bin/env python3
"""Compute–accuracy Pareto plot (2×2, one panel per benchmark).

x-axis : N_token (canvas tokens allocated)
y-axis : Accuracy / Pass@1 (%)

Curves:
  fixed_length  — gray connected line, N ∈ {64, 128, 256, 512, 1024, 2048}
  daedal        — single marker
  rho_eos       — single marker (Asym)
  smartcrop     — solid colored line, q ∈ {0.50, 0.75, 0.90, 0.95, 0.99}
  smartcrop_ef  — same line dashed, hollow markers (eos_free; q=0.90 only)

Usage:
    python scripts/plot_pareto.py --out figures/pareto.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# All numbers are the paper's table cells, so this figure needs no result files.

DATA = {
    "GSM8K": {
        "n_samples": 1319,
        "fixed_length": {
            "n":   [64,    128,   256,   512,   1024,  2048],
            "acc": [45.9,  69.2,  78.8,  83.2,  83.2,  82.8],
        },
        "daedal":    {"n": 1040.0, "acc": 84.3},
        "rho_asym":  {"n": 663.7,  "acc": 83.6},
        "smartcrop": {
            "q":   [0.50,  0.75,  0.90,  0.95,  0.99],
            "n":   [238.9, 284.6, 336.6, 376.8, 476.7],
            "acc": [77.6,  78.8,  81.1,  82.2,  82.9],
        },
        "smartcrop_ef": {   # eos_free (q=0.90 only)
            "n":   [336.6],
            "acc": [80.7],
        },
    },
    "MATH500": {
        "n_samples": 500,
        "fixed_length": {
            "n":   [64,   128,   256,   512,   1024,  2048],
            "acc": [20.8, 28.2,  33.8,  38.8,  41.0,  40.0],
        },
        "daedal":    {"n": 1059.1, "acc": 40.8},
        "rho_asym":  {"n": 878.0,  "acc": 40.0},
        "smartcrop": {
            "q":   [0.50,  0.75,  0.90,  0.95,  0.99],
            "n":   [523.6, 633.5, 721.0, 772.9, 866.7],
            "acc": [36.6,  41.2,  37.8,  39.0,  39.4],
        },
        "smartcrop_ef": {
            "n":   [721.0],
            "acc": [37.0],
        },
    },
    "HumanEval": {
        "n_samples": 164,
        "fixed_length": {
            "n":   [64,   128,   256,   512,   1024,  2048],
            "acc": [20.1, 21.3,  39.6,  43.9,  47.6,  46.3],
        },
        "daedal":    {"n": 1047.7, "acc": 45.1},
        "rho_asym":  {"n": 990.6,  "acc": 48.8},
        "smartcrop": {
            "q":   [0.50,  0.75,  0.90,  0.95,  0.99],
            "n":   [558.7, 668.6, 812.6, 919.1, 1013.8],
            "acc": [43.3,  45.7,  45.7,  47.6,  48.2],
        },
        "smartcrop_ef": {
            "n":   [812.6],
            "acc": [45.7],
        },
    },
    "MBPP": {
        "n_samples": 500,
        "fixed_length": {
            "n":   [64,   128,   256,   512,   1024,  2048],
            "acc": [21.0, 28.6,  38.6,  36.6,  37.2,  40.6],
        },
        "daedal":    {"n": 1046.5, "acc": 36.2},
        "rho_asym":  {"n": 921.5,  "acc": 37.8},
        "smartcrop": {
            "q":   [0.50,  0.75,  0.90,  0.95,  0.99],
            "n":   [495.3, 573.9, 653.0, 721.3, 856.0],
            "acc": [37.4,  36.1,  38.0,  36.9,  35.0],
        },
        "smartcrop_ef": {
            "n":   [587.7],
            "acc": [36.6],
        },
    },
}

BENCHMARKS = ["GSM8K", "MATH500", "HumanEval", "MBPP"]


# ── Stats ─────────────────────────────────────────────────────────────────────

def wilson_halfwidth(p_pct: float, n: int, z: float = 1.96) -> float:
    """Wilson 95% CI half-width on accuracy (in percentage points)."""
    p = p_pct / 100.0
    denom = 1.0 + z * z / n
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return half * 100.0


def wilson_array(acc, n):
    return np.array([wilson_halfwidth(a, n) for a in acc])

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE    = "#2166AC"
GRAY    = "#888888"
ORANGE  = "#D6604D"
GREEN   = "#4DAC26"
PURPLE  = "#9970AB"

RC = {
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "lines.linewidth":   1.4,
    "lines.markersize":  5,
    "figure.dpi":        150,
}


def plot_panel(ax, benchmark: str):
    d = DATA[benchmark]
    n_samples = d["n_samples"]

    # Fixed-length — gray dashed line with errorbars
    fl = d["fixed_length"]
    fl_err = wilson_array(fl["acc"], n_samples)
    ax.errorbar(
        fl["n"], fl["acc"], yerr=fl_err,
        color=GRAY, lw=1.2, ls="--",
        marker="s", ms=4, zorder=2,
        ecolor=GRAY, elinewidth=0.9, capsize=2.5, capthick=0.9,
        label="Fixed-length",
    )

    # DAEDAL
    dae = d["daedal"]
    dae_err = wilson_halfwidth(dae["acc"], n_samples)
    ax.errorbar(
        dae["n"], dae["acc"], yerr=dae_err,
        marker="D", ms=6, color=ORANGE, mec=ORANGE, mfc=ORANGE, ls="none",
        ecolor=ORANGE, elinewidth=1.0, capsize=2.5, capthick=1.0,
        zorder=4, label="DAEDAL",
    )

    # ρ-EOS — single marker (Asym)
    rho_asym = d["rho_asym"]
    rho_asym_err = wilson_halfwidth(rho_asym["acc"], n_samples)
    ax.errorbar(
        rho_asym["n"], rho_asym["acc"], yerr=rho_asym_err,
        marker="v", ms=6, color=GREEN, mec=GREEN, mfc=GREEN, ls="none",
        ecolor=GREEN, elinewidth=1.0, capsize=2.5, capthick=1.0,
        zorder=4, label=r"$\rho$-EOS",
    )

    # SmartCrop — solid blue line with errorbars
    sc = d["smartcrop"]
    sc_err = wilson_array(sc["acc"], n_samples)
    ax.errorbar(
        sc["n"], sc["acc"], yerr=sc_err,
        color=BLUE, lw=1.6, marker="o", ms=5,
        ecolor=BLUE, elinewidth=1.0, capsize=2.5, capthick=1.0,
        zorder=5, label="SmartCrop",
    )

    # Annotate only q=0.50, q=0.90, q=0.99 to avoid clutter
    annotate_qs = {0.50, 0.90, 0.99}
    for q, n, acc in zip(sc["q"], sc["n"], sc["acc"]):
        if q in annotate_qs:
            ax.annotate(f"$q$={q}", xy=(n, acc), xytext=(5, 3),
                        textcoords="offset points", fontsize=6.5, color=BLUE)

    # SmartCrop + eos_free
    ef = d["smartcrop_ef"]
    ef_err = wilson_array(ef["acc"], n_samples)
    ax.errorbar(
        ef["n"], ef["acc"], yerr=ef_err,
        marker="o", ms=7, color=PURPLE, mec=PURPLE, mfc=PURPLE, ls="none",
        ecolor=PURPLE, elinewidth=1.0, capsize=2.5, capthick=1.0,
        zorder=6, label="SmartCrop + eos_free",
    )

    # Zoom x-axis to the interesting region; fixed-length 1024/2048 shown as ghosts
    sc_max_n = max(sc["n"])
    rho_max_n = rho_asym["n"]
    x_right = max(sc_max_n, rho_max_n, dae["n"]) * 1.12
    ax.set_xlim(left=0, right=x_right)

    ax.set_title(f"{benchmark} (n={n_samples})")
    ax.set_xlabel(r"$N_{\rm token}$")
    ax.set_ylabel("Accuracy / Pass@1 (%)")
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures/pareto.pdf"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    for ax, bm in zip(axes.flat, BENCHMARKS):
        plot_panel(ax, bm)

    # Shared legend from first panel
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
