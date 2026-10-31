#!/usr/bin/env python3
"""SmartCrop method diagram.

Usage:
    python scripts/plot_method.py --out figures/method.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Palette ────────────────────────────────────────────────────────────────────
BG        = "#FFFFFF"
PROMPT_FC = "#D5E8F5"
PROMPT_EC = "#2166AC"
MASK_FC   = "#E8D5F5"
MASK_EC   = "#9970AB"
CROP_EC   = "#D6604D"
GREEN     = "#4DAC26"
THRESHOLD = "#D6604D"

# ── Data ───────────────────────────────────────────────────────────────────────
PROMPT_TOKENS = ["What", "is", "the", "capital", "of", "France", "?"]
N_PROMPT      = len(PROMPT_TOKENS)
N_MASK        = 10
N_TOTAL       = N_PROMPT + N_MASK

_P_EOS_RAW = np.array([
    0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.03,        # prompt (7)
    0.08, 0.15, 0.25, 0.35, 0.40, 0.65, 0.20, 0.08, 0.03, 0.01,  # mask (10)
])
CDF = 1.0 - np.cumprod(1.0 - _P_EOS_RAW)
THRESHOLD_Q = 0.90

# First position where CDF >= q is the first token we DON'T need — canvas ends before it
CROP_IDX    = int(np.argmax(CDF >= THRESHOLD_Q))   # = 12; canvas = [0, CROP_IDX)
N_CROP_MASK = CROP_IDX - N_PROMPT                  # = 5 mask tokens in canvas
BLOCK_SIZE  = 2
REVEALED    = ["Paris", "is", "the", "capital", "France"]


# ── Drawing helpers ────────────────────────────────────────────────────────────
def token_box(ax, col, row_y, label, fc, ec, step, tok_w=1.0, tok_h=0.68, fontsize=11):
    x = col * step
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, row_y), tok_w, tok_h,
        boxstyle="round,pad=0.05",
        fc=fc, ec=ec, lw=1.5, zorder=3,
    ))
    ax.text(x + tok_w / 2, row_y + tok_h / 2, label,
            ha="center", va="center", fontsize=fontsize, zorder=4,
            color=ec if label != "[MASK]" else MASK_EC)


def draw_denoising_row(ax, row_y, n_revealed_blocks, step, tok_w, tok_h):
    for i in range(CROP_IDX):  # canvas = [0, CROP_IDX)
        col_x = i * step
        if i < N_PROMPT:
            fc, ec, lbl = PROMPT_FC, PROMPT_EC, PROMPT_TOKENS[i]
        else:
            mask_i  = i - N_PROMPT
            block_i = mask_i // BLOCK_SIZE
            if block_i < n_revealed_blocks:
                lbl    = REVEALED[mask_i] if mask_i < len(REVEALED) else "..."
                fc, ec = "#D5F0E0", GREEN
            else:
                lbl, fc, ec = "[MASK]", MASK_FC, MASK_EC
        ax.add_patch(mpatches.FancyBboxPatch(
            (col_x, row_y), tok_w, tok_h,
            boxstyle="round,pad=0.05",
            fc=fc, ec=ec, lw=1.5, zorder=3,
        ))
        ax.text(col_x + tok_w / 2, row_y + tok_h / 2, lbl,
                ha="center", va="center", fontsize=10, zorder=4, color=ec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures/method.pdf"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    TOK_W = 1.0
    GAP   = 0.18
    STEP  = TOK_W + GAP
    TOK_H = 0.68

    # denoising row positions
    row0_y = 2.0
    row1_y = 0.75
    rowN_y = -1.10

    ylim_top    = 5.8
    ylim_bottom = rowN_y - 0.55
    y_range     = ylim_top - ylim_bottom

    total_w = N_TOTAL * STEP
    fig_w   = total_w + 3.0
    fig_h   = y_range * 0.95

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.6, total_w + 0.6)
    ax.set_ylim(ylim_bottom, ylim_top)
    ax.axis("off")

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1 — t=1 probe: tokens + CDF(EOS) bars
    # ══════════════════════════════════════════════════════════════════════════
    ax.text(-0.3, 5.65, "t = 1  (fully masked canvas — one forward pass)",
            fontsize=12, fontweight="bold", va="bottom", color="#222222")

    BAR_BASE  = 3.8
    BAR_MAX_H = 1.4
    TOK_ROW   = 3.0

    # shaded region for discarded tokens — starts at CROP_IDX (inclusive)
    excl_x = CROP_IDX * STEP
    ax.add_patch(mpatches.FancyBboxPatch(
        (excl_x - GAP / 2, TOK_ROW - 0.12), total_w - excl_x + GAP / 2 + 0.1, BAR_MAX_H + 0.9,
        boxstyle="square,pad=0.0", fc="#F0F0F0", ec="none", zorder=1, alpha=0.8,
    ))
    ax.text((excl_x + total_w) / 2, BAR_BASE + BAR_MAX_H + 0.15,
            "discarded", ha="center", va="bottom", fontsize=10, color="#AAAAAA")

    for i in range(N_TOTAL):
        x_left   = i * STEP
        x_center = x_left + TOK_W / 2

        bar_h  = CDF[i] * BAR_MAX_H
        if i >= CROP_IDX:
            fc_bar = "#DDDDDD"
        else:
            fc_bar = "#AACCE8"
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_left, BAR_BASE), TOK_W, max(bar_h, 0.01),
            boxstyle="round,pad=0.02", fc=fc_bar, ec="none", zorder=2,
        ))
        if CDF[i] >= 0.10:
            ax.text(x_center, BAR_BASE + bar_h + 0.07,
                    f"{CDF[i]:.2f}", ha="center", va="bottom",
                    fontsize=9, color="#888888" if i >= CROP_IDX else "#444444")
        if i < N_PROMPT:
            token_box(ax, i, TOK_ROW, PROMPT_TOKENS[i], PROMPT_FC, PROMPT_EC, STEP)
        else:
            token_box(ax, i, TOK_ROW, "[MASK]", MASK_FC, MASK_EC, STEP)

    # q threshold — dashed line only in the discarded region
    thresh_y = BAR_BASE + THRESHOLD_Q * BAR_MAX_H
    excl_start_x = CROP_IDX * STEP
    ax.plot([excl_start_x - GAP / 2, total_w + 0.1], [thresh_y, thresh_y],
            color=THRESHOLD, lw=1.5, ls="--", zorder=5)
    ax.text(total_w + 0.25, thresh_y, f"q = {THRESHOLD_Q}",
            ha="left", va="center", fontsize=11, color=THRESHOLD, fontweight="bold")

    ax.text(-0.45, BAR_BASE + BAR_MAX_H / 2, "CDF(EOS)",
            ha="center", va="center", fontsize=11, rotation=90, color="#555555")

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2 — block denoising: step 0 / step 1 / ... / step N
    # ══════════════════════════════════════════════════════════════════════════


    label_x  = CROP_IDX * STEP + 0.30
    center_x = CROP_IDX * STEP / 2
    arrow_kw = dict(arrowstyle="-|>", color="#888888", lw=1.4)

    # step 0: all masked
    draw_denoising_row(ax, row0_y, n_revealed_blocks=0, step=STEP, tok_w=TOK_W, tok_h=TOK_H)
    ax.text(label_x, row0_y + TOK_H / 2, "step 0",
            va="center", fontsize=11, fontweight="bold", color="#555555")

    # arrow row 0 → row 1
    ax.annotate("", xy=(center_x, row1_y + TOK_H + 0.06),
                xytext=(center_x, row0_y - 0.06), arrowprops=arrow_kw)

    # step 1: block 0 revealed
    draw_denoising_row(ax, row1_y, n_revealed_blocks=1, step=STEP, tok_w=TOK_W, tok_h=TOK_H)
    ax.text(label_x, row1_y + TOK_H / 2, "step 1",
            va="center", fontsize=11, fontweight="bold", color="#555555")

    # two arrows + dots between step 1 and step N
    dots_y = (row1_y + rowN_y + TOK_H) / 2
    ax.annotate("", xy=(center_x, dots_y + 0.18),
                xytext=(center_x, row1_y - 0.06), arrowprops=arrow_kw)
    ax.text(center_x, dots_y, "· · ·",
            ha="center", va="center", fontsize=20, color="#999999", zorder=4)
    ax.annotate("", xy=(center_x, rowN_y + TOK_H + 0.06),
                xytext=(center_x, dots_y - 0.18), arrowprops=arrow_kw)

    # step N: all revealed
    draw_denoising_row(ax, rowN_y, n_revealed_blocks=N_CROP_MASK, step=STEP, tok_w=TOK_W, tok_h=TOK_H)
    ax.text(label_x, rowN_y + TOK_H / 2, "step N",
            va="center", fontsize=11, fontweight="bold", color="#333333")

    fig.savefig(args.out, bbox_inches="tight", facecolor=BG, dpi=150)
    png = args.out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", facecolor=BG, dpi=150)
    print(f"Saved: {args.out}  {png}")


if __name__ == "__main__":
    main()
