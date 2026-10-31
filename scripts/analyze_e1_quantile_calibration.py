"""Cross-fit a CDF threshold for nominal length coverage from E1 data.

Each E1 prompt has one t=1 EOS-CDF curve and K=16 temperature-1.0 sampled
completion lengths.  This script selects the raw CDF threshold that gives
nominal coverage on four folds and evaluates it on the fifth.  Prompts, not
individual samples, are resampled for bootstrap intervals.

Usage:
    python scripts/analyze_e1_quantile_calibration.py \
        results/e1_pilot/gsm8k_n200_k16.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def predicted_lengths(curves: np.ndarray, quantile: float) -> np.ndarray:
    """Return the first position at which each raw EOS-CDF crosses quantile."""
    hits = curves >= quantile
    return np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, curves.shape[1])


def coverage(curves: np.ndarray, lengths: np.ndarray, quantile: float) -> float:
    predicted = predicted_lengths(curves, quantile)
    return float((lengths <= predicted[:, None]).mean())


def select_quantile(
    curves: np.ndarray,
    lengths: np.ndarray,
    target_coverage: float,
    candidates: np.ndarray,
) -> float:
    observed = np.array([coverage(curves, lengths, q) for q in candidates])
    return float(candidates[np.argmin(np.abs(observed - target_coverage))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    args = parser.parse_args()

    rows = json.loads(args.json_path.read_text())
    curves = np.asarray([row["p_eos_curve"] for row in rows], dtype=float)
    lengths = np.asarray([row["b_ks"] for row in rows], dtype=int)
    candidates = np.linspace(0.50, 0.999, 500)
    rng = np.random.default_rng(args.seed)
    folds = np.array_split(rng.permutation(len(rows)), args.folds)
    all_indices = np.arange(len(rows))
    crossfit_predictions = np.empty(len(rows), dtype=int)

    print(f"Prompts: {len(rows)}; samples/prompt: {lengths.shape[1]}")
    print(f"Target coverage: {args.target_coverage:.3f}\n")
    print("fold  fitted q  held-out coverage  raw q=0.90 coverage")
    print("----  --------  -----------------  -------------------")
    for fold_number, test_indices in enumerate(folds, start=1):
        train_indices = np.setdiff1d(all_indices, test_indices)
        fitted_q = select_quantile(
            curves[train_indices],
            lengths[train_indices],
            args.target_coverage,
            candidates,
        )
        crossfit_predictions[test_indices] = predicted_lengths(
            curves[test_indices], fitted_q
        )
        held_out = coverage(curves[test_indices], lengths[test_indices], fitted_q)
        raw = coverage(curves[test_indices], lengths[test_indices], 0.90)
        print(f"{fold_number:>4}  {fitted_q:>8.3f}  {held_out:>17.3f}  {raw:>19.3f}")

    calibrated_covered = lengths <= crossfit_predictions[:, None]
    raw_covered = lengths <= predicted_lengths(curves, 0.90)[:, None]
    bootstrap = np.empty((args.bootstrap, 2), dtype=float)
    for draw in range(args.bootstrap):
        prompt_indices = rng.integers(0, len(rows), len(rows))
        bootstrap[draw] = (
            calibrated_covered[prompt_indices].mean(),
            raw_covered[prompt_indices].mean(),
        )

    calibrated_interval = np.quantile(bootstrap[:, 0], [0.025, 0.975])
    raw_interval = np.quantile(bootstrap[:, 1], [0.025, 0.975])
    print()
    print(
        "Cross-fit calibrated coverage: "
        f"{calibrated_covered.mean():.3f} "
        f"(95% prompt-bootstrap CI {calibrated_interval[0]:.3f}–{calibrated_interval[1]:.3f})"
    )
    print(
        "Raw q=0.90 coverage:          "
        f"{raw_covered.mean():.3f} "
        f"(95% prompt-bootstrap CI {raw_interval[0]:.3f}–{raw_interval[1]:.3f})"
    )


if __name__ == "__main__":
    main()
