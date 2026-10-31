"""E1 analysis — does the t=1 instance-level readout track realized length?

Loads results/e1_pilot/{task}_n{N}_k{K}.json (one row per prompt: pred_len_q90
from a single t=1 forward pass, plus K stochastic realized completion lengths
b_ks). For each prompt, mean(b_ks) is the best available estimate of the true
prompt-conditional expected length.

Three questions, following the H1/H2/H3 ladder:
  1. Within-prompt noise floor: distribution of CV(B_k) = std(b_ks)/mean(b_ks)
     across prompts. If this is large, there's a real aleatoric ceiling on
     how well any single-shot readout could ever track individual samples.
  2. Does pred_len_q90 explain across-prompt variance in mean(b_ks)? Pearson
     r and R^2 of pred_len_q90 vs mean(b_ks).
  3. Does the instance-level readout beat the naive task-mean-canvas baseline
     (same predicted length for every prompt) at predicting mean(b_ks)? This
     is the actual test behind Table 5's n.s. instance-vs-task-mean result:
     if pred_len_q90 has near-zero skill over the task mean, that supports H1
     (the accuracy table found no difference because there's nothing to find,
     not because the model failed to compute or expose it).

Usage:
    python scripts/analyze_e1.py results/e1_pilot/gsm8k_n200_k16.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    args = parser.parse_args()

    with open(args.json_path) as f:
        results = json.load(f)

    n = len(results)
    pred = np.array([r["pred_len_q90"] for r in results], dtype=float)
    means = np.array([np.mean(r["b_ks"]) for r in results], dtype=float)
    stds = np.array([np.std(r["b_ks"]) for r in results], dtype=float)
    q90s = np.array([np.percentile(r["b_ks"], 90) for r in results], dtype=float)
    cvs = np.array([s / m for s, m in zip(stds, means) if m > 0])

    print(f"Prompts: {n}, K per prompt: {len(results[0]['b_ks'])}")
    print()

    # ── 1. Within-prompt noise floor ────────────────────────────────────────
    print("── Within-prompt noise (CV = std(B_k)/mean(B_k)) ──")
    print(f"  mean CV:   {cvs.mean():.3f}")
    print(f"  median CV: {np.median(cvs):.3f}")
    print(f"  p90 CV:    {np.percentile(cvs, 90):.3f}")
    print(f"  max CV:    {cvs.max():.3f}  (doc={results[int(np.argmax([s/m for s,m in zip(stds,means) if m>0]))]['doc_id']})")
    print()

    # ── 2. Instance-level readout vs realized mean length ──────────────────
    r = np.corrcoef(pred, means)[0, 1]
    ss_res = np.sum((means - pred) ** 2)
    ss_tot = np.sum((means - means.mean()) ** 2)
    r2_readout = 1 - ss_res / ss_tot
    mae_readout = np.mean(np.abs(means - pred))

    print("── Instance-level readout (pred_len_q90) vs mean(B_k) ──")
    print(f"  Pearson r: {r:.3f}")
    print(f"  R^2 (readout as direct predictor): {r2_readout:.3f}")
    print(f"  MAE: {mae_readout:.1f} tokens")
    print()

    # Affine recalibration: fit mean(B_k) ~ a + b*pred_len_q90. If this R^2
    # is much higher than the direct-predictor R^2 above, the raw signal is
    # present and rank-correlated but miscalibrated in scale/offset -- a
    # readout-gap (H2) signature, not a computation gap (H3) one.
    b_coef, a_coef = np.polyfit(pred, means, 1)
    fitted = a_coef + b_coef * pred
    ss_res_fit = np.sum((means - fitted) ** 2)
    r2_fit = 1 - ss_res_fit / ss_tot
    mae_fit = np.mean(np.abs(means - fitted))

    print("── Same readout, affine-recalibrated (mean(B_k) ~ a + b*pred) ──")
    print(f"  fit: mean(B_k) = {a_coef:.1f} + {b_coef:.3f} * pred_len_q90")
    print(f"  R^2 (in-sample, recalibrated): {r2_fit:.3f}")
    print(f"  MAE (in-sample, recalibrated): {mae_fit:.1f} tokens")
    print()

    # ── 2b. Same readout vs the matching statistic: q90(B_k), not mean(B_k) ─
    # pred_len_q90 targets a 90th-percentile boundary crossing, not the mean
    # of the sampled lengths -- q90(b_ks) (K=16 samples, so this is a coarse
    # empirical estimate) is the theoretically correct comparison point.
    r_q90 = np.corrcoef(pred, q90s)[0, 1]
    ss_tot_q90 = np.sum((q90s - q90s.mean()) ** 2)
    ss_res_q90 = np.sum((q90s - pred) ** 2)
    r2_direct_q90 = 1 - ss_res_q90 / ss_tot_q90
    mae_direct_q90 = np.mean(np.abs(q90s - pred))
    mae_taskmean_q90 = np.mean(np.abs(q90s - q90s.mean()))

    print("── Same readout vs the matching statistic: q90(B_k) (K=16 empirical) ──")
    print(f"  Pearson r: {r_q90:.3f}")
    print(f"  R^2 (direct): {r2_direct_q90:.3f}")
    print(f"  MAE (direct): {mae_direct_q90:.1f} tokens")
    print(f"  MAE (task-mean-q90 baseline): {mae_taskmean_q90:.1f} tokens")
    print()

    # ── 3. Readout vs naive task-mean-canvas baseline ───────────────────────
    task_mean = means.mean()
    mae_taskmean = np.mean(np.abs(means - task_mean))
    ss_res_taskmean = np.sum((means - task_mean) ** 2)
    r2_taskmean = 1 - ss_res_taskmean / ss_tot  # ~0 by construction

    print("── Readout vs naive task-mean baseline ──")
    print(f"  Task-mean canvas: {task_mean:.1f} tokens (constant for all prompts)")
    print(f"  MAE (task-mean baseline): {mae_taskmean:.1f} tokens")
    print(f"  MAE (instance readout):   {mae_readout:.1f} tokens")
    print(f"  MAE reduction: {100 * (1 - mae_readout / mae_taskmean):.1f}%")
    print()

    # ── Verdict hint ─────────────────────────────────────────────────────────
    # The mean(B_k) comparisons above are a diagnostic, not the real test:
    # pred_len_q90 targets a 90th-percentile crossing, so the theoretically
    # matched comparison is against q90(b_ks), not mean(b_ks). Weight that one.
    mae_reduction_q90 = 100 * (1 - mae_direct_q90 / mae_taskmean_q90)
    print("── Verdict hint (weighted on the q90-vs-q90 comparison) ──")
    if r2_direct_q90 > 0.15 and mae_reduction_q90 > 15:
        print(f"  Against the theoretically matched target (q90(B_k)), the raw,")
        print(f"  uncalibrated readout already beats the task-mean baseline by")
        print(f"  {mae_reduction_q90:.1f}% MAE (R^2={r2_direct_q90:.3f}, r={r_q90:.3f}).")
        print("  Instance-level signal IS present and IS correctly read at t=1 --")
        print("  this rules out both H1 (aleatoric ceiling: there would be nothing")
        print("  to win) and H3 (computation gap: the encoder clearly resolves")
        print("  instance identity). Table 5's n.s. accuracy result is then NOT a")
        print("  signal problem -- look at whether downstream accuracy is simply")
        print("  insensitive to canvas size in this regime (a decoding-side,")
        print("  not encoding-side, question).")
        print(f"  (Note: the naive mean(B_k)-vs-pred comparison above looked flat")
        print(f"  -- MAE reduction {100*(1-mae_readout/mae_taskmean):.1f}% -- purely")
        print(f"  because it compares a q90-target predictor to the wrong statistic.)")
    elif r2_fit > 0.3 and r2_readout < 0.15:
        print(f"  Raw readout underperforms (R^2={r2_readout:.3f}, MAE {mae_readout:.1f}")
        print(f"  vs task-mean {mae_taskmean:.1f}) but affine recalibration recovers")
        print(f"  R^2={r2_fit:.3f} (r={r:.3f}). The instance-level signal IS present")
        print("  and rank-correlates with true length -- this is H2 (readout gap):")
        print("  the q=0.90 threshold estimator is miscalibrated in scale/offset,")
        print("  not that the encoder fails to compute instance identity (rules out")
        print("  H3) or that there's nothing to compute (rules out H1).")
    elif r2_direct_q90 > 0.05 or r2_readout > 0.05:
        print("  Weak-to-moderate tracking. Partial support for H2 (readout gap):")
        print("  some instance-level signal exists but is diluted/undertrained.")
    else:
        print("  Readout ~uncorrelated with realized per-prompt length under either")
        print("  matching. Either H1 (aleatoric ceiling dominates, task and instance")
        print("  length are the same distribution) or H3 (computation gap, t=1")
        print("  encoder never resolves instance identity) -- the CV distribution")
        print("  above disambiguates: high within-prompt CV relative to across-prompt")
        print("  spread of means favors H1; low within-prompt CV with a flat readout")
        print("  favors H3.")


if __name__ == "__main__":
    main()
