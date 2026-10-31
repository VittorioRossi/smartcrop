#!/usr/bin/env python3
"""Quantile calibration of the t=1 length readout.

For each prompt c, SmartCrop at quantile q records its crop length M_q(c)
(`ntoken`). The fixed-canvas run records the realized response length R(c)
(`etoken`, content tokens before trailing EoS). If the readout is a calibrated
CDF of the boundary B = R + 1, the coverage Pr_c[B <= M_q(c)] should be ~q.

Also reports Spearman rho(M_q, R) across prompts: the prompt-specific part of
the signal that a task-mean canvas cannot capture.

With --retention, also prints the under-allocation table for LLaDA-8B on GSM8K:
prompts with M_q < R + 1, their median deficit, and the share of full-canvas
correct answers kept after cropping, for under-allocated prompts and for the rest.

Usage:
    python scripts/calibration.py [--results-dir results] [--ref 1024] [--retention]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from run_mcnemar import score_smartcrop_gsm8k_math

QS = [50, 75, 90, 95, 99]
TASKS = {"gsm8k": 1319, "math500": 500, "humaneval": 164, "mbpp": None}
MODELS = {"llada_8b_instruct": "LLaDA-8B", "dream_v0_instruct_7b": "DREAM-7B"}

log = logging.getLogger(__name__)


def load(path: Path) -> dict[str, dict]:
    return {str(r["doc_id"]): r for r in json.loads(path.read_text())}


def find(root: Path, task: str, pattern: str) -> Path | None:
    hits = sorted((root / task).glob(pattern))
    hits = [h for h in hits if not h.name.endswith(".stale") and "_n2_" not in h.name and "_n10_" not in h.name]
    return max(hits, key=lambda h: h.stat().st_size) if hits else None


def main(root: Path, ref: int) -> None:
    rows = []
    for model, name in MODELS.items():
        for task in TASKS:
            fixed = find(root, task, f"{model}_low_conf_n*_gl{ref}_s{ref}.json")
            if fixed is None:
                continue
            R = {k: r["etoken"] for k, r in load(fixed).items()}
            for q in QS:
                sc = find(root, task, f"{model}_cdf_q{q}_n*_gl1024_s1024.json")
                if sc is None:
                    continue
                M = {k: r["ntoken"] for k, r in load(sc).items()}
                ids = sorted(set(M) & set(R))
                m = np.array([M[i] for i in ids])
                r = np.array([R[i] for i in ids])
                cov = np.mean(r + 1 <= m)
                rho = spearmanr(m, r).statistic
                slack = np.mean(np.clip(m - r, 0, None))
                rows.append((name, task, q / 100, len(ids), cov, rho, np.median(m), np.median(r), slack))

    print(f"reference: fixed canvas M={ref}")
    print(f"{'model':10s} {'task':10s} {'q':>5s} {'n':>5s} {'cover':>6s} {'rho':>6s} {'med M_q':>8s} {'med R':>6s} {'slack':>6s}")
    for row in rows:
        print("{:10s} {:10s} {:5.2f} {:5d} {:6.3f} {:6.3f} {:8.0f} {:6.0f} {:6.1f}".format(*row))


def retention(root: Path, ref: int) -> None:
    model, task = "llada_8b_instruct", "gsm8k"
    fixed_path = find(root, task, f"{model}_low_conf_n*_gl{ref}_s{ref}.json")
    fixed = load(fixed_path)
    fixed_ok = score_smartcrop_gsm8k_math(fixed_path)
    print(f"{'q':>5s} {'under':>6s} {'deficit':>8s} {'kept':>6s} {'kept (room)':>12s}")
    for q in QS[:-1]:
        crop_path = find(root, task, f"{model}_cdf_q{q}_n*_gl1024_s1024.json")
        crop = load(crop_path)
        crop_ok = score_smartcrop_gsm8k_math(crop_path)
        # Prompts that never emit EoS on the full canvas have no known length.
        ids = sorted(i for i in set(crop) & set(fixed) if fixed[i]["etoken"] < ref)
        need = {i: fixed[i]["etoken"] + 1 for i in ids}
        under = [i for i in ids if crop[i]["ntoken"] < need[i]]
        room = [i for i in ids if crop[i]["ntoken"] >= need[i]]
        deficit = np.median([1 - crop[i]["ntoken"] / need[i] for i in under])

        def kept(group: list[str]) -> float:
            base = [int(i) for i in group if fixed_ok[int(i)]]
            return np.mean([crop_ok[i] for i in base])

        print(f"{q / 100:5.2f} {len(under):6d} {deficit:8.0%} {kept(under):6.0%} {kept(room):12.0%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--ref", type=int, default=1024)
    ap.add_argument("--retention", action="store_true")
    args = ap.parse_args()
    main(args.results_dir, args.ref)
    if args.retention:
        retention(args.results_dir, args.ref)
