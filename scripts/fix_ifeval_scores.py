#!/usr/bin/env python3
"""
Re-evaluate IFEval step analysis results with the corrected checker.

The original run called INSTRUCTION_DICT[i](instruction_id=i, **kwargs) but
the kwargs belong in build_description(**kwargs), not __init__. This caused
check_following() to have no constraints → always returned True (or always 0
due to empty instructions list). All scores were 0.

This script reloads the saved predictions, reconstructs the docs in the same
shuffle order (SEED=42), re-evaluates with the fixed checker, and overwrites
generation_results.csv in-place.

Usage:
    python scripts/fix_ifeval_scores.py <results_dir>
    python scripts/fix_ifeval_scores.py results/rebuttal/step_analysis_ifeval_GSAI-ML_LLaDA-8B-Instruct_476867
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42


def ifeval_correct_fixed(prediction: str, doc: dict) -> int:
    import inspect
    from lm_eval.tasks.ifeval.instructions_registry import INSTRUCTION_DICT
    instructions = []
    for i, k in zip(doc["instruction_id_list"], doc["kwargs"]):
        if i not in INSTRUCTION_DICT:
            continue
        instr = INSTRUCTION_DICT[i](instruction_id=i)
        # Filter kwargs to only those accepted by this checker's build_description
        sig = inspect.signature(instr.build_description)
        valid = set(sig.parameters.keys()) - {"self"}
        filtered_k = {key: val for key, val in k.items() if key in valid}
        try:
            instr.build_description(**filtered_k)
        except Exception:
            continue
        instructions.append(instr)
    if not instructions:
        return 0
    return int(all(instr.check_following(prediction) for instr in instructions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    gen_csv = out_dir / "generation_results.csv"

    if not gen_csv.exists():
        print(f"No generation_results.csv in {out_dir}")
        sys.exit(1)

    # Load existing rows
    with open(gen_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    # Reconstruct IFEval docs in same order as original run
    from datasets import load_dataset
    dataset = load_dataset("google/IFEval", split="train")
    docs = list(dataset)
    rng = random.Random(SEED)
    rng.shuffle(docs)
    n_examples = len(set(r["example_id"] for r in rows))
    docs = docs[:n_examples]

    print(f"Re-evaluating {n_examples} examples × {len(rows)//n_examples} checkpoints")

    # Re-score
    fixed = 0
    total = 0
    for row in rows:
        ex_id = int(row["example_id"])
        doc = docs[ex_id]
        new_correct = ifeval_correct_fixed(row["prediction"], doc)
        if int(row["correct"]) != new_correct:
            fixed += 1
        row["correct"] = str(new_correct)
        total += 1

    print(f"Changed {fixed}/{total} scores")

    # Group by checkpoint for accuracy summary
    from collections import defaultdict
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["checkpoint_label"]].append(int(row["correct"]))

    print("\nCorrected accuracy:")
    for label in sorted(by_label):
        scores = by_label[label]
        print(f"  {label:<35s}  acc={sum(scores)/len(scores):.4f}  ({sum(scores)}/{len(scores)})")

    # Overwrite CSV
    with open(gen_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOverwrote {gen_csv}")

    # Update summary if it exists
    summary_path = out_dir / "step_analysis_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        summary["ifeval_scores_fixed"] = True
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
