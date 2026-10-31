"""Backfill the `correct` field on a run_benchmark.py results JSON for a code
task (mbpp/humaneval) that was never scored server-side. Needs code execution
(HF `evaluate`'s code_eval), so must run somewhere with that environment set
up and HF_ALLOW_CODE_EVAL permitted. This executes model-written code, so run
it in a sandboxed job. Reuses metrics/{mbpp,humaneval}.score_batch from this repo.

Usage:
    python scripts/backfill_code_correct.py --task humaneval \
        --path results/humaneval/some_run.json \
        --repo-root .
"""
import argparse
import json
import os
import sys


def backfill(task, path, repo_root):
    sys.path.insert(0, repo_root)
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    if task == "mbpp":
        from metrics.mbpp import score_batch
    elif task == "humaneval":
        from metrics.humaneval import score_batch
    else:
        raise ValueError(f"unsupported task: {task}")

    d = json.load(open(path))
    corrects = score_batch(d)
    for r, c in zip(d, corrects):
        r["correct"] = bool(c)
    n_correct = sum(corrects)
    print(f"{path}: n={len(d)} acc={n_correct / len(d):.4f} ({n_correct}/{len(d)})")
    with open(path, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print("wrote scored results back to", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["mbpp", "humaneval"], required=True)
    p.add_argument("--path", required=True, help="results JSON to backfill in place")
    p.add_argument("--repo-root", default=".",
                    help="repo root, for importing metrics/*.py")
    args = p.parse_args()
    backfill(args.task, args.path, args.repo_root)
