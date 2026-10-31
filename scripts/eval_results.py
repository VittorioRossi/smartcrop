"""Evaluate result JSON files using the canonical metrics from metrics/.

Usage:
    python scripts/eval_results.py results/math500/llada_8b_instruct_cdf_q90_n500_gl1024_s1024.json
    python scripts/eval_results.py results/gsm8k/llada_8b_instruct_cdf_q90_n1319*.json
    python scripts/eval_results.py results/humaneval/*.json   # requires HF_ALLOW_CODE_EVAL=1
    python scripts/eval_results.py results/mbpp/*.json

Math scoring uses is_equiv() with LaTeX normalization from metrics/math500.py.
Code scoring uses sanitize() + code_eval from metrics/humaneval.py / metrics/mbpp.py.
"""

import sys
import json
import os
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.math500 import score_batch as _math_score_batch
from metrics.humaneval import score_batch as _he_score_batch, sanitize as he_sanitize
from metrics.mbpp import score_batch as _mbpp_score_batch, sanitize as mbpp_sanitize, extract_python_code, find_entry_point


def summarize_efficiency(data):
    etokens = [r["etoken"] for r in data]
    ntokens = [r["ntoken"] for r in data]
    eratios = [r["etoken"] / r["ntoken"] for r in data if r["ntoken"] > 0]
    return np.mean(etokens), np.mean(ntokens), np.mean(eratios)


def eval_math(path: Path):
    data = json.loads(path.read_text())
    n = len(data)
    nc = sum(_math_score_batch(data))
    et, nt, er = summarize_efficiency(data)
    print(f"{path.name}")
    print(f"  n={n}  Acc={100*nc/n:.1f}%  Etoken={et:.1f}  Ntoken={nt:.1f}  Eratio={er:.3f}")


def eval_humaneval(path: Path):
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    import evaluate as hf_evaluate
    data = json.loads(path.read_text())
    n = len(data)

    refs, preds = [], []
    for r in data:
        raw = r["response"]
        prompt = r["gt"]["prompt"]
        entry_point = r["gt"]["entry_point"]
        test = r["gt"]["test"]
        code_block = raw.split("```python\n", 1)[-1].split("```")[0]
        full = prompt + "\n" + code_block
        sanitized = he_sanitize(full, entry_point)
        refs.append(test + f"\ncheck({entry_point})")
        preds.append([sanitized])

    code_eval = hf_evaluate.load("code_eval")
    results, _ = code_eval.compute(references=refs, predictions=preds, k=[1], timeout=10)
    et, nt, er = summarize_efficiency(data)
    print(f"{path.name}")
    print(f"  n={n}  Pass@1={results['pass@1']*100:.1f}%  Etoken={et:.1f}  Ntoken={nt:.1f}  Eratio={er:.3f}")


def eval_mbpp(path: Path):
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    import evaluate as hf_evaluate
    data = json.loads(path.read_text())

    refs, preds = [], []
    skipped = 0
    for r in data:
        raw = r["response"]
        test_list = r["gt"]["test_list"]
        code = extract_python_code(raw)
        if not code:
            skipped += 1
            continue
        ep = find_entry_point(code)
        if not ep:
            skipped += 1
            continue
        sanitized = mbpp_sanitize(code, entrypoint=ep)
        refs.append("\n".join(test_list))
        preds.append([sanitized])

    n = len(data)
    code_eval = hf_evaluate.load("code_eval")
    results, _ = code_eval.compute(references=refs, predictions=preds, k=[1], timeout=10)
    pass1 = results["pass@1"] * len(preds) / n
    et, nt, er = summarize_efficiency(data)
    print(f"{path.name}")
    print(f"  n={n}  Pass@1={pass1*100:.1f}%  (skipped={skipped})  Etoken={et:.1f}  Ntoken={nt:.1f}  Eratio={er:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    for path in sorted(args.files):
        if not path.exists():
            print(f"{path.name}: MISSING")
            continue
        data = json.loads(path.read_text())
        if not data:
            print(f"{path.name}: EMPTY")
            continue
        task = data[0].get("task", "")
        if task in ("gsm8k", "math500"):
            eval_math(path)
        elif task == "humaneval":
            eval_humaneval(path)
        elif task == "mbpp":
            eval_mbpp(path)
        else:
            print(f"{path.name}: unknown task '{task}'")


if __name__ == "__main__":
    main()
