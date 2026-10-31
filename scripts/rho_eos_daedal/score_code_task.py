"""Per-example pass/fail scorer for a dllm_eval samples_{task}_*.jsonl file
(mbpp or humaneval), for methods run through the yjyddq/rho-EOS reference
harness (DAEDAL, rho-EOS Sym/Asym). The harness's own metrics/{task}.py
scripts only ever print an aggregate pass@1 to the log and discard the
per-example verdicts that HF's code_eval.compute() already returns
internally -- this recovers them, keyed by the dataset's own task_id, for
paired significance testing (McNemar) against SmartCrop/Fixed-Length.

Needs code execution (HF `evaluate`'s code_eval, HF_ALLOW_CODE_EVAL=1) so
must run somewhere with that environment already set up. It executes
model-written code, so use a sandboxed CPU job. No GPU needed.

Usage:
    python score_code_task.py --task mbpp \
        --samples /path/to/samples_mbpp_*.jsonl \
        --out daedal_mbpp.json \
        --repo-root .
"""
import argparse
import json
import os
import sys


def score_mbpp(samples_path, out_path, repo_root):
    sys.path.insert(0, repo_root)
    import evaluate as hf_evaluate
    from metrics.mbpp import extract_python_code, find_entry_point, sanitize as mbpp_sanitize

    doc_ids, preds, refs, failed = [], [], [], []
    with open(samples_path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task_id = item["doc"]["task_id"]
            raw = item["resps"][0][0]
            ref = item["target"]
            code = extract_python_code(raw)
            if not code:
                failed.append(task_id)
                continue
            ep = find_entry_point(code)
            if not ep:
                failed.append(task_id)
                continue
            sanitized = mbpp_sanitize(code, entrypoint=ep)
            doc_ids.append(task_id)
            preds.append([sanitized])
            refs.append(ref)

    code_eval = hf_evaluate.load("code_eval")
    _, results = code_eval.compute(
        references=refs, predictions=preds, k=[1],
        num_workers=max(1, os.cpu_count() // 2),
    )
    out = {}
    for test_id, task_id in enumerate(doc_ids):
        out[str(task_id)] = bool(results[test_id][0][1]["passed"])
    for task_id in failed:
        out[str(task_id)] = False

    json.dump(out, open(out_path, "w"), indent=2)
    n = len(out)
    acc = sum(out.values()) / n
    print(f"MBPP {samples_path}: n={n} acc={acc:.4f} (extraction/entrypoint failures: {len(failed)})")


def score_humaneval(samples_path, out_path, repo_root):
    sys.path.insert(0, repo_root)
    import evaluate as hf_evaluate
    from metrics.humaneval import sanitize as he_sanitize

    doc_ids, preds, refs = [], [], []
    with open(samples_path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task_id = item["doc"]["task_id"]
            raw = item["resps"][0][0]
            prompt = item["doc"]["prompt"]
            entry_point = item["doc"]["entry_point"]
            ref = item["target"]
            code_to_sanitize = raw.split("```python\n", 1)[-1].split("```")[0]
            full_text = prompt + "\n" + code_to_sanitize
            sanitized = he_sanitize(full_text, entry_point)
            doc_ids.append(task_id)
            preds.append([sanitized])
            refs.append(ref)

    code_eval = hf_evaluate.load("code_eval")
    _, results = code_eval.compute(
        references=refs, predictions=preds, k=[1],
        num_workers=max(1, os.cpu_count() // 2),
    )
    out = {}
    for test_id, task_id in enumerate(doc_ids):
        out[str(task_id)] = bool(results[test_id][0][1]["passed"])

    json.dump(out, open(out_path, "w"), indent=2)
    n = len(out)
    acc = sum(out.values()) / n
    print(f"HumanEval {samples_path}: n={n} acc={acc:.4f}")


if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["mbpp", "humaneval"], required=True)
    p.add_argument("--samples", required=True, help="samples_{task}_*.jsonl path")
    p.add_argument("--out", required=True, help="output {task_id: bool} JSON path")
    p.add_argument("--repo-root", default=".",
                    help="yjyddq/rho-EOS repo root, for importing metrics/*.py")
    args = p.parse_args()
    if args.task == "mbpp":
        score_mbpp(args.samples, args.out, args.repo_root)
    else:
        score_humaneval(args.samples, args.out, args.repo_root)
