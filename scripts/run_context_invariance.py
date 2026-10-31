"""Context invariance test for the t=1 length estimator.

For each prompt, run the t=1 forward pass at multiple canvas sizes and compare
the predicted crop lengths. A context-invariant estimator should give the same
prediction regardless of how many mask tokens are appended.

Usage:
    python scripts/run_context_invariance.py --model GSAI-ML/LLaDA-8B-Instruct \\
        --tasks gsm8k math500 humaneval mbpp --n 200
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time

import numpy as np
import torch
import dllm

from diffusion_llms.generators.cdf_remasking_generator import _predict_lengths_from_t1

PROJECT_ROOT  = Path(__file__).parent.parent
DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
CONTEXT_SIZES = [256, 512, 1024, 2048, 4096]
SUPPORTED_TASKS = ["gsm8k", "math500", "humaneval", "mbpp"]


def _npz_key(doc_id: str) -> str:
    """Sanitize doc_id for use as a numpy .npz array key (no slashes/dots)."""
    return doc_id.replace("/", "_").replace(".", "_")


def _is_dream(model_name: str) -> bool:
    return "dream" in model_name.lower()


def _is_llada15(model_name: str) -> bool:
    return "1.5" in model_name.lower()


def load_examples(task, tokenizer, n, seed=42, assistant_prefix=""):
    from lm_eval.tasks import TaskManager
    import lm_eval.tasks as lm_tasks

    task_manager = TaskManager(
        include_path=str(PROJECT_ROOT / "diffusion_llms" / "tasks")
    )
    task_dict  = lm_tasks.get_task_dict([task], task_manager)
    lm_task    = task_dict[task]
    docs       = list(lm_task.test_docs())
    docs       = sorted(docs, key=lambda d: d.get("task_id", d.get("idx", 0)))[:n]

    examples = []
    for i, doc in enumerate(docs):
        raw_text  = lm_task.doc_to_text(doc)
        messages  = [{"role": "user", "content": raw_text}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if assistant_prefix:
            templated += assistant_prefix
        ids    = tokenizer.encode(templated, add_special_tokens=False)
        doc_id = str(doc.get("task_id", doc.get("idx", i)))
        examples.append({"prompt_ids": ids, "doc_id": doc_id})

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default=DEFAULT_MODEL)
    parser.add_argument("--tasks",  type=str, nargs="+", default=SUPPORTED_TASKS)
    parser.add_argument("--n",      type=int, default=200)
    parser.add_argument("--q",      type=float, default=0.90)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None, dest="out_dir")
    parser.add_argument("--right-shift-logits", action="store_true", default=False,
                        dest="right_shift_logits")
    args = parser.parse_args()

    if _is_dream(args.model) and not args.right_shift_logits:
        args.right_shift_logits = True

    model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    if args.out_dir is None:
        args.out_dir = Path(f"results/context_invariance/{model_short}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}  Model: {args.model}")

    # ── load model (mirrors run_benchmark.py) ────────────────────────────────
    if _is_dream(args.model):
        from diffusion_llms.utils import get_model as _get_model, get_tokenizer as _get_tokenizer
        model     = _get_model(model_name_or_path=args.model, dtype=torch.bfloat16,
                               device_map="auto", trust_remote_code=True).eval()
        tokenizer = _get_tokenizer(model_name_or_path=args.model, trust_remote_code=True)
        _orig_fwd = model.forward
        def _dream_fwd_patch(*a, **kw):
            mask = kw.get("attention_mask", a[1] if len(a) > 1 else None)
            if isinstance(mask, torch.Tensor) and mask.dim() == 2:
                mask = torch.logical_and(
                    mask.unsqueeze(1).unsqueeze(-2),
                    mask.unsqueeze(1).unsqueeze(-1),
                ).to(torch.bool)
                if "attention_mask" in kw:
                    kw["attention_mask"] = mask
                else:
                    a = list(a); a[1] = mask; a = tuple(a)
            return _orig_fwd(*a, **kw)
        model.forward = _dream_fwd_patch
    else:
        import dataclasses
        @dataclasses.dataclass
        class _MA:
            model_name_or_path: str = DEFAULT_MODEL
        model     = dllm.utils.get_model(model_args=_MA(args.model)).eval().to(device)
        tokenizer = dllm.utils.get_tokenizer(model_args=_MA(args.model))

    mask_id = tokenizer.mask_token_id
    eos_id  = tokenizer.eos_token_id

    # ── run each task ─────────────────────────────────────────────────────────
    for task in args.tasks:
        out_path = args.out_dir / f"{task}.json"

        # resume — only skip samples that have ALL context sizes already done
        records: list = []
        records_by_id: dict = {}
        if out_path.exists():
            records = json.loads(out_path.read_text())
            records_by_id = {r["doc_id"]: r for r in records}
            done_ids = {
                r["doc_id"] for r in records
                if all(r["preds"].get(str(C)) is not None for C in CONTEXT_SIZES)
            }
            print(f"[{task}] resuming — {len(done_ids)}/{len(records)} fully done")
        else:
            done_ids = set()

        assistant_prefix = "" if task in ("humaneval", "mbpp") else "<reasoning>"
        examples = load_examples(task, tokenizer, args.n, seed=args.seed,
                                 assistant_prefix=assistant_prefix)
        pending  = [ex for ex in examples if ex["doc_id"] not in done_ids]
        print(f"[{task}] {len(pending)} samples × {len(CONTEXT_SIZES)} context sizes")

        probs_dir = args.out_dir / "probs" / task
        probs_dir.mkdir(parents=True, exist_ok=True)

        for ex in pending:
            prompt = torch.as_tensor(ex["prompt_ids"], dtype=torch.long,
                                     device=model.device)
            prompt_len = len(ex["prompt_ids"])

            # start from existing preds if this sample was partially done
            existing = records_by_id.get(ex["doc_id"])
            preds = dict(existing["preds"]) if existing else {}

            for C in CONTEXT_SIZES:
                if str(C) in preds:  # already computed for this context size
                    continue
                max_new = C - prompt_len
                if max_new <= 0:
                    preds[str(C)] = None
                    continue
                t0 = time.time()
                (length,), (probs,) = _predict_lengths_from_t1(
                    model, [prompt], mask_id, eos_id,
                    max_new_tokens=max_new,
                    quantile=args.q,
                    right_shift_logits=args.right_shift_logits,
                    return_probs=True,
                )
                elapsed = round(time.time() - t0, 3)
                preds[str(C)] = {"pred_new": length, "elapsed": elapsed}
                npz_path = probs_dir / f"{_npz_key(ex['doc_id'])}_C{C}.npz"
                np.savez_compressed(str(npz_path), probs=probs)

            if existing:
                existing["preds"] = preds
            else:
                rec = {
                    "doc_id":     ex["doc_id"],
                    "npz_key":    _npz_key(ex["doc_id"]),
                    "prompt_len": prompt_len,
                    "task":       task,
                    "model":      args.model,
                    "q":          args.q,
                    "preds":      preds,
                }
                records.append(rec)
                records_by_id[ex["doc_id"]] = rec
            out_path.write_text(json.dumps(records, indent=2))

        # summary
        valid = [r for r in records if all(r["preds"].get(str(C)) is not None for C in CONTEXT_SIZES)]
        if valid:
            sizes = [str(C) for C in CONTEXT_SIZES]
            mat   = np.array([[r["preds"][s]["pred_new"] for s in sizes] for r in valid])
            stds  = mat.std(axis=1)
            print(f"[{task}] n={len(valid)}  mean_pred={mat[:,2].mean():.0f}  "
                  f"std_across_sizes={stds.mean():.1f}  max_std={stds.max():.1f}")

    print(f"\nDone. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
