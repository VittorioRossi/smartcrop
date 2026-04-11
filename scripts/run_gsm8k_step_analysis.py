#!/usr/bin/env python3
"""
SmartCrop Survival-Function Sweep — GSM8K / HumanEval / MBPP / IFEval / MATH500

For each canvas_size, run ONE full FC generation (steps=canvas_size, one token
unmasked per step). A step_callback computes the EOS survival function and
records the resulting predicted length every RECORD_EVERY callback fires.
No logit tensors are stored — the CDF is computed and discarded inline.

Fresh generation + accuracy evaluation is run at 4 checkpoints:
  k=0    (all-masked SmartCrop baseline)
  k=1%   of canvas steps
  k=5%   of canvas steps
  stable_median: median of predicted_len over the "stable region"
                 (drop first 10 steps as transient, last 5% as collapse zone)

Outputs:
  survival_curve_raw.csv  — (example_id, step, frac, predicted_len) for every
                             RECORD_EVERY steps of every example
  generation_results.csv  — (example_id, checkpoint_k, checkpoint_label,
                             predicted_len, stable_median_len, correct,
                             prediction, gold)  for the 4 checkpoints
  survival_arrays_canvas{N}.npz  — full S(t) arrays per step/example

Usage:
    python scripts/run_gsm8k_step_analysis.py --task gsm8k
    python scripts/run_gsm8k_step_analysis.py --task humaneval
    python scripts/run_gsm8k_step_analysis.py --task mbpp
    python scripts/run_gsm8k_step_analysis.py --task ifeval
    python scripts/run_gsm8k_step_analysis.py --task math500
    sbatch scripts/slurm/run_step_analysis.slurm
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer
from diffusion_llms.utils.model_utils import get_model
from diffusion_llms.generators.base_length_generator import (
    BaseLengthGenerator,
    BaseLengthGeneratorConfig,
)
from diffusion_llms.utils.common import ensure_token_id

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
CANVAS_SIZES  = [2048]
RECORD_EVERY  = 5         # record survival function every N callback fires
EVAL_FRACS    = [0.0, 0.01, 0.05]   # fraction-based eval checkpoints
EOS_QUANTILE  = 0.9
N_EXAMPLES    = 50
SEED          = 42


# ---------------------------------------------------------------------------
# GSM8K helpers
# ---------------------------------------------------------------------------
ANS_RE = re.compile(r"-?[\d,]*\.?\d+")


def extract_last_number(text: str) -> str | None:
    matches = ANS_RE.findall(text.replace(",", ""))
    return matches[-1] if matches else None


def flexible_match(prediction: str, gold: str) -> bool:
    p, g = extract_last_number(prediction), extract_last_number(gold)
    if p is None or g is None:
        return False
    try:
        return float(p) == float(g)
    except ValueError:
        return False


def load_gsm8k_examples(n: int, seed: int, tokenizer, device):
    """Return list of (prompt_ids, gold_str, doc_or_None)."""
    import random
    from lm_eval.tasks import TaskManager
    import lm_eval.tasks as lm_tasks

    task_manager = TaskManager(
        include_path=str(PROJECT_ROOT / "diffusion_llms" / "tasks")
    )
    task_dict = lm_tasks.get_task_dict(["gsm8k_cot_llama"], task_manager)
    task = task_dict["gsm8k_cot_llama"]

    docs = list(task.test_docs())
    rng = random.Random(seed)
    rng.shuffle(docs)
    docs = docs[:n]

    examples = []
    for doc in docs:
        raw_text  = task.doc_to_text(doc)
        gold      = task.doc_to_target(doc).strip()
        messages  = [{"role": "user", "content": raw_text}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids        = tokenizer.encode(templated, add_special_tokens=False)
        prompt_ids = torch.tensor(ids, dtype=torch.long, device=device)
        examples.append((prompt_ids, gold, None))

    return examples


# ---------------------------------------------------------------------------
# HumanEval helpers
# ---------------------------------------------------------------------------
_code_eval_metric = None


def _get_code_eval():
    global _code_eval_metric
    if _code_eval_metric is None:
        import evaluate as hf_evaluate
        _code_eval_metric = hf_evaluate.load("code_eval")
    return _code_eval_metric


def humaneval_correct(prediction: str, doc: dict) -> int:
    blocks = re.findall(r'```\w*\n(.*?)```', prediction, re.DOTALL)
    code = blocks[0] if blocks else prediction.lstrip()
    full_code = doc["prompt"] + code
    test_str  = doc["test"] + f"\ncheck({doc['entry_point']})"
    try:
        result, _ = _get_code_eval().compute(
            references=[test_str],
            predictions=[[full_code]],
            k=[1],
            timeout=10.0,
        )
        return int(result.get("pass@1", 0.0) == 1.0)
    except Exception:
        return 0


def load_humaneval_examples(n: int, seed: int, tokenizer, device):
    """Return list of (prompt_ids, gold_str, doc)."""
    import random
    from lm_eval.tasks import TaskManager
    import lm_eval.tasks as lm_tasks

    task_manager = TaskManager(
        include_path=str(PROJECT_ROOT / "diffusion_llms" / "tasks")
    )
    task_dict = lm_tasks.get_task_dict(["humaneval_instruct_local"], task_manager)
    task = task_dict["humaneval_instruct_local"]

    docs = list(task.test_docs())
    rng = random.Random(seed)
    rng.shuffle(docs)
    docs = docs[:n]

    examples = []
    for doc in docs:
        raw_text  = task.doc_to_text(doc)
        gold      = task.doc_to_target(doc).strip()
        messages  = [{"role": "user", "content": raw_text}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids        = tokenizer.encode(templated, add_special_tokens=False)
        prompt_ids = torch.tensor(ids, dtype=torch.long, device=device)
        examples.append((prompt_ids, gold, doc))

    return examples


def check_correct(task: str, prediction: str, gold: str, doc) -> int:
    if task == "gsm8k":
        return int(flexible_match(prediction, gold))
    elif task == "humaneval":
        return humaneval_correct(prediction, doc)
    elif task == "mbpp":
        return mbpp_correct(prediction, doc)
    elif task == "ifeval":
        return ifeval_correct(prediction, doc)
    elif task == "math500":
        return math500_correct(prediction, gold)
    else:
        raise ValueError(f"Unknown task: {task}")


# ---------------------------------------------------------------------------
# MBPP helpers
# ---------------------------------------------------------------------------

def mbpp_correct(prediction: str, doc: dict) -> int:
    """Check MBPP correctness by running the test assertions."""
    blocks = re.findall(r'```\w*\n(.*?)```', prediction, re.DOTALL)
    code = blocks[0] if blocks else prediction.lstrip()
    test_str = "\n".join(doc["test_list"])
    try:
        result, _ = _get_code_eval().compute(
            references=[test_str],
            predictions=[[code]],
            k=[1],
            timeout=10.0,
        )
        return int(result.get("pass@1", 0.0) == 1.0)
    except Exception:
        return 0


def load_mbpp_examples(n: int, seed: int, tokenizer, device):
    """Return list of (prompt_ids, gold_str, doc)."""
    import random
    from lm_eval.tasks import TaskManager
    import lm_eval.tasks as lm_tasks

    task_manager = TaskManager()
    task_dict = lm_tasks.get_task_dict(["mbpp_instruct"], task_manager)
    task = task_dict["mbpp_instruct"]

    docs = list(task.test_docs())
    rng = random.Random(seed)
    rng.shuffle(docs)
    docs = docs[:n]

    examples = []
    for doc in docs:
        raw_text = (
            "You are an expert Python programmer, and here is your task:\n"
            f"{doc['text']}\n"
            "Your code should pass these tests:\n"
            + "\n".join(doc["test_list"][:3])
        )
        messages = [{"role": "user", "content": raw_text}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer.encode(templated, add_special_tokens=False)
        prompt_ids = torch.tensor(ids, dtype=torch.long, device=device)
        examples.append((prompt_ids, doc.get("code", ""), doc))

    return examples


# ---------------------------------------------------------------------------
# IFEval helpers
# ---------------------------------------------------------------------------

def ifeval_correct(prediction: str, doc: dict) -> int:
    """Check IFEval prompt-level strict accuracy."""
    try:
        import inspect
        from lm_eval.tasks.ifeval.instructions_registry import INSTRUCTION_DICT
        instructions = []
        for i, k in zip(doc["instruction_id_list"], doc["kwargs"]):
            if i not in INSTRUCTION_DICT:
                continue
            instr = INSTRUCTION_DICT[i](instruction_id=i)
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
    except Exception:
        return 0


def load_ifeval_examples(n: int, seed: int, tokenizer, device):
    """Return list of (prompt_ids, gold_str, doc)."""
    import random
    from datasets import load_dataset

    dataset = load_dataset("google/IFEval", split="train")
    docs = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(docs)
    docs = docs[:n]

    examples = []
    for doc in docs:
        messages = [{"role": "user", "content": doc["prompt"]}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer.encode(templated, add_special_tokens=False)
        prompt_ids = torch.tensor(ids, dtype=torch.long, device=device)
        examples.append((prompt_ids, doc["prompt"], doc))

    return examples


# ---------------------------------------------------------------------------
# MATH500 helpers
# ---------------------------------------------------------------------------

def math500_correct(prediction: str, gold: str) -> int:
    """Check MATH500 correctness using boxed-answer equivalence."""
    from diffusion_llms.tasks.math500.utils import extract_answer, is_equiv
    pred_ans = extract_answer(prediction)
    return int(is_equiv(pred_ans, gold))


def load_math500_examples(n: int, seed: int, tokenizer, device):
    """Return list of (prompt_ids, gold_str, doc)."""
    import random
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    docs = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(docs)
    docs = docs[:n]

    examples = []
    for doc in docs:
        raw_text = (
            "Solve the following problem step by step. "
            "Express your final answer in \\boxed{}.\n\n"
            f"Problem: {doc['problem']}\n\nSolution:"
        )
        messages = [{"role": "user", "content": raw_text}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer.encode(templated, add_special_tokens=False)
        prompt_ids = torch.tensor(ids, dtype=torch.long, device=device)
        examples.append((prompt_ids, doc["answer"], doc))

    return examples


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_stable_median(survival_curve: dict[int, int]) -> int:
    """Stable median: drop first 10 steps (transient) and last 5% (collapse)."""
    sorted_steps = sorted(survival_curve.keys())
    predictions  = [survival_curve[k] for k in sorted_steps]
    n = len(predictions)
    stable = predictions[10 : int(0.95 * n)]
    if not stable:
        stable = predictions  # fallback if very few steps recorded
    return int(np.median(stable))


def run_one_example(
    generator,
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    gold: str,
    doc,            # dict for HumanEval, None for GSM8K
    task: str,
    canvas_size: int,
    record_every: int,
    eval_steps: list[int],
    mask_id: int,
    eos_id: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], dict[int, np.ndarray]]:
    """
    Returns:
      curve_rows      — predicted_len at every record_every callback fires
      gen_rows        — generation + accuracy at 4 checkpoints
      survival_arrays — {step: S(t) array of shape [canvas_size]}
    """
    prompt_len = prompt_ids.shape[0]

    survival_curve: dict[int, int]       = {}
    survival_arrays: dict[int, np.ndarray] = {}

    def _callback(state: dict):
        step = state["step"]
        if step % record_every != 0:
            return
        logits_k = state["logits"][:, prompt_len:, :]  # [1, canvas_size, V]
        with torch.no_grad():
            step_probs  = torch.softmax(logits_k[0], dim=-1)     # [canvas_size, V]
            eos_probs   = step_probs[:, eos_id]                   # [canvas_size]
            not_stopped = torch.cumprod(1.0 - eos_probs, dim=0)  # S(t)
            stopped     = 1.0 - not_stopped
            cutoff      = torch.where(stopped >= EOS_QUANTILE)[0]
            gen_tokens  = int(cutoff[0].item()) + 1 if cutoff.numel() > 0 else canvas_size
        survival_curve[step]  = max(1, gen_tokens)
        survival_arrays[step] = not_stopped.cpu().float().numpy()

    fc_cfg = BaseLengthGeneratorConfig(
        max_new_tokens  = canvas_size,
        block_length    = canvas_size,
        steps           = canvas_size,
        temperature     = 0.0,
        remasking       = "low_confidence",
        logits_eos_inf  = False,   # keep EOS signal intact for CDF
        step_callback   = _callback,
    )
    generator.generate([prompt_ids], config=fc_cfg)

    # --- Survival curve rows ---
    curve_rows = [
        {
            "checkpoint_k":  k,
            "frac":          round(k / canvas_size, 4),
            "predicted_len": survival_curve[k],
        }
        for k in sorted(survival_curve.keys())
    ]

    # --- Stable median ---
    stable_median_len = compute_stable_median(survival_curve)

    # --- Eval points: fraction-based + stable_median ---
    eval_points = []
    for k in sorted(eval_steps):
        nearest    = min(survival_curve.keys(), key=lambda s: abs(s - k))
        gen_tokens = survival_curve[nearest]
        label      = f"k={k} ({k / canvas_size:.1%})"
        eval_points.append((k, label, gen_tokens))
    # stable_median uses checkpoint_k = -1 as sentinel
    eval_points.append((-1, "stable_median", stable_median_len))

    gen_rows = []
    for (ck, label, gen_tokens) in eval_points:
        predicted_total = prompt_len + gen_tokens

        gen_cfg = BaseLengthGeneratorConfig(
            max_new_tokens    = gen_tokens,
            block_length      = gen_tokens,
            steps             = gen_tokens,
            temperature       = 0.0,
            remasking         = "low_confidence",
            length_prediction = predicted_total,
            logits_eos_inf    = True,
        )
        out     = generator.generate([prompt_ids], config=gen_cfg)
        gen_ids = out.sequences[0, prompt_len:]
        eos_pos = (gen_ids == eos_id).nonzero()
        if eos_pos.numel() > 0:
            gen_ids = gen_ids[: eos_pos[0].item()]
        prediction = tokenizer.decode(gen_ids, skip_special_tokens=True)

        gen_rows.append({
            "checkpoint_k":      ck,
            "checkpoint_label":  label,
            "predicted_len":     gen_tokens,
            "stable_median_len": stable_median_len,
            "correct":           check_correct(task, prediction, gold, doc),
            "prediction":        prediction,
            "gold":              gold,
        })

    return curve_rows, gen_rows, survival_arrays


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        choices=["gsm8k", "humaneval", "mbpp", "ifeval", "math500"],
                        default="gsm8k")
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--n_examples",   type=int, default=N_EXAMPLES)
    parser.add_argument("--canvas_sizes", type=int, nargs="+", default=CANVAS_SIZES)
    parser.add_argument("--record_every", type=int, default=RECORD_EVERY)
    parser.add_argument("--output_dir",   default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Task:  {args.task}")
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = get_model(
        model_name_or_path=args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    model.requires_grad_(False)

    mask_id = ensure_token_id(
        getattr(tokenizer, "mask_token_id", None), "mask_token_id",
        tokenizer=tokenizer,
        candidates=["<mask>", "[MASK]", "<|mask|>", "<|mdm_mask|>", "[gMASK]"],
    )
    eos_id = ensure_token_id(
        getattr(tokenizer, "eos_token_id", None), "eos_token_id",
        tokenizer=tokenizer,
    )
    print(f"mask_id={mask_id}  eos_id={eos_id}")

    generator = BaseLengthGenerator(model=model, tokenizer=tokenizer)

    print(f"Loading {args.n_examples} {args.task} examples...")
    if args.task == "gsm8k":
        examples = load_gsm8k_examples(args.n_examples, SEED, tokenizer, device)
    elif args.task == "humaneval":
        examples = load_humaneval_examples(args.n_examples, SEED, tokenizer, device)
        _get_code_eval()  # warm up
    elif args.task == "mbpp":
        examples = load_mbpp_examples(args.n_examples, SEED, tokenizer, device)
        _get_code_eval()  # warm up
    elif args.task == "ifeval":
        examples = load_ifeval_examples(args.n_examples, SEED, tokenizer, device)
    elif args.task == "math500":
        examples = load_math500_examples(args.n_examples, SEED, tokenizer, device)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    print(f"Loaded {len(examples)} examples.")

    run_id  = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir = Path(args.output_dir or
                   PROJECT_ROOT / "results" / "rebuttal" /
                   f"step_analysis_{args.task}_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_csv_path = out_dir / "survival_curve_raw.csv"
    gen_csv_path   = out_dir / "generation_results.csv"

    # --- Resume: find already-completed example_ids from existing gen CSV ---
    completed_ids: set[int] = set()
    curve_header_written = gen_csv_path.exists() and gen_csv_path.stat().st_size > 0
    gen_header_written   = curve_csv_path.exists() and curve_csv_path.stat().st_size > 0
    if gen_csv_path.exists() and gen_csv_path.stat().st_size > 0:
        with open(gen_csv_path, newline="") as f:
            for row in csv.DictReader(f):
                completed_ids.add(int(row["example_id"]))
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} examples already done, skipping them.")

    for canvas_size in args.canvas_sizes:
        eval_steps = sorted(set(int(f * canvas_size) for f in EVAL_FRACS))

        print(f"\n{'='*60}")
        print(f"Task={args.task}  Canvas={canvas_size}  "
              f"record_every={args.record_every}  "
              f"eval_steps={eval_steps} + stable_median")
        print(f"{'='*60}")

        correct_by_label: dict[str, int] = {}
        canvas_t0 = time.time()

        for i, (prompt_ids, gold, doc) in enumerate(examples):
            if i in completed_ids:
                print(f"  ex {i+1:3d}/{len(examples)}  [SKIPPED]", flush=True)
                continue
            ex_t0 = time.time()
            curve_rows, gen_rows, survival_arrays = run_one_example(
                generator=generator,
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                gold=gold,
                doc=doc,
                task=args.task,
                canvas_size=canvas_size,
                record_every=args.record_every,
                eval_steps=eval_steps,
                mask_id=mask_id,
                eos_id=eos_id,
                device=device,
            )
            ex_elapsed = time.time() - ex_t0

            gen_str = "  ".join(
                f"{r['checkpoint_label']}:{'✓' if r['correct'] else '✗'}(L={r['predicted_len']})"
                for r in gen_rows
            )
            print(f"  ex {i+1:3d}/{len(examples)}  ({ex_elapsed:.1f}s)  [{gen_str}]",
                  flush=True)

            for r in gen_rows:
                lbl = r["checkpoint_label"]
                correct_by_label[lbl] = correct_by_label.get(lbl, 0) + r["correct"]

            # Save survival arrays to npz
            npz_path = out_dir / f"survival_arrays_canvas{canvas_size}.npz"
            existing = dict(np.load(npz_path)) if npz_path.exists() else {}
            for step, arr in survival_arrays.items():
                existing[f"ex{i}_step{step}"] = arr
            np.savez_compressed(npz_path, **existing)

            # Flush survival curve rows
            with open(curve_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["canvas_size", "example_id"] + list(curve_rows[0].keys())
                )
                if not curve_header_written:
                    writer.writeheader()
                    curve_header_written = True
                writer.writerows(
                    {"canvas_size": canvas_size, "example_id": i, **r}
                    for r in curve_rows
                )

            # Flush generation rows
            with open(gen_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["canvas_size", "example_id"] + list(gen_rows[0].keys())
                )
                if not gen_header_written:
                    writer.writeheader()
                    gen_header_written = True
                writer.writerows(
                    {"canvas_size": canvas_size, "example_id": i, **r}
                    for r in gen_rows
                )

            if (i + 1) % 5 == 0:
                elapsed = time.time() - canvas_t0
                acc_str = "  ".join(
                    f"{lbl}:{correct_by_label.get(lbl, 0)/(i+1):.3f}"
                    for lbl in sorted(correct_by_label)
                )
                print(f"  --- [{i+1}/{len(examples)}] elapsed={elapsed:.0f}s  acc: {acc_str}",
                      flush=True)

        print(f"\n  Final accuracy (task={args.task}, canvas={canvas_size}, n={len(examples)}):")
        for lbl in sorted(correct_by_label):
            acc = correct_by_label[lbl] / len(examples)
            print(f"    {lbl:<35s}  acc={acc:.4f}  ({correct_by_label[lbl]}/{len(examples)})")

    summary_path = out_dir / "step_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "task":         args.task,
            "model":        args.model,
            "n_examples":   len(examples),
            "eos_quantile": EOS_QUANTILE,
            "record_every": args.record_every,
            "eval_fracs":   EVAL_FRACS,
            "eval_note":    "4th checkpoint = stable_median (checkpoint_k=-1)",
            "canvas_sizes": args.canvas_sizes,
            "run_id":       run_id,
        }, f, indent=2)
    print(f"\nSummary: {summary_path}")
    print(f"Curve CSV: {curve_csv_path}")
    print(f"Gen CSV:   {gen_csv_path}")


if __name__ == "__main__":
    main()
