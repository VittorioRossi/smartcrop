#!/usr/bin/env python3
"""
SmartCrop Survival-Function Sweep — GSM8K

For each canvas_size, run ONE full FC generation (steps=canvas_size, one token
unmasked per step). A step_callback computes the EOS survival function and
records the resulting predicted length every RECORD_EVERY callback fires.
No logit tensors are stored — the CDF is computed and discarded inline.

Fresh generation + accuracy evaluation is only run at EVAL_FRACS checkpoints
(sparse), using the predicted length from the survival curve at that step.

Outputs:
  survival_curve_raw.csv  — (example_id, step, frac, predicted_len)  for every
                             RECORD_EVERY steps of every example
  generation_results.csv  — (example_id, step, frac, predicted_len, correct,
                             prediction, gold)  for EVAL_FRACS checkpoints only

step=0 (all-masked pass) is the SmartCrop baseline.

Usage:
    python scripts/run_gsm8k_step_analysis.py
    sbatch scripts/slurm/run_gsm8k_step_analysis.slurm
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

import torch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer
from diffusion_llms.utils.model_utils import get_model
from diffusion_llms.generators.base_length_generator import (
    BaseLengthGenerator,
    BaseLengthGeneratorConfig,
)
from diffusion_llms.utils.generation_utils import compute_zero_shot_length_prediction
from diffusion_llms.utils.common import ensure_token_id

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
CANVAS_SIZES  = [512]
# Record predicted_len from survival function at every N callback fires
RECORD_EVERY  = 5
# Run fresh generation + evaluation only at these fractions of canvas steps
EVAL_FRACS    = [0.0, 0.05, 0.10, 0.25, 0.50]
EOS_QUANTILE  = 0.9
N_EXAMPLES    = 50
SEED          = 42


# ---------------------------------------------------------------------------
# Helpers
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
    """Load n GSM8K examples via lm_eval, return list of (prompt_ids, gold_str)."""
    import random
    from lm_eval.tasks import TaskManager
    import lm_eval.tasks as lm_tasks

    task_manager = TaskManager(
        include_path=str(PROJECT_ROOT / "diffusion_llms" / "tasks")
    )
    task_dict = lm_tasks.get_task_dict(["gsm8k_cot_llama"], task_manager)
    task      = task_dict["gsm8k_cot_llama"]

    docs = list(task.test_docs())
    rng  = random.Random(seed)
    rng.shuffle(docs)
    docs = docs[:n]

    examples = []
    for doc in docs:
        raw_text = task.doc_to_text(doc)
        gold     = task.doc_to_target(doc).strip()
        messages  = [{"role": "user", "content": raw_text}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids        = tokenizer.encode(templated, add_special_tokens=False)
        prompt_ids = torch.tensor(ids, dtype=torch.long, device=device)
        examples.append((prompt_ids, gold))

    return examples


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_one_example(
    generator,
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    gold: str,
    canvas_size: int,
    record_every: int,
    eval_steps: set[int],
    mask_id: int,
    eos_id: int,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    """
    Returns:
      curve_rows  — predicted_len at every record_every callback fires
      gen_rows    — generation + accuracy at eval_steps checkpoints
    """
    prompt_len = prompt_ids.shape[0]

    # step -> predicted new token count (recorded every record_every steps)
    survival_curve: dict[int, int] = {}

    def _callback(state: dict):
        step = state["step"]
        if step % record_every != 0:
            return
        logits_k = state["logits"][:, prompt_len:, :]
        full_logits = torch.cat([
            torch.zeros(1, prompt_len, logits_k.shape[-1], device=device),
            logits_k,
        ], dim=1)
        pred = compute_zero_shot_length_prediction(
            model=model,
            prompt=prompt_ids,
            mask_id=mask_id,
            eos_id=eos_id,
            max_new_tokens=canvas_size,
            eos_quantile=EOS_QUANTILE,
            safe_margin=0,
            device=device,
            logits=full_logits,
        )
        survival_curve[step] = max(1, pred - prompt_len) if pred is not None else canvas_size
        # full_logits released here — no persistent storage

    fc_cfg = BaseLengthGeneratorConfig(
        max_new_tokens  = canvas_size,
        block_length    = canvas_size,
        steps           = canvas_size,
        temperature     = 0.0,
        remasking       = "low_confidence",
        logits_eos_inf  = False,  # keep EOS signal intact for CDF
        step_callback   = _callback,
    )
    generator.generate([prompt_ids], config=fc_cfg)

    # --- Survival curve rows (cheap — already computed) ---
    curve_rows = [
        {
            "checkpoint_k":  k,
            "frac":          round(k / canvas_size, 4),
            "predicted_len": survival_curve[k],
        }
        for k in sorted(survival_curve.keys())
    ]

    # --- Fresh generation only at eval checkpoints ---
    gen_rows = []
    for k in sorted(eval_steps):
        # Use the predicted_len from the survival curve at the nearest recorded step
        nearest = min(survival_curve.keys(), key=lambda s: abs(s - k))
        gen_tokens      = survival_curve[nearest]
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
            "checkpoint_k":  k,
            "frac":          round(k / canvas_size, 4),
            "predicted_len": gen_tokens,
            "correct":       int(flexible_match(prediction, gold)),
            "prediction":    prediction,
            "gold":          gold,
        })

    return curve_rows, gen_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--n_examples",   type=int, default=N_EXAMPLES)
    parser.add_argument("--canvas_sizes", type=int, nargs="+", default=CANVAS_SIZES)
    parser.add_argument("--record_every", type=int, default=RECORD_EVERY)
    parser.add_argument("--output_dir",   default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model     = get_model(
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

    print(f"Loading {args.n_examples} GSM8K examples...")
    examples = load_gsm8k_examples(args.n_examples, SEED, tokenizer, device)
    print(f"Loaded {len(examples)} examples.")

    run_id  = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir = Path(args.output_dir or
                   PROJECT_ROOT / "results" / "rebuttal" / f"step_analysis_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_csv_path = out_dir / "survival_curve_raw.csv"
    gen_csv_path   = out_dir / "generation_results.csv"

    curve_header_written = False
    gen_header_written   = False

    for canvas_size in args.canvas_sizes:
        eval_steps = sorted(set(int(f * canvas_size) for f in EVAL_FRACS))

        print(f"\n{'='*60}")
        print(f"Canvas={canvas_size}  record_every={args.record_every}  "
              f"eval_steps={eval_steps}")
        print(f"{'='*60}")

        correct_counts = {k: 0 for k in eval_steps}
        canvas_t0 = time.time()

        for i, (prompt_ids, gold) in enumerate(examples):
            ex_t0 = time.time()
            curve_rows, gen_rows = run_one_example(
                generator=generator,
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                gold=gold,
                canvas_size=canvas_size,
                record_every=args.record_every,
                eval_steps=set(eval_steps),
                mask_id=mask_id,
                eos_id=eos_id,
                device=device,
            )
            ex_elapsed = time.time() - ex_t0

            # Per-example log: show a few curve points + gen results
            curve_sample = [r for r in curve_rows if r["checkpoint_k"] % 50 == 0]
            curve_str = "  ".join(
                f"k={r['checkpoint_k']}:L={r['predicted_len']}" for r in curve_sample
            )
            gen_str = "  ".join(
                f"k={r['checkpoint_k']}:{'✓' if r['correct'] else '✗'}"
                for r in gen_rows
            )
            print(f"  ex {i+1:3d}/{len(examples)}  ({ex_elapsed:.1f}s)  "
                  f"curve=[{curve_str}]  gen=[{gen_str}]", flush=True)

            # Update accuracy counts
            for r in gen_rows:
                correct_counts[r["checkpoint_k"]] += r["correct"]

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
                    f"k={k}:{correct_counts[k]/(i+1):.3f}" for k in eval_steps
                )
                print(f"  --- [{i+1}/{len(examples)}] elapsed={elapsed:.0f}s  acc: {acc_str}",
                      flush=True)

        # Final accuracy for this canvas
        print(f"\n  Final accuracy (canvas={canvas_size}, n={len(examples)}):")
        for k in eval_steps:
            acc  = correct_counts[k] / len(examples)
            frac = k / canvas_size
            print(f"    k={k:4d}  frac={frac:.2f}  acc={acc:.4f}")

    # Save JSON summary
    summary_path = out_dir / "step_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model":        args.model,
            "n_examples":   len(examples),
            "eos_quantile": EOS_QUANTILE,
            "record_every": args.record_every,
            "eval_fracs":   EVAL_FRACS,
            "canvas_sizes": args.canvas_sizes,
            "run_id":       run_id,
        }, f, indent=2)
    print(f"\nSummary: {summary_path}")
    print(f"Curve CSV: {curve_csv_path}")
    print(f"Gen CSV:   {gen_csv_path}")


if __name__ == "__main__":
    main()
