#!/usr/bin/env python3
"""
SmartCrop Step Analysis — GSM8K

For each canvas_size in [512, 2048] and checkpoint fraction in [0%, 5%, 10%, 25%, 50%]:
  1. Run ONE full FC generation (steps=canvas_size, i.e. one token unmasked per step)
     with a step_callback that caches logits at the checkpoint steps.
  2. For each checkpoint k:
       a. Apply EOS CDF (q=0.9) to logits_k  → predicted length L_k
       b. Run fresh generation at L_k
       c. Evaluate with flexible-match
  3. Output accuracy table + raw CSV.

k=0 (all-masked pass) is identical to current SmartCrop.

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
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

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
DEFAULT_MODEL      = "GSAI-ML/LLaDA-8B-Instruct"
CANVAS_SIZES       = [512, 2048]
# Checkpoint fractions of total steps (0 = all-masked pass = current SmartCrop)
CHECKPOINT_FRACS   = [0.0, 0.05, 0.10, 0.25, 0.50]
EOS_QUANTILE       = 0.9
N_EXAMPLES         = 50
SEED               = 42


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
    checkpoint_steps: list[int],
    mask_id: int,
    eos_id: int,
    device: torch.device,
) -> list[dict]:
    """
    1. FC generation at canvas_size (steps=canvas_size) with logit caching.
    2. For each checkpoint k: predict length → fresh generation → evaluate.
    Returns list of result dicts (one per checkpoint).
    """
    prompt_len = prompt_ids.shape[0]
    target_steps = set(checkpoint_steps)

    # --- Step 1: FC probe run, caching logits at checkpoints ---
    cached_logits: dict[int, torch.Tensor] = {}

    def _cache_callback(state: dict):
        step = state["step"]
        if step in target_steps:
            # logits shape: [1, seq_len, vocab] — keep only generation part, detach
            cached_logits[step] = state["logits"][:, prompt_len:, :].detach().clone()

    fc_cfg = BaseLengthGeneratorConfig(
        max_new_tokens  = canvas_size,
        block_length    = canvas_size,   # single block
        steps           = canvas_size,   # one token unmasked per step
        temperature     = 0.0,
        remasking       = "low_confidence",
        logits_eos_inf  = True,
        step_callback   = _cache_callback,
    )
    # We discard the FC output — we only need the cached logits
    generator.generate([prompt_ids], config=fc_cfg)

    # --- Step 2: For each checkpoint, predict + generate fresh + evaluate ---
    results = []
    for k in checkpoint_steps:
        logits_k = cached_logits.get(k)

        # k=0: logits_k is None (no denoising step 0 in the loop); use all-masked pass
        if logits_k is None:
            predicted_total = compute_zero_shot_length_prediction(
                model=model,
                prompt=prompt_ids,
                mask_id=mask_id,
                eos_id=eos_id,
                max_new_tokens=canvas_size,
                eos_quantile=EOS_QUANTILE,
                safe_margin=0,
                device=device,
            )
        else:
            # Reconstruct full-sequence logits tensor for the helper
            # It only needs the generation slice, but the helper signature expects [1,T,V]
            # We pass via the logits kwarg which skips the forward pass
            predicted_total = compute_zero_shot_length_prediction(
                model=model,
                prompt=prompt_ids,
                mask_id=mask_id,
                eos_id=eos_id,
                max_new_tokens=canvas_size,
                eos_quantile=EOS_QUANTILE,
                safe_margin=0,
                device=device,
                logits=torch.cat([
                    torch.zeros(1, prompt_len, logits_k.shape[-1], device=device),
                    logits_k,
                ], dim=1),
            )

        if predicted_total is None:
            predicted_total = prompt_len + canvas_size

        gen_tokens = max(1, predicted_total - prompt_len)

        # Fresh generation at predicted length
        gen_cfg = BaseLengthGeneratorConfig(
            max_new_tokens  = gen_tokens,
            block_length    = gen_tokens,
            steps           = gen_tokens,
            temperature     = 0.0,
            remasking       = "low_confidence",
            length_prediction = predicted_total,
            logits_eos_inf  = True,
        )
        out     = generator.generate([prompt_ids], config=gen_cfg)
        gen_ids = out.sequences[0, prompt_len:]
        # Strip EOS padding
        eos_pos = (gen_ids == eos_id).nonzero()
        if eos_pos.numel() > 0:
            gen_ids = gen_ids[: eos_pos[0].item()]
        prediction = tokenizer.decode(gen_ids, skip_special_tokens=True)

        is_correct = flexible_match(prediction, gold)
        results.append({
            "checkpoint_k":  k,
            "predicted_len": gen_tokens,
            "prediction":    prediction[:300],
            "gold":          gold,
            "correct":       int(is_correct),
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--n_examples",   type=int,   default=N_EXAMPLES)
    parser.add_argument("--canvas_sizes", type=int,   nargs="+", default=CANVAS_SIZES)
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

    all_rows   = []
    # summary[canvas_size][k] = accuracy
    summary: dict[int, dict[int, float]] = {}

    for canvas_size in args.canvas_sizes:
        summary[canvas_size] = {}

        # Compute checkpoint steps from fractions
        checkpoint_steps = sorted(set(
            [int(f * canvas_size) for f in CHECKPOINT_FRACS]
        ))
        # Ensure 0 is always included (all-masked pass = SmartCrop baseline)
        if 0 not in checkpoint_steps:
            checkpoint_steps = [0] + checkpoint_steps

        print(f"\n{'='*60}")
        print(f"Canvas={canvas_size}  checkpoints={checkpoint_steps}")
        print(f"{'='*60}")

        # Accumulators per checkpoint
        correct_counts = {k: 0 for k in checkpoint_steps}

        for i, (prompt_ids, gold) in enumerate(examples):
            ex_results = run_one_example(
                generator=generator,
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                gold=gold,
                canvas_size=canvas_size,
                checkpoint_steps=checkpoint_steps,
                mask_id=mask_id,
                eos_id=eos_id,
                device=device,
            )

            for r in ex_results:
                k = r["checkpoint_k"]
                correct_counts[k] += r["correct"]
                all_rows.append({
                    "canvas_size":    canvas_size,
                    "checkpoint_k":   k,
                    "frac":           round(k / canvas_size, 3) if canvas_size else 0,
                    "example_id":     i,
                    "predicted_len":  r["predicted_len"],
                    "correct":        r["correct"],
                    "gold":           r["gold"],
                    "prediction":     r["prediction"],
                })

            if (i + 1) % 10 == 0:
                running = {k: correct_counts[k] / (i + 1) for k in checkpoint_steps}
                print(f"  [{i+1}/{len(examples)}] running acc: "
                      + "  ".join(f"k={k}:{v:.3f}" for k, v in running.items()))

        for k in checkpoint_steps:
            acc = correct_counts[k] / len(examples)
            summary[canvas_size][k] = acc

    # Save CSV
    csv_path = out_dir / "step_analysis_raw.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV: {csv_path}")

    # Print summary table
    print("\n" + "="*65)
    print("ACCURACY vs CHECKPOINT (flexible-match, GSM8K)")
    print("="*65)
    all_canvas = args.canvas_sizes
    header = f"{'k (steps)':>10}  {'frac':>6}" + "".join(f"  c={c:4d}" for c in all_canvas)
    print(header)
    print("-" * len(header))
    # Collect all k values seen
    all_ks = sorted(set(k for c in all_canvas for k in summary.get(c, {})))
    for k in all_ks:
        frac = k / all_canvas[0] if all_canvas else 0
        row  = f"{k:>10}  {frac:>6.2f}"
        for c in all_canvas:
            frac_c = k / c if c else 0
            acc    = summary.get(c, {}).get(k, float("nan"))
            row   += f"  {acc:.4f}({frac_c:.2f})"
        print(row)

    # Save JSON summary
    summary_path = out_dir / "step_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model":             args.model,
            "n_examples":        len(examples),
            "eos_quantile":      EOS_QUANTILE,
            "checkpoint_fracs":  CHECKPOINT_FRACS,
            "canvas_sizes":      args.canvas_sizes,
            "results":           {
                str(c): {str(k): v for k, v in d.items()}
                for c, d in summary.items()
            },
            "run_id":            run_id,
        }, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
