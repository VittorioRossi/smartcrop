"""E1 pilot — per-prompt length-variance check.

For a small set of prompts, take one t=1 forward pass (the CDF readout
G_1(i;c) = p_eos(i), already a CDF by the EOS-padding convention — no cumsum
needed) and K stochastic completions at temperature>0 (low_confidence
remasking still needs temp>0 for token-level diversity, since remasking
order is confidence-driven, not random).

Goal: check whether repeated sampling produces enough spread in the realized
boundary B_k to make the E1 PIT/calibration test meaningful. If std(B_k) per
prompt is tiny at this temperature, rerun with a higher temperature or a
larger K before committing to the full ~200-prompt run.

Usage:
    python scripts/run_e1_pilot.py --task gsm8k --n 20 --k 10 --temperature 1.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import dataclasses
import json
import time

import numpy as np
import torch
import dllm

from diffusion_llms.generators.cdf_remasking_generator import _predict_lengths_from_t1
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

from run_benchmark import load_examples, actual_length, _ModelArgs, DEFAULT_MODEL

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=20, help="number of prompts")
    parser.add_argument("--k", type=int, default=10, help="completions per prompt")
    parser.add_argument("--gen-length", type=int, default=1024, dest="gen_length")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=32, dest="block_size")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cdf-quantile", type=float, default=0.90, dest="cdf_quantile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = PROJECT_ROOT / "results" / "e1_pilot" / f"{args.task}_n{args.n}_k{args.k}.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model_args = _ModelArgs(model_name_or_path=args.model)
    model = dllm.utils.get_model(model_args=model_args).eval().to(device)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    eos_id = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id

    print(f"Loading {args.n} {args.task} examples...")
    assistant_prefix = "" if args.task in ("humaneval", "mbpp") else "<reasoning>"
    examples = load_examples(args.task, tokenizer, args.n, seed=args.seed,
                              assistant_prefix=assistant_prefix)
    print(f"Loaded {len(examples)} examples")

    cfg = MDLMSamplerConfig(
        max_new_tokens=args.gen_length,
        steps=args.steps,
        block_size=args.block_size,
        temperature=args.temperature,
        remasking="low_confidence",
    )
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)

    results = []
    if args.out.exists():
        with open(args.out) as f:
            results = json.load(f)
    done_ids = {r["doc_id"] for r in results}

    for i, ex in enumerate(examples):
        doc_id = ex["doc_id"]
        if doc_id in done_ids:
            print(f"[{i+1:3d}/{len(examples)}] doc={doc_id}  skipped")
            continue

        prompt_ids = ex["prompt_ids"]
        prompt_tensor = torch.as_tensor(prompt_ids, dtype=torch.long, device=model.device)

        # single t=1 readout — G_1(i;c) = p_eos(i), already a CDF, no transform needed
        pred_lens, prob_arrays = _predict_lengths_from_t1(
            model, [prompt_tensor], mask_id, eos_id, args.gen_length,
            quantile=args.cdf_quantile, return_probs=True,
        )
        p_eos_curve = prob_arrays[0][:, 0].tolist()  # [max_new_tokens]

        b_ks = []
        t0 = time.time()
        for k in range(args.k):
            torch.manual_seed(args.seed * 10_000 + i * 100 + k)
            out = sampler.sample([prompt_ids], cfg, return_dict=True)
            seq = out.sequences[0].tolist()
            b_ks.append(actual_length(seq, len(prompt_ids), eos_id))
        elapsed = time.time() - t0

        results.append({
            "doc_id": doc_id,
            "pred_len_q90": pred_lens[0],
            "p_eos_curve": p_eos_curve,
            "b_ks": b_ks,
            "elapsed": elapsed,
        })

        with open(args.out, "w") as f:
            json.dump(results, f)

        b_arr = np.array(b_ks)
        print(f"[{i+1:3d}/{len(examples)}] doc={doc_id}  pred_len={pred_lens[0]}  "
              f"mean(B_k)={b_arr.mean():.1f}  std(B_k)={b_arr.std():.1f}  "
              f"min={b_arr.min()} max={b_arr.max()}  {elapsed:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────────────
    all_stds = [np.std(r["b_ks"]) for r in results]
    all_means = [np.mean(r["b_ks"]) for r in results]
    print(f"\n{'─'*55}")
    print(f"  Prompts:            {len(results)}")
    print(f"  Mean per-prompt std(B_k):  {np.mean(all_stds):.2f}")
    print(f"  Median per-prompt std(B_k): {np.median(all_stds):.2f}")
    print(f"  Mean per-prompt mean(B_k): {np.mean(all_means):.2f}")
    print(f"  Coefficient of variation (std/mean), avg: {np.mean([s/m for s, m in zip(all_stds, all_means) if m > 0]):.3f}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
