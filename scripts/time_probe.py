"""Wall-clock timing of SmartCrop's t=1 probe vs the denoising loop.

Replicates CDFRemaskingSampler.sample step by step (same probe function, same
cropped config, same MDLMSampler call) and wraps each phase with
torch.cuda.synchronize(). Sampler code is untouched.

Usage:
    python scripts/time_probe.py --tasks gsm8k math500 mbpp humaneval --n 50 50 50 30
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import logging
import time
from copy import copy

import torch
import dllm
from dllm.core.samplers import MDLMSampler

from diffusion_llms.generators import CDFRemaskingSampler, CDFRemaskingSamplerConfig
from diffusion_llms.generators.cdf_remasking_generator import _predict_lengths_from_t1
from run_benchmark import _ModelArgs, actual_length, load_examples

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("time_probe")


def sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def run_one(sampler, cfg: CDFRemaskingSamplerConfig, prompt_ids: list[int]) -> dict:
    model, tok = sampler.model, sampler.tokenizer
    eos_id = tok.eos_token_id

    t0 = sync()
    inputs = [torch.as_tensor(prompt_ids, dtype=torch.long, device=model.device)]
    pred_lens = _predict_lengths_from_t1(
        model, inputs, tok.mask_token_id, eos_id, cfg.max_new_tokens,
        quantile=cfg.cdf_quantile, right_shift_logits=cfg.right_shift_logits,
    )
    cropped = max(pred_lens)
    ccfg = copy(cfg)
    ccfg.remasking = "low_confidence"
    ccfg.max_new_tokens = cropped
    ccfg.steps = cropped
    t1 = sync()
    out = MDLMSampler.sample(sampler, inputs, ccfg, return_dict=True)
    t2 = sync()

    seq = out.sequences[0].tolist()
    return {
        "probe_s": t1 - t0,
        "denoise_s": t2 - t1,
        "total_s": t2 - t0,
        "cropped_len": cropped,
        "prompt_len": len(prompt_ids),
        "etoken": actual_length(seq, len(prompt_ids), eos_id),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["gsm8k", "math500", "mbpp", "humaneval"])
    p.add_argument("--n", nargs="+", type=int, default=[50, 50, 50, 30])
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--gen-length", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--q", type=float, default=0.90)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("results/probe_timing/probe_timing.json"))
    args = p.parse_args()
    assert len(args.tasks) == len(args.n)

    model_args = _ModelArgs(model_name_or_path=args.model)
    model = dllm.utils.get_model(model_args=model_args).eval().to("cuda")
    tok = dllm.utils.get_tokenizer(model_args=model_args)
    sampler = CDFRemaskingSampler(model=model, tokenizer=tok)
    cfg = CDFRemaskingSamplerConfig(
        steps=args.gen_length, max_new_tokens=args.gen_length,
        block_size=args.block_size, temperature=0.0, cdf_quantile=args.q,
    )

    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "model": args.model, "gen_length": args.gen_length,
        "block_size": args.block_size, "q": args.q, "batch_size": 1,
        "warmup": args.warmup,
    }
    log.info("meta %s", meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for task, n in zip(args.tasks, args.n):
        prefix = "" if task in ("humaneval", "mbpp") else "<reasoning>"
        exs = load_examples(task, tok, n, assistant_prefix=prefix)
        # Warmup on the task's own first prompts: excludes first-call CUDA/kernel
        # overhead and lets the allocator settle at this task's shapes.
        for ex in exs[: args.warmup]:
            w = run_one(sampler, cfg, ex["prompt_ids"])
            log.info("[%s warmup] probe=%.4fs total=%.2fs", task, w["probe_s"], w["total_s"])
        for i, ex in enumerate(exs):
            r = run_one(sampler, cfg, ex["prompt_ids"])
            r.update(task=task, doc_id=ex["doc_id"])
            records.append(r)
            log.info("[%s %d/%d] L=%d probe=%.4fs denoise=%.2fs share=%.3f%%",
                     task, i + 1, n, r["cropped_len"], r["probe_s"], r["denoise_s"],
                     100 * r["probe_s"] / r["total_s"])
            with open(args.out, "w") as f:
                json.dump({"meta": meta, "records": records}, f, indent=1)


if __name__ == "__main__":
    main()
