"""Unified generation + evaluation script for masked diffusion LM benchmarks.

Usage:
    python scripts/run_benchmark.py --task gsm8k --sampler cdf
    python scripts/run_benchmark.py --task humaneval --sampler cdf --eos-free
    python scripts/run_benchmark.py --task math500 --n 500
    python scripts/run_benchmark.py --task gsm8k --sampler cdf --threshold 0.9

Supported tasks:    gsm8k, math500, humaneval, mbpp
Supported samplers: cdf, low_conf
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import dataclasses
import json
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
import dllm

from copy import copy

def _is_dream(model_name: str) -> bool:
    return "dream" in model_name.lower()

from diffusion_llms.generators import (
    CDFRemaskingSampler, CDFRemaskingSamplerConfig,
    CDFRemaskingSamplerBatched,
)
from diffusion_llms.generators.cdf_remasking_generator import _predict_lengths_from_t1
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"

SAMPLER_REGISTRY = {
    "cdf":      (CDFRemaskingSampler, CDFRemaskingSamplerConfig),
    "low_conf": (MDLMSampler,        MDLMSamplerConfig),
}

SUPPORTED_TASKS = ["gsm8k", "math500", "humaneval", "mbpp"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_examples(task, tokenizer, n, seed=42, assistant_prefix=""):
    from lm_eval.tasks import TaskManager
    import lm_eval.tasks as lm_tasks

    task_manager = TaskManager(
        include_path=str(PROJECT_ROOT / "diffusion_llms" / "tasks")
    )
    task_dict = lm_tasks.get_task_dict([task], task_manager)
    lm_task   = task_dict[task]

    docs = list(lm_task.test_docs())
    docs = sorted(docs, key=lambda d: d.get("task_id", d.get("idx", 0)))[:n]

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
        examples.append({"prompt_ids": ids, "doc_id": doc_id, "doc": doc})

    return examples


# ── Ground truth extraction ───────────────────────────────────────────────────

def get_gt(task, doc):
    if task == "gsm8k":
        return doc.get("answer", "")
    if task == "math500":
        return doc.get("answer", "")
    if task == "humaneval":
        return {
            "test":        doc.get("test", ""),
            "entry_point": doc.get("entry_point", ""),
            "prompt":      doc.get("prompt", ""),
        }
    if task == "mbpp":
        return {
            "test_list": doc.get("test_list", []),
        }
    return ""


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_all(task: str, results: list) -> list:
    if task in ("gsm8k", "math500"):
        from metrics.math500 import score_batch
    elif task == "humaneval":
        from metrics.humaneval import score_batch
    elif task == "mbpp":
        from metrics.mbpp import score_batch
    return score_batch(results)


# ── Config helpers ────────────────────────────────────────────────────────────

def build_config(ConfigClass, base_kwargs, extra_kwargs):
    accepted = {f.name for f in dataclasses.fields(ConfigClass)}
    merged   = {k: v for k, v in {**base_kwargs, **extra_kwargs}.items() if k in accepted}
    return ConfigClass(**merged)


def _git_info(repo_dir: Path) -> dict:
    def _run(cmd):
        try:
            return subprocess.check_output(
                cmd, cwd=repo_dir, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty":  bool(_run(["git", "status", "--porcelain"])),
    }


def _package_versions() -> dict:
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for pkg in ("transformers", "dllm"):
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = None
    return versions


def _write_json(path: Path, obj) -> None:
    """Write via a temp file and rename, so a job killed mid-write keeps the last complete file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def actual_length(token_ids, prompt_len, eos_id):
    for i, t in enumerate(token_ids[prompt_len:]):
        if t == eos_id:
            return i + 1
    return len(token_ids) - prompt_len


# ── Main ──────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _ModelArgs:
    model_name_or_path: str = DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    type=str, required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--sampler", type=str, default="cdf", choices=sorted(SAMPLER_REGISTRY))
    parser.add_argument("--model",   type=str, default=DEFAULT_MODEL)
    parser.add_argument("--n",           type=int, default=50)
    parser.add_argument("--gen-length",  type=int, default=1024, dest="gen_length")
    parser.add_argument("--steps",       type=int, default=128)
    parser.add_argument("--block-size",  type=int, default=32,   dest="block_size")
    parser.add_argument("--q",           type=float, default=0.90)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--out",         type=Path, default=None)
    parser.add_argument("--batch-size",  type=int,  default=1,  dest="batch_size")
    # cdf-specific
    parser.add_argument("--eos-free",  action="store_true", default=False, dest="eos_free")
    parser.add_argument("--threshold", type=float, default=None,
                        help="confidence-threshold parallel decoding (e.g. 0.9); default one token per step")
    parser.add_argument("--no-crop", action="store_true", default=False, dest="no_crop",
                        help="skip the t=1 probe and denoise the full --gen-length canvas")
    parser.add_argument("--right-shift-logits", action="store_true", default=False,
                        dest="right_shift_logits")
    # shared
    parser.add_argument("--assistant-prefix", type=str, default=None,
                        dest="assistant_prefix")
    args = parser.parse_args()
    run_start = datetime.now(timezone.utc)
    if args.threshold is not None and (args.sampler != "cdf" or args.batch_size > 1):
        parser.error("--threshold requires --sampler cdf and --batch-size 1")
    if args.no_crop and (args.sampler != "cdf" or args.batch_size > 1):
        parser.error("--no-crop requires --sampler cdf and --batch-size 1")

    # Default assistant prefix: reasoning tag for math, empty for code tasks
    if args.assistant_prefix is None:
        args.assistant_prefix = "" if args.task in ("humaneval", "mbpp") else "<reasoning>"

    # DREAM models need AR-shift logits; auto-enable unless user already passed the flag
    if _is_dream(args.model) and not args.right_shift_logits:
        args.right_shift_logits = True

    if args.out is None:
        model_short = args.model.split("/")[-1].lower().replace("-", "_")
        tag = args.sampler
        if args.sampler == "cdf":
            tag += "_nocrop" if args.no_crop else f"_q{int(args.q * 100)}"
        if args.threshold is not None:
            tag += f"_thr{int(round(args.threshold * 100))}"
        if args.eos_free:
            tag += "_eosfree"
        if args.batch_size > 1:
            tag += f"_b{args.batch_size}"
        args.out = Path(
            f"results/{args.task}/{model_short}_{tag}_n{args.n}_gl{args.gen_length}_s{args.steps}.json"
        )

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if _is_dream(args.model):
        from diffusion_llms.utils import get_model as _get_model, get_tokenizer as _get_tokenizer
        model = _get_model(
            model_name_or_path=args.model,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        tokenizer = _get_tokenizer(model_name_or_path=args.model, trust_remote_code=True)
        # The cached HF DREAM model skips the 2D long→[B,1,L,L] bool conversion that
        # the local copy does. Patch forward to apply it so both the t=1 probe and
        # MDLMSampler denoising work correctly.
        _orig_fwd = model.forward
        def _dream_fwd_patch(*fwd_args, **fwd_kwargs):
            mask = fwd_kwargs.get('attention_mask', fwd_args[1] if len(fwd_args) > 1 else None)
            if isinstance(mask, torch.Tensor) and mask.dim() == 2:
                mask = torch.logical_and(
                    mask.unsqueeze(1).unsqueeze(-2),
                    mask.unsqueeze(1).unsqueeze(-1),
                ).to(torch.bool)
                if 'attention_mask' in fwd_kwargs:
                    fwd_kwargs['attention_mask'] = mask
                else:
                    fwd_args = list(fwd_args)
                    fwd_args[1] = mask
                    fwd_args = tuple(fwd_args)
            return _orig_fwd(*fwd_args, **fwd_kwargs)
        model.forward = _dream_fwd_patch
    else:
        model_args = _ModelArgs(model_name_or_path=args.model)
        model      = dllm.utils.get_model(model_args=model_args).eval().to(device)
        tokenizer  = dllm.utils.get_tokenizer(model_args=model_args)

    print(f"Loading {args.n} {args.task} examples...")
    examples = load_examples(args.task, tokenizer, args.n, seed=args.seed,
                             assistant_prefix=args.assistant_prefix)
    print(f"Loaded {len(examples)} examples")

    use_batched = args.batch_size > 1 and args.sampler == "cdf"
    SamplerClass, ConfigClass = SAMPLER_REGISTRY[args.sampler]
    if use_batched:
        SamplerClass = CDFRemaskingSamplerBatched
    cfg = build_config(
        ConfigClass,
        base_kwargs=dict(
            steps=args.steps,
            max_new_tokens=args.gen_length,
            block_size=args.block_size,
            temperature=0.0,
        ),
        extra_kwargs=dict(
            cdf_quantile=args.q,
            eos_free=args.eos_free,
            threshold=args.threshold,
            no_crop=args.no_crop,
            right_shift_logits=args.right_shift_logits,
        ),
    )
    sampler = SamplerClass(model=model, tokenizer=tokenizer)
    print(f"Sampler: {args.sampler}  Config: {cfg}")
    print(f"Output:  {args.out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set = set()
    results = []
    if args.out.exists():
        with open(args.out) as f:
            results = json.load(f)
        done_ids = {r["doc_id"] for r in results}
        print(f"Resuming: {len(done_ids)} already done")

    eos_id = tokenizer.eos_token_id

    # Count top-level forward passes (t=1 probe included) per sample.
    nfe = [0]
    def _count_forward(module, fwd_args):
        nfe[0] += 1
    model.register_forward_pre_hook(_count_forward)

    if use_batched:
        pending = [ex for ex in examples if ex["doc_id"] not in done_ids]
        n_buckets = (len(pending) + args.batch_size - 1) // args.batch_size
        print(f"Batched mode: batch_size={args.batch_size}  pending={len(pending)}  buckets={n_buckets}")

        # t=1 + denoising interleaved per bucket
        for start in range(0, len(pending), args.batch_size):
            chunk_idxs = range(start, min(start + args.batch_size, len(pending)))
            bucket_ins = [
                torch.as_tensor(pending[i]["prompt_ids"], dtype=torch.long, device=model.device)
                for i in chunk_idxs
            ]

            pred_lens = _predict_lengths_from_t1(
                model, bucket_ins, tokenizer.mask_token_id, eos_id,
                cfg.max_new_tokens, quantile=cfg.cdf_quantile,
            )
            canvas = max(pred_lens)

            bucket_cfg = copy(cfg)
            bucket_cfg.remasking = "low_confidence"
            bucket_cfg.max_new_tokens = canvas
            bucket_cfg.steps = canvas
            if cfg.eos_free:
                existing = list(bucket_cfg.suppress_tokens or [])
                if eos_id not in existing:
                    existing.append(eos_id)
                bucket_cfg.suppress_tokens = existing

            t0  = time.time()
            out = MDLMSampler.sample(sampler, bucket_ins, bucket_cfg, return_dict=True)
            elapsed_per = (time.time() - t0) / len(chunk_idxs)

            bucket_num = start // args.batch_size + 1
            print(f"[bucket {bucket_num:3d}/{n_buckets}] B={len(chunk_idxs)}  canvas={canvas}  {elapsed_per:.2f}s/sample")

            for local_j, global_i in enumerate(chunk_idxs):
                ex         = pending[global_i]
                seq        = out.sequences[local_j].tolist()
                prompt_ids = ex["prompt_ids"]
                gen_tokens = seq[len(prompt_ids):]
                try:
                    eos_pos = gen_tokens.index(eos_id)
                    gen_tokens_trimmed = gen_tokens[:eos_pos]
                except ValueError:
                    gen_tokens_trimmed = gen_tokens
                response = tokenizer.decode(gen_tokens_trimmed, skip_special_tokens=True).strip()
                gt = get_gt(args.task, ex["doc"])
                results.append({
                    "doc_id":   ex["doc_id"],
                    "response": response,
                    "gt":       gt,
                    "etoken":   actual_length(seq, len(prompt_ids), eos_id),
                    "ntoken":   canvas,
                    "elapsed":  elapsed_per,
                    "sampler":  args.sampler,
                    "task":     args.task,
                })

            _write_json(args.out, results)

        # skip the per-sample loop below
        examples = []

    for i, ex in enumerate(examples):
        prompt_ids = ex["prompt_ids"]
        doc_id     = ex["doc_id"]

        if doc_id in done_ids:
            print(f"[{i+1:3d}/{len(examples)}] doc={doc_id}  skipped")
            continue

        nfe[0] = 0
        t0  = time.time()
        out = sampler.sample([prompt_ids], cfg, return_dict=True)
        seq = out.sequences[0].tolist()

        gen_tokens = seq[len(prompt_ids):]
        try:
            eos_pos           = gen_tokens.index(eos_id)
            gen_tokens_trimmed = gen_tokens[:eos_pos]
        except ValueError:
            gen_tokens_trimmed = gen_tokens

        response = tokenizer.decode(gen_tokens_trimmed, skip_special_tokens=True).strip()
        etoken   = actual_length(seq, len(prompt_ids), eos_id)
        ntoken   = len(seq) - len(prompt_ids)   # total canvas tokens (cropped or full)
        gt       = get_gt(args.task, ex["doc"])
        elapsed  = time.time() - t0

        results.append({
            "doc_id":  doc_id,
            "response": response,
            "gt":       gt,
            "etoken":   etoken,
            "ntoken":   ntoken,
            "elapsed":  elapsed,
            "nfe":      nfe[0],
            "sampler":  args.sampler,
            "task":     args.task,
        })

        _write_json(args.out, results)

        eratio = etoken / ntoken if ntoken > 0 else 0.0
        print(f"[{i+1:3d}/{len(examples)}] doc={doc_id}  etoken={etoken}  ntoken={ntoken}  eratio={eratio:.2f}  nfe={nfe[0]}  {elapsed:.1f}s")

    # ── Summary (Table 1 metrics) ─────────────────────────────────────────────
    etokens  = [r["etoken"]  for r in results]
    ntokens  = [r["ntoken"]  for r in results]
    eratios  = [r["etoken"] / r["ntoken"] for r in results if r["ntoken"] > 0]
    truntime = sum(r["elapsed"] for r in results)

    print(f"\n{'─'*55}")
    print(f"  Etoken  (avg): {np.mean(etokens):.1f}")
    print(f"  Ntoken  (avg): {np.mean(ntokens):.1f}")
    print(f"  Eratio  (avg): {np.mean(eratios):.3f}")
    print(f"  Truntime:      {truntime:.1f}s")
    if all("nfe" in r for r in results):
        print(f"  NFE     (avg): {np.mean([r['nfe'] for r in results]):.1f}")

    try:
        corrects = score_all(args.task, results)
        for r, c in zip(results, corrects):
            r["correct"] = c
        n_correct = sum(corrects)
        label = "Acc" if args.task in ("gsm8k", "math500") else "Pass@1"
        print(f"  {label}:        {100*n_correct/len(results):.1f}%  ({n_correct}/{len(results)})")
    except Exception as e:
        print(f"  Scoring:       skipped ({e})")
    print(f"{'─'*55}")

    _write_json(args.out, results)

    # ── Provenance sidecar ────────────────────────────────────────────────────
    run_end = datetime.now(timezone.utc)
    meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "sampler_config": dataclasses.asdict(cfg),
        "git": _git_info(PROJECT_ROOT),
        "versions": _package_versions(),
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "run_start_utc": run_start.isoformat(),
        "run_end_utc": run_end.isoformat(),
        "wall_time_seconds": (run_end - run_start).total_seconds(),
        "sum_per_sample_elapsed_seconds": truntime,
        "n_samples": len(results),
    }
    meta_path = args.out.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Provenance:    {meta_path}")


if __name__ == "__main__":
    main()
