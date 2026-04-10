#!/usr/bin/env python3
"""
LLaDA Canvas Size Sweep — Correlation Analysis (Rebuttal)

Tests how LLaDA performance varies with canvas size, complementing the DREAM
canvas sweep. The hypothesis is that LLaDA maintains or improves accuracy on
shorter canvases due to EoS-as-padding training, while DREAM degrades.

Canvas sizes tested mirror the DREAM sweep plus the SC-0.9 mean predicted
length for HumanEval (~379 tokens at τ=0.9) to make the comparison direct.

Uses SmartCrop (generation_mode=zero_shot, eos_quantile=0.9) so the predicted
length is always at most canvas_size tokens.

Usage:
    python scripts/run_llada_canvas_sweep.py
    python scripts/run_llada_canvas_sweep.py --model GSAI-ML/LLaDA-1.5
    sbatch scripts/slurm/run_llada_canvas_sweep.slurm
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_llms.utils.experiment_dirs import ExperimentDirectoryManager
from diffusion_llms.utils.merge_logs import merge_rank_logs, verify_merged_logs

DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_CANVAS_SIZES = [128, 256, 379, 512, 1024]

TASK = "humaneval_instruct_local"
EOS_QUANTILE = 0.9

# Sentinel: encode canvas size as negative float for ExperimentDirectoryManager
def canvas_sentinel(canvas_size: int) -> float:
    return -(canvas_size / 10000.0)


def is_completed(exp_dirs: ExperimentDirectoryManager, canvas_size: int) -> bool:
    q = canvas_sentinel(canvas_size)
    task_dir = exp_dirs.get_task_quantile_dir(TASK, q)
    if not task_dir.exists():
        return False
    if list(task_dir.glob("**/results_*.json")):
        print(f"  [RESUME] Already completed: canvas={canvas_size}")
        return True
    return False


def run_canvas_eval(
    model: str,
    canvas_size: int,
    exp_dirs: ExperimentDirectoryManager,
    device: str = "cuda",
    seed: int = 0,
    skip_completed: bool = True,
) -> dict:
    q = canvas_sentinel(canvas_size)

    if skip_completed and is_completed(exp_dirs, canvas_size):
        return {"canvas_size": canvas_size, "status": "skipped", "return_code": 0}

    task_dir = exp_dirs.setup_task_quantile(TASK, q)
    rank_log_dir = exp_dirs.get_rank_log_dir(TASK, q)
    samples_path = exp_dirs.get_samples_path(TASK, q)
    metrics_path = exp_dirs.get_metrics_path(TASK, q)

    model_args = ",".join([
        f"pretrained={model}",
        "generation_mode=zero_shot",
        f"max_new_tokens={canvas_size}",
        f"steps={canvas_size}",
        f"block_length={canvas_size}",
        f"eos_quantile={EOS_QUANTILE}",
        "temperature=0.0",
        "cfg=0.0",
        "trust_remote_code=True",
        f"task_name={TASK}",
        f"debug_logpath={rank_log_dir}/generation.jsonl",
        "measure_flops=True",
        "logits_eos_inf=True",
    ])

    cmd = [sys.executable, str(PROJECT_ROOT / "diffusion_llms" / "eval.py"),
        "--include_path", str(PROJECT_ROOT / "diffusion_llms" / "tasks"),
        "--tasks", TASK,
        "--model", "llada",
        "--device", device,
        "--num_fewshot", "0",
        "--model_args", model_args,
        "--seed", str(seed),
        "--output_path", str(task_dir),
        "--confirm_run_unsafe_code",
    ]

    print(f"\n{'='*60}")
    print(f"LLaDA canvas sweep  |  model={model}  |  canvas={canvas_size}")
    print(f"Output: {task_dir}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))

    run_result = {
        "canvas_size": canvas_size,
        "status": "success" if result.returncode == 0 else "failed",
        "return_code": result.returncode,
        "task_dir": str(task_dir),
    }

    if result.returncode == 0:
        num_merged = merge_rank_logs(rank_log_dir, samples_path)
        verification = verify_merged_logs(samples_path)
        run_result["samples_count"] = num_merged
        run_result["samples_valid"] = verification["valid"]

        results_files = list(task_dir.glob("**/results_*.json"))
        if results_files:
            latest = max(results_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                lm_metrics = json.load(f)
            with open(metrics_path, "w") as f:
                json.dump(lm_metrics, f, indent=2)
            run_result["metrics"] = lm_metrics.get("results", {}).get(TASK, {})

        print(f"SUCCESS: canvas={canvas_size} ({num_merged} samples)")
    else:
        print(f"FAILED: canvas={canvas_size}")

    return run_result


def main():
    parser = argparse.ArgumentParser(description="LLaDA canvas-size sweep (rebuttal)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--canvas_sizes", nargs="+", type=int, default=DEFAULT_CANVAS_SIZES)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    run_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    model_tag = args.model.replace("/", "__")
    output_dir = args.output_dir or str(
        PROJECT_ROOT / "results" / "rebuttal" / f"llada_canvas_sweep_{model_tag}_{run_id}"
    )
    exp_dirs = ExperimentDirectoryManager(output_dir)

    print("=" * 60)
    print("LLaDA Canvas Size Sweep (Rebuttal)")
    print("=" * 60)
    print(f"Model:        {args.model}")
    print(f"Canvas sizes: {args.canvas_sizes}")
    print(f"  (379 = SC-0.9 mean predicted length for HumanEval)")
    print(f"Task:         {TASK}  (EOS quantile τ={EOS_QUANTILE})")
    print(f"Output:       {output_dir}")
    print("=" * 60)

    all_results = []
    for canvas_size in args.canvas_sizes:
        result = run_canvas_eval(
            model=args.model,
            canvas_size=canvas_size,
            exp_dirs=exp_dirs,
            device=args.device,
            seed=args.seed,
            skip_completed=not args.no_resume,
        )
        all_results.append(result)

    summary = {
        "run_id": run_id,
        "model": args.model,
        "method": "llada_canvas_sweep",
        "canvas_sizes": args.canvas_sizes,
        "task": TASK,
        "eos_quantile": EOS_QUANTILE,
        "smartcrop_mean_length_humaneval": 379,
        "results": all_results,
        "completed_at": datetime.now().isoformat(),
    }
    summary_path = Path(output_dir) / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Canvas':>8}  {'pass@1':>8}  {'gen_len':>8}  {'Status'}")
    print("-" * 40)
    for r in all_results:
        metrics = r.get("metrics", {})
        score = metrics.get("pass@1,create_test", float("nan"))
        gen_len = metrics.get("gen_len,none", float("nan"))
        marker = " <- SC-0.9 mean" if r["canvas_size"] == 379 else ""
        print(f"{r['canvas_size']:>8}  {score:>8.4f}  {gen_len:>8.1f}  {r['status']}{marker}")

    n_ok = sum(1 for r in all_results if r["status"] in ("success", "skipped"))
    n_fail = sum(1 for r in all_results if r["status"] == "failed")
    print(f"\nDone. {n_ok}/{len(all_results)} completed, {n_fail} failed.")
    print(f"Summary: {summary_path}")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
