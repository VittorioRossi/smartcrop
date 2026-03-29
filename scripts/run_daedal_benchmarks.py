#!/usr/bin/env python3
"""
DAEDAL Benchmark Runner (Rebuttal Baseline)

Evaluates DAEDAL (Li et al., 2025, arXiv 2508.00819) on the same four benchmarks
used by SmartCrop (Table 2) so we can provide a direct comparison in the rebuttal.

DAEDAL is a single method (no quantile sweep), so this runs one evaluation per task.

Usage:
    # Local single GPU
    python scripts/run_daedal_benchmarks.py --device cuda

    # Specific tasks only
    python scripts/run_daedal_benchmarks.py --tasks gsm8k_cot_llama ifeval

    # Resume from previous run
    python scripts/run_daedal_benchmarks.py --output_dir results/rebuttal/daedal

    # SLURM submission
    sbatch scripts/slurm/run_daedal_benchmarks.slurm
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

# Task configs for DAEDAL rebuttal comparison.
# max_tokens follows DAEDAL paper where available; 2048 for new tasks.
# num_fewshot: None = use task default (e.g. BBH uses 3-shot by default)
TASK_CONFIGS = {
    "humaneval_instruct_local":    {"max_tokens": 2048, "block_length": 32, "num_fewshot": 0},
    "gsm8k_cot_llama":             {"max_tokens": 2048, "block_length": 32, "num_fewshot": 0},
    "ifeval":                      {"max_tokens": 2048, "block_length": 32, "num_fewshot": 0},
    "bbh_cot_fewshot":   {"max_tokens": 1024, "block_length": 32, "num_fewshot": 3},
    "math500":                     {"max_tokens": 2048, "block_length": 32, "num_fewshot": 0},
}

# DAEDAL hyperparameters (reference values from Li et al. eval scripts)
DAEDAL_HPARAMS = {
    "initial_gen_length":            64,
    "high_conf_threshold":           0.90,
    "low_conf_threshold":            0.10,
    "eos_confidence_threshold":      0.5,
    "expand_eos_confidence_threshold": 0.9,
    "expansion_factor":              8,
    "eos_check_tokens":              32,
}

# Sentinel quantile value written to logs so DAEDAL rows appear alongside
# SmartCrop rows in downstream analysis scripts.
DAEDAL_QUANTILE_SENTINEL = -1.0


def is_task_completed(exp_dirs: ExperimentDirectoryManager, task: str) -> bool:
    task_dir = exp_dirs.get_task_quantile_dir(task, DAEDAL_QUANTILE_SENTINEL)
    if not task_dir.exists():
        return False
    lm_eval_results = list(task_dir.glob("**/results_*.json"))
    if lm_eval_results:
        print(f"  [RESUME] Already completed: {task}")
        return True
    return False


def run_single_evaluation(
    model: str,
    task: str,
    exp_dirs: ExperimentDirectoryManager,
    device: str = "cuda",
    num_fewshot: int = 0,
    seed: int = 0,
    skip_completed: bool = True,
    num_processes: int = 1,
    master_port: int = 29500,
) -> dict:
    if skip_completed and is_task_completed(exp_dirs, task):
        return {
            "task": task,
            "status": "skipped",
            "return_code": 0,
            "task_dir": str(exp_dirs.setup_task_quantile(task, DAEDAL_QUANTILE_SENTINEL)),
        }

    task_dir     = exp_dirs.setup_task_quantile(task, DAEDAL_QUANTILE_SENTINEL)
    rank_log_dir = exp_dirs.get_rank_log_dir(task, DAEDAL_QUANTILE_SENTINEL)
    samples_path = exp_dirs.get_samples_path(task, DAEDAL_QUANTILE_SENTINEL)
    metrics_path = exp_dirs.get_metrics_path(task, DAEDAL_QUANTILE_SENTINEL)

    task_cfg    = TASK_CONFIGS.get(task, {"max_tokens": 512, "block_length": 32, "num_fewshot": 0})
    max_tokens   = task_cfg["max_tokens"]
    block_length = task_cfg["block_length"]
    # Use per-task num_fewshot unless the caller overrides
    effective_fewshot = num_fewshot if num_fewshot != 0 else task_cfg.get("num_fewshot", 0)

    model_args_parts = [
        f"pretrained={model}",
        "generation_mode=daedal",
        f"max_new_tokens={max_tokens}",
        f"block_length={block_length}",
        # DAEDAL hyperparameters
        f"initial_gen_length={DAEDAL_HPARAMS['initial_gen_length']}",
        f"high_conf_threshold={DAEDAL_HPARAMS['high_conf_threshold']}",
        f"low_conf_threshold={DAEDAL_HPARAMS['low_conf_threshold']}",
        f"eos_confidence_threshold={DAEDAL_HPARAMS['eos_confidence_threshold']}",
        f"expand_eos_confidence_threshold={DAEDAL_HPARAMS['expand_eos_confidence_threshold']}",
        f"expansion_factor={DAEDAL_HPARAMS['expansion_factor']}",
        f"eos_check_tokens={DAEDAL_HPARAMS['eos_check_tokens']}",
        "temperature=0.0",
        "cfg=0.0",
        "trust_remote_code=True",
        # Logging
        f"task_name={task}",
        f"debug_logpath={rank_log_dir}/generation.jsonl",
    ]
    model_args = ",".join(model_args_parts)

    if num_processes > 1:
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(num_processes),
            "--num_machines", "1",
            "--machine_rank", "0",
            "--main_process_port", str(master_port),
            "--mixed_precision", "bf16",
            str(PROJECT_ROOT / "diffusion_llms" / "eval.py"),
        ]
    else:
        cmd = [sys.executable, str(PROJECT_ROOT / "diffusion_llms" / "eval.py")]

    cmd.extend([
        "--include_path", str(PROJECT_ROOT / "diffusion_llms" / "tasks"),
        "--tasks", task,
        "--model", "llada",
        "--device", device,
        "--num_fewshot", str(effective_fewshot),
        "--model_args", model_args,
        "--seed", str(seed),
        "--output_path", str(task_dir),
        "--confirm_run_unsafe_code",
    ])

    print(f"\n{'='*60}")
    print(f"DAEDAL  |  Task: {task}")
    print(f"Output: {task_dir}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))

    run_result = {
        "task": task,
        "status": "success" if result.returncode == 0 else "failed",
        "return_code": result.returncode,
        "task_dir": str(task_dir),
    }

    if result.returncode == 0:
        num_merged = merge_rank_logs(rank_log_dir, samples_path)
        verification = verify_merged_logs(samples_path)
        run_result["samples_count"] = num_merged
        run_result["samples_valid"] = verification["valid"]
        if not verification["valid"]:
            print(f"WARNING: Log verification failed for {task}")

        lm_eval_results = list(task_dir.glob("results_*.json"))
        if lm_eval_results:
            latest = max(lm_eval_results, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                lm_eval_metrics = json.load(f)
            with open(metrics_path, "w") as f:
                json.dump(lm_eval_metrics, f, indent=2)
            run_result["metrics_path"] = str(metrics_path)

        print(f"SUCCESS: {task} ({num_merged} samples)")
    else:
        print(f"FAILED: {task}")

    return run_result


def main():
    parser = argparse.ArgumentParser(description="DAEDAL benchmark runner (rebuttal baseline)")
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--tasks",        nargs="+", default=list(TASK_CONFIGS.keys()))
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--output_dir",   default=None)
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--num_processes",type=int, default=1)
    parser.add_argument("--master_port",  type=int, default=29500)
    parser.add_argument("--no_resume",    action="store_true")
    args = parser.parse_args()

    run_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = args.output_dir or str(PROJECT_ROOT / "results" / "rebuttal" / f"daedal_{run_id}")
    exp_dirs = ExperimentDirectoryManager(output_dir)

    print("=" * 60)
    print("DAEDAL Benchmark Evaluation (Rebuttal Baseline)")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Tasks:   {args.tasks}")
    print(f"Output:  {output_dir}")
    print(f"DAEDAL hparams: {DAEDAL_HPARAMS}")
    print("=" * 60)

    all_results = []
    for task in args.tasks:
        result = run_single_evaluation(
            model=args.model,
            task=task,
            exp_dirs=exp_dirs,
            device=args.device,
            seed=args.seed,
            skip_completed=not args.no_resume,
            num_processes=args.num_processes,
            master_port=args.master_port,
        )
        all_results.append(result)

    # Save run summary
    summary = {
        "run_id": run_id,
        "model": args.model,
        "method": "daedal",
        "daedal_hparams": DAEDAL_HPARAMS,
        "tasks": args.tasks,
        "results": all_results,
        "completed_at": datetime.now().isoformat(),
    }
    summary_path = Path(output_dir) / "daedal_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    n_ok   = sum(1 for r in all_results if r["status"] in ("success", "skipped"))
    n_fail = sum(1 for r in all_results if r["status"] == "failed")
    print(f"\nDone. {n_ok}/{len(all_results)} tasks completed, {n_fail} failed.")
    print(f"Summary: {summary_path}")

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
