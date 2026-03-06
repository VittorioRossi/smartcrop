#!/usr/bin/env python3
"""
SLURM-aware Benchmark Runner for SmartCrop: Zero-Shot Length Prediction

This script runs all benchmark evaluations with proper multi-GPU support,
structured result saving, and log merging.

Usage:
    # Single GPU (local testing)
    python scripts/run_slurm_benchmarks.py --device cuda

    # Multi-GPU (local testing)
    python scripts/run_slurm_benchmarks.py --num_processes 2 --master_port 29500

    # Specific task/quantile (for testing)
    python scripts/run_slurm_benchmarks.py --tasks gsm8k_cot_llama --quantiles 0.99

    # SLURM submission (recommended)
    sbatch scripts/slurm/run_benchmarks.slurm

    # Resume from previous run
    python scripts/run_slurm_benchmarks.py --output_dir results/preprint/453436_benchmark

Features:
    - Proper multi-GPU data sharding via accelerate launch (internally)
    - Checkpoint resume: skips completed tasks based on results_*.json
    - Structured output: results/preprint/{experiment}/by_task/{task}/q{quantile}/
    - Complete per-sample results (FLOPs, queries, answers)
    - Automatic log merging after each task/quantile
    - Aggregated summary across all runs
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_llms.utils.experiment_dirs import ExperimentDirectoryManager
from diffusion_llms.utils.merge_logs import merge_rank_logs, verify_merged_logs

# NOTE: We do NOT create an Accelerator here. Each subprocess (eval.py) manages its own
# distributed environment via accelerate launch. Creating one here causes NCCL errors
# when subprocesses exit at different times.

# Default configuration
DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_QUANTILES = [0.0, 0.5, 0.75, 0.9, 0.95, 0.99]

# Task configurations (task_name: {max_tokens, steps})
# Note: gsm8k_cot_llama uses the instruct prompt format ("The final answer is...")
TASK_CONFIGS = {
    "humaneval_instruct_local": {"max_tokens": 512, "steps": 512},
    "gsm8k_cot_llama": {"max_tokens": 256, "steps": 256},
    "ifeval": {"max_tokens": 1280, "steps": 1280},
    "longformqa": {"max_tokens": 512, "steps": 128},
}


def is_task_completed(exp_dirs: ExperimentDirectoryManager, task: str, quantile: float) -> bool:
    """Check if a task/quantile combination has already been completed.

    A task is considered complete if lm-eval results file exists (results_*.json).
    This is more reliable than checking samples.jsonl/metrics.json since those are
    created after the run in post-processing which may not complete if job is killed.
    """
    task_dir = exp_dirs.get_task_quantile_dir(task, quantile)
    if not task_dir.exists():
        return False

    # Check for lm-eval results files (created by lm-eval at the end of evaluation)
    lm_eval_results = list(task_dir.glob("**/results_*.json"))
    if lm_eval_results:
        print(f"  [RESUME CHECK] Found results for {task} q={quantile}: {lm_eval_results[0].name}")
        return True

    return False


def run_single_evaluation(
    model: str,
    task: str,
    quantile: float,
    exp_dirs: ExperimentDirectoryManager,
    device: str = "cuda",
    num_fewshot: int = 0,
    seed: int = 0,
    skip_completed: bool = True,
    num_processes: int = 1,
    master_port: int = 29500,
) -> dict:
    """Run a single (task, quantile) evaluation.

    Args:
        model: Model path or name
        task: Task name (e.g., "gsm8k_cot")
        quantile: EOS quantile value
        exp_dirs: Experiment directory manager
        device: Device to use
        num_fewshot: Number of few-shot examples
        seed: Random seed
        skip_completed: Whether to skip already-completed tasks
        num_processes: Number of GPU processes for accelerate
        master_port: Port for distributed communication

    Returns:
        Dict with run results (status, metrics, etc.)
    """
    # Check if already completed (for checkpoint resume)
    if skip_completed and is_task_completed(exp_dirs, task, quantile):
        print(f"\n{'='*60}")
        print(f"SKIPPING (already completed): {task} | Quantile: {quantile}")
        print(f"{'='*60}")
        return {
            "task": task,
            "quantile": quantile,
            "status": "skipped",
            "return_code": 0,
            "task_dir": str(exp_dirs.setup_task_quantile(task, quantile)),
        }
    
    task_config = TASK_CONFIGS.get(task, {"max_tokens": 512, "steps": 512})
    max_tokens = task_config["max_tokens"]
    steps = task_config["steps"]

    # Setup directories
    task_dir = exp_dirs.setup_task_quantile(task, quantile)
    rank_log_dir = exp_dirs.get_rank_log_dir(task, quantile)
    samples_path = exp_dirs.get_samples_path(task, quantile)
    metrics_path = exp_dirs.get_metrics_path(task, quantile)

    # Build model args with task metadata
    model_args = ",".join([
        f"pretrained={model}",
        "generation_mode=zero_shot",
        f"eos_quantile={quantile}",
        "safe_margin=0",
        "measure_flops=True",
        f"max_new_tokens={max_tokens}",
        "temperature=0.1",
        f"steps={steps}",
        f"block_length={max_tokens}",
        f"gen_length={max_tokens}",  # Required for proper generation length
        "trust_remote_code=True",
        "logits_eos_inf=True",
        # Task metadata for logging
        f"task_name={task}",
        # Log path (rank suffix added automatically in multi-GPU)
        f"debug_logpath={rank_log_dir}/generation.jsonl",
    ])

    # Build command with accelerate launch for multi-GPU support
    # This ensures proper distributed coordination for each evaluation task
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
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "diffusion_llms" / "eval.py"),
        ]

    # Add eval.py arguments
    cmd.extend([
        "--include_path", str(PROJECT_ROOT / "diffusion_llms" / "tasks"),
        "--tasks", task,
        "--model", "llada",
        "--device", device,
        "--num_fewshot", str(num_fewshot),
        "--model_args", model_args,
        "--seed", str(seed),
        "--output_path", str(task_dir),
        "--confirm_run_unsafe_code",
    ])

    print(f"\n{'='*60}")
    print(f"Running: {task} | Quantile: {quantile}")
    print(f"Output: {task_dir}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))

    # NOTE: No synchronization needed here. The eval.py subprocess handles its own
    # distributed coordination via accelerate. When subprocess.run() returns, the
    # evaluation is complete on this process.

    run_result = {
        "task": task,
        "quantile": quantile,
        "status": "success" if result.returncode == 0 else "failed",
        "return_code": result.returncode,
        "task_dir": str(task_dir),
    }

    if result.returncode == 0:
        # Merge rank logs into single samples.jsonl
        num_merged = merge_rank_logs(rank_log_dir, samples_path)
        verification = verify_merged_logs(samples_path)

        run_result["samples_count"] = num_merged
        run_result["samples_valid"] = verification["valid"]

        if not verification["valid"]:
            print(f"WARNING: Log verification failed!")
            print(f"  Duplicates: {verification['duplicates']}")
            print(f"  Gaps: {verification['gaps']}")

        # Copy lm-eval metrics if they exist
        lm_eval_results = list(task_dir.glob("results_*.json"))
        if lm_eval_results:
            # Copy the latest results file
            latest_results = max(lm_eval_results, key=lambda p: p.stat().st_mtime)
            with open(latest_results) as f:
                lm_eval_metrics = json.load(f)
            with open(metrics_path, "w") as f:
                json.dump(lm_eval_metrics, f, indent=2)
            run_result["metrics_path"] = str(metrics_path)

        print(f"SUCCESS: {task} q={quantile} ({num_merged} samples)")
    else:
        print(f"FAILED: {task} q={quantile}")

    return run_result


def run_all_benchmarks(
    model: str,
    tasks: list[str],
    quantiles: list[float],
    exp_dirs: ExperimentDirectoryManager,
    device: str = "cuda",
    seed: int = 0,
    skip_completed: bool = True,
    num_processes: int = 1,
    master_port: int = 29500,
) -> list[dict]:
    """Run all benchmark evaluations.

    Args:
        model: Model path or name
        tasks: List of tasks to run
        quantiles: List of quantiles to test
        exp_dirs: Experiment directory manager
        device: Device to use
        seed: Random seed for reproducibility
        skip_completed: Whether to skip already-completed tasks
        num_processes: Number of GPU processes for accelerate
        master_port: Port for distributed communication

    Returns:
        List of run results
    """
    all_results = []

    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"# TASK: {task}")
        print(f"{'#'*60}")

        for quantile in quantiles:
            result = run_single_evaluation(
                model=model,
                task=task,
                quantile=quantile,
                exp_dirs=exp_dirs,
                device=device,
                seed=seed,
                skip_completed=skip_completed,
                num_processes=num_processes,
                master_port=master_port,
            )
            all_results.append(result)

    return all_results


def create_summary(results: list[dict], exp_dirs: ExperimentDirectoryManager) -> dict:
    """Create and save experiment summary.

    Args:
        results: List of run results
        exp_dirs: Experiment directory manager

    Returns:
        Summary dictionary
    """
    success_count = sum(1 for r in results if r["status"] == "success")
    total_count = len(results)

    summary = {
        "experiment_name": exp_dirs.experiment_name,
        "timestamp": datetime.now().isoformat(),
        "total_runs": total_count,
        "successful_runs": success_count,
        "failed_runs": total_count - success_count,
        "results": results,
    }

    # Group by task for easier analysis
    by_task = {}
    for r in results:
        task = r["task"]
        if task not in by_task:
            by_task[task] = []
        by_task[task].append({
            "quantile": r["quantile"],
            "status": r["status"],
            "samples_count": r.get("samples_count"),
        })
    summary["by_task"] = by_task

    exp_dirs.save_summary(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="SmartCrop SLURM Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to evaluate (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=list(TASK_CONFIGS.keys()),
        choices=list(TASK_CONFIGS.keys()),
        help="Tasks to evaluate (default: all tasks)",
    )

    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_QUANTILES,
        help=f"EOS quantiles to test (default: {DEFAULT_QUANTILES})",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: results/preprint/<timestamp>_benchmark)",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name (default: <timestamp>_benchmark)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume (re-run all tasks even if completed)",
    )

    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of GPU processes for accelerate (default: 1, use 2 for multi-GPU)",
    )

    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Port for distributed communication (default: 29500)",
    )

    args = parser.parse_args()

    # Setup experiment directories
    if args.output_dir:
        # If output_dir is provided, use it directly as the root
        exp_dirs = ExperimentDirectoryManager(
            base_dir=Path(args.output_dir).parent.parent,
            experiment_name=Path(args.output_dir).name,
            submission_name=Path(args.output_dir).parent.name,
        )
        # Override the root to match exact output_dir
        exp_dirs.root = Path(args.output_dir)
        exp_dirs.by_task = exp_dirs.root / "by_task"
        exp_dirs.aggregated = exp_dirs.root / "aggregated"
    else:
        exp_dirs = ExperimentDirectoryManager(
            base_dir=PROJECT_ROOT / "results",
            experiment_name=args.experiment_name,
        )

    exp_dirs.setup()

    # Save configuration
    config = {
        "model": args.model,
        "tasks": args.tasks,
        "quantiles": args.quantiles,
        "device": args.device,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "task_configs": TASK_CONFIGS,
    }
    exp_dirs.save_config(config)

    print("=" * 60)
    print("SmartCrop SLURM Benchmark Runner")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Tasks: {args.tasks}")
    print(f"Quantiles: {args.quantiles}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Num Processes: {args.num_processes}")
    print(f"Master Port: {args.master_port}")
    print(f"Output: {exp_dirs.root}")
    print("=" * 60)

    # Run all benchmarks
    skip_completed = not args.no_resume
    if skip_completed:
        print("Checkpoint resume ENABLED: will skip already-completed tasks")
    else:
        print("Checkpoint resume DISABLED: will re-run all tasks")
    print("=" * 60)
    
    results = run_all_benchmarks(
        model=args.model,
        tasks=args.tasks,
        quantiles=args.quantiles,
        exp_dirs=exp_dirs,
        device=args.device,
        seed=args.seed,
        skip_completed=skip_completed,
        num_processes=args.num_processes,
        master_port=args.master_port,
    )

    # Create and save summary
    summary = create_summary(results, exp_dirs)

    # Print final summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"Completed: {summary['successful_runs']}/{summary['total_runs']}")

    for r in results:
        status_icon = "[OK]" if r["status"] == "success" else "[FAIL]"
        samples_info = f"({r.get('samples_count', '?')} samples)" if r["status"] == "success" else ""
        print(f"  {status_icon} {r['task']} q={r['quantile']} {samples_info}")

    print(f"\nResults saved to: {exp_dirs.root}")
    print(f"Summary: {exp_dirs.aggregated / 'summary.json'}")


if __name__ == "__main__":
    main()
