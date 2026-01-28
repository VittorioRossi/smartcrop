#!/usr/bin/env python3
"""
Unified Experiment Runner for SmartCrop: Zero-Shot Length Prediction for Diffusion LLMs

This script provides a single entrypoint to replicate all experiments from the paper.
Run with --help to see available experiments and options.

Usage:
    # Run all benchmark evaluations
    python scripts/run_experiments.py --experiment benchmarks --device cuda

    # Run a specific benchmark with specific quantile
    python scripts/run_experiments.py --experiment benchmarks --tasks gsm8k_cot --quantiles 0.99 --device cuda

    # Run all experiments (benchmarks + quality-length analysis)
    python scripts/run_experiments.py --experiment all --device cuda

Requirements:
    - NVIDIA GPU with CUDA support (recommended: A100 40GB or higher)
    - ~16GB VRAM minimum for LLaDA-8B-Instruct
    - Python 3.10+
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Default configuration
DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_QUANTILES = [0.0, 0.5, 0.75, 0.9, 0.95, 0.99]

# Task configurations (task_name: {max_tokens, steps})
TASK_CONFIGS = {
    "humaneval_instruct_local": {"max_tokens": 512, "steps": 512},
    "gsm8k_cot": {"max_tokens": 256, "steps": 256},
    "ifeval": {"max_tokens": 1280, "steps": 1280},
    "longformqa": {"max_tokens": 512, "steps": 128},
}


def run_benchmark_evaluation(
    model: str,
    task: str,
    quantiles: list[float],
    device: str,
    output_dir: Path,
    num_fewshot: int = 0,
    seed: int = 0,
):
    """Run lm-eval benchmark for a given task across multiple quantiles."""

    task_config = TASK_CONFIGS.get(task, {"max_tokens": 512, "steps": 512})
    max_tokens = task_config["max_tokens"]
    steps = task_config["steps"]

    results = []

    for q in quantiles:
        print(f"\n{'='*60}")
        print(f"Running: {task} | Quantile: {q}")
        print(f"{'='*60}")

        # Build model args string
        model_args = ",".join([
            f"pretrained={model}",
            "generation_mode=zero_shot",
            f"eos_quantile={q}",
            "safe_margin=0",
            "measure_flops=True",
            f"max_new_tokens={max_tokens}",
            "temperature=0.1",
            f"steps={steps}",
            f"block_length={max_tokens}",
            "trust_remote_code=True",
            "logits_eos_inf=True",
            f"debug_logpath={output_dir}/logs/{task}_q{q}.jsonl",
        ])

        cmd = [
            sys.executable, "-m", "lm_eval",
            "--include_path", str(PROJECT_ROOT / "diffusion_llms" / "tasks"),
            "--tasks", task,
            "--model", "GSAI-ML/LLaDA-8B-Instruct",  # Our registered model name
            "--device", device,
            "--num_fewshot", str(num_fewshot),
            "--model_args", model_args,
            "--seed", str(seed),
            "--output_path", str(output_dir / "results"),
            "--confirm_run_unsafe_code",
        ]

        print(f"Command: {' '.join(cmd)}")

        env = os.environ.copy()
        env["HF_ALLOW_CODE_EVAL"] = "1"
        env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")

        result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))

        if result.returncode == 0:
            print(f"SUCCESS: {task} q={q}")
            results.append({"task": task, "quantile": q, "status": "success"})
        else:
            print(f"FAILED: {task} q={q}")
            results.append({"task": task, "quantile": q, "status": "failed"})

    return results


def run_all_benchmarks(
    model: str,
    tasks: list[str],
    quantiles: list[float],
    device: str,
    output_dir: Path,
):
    """Run all benchmark evaluations."""

    all_results = []

    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"# TASK: {task}")
        print(f"{'#'*60}")

        results = run_benchmark_evaluation(
            model=model,
            task=task,
            quantiles=quantiles,
            device=device,
            output_dir=output_dir,
        )
        all_results.extend(results)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="SmartCrop Experiment Runner - Replicate all paper experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["benchmarks", "all"],
        default="benchmarks",
        help="Which experiment to run (default: benchmarks)",
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
        help="Output directory (default: results/<timestamp>)",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / timestamp
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)

    print("="*60)
    print("SmartCrop Experiment Runner")
    print("="*60)
    print(f"Experiment: {args.experiment}")
    print(f"Model: {args.model}")
    print(f"Tasks: {args.tasks}")
    print(f"Quantiles: {args.quantiles}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("="*60)

    if args.dry_run:
        print("\n[DRY RUN - No commands will be executed]\n")
        return

    # Run experiments
    if args.experiment in ["benchmarks", "all"]:
        results = run_all_benchmarks(
            model=args.model,
            tasks=args.tasks,
            quantiles=args.quantiles,
            device=args.device,
            output_dir=output_dir,
        )

        # Print summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

        success_count = sum(1 for r in results if r["status"] == "success")
        total_count = len(results)

        print(f"Completed: {success_count}/{total_count}")

        for r in results:
            status_icon = "[OK]" if r["status"] == "success" else "[FAIL]"
            print(f"  {status_icon} {r['task']} q={r['quantile']}")

        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
