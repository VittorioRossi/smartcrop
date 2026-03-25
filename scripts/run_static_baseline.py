#!/usr/bin/env python3
"""
Static-Mean Baseline Evaluation (Rebuttal Baseline)

Evaluates a baseline where every sample in a task is generated with a FIXED
canvas equal to the per-task mean predicted length from SmartCrop (τ=0.9).

This directly answers the reviewer question:
  "Does SmartCrop beat simply knowing the task-level average length?"
  (distinct from the cross-task shuffled control in Figure 2)

The baseline uses generation_mode=native (standard fixed-canvas diffusion)
with max_new_tokens = mean(L̂_task), so T = max_new_tokens (1 token/step).

Mean generation lengths are derived from the q0.90 SmartCrop run logs and
can be recomputed from any samples.jsonl via --compute_means_from <results_dir>.

Usage:
    python scripts/run_static_baseline.py --device cuda
    sbatch scripts/slurm/run_static_baseline.slurm
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

# Per-task mean generation lengths from SmartCrop τ=0.9 (run 453436).
# These are mean(generation_length) across all samples in q0.90/samples.jsonl.
STATIC_MEAN_LENGTHS = {
    "gsm8k_cot_llama":          162,
    "humaneval_instruct_local": 377,
    "ifeval":                   145,
    "longformqa":               97,
}

# Mirror SmartCrop max_tokens caps so the static length never exceeds the task budget.
TASK_MAX_TOKENS = {
    "gsm8k_cot_llama":          256,
    "humaneval_instruct_local": 512,
    "ifeval":                   1280,
    "longformqa":               512,
}

# Sentinel so rows appear alongside SmartCrop/DAEDAL in analysis scripts.
# We use -2.0 to distinguish from DAEDAL's -1.0.
STATIC_QUANTILE_SENTINEL = -2.0


def compute_means_from_logs(results_dir: Path, quantile: float = 0.90) -> dict:
    """Compute per-task mean generation_length from existing samples.jsonl files."""
    q_str = f"q{quantile:.2f}"
    means = {}
    for task in TASK_MAX_TOKENS:
        samples_path = results_dir / "by_task" / task / q_str / "samples.jsonl"
        if not samples_path.exists():
            print(f"  [WARN] Not found: {samples_path}")
            continue
        lengths = [json.loads(l)["generation_length"] for l in open(samples_path)]
        means[task] = round(sum(lengths) / len(lengths))
        print(f"  {task}: mean={means[task]} (n={len(lengths)})")
    return means


def is_task_completed(exp_dirs: ExperimentDirectoryManager, task: str) -> bool:
    task_dir = exp_dirs.get_task_quantile_dir(task, STATIC_QUANTILE_SENTINEL)
    if not task_dir.exists():
        return False
    if list(task_dir.glob("**/results_*.json")):
        print(f"  [RESUME] Already completed: {task}")
        return True
    return False


def run_single_evaluation(
    model: str,
    task: str,
    static_length: int,
    exp_dirs: ExperimentDirectoryManager,
    device: str = "cuda",
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
            "static_length": static_length,
            "task_dir": str(exp_dirs.setup_task_quantile(task, STATIC_QUANTILE_SENTINEL)),
        }

    max_tokens = TASK_MAX_TOKENS.get(task, 512)
    # Clamp to task budget
    gen_length = min(static_length, max_tokens)

    task_dir = exp_dirs.setup_task_quantile(task, STATIC_QUANTILE_SENTINEL)
    rank_log_dir = exp_dirs.get_rank_log_dir(task, STATIC_QUANTILE_SENTINEL)
    samples_path = exp_dirs.get_samples_path(task, STATIC_QUANTILE_SENTINEL)
    metrics_path = exp_dirs.get_metrics_path(task, STATIC_QUANTILE_SENTINEL)

    model_args = ",".join([
        f"pretrained={model}",
        "generation_mode=native",
        f"max_new_tokens={gen_length}",
        f"steps={gen_length}",
        f"block_length={gen_length}",
        "temperature=0.1",
        "cfg=0.0",
        "trust_remote_code=True",
        f"task_name={task}",
        f"eos_quantile={STATIC_QUANTILE_SENTINEL}",
        f"debug_logpath={rank_log_dir}/generation.jsonl",
        "measure_flops=True",
    ])

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
        "--num_fewshot", "0",
        "--model_args", model_args,
        "--seed", str(seed),
        "--output_path", str(task_dir),
        "--confirm_run_unsafe_code",
    ])

    print(f"\n{'='*60}")
    print(f"Static-Mean  |  Task: {task}  |  L={gen_length}")
    print(f"Output: {task_dir}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))

    run_result = {
        "task": task,
        "status": "success" if result.returncode == 0 else "failed",
        "return_code": result.returncode,
        "static_length": gen_length,
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

        print(f"SUCCESS: {task} L={gen_length} ({num_merged} samples)")
    else:
        print(f"FAILED: {task}")

    return run_result


def main():
    parser = argparse.ArgumentParser(description="Static-mean baseline (rebuttal)")
    parser.add_argument("--model",           default=DEFAULT_MODEL)
    parser.add_argument("--tasks",           nargs="+", default=list(STATIC_MEAN_LENGTHS.keys()))
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--output_dir",      default=None)
    parser.add_argument("--seed",            type=int, default=0)
    parser.add_argument("--num_processes",   type=int, default=1)
    parser.add_argument("--master_port",     type=int, default=29500)
    parser.add_argument("--no_resume",       action="store_true")
    parser.add_argument(
        "--compute_means_from",
        default=None,
        help="Path to SmartCrop results dir to recompute means (e.g. results/icml_2025/453436_benchmark). "
             "If omitted, uses hardcoded values from the q0.90 run.",
    )
    parser.add_argument(
        "--source_quantile",
        type=float, default=0.90,
        help="Which quantile run to derive means from (default: 0.90)",
    )
    args = parser.parse_args()

    # Resolve mean lengths
    mean_lengths = dict(STATIC_MEAN_LENGTHS)
    if args.compute_means_from:
        print(f"Computing means from: {args.compute_means_from}")
        computed = compute_means_from_logs(
            Path(args.compute_means_from), quantile=args.source_quantile
        )
        mean_lengths.update(computed)

    run_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = args.output_dir or str(PROJECT_ROOT / "results" / "rebuttal" / f"static_{run_id}")
    exp_dirs = ExperimentDirectoryManager(output_dir)

    print("=" * 60)
    print("Static-Mean Baseline Evaluation (Rebuttal)")
    print("=" * 60)
    print(f"Model:  {args.model}")
    print(f"Tasks:  {args.tasks}")
    print(f"Output: {output_dir}")
    print("Mean lengths (from SmartCrop τ=0.9):")
    for t in args.tasks:
        print(f"  {t}: {mean_lengths.get(t, '?')}")
    print("=" * 60)

    all_results = []
    for task in args.tasks:
        if task not in mean_lengths:
            print(f"WARNING: No mean length for {task}, skipping.")
            continue
        result = run_single_evaluation(
            model=args.model,
            task=task,
            static_length=mean_lengths[task],
            exp_dirs=exp_dirs,
            device=args.device,
            seed=args.seed,
            skip_completed=not args.no_resume,
            num_processes=args.num_processes,
            master_port=args.master_port,
        )
        all_results.append(result)

    summary = {
        "run_id": run_id,
        "model": args.model,
        "method": "static_mean",
        "mean_lengths": mean_lengths,
        "source_quantile": args.source_quantile,
        "tasks": args.tasks,
        "results": all_results,
        "completed_at": datetime.now().isoformat(),
    }
    summary_path = Path(output_dir) / "static_summary.json"
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
