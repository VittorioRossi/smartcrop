#!/usr/bin/env python3
"""
SmartCrop Rebuttal Benchmark Runner (SC-0.9 @ 2048 tokens)

Runs SmartCrop with eos_quantile=0.9 and max_new_tokens=2048 on all five
rebuttal tasks: GSM8K, HumanEval, IfEval, BBH, MATH-500.

This mirrors run_daedal_benchmarks.py so results are directly comparable.

Usage:
    python scripts/run_smartcrop_rebuttal.py
    python scripts/run_smartcrop_rebuttal.py --tasks gsm8k_cot_llama math500
    sbatch scripts/slurm/run_rebuttal_smartcrop.slurm
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
SC_QUANTILE   = 0.9   # stored as directory sentinel (mirrors DAEDAL_QUANTILE_SENTINEL)
MAX_NEW_TOKENS = 2048

# All five rebuttal tasks.
# num_fewshot: None = use task default (BBH: 3-shot); 0 = override to zero-shot.
TASK_CONFIGS = {
    "humaneval_instruct_local":  {"num_fewshot": 0},
    "gsm8k_cot_llama":           {"num_fewshot": 0},
    "ifeval":                    {"num_fewshot": 0},
    "bbh_cot_fewshot": {"num_fewshot": 3},
    "math500":                   {"num_fewshot": 0},
}

# SmartCrop generation hyperparameters (temperature=0.0 for fair DAEDAL comparison)
SC_HPARAMS = {
    "generation_mode":  "zero_shot",
    "eos_quantile":     SC_QUANTILE,
    "max_new_tokens":   MAX_NEW_TOKENS,
    "steps":            MAX_NEW_TOKENS,
    "block_length":     MAX_NEW_TOKENS,
    "temperature":      0.0,
    "cfg":              0.0,
    "logits_eos_inf":   True,
    "measure_flops":    True,
}


def is_task_completed(exp_dirs: ExperimentDirectoryManager, task: str) -> bool:
    task_dir = exp_dirs.get_task_quantile_dir(task, SC_QUANTILE)
    if not task_dir.exists():
        return False
    if list(task_dir.glob("**/results_*.json")):
        print(f"  [RESUME] Already completed: {task}")
        return True
    return False


def run_single_evaluation(
    model: str,
    task: str,
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
            "task_dir": str(exp_dirs.setup_task_quantile(task, SC_QUANTILE)),
        }

    task_cfg     = TASK_CONFIGS.get(task, {"num_fewshot": 0})
    num_fewshot  = task_cfg["num_fewshot"]

    task_dir     = exp_dirs.setup_task_quantile(task, SC_QUANTILE)
    rank_log_dir = exp_dirs.get_rank_log_dir(task, SC_QUANTILE)
    samples_path = exp_dirs.get_samples_path(task, SC_QUANTILE)
    metrics_path = exp_dirs.get_metrics_path(task, SC_QUANTILE)

    model_args_parts = [
        f"pretrained={model}",
        f"generation_mode={SC_HPARAMS['generation_mode']}",
        f"eos_quantile={SC_HPARAMS['eos_quantile']}",
        f"max_new_tokens={SC_HPARAMS['max_new_tokens']}",
        f"steps={SC_HPARAMS['steps']}",
        f"block_length={SC_HPARAMS['block_length']}",
        f"temperature={SC_HPARAMS['temperature']}",
        f"cfg={SC_HPARAMS['cfg']}",
        f"logits_eos_inf={SC_HPARAMS['logits_eos_inf']}",
        f"measure_flops={SC_HPARAMS['measure_flops']}",
        "trust_remote_code=True",
        f"task_name={task}",
        f"debug_logpath={rank_log_dir}/generation.jsonl",
    ]
    model_args = ",".join(str(p) for p in model_args_parts)

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
        "--num_fewshot", str(num_fewshot),
        "--model_args", model_args,
        "--seed", str(seed),
        "--output_path", str(task_dir),
        "--confirm_run_unsafe_code",
    ])

    print(f"\n{'='*60}")
    print(f"SmartCrop (q={SC_QUANTILE}, max={MAX_NEW_TOKENS})  |  Task: {task}")
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
        num_merged   = merge_rank_logs(rank_log_dir, samples_path)
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
    parser = argparse.ArgumentParser(description="SmartCrop rebuttal benchmark (SC-0.9 @ 2048)")
    parser.add_argument("--model",         default=DEFAULT_MODEL)
    parser.add_argument("--tasks",         nargs="+", default=list(TASK_CONFIGS.keys()))
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--output_dir",    default=None)
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--master_port",   type=int, default=29500)
    parser.add_argument("--no_resume",     action="store_true")
    args = parser.parse_args()

    run_id     = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = args.output_dir or str(
        PROJECT_ROOT / "results" / "rebuttal" / f"smartcrop_rebuttal_{run_id}"
    )
    exp_dirs = ExperimentDirectoryManager(output_dir)

    print("=" * 60)
    print(f"SmartCrop Rebuttal Benchmarks  (q={SC_QUANTILE}, max={MAX_NEW_TOKENS})")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Tasks:    {args.tasks}")
    print(f"Output:   {output_dir}")
    print(f"SC hparams: {SC_HPARAMS}")
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

    summary = {
        "run_id":       run_id,
        "model":        args.model,
        "method":       "smartcrop",
        "sc_hparams":   SC_HPARAMS,
        "tasks":        args.tasks,
        "results":      all_results,
        "completed_at": datetime.now().isoformat(),
    }
    summary_path = Path(output_dir) / "smartcrop_summary.json"
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
