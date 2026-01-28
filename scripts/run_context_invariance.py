#!/usr/bin/env python3
"""
Context Invariance Test for Zero-Shot Length Prediction

Tests whether the predicted generation length is invariant to the context size
used during the forward pass. This validates that the length predictor gives
consistent estimates regardless of how much padding is allocated.

Usage:
    # Single task
    python scripts/run_context_invariance.py --tasks gsm8k --num_samples 100

    # All tasks
    python scripts/run_context_invariance.py --tasks all --num_samples 500

    # Full run via SLURM (see run_context_invariance.slurm)
    python scripts/run_context_invariance.py --output_dir results/context_invariance/run_001 --tasks all

Features:
    - Tests context sizes: 256, 512, 1024, 2048, 4096
    - Supports tasks: gsm8k, humaneval, ifeval, longformqa
    - Resume capability: skips already processed doc_ids
    - Incremental saving: results saved per sample
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from lm_eval.tasks import TaskManager
from diffusion_llms.utils import get_model, get_tokenizer
from diffusion_llms.utils.generation_utils import compute_zero_shot_length_prediction

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# Context sizes to test
CONTEXT_SIZES = [256, 512, 1024, 2048, 4096]

# Task configurations matching run_slurm_benchmarks.py
# task_name -> lm_eval task name for loading
TASK_CONFIGS = {
    "gsm8k": {
        "lm_eval_task": "gsm8k_cot_llama",
        "max_context": 4096,  # Maximum context size to test
    },
    "humaneval": {
        "lm_eval_task": "humaneval_instruct_local",
        "max_context": 4096,
    },
    "ifeval": {
        "lm_eval_task": "ifeval",
        "max_context": 4096,
    },
    "longformqa": {
        "lm_eval_task": "longformqa",
        "max_context": 4096,
    },
}

ALL_TASKS = list(TASK_CONFIGS.keys())


def load_existing_results(output_path: Path) -> set[int]:
    """Load already processed doc_ids for resume capability."""
    processed = set()
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed.add(data['doc_id'])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return processed


def get_task_and_dataset(task_name: str):
    """Load task and dataset for a given task name."""
    config = TASK_CONFIGS.get(task_name)
    if not config:
        raise ValueError(f"Unknown task: {task_name}. Available: {ALL_TASKS}")
    
    lm_eval_task = config["lm_eval_task"]
    
    # Try loading with custom tasks first
    tm = TaskManager(include_path="diffusion_llms/tasks")
    try:
        task_dict = tm.load_task_or_group([lm_eval_task])
        task = task_dict[lm_eval_task]
    except Exception as e:
        print(f"Failed to load {lm_eval_task} with custom tasks: {e}")
        # Fallback to standard tasks
        tm = TaskManager()
        try:
            task_dict = tm.load_task_or_group([lm_eval_task])
            task = task_dict[lm_eval_task]
        except Exception as e2:
            raise RuntimeError(f"Failed to load {lm_eval_task}: {e2}")
    
    # Get dataset
    dataset = None
    if hasattr(task, "dataset"):
        if "test" in task.dataset:
            dataset = task.dataset["test"]
        elif "validation" in task.dataset:
            dataset = task.dataset["validation"]
    
    if dataset is None and hasattr(task, "test_docs"):
        test_docs = task.test_docs()
        if test_docs:
            dataset = list(test_docs)
    
    if not dataset:
        raise RuntimeError(f"Could not find dataset docs for {task_name}")
    
    return task, list(dataset)


def run_invariance_test(
    model,
    tokenizer,
    task,
    samples,
    task_name: str,
    output_path: Path,
    eos_quantile: float,
    device: str,
    processed_ids: set,
):
    """Run invariance test for a single task."""
    
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = 126336  # <|mdm_mask|> fallback for LLaDA
    eos_id = tokenizer.eos_token_id
    
    total_processed = len(processed_ids)
    
    for i, doc in enumerate(tqdm(samples, desc=f"Processing {task_name}")):
        # Skip if already processed
        if i in processed_ids:
            continue
        
        # Format prompt using task's doc_to_text
        prompt_text = task.doc_to_text(doc)
        
        # Apply chat template
        messages = [{"role": "user", "content": prompt_text}]
        templated_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompt_tensor = tokenizer(
            templated_prompt, 
            return_tensors="pt", 
            add_special_tokens=False
        ).input_ids.to(device)
        
        prompt_len = prompt_tensor.shape[1]
        
        doc_result = {
            "doc_id": i,
            "prompt_len": prompt_len,
            "task": task_name,
            "eos_quantile": eos_quantile,
            "predictions": {}
        }
        
        for C in CONTEXT_SIZES:
            # Check if prompt fits in context
            if prompt_len >= C:
                doc_result["predictions"][str(C)] = {
                    "error": "prompt_too_long", 
                    "predicted_length": None
                }
                continue
            
            max_new_tokens = C - prompt_len
            
            try:
                predicted_length = compute_zero_shot_length_prediction(
                    model=model,
                    prompt=prompt_tensor[0],
                    mask_id=mask_id,
                    eos_id=eos_id,
                    max_new_tokens=max_new_tokens,
                    eos_quantile=eos_quantile,
                    safe_margin=0,
                    device=device
                )
                
                # Compute the predicted NEW tokens (subtract prompt)
                predicted_new_tokens = predicted_length - prompt_len if predicted_length else None
                
                doc_result["predictions"][str(C)] = {
                    "predicted_length": predicted_length,
                    "predicted_new_tokens": predicted_new_tokens,
                    "max_new_tokens": max_new_tokens,
                    "context_size": C
                }
            except Exception as e:
                doc_result["predictions"][str(C)] = {
                    "error": str(e), 
                    "predicted_length": None
                }
        
        # Incremental save
        with open(output_path, "a") as f:
            f.write(json.dumps(doc_result) + "\n")
        
        total_processed += 1
    
    return total_processed


def main():
    parser = argparse.ArgumentParser(
        description="Context Invariance Test for Zero-Shot Length Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="GSAI-ML/LLaDA-8B-Instruct", 
        help="Model name or path"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory. Results saved to {output_dir}/{task}/results.jsonl"
    )
    parser.add_argument(
        "--tasks", 
        type=str, 
        nargs="+",
        default=["gsm8k"],
        help=f"Tasks to run. Options: {ALL_TASKS} or 'all'"
    )
    parser.add_argument(
        "--eos_quantile",
        type=float,
        default=0.9,
        help="EOS quantile for length prediction"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Device"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (re-run all samples)"
    )
    args = parser.parse_args()

    # Determine tasks to run
    if args.tasks == ["all"] or "all" in args.tasks:
        tasks_to_run = ALL_TASKS
    else:
        tasks_to_run = args.tasks
    
    # Validate tasks
    for t in tasks_to_run:
        if t not in TASK_CONFIGS:
            print(f"ERROR: Unknown task '{t}'. Available: {ALL_TASKS}")
            return
    
    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/context_invariance")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model and Tokenizer (once for all tasks)
    print(f"Loading model: {args.model}...")
    device_map = "auto" if args.device != "cpu" else None
    
    model = get_model(
        args.model, 
        device_map=device_map, 
        trust_remote_code=True, 
        dtype=torch.bfloat16
    )
    if args.device == "cpu":
        model = model.to("cpu")
        
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    model.eval()

    # 2. Run each task
    all_results = {}
    
    for task_name in tasks_to_run:
        print(f"\n{'='*60}")
        print(f"TASK: {task_name}")
        print(f"{'='*60}")
        
        # Create task-specific output directory
        task_output_dir = output_dir / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_output_dir / "results.jsonl"
        
        # Load existing results for resume
        processed_ids = set()
        if not args.no_resume:
            processed_ids = load_existing_results(output_path)
            if processed_ids:
                print(f"Resuming: {len(processed_ids)} samples already processed")
        
        # Load task and dataset
        try:
            task, dataset = get_task_and_dataset(task_name)
        except Exception as e:
            print(f"ERROR loading task {task_name}: {e}")
            continue
        
        # Select samples
        samples = dataset.copy()
        print(f"Running on {len(samples)} samples (contexts: {CONTEXT_SIZES})")
        print(f"Output: {output_path}")
        
        # Run invariance test
        total_processed = run_invariance_test(
            model=model,
            tokenizer=tokenizer,
            task=task,
            samples=samples,
            task_name=task_name,
            output_path=output_path,
            eos_quantile=args.eos_quantile,
            device=args.device,
            processed_ids=processed_ids,
        )
        
        all_results[task_name] = {
            "total_processed": total_processed,
            "output_path": str(output_path)
        }
        
        # Save task summary
        summary_path = task_output_dir / "summary.json"
        summary = {
            "model": args.model,
            "task": task_name,
            "lm_eval_task": TASK_CONFIGS[task_name]["lm_eval_task"],
            "num_samples": args.num_samples,
            "total_processed": total_processed,
            "context_sizes": CONTEXT_SIZES,
            "eos_quantile": args.eos_quantile,
            "output_path": str(output_path)
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Completed: {total_processed} samples")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for task_name, result in all_results.items():
        print(f"  {task_name}: {result['total_processed']} samples -> {result['output_path']}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()