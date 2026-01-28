#!/usr/bin/env python3
"""
Quality-Length Sweep Experiment for SmartCrop

This script analyzes the quality-length trade-off by testing how model performance
varies when generating at different lengths relative to the predicted optimal length.

For each sample, we:
1. Predict the optimal generation length using zero-shot EOS distribution
2. Generate outputs at various deviations from this prediction (-50% to +50%)
3. Compare against shuffled lengths (random baseline) and full-length generation

This experiment demonstrates that our length prediction method identifies a
near-optimal generation length - deviating from it degrades performance.

Usage:
    # Run on IFEval (default)
    python scripts/run_quality_length_sweep.py --device cuda

    # Run on HumanEval
    python scripts/run_quality_length_sweep.py --task humaneval --device cuda

    # Quick test with dummy model
    python scripts/run_quality_length_sweep.py --dummy --num_samples 5

Requirements:
    - NVIDIA GPU with CUDA support (recommended: A100 40GB)
    - ~16GB VRAM for LLaDA-8B-Instruct
"""

import os
import sys
import json
import torch
import argparse
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from lm_eval.tasks import TaskManager
from diffusion_llms.utils import get_model, get_tokenizer
from diffusion_llms.generators import BaseLengthGenerator, BaseLengthGeneratorConfig
from diffusion_llms.utils.generation_utils import compute_zero_shot_length_prediction


class DummyModel(torch.nn.Module):
    """Mock model for testing the pipeline without loading a real model."""

    def __init__(self, device, vocab_size=128256):
        super().__init__()
        self.device = device
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': 4096,
            'num_hidden_layers': 2,
            'num_attention_heads': 32,
            'intermediate_size': 11008,
        })()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).to(self.device)
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.config.vocab_size, device=self.device)
        return type('Output', (), {'logits': logits})()

    def generate(self, input_ids, **kwargs):
        max_new_tokens = kwargs.get('max_new_tokens', 100)
        if hasattr(input_ids, "shape"):
            batch_size = input_ids.shape[0]
            device = input_ids.device
        else:
            batch_size = len(input_ids)
            device = self.device
            input_ids = torch.tensor(input_ids, device=device)
        new_tokens = torch.randint(0, self.config.vocab_size, (batch_size, max_new_tokens), device=device)
        return torch.cat([input_ids, new_tokens], dim=1)


def load_task_dataset(task_name: str, include_path: str):
    """Load dataset for a given task using lm-eval TaskManager."""
    tm = TaskManager(include_path=include_path)
    task_dict = tm.load_task_or_group([task_name])
    task = task_dict[task_name]

    dataset = None
    if hasattr(task, "dataset"):
        if "test" in task.dataset:
            dataset = task.dataset["test"]
        elif "validation" in task.dataset:
            dataset = task.dataset["validation"]

    if dataset is None:
        if hasattr(task, "test_docs"):
            test_docs = task.test_docs()
            if test_docs is not None:
                dataset = list(test_docs)
        elif hasattr(task, "validation_docs"):
            val_docs = task.validation_docs()
            if val_docs is not None:
                dataset = list(val_docs)

    return task, list(dataset) if dataset else None


def run_sweep(
    model,
    tokenizer,
    task,
    samples: list,
    output_path: Path,
    device: str,
    is_dummy: bool = False,
):
    """Run the quality-length sweep experiment."""

    generator = BaseLengthGenerator(model=model, tokenizer=tokenizer)
    mask_id = tokenizer.mask_token_id or 126336  # LLaDA fallback

    # Phase 1: Predict lengths for all samples
    print("\nPhase 1: Predicting optimal lengths...")
    precomputed_data = []
    predicted_lengths = []

    for i, doc in enumerate(tqdm(samples, desc="Predicting lengths")):
        prompt_text = task.doc_to_text(doc)
        messages = [{"role": "user", "content": prompt_text}]
        templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tensor = tokenizer(templated, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        predicted_length = compute_zero_shot_length_prediction(
            model=model,
            prompt=prompt_tensor[0],
            mask_id=mask_id,
            eos_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            eos_quantile=0.99,
            safe_margin=0,
            device=device,
        )

        if predicted_length is None:
            predicted_length = prompt_tensor.shape[1] + (128 if is_dummy else 256)

        precomputed_data.append({
            "doc": doc,
            "prompt_text": templated,
            "prompt_tensor": prompt_tensor,
            "predicted_total_length": predicted_length,
        })
        predicted_lengths.append(predicted_length)

    # Create shuffled baseline
    shuffled_lengths = predicted_lengths.copy()
    random.shuffle(shuffled_lengths)

    # Phase 2: Run sweep
    print("\nPhase 2: Running generation sweep...")

    # Clear output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("")

    for i, data in enumerate(tqdm(precomputed_data, desc="Sweep")):
        doc = data["doc"]
        prompt_tensor = data["prompt_tensor"]
        predicted_total_len = data["predicted_total_length"]
        shuffled_total_len = shuffled_lengths[i]

        prompt_len = prompt_tensor.shape[1]
        gen_len_pred = max(10, predicted_total_len - prompt_len)
        gen_len_shuffled = max(10, shuffled_total_len - prompt_len)

        # Define deviations to test
        deviations = {}
        for percent in range(-50, 51, 10):
            multiplier = 1.0 + (percent / 100.0)
            length = max(10, int(gen_len_pred * multiplier))
            deviations[f"pred_{percent:+d}%"] = length

        deviations["shuffled"] = gen_len_shuffled
        deviations["full"] = 512  # Fixed baseline

        doc_results = {
            "doc_id": i,
            "prompt": data["prompt_text"],
            "predicted_gen_length": gen_len_pred,
            "shuffled_gen_length": gen_len_shuffled,
            "deviations": {},
        }

        for name, gen_len in deviations.items():
            target_total_len = prompt_len + gen_len
            steps = 2 if is_dummy else 512

            config = BaseLengthGeneratorConfig(
                max_new_tokens=gen_len,
                max_length=target_total_len,
                length_prediction=target_total_len,
                steps=steps,
                block_length=target_total_len,
                temperature=0.0,
                remasking="low_confidence",
            )
            config.measure_flops = False

            try:
                out = generator.generate(prompts=[prompt_tensor[0]], config=config)
                generated_seq = out.sequences[0]
                generated_text = tokenizer.decode(generated_seq[prompt_len:], skip_special_tokens=True)

                metrics = task.process_results(doc, [generated_text])

                doc_results["deviations"][name] = {
                    "gen_length": gen_len,
                    "text": generated_text[:500],  # Truncate for storage
                    "metrics": metrics,
                }

            except Exception as e:
                print(f"Error generating for {name}: {e}")
                doc_results["deviations"][name] = {"error": str(e)}

        # Append result
        with open(output_path, "a") as f:
            f.write(json.dumps(doc_results) + "\n")

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quality-Length Sweep Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Model to evaluate",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="ifeval",
        choices=["ifeval", "humaneval_instruct_local", "gsm8k_cot", "longformqa"],
        help="Task to run sweep on (default: ifeval)",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output file path (default: results/quality_length_sweep_<task>.jsonl)",
    )

    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy model for testing pipeline",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup output path
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / "results" / f"quality_length_sweep_{args.task}_{timestamp}.jsonl"
    else:
        output_path = Path(args.output_path)

    print("="*60)
    print("Quality-Length Sweep Experiment")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print(f"Output: {output_path}")
    print(f"Dummy mode: {args.dummy}")
    print("="*60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)

    # Load model
    print("Loading model...")
    if args.dummy:
        model = DummyModel(args.device, vocab_size=len(tokenizer))
        model = model.to(args.device)
    else:
        device_map = "auto" if args.device != "cpu" else None
        model = get_model(args.model, device_map=device_map, trust_remote_code=True, dtype=torch.bfloat16)

    model.eval()

    # Load task and dataset
    print(f"Loading {args.task} dataset...")
    include_path = str(PROJECT_ROOT / "diffusion_llms" / "tasks")
    task, dataset = load_task_dataset(args.task, include_path)

    if dataset is None:
        print(f"ERROR: Could not load dataset for {args.task}")
        sys.exit(1)

    samples = dataset[:args.num_samples]
    print(f"Loaded {len(samples)} samples")

    # Run experiment
    run_sweep(
        model=model,
        tokenizer=tokenizer,
        task=task,
        samples=samples,
        output_path=output_path,
        device=args.device,
        is_dummy=args.dummy,
    )


if __name__ == "__main__":
    main()
