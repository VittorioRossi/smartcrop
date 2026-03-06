# SmartCrop: Zero-Shot Length Prediction for Efficient Diffusion Language Models

This repository contains the evaluation code and scripts to replicate all experiments from the paper "More Efficient Text Diffusion via Length Prediction."

`diffusion_llms/` is the core package containing the evaluation code, generation strategies, and utilities for tracking efficiency metrics. The `scripts/` directory contains the main entry points for running the benchmark evaluations and quality-length trade-off analysis. Pre-computed results and visualizations are available in the `figures/` directory.

## Usage

```bash
# 1. Clone and setup
git clone <repository-url>
cd smartcrop-submission

# 2. Create environment
conda create -n smartcrop python=3.10 -y
conda activate smartcrop
pip install -r requirements.txt

# 3. Run all benchmark evaluations (Table 1)
python scripts/run_experiments.py --experiment benchmarks --device cuda

# 4. Run quality-length sweep (Figure 2)
python scripts/run_quality_length_sweep.py --task ifeval --device cuda
```

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 16GB VRAM (e.g., V100, A100)
- **Recommended**: NVIDIA A100 40GB or higher
- **CPU-only**: Supported but extremely slow (not recommended)

Estimated runtime on A100:
- Full benchmark suite (all tasks, all quantiles): ~24-48 hours
- Single task, single quantile: ~30-60 minutes
- Quality-length sweep (100 samples): ~2-4 hours

## Experiments

### Experiment 1: Benchmark Evaluations (Table 1)

Evaluates the model across multiple benchmarks with different EOS quantile thresholds.

```bash
# Run all benchmarks with all quantiles
python scripts/run_experiments.py --experiment benchmarks --device cuda

# Run specific task
python scripts/run_experiments.py --tasks gsm8k_cot --quantiles 0.99 --device cuda

# Run specific quantiles
python scripts/run_experiments.py --tasks ifeval --quantiles 0.0 0.99 --device cuda
```

**Tasks available:**
- `humaneval_instruct_local` - Code generation (HumanEval)
- `gsm8k_cot` - Math reasoning (GSM8K with chain-of-thought)
- `ifeval` - Instruction following (IFEval)
- `longformqa` - Long-form question answering (LongFormQA)

**Quantiles:**
- `0.0` - Native baseline (no length prediction)
- `0.5, 0.75, 0.9, 0.95, 0.99` - Increasing conservatism in length prediction

### Experiment 2: Quality-Length Trade-off (Figure 2)

Analyzes how performance varies when generating at different lengths relative to the predicted optimal.

```bash
# Run on IFEval (default)
python scripts/run_quality_length_sweep.py --device cuda

# Run on HumanEval
python scripts/run_quality_length_sweep.py --task humaneval_instruct_local --device cuda

# Quick test with dummy model
python scripts/run_quality_length_sweep.py --dummy --num_samples 5
```

This experiment predicts optimal generation length for each sample, generates at various deviations (-50% to +50% from predicted) and compares against shuffled lengths (random baseline) and full-length generation.

## Model

All experiments use **LLaDA-8B-Instruct** ([`GSAI-ML/LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)), a state-of-the-art diffusion language model.

The model is automatically downloaded from Hugging Face Hub on first run.

## Output Format

### Benchmark Results

Results are saved to `results/<timestamp>/`:
- `results/` - lm-eval output files (JSON)
- `logs/` - Per-sample generation logs (JSONL)

Each log entry contains:
```json
{
  "example_id": 0,
  "generation_method": "ZeroShotGenerator",
  "generation_length": 128,
  "total_flops": "1.23e+12",
  "nfe": 256,
  "nfe_per_token": 2.0,
  "messages": {
    "prompt": "...",
    "raw_generated": "...",
    "final_answer": "..."
  }
}
```

### Sweep Results

Sweep results are saved as JSONL with one entry per sample:
```json
{
  "doc_id": 0,
  "prompt": "...",
  "predicted_gen_length": 128,
  "shuffled_gen_length": 95,
  "deviations": {
    "pred_-50%": {"gen_length": 64, "text": "...", "metrics": {...}},
    "pred_+0%": {"gen_length": 128, "text": "...", "metrics": {...}},
    "pred_+50%": {"gen_length": 192, "text": "...", "metrics": {...}},
    "shuffled": {"gen_length": 95, "text": "...", "metrics": {...}},
    "full": {"gen_length": 512, "text": "...", "metrics": {...}}
  }
}
```

## Pre-computed Results

The `figures/` directory contains pre-computed visualizations from our experiments:

| File | Description |
|------|-------------|
| `table_1.pdf` | Main results table comparing quantiles across benchmarks |
| `distribution_q0.99.png` | Distribution of predicted vs actual lengths at q=0.99 |
| `quality_length_sweep.png` | Quality-length trade-off on HumanEval |
| `quality_length_sweep_ifeval.png` | Quality-length trade-off on IFEval |
| `*_len_perf_corr.png` | Length-performance correlation plots |

## Troubleshooting

### Out of Memory

Reduce batch size or use a smaller quantile subset:
```bash
python scripts/run_experiments.py --tasks gsm8k_cot --quantiles 0.99 --device cuda
```

### CUDA Not Available

Check your CUDA installation:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

### Task Loading Errors

Ensure the tasks directory is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Citation

If you use this code, please cite:

```bibtex
@article{smartcrop2026,
  title={Diffusion Language Models Are Natively Length-Aware},
  author={Rossi, Vittorio and Cir\`{o}, Giacomo and Beltrame, Davide and Gandolfi, Luca and R{\"o}ttger, Paul and Hovy, Dirk},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This code is released under the MIT License. See LICENSE for details.

## Acknowledgments

This work builds upon:
- [LLaDA](https://github.com/GSAI-ML/LLaDA) - Large Language Diffusion with mAsking
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Evaluation framework
- [DLLM](https://github.com/ML-GSAI/dllm) - Diffusion LLM utilities
