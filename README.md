# Diffusion Language Models Are Natively Length Aware

Code for SmartCrop: read a prompt-specific length from the `t=1` EOS probabilities of a diffusion language model, crop the canvas to that length, then denoise.

The method is in `diffusion_llms/generators/cdf_remasking_generator.py`. Every paper number comes from the scripts in `scripts/`.

## Setup

Python 3.10. A GPU is needed only for generation.

```bash
git clone --recurse-submodules https://github.com/VittorioRossi/smartcrop
cd smartcrop
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
pytest
```

`requirements.txt` holds the exact versions of the environment behind the paper. `requirements-lock.txt` lists that whole environment (Linux, CUDA 12.8). `vendor/dllm` is [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm) at `b8d76ff`.

Models: `GSAI-ML/LLaDA-8B-Instruct`, `GSAI-ML/LLaDA-1.5`, `Dream-org/Dream-v0-Instruct-7B`.

**Result files are not in this repo.** Everything marked CPU below reads what the GPU runs write under `results/`, except the three figures marked "no input".

## Generation

All runs use seed 42 and batch size 1, on one H100 NVL for the published times. Timings depend on the hardware; accuracy and token counts do not need it.

`N` is 1319 for GSM8K, 500 for MATH500, 164 for HumanEval. MBPP is 500 in Table 1 and 374 in every other table. One token per step means `--steps` equals `--gen-length`.

```bash
# SmartCrop, one token per step
python scripts/run_benchmark.py --task $TASK --n $N --sampler cdf --q 0.90 --gen-length 1024 --steps 1024
# Fixed canvas, one token per step
python scripts/run_benchmark.py --task $TASK --n $N --sampler low_conf --gen-length $M --steps $M
# SmartCrop, confidence-threshold decoding
python scripts/run_benchmark.py --task $TASK --n $N --sampler cdf --threshold 0.9 --gen-length 1024 --steps 1024
# Fixed 1024, confidence-threshold decoding
python scripts/run_benchmark.py --task $TASK --n $N --sampler cdf --no-crop --threshold 0.9 --gen-length 1024 --steps 1024
```

Add `--model` for the other models and `--eos-free` to suppress EOS on the crop. Code tasks execute model-written code: set `HF_ALLOW_CODE_EVAL=1` and use a sandbox.

`scripts/slurm/` wraps these commands. Submit from the repo root after `mkdir -p logs/slurm`, and add your own `--account` and partition. The wrappers set the Hugging Face offline flags, so download models and datasets first.

## Paper items

| Paper item | Label | Command | Needs |
|---|---|---|---|
| Table 1, left block | `tab:main` | SmartCrop and Fixed above with `M` in 64...2048, plus DAEDAL and ρ-EOS below | GPU |
| Table 1, right block | `tab:main` | The two threshold commands above, plus `scripts/rho_eos_daedal/eval_baseline_init64_bs1.sh` | GPU |
| q-sweep | `tab:ablation_q` | SmartCrop with `--q` in 0.50, 0.75, 0.90, 0.95, 0.99 | GPU |
| DREAM-7B results | `tab:dream_main` | One-token block: `sbatch scripts/slurm/run_dream_fixed_all.slurm`, `run_dream_cdf_all.slurm`. Threshold block: the SmartCrop threshold command with `--model Dream-org/Dream-v0-Instruct-7B`, with and without `--eos-free`, MBPP `N` 500 | GPU |
| LLaDA-1.5 results | `tab:llada15_main` | `sbatch scripts/slurm/run_llada_1_5_cdf_h100.slurm`; baselines are cited from Yang et al. | GPU |
| eos_free | `tab:eos_free` | SmartCrop with `--eos-free` (LLaDA-1.5 at `--q 0.99`), then `run_mcnemar.py --pair` | GPU, then CPU |
| Instance vs. task mean | `tab:instance_vs_mean` | Fixed at the task-mean canvas, then `python scripts/run_mcnemar.py --pair $TASK $SMARTCROP.json $FIXED.json` | GPU, then CPU |
| McNemar p-values | `tab:pvals` | `python scripts/run_mcnemar.py --input-dir results/mcnemar_input` | CPU |
| Calibration | `tab:calibration` | `python scripts/calibration.py --results-dir results` | CPU |
| Under-allocation | `tab:retention` | `python scripts/calibration.py --results-dir results --retention` | CPU |
| Sampled-length check | `app:calibration` | `run_e1_pilot.py --task gsm8k --n 200 --k 16`, then `analyze_e1.py` and `analyze_e1_quantile_calibration.py` on its output | GPU, then CPU |
| Probe cost | `tab:probe` | `python scripts/time_probe.py`, then `python scripts/analyze_probe_timing.py $OUT.json` | GPU, then CPU |
| Baselines from 1024 | `tab:start1024` | `eval_daedal_1024.sh`, `eval_rho_eos_asym_1024.sh`, then `compute_daedal_rho_tokens.py $SAMPLES.jsonl` | GPU, then CPU |

The task-mean canvases `M` are 337, 721, 813, 588 for LLaDA-8B and 204, 285, 301, 267 for DREAM-7B (GSM8K, MATH500, HumanEval, MBPP).

`scripts/eval_results.py` rescores result files. `backfill_boxed_correct.py` and `backfill_code_correct.py` add a missing `correct` field.

## Figures

| File in the paper | Command | Needs |
|---|---|---|
| `img/method.png` | `python scripts/plot_method.py --out figures/method.pdf` (also writes the PNG) | CPU, no input |
| `img/pareto_llada_1.png` | `python scripts/plot_pareto.py --out figures/pareto_llada_1.png` | CPU, no input |
| `img/pareto_dream.png` | `python scripts/plot_pareto_dream.py --out figures/pareto_dream.png` | CPU, no input |
| `img/eos_cdf_examples.png` | `python scripts/plot_eos_cdf_examples.py` | context-invariance runs |
| `img/context_invariance.png` | `python scripts/plot_context_invariance.py` | context-invariance runs |
| `img/peos_llada8_all.png` | `python scripts/plot_peos_grid.py --model llada_8b_instruct --out figures/peos_llada8_all.png` | context-invariance runs |
| `img/peos_dream_all.png` | `python scripts/plot_peos_grid.py --model dream_v0_instruct_7b --out figures/peos_dream_all.png` | context-invariance runs |

Context-invariance runs need a GPU, once per model: `python scripts/run_context_invariance.py --model $MODEL --n 200`.

## Baselines

DAEDAL and ρ-EOS run in the authors' harness, [yjyddq/rho-EOS](https://github.com/yjyddq/rho-EOS) at `69992ca`, in its own environment. `scripts/rho_eos_daedal/README.md` has the scripts, the fixes they need, and how to read their accuracy. `score_code_task.py` there recovers the per-problem results that `run_mcnemar.py` needs.

## License

MIT.
