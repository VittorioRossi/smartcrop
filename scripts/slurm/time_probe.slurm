#!/bin/bash
#SBATCH --job-name="time_probe"
#SBATCH --partition=medium_gpunew
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/slurm/time_probe-%j.out
#SBATCH --error=logs/slurm/time_probe-%j.err

# Probe vs denoise wall-clock timing for SmartCrop (LLaDA-8B-Instruct, M=1024, q=0.90, bs=1).
# Uses the default ~/.cache/huggingface cache (no HF_HOME override).
set -e
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate diffusion_llms
export PYTHONPATH="${PYTHONPATH}:$PWD"
export GIT_PYTHON_REFRESH=quiet OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
nvidia-smi

python scripts/time_probe.py \
    --tasks gsm8k math500 mbpp humaneval \
    --n 50 50 50 30 \
    --gen-length 1024 --block-size 32 --q 0.90 --warmup 3 \
    --out "results/probe_timing/probe_timing_${SLURM_JOB_ID}.json"

echo "Finished: $(date)"
