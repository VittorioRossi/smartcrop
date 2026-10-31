#!/bin/bash
#SBATCH --job-name=ctx-inv
#SBATCH --partition=gpuh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/ctx-inv-%j.out
#SBATCH --error=logs/slurm/ctx-inv-%j.err

# Usage:
#   MODEL=GSAI-ML/LLaDA-8B-Instruct sbatch scripts/slurm/run_context_invariance.slurm
#   MODEL=GSAI-ML/LLaDA-1.5         sbatch scripts/slurm/run_context_invariance.slurm
#   MODEL=Dream-org/Dream-v0-Instruct-7B sbatch scripts/slurm/run_context_invariance.slurm

set -e

module load miniconda3
eval "$(conda shell.bash hook)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cd "$PROJECT_DIR" || exit 1
conda activate diffusion_llms
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

MODEL="${MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
N="${N:-200}"
Q="${Q:-0.90}"

echo "Model:  $MODEL"
echo "N:      $N"
echo "Q:      $Q"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"

python scripts/run_context_invariance.py \
    --model   "$MODEL" \
    --tasks   gsm8k math500 humaneval mbpp \
    --n       "$N" \
    --q       "$Q"

echo "Done: $(date)"
