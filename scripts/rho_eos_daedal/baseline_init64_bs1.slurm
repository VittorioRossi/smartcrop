#!/bin/bash
#SBATCH --job-name="baseline_init64_bs1"
#SBATCH --partition=gpunew
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/baseline_init64_bs1-%j.out
#SBATCH --error=logs/slurm/baseline_init64_bs1-%j.err

# Usage: METHOD=daedal TASK=gsm8k [LIMIT=4] sbatch --export=ALL baseline_init64_bs1.slurm
set -e
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate rho-EOS

bash scripts/eval_baseline_init64_bs1.sh
