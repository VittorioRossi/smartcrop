#!/bin/bash
#SBATCH --job-name="daedal_full"
#SBATCH --partition=gpunew
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/daedal_full-%j.out
#SBATCH --error=logs/slurm/daedal_full-%j.err

set -e
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate rho-EOS

bash scripts/eval_daedal_1024.sh
