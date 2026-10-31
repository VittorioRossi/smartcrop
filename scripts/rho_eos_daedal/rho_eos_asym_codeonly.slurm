#!/bin/bash
#SBATCH --job-name="rho_eos_asym_codeonly"
#SBATCH --partition=long_gpunew
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/rho_eos_asym_codeonly-%j.out
#SBATCH --error=logs/slurm/rho_eos_asym_codeonly-%j.err

set -e
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate rho-EOS

bash scripts/eval_rho_eos_asym_1024_codeonly.sh
