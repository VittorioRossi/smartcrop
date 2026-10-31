#!/bin/bash
#SBATCH --job-name="backfill_boxed"
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/backfill_boxed-%j.out
#SBATCH --error=logs/slurm/backfill_boxed-%j.err

# ── Overrides (pass as env vars before sbatch) ────────────────────────────────
#   SC_PATH : results JSON to backfill in place (required)
#
# Example:
#   SC_PATH=results/gsm8k/llada_8b_instruct_cdf_q99_n1319_gl1024_s1024.json \
#     sbatch scripts/slurm/backfill_boxed_correct.slurm

set -e
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate diffusion_llms

python3 scripts/backfill_boxed_correct.py --path "$SC_PATH"
