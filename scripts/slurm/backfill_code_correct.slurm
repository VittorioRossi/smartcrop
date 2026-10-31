#!/bin/bash
#SBATCH --job-name="backfill_code"
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/backfill_code-%j.out
#SBATCH --error=logs/slurm/backfill_code-%j.err

# ── Overrides (pass as env vars before sbatch) ────────────────────────────────
#   TASK    : mbpp | humaneval                              (required)
#   SC_PATH : results JSON to backfill in place              (required)
#
# Example:
#   TASK=humaneval SC_PATH=results/humaneval/llada_8b_instruct_cdf_q99_n164_gl1024_s1024.json \
#     sbatch scripts/slurm/backfill_code_correct.slurm

set -e
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate diffusion_llms
export HF_ALLOW_CODE_EVAL=1

python3 scripts/backfill_code_correct.py --task "$TASK" --path "$SC_PATH" \
  --repo-root .
