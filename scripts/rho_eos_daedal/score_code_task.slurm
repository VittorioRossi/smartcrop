#!/bin/bash
#SBATCH --job-name="score_code_task"
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/score_code_task-%j.out
#SBATCH --error=logs/slurm/score_code_task-%j.err

# ── Overrides (pass as env vars before sbatch) ────────────────────────────────
#   TASK    : mbpp | humaneval                                        (required)
#   SAMPLES : samples_{task}_*.jsonl path to score                    (required)
#   OUT     : output {task_id: bool} JSON path                        (required)
#
# Example:
#   TASK=mbpp \
#   SAMPLES=results/daedal-1024-codeonly-20260718_212507/mbpp_1024/samples_mbpp_2026-07-19T00-18-02.336384.jsonl \
#   OUT=mcnemar_code_scores/daedal_mbpp.json \
#     sbatch scripts/rho_eos_daedal/score_code_task.slurm

set -e
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate rho-EOS
export HF_ALLOW_CODE_EVAL=1

mkdir -p "$(dirname "$OUT")"
python3 scripts/score_code_task.py --task "$TASK" \
  --samples "$SAMPLES" --out "$OUT" --repo-root .
