#!/bin/bash
#SBATCH --job-name="e1_pilot"
#SBATCH --partition=short_gpunew
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/slurm/e1_pilot-%j.out
#SBATCH --error=logs/slurm/e1_pilot-%j.err

# ── Overrides (pass as env vars before sbatch) ────────────────────────────────
#   TASK        : gsm8k | math500 | humaneval | mbpp   (default: gsm8k)
#   N           : number of prompts                    (default: 20)
#   K           : completions per prompt                (default: 10)
#   TEMPERATURE : sampling temperature                  (default: 1.0)
#   GEN_LENGTH  : canvas size                           (default: 1024)
#
# Example:
#   TASK=gsm8k N=20 K=10 sbatch scripts/slurm/run_e1_pilot.slurm

set -e

module load miniconda3
eval "$(conda shell.bash hook)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cd "$PROJECT_DIR" || exit 1
conda activate diffusion_llms
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
export GIT_PYTHON_REFRESH=quiet
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1

TASK="${TASK:-gsm8k}"
N="${N:-20}"
K="${K:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
GEN_LENGTH="${GEN_LENGTH:-1024}"

echo "=========================================="
echo "E1 pilot: $TASK  N=$N  K=$K  temp=$TEMPERATURE  gen_length=$GEN_LENGTH"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "=========================================="
nvidia-smi
echo ""

python scripts/run_e1_pilot.py \
    --task        "$TASK" \
    --n           "$N" \
    --k           "$K" \
    --temperature "$TEMPERATURE" \
    --gen-length  "$GEN_LENGTH"

EXIT_CODE=$?
echo "Exit: $EXIT_CODE  Finished: $(date)"
exit $EXIT_CODE
