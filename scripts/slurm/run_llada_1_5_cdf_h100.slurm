#!/bin/bash
#SBATCH --job-name="llada1.5-cdf-h100"
#SBATCH --partition=medium_gpunew
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/slurm/llada1_5_cdf_h100-%j.out
#SBATCH --error=logs/slurm/llada1_5_cdf_h100-%j.err

# ── Overrides (pass as env vars before sbatch) ────────────────────────────────
#   MODEL      : HF model id          (default: GSAI-ML/LLaDA-1.5)
#   GEN_LENGTH : canvas size          (default: 1024)
#   BLOCK_SIZE : block size           (default: 32)
#   Q          : CDF quantile         (default: 0.90)
#
# H100 (gpunew) variant — single GPU. Runs all 4 tasks sequentially.

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

MODEL="${MODEL:-GSAI-ML/LLaDA-1.5}"
GEN_LENGTH="${GEN_LENGTH:-1024}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
Q="${Q:-0.90}"

# Full dataset sizes
N_GSM8K="${N_GSM8K:-1319}"
N_MATH500="${N_MATH500:-500}"
N_HUMANEVAL="${N_HUMANEVAL:-164}"
N_MBPP="${N_MBPP:-374}"

echo "=========================================="
echo "LLaDA-1.5 CDF (H100) — all 4 tasks  (sequential)"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL  Q: $Q  GL: $GEN_LENGTH  BS: $BLOCK_SIZE"
echo "N: gsm8k=$N_GSM8K  math500=$N_MATH500  humaneval=$N_HUMANEVAL  mbpp=$N_MBPP"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
nvidia-smi
echo ""

run_task() {
    local TASK=$1
    local N=$2
    echo ">>> $TASK (n=$N) start  $(date)"
    python scripts/run_benchmark.py \
        --task       "$TASK" \
        --sampler    cdf \
        --model      "$MODEL" \
        --n          "$N" \
        --gen-length "$GEN_LENGTH" \
        --steps      "$GEN_LENGTH" \
        --block-size "$BLOCK_SIZE" \
        --q          "$Q"
    echo "<<< $TASK done   $(date)"
}

run_task gsm8k     $N_GSM8K
run_task math500   $N_MATH500
run_task humaneval $N_HUMANEVAL
run_task mbpp      $N_MBPP

echo ""
echo "All 4 tasks complete.  Finished: $(date)"
