#!/bin/bash
#SBATCH --job-name="dream-fixed-gl"
#SBATCH --partition=long_gpuh200
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/slurm/dream_fixed_gl%x-%j.out
#SBATCH --error=logs/slurm/dream_fixed_gl%x-%j.err

# Required: GL — canvas size (e.g. GL=512 sbatch ...)
# Optional: MODEL, N_GSM8K, N_MATH500, N_HUMANEVAL, N_MBPP

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

if [ -z "$GL" ]; then
    echo "ERROR: GL env var is required (e.g. GL=512 sbatch ...)"
    exit 1
fi

MODEL="${MODEL:-Dream-org/Dream-v0-Instruct-7B}"
N_GSM8K="${N_GSM8K:-1319}"
N_MATH500="${N_MATH500:-500}"
N_HUMANEVAL="${N_HUMANEVAL:-164}"
N_MBPP="${N_MBPP:-374}"

echo "=========================================="
echo "DREAM Fixed-Length — GL=$GL — all 4 tasks"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL"
echo "N: gsm8k=$N_GSM8K  math500=$N_MATH500  humaneval=$N_HUMANEVAL  mbpp=$N_MBPP"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
nvidia-smi
echo ""

run_task() {
    local TASK=$1
    local N=$2
    echo ">>> $TASK  GL=$GL  (n=$N)  start  $(date)"
    python scripts/run_benchmark.py \
        --task       "$TASK" \
        --sampler    low_conf \
        --model      "$MODEL" \
        --n          "$N" \
        --gen-length "$GL" \
        --steps      "$GL"
    echo "<<< $TASK  GL=$GL  done   $(date)"
}

# SKIP_TASKS: space-separated list of tasks to skip (e.g. "gsm8k")
SKIP_TASKS="${SKIP_TASKS:-}"

maybe_run() {
    local TASK=$1; local N=$2
    if echo "$SKIP_TASKS" | grep -qw "$TASK"; then
        echo "--- skipping $TASK (SKIP_TASKS=$SKIP_TASKS)"
    else
        run_task "$TASK" "$N"
    fi
}

maybe_run gsm8k     $N_GSM8K
maybe_run math500   $N_MATH500
maybe_run humaneval $N_HUMANEVAL
maybe_run mbpp      $N_MBPP

echo ""
echo "GL=$GL complete.  Finished: $(date)"
