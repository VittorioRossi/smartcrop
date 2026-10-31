#!/bin/bash
#SBATCH --job-name="dream-fixed-all"
#SBATCH --partition=long_gpuh200
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/slurm/dream_fixed_all-%j.out
#SBATCH --error=logs/slurm/dream_fixed_all-%j.err

# ── Overrides (pass as env vars before sbatch) ────────────────────────────────
#   MODEL      : HF model id          (default: Dream-org/Dream-v0-Instruct-7B)
#   GEN_LENGTHS: space-separated list (default: "64 128 256 512 1024 2048")
#
# N is set per task to the full dataset size (gsm8k=1319, math500=500,
# humaneval=164, mbpp=374). Override per task via N_GSM8K, N_MATH500, etc.
#
# Runs all 6 canvas sizes × 4 tasks sequentially; each task gets all 4 GPUs
# via device_map="auto".

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

MODEL="${MODEL:-Dream-org/Dream-v0-Instruct-7B}"
GEN_LENGTHS="${GEN_LENGTHS:-64 128 256 512 1024 2048}"

N_GSM8K="${N_GSM8K:-1319}"
N_MATH500="${N_MATH500:-500}"
N_HUMANEVAL="${N_HUMANEVAL:-164}"
N_MBPP="${N_MBPP:-374}"

echo "=========================================="
echo "DREAM Fixed-Length — all 6 canvas sizes × 4 tasks  (2 GPUs)"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL  GL: $GEN_LENGTHS"
echo "N: gsm8k=$N_GSM8K  math500=$N_MATH500  humaneval=$N_HUMANEVAL  mbpp=$N_MBPP"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
nvidia-smi
echo ""

run_task() {
    local TASK=$1
    local N=$2
    local GL=$3
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

for GL in $GEN_LENGTHS; do
    echo ""
    echo "══════════  GL=$GL  ══════════"
    run_task gsm8k     $N_GSM8K     $GL
    run_task math500   $N_MATH500   $GL
    run_task humaneval $N_HUMANEVAL $GL
    run_task mbpp      $N_MBPP      $GL
done

echo ""
echo "All canvas sizes complete.  Finished: $(date)"
