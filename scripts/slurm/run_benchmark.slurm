#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --partition=gpuh200
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/slurm/benchmark-%j.out
#SBATCH --error=logs/slurm/benchmark-%j.err

# ── Overrides (pass as env vars before sbatch) ────────────────────────────────
#   TASK             : gsm8k | math500 | humaneval | mbpp   (default: gsm8k)
#   SAMPLER          : cdf | low_conf                       (default: cdf)
#   MODEL            : HF model id                          (default: LLaDA-8B-Instruct)
#   N                : number of examples                   (default: 50)
#   GEN_LENGTH       : canvas size                          (default: 1024)
#   STEPS            : denoising steps = block_size         (default: GEN_LENGTH)
#   Q                : CDF quantile                         (default: 0.90)
#   EOS_FREE         : 1 to suppress EOS from vocab         (default: 0)
#   THRESHOLD        : confidence-threshold decoding, e.g. 0.9 (default: unset = one token/step)
#   NO_CROP          : 1 to skip the t=1 probe and denoise the full canvas (default: 0)
#   MAX_RESUBMITS    : resubmit up to this many times when the time limit is near (default: 0).
#                      Needs --requeue --signal=B:USR1@300 on the sbatch line; the run resumes by doc_id.
#   RIGHT_SHIFT_LOGITS: 1 for AR-shift models (auto-set for DREAM) (default: 0)
#   ASSISTANT_PREFIX : prefix after generation prompt       (default: <reasoning>)
#
# Examples:
#   TASK=gsm8k N=1319 sbatch scripts/slurm/run_benchmark.slurm
#   TASK=math500 N=500 sbatch scripts/slurm/run_benchmark.slurm
#   TASK=humaneval Q=0.95 sbatch scripts/slurm/run_benchmark.slurm
#   MODEL=Dream-org/Dream-v0-Instruct-7B N=50 sbatch --partition=medium_gpuh200 scripts/slurm/run_benchmark.slurm

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
SAMPLER="${SAMPLER:-cdf}"
MODEL="${MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
N="${N:-50}"
GEN_LENGTH="${GEN_LENGTH:-1024}"
STEPS="${STEPS:-$GEN_LENGTH}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
Q="${Q:-0.90}"
EOS_FREE="${EOS_FREE:-0}"
RIGHT_SHIFT_LOGITS="${RIGHT_SHIFT_LOGITS:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"

EOS_FREE_FLAG=""
[ "$EOS_FREE" = "1" ] && EOS_FREE_FLAG="--eos-free"

THRESHOLD_FLAG=""
[ -n "${THRESHOLD:-}" ] && THRESHOLD_FLAG="--threshold $THRESHOLD"

NO_CROP_FLAG=""
[ "${NO_CROP:-0}" = "1" ] && NO_CROP_FLAG="--no-crop"

RIGHT_SHIFT_FLAG=""
[ "$RIGHT_SHIFT_LOGITS" = "1" ] && RIGHT_SHIFT_FLAG="--right-shift-logits"

# Let run_benchmark.py choose the default prefix per task (empty for code, <reasoning> for math).
# Only pass --assistant-prefix if explicitly overridden via env var.
ASSISTANT_PREFIX_FLAG=""
[ -n "${ASSISTANT_PREFIX+x}" ] && ASSISTANT_PREFIX_FLAG="--assistant-prefix $ASSISTANT_PREFIX"

echo "=========================================="
echo "Benchmark: $TASK  Sampler: $SAMPLER"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "N: $N  GenLength: $GEN_LENGTH  Steps: $STEPS  BlockSize: $BLOCK_SIZE  Q: $Q  EosFree: $EOS_FREE  Threshold: ${THRESHOLD:-none}  NoCrop: ${NO_CROP:-0}  Restarts: ${SLURM_RESTART_COUNT:-0}/${MAX_RESUBMITS:-0}  RightShift: $RIGHT_SHIFT_LOGITS"
echo "=========================================="
nvidia-smi
echo ""

MAX_RESUBMITS="${MAX_RESUBMITS:-0}"
RESTARTS="${SLURM_RESTART_COUNT:-0}"

# Requeue keeps the job ID, so it needs no free slot under the per-user submit limit.
resubmit() {
    echo "Time limit near: stopping job $SLURM_JOB_ID ($(date))"
    kill -TERM "$PY_PID" 2>/dev/null || true
    wait "$PY_PID" || true
    if [ "$RESTARTS" -lt "$MAX_RESUBMITS" ]; then
        echo "Requeueing (restart $((RESTARTS + 1))/$MAX_RESUBMITS)"
        scontrol requeue "$SLURM_JOB_ID"
    else
        echo "No resubmits left ($RESTARTS/$MAX_RESUBMITS)"
    fi
    exit 0
}
[ "$MAX_RESUBMITS" -gt 0 ] && trap resubmit USR1

python scripts/run_benchmark.py \
    --task             "$TASK" \
    --sampler          "$SAMPLER" \
    --model            "$MODEL" \
    --n                "$N" \
    --gen-length       "$GEN_LENGTH" \
    --steps            "$STEPS" \
    --block-size       "$BLOCK_SIZE" \
    --q                "$Q" \
    --batch-size       "$BATCH_SIZE" \
    $ASSISTANT_PREFIX_FLAG \
    $EOS_FREE_FLAG \
    $THRESHOLD_FLAG \
    $NO_CROP_FLAG \
    $RIGHT_SHIFT_FLAG &
PY_PID=$!

EXIT_CODE=0
wait "$PY_PID" || EXIT_CODE=$?
echo "Exit: $EXIT_CODE  Finished: $(date)"
exit $EXIT_CODE
