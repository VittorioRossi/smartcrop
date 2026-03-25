#!/bin/bash
#SBATCH --job-name="static-rebuttal"
#SBATCH --account=3188641
#SBATCH --partition=medium_gpunew
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/home/3188641/smartcrop
#SBATCH --output=/home/3188641/logs/slurm/static_eval-%j.out
#SBATCH --error=/home/3188641/logs/slurm/static_eval-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vittorio.rossi2@studbocconi.it

# ============================================================================
# Static-Mean Baseline Evaluation (Rebuttal)
#
# Runs standard fixed-canvas diffusion where max_new_tokens = per-task mean
# predicted length from SmartCrop (τ=0.9). Directly answers the reviewer
# question: does SmartCrop beat simply knowing the task-level average length?
#
# Mean lengths (from run 453436, τ=0.9):
#   gsm8k_cot_llama:          162 tokens
#   humaneval_instruct_local: 377 tokens
#   ifeval:                   145 tokens
#   longformqa:               97  tokens
# ============================================================================

set -e

# ---------------- ENVIRONMENT SETUP ----------------
module load miniconda3
eval "$(conda shell.bash hook)"

USER_HOME=$(eval echo ~$USER)
PROJECT_DIR="$USER_HOME/smartcrop"

cd "$PROJECT_DIR" || { echo "Error: Project dir not found at $PROJECT_DIR"; exit 1; }

if ! conda env list | grep -q "^diffusion_llms "; then
    echo "Error: Conda environment 'diffusion_llms' not found."
    exit 1
fi

conda activate diffusion_llms
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
export GIT_PYTHON_REFRESH=quiet

# ---------------- GPU SETUP ----------------
export OMP_NUM_THREADS=1
export ACCELERATE_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset ACCELERATE_CONFIG_FILE

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

export OUTPUT_DIR="${PROJECT_DIR}/results/rebuttal/static_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$USER_HOME/logs/slurm"

# ---------------- LOGGING ----------------
echo "=========================================="
echo "Static-Mean Baseline Evaluation (Rebuttal)"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"
echo "Output:    $OUTPUT_DIR"
echo "Mean lengths (τ=0.9 from run 453436):"
echo "  gsm8k_cot_llama=162  humaneval_instruct_local=377"
echo "  ifeval=145           longformqa=97"
echo "=========================================="
echo ""

echo "--- Environment ---"
echo "Python: $(which python)"
echo "Conda:  $CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

echo "--- GPU Status ---"
nvidia-smi
echo ""

echo "--- Starting Evaluation ---"

# ---------------- RUN STATIC BASELINE ----------------
python scripts/run_static_baseline.py \
    --output_dir "$OUTPUT_DIR" \
    --num_processes 1 \
    --master_port $MASTER_PORT

EXIT_CODE=$?

# ---------------- SUMMARY ----------------
echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time:  $(date)"
echo "Results:   $OUTPUT_DIR"
echo "=========================================="

echo ""
echo "--- Output Files ---"
find "$OUTPUT_DIR" -type f \( -name "*.jsonl" -o -name "*.json" \) | head -20

exit $EXIT_CODE
