#!/bin/bash
#SBATCH --job-name="ctx-invariance"
#SBATCH --account=3188641
#SBATCH --partition=gpunew
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/home/3188641/smartcrop
#SBATCH --output=/home/3188641/logs/slurm/ctx_invariance-%j.out
#SBATCH --error=/home/3188641/logs/slurm/ctx_invariance-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vittorio.rossi2@studbocconi.it

# ============================================================================
# Context Invariance Test for Zero-Shot Length Prediction
# Tests whether predicted length is invariant to context size
# ============================================================================

set -e  # Exit on error

# ---------------- ENVIRONMENT SETUP ----------------
module load miniconda3
eval "$(conda shell.bash hook)"

USER_HOME=$(eval echo ~$USER)
PROJECT_DIR="$USER_HOME/smartcrop"

cd "$PROJECT_DIR" || { echo "Error: Project dir not found at $PROJECT_DIR"; exit 1; }

# Check conda environment exists
if ! conda env list | grep -q "^diffusion_llms "; then
    echo "Error: Environment 'diffusion_llms' not found."
    exit 1
fi

conda activate diffusion_llms
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
export GIT_PYTHON_REFRESH=quiet

# ---------------- OUTPUT SETUP ----------------
export OUTPUT_DIR="${PROJECT_DIR}/results/context_invariance/${SLURM_JOB_ID}_run"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$USER_HOME/logs/slurm"

# ---------------- LOGGING ----------------
echo "==========================================="
echo "Context Invariance Test"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "==========================================="
echo ""

echo "--- Environment Info ---"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

echo "--- GPU Status ---"
nvidia-smi
echo ""

echo "--- Starting Evaluation ---"
echo ""

# ---------------- RUN EXPERIMENT ----------------
python scripts/run_context_invariance.py \
    --output_dir "$OUTPUT_DIR" \
    --tasks all \
    --eos_quantile 0.9

EXIT_CODE=$?

# ---------------- GENERATE PLOTS ----------------
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "--- Generating Plots ---"
    for TASK_DIR in "$OUTPUT_DIR"/*/; do
        if [ -f "${TASK_DIR}results.jsonl" ]; then
            TASK_NAME=$(basename "$TASK_DIR")
            echo "Generating plots for $TASK_NAME..."
            python scripts/plot_context_invariance.py \
                --input_path "${TASK_DIR}results.jsonl" \
                --output_dir "$TASK_DIR"
        fi
    done
fi

# ---------------- CLEANUP & SUMMARY ----------------
echo ""
echo "==========================================="
echo "Experiment Complete"
echo "==========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Results: $OUTPUT_DIR"
echo "==========================================="

# List output files
echo ""
echo "--- Output Files ---"
find "$OUTPUT_DIR" -type f -name "*.png" -o -name "*.json" | head -30

exit $EXIT_CODE
