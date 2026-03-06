#!/bin/bash
#SBATCH --job-name="smartcrop-full-eval"
#SBATCH --account=3188641
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/home/3188641/smartcrop
#SBATCH --output=/home/3188641/logs/slurm/smartcrop_eval-%j.out
#SBATCH --error=/home/3188641/logs/slurm/smartcrop_eval-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vittorio.rossi2@studbocconi.it

# ============================================================================
# SmartCrop Full Benchmark Evaluation
# Runs all 4 tasks x 6 quantiles with proper multi-GPU support
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
    echo "Please run 'conda create -n diffusion_llms python=3.10 -y' manually first."
    exit 1
fi

conda activate diffusion_llms
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
export GIT_PYTHON_REFRESH=quiet

# ---------------- MULTI-GPU SETUP ----------------
export OMP_NUM_THREADS=1
export ACCELERATE_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset ACCELERATE_CONFIG_FILE

# Get a random port to avoid collisions with other jobs
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

export OUTPUT_DIR="${PROJECT_DIR}/results/preprint/453436_benchmark"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$USER_HOME/logs/slurm"  # Ensure log directory exists

# ---------------- LOGGING ----------------
echo "=========================================="
echo "SmartCrop Full Benchmark Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

echo "--- Environment Info ---"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MASTER_PORT: $MASTER_PORT"
echo ""

echo "--- GPU Status ---"
nvidia-smi
echo ""

echo "--- Starting Evaluation ---"
echo ""

# ---------------- RUN BENCHMARKS ----------------
# Run the benchmark runner script with plain python.
# The runner internally uses accelerate launch for each task's eval.py call.
python scripts/run_slurm_benchmarks.py \
    --output_dir "$OUTPUT_DIR" \
    --num_processes 1 \
    --master_port $MASTER_PORT

EXIT_CODE=$?

# ---------------- CLEANUP & SUMMARY ----------------
echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="

# List output files
echo ""
echo "--- Output Files ---"
find "$OUTPUT_DIR" -type f -name "*.jsonl" -o -name "*.json" | head -20

exit $EXIT_CODE
