#!/bin/bash
#SBATCH --job-name="llada-canvas"
#SBATCH --account=3188641
#SBATCH --partition=long_gpunew
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/home/3188641/smartcrop
#SBATCH --output=/home/3188641/logs/slurm/llada_canvas-%j.out
#SBATCH --error=/home/3188641/logs/slurm/llada_canvas-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vittorio.rossi2@studbocconi.it

# Usage:
#   MODEL=GSAI-ML/LLaDA-8B-Instruct sbatch scripts/slurm/run_llada_canvas_sweep.slurm
#   MODEL=GSAI-ML/LLaDA-1.5 sbatch scripts/slurm/run_llada_canvas_sweep.slurm

set -e

module load miniconda3
eval "$(conda shell.bash hook)"

USER_HOME=$(eval echo ~$USER)
PROJECT_DIR="$USER_HOME/smartcrop"

cd "$PROJECT_DIR" || { echo "Error: Project dir not found"; exit 1; }

conda activate diffusion_llms
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
export GIT_PYTHON_REFRESH=quiet
export HF_ALLOW_CODE_EVAL=1
export OMP_NUM_THREADS=1
unset ACCELERATE_CONFIG_FILE

MODEL="${MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
MODEL_TAG=$(echo "$MODEL" | tr '/' '__')

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/rebuttal/llada_canvas_sweep_${MODEL_TAG}_${SLURM_JOB_ID}}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$USER_HOME/logs/slurm"

echo "=========================================="
echo "LLaDA Canvas Size Sweep (Rebuttal)"
echo "=========================================="
echo "Model:   $MODEL"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Start:   $(date)"
echo "Output:  $OUTPUT_DIR"
echo "=========================================="

nvidia-smi

python scripts/run_llada_canvas_sweep.py \
    --model "$MODEL" \
    --canvas_sizes 128 256 379 512 1024 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

EXIT_CODE=$?

echo ""
echo "Done. Exit: $EXIT_CODE  |  $(date)"
echo "Results: $OUTPUT_DIR"
exit $EXIT_CODE
