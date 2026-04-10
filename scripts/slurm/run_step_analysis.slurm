#!/bin/bash
#SBATCH --job-name="step-analysis"
#SBATCH --account=3188641
#SBATCH --partition=medium_gpunew
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/home/3188641/smartcrop
#SBATCH --output=/home/3188641/logs/slurm/step_analysis-%j.out
#SBATCH --error=/home/3188641/logs/slurm/step_analysis-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vittorio.rossi2@studbocconi.it

# Usage:
#   TASK=gsm8k MODEL=GSAI-ML/LLaDA-8B-Instruct sbatch scripts/slurm/run_step_analysis.slurm
#
# Supported TASK values: gsm8k, humaneval, mbpp, ifeval, math500
# Supported MODEL values: GSAI-ML/LLaDA-8B-Instruct, GSAI-ML/LLaDA-1.5

set -e

module load miniconda3
eval "$(conda shell.bash hook)"

USER_HOME=$(eval echo ~$USER)
PROJECT_DIR="$USER_HOME/smartcrop"
cd "$PROJECT_DIR"
conda activate diffusion_llms
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
export GIT_PYTHON_REFRESH=quiet
export OMP_NUM_THREADS=1
export HF_ALLOW_CODE_EVAL=1

TASK="${TASK:-gsm8k}"
MODEL="${MODEL:-GSAI-ML/LLaDA-8B-Instruct}"

# Safe model tag for directory names (replace / with __)
MODEL_TAG=$(echo "$MODEL" | tr '/' '__')

OUTPUT_DIR="${PROJECT_DIR}/results/rebuttal/step_analysis_${TASK}_${MODEL_TAG}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$USER_HOME/logs/slurm"

echo "Step Analysis"
echo "Task:   $TASK"
echo "Model:  $MODEL"
echo "Job:    $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
echo "Output: $OUTPUT_DIR"
nvidia-smi && echo ""

python scripts/run_gsm8k_step_analysis.py \
    --task "$TASK" \
    --model "$MODEL" \
    --canvas_sizes 2048 \
    --output_dir "$OUTPUT_DIR" \
    ${N_EXAMPLES:+--n_examples $N_EXAMPLES}

EXIT_CODE=$?
echo "Exit: $EXIT_CODE  Finished: $(date)"
exit $EXIT_CODE
