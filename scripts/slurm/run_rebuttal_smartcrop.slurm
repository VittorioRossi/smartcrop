#!/bin/bash
#SBATCH --job-name="rebuttal-sc"
#SBATCH --account=3188641
#SBATCH --partition=long_gpunew
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/home/3188641/smartcrop
#SBATCH --output=/home/3188641/logs/slurm/rebuttal_sc-%j.out
#SBATCH --error=/home/3188641/logs/slurm/rebuttal_sc-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vittorio.rossi2@studbocconi.it

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

OUTPUT_DIR="${PROJECT_DIR}/results/rebuttal/smartcrop_rebuttal_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$USER_HOME/logs/slurm"

echo "SmartCrop Rebuttal Benchmarks (q=0.9, max_new_tokens=2048, block_length=2048)"
echo "Tasks: humaneval gsm8k ifeval bbh_cot_fewshot math500"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
echo "Output: $OUTPUT_DIR"
nvidia-smi && echo ""

python scripts/run_smartcrop_rebuttal.py \
    --output_dir "$OUTPUT_DIR" \
    --tasks humaneval_instruct_local gsm8k_cot_llama ifeval bbh_cot_fewshot math500

EXIT_CODE=$?
echo "Exit: $EXIT_CODE  Finished: $(date)"
exit $EXIT_CODE
