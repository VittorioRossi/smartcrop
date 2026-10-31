#!/bin/bash
set -e

LOG_FILE="./logs/run_rho_eos_asym_1024_$(date +%Y%m%d_%H%M%S).log"
mkdir -p ./logs
exec > >(tee -i "$LOG_FILE") 2>&1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RHO_LOW=0.4
RHO_HIGH=0.8
SCHEDULER="exp"
LIMIT="${LIMIT:-}"   # set to e.g. 4 for smoke test, empty for full run

NUM_GPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
case "$NUM_GPU" in
    1)  ACCEL_CONFIG="accelerate_config_1gpu.yaml" ;;
    2)  ACCEL_CONFIG="accelerate_config_2gpu.yaml" ;;
    4)  ACCEL_CONFIG="accelerate_config_4gpu.yaml" ;;
    8)  ACCEL_CONFIG="accelerate_config.yaml" ;;
    *)  echo "Unsupported GPU count: $NUM_GPU"; exit 1 ;;
esac
echo "Detected $NUM_GPU GPU(s), using $ACCEL_CONFIG"

BASE_OUTPUT_PATH="./results/rho-eos-asym-1024-$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"

TASKS_MATH=("gsm8k" "math500")
TASKS_CODE=("mbpp" "humaneval")

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit ${LIMIT}"
fi

for task in "${TASKS_MATH[@]}"; do
    echo "======================================================"
    echo "<<rho-EOS-Asym>> -> Task: ${task}, L_init: 1024"
    echo "======================================================"
    OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_1024"

    accelerate launch --config_file "$ACCEL_CONFIG" evaluation_script.py \
        -m dllm_eval \
        --model LLaDA_rho_EOS \
        --tasks "${task}" \
        --batch_size 8 \
        --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning>" \
        --gen_kwargs "low_density_threshold=${RHO_LOW},high_density_threshold=${RHO_HIGH},scheduler=${SCHEDULER},block_length=32,initial_gen_length=1024,max_gen_length=2048,cfg_scale=0.0,high_conf_threshold=0.9,low_conf_threshold=0.1,eos_confidence_threshold=0.5,expand_eos_confidence_threshold=0.9,expansion_factor=8,eos_check_tokens=32" \
        --num_fewshot 0 \
        --output_path "${OUTPUT_PATH}" \
        --log_samples \
        --apply_chat_template \
        --fewshot_as_multiturn \
        ${LIMIT_ARG}

    python metrics/${task}.py \
        --model_path "${MODEL_PATH}" \
        --res_path "${OUTPUT_PATH}"
done

for task in "${TASKS_CODE[@]}"; do
    echo "======================================================"
    echo "<<rho-EOS-Asym>> -> Task: ${task}, L_init: 1024"
    echo "======================================================"
    OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_1024"

    accelerate launch --config_file "$ACCEL_CONFIG" evaluation_script.py \
        -m dllm_eval \
        --model LLaDA_rho_EOS \
        --tasks "${task}" \
        --batch_size 8 \
        --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
        --gen_kwargs "low_density_threshold=${RHO_LOW},high_density_threshold=${RHO_HIGH},scheduler=${SCHEDULER},block_length=32,initial_gen_length=1024,max_gen_length=2048,cfg_scale=0.0,high_conf_threshold=0.9,low_conf_threshold=0.1,eos_confidence_threshold=0.5,expand_eos_confidence_threshold=0.9,expansion_factor=8,eos_check_tokens=32 " \
        --num_fewshot 0 \
        --output_path "${OUTPUT_PATH}" \
        --log_samples \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        ${LIMIT_ARG}

    python metrics/${task}.py \
        --model_path "${MODEL_PATH}" \
        --res_path "${OUTPUT_PATH}"
done
