#!/bin/bash
# DAEDAL / rho-EOS Asym on LLaDA-8B-Instruct at the authors' L_init=64, batch size 1.
# Batch size 1 matches SmartCrop's timing protocol and avoids DAEDAL padding every
# sequence in a batch to the batch's longest canvas.
#
# Env:
#   METHOD : daedal | rho_asym      (required)
#   TASK   : gsm8k | math500 | mbpp | humaneval   (required)
#   LIMIT  : e.g. 4 for a smoke test, empty for the full run
set -e

: "${METHOD:?METHOD required}" "${TASK:?TASK required}"
L_INIT=64
BATCH_SIZE=1
LIMIT="${LIMIT:-}"

mkdir -p ./logs
LOG_FILE="./logs/run_${METHOD}_init${L_INIT}_bs${BATCH_SIZE}_${TASK}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -i "$LOG_FILE") 2>&1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

COMMON="block_length=32,initial_gen_length=${L_INIT},max_gen_length=2048,cfg_scale=0.0,high_conf_threshold=0.9,low_conf_threshold=0.1,eos_confidence_threshold=0.5,expand_eos_confidence_threshold=0.9,expansion_factor=8,eos_check_tokens=32"
case "$METHOD" in
    daedal)   MODEL=LLaDA_DAEDAL;  GEN_KWARGS="$COMMON" ;;
    rho_asym) MODEL=LLaDA_rho_EOS; GEN_KWARGS="low_density_threshold=0.4,high_density_threshold=0.8,scheduler=exp,$COMMON" ;;
    *) echo "Unknown METHOD: $METHOD"; exit 1 ;;
esac

EXTRA_ARGS=""
MODEL_ARGS="pretrained=GSAI-ML/LLaDA-8B-Instruct,assistant_prefix=<reasoning>"
# Same dataset caches as the L_init=1024 runs: math from the default cache, code from the rho-EOS one.
case "$TASK" in
    mbpp|humaneval)
        EXTRA_ARGS="--confirm_run_unsafe_code"
        export HF_DATASETS_CACHE=$HOME/.cache/huggingface_rho_eos/datasets ;;
esac
[ -n "$LIMIT" ] && EXTRA_ARGS="$EXTRA_ARGS --limit ${LIMIT}"

OUTPUT_PATH="./results/${METHOD}-init${L_INIT}-bs${BATCH_SIZE}${LIMIT:+-smoke}-$(date +%Y%m%d_%H%M%S)/${TASK}_${L_INIT}"
echo "<<${METHOD}>> task=${TASK} L_init=${L_INIT} batch_size=${BATCH_SIZE} limit=${LIMIT:-none} -> ${OUTPUT_PATH}"

accelerate launch --config_file accelerate_config_1gpu.yaml evaluation_script.py \
    -m dllm_eval \
    --model "$MODEL" \
    --tasks "$TASK" \
    --batch_size "$BATCH_SIZE" \
    --model_args "$MODEL_ARGS" \
    --gen_kwargs "$GEN_KWARGS" \
    --num_fewshot 0 \
    --output_path "$OUTPUT_PATH" \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn \
    $EXTRA_ARGS

python metrics/${TASK}.py --model_path GSAI-ML/LLaDA-8B-Instruct --res_path "$OUTPUT_PATH"
