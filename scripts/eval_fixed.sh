#!/usr/bin/env bash
set -euo pipefail

cd /home/tjn2004/uav/UAV_ON

DATASET_PATH="${DATASET_PATH:-/home/tjn2004/uav/DATASET/UAV-ON-data/valset/CabinLake.json}"
EVAL_SAVE_PATH="${EVAL_SAVE_PATH:-/home/tjn2004/uav/UAV_ON/logs/scene}"
MAX_ACTIONS="${MAX_ACTIONS:-150}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SIMULATOR_TOOL_PORT="${SIMULATOR_TOOL_PORT:-30000}"
IS_FIXED="${IS_FIXED:-true}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASKS="${1:-${EVAL_TASKS:-}}"

if [ -n "${TASKS}" ]; then
    python scripts/eval_tasks.py \
        --tasks "${TASKS}" \
        --dataset_path "${DATASET_PATH}" \
        --eval_save_path "${EVAL_SAVE_PATH}" \
        --maxActions "${MAX_ACTIONS}" \
        --is_fixed "${IS_FIXED}" \
        --gpu_id "${GPU_ID}" \
        --batchSize "${BATCH_SIZE}" \
        --simulator_tool_port "${SIMULATOR_TOOL_PORT}" \
        --cuda_visible_devices "${CUDA_DEVICES}"
else
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

    python -u src/eval_2.py \
        --maxActions "${MAX_ACTIONS}" \
        --eval_save_path "${EVAL_SAVE_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --is_fixed "${IS_FIXED}" \
        --gpu_id "${GPU_ID}" \
        --batchSize "${BATCH_SIZE}" \
        --simulator_tool_port "${SIMULATOR_TOOL_PORT}"
fi