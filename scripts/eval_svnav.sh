
# ----------------------------------------------------------------------
# SVNav auto log capture
# ----------------------------------------------------------------------
# Save every eval_svnav.sh run to logs/svnav_debug automatically.
# The script re-executes itself once through tee. The guard variable prevents
# recursion. Set SVNAV_DISABLE_AUTO_LOG=1 to disable this behavior.
if [ "${SVNAV_DISABLE_AUTO_LOG:-0}" != "1" ] && [ "${SVNAV_EVAL_LOG_TEE_ACTIVE:-0}" != "1" ]; then
    export SVNAV_EVAL_LOG_TEE_ACTIVE=1

    SVNAV_LOG_DIR="${SVNAV_LOG_DIR:-logs/svnav_debug}"
    mkdir -p "$SVNAV_LOG_DIR"

    SVNAV_LOG_FILE="${SVNAV_LOG_FILE:-$SVNAV_LOG_DIR/svnav_eval_$(date +%Y%m%d_%H%M%S).log}"
    SVNAV_LATEST_LOG="$SVNAV_LOG_DIR/latest.log"

    echo "[SVNavLog] saving eval output to: $SVNAV_LOG_FILE"
    echo "[SVNavLog] latest log mirror: $SVNAV_LATEST_LOG"

    set -o pipefail
    PYTHONUNBUFFERED=1 bash "$0" "$@" 2>&1 | tee "$SVNAV_LOG_FILE" "$SVNAV_LATEST_LOG"
    exit ${PIPESTATUS[0]}
fi

cd "$(dirname "$0")/.."

root_dir=.
echo $PWD

CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/eval_svnav.py \
  --name SVNav-Step1 \
  --maxActions 150 \
  --eval_save_path $root_dir/logs/scene \
  --dataset_path /home/tjn2004/uav/DATASET/UAV-ON-data/valset/Slum.json \
  --is_fixed true \
  --gpu_id 0 \
  --batchSize 1 \
  --simulator_tool_port 30000
