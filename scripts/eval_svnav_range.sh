#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ $# -lt 2 ]; then
  echo "Usage:"
  echo "  bash scripts/eval_svnav_range.sh <scene|scene.json|json_path> <task_id|start-end> [max_actions]"
  echo ""
  echo "Examples:"
  echo "  bash scripts/eval_svnav_range.sh Slum 1-4"
  echo "  bash scripts/eval_svnav_range.sh Slum 1-1"
  echo "  bash scripts/eval_svnav_range.sh CabinLake 3 150"
  echo "  bash scripts/eval_svnav_range.sh /home/tjn2004/uav/DATASET/UAV-ON-data/valset/Slum.json 2-5 150"
  exit 1
fi

SCENE_ARG="$1"
RANGE_ARG="$2"
MAX_ACTIONS="${3:-150}"

DATA_ROOT="${UAVON_DATA_ROOT:-/home/tjn2004/uav/DATASET/UAV-ON-data/valset}"
TMP_ROOT="${SVNAV_TMP_DATASET_DIR:-/tmp/svnav_eval_ranges}"
mkdir -p "$TMP_ROOT"

if [[ "$SCENE_ARG" == /* && -f "$SCENE_ARG" ]]; then
  SRC_JSON="$SCENE_ARG"
  SCENE_NAME="$(basename "$SCENE_ARG" .json)"
elif [[ "$SCENE_ARG" == *.json && -f "$SCENE_ARG" ]]; then
  SRC_JSON="$SCENE_ARG"
  SCENE_NAME="$(basename "$SCENE_ARG" .json)"
else
  SCENE_NAME="${SCENE_ARG%.json}"
  SRC_JSON="$DATA_ROOT/${SCENE_NAME}.json"
fi

if [ ! -f "$SRC_JSON" ]; then
  echo "[ERROR] Scene json not found: $SRC_JSON"
  echo "Set UAVON_DATA_ROOT if your dataset root is different."
  exit 1
fi

if [[ "$RANGE_ARG" == *-* ]]; then
  START_ID="${RANGE_ARG%-*}"
  END_ID="${RANGE_ARG#*-}"
else
  START_ID="$RANGE_ARG"
  END_ID="$RANGE_ARG"
fi

if ! [[ "$START_ID" =~ ^[0-9]+$ && "$END_ID" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] Range must be like 1, 1-1, or 1-4. Got: $RANGE_ARG"
  exit 1
fi

if [ "$START_ID" -gt "$END_ID" ]; then
  echo "[ERROR] start task id must be <= end task id. Got: $RANGE_ARG"
  exit 1
fi

OUT_JSON="$TMP_ROOT/${SCENE_NAME}_tasks_${START_ID}_${END_ID}.json"

python - <<PY
import json
from pathlib import Path

src = Path("$SRC_JSON")
out = Path("$OUT_JSON")
start = int("$START_ID")
end = int("$END_ID")

data = json.loads(src.read_text())

def slice_list(items):
    n = len(items)
    if start < 0 or end >= n:
        raise IndexError(f"Task range {start}-{end} out of bounds for {src}, total tasks={n}")
    selected = items[start:end + 1]
    return selected, n

if isinstance(data, list):
    selected, total = slice_list(data)
    output = selected
elif isinstance(data, dict):
    output = dict(data)
    total = None
    task_key = None
    for key in ["episodes", "tasks", "data"]:
        if key in output and isinstance(output[key], list):
            selected, total = slice_list(output[key])
            output[key] = selected
            task_key = key
            break
    if task_key is None:
        raise RuntimeError(f"Cannot find task list key in {src}. Expected one of: episodes/tasks/data")
else:
    raise RuntimeError(f"Unsupported json root type: {type(data)}")

out.write_text(json.dumps(output, ensure_ascii=False, indent=2))

print("[SVNavRange] source:", src)
print("[SVNavRange] scene:", "$SCENE_NAME")
print("[SVNavRange] original task range:", f"{start}-{end}")
print("[SVNavRange] total tasks in source:", total)
print("[SVNavRange] selected tasks:", end - start + 1)
print("[SVNavRange] saved subset:", out)

# Print brief task preview for sanity.
preview_items = selected[:5]
for offset, item in enumerate(preview_items):
    original_id = start + offset
    if isinstance(item, dict):
        info = item.get("info", {})
        name = (
            info.get("true_name")
            or info.get("object_name")
            or item.get("true_name")
            or item.get("object_name")
            or item.get("name")
            or "unknown"
        )
        start_pose = item.get("start_pose", {})
        start_position = None
        if isinstance(start_pose, dict):
            start_position = start_pose.get("start_position")
        print(f"[SVNavRange] original_task={original_id} target={name} start={start_position}")
    else:
        print(f"[SVNavRange] original_task={original_id} type={type(item).__name__}")
PY

RUN_NAME="SVNav-${SCENE_NAME}-tasks-${START_ID}-${END_ID}"

echo "[SVNavRange] run_name: $RUN_NAME"
echo "[SVNavRange] max_actions: $MAX_ACTIONS"
echo "[SVNavRange] dataset_path: $OUT_JSON"

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u "$ROOT_DIR/src/eval_svnav.py" \
  --name "$RUN_NAME" \
  --maxActions "$MAX_ACTIONS" \
  --eval_save_path "$ROOT_DIR/logs/scene" \
  --dataset_path "$OUT_JSON" \
  --is_fixed true \
  --gpu_id 0 \
  --batchSize 1 \
  --simulator_tool_port 30000
