#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

root_dir="/home/tjn2004/uav/UAV_ON"

# ============================================================
# Stable batched random eval config
# ============================================================

# 场景名。也可以运行时临时指定：
#   SCENE_NAME=Slum bash scripts/eval_random_batched.sh 20 42 5
SCENE_NAME="${SCENE_NAME:-CabinLake}"

DATASET_ROOT="/home/tjn2004/uav/DATASET/UAV-ON-data/valset"
DATASET_PATH="/home/tjn2004/uav/DATASET/UAV-ON-data/valset/Barnyard.json"

# 用法：
#   bash scripts/eval_random_batched.sh
#   bash scripts/eval_random_batched.sh 20
#   bash scripts/eval_random_batched.sh 20 42
#   bash scripts/eval_random_batched.sh 20 42 5
TOTAL_TASKS="${1:-20}"
SEED="${2:-42}"
TASKS_PER_BATCH="${3:-5}"

MAX_ACTIONS="${MAX_ACTIONS:-100}"
BATCH_SIZE=1
GPU_ID="${GPU_ID:-0}"
SIMULATOR_TOOL_PORT="${SIMULATOR_TOOL_PORT:-30000}"
EVAL_SAVE_PATH="$root_dir/logs/scene"

SAMPLE_MODE="stratified_random"

# UE 窗口化设置。脚本会尽量用 UE 配置 + wmctrl 窗口管理两种方式处理。
UE_RES_X="${UE_RES_X:-1280}"
UE_RES_Y="${UE_RES_Y:-720}"
UE_WINDOW_ARGS="-windowed -ResX=${UE_RES_X} -ResY=${UE_RES_Y}"

# 每个 batch 结束后是否清理 UE/场景残留。
CLEANUP_UE_AFTER_BATCH="${CLEANUP_UE_AFTER_BATCH:-1}"

# 每个 batch 结束后是否清 Linux cache。
DROP_CACHE_AFTER_BATCH="${DROP_CACHE_AFTER_BATCH:-1}"

# batch 出错后是否继续下一批。
# 0：继续；1：停止。
STOP_ON_BATCH_ERROR="${STOP_ON_BATCH_ERROR:-0}"

# ============================================================

if [ ! -f "$DATASET_PATH" ]; then
    echo "[ERROR] dataset not found: $DATASET_PATH"
    exit 1
fi

mkdir -p "$root_dir/tmp_random_tasks"

DATASET_NAME="$(basename "$DATASET_PATH" .json)"
MASTER_JSON="$root_dir/tmp_random_tasks/${DATASET_NAME}_${SAMPLE_MODE}_${TOTAL_TASKS}_seed_${SEED}.json"
MASTER_MAP_JSON="$root_dir/tmp_random_tasks/${DATASET_NAME}_${SAMPLE_MODE}_${TOTAL_TASKS}_seed_${SEED}_mapping.json"
BATCH_DIR="$root_dir/tmp_random_tasks/${DATASET_NAME}_${SAMPLE_MODE}_${TOTAL_TASKS}_seed_${SEED}_batches"

mkdir -p "$BATCH_DIR"

echo "[BatchedEval] scene=${SCENE_NAME}"
echo "[BatchedEval] dataset=${DATASET_PATH}"
echo "[BatchedEval] total_tasks=${TOTAL_TASKS}, seed=${SEED}, tasks_per_batch=${TASKS_PER_BATCH}"
echo "[BatchedEval] max_actions=${MAX_ACTIONS}"
echo "[BatchedEval] UE window args=${UE_WINDOW_ARGS}"

force_ue_windowed_config() {
    echo "[WindowedUE] trying to force UE GameUserSettings.ini to windowed mode..."

    python - "$UE_RES_X" "$UE_RES_Y" <<'PY'
import os
import re
import sys
from pathlib import Path

res_x = sys.argv[1]
res_y = sys.argv[2]

search_roots = [
    Path.home(),
    Path("/home/tjn2004/uav"),
]

ini_files = []
for root in search_roots:
    if not root.exists():
        continue
    try:
        for path in root.rglob("GameUserSettings.ini"):
            p = str(path)
            if "/Saved/Config/" in p:
                ini_files.append(path)
    except Exception:
        pass

ini_files = sorted(set(ini_files))

if not ini_files:
    print("[WindowedUE] no existing GameUserSettings.ini found; window guard will still try wmctrl.")
    sys.exit(0)

def set_key(text, key, value):
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    line = f"{key}={value}"
    if pattern.search(text):
        return pattern.sub(line, text)
    return text.rstrip() + "\n" + line + "\n"

for path in ini_files:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "[/Script/Engine.GameUserSettings]" not in text:
            text = text.rstrip() + "\n\n[/Script/Engine.GameUserSettings]\n"

        text = set_key(text, "ResolutionSizeX", res_x)
        text = set_key(text, "ResolutionSizeY", res_y)
        text = set_key(text, "LastUserConfirmedResolutionSizeX", res_x)
        text = set_key(text, "LastUserConfirmedResolutionSizeY", res_y)
        text = set_key(text, "DesiredScreenWidth", res_x)
        text = set_key(text, "DesiredScreenHeight", res_y)
        text = set_key(text, "LastUserConfirmedDesiredScreenWidth", res_x)
        text = set_key(text, "LastUserConfirmedDesiredScreenHeight", res_y)

        # UE: 0=Fullscreen, 1=WindowedFullscreen, 2=Windowed
        text = set_key(text, "FullscreenMode", "2")
        text = set_key(text, "LastConfirmedFullscreenMode", "2")
        text = set_key(text, "PreferredFullscreenMode", "2")
        text = set_key(text, "bUseDesktopResolutionForFullscreen", "False")

        path.write_text(text, encoding="utf-8")
        print(f"[WindowedUE] patched {path}")
    except Exception as e:
        print(f"[WindowedUE] failed to patch {path}: {e}")
PY
}

WINDOW_GUARD_PID=""

start_window_guard() {
    if ! command -v wmctrl >/dev/null 2>&1; then
        echo "[WindowedUE] wmctrl not found; GameUserSettings.ini patch applied, but runtime resize guard is disabled."
        echo "[WindowedUE] To enable runtime window forcing: sudo apt install wmctrl"
        return
    fi

    echo "[WindowedUE] starting wmctrl window guard..."
    (
        while true; do
            # 目标窗口标题通常包含场景名，部分 UE 窗口可能包含 Unreal/AirSim。
            ids="$(wmctrl -l | grep -Ei "${SCENE_NAME}|Unreal|AirSim" | awk '{print $1}' || true)"
            for wid in $ids; do
                wmctrl -i -r "$wid" -b remove,fullscreen 2>/dev/null || true
                wmctrl -i -r "$wid" -e "0,40,40,${UE_RES_X},${UE_RES_Y}" 2>/dev/null || true
            done
            sleep 5
        done
    ) &
    WINDOW_GUARD_PID="$!"
}

stop_window_guard() {
    if [ -n "${WINDOW_GUARD_PID}" ]; then
        kill "${WINDOW_GUARD_PID}" 2>/dev/null || true
        WINDOW_GUARD_PID=""
    fi
}

cleanup_after_batch() {
    echo "[Cleanup] cleaning after batch..."

    # eval 进程正常应该已经退出；这里只清残留。
    pkill -f "src/eval_2.py" 2>/dev/null || true

    if [ "$CLEANUP_UE_AFTER_BATCH" = "1" ]; then
        # 只杀当前场景名相关的 UE/打包场景进程，尽量不动 simulator socket server。
        pkill -TERM -f "${SCENE_NAME}" 2>/dev/null || true
        sleep 3
        pkill -KILL -f "${SCENE_NAME}" 2>/dev/null || true
    fi

    if [ "$DROP_CACHE_AFTER_BATCH" = "1" ]; then
        sudo -n /usr/local/bin/uav_drop_caches.sh 2>/dev/null || {
            echo "[Cleanup] skip drop_caches because sudo permission is not available."
        }
    fi

    free -h || true
    nvidia-smi || true
}

generate_master_subset() {
    echo "[BatchedEval] generating fixed master subset..."

    python - "$DATASET_PATH" "$MASTER_JSON" "$MASTER_MAP_JSON" "$TOTAL_TASKS" "$SEED" "$SAMPLE_MODE" <<'PY'
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

src_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
map_path = Path(sys.argv[3])
num_tasks = int(sys.argv[4])
seed = int(sys.argv[5])
sample_mode = sys.argv[6]

with src_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise RuntimeError(
        "UAV-ON valset JSON should be a list of tasks. "
        f"Got type={type(data).__name__}."
    )

tasks = data
total = len(tasks)

if total == 0:
    raise RuntimeError("Dataset contains 0 tasks.")

if num_tasks <= 0:
    raise RuntimeError("TOTAL_TASKS must be positive.")

if num_tasks > total:
    print(f"[RandomTasks] requested {num_tasks}, but dataset only has {total}; use all tasks.")
    num_tasks = total

rng = random.Random(seed)

def stratified_sample(total_count, sample_count, rng_obj):
    if sample_count >= total_count:
        return list(range(total_count))

    selected = []
    for bucket_id in range(sample_count):
        start = int(bucket_id * total_count / sample_count)
        end = int((bucket_id + 1) * total_count / sample_count)

        start = max(0, min(start, total_count - 1))
        end = max(start + 1, min(end, total_count))

        candidates = list(range(start, end))
        selected.append(rng_obj.choice(candidates))

    selected = sorted(set(selected))

    while len(selected) < sample_count:
        candidate = rng_obj.randrange(total_count)
        if candidate not in selected:
            selected.append(candidate)

    return sorted(selected)

if sample_mode == "stratified_random":
    selected_indices = stratified_sample(total, num_tasks, rng)
else:
    selected_indices = sorted(rng.sample(range(total), num_tasks))

selected_tasks = []
mapping = []
seen_episode_ids = set()

for subset_episode_id, original_index in enumerate(selected_indices):
    task = deepcopy(tasks[original_index])

    if not isinstance(task, dict):
        raise RuntimeError(f"Task at original index {original_index + 1} is not a dict.")

    if "episode_id" not in task:
        raise RuntimeError(
            f"Task at original index {original_index + 1} has no episode_id. "
            "UAV-ON env_uav.py uses item['episode_id'] to create task_id."
        )

    original_episode_id = task["episode_id"]

    if original_episode_id in seen_episode_ids:
        raise RuntimeError(
            f"Duplicate episode_id found in selected tasks: {original_episode_id}. "
            "This would make save folders ambiguous."
        )

    seen_episode_ids.add(original_episode_id)

    # 不改 episode_id，保证 UAV-ON 保存目录 task_xxx 仍对应原始 episode_id。
    task["_random_subset_episode_id"] = subset_episode_id
    task["_original_episode_id"] = original_episode_id
    task["_original_task_index_0based"] = original_index
    task["_original_task_index_1based"] = original_index + 1
    task["_random_seed"] = seed
    task["_sample_mode"] = sample_mode
    task["_source_dataset"] = str(src_path)

    selected_tasks.append(task)

    mapping.append({
        "subset_episode_id": subset_episode_id,
        "original_task_index_0based": original_index,
        "original_task_index_1based": original_index + 1,
        "original_episode_id": original_episode_id,
        "expected_eval_task_id": original_episode_id,
        "expected_save_folder": f"task_{original_episode_id}",
    })

out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    json.dump(selected_tasks, f, ensure_ascii=False, indent=2)

with map_path.open("w", encoding="utf-8") as f:
    json.dump({
        "source_dataset": str(src_path),
        "random_subset_dataset": str(out_path),
        "num_source_tasks": total,
        "num_selected_tasks": len(selected_tasks),
        "seed": seed,
        "sample_mode": sample_mode,
        "mapping": mapping,
    }, f, ensure_ascii=False, indent=2)

print(f"[RandomTasks] source: {src_path}")
print(f"[RandomTasks] output: {out_path}")
print(f"[RandomTasks] mapping: {map_path}")
print(f"[RandomTasks] total tasks: {total}")
print(f"[RandomTasks] selected count: {len(selected_tasks)}")
print(f"[RandomTasks] seed: {seed}")
print(f"[RandomTasks] sample mode: {sample_mode}")
print("[RandomTasks] selected tasks:")
for item in mapping:
    print(
        "  subset_episode={subset_episode_id} -> "
        "original_index_1based={original_task_index_1based}, "
        "episode_id={original_episode_id}, "
        "save_folder={expected_save_folder}".format(**item)
    )
PY
}

create_batch_subset() {
    local start_1based="$1"
    local end_1based="$2"
    local batch_json="$3"
    local batch_map_json="$4"
    local batch_id="$5"

    python - "$MASTER_JSON" "$batch_json" "$batch_map_json" "$start_1based" "$end_1based" "$batch_id" <<'PY'
import json
import sys
from copy import deepcopy
from pathlib import Path

master_path = Path(sys.argv[1])
batch_path = Path(sys.argv[2])
batch_map_path = Path(sys.argv[3])
start_1based = int(sys.argv[4])
end_1based = int(sys.argv[5])
batch_id = int(sys.argv[6])

with master_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise RuntimeError("Master subset JSON should be a list.")

start = max(0, start_1based - 1)
end = min(len(data), end_1based)

selected = []
mapping = []

for local_id, task in enumerate(data[start:end]):
    task = deepcopy(task)

    master_subset_id = task.get("_random_subset_episode_id", start + local_id)
    original_episode_id = task.get("_original_episode_id", task.get("episode_id", ""))

    task["_batch_id"] = batch_id
    task["_batch_local_episode_id"] = local_id
    task["_batch_range_1based"] = [start_1based, end_1based]
    task["_master_subset_episode_id"] = master_subset_id

    selected.append(task)

    mapping.append({
        "batch_id": batch_id,
        "batch_local_episode_id": local_id,
        "master_subset_episode_id": master_subset_id,
        "original_episode_id": original_episode_id,
        "original_task_index_1based": task.get("_original_task_index_1based", ""),
        "expected_save_folder": f"task_{original_episode_id}",
    })

batch_path.parent.mkdir(parents=True, exist_ok=True)

with batch_path.open("w", encoding="utf-8") as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)

with batch_map_path.open("w", encoding="utf-8") as f:
    json.dump({
        "master_subset": str(master_path),
        "batch_subset": str(batch_path),
        "batch_id": batch_id,
        "batch_range_1based": [start_1based, end_1based],
        "num_selected_tasks": len(selected),
        "mapping": mapping,
    }, f, ensure_ascii=False, indent=2)

print(f"[BatchTasks] batch={batch_id}, range={start_1based}-{end_1based}, output={batch_path}")
for item in mapping:
    print(
        "  batch_local={batch_local_episode_id} -> "
        "master_subset={master_subset_episode_id}, "
        "episode_id={original_episode_id}, "
        "save_folder={expected_save_folder}".format(**item)
    )
PY
}

run_one_batch() {
    local batch_id="$1"
    local start_1based="$2"
    local end_1based="$3"

    local batch_json="$BATCH_DIR/${DATASET_NAME}_${SAMPLE_MODE}_${TOTAL_TASKS}_seed_${SEED}_batch_${batch_id}_${start_1based}_${end_1based}.json"
    local batch_map_json="$BATCH_DIR/${DATASET_NAME}_${SAMPLE_MODE}_${TOTAL_TASKS}_seed_${SEED}_batch_${batch_id}_${start_1based}_${end_1based}_mapping.json"

    create_batch_subset "$start_1based" "$end_1based" "$batch_json" "$batch_map_json" "$batch_id"

    echo ""
    echo "===================================================================================================="
    echo "[BatchedEval] Running batch ${batch_id}: tasks ${start_1based}-${end_1based}"
    echo "[BatchedEval] dataset=${batch_json}"
    echo "===================================================================================================="

    export SDL_VIDEO_FULLSCREEN=0
    export UAVON_UE_WINDOWED=1
    export UAVON_UE_WINDOW_ARGS="${UE_WINDOW_ARGS}"
    export UE_WINDOWED_ARGS="${UE_WINDOW_ARGS}"

    start_window_guard

    set +e
    CUDA_VISIBLE_DEVICES="$GPU_ID" python -u "$root_dir/src/eval_2.py" \
        --maxActions "$MAX_ACTIONS" \
        --eval_save_path "$EVAL_SAVE_PATH" \
        --dataset_path "$batch_json" \
        --is_fixed true \
        --gpu_id "$GPU_ID" \
        --batchSize "$BATCH_SIZE" \
        --simulator_tool_port "$SIMULATOR_TOOL_PORT"
    local status="$?"
    set -e

    stop_window_guard

    echo "[BatchedEval] batch ${batch_id} exit status=${status}"

    cleanup_after_batch

    if [ "$status" != "0" ] && [ "$STOP_ON_BATCH_ERROR" = "1" ]; then
        echo "[BatchedEval] batch failed and STOP_ON_BATCH_ERROR=1, stop."
        exit "$status"
    fi

    return 0
}

# 检查免密 drop_caches 是否可用；不要使用 sudo -v，避免中途要求密码。
if [ "$DROP_CACHE_AFTER_BATCH" = "1" ]; then
    sudo -n /usr/local/bin/uav_drop_caches.sh 2>/dev/null || {
        echo "[Cleanup] warning: /usr/local/bin/uav_drop_caches.sh cannot run without password."
        echo "[Cleanup] batch cleanup will skip drop_caches if permission is unavailable."
    }
fi

force_ue_windowed_config
generate_master_subset

batch_id=1
start=1

while [ "$start" -le "$TOTAL_TASKS" ]; do
    end=$((start + TASKS_PER_BATCH - 1))
    if [ "$end" -gt "$TOTAL_TASKS" ]; then
        end="$TOTAL_TASKS"
    fi

    run_one_batch "$batch_id" "$start" "$end"

    start=$((end + 1))
    batch_id=$((batch_id + 1))
done

echo ""
echo "===================================================================================================="
echo "[BatchedEval] all batches finished"
echo "[BatchedEval] master subset: $MASTER_JSON"
echo "[BatchedEval] batch dir:     $BATCH_DIR"
echo "[BatchedEval] logs saved to: $EVAL_SAVE_PATH"
echo "===================================================================================================="
