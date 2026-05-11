#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def to_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def to_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    bad_lines = 0

    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad_lines += 1

    rows.sort(key=lambda item: to_int(item.get("frame"), 0))
    return rows, bad_lines


def get_nested_position(record):
    try:
        pos = record.get("sensors", {}).get("state", {}).get("position", None)
        if isinstance(pos, list) and len(pos) >= 3:
            return [float(pos[0]), float(pos[1]), float(pos[2])]
    except Exception:
        pass
    return None


def get_target_position(obj_info):
    pose = obj_info.get("pose", None)
    if isinstance(pose, list) and len(pose) > 0:
        first = pose[0]
        if isinstance(first, list) and len(first) >= 3:
            return [float(first[0]), float(first[1]), float(first[2])]
    return None


def euclidean_distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    if len(p1) < 3 or len(p2) < 3:
        return None
    return math.sqrt(
        (float(p1[0]) - float(p2[0])) ** 2
        + (float(p1[1]) - float(p2[1])) ** 2
        + (float(p1[2]) - float(p2[2])) ** 2
    )


def get_distance_to_end(record, target_position=None):
    value = to_float(record.get("distance_to_end"), None)
    if value is not None:
        return value

    pos = get_nested_position(record)
    return euclidean_distance(pos, target_position)


def get_path_length(records):
    if not records:
        return 0.0

    final_move_distance = to_float(records[-1].get("move_distance"), None)
    if final_move_distance is not None:
        return final_move_distance

    length = 0.0
    prev = None

    for record in records:
        pos = get_nested_position(record)
        if pos is None:
            continue
        if prev is not None:
            step_dist = euclidean_distance(prev, pos)
            if step_dist is not None:
                length += step_dist
        prev = pos

    return length


def get_action_count(records):
    return sum(1 for item in records if item.get("action") is not None)


def get_stop_info(records):
    stop_records = []
    for item in records:
        action = item.get("action")
        if isinstance(action, str) and action.lower() == "stop":
            stop_records.append(item)

    if not stop_records:
        return False, None

    return True, to_int(stop_records[-1].get("frame"), None)


def get_size_group(size_text):
    text = str(size_text or "").strip().lower()
    if text.startswith("small"):
        return "small"
    if text.startswith("mid") or text.startswith("medium"):
        return "medium"
    if text.startswith("large"):
        return "large"
    return "unknown"


def count_prompt_steps(prompt_path):
    path = Path(prompt_path)
    if not path.exists():
        return 0
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0
    return len(re.findall(r"\[STEP\s+\d+\s+PROMPT\]", text))


def find_trajectory_path(task_dir):
    preferred = Path(task_dir) / "log" / "trajectory.jsonl"
    if preferred.exists():
        return preferred

    matches = sorted(Path(task_dir).rglob("trajectory.jsonl"))
    if matches:
        return matches[0]

    return None


def infer_folder_label(task_dir):
    """
    根据 task_xx 所在的上级目录推断分类标签。
    比如：
      success_CabinLake.../task_2 -> success
      oracle_CabinLake.../task_2  -> oracle
      failure_CabinLake.../task_2 -> failure
    """
    parent = Path(task_dir).parent.name.lower()

    if parent.startswith("success"):
        return "success"
    if parent.startswith("oracle"):
        return "oracle"
    if parent.startswith("failure") or parent.startswith("fail"):
        return "failure"
    if parent.startswith("collision"):
        return "collision"
    if parent.startswith("max"):
        return "max_step"

    return parent


def analyze_task(task_dir, success_threshold, max_actions, allow_collision_success=False):
    task_dir = Path(task_dir)
    obj_path = task_dir / "object_description.json"
    prompt_path = task_dir / "prompt_info.txt"
    traj_path = find_trajectory_path(task_dir)

    if not obj_path.exists() or traj_path is None:
        return None

    obj_info = read_json(obj_path)
    records, bad_lines = read_jsonl(traj_path)

    if not records:
        return {
            "task_dir": str(task_dir),
            "valid": False,
            "invalid_reason": "empty trajectory",
        }

    target_position = get_target_position(obj_info)
    distances = []

    for record in records:
        distance = get_distance_to_end(record, target_position=target_position)
        if distance is not None:
            distances.append((to_int(record.get("frame"), 0), float(distance), record))

    if not distances:
        return {
            "task_dir": str(task_dir),
            "valid": False,
            "invalid_reason": "missing distance_to_end",
        }

    final_frame, final_distance, final_record = distances[-1]
    min_frame, min_distance, min_record = min(distances, key=lambda item: item[1])

    geodesic_distance = to_float(obj_info.get("info", {}).get("geodesic_distance"), None)
    euclidean_start_distance = to_float(obj_info.get("info", {}).get("euclidean_distance"), None)

    if geodesic_distance is None:
        geodesic_distance = euclidean_start_distance

    if geodesic_distance is None:
        start_pos = obj_info.get("start_pose", {}).get("start_position")
        geodesic_distance = euclidean_distance(start_pos, target_position)

    if geodesic_distance is None or geodesic_distance <= 0:
        geodesic_distance = 0.0

    path_length = get_path_length(records)
    action_count = get_action_count(records)
    collision = any(bool(item.get("is_collision", False)) for item in records)
    stopped, stop_step = get_stop_info(records)

    success_by_distance = final_distance <= success_threshold
    success = success_by_distance if allow_collision_success else (success_by_distance and not collision)
    oracle_success = min_distance <= success_threshold

    wrong_stop = bool(stopped and final_distance > success_threshold)
    max_step = action_count >= max_actions

    if collision:
        termination_type = "collision"
    elif stopped:
        termination_type = "stop"
    elif success_by_distance:
        termination_type = "success_distance"
    elif max_step:
        termination_type = "max_step"
    else:
        termination_type = "ended"

    if success:
        failure_type = "success"
    elif collision:
        failure_type = "collision"
    elif wrong_stop:
        failure_type = "wrong_stop"
    elif max_step:
        failure_type = "max_step"
    elif oracle_success:
        failure_type = "oracle_success_only"
    else:
        failure_type = "not_found"

    if success and geodesic_distance > 0:
        spl = geodesic_distance / max(path_length, geodesic_distance)
    else:
        spl = 0.0

    oracle_path_length = None
    if oracle_success:
        oracle_path_length = to_float(min_record.get("move_distance"), None)
        if oracle_path_length is None:
            oracle_path_length = path_length

    if oracle_success and geodesic_distance > 0 and oracle_path_length is not None:
        oracle_spl = geodesic_distance / max(oracle_path_length, geodesic_distance)
    else:
        oracle_spl = 0.0

    object_name = str(obj_info.get("true_name") or obj_info.get("object_name") or "").strip()
    if object_name == "":
        object_name = "unknown"

    map_name = str(obj_info.get("map_name") or "unknown")
    source_dataset = str(obj_info.get("_source_dataset") or "")

    return {
        "valid": True,
        "folder_label": infer_folder_label(task_dir),
        "task_dir": str(task_dir),
        "trajectory_path": str(traj_path),
        "prompt_path": str(prompt_path) if prompt_path.exists() else "",
        "episode_id": str(obj_info.get("episode_id", "")),
        "original_episode_id": str(obj_info.get("_original_episode_id", obj_info.get("episode_id", ""))),
        "original_task_index_1based": obj_info.get("_original_task_index_1based", ""),
        "random_subset_episode_id": obj_info.get("_random_subset_episode_id", ""),
        "random_seed": obj_info.get("_random_seed", ""),
        "sample_mode": obj_info.get("_sample_mode", ""),
        "source_dataset": source_dataset,
        "map_name": map_name,
        "scene": map_name.replace("_test", ""),
        "object_name": object_name,
        "object_size": str(obj_info.get("size", "")).strip(),
        "size_group": get_size_group(obj_info.get("size", "")),
        "geodesic_distance": round(geodesic_distance, 4),
        "start_euclidean_distance": round(euclidean_start_distance, 4) if euclidean_start_distance is not None else "",
        "final_distance": round(final_distance, 4),
        "min_distance": round(min_distance, 4),
        "path_length": round(path_length, 4),
        "oracle_path_length": round(oracle_path_length, 4) if oracle_path_length is not None else "",
        "action_count": action_count,
        "final_frame": final_frame,
        "first_success_frame": min_frame if oracle_success else "",
        "first_success_step": min_frame if oracle_success else "",
        "stopped": int(stopped),
        "stop_step": stop_step if stop_step is not None else "",
        "collision": int(collision),
        "max_step": int(max_step),
        "wrong_stop": int(wrong_stop),
        "success": int(success),
        "success_by_distance": int(success_by_distance),
        "oracle_success": int(oracle_success),
        "spl": round(spl, 6),
        "oracle_spl": round(oracle_spl, 6),
        "dts": round(final_distance, 4),
        "termination_type": termination_type,
        "failure_type": failure_type,
        "prompt_step_count": count_prompt_steps(prompt_path),
        "bad_trajectory_lines": bad_lines,
    }


def mean(values):
    values = [v for v in values if v is not None and v != ""]
    if not values:
        return None
    return sum(float(v) for v in values) / len(values)


def rate(rows, key):
    if not rows:
        return 0.0
    return sum(int(row.get(key, 0)) for row in rows) / len(rows)


def summarize(rows):
    rows = [row for row in rows if row.get("valid")]

    if not rows:
        return {"num_tasks": 0}

    first_success_steps = [row["first_success_step"] for row in rows if row.get("first_success_step") != ""]
    stop_steps = [row["stop_step"] for row in rows if row.get("stop_step") != ""]

    termination_counter = Counter(row["termination_type"] for row in rows)
    failure_counter = Counter(row["failure_type"] for row in rows)
    folder_counter = Counter(row["folder_label"] for row in rows)

    return {
        "num_tasks": len(rows),
        "SR": round(rate(rows, "success"), 6),
        "OSR": round(rate(rows, "oracle_success"), 6),
        "SPL": round(mean([row["spl"] for row in rows]) or 0.0, 6),
        "Oracle_SPL": round(mean([row["oracle_spl"] for row in rows]) or 0.0, 6),
        "DTS": round(mean([row["dts"] for row in rows]) or 0.0, 4),
        "Min_Distance": round(mean([row["min_distance"] for row in rows]) or 0.0, 4),
        "Avg_Path_Length": round(mean([row["path_length"] for row in rows]) or 0.0, 4),
        "Avg_Steps": round(mean([row["action_count"] for row in rows]) or 0.0, 4),
        "Collision_Rate": round(rate(rows, "collision"), 6),
        "MaxStep_Rate": round(rate(rows, "max_step"), 6),
        "Stop_Rate": round(rate(rows, "stopped"), 6),
        "WrongStop_Rate": round(rate(rows, "wrong_stop"), 6),
        "Success_Count": sum(int(row["success"]) for row in rows),
        "Oracle_Success_Count": sum(int(row["oracle_success"]) for row in rows),
        "Collision_Count": sum(int(row["collision"]) for row in rows),
        "MaxStep_Count": sum(int(row["max_step"]) for row in rows),
        "WrongStop_Count": sum(int(row["wrong_stop"]) for row in rows),
        "Avg_First_Success_Step_Only_OSR": round(mean(first_success_steps), 4) if first_success_steps else "",
        "Avg_Stop_Step_Only_Stop": round(mean(stop_steps), 4) if stop_steps else "",
        "Termination_Type_Count": dict(termination_counter),
        "Failure_Type_Count": dict(failure_counter),
        "Folder_Label_Count": dict(folder_counter),
    }


def group_by(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        if not row.get("valid"):
            continue
        value = str(row.get(key, "") or "unknown")
        grouped[value].append(row)

    return {
        value: summarize(group_rows)
        for value, group_rows in sorted(grouped.items(), key=lambda item: item[0])
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_group_csv(path, grouped):
    rows = []
    for group_name, summary in grouped.items():
        row = {"group": group_name}
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False)
            else:
                row[key] = value
        rows.append(row)
    write_csv(path, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/tjn2004/uav/UAV_ON/logs/scene",
        help="要分析的目录。可以是整个 logs/scene，也可以是某一次运行目录，也可以是某个场景目录。",
    )
    parser.add_argument(
        "--out",
        default="",
        help="输出目录。默认输出到 <root>/eval_metrics/<timestamp>。",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=20.0,
        help="UAV-ON 成功距离阈值，默认 20。",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=150,
        help="最大动作步数，默认 150。",
    )
    parser.add_argument(
        "--allow-collision-success",
        action="store_true",
        help="默认 collision 不算 success；加这个参数后，只要 final_distance<=20 就算 success。",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"[ERROR] root does not exist: {root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
    else:
        out_dir = root / "eval_metrics" / timestamp

    object_files = sorted(root.rglob("object_description.json"))
    task_dirs = [path.parent for path in object_files]

    print(f"[EvalMetrics] root={root}")
    print(f"[EvalMetrics] found task candidates={len(task_dirs)}")

    rows = []
    invalid_rows = []

    seen_episode_ids = defaultdict(list)

    for task_dir in task_dirs:
        row = analyze_task(
            task_dir=task_dir,
            success_threshold=args.success_threshold,
            max_actions=args.max_actions,
            allow_collision_success=args.allow_collision_success,
        )

        if row is None:
            continue

        if row.get("valid"):
            rows.append(row)
            key = row.get("original_episode_id") or row.get("episode_id") or row.get("task_dir")
            seen_episode_ids[str(key)].append(row.get("task_dir"))
        else:
            invalid_rows.append(row)

    duplicate_episode_ids = {
        key: value for key, value in seen_episode_ids.items()
        if len(value) > 1
    }

    overall = summarize(rows)
    by_folder_label = group_by(rows, "folder_label")
    by_scene = group_by(rows, "scene")
    by_map = group_by(rows, "map_name")
    by_size = group_by(rows, "size_group")
    by_object = group_by(rows, "object_name")
    by_source_dataset = group_by(rows, "source_dataset")

    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "per_task_metrics.csv", rows)
    write_csv(out_dir / "invalid_tasks.csv", invalid_rows)
    write_group_csv(out_dir / "by_folder_label_metrics.csv", by_folder_label)
    write_group_csv(out_dir / "by_scene_metrics.csv", by_scene)
    write_group_csv(out_dir / "by_map_metrics.csv", by_map)
    write_group_csv(out_dir / "by_size_metrics.csv", by_size)
    write_group_csv(out_dir / "by_object_metrics.csv", by_object)
    write_group_csv(out_dir / "by_source_dataset_metrics.csv", by_source_dataset)

    summary = {
        "generated_at": timestamp,
        "root": str(root),
        "output_dir": str(out_dir),
        "config": {
            "success_threshold": args.success_threshold,
            "max_actions": args.max_actions,
            "allow_collision_success": bool(args.allow_collision_success),
        },
        "overall": overall,
        "by_folder_label": by_folder_label,
        "by_scene": by_scene,
        "by_map": by_map,
        "by_size": by_size,
        "by_object": by_object,
        "by_source_dataset": by_source_dataset,
        "invalid_task_count": len(invalid_rows),
        "duplicate_episode_ids": duplicate_episode_ids,
    }

    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("[EvalMetrics] done")
    print(f"[EvalMetrics] output_dir={out_dir}")

    if duplicate_episode_ids:
        print()
        print("[Warning] duplicate episode ids found:")
        for key, paths in duplicate_episode_ids.items():
            print(f"  episode={key}")
            for p in paths:
                print(f"    {p}")

    print()
    print("[Overall]")
    print(f"  num_tasks         = {overall.get('num_tasks', 0)}")
    print(f"  SR                = {overall.get('SR', 0.0):.4f}")
    print(f"  OSR               = {overall.get('OSR', 0.0):.4f}")
    print(f"  SPL               = {overall.get('SPL', 0.0):.4f}")
    print(f"  Oracle_SPL        = {overall.get('Oracle_SPL', 0.0):.4f}")
    print(f"  DTS               = {overall.get('DTS', 0.0):.4f}")
    print(f"  Avg_Steps         = {overall.get('Avg_Steps', 0.0):.4f}")
    print(f"  Avg_Path_Length   = {overall.get('Avg_Path_Length', 0.0):.4f}")
    print(f"  Stop_Rate         = {overall.get('Stop_Rate', 0.0):.4f}")
    print(f"  Collision_Rate    = {overall.get('Collision_Rate', 0.0):.4f}")
    print(f"  MaxStep_Rate      = {overall.get('MaxStep_Rate', 0.0):.4f}")
    print(f"  WrongStop_Rate    = {overall.get('WrongStop_Rate', 0.0):.4f}")
    print(f"  Folder_Label      = {overall.get('Folder_Label_Count', {})}")
    print(f"  Failure_Type      = {overall.get('Failure_Type_Count', {})}")

    print()
    print("[Files]")
    print(f"  {out_dir / 'metrics_summary.json'}")
    print(f"  {out_dir / 'per_task_metrics.csv'}")
    print(f"  {out_dir / 'by_folder_label_metrics.csv'}")
    print(f"  {out_dir / 'by_scene_metrics.csv'}")


if __name__ == "__main__":
    main()
