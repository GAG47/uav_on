#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_DATASET_PATH = "/home/tjn2004/uav/DATASET/UAV-ON-data/valset/CabinLake.json"
DEFAULT_EVAL_SAVE_PATH = "/home/tjn2004/uav/UAV_ON/logs/scene"


def parse_task_ids(task_string):
    task_ids = []
    for item in task_string.split(","):
        item = item.strip()
        if item == "":
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            start = int(start.strip())
            end = int(end.strip())
            if end < start:
                raise ValueError(f"invalid task range: {item}")
            task_ids.extend(list(range(start, end + 1)))
        else:
            task_ids.append(int(item))

    task_ids = list(dict.fromkeys(task_ids))
    if len(task_ids) == 0:
        raise ValueError("no task id is provided")
    return task_ids


def get_item_task_id(item, fallback_index):
    for key in ["task_id", "episode_id", "id"]:
        if key in item:
            try:
                return int(item[key])
            except Exception:
                pass
    return int(fallback_index)


def load_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"dataset should be a list: {dataset_path}")
    return data


def build_subset(data, task_ids):
    task_id_set = set(task_ids)
    subset = []
    selected_ids = []

    for index, item in enumerate(data):
        task_id = get_item_task_id(item, index)
        if task_id in task_id_set:
            subset.append(item)
            selected_ids.append(task_id)

    missing_ids = [task_id for task_id in task_ids if task_id not in selected_ids]
    if len(missing_ids) > 0:
        raise ValueError(
            "these task ids are not found in dataset: "
            + ",".join(str(item) for item in missing_ids)
        )

    return subset, selected_ids


def write_subset_dataset(subset, output_path):
    with open(output_path, "w") as f:
        json.dump(subset, f, indent=2)


def build_eval_command(args, subset_path):
    command = [
        sys.executable,
        "-u",
        "src/eval_2.py",
        "--maxActions",
        str(args.max_actions),
        "--eval_save_path",
        args.eval_save_path,
        "--dataset_path",
        str(subset_path),
        "--is_fixed",
        str(args.is_fixed).lower(),
        "--gpu_id",
        str(args.gpu_id),
        "--batchSize",
        str(args.batch_size),
        "--simulator_tool_port",
        str(args.simulator_tool_port),
    ]

    if args.name is not None and args.name != "":
        command.extend(["--name", args.name])

    return command


def main():
    parser = argparse.ArgumentParser(
        description="Run selected UAV-ON tasks with eval_2.py."
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Task ids to run, for example: 1,3,4 or 1-4 or 1,3,7-9.",
    )
    parser.add_argument(
        "--dataset_path",
        default=DEFAULT_DATASET_PATH,
        help="Original dataset json path.",
    )
    parser.add_argument(
        "--eval_save_path",
        default=DEFAULT_EVAL_SAVE_PATH,
        help="Evaluation output directory.",
    )
    parser.add_argument(
        "--maxActions",
        "--max_actions",
        dest="max_actions",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--is_fixed",
        default="true",
        choices=["true", "false", "True", "False"],
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--batchSize",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--simulator_tool_port",
        type=int,
        default=30000,
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default="0",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional eval name passed to eval_2.py.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print selected tasks and command.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    task_ids = parse_task_ids(args.tasks)
    dataset_path = Path(args.dataset_path)

    data = load_dataset(dataset_path)
    subset, selected_ids = build_subset(data, task_ids)

    print(f"[EvalTasks] dataset: {dataset_path}")
    print(f"[EvalTasks] selected task ids: {selected_ids}")
    print(f"[EvalTasks] selected count: {len(subset)}")

    with tempfile.TemporaryDirectory(prefix="uav_on_eval_tasks_") as tmp_dir:
        subset_path = Path(tmp_dir) / (
            dataset_path.stem
            + "_tasks_"
            + "_".join(str(item) for item in selected_ids)
            + ".json"
        )
        write_subset_dataset(subset, subset_path)

        command = build_eval_command(args, subset_path)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

        print("[EvalTasks] command:")
        print(" ".join(command))

        if args.dry_run:
            return

        result = subprocess.run(
            command,
            cwd=str(repo_root),
            env=env,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()