import json
import os
import time
from pathlib import Path


class AirHuntMetricsLogger:
    """
    Structured JSONL logger for AirHunt-style planner-driven navigation.

    This logger is observation-only. It must not affect target selection,
    viewpoint generation, local planning, path following, or stop decisions.
    """

    def __init__(self, base_dir=None, enabled=None):
        if enabled is None:
            enabled = os.environ.get("AIRHUNT_ENABLE_TRACE", "1") != "0"

        self.enabled = bool(enabled)
        self.file_path = None

        if not self.enabled:
            return

        if base_dir is None:
            base_dir = self.default_base_dir()

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.file_path = self.base_dir / f"airhunt_trace_{timestamp}.jsonl"

    def default_base_dir(self):
        try:
            from src.common.param import args
            eval_save_path = getattr(args, "eval_save_path", None)
            name = getattr(args, "name", None)

            if eval_save_path is not None and name is not None:
                return Path(eval_save_path) / name / "airhunt_eval_logs"
        except Exception:
            pass

        try:
            from common.param import args
            eval_save_path = getattr(args, "eval_save_path", None)
            name = getattr(args, "name", None)

            if eval_save_path is not None and name is not None:
                return Path(eval_save_path) / name / "airhunt_eval_logs"
        except Exception:
            pass

        return Path("logs") / "airhunt_eval_logs"

    def log_step(self, record):
        if not self.enabled:
            return

        if self.file_path is None:
            return

        record = self.make_json_safe(record)

        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def make_step_record(
        self,
        episode_index,
        step_num,
        current_pose,
        fixed,
        semantic_result,
        memory_summary,
        grounding_dino_result,
        tracker_info,
        verification_info,
        navigation_info,
        selected_target,
        executable_viewpoint,
        planned_path,
        path_follower_info,
        planner_feedback,
        stop_gate_info,
        action,
        value,
        done,
    ):
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "episode_index": episode_index,
            "step_num": step_num,
            "fixed": bool(fixed),
            "current_pose": current_pose,
            "action": action,
            "step_size": value,
            "done": bool(done),
            "semantic": self.summarize_semantic_result(semantic_result),
            "memory": self.summarize_memory(memory_summary),
            "grounding_dino": self.summarize_grounding_dino(grounding_dino_result),
            "tracker": self.summarize_tracker(tracker_info),
            "verification": self.summarize_verification(verification_info),
            "navigation": self.summarize_navigation(navigation_info),
            "target": self.summarize_target(selected_target),
            "viewpoint": self.summarize_viewpoint(executable_viewpoint),
            "path": self.summarize_path(planned_path),
            "path_follower": self.summarize_path_follower(path_follower_info),
            "planner_feedback": self.summarize_planner_feedback(planner_feedback),
            "stop_gate": self.summarize_stop_gate(stop_gate_info),
        }

        record["derived"] = self.derive_step_status(record)
        return record

    def summarize_semantic_result(self, semantic_result):
        if not isinstance(semantic_result, dict):
            return {}

        return {
            "best_region": semantic_result.get("best_region", ""),
            "target_visible": semantic_result.get("target_visible", False),
            "target_confidence": semantic_result.get("target_confidence", 0.0),
            "region_scores": semantic_result.get("region_scores", {}),
            "safety_scores": semantic_result.get("safety_scores", {}),
            "novelty_scores": semantic_result.get("novelty_scores", {}),
            "reason": semantic_result.get("reason", ""),
            "raw_text": semantic_result.get("raw_text", ""),
            "parse_error": semantic_result.get("parse_error", ""),
        }

    def summarize_memory(self, memory_summary):
        if not isinstance(memory_summary, dict):
            return {}

        keys = [
            "updated",
            "best_region",
            "best_score",
            "num_updated_cells",
            "max_semantic_value",
            "mean_confidence",
            "reason",
        ]
        return {key: memory_summary.get(key, None) for key in keys if key in memory_summary}

    def summarize_grounding_dino(self, grounding_dino_result):
        if not isinstance(grounding_dino_result, dict):
            return {}

        best_detection = grounding_dino_result.get("best_detection", None)
        if not isinstance(best_detection, dict):
            best_detection_summary = None
        else:
            best_detection_summary = {
                "score": best_detection.get("score", best_detection.get("confidence", None)),
                "label": best_detection.get("label", best_detection.get("phrase", "")),
                "image_index": best_detection.get("image_index", best_detection.get("image_id", None)),
                "bbox": best_detection.get("bbox", None),
                "center": best_detection.get("center", None),
                "target_world_position": best_detection.get("target_world_position", None),
            }

        return {
            "available": grounding_dino_result.get("available", False),
            "num_detections": grounding_dino_result.get("num_detections", 0),
            "best_score": grounding_dino_result.get("best_score", 0.0),
            "best_detection": best_detection_summary,
            "error": grounding_dino_result.get("error", ""),
        }

    def summarize_tracker(self, tracker_info):
        if not isinstance(tracker_info, dict):
            return {}

        keys = [
            "candidate",
            "confirmed",
            "verified",
            "verification_required",
            "stop_ready",
            "lost_count",
            "confirm_count",
            "required_count",
            "navigate_count",
            "candidate_id",
            "verified_candidate_id",
            "verified_target_position",
            "planner_target",
            "reason",
            "reject_reason",
        ]
        return {key: tracker_info.get(key, None) for key in keys if key in tracker_info}

    def summarize_verification(self, verification_info):
        if not isinstance(verification_info, dict):
            return {}

        keys = [
            "checked",
            "verified",
            "selected",
            "selected_candidate_id",
            "selected_index",
            "reason",
            "raw_response",
            "num_candidates",
            "error",
        ]
        return {key: verification_info.get(key, None) for key in keys if key in verification_info}

    def summarize_navigation(self, navigation_info):
        if not isinstance(navigation_info, dict):
            return {}

        target_evidence = navigation_info.get("target_evidence", {})
        planner_policy = navigation_info.get("planner_policy", {})

        return {
            "mode": navigation_info.get("mode", ""),
            "prev_mode": navigation_info.get("prev_mode", ""),
            "changed": navigation_info.get("changed", False),
            "reason": navigation_info.get("reason", ""),
            "target_evidence": target_evidence if isinstance(target_evidence, dict) else {},
            "planner_policy": planner_policy if isinstance(planner_policy, dict) else {},
        }

    def summarize_target(self, target):
        if not isinstance(target, dict):
            return {}

        keys = [
            "valid",
            "target_id",
            "target_type",
            "source",
            "position",
            "target_world_position",
            "verified_target_position",
            "anchor_position",
            "viewpoint_position",
            "score",
            "semantic_value",
            "confidence",
            "reason",
            "stop_reason",
            "stop_gate_info",
        ]
        return {key: target.get(key, None) for key in keys if key in target}

    def summarize_viewpoint(self, viewpoint):
        if not isinstance(viewpoint, dict):
            return {}

        executable = viewpoint.get("executable_viewpoint", None)
        if isinstance(executable, dict):
            executable_summary = {
                "valid": executable.get("valid", None),
                "viewpoint_id": executable.get("viewpoint_id", ""),
                "viewpoint_type": executable.get("viewpoint_type", ""),
                "position": executable.get("position", None),
                "yaw": executable.get("yaw", None),
                "anchor_position": executable.get("anchor_position", None),
                "safety_score": executable.get("safety_score", None),
                "information_gain": executable.get("information_gain", None),
                "reason": executable.get("reason", ""),
            }
        else:
            executable_summary = None

        return {
            "valid": viewpoint.get("valid", False),
            "viewpoint_id": viewpoint.get("viewpoint_id", ""),
            "viewpoint_type": viewpoint.get("viewpoint_type", ""),
            "viewpoint_position": viewpoint.get("viewpoint_position", viewpoint.get("position", None)),
            "viewpoint_yaw": viewpoint.get("viewpoint_yaw", viewpoint.get("target_yaw", None)),
            "anchor_position": viewpoint.get("anchor_position", None),
            "safety_score": viewpoint.get("safety_score", None),
            "information_gain": viewpoint.get("information_gain", None),
            "reason": viewpoint.get("viewpoint_reason", viewpoint.get("reason", "")),
            "executable_viewpoint": executable_summary,
        }

    def summarize_path(self, planned_path):
        if not isinstance(planned_path, dict):
            return {}

        return {
            "valid": planned_path.get("valid", False),
            "reason": planned_path.get("reason", ""),
            "path_len": planned_path.get("path_len", 0),
            "raw_grid_path_len": planned_path.get("raw_grid_path_len", 0),
            "path_length": planned_path.get("path_length", 0.0),
            "next_waypoint": planned_path.get("next_waypoint", None),
            "target_position": planned_path.get("target_position", None),
            "start_grid": planned_path.get("start_grid", None),
            "goal_grid": planned_path.get("goal_grid", None),
            "viewpoint_type": planned_path.get("viewpoint_type", ""),
            "viewpoint_position": planned_path.get("viewpoint_position", None),
            "debug": planned_path.get("debug", {}),
        }

    def summarize_path_follower(self, path_follower_info):
        if not isinstance(path_follower_info, dict):
            return {}

        return {
            "valid": path_follower_info.get("valid", False),
            "action": path_follower_info.get("action", None),
            "step_size": path_follower_info.get("step_size", 0.0),
            "done": path_follower_info.get("done", False),
            "reason": path_follower_info.get("reason", ""),
            "action_source": path_follower_info.get("action_source", ""),
            "next_waypoint": path_follower_info.get("next_waypoint", None),
            "distance_to_waypoint": path_follower_info.get("distance_to_waypoint", None),
            "horizontal_distance_to_waypoint": path_follower_info.get("horizontal_distance_to_waypoint", None),
            "yaw_error": path_follower_info.get("yaw_error", None),
            "path_reason": path_follower_info.get("path_reason", ""),
            "path_len": path_follower_info.get("path_len", 0),
            "path_length": path_follower_info.get("path_length", 0.0),
        }

    def summarize_planner_feedback(self, planner_feedback):
        if not isinstance(planner_feedback, dict):
            return {}

        return {
            "valid": planner_feedback.get("valid", False),
            "reason": planner_feedback.get("reason", ""),
            "target_type": planner_feedback.get("target_type", ""),
            "viewpoint_type": planner_feedback.get("viewpoint_type", ""),
            "path_len": planner_feedback.get("path_len", 0),
            "path_length": planner_feedback.get("path_length", 0.0),
            "action_source": planner_feedback.get("action_source", ""),
            "replan_required": planner_feedback.get("replan_required", False),
            "should_reselect_viewpoint": planner_feedback.get("should_reselect_viewpoint", False),
            "should_update_target_priority": planner_feedback.get("should_update_target_priority", False),
            "path_follower_valid": planner_feedback.get("path_follower_valid", None),
            "path_follower_reason": planner_feedback.get("path_follower_reason", ""),
        }

    def summarize_stop_gate(self, stop_gate_info):
        if not isinstance(stop_gate_info, dict):
            return {}

        return {
            "pass": stop_gate_info.get("pass", False),
            "reason": stop_gate_info.get("reason", ""),
            "mode": stop_gate_info.get("mode", ""),
            "distance": stop_gate_info.get("distance", None),
            "threshold": stop_gate_info.get("threshold", None),
            "target_position": stop_gate_info.get("target_position", None),
            "evidence_status": stop_gate_info.get("evidence_status", ""),
            "tracker_verified": stop_gate_info.get("tracker_verified", None),
            "tracker_confirmed": stop_gate_info.get("tracker_confirmed", None),
            "navigate_count": stop_gate_info.get("navigate_count", None),
        }

    def derive_step_status(self, record):
        action_source = record.get("path_follower", {}).get("action_source", "")
        action = record.get("action", None)
        path = record.get("path", {})
        stop_gate = record.get("stop_gate", {})
        navigation = record.get("navigation", {})
        tracker = record.get("tracker", {})
        verification = record.get("verification", {})
        target = record.get("target", {})
        viewpoint = record.get("viewpoint", {})

        status = {
            "action_source": action_source,
            "is_stop_action": action == "stop",
            "stop_gate_pass": bool(stop_gate.get("pass", False)),
            "path_valid": bool(path.get("valid", False)),
            "path_reason": path.get("reason", ""),
            "mode": navigation.get("mode", ""),
            "target_evidence_status": navigation.get("target_evidence", {}).get("status", ""),
            "target_type": target.get("target_type", ""),
            "viewpoint_type": viewpoint.get("viewpoint_type", ""),
            "verified": bool(tracker.get("verified", False)),
            "verification_checked": bool(verification.get("checked", False)),
            "verification_verified": bool(verification.get("verified", False)),
        }

        if action == "stop" and not stop_gate.get("pass", False):
            status["warning"] = "stop_without_stop_gate_pass"
        elif stop_gate.get("pass", False):
            status["event"] = "stop_gate_pass"
        elif not path.get("valid", False):
            status["event"] = "path_invalid"
        elif action_source == "path_follower":
            status["event"] = "path_following"
        else:
            status["event"] = "unknown"

        return status

    def make_json_safe(self, value):
        if isinstance(value, dict):
            return {
                str(k): self.make_json_safe(v)
                for k, v in value.items()
            }

        if isinstance(value, list):
            return [
                self.make_json_safe(v)
                for v in value
            ]

        if isinstance(value, tuple):
            return [
                self.make_json_safe(v)
                for v in value
            ]

        if isinstance(value, set):
            return [
                self.make_json_safe(v)
                for v in sorted(list(value))
            ]

        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)
