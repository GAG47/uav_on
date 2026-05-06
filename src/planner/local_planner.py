try:
    from src.planner.geometry_planner import OccupancyGeometryPlanner
except Exception:
    from planner.geometry_planner import OccupancyGeometryPlanner


class LocalPlanner:
    def __init__(
        self,
        memory,
        unknown_cost=6.0,
        safety_weight=3.0,
        semantic_weight=0.3,
        visited_weight=0.2,
        blocked_safety_threshold=0.18,
        blocked_confidence_threshold=0.05,
        free_confidence_threshold=0.03,
        free_safety_threshold=0.28,
        obstacle_confidence_threshold=0.45,
        obstacle_inflation_radius=3.0,
        risk_inflation_radius=5.0,
        risk_safety_threshold=0.45,
        risk_weight=4.0,
        block_unknown=True,
        max_path_points=30,
    ):
        self.memory = memory
        self.geometry_planner = OccupancyGeometryPlanner(
            memory=memory,
            unknown_cost=unknown_cost,
            safety_weight=safety_weight,
            semantic_weight=semantic_weight,
            visited_weight=visited_weight,
            blocked_safety_threshold=blocked_safety_threshold,
            blocked_confidence_threshold=blocked_confidence_threshold,
            free_confidence_threshold=free_confidence_threshold,
            free_safety_threshold=free_safety_threshold,
            obstacle_confidence_threshold=obstacle_confidence_threshold,
            obstacle_inflation_radius=obstacle_inflation_radius,
            risk_inflation_radius=risk_inflation_radius,
            risk_safety_threshold=risk_safety_threshold,
            risk_weight=risk_weight,
            block_unknown=block_unknown,
            max_path_points=max_path_points,
        )

    def plan_path(self, current_pose, memory_target):
        if memory_target is None or not memory_target.get("valid", False):
            return self.default_plan(reason="invalid memory target")

        target_position = self.get_target_position(memory_target)
        if target_position is None:
            return self.default_plan(reason="target position is None")

        target_type = memory_target.get("target_type", "")
        allow_arrived = (
            target_type in [
                "verified_target_stop",
                "gdino_stop",
                "gdino_verified_stop",
                "gdino_position_stop",
            ]
            or memory_target.get("stop_reason", "") != ""
        )

        geometry_result = self.geometry_planner.evaluate_path(
            current_pose=current_pose,
            target_position=target_position,
            z=current_pose[2],
            allow_goal_adjustment=True,
            allow_arrived=allow_arrived,
        )

        if not geometry_result.get("reachable", False):
            return self.default_plan_from_geometry(geometry_result)

        if (
            not allow_arrived
            and geometry_result.get("path_len", 0) <= 1
        ):
            geometry_result["reason"] = "non-stop path is too short for movement"
            geometry_result["reachable"] = False
            return self.default_plan_from_geometry(geometry_result)

        return {
            "valid": True,
            "reason": geometry_result.get("reason", "ok"),
            "arrived": geometry_result.get("arrived", False),
            "start_grid": geometry_result.get("start_grid", None),
            "goal_grid": geometry_result.get("goal_grid", None),
            "original_goal_grid": geometry_result.get("original_goal_grid", None),
            "goal_adjusted": geometry_result.get("goal_adjusted", False),
            "target_position": target_position,
            "path": geometry_result.get("path", []),
            "grid_path": geometry_result.get("grid_path", []),
            "raw_grid_path_len": geometry_result.get("raw_grid_path_len", 0),
            "path_len": geometry_result.get("path_len", 0),
            "path_length": geometry_result.get("path_length", 0.0),
            "straight_distance": geometry_result.get("straight_distance", 0.0),
            "path_to_straight_ratio": geometry_result.get("path_to_straight_ratio", 1.0),
            "geometry_cost": geometry_result.get("geometry_cost", 0.0),
            "geometry_score": geometry_result.get("geometry_score", 1.0),
            "blocked_cells": geometry_result.get("blocked_cells", 0),
            "obstacle_cells": geometry_result.get("obstacle_cells", 0),
            "unknown_blocked_cells": geometry_result.get("unknown_blocked_cells", 0),
            "inflation_cells": geometry_result.get("inflation_cells", 0),
        }

    def get_target_position(self, memory_target):
        if memory_target.get("viewpoint_position", None) is not None:
            return memory_target["viewpoint_position"]
        if memory_target.get("position", None) is not None:
            return memory_target["position"]
        if memory_target.get("frontier_position", None) is not None:
            return memory_target["frontier_position"]
        return None

    def default_plan_from_geometry(self, geometry_result):
        return {
            "valid": False,
            "reason": geometry_result.get("reason", "unknown"),
            "arrived": geometry_result.get("arrived", False),
            "start_grid": geometry_result.get("start_grid", None),
            "goal_grid": geometry_result.get("goal_grid", None),
            "original_goal_grid": geometry_result.get("original_goal_grid", None),
            "goal_adjusted": geometry_result.get("goal_adjusted", False),
            "target_position": geometry_result.get("target_position", None),
            "path": geometry_result.get("path", []),
            "grid_path": geometry_result.get("grid_path", []),
            "raw_grid_path_len": geometry_result.get("raw_grid_path_len", 0),
            "path_len": geometry_result.get("path_len", 0),
            "path_length": geometry_result.get("path_length", 0.0),
            "straight_distance": geometry_result.get("straight_distance", 0.0),
            "path_to_straight_ratio": geometry_result.get("path_to_straight_ratio", 1.0),
            "geometry_cost": geometry_result.get("geometry_cost", 1.0),
            "geometry_score": geometry_result.get("geometry_score", 0.0),
            "blocked_cells": geometry_result.get("blocked_cells", 0),
            "obstacle_cells": geometry_result.get("obstacle_cells", 0),
            "unknown_blocked_cells": geometry_result.get("unknown_blocked_cells", 0),
            "inflation_cells": geometry_result.get("inflation_cells", 0),
        }

    def default_plan(self, reason="unknown"):
        return {
            "valid": False,
            "reason": reason,
            "arrived": False,
            "start_grid": None,
            "goal_grid": None,
            "original_goal_grid": None,
            "goal_adjusted": False,
            "target_position": None,
            "path": [],
            "grid_path": [],
            "raw_grid_path_len": 0,
            "path_len": 0,
            "path_length": 0.0,
            "straight_distance": 0.0,
            "path_to_straight_ratio": 1.0,
            "geometry_cost": 1.0,
            "geometry_score": 0.0,
            "blocked_cells": 0,
            "obstacle_cells": 0,
            "unknown_blocked_cells": 0,
            "inflation_cells": 0,
        }