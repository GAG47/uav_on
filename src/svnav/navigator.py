from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from svnav.action_adapter import ActionAdapter, ActionAdapterConfig
from svnav.semantic_map import SemanticMap
from svnav.types import (
    ActionSource,
    NavDecision,
    NavMode,
    ObservationRecord,
    PoseRecord,
)
from svnav.viewpoint_planner import (
    SearchViewpoint,
    SearchViewpointPlanner,
    SearchViewpointPlannerConfig,
)


@dataclass
class SearchNavigatorConfig:
    max_active_viewpoint_steps: int = 12
    no_progress_limit: int = 4
    progress_epsilon: float = 0.5

    replan_score_margin: float = 0.15

    def __post_init__(self) -> None:
        self.max_active_viewpoint_steps = int(self.max_active_viewpoint_steps)
        self.no_progress_limit = int(self.no_progress_limit)
        self.progress_epsilon = float(self.progress_epsilon)
        self.replan_score_margin = float(self.replan_score_margin)


@dataclass
class ActiveViewpointState:
    viewpoint: SearchViewpoint
    start_step: int
    last_distance: Optional[float] = None
    last_progress_metric: Optional[float] = None
    no_progress_count: int = 0
    last_replan_reason: str = "new_viewpoint"

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint": self.viewpoint.to_log_dict(),
            "start_step": int(self.start_step),
            "last_distance": self.last_distance,
            "last_progress_metric": self.last_progress_metric,
            "no_progress_count": int(self.no_progress_count),
            "last_replan_reason": self.last_replan_reason,
        }


class SearchNavigator:
    """
    Search-stage navigator with viewpoint / waypoint tracking.

    It does not call VLM, GDINO, Task2, or StopGate.

    Planning loop:
        current SemanticMap
            -> select SearchRegion
            -> generate SearchViewpoint
            -> keep active_viewpoint until reached / invalid / preempted
            -> ActionAdapter follows active_viewpoint.position
    """

    def __init__(
        self,
        config: Optional[SearchNavigatorConfig] = None,
        viewpoint_planner: Optional[SearchViewpointPlanner] = None,
        viewpoint_planner_config: Optional[SearchViewpointPlannerConfig] = None,
        action_adapter: Optional[ActionAdapter] = None,
        action_adapter_config: Optional[ActionAdapterConfig] = None,
    ) -> None:
        self.config = config or SearchNavigatorConfig()
        self.viewpoint_planner = viewpoint_planner or SearchViewpointPlanner(
            viewpoint_planner_config
        )
        self.action_adapter = action_adapter or ActionAdapter(action_adapter_config)

        self.active_by_episode: Dict[str, ActiveViewpointState] = {}

    def decide_search(
        self,
        episode_id: str,
        step_id: int,
        observation: ObservationRecord,
        semantic_map: SemanticMap,
    ) -> NavDecision:
        current_pose = observation.pose
        episode_id = str(episode_id)

        if not semantic_map.is_pose_in_bounds(current_pose):
            target_position = semantic_map.nearest_in_bounds_position(current_pose)
            self.active_by_episode.pop(episode_id, None)

            return self.action_adapter.target_to_action(
                episode_id=episode_id,
                step_id=step_id,
                current_pose=current_pose,
                target_position=target_position,
                observation=observation,
                semantic_map=semantic_map,
                mode=NavMode.SEARCH,
                target_type="in_bounds_region",
                action_source=ActionSource.GEOMETRIC_EXPLORE,
                reason="uav outside semantic map; return to nearest in-bounds position",
                debug_info={
                    "navigator_reason": "out_of_bounds",
                    "target_position": list(target_position),
                },
            )

        active = self.active_by_episode.get(episode_id)
        replan_reason = self._need_replan(
            active=active,
            current_pose=current_pose,
            semantic_map=semantic_map,
            step_id=step_id,
        )

        created_new = False

        if active is None or replan_reason is not None:
            viewpoint = self.viewpoint_planner.plan_best_viewpoint(
                current_pose=current_pose,
                semantic_map=semantic_map,
                current_step=step_id,
            )

            if viewpoint is None:
                self.active_by_episode.pop(episode_id, None)
                return NavDecision(
                    episode_id=episode_id,
                    step_id=step_id,
                    mode=NavMode.SEARCH,
                    target_type="none",
                    action="rotl",
                    step_size=self.action_adapter.config.rotation_step_min,
                    action_source=ActionSource.GEOMETRIC_EXPLORE,
                    reason="no valid search viewpoint; rotate to observe",
                    debug_info={
                        "navigator_reason": "no_valid_viewpoint",
                        "replan_reason": replan_reason,
                    },
                )

            active = ActiveViewpointState(
                viewpoint=viewpoint,
                start_step=step_id,
                last_distance=self._distance_to_viewpoint(current_pose, viewpoint),
                last_progress_metric=self._progress_metric(current_pose, viewpoint),
                last_replan_reason=replan_reason or "new_viewpoint",
            )
            self.active_by_episode[episode_id] = active
            created_new = True

        if not created_new and int(step_id) > int(active.start_step):
            self._update_progress(active, current_pose)

        viewpoint = active.viewpoint
        source = (
            ActionSource.SEMANTIC_MAP
            if viewpoint.region.source == "semantic"
            else ActionSource.GEOMETRIC_EXPLORE
        )

        return self.action_adapter.follow_viewpoint(
            episode_id=episode_id,
            step_id=step_id,
            current_pose=current_pose,
            viewpoint=viewpoint,
            observation=observation,
            semantic_map=semantic_map,
            mode=NavMode.SEARCH,
            target_type="search_viewpoint",
            action_source=source,
            reason="follow active search viewpoint",
            debug_info={
                "navigator_reason": "follow_active_viewpoint",
                "active_viewpoint": active.to_log_dict(),
                "viewpoint_distance": self._distance_to_viewpoint(
                    current_pose,
                    viewpoint,
                ),
                "viewpoint_yaw_error": self._view_yaw_error(
                    current_pose,
                    viewpoint,
                ),
                "progress_metric": self._progress_metric(
                    current_pose,
                    viewpoint,
                ),
                "map_debug": self._build_map_debug(
                    semantic_map,
                    top_k=3,
                ),
                "region_debug": self._build_region_debug(
                    current_pose,
                    semantic_map,
                    top_k=3,
                ),
            },
        )

    def _need_replan(
        self,
        active: Optional[ActiveViewpointState],
        current_pose: PoseRecord,
        semantic_map: SemanticMap,
        step_id: int,
    ) -> Optional[str]:
        if active is None:
            return "no_active_viewpoint"

        viewpoint = active.viewpoint

        if not semantic_map.is_pose_in_bounds(
            PoseRecord(
                x=viewpoint.position[0],
                y=viewpoint.position[1],
                z=viewpoint.position[2],
                yaw=viewpoint.yaw,
            )
        ):
            return "active_viewpoint_out_of_bounds"

        if self._is_viewpoint_reached(current_pose, viewpoint):
            return "active_viewpoint_reached"

        age = int(step_id) - int(active.start_step)
        if age >= self.config.max_active_viewpoint_steps:
            return "active_viewpoint_timeout"

        if active.no_progress_count >= self.config.no_progress_limit:
            return "active_viewpoint_no_progress"

        new_best_score = self.viewpoint_planner.best_region_score(
            current_pose=current_pose,
            semantic_map=semantic_map,
        )
        active_score = float(viewpoint.region.score)

        if new_best_score > active_score + self.config.replan_score_margin:
            return "significantly_better_region"

        return None

    def _is_viewpoint_reached(
        self,
        pose: PoseRecord,
        viewpoint: SearchViewpoint,
    ) -> bool:
        # Search viewpoint completion is position-based.
        # Exact view-yaw alignment is not required in Search stage; otherwise
        # UAV-ON's coarse rotation easily causes in-place yaw oscillation.
        distance = self._distance_to_viewpoint(pose, viewpoint)
        return distance <= self.action_adapter.config.position_tolerance

    def _update_progress(
        self,
        active: ActiveViewpointState,
        current_pose: PoseRecord,
    ) -> None:
        distance = self._distance_to_viewpoint(
            current_pose,
            active.viewpoint,
        )
        metric = self._progress_metric(
            current_pose,
            active.viewpoint,
        )

        if active.last_progress_metric is None:
            active.last_progress_metric = metric
            active.last_distance = distance
            return

        if metric < active.last_progress_metric - self.config.progress_epsilon:
            active.no_progress_count = 0
        else:
            active.no_progress_count += 1

        active.last_distance = distance
        active.last_progress_metric = metric

    def _progress_metric(
        self,
        pose: PoseRecord,
        viewpoint: SearchViewpoint,
    ) -> float:
        # Composite navigation progress:
        # - distance to waypoint should decrease during translation;
        # - heading error to waypoint should decrease during turning.
        distance = self._distance_to_viewpoint(pose, viewpoint)
        yaw_error = abs(self._yaw_to_waypoint_error(pose, viewpoint))

        yaw_equivalent_distance = 0.05 * yaw_error
        return float(distance + yaw_equivalent_distance)


    def _build_map_debug(
        self,
        semantic_map: SemanticMap,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        values = []
        high_value_count = 0
        top_cells = []

        for cell in semantic_map.iter_cells():
            effective = float(cell.semantic_value * cell.semantic_conf)
            values.append(effective)

            status = getattr(cell.status, "value", str(cell.status))
            if status == "high_value":
                high_value_count += 1

            top_cells.append(
                {
                    "gx": int(cell.gx),
                    "gy": int(cell.gy),
                    "semantic_value": float(cell.semantic_value),
                    "semantic_conf": float(cell.semantic_conf),
                    "effective_value": effective,
                    "visited_count": int(cell.visited_count),
                    "status": status,
                }
            )

        values.sort(reverse=True)
        best = float(values[0]) if values else 0.0
        second = float(values[1]) if len(values) > 1 else 0.0

        top_cells.sort(
            key=lambda item: item["effective_value"],
            reverse=True,
        )

        try:
            summary = semantic_map.get_summary(top_k=top_k)
            status_counts = summary.get("status_counts", {})
        except Exception:
            status_counts = {}

        return {
            "best_semantic": best,
            "second_semantic": second,
            "semantic_margin": best - second,
            "high_value_count": int(high_value_count),
            "status_counts": status_counts,
            "top_semantic_cells": top_cells[:top_k],
        }

    def _build_region_debug(
        self,
        current_pose: PoseRecord,
        semantic_map: SemanticMap,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        try:
            regions = self.viewpoint_planner.build_search_regions(
                current_pose=current_pose,
                semantic_map=semantic_map,
            )
        except Exception as exc:
            return {
                "error": str(exc),
                "top_regions": [],
                "top_semantic_regions": [],
            }

        top_regions = []
        top_semantic_regions = []

        for region in regions[:top_k]:
            top_regions.append(region.to_log_dict())

        for region in regions:
            if region.source == "semantic":
                top_semantic_regions.append(region.to_log_dict())
            if len(top_semantic_regions) >= top_k:
                break

        return {
            "top_regions": top_regions,
            "top_semantic_regions": top_semantic_regions,
            "region_count": len(regions),
        }


    @staticmethod
    def _distance_to_viewpoint(
        pose: PoseRecord,
        viewpoint: SearchViewpoint,
    ) -> float:
        return math.hypot(
            float(viewpoint.position[0]) - float(pose.x),
            float(viewpoint.position[1]) - float(pose.y),
        )

    def _yaw_to_waypoint_error(
        self,
        pose: PoseRecord,
        viewpoint: SearchViewpoint,
    ) -> float:
        current_yaw = self._yaw_to_rad(pose.yaw)
        target_angle = math.atan2(
            float(viewpoint.position[1]) - float(pose.y),
            float(viewpoint.position[0]) - float(pose.x),
        )
        return math.degrees(self._angle_diff(target_angle, current_yaw))

    def _view_yaw_error(
        self,
        pose: PoseRecord,
        viewpoint: SearchViewpoint,
    ) -> float:
        current_yaw = self._yaw_to_rad(pose.yaw)
        desired_yaw = self._yaw_to_rad(viewpoint.yaw)
        return math.degrees(self._angle_diff(desired_yaw, current_yaw))

    @staticmethod
    def _yaw_to_rad(yaw: float) -> float:
        yaw = float(yaw)
        if abs(yaw) > 2.0 * math.pi + 1e-3:
            return math.radians(yaw)
        return yaw

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = a - b
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff <= -math.pi:
            diff += 2.0 * math.pi
        return diff


__all__ = [
    "SearchNavigator",
    "SearchNavigatorConfig",
    "ActiveViewpointState",
]
