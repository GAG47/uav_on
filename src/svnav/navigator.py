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
    NavigationWaypoint,
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
        waypoint = self.select_search_waypoint(
            episode_id=episode_id,
            step_id=step_id,
            observation=observation,
            semantic_map=semantic_map,
        )
        command, feedback = self.action_adapter.follow_waypoint(
            current_pose=observation.pose,
            waypoint=waypoint,
            observation=observation,
            semantic_map=semantic_map,
        )

        debug = dict(waypoint.debug_info or {})
        debug.update(command.debug_info or {})
        debug["adapter_phase"] = command.phase
        debug["navigation_waypoint"] = waypoint.to_log_dict()
        debug["control_command"] = command.to_log_dict()
        debug["controller_feedback"] = feedback.to_log_dict()

        action_source = self._source_to_action_source(waypoint.source)
        if command.source == "safety_hold":
            action_source = ActionSource.SAFETY_HOLD

        return NavDecision(
            episode_id=episode_id,
            step_id=step_id,
            mode=NavMode.SEARCH,
            target_type=waypoint.target_type,
            target_position=waypoint.position,
            target_id=waypoint.target_id,
            action=command.action,
            step_size=command.step_size,
            action_source=action_source,
            reason=command.reason or waypoint.reason,
            debug_info=debug,
        )

    def select_search_waypoint(
        self,
        episode_id: str,
        step_id: int,
        observation: ObservationRecord,
        semantic_map: SemanticMap,
    ) -> NavigationWaypoint:
        current_pose = observation.pose
        episode_id = str(episode_id)

        if not semantic_map.is_pose_in_bounds(current_pose):
            target_position = semantic_map.nearest_in_bounds_position(current_pose)
            self.active_by_episode.pop(episode_id, None)
            return NavigationWaypoint(
                episode_id=episode_id,
                step_id=step_id,
                waypoint_id="return_in_bounds_{}".format(step_id),
                mode="search",
                position=tuple(target_position),
                desired_yaw=current_pose.yaw,
                source=ActionSource.GEOMETRIC_EXPLORE.value,
                target_type="in_bounds_region",
                target_id=None,
                region_id=None,
                arrive_radius=self.action_adapter.config.position_tolerance,
                reason="uav outside semantic map; return to nearest in-bounds position",
                debug_info={
                    "navigator_reason": "out_of_bounds",
                    "target_position": list(target_position),
                    "map_debug": self._build_map_debug(semantic_map, top_k=3),
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
                return NavigationWaypoint(
                    episode_id=episode_id,
                    step_id=step_id,
                    waypoint_id="observe_rotate_{}".format(step_id),
                    mode="search",
                    position=(float(current_pose.x), float(current_pose.y), float(current_pose.z)),
                    desired_yaw=float(current_pose.yaw) + self.action_adapter.config.rotation_step_min,
                    source=ActionSource.GEOMETRIC_EXPLORE.value,
                    target_type="none",
                    arrive_radius=self.action_adapter.config.position_tolerance,
                    reason="no valid search viewpoint; rotate to observe",
                    debug_info={
                        "navigator_reason": "no_valid_viewpoint",
                        "replan_reason": replan_reason,
                        "fallback_action": "rotate_observe",
                        "map_debug": self._build_map_debug(semantic_map, top_k=3),
                        "region_debug": self._build_region_debug(
                            current_pose,
                            semantic_map,
                            top_k=3,
                        ),
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
            ActionSource.SEMANTIC_MAP.value
            if viewpoint.region.source == "semantic"
            else ActionSource.GEOMETRIC_EXPLORE.value
        )

        debug_info = {
            "navigator_reason": "follow_active_viewpoint",
            "active_viewpoint": active.to_log_dict(),
            "viewpoint_id": getattr(viewpoint, "viewpoint_id", None),
            "region_id": getattr(viewpoint, "region_id", None),
            "viewpoint_distance": self._distance_to_viewpoint(current_pose, viewpoint),
            "viewpoint_yaw_error": self._view_yaw_error(current_pose, viewpoint),
            "progress_metric": self._progress_metric(current_pose, viewpoint),
            "map_debug": self._build_map_debug(semantic_map, top_k=3),
            "region_debug": self._build_region_debug(
                current_pose,
                semantic_map,
                top_k=3,
            ),
        }

        return NavigationWaypoint(
            episode_id=episode_id,
            step_id=step_id,
            waypoint_id=str(getattr(viewpoint, "viewpoint_id", "search_viewpoint")),
            mode="search",
            position=tuple(viewpoint.position),
            desired_yaw=float(viewpoint.yaw),
            source=source,
            target_type="search_viewpoint",
            target_id=getattr(viewpoint, "viewpoint_id", None),
            region_id=getattr(viewpoint, "region_id", None),
            arrive_radius=self.action_adapter.config.position_tolerance,
            yaw_tolerance=self.action_adapter.config.observe_yaw_tolerance_deg,
            reason="follow active search viewpoint",
            debug_info=debug_info,
        )

    @staticmethod
    def _source_to_action_source(source: str) -> ActionSource:
        try:
            return ActionSource(source)
        except Exception:
            return ActionSource.GEOMETRIC_EXPLORE

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

# ----------------------------------------------------------------------
# SVNav target approach navigation
# ----------------------------------------------------------------------
#
# Step 14 contract:
#   - Step 13 decides whether a candidate can be approached:
#       task2_admission = passed / pending / rejected
#   - Step 13 also provides where to look:
#       approach_anchor.type = visual / spatial
#   - Step 14 only executes the anchor. It must not use raw depth/position_3d
#     to override Task2 admission or anchor type.

import math as _svnav_approach_math

try:
    from common.param import args as _svnav_args
except Exception:  # pragma: no cover
    _svnav_args = None

from svnav.types import (
    ActionSource as _SVNavActionSource,
    NavDecision as _SVNavDecision,
    NavMode as _SVNavMode,
)


def _svnav_approach_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _svnav_approach_position(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (
                float(value[0]),
                float(value[1]),
                float(value[2]),
            )
        except Exception:
            return None
    return None


def _svnav_approach_angle_diff_deg(target_deg, current_deg):
    diff = float(target_deg) - float(current_deg)
    while diff > 180.0:
        diff -= 360.0
    while diff <= -180.0:
        diff += 360.0
    return diff


def _svnav_approach_default_step_size():
    if _svnav_args is not None:
        return float(getattr(_svnav_args, "xOy_step_size", 5.0))
    return 5.0


def _svnav_approach_default_rotate_angle():
    if _svnav_args is not None:
        return float(getattr(_svnav_args, "rotateAngle", 30.0))
    return 30.0


def _svnav_view_yaw_offset(view_id):
    view = str(view_id or "").lower()
    if view == "left":
        return -90.0
    if view == "right":
        return 90.0
    if view == "back":
        return 180.0
    return 0.0


def _svnav_bbox_yaw_offset(bbox_center_norm):
    if not isinstance(bbox_center_norm, (list, tuple)) or len(bbox_center_norm) < 2:
        return 0.0

    cx = _svnav_approach_float(bbox_center_norm[0], 0.5)
    return (cx - 0.5) * 70.0


def _svnav_get_metadata(target_evidence):
    metadata = getattr(target_evidence, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return {}


def _svnav_get_approach_anchor(target_evidence):
    """
    Read Step 13 output.

    Do not infer spatial/visual from raw position_3d here. Step 13 owns that
    decision through anchor_type / approach_anchor.
    """
    metadata = _svnav_get_metadata(target_evidence)

    admission = metadata.get("task2_admission")
    anchor_type = metadata.get("anchor_type") or metadata.get("evidence_kind")
    anchor = metadata.get("approach_anchor")

    if not isinstance(anchor, dict):
        if anchor_type == "spatial":
            anchor = metadata.get("spatial_anchor")
        else:
            anchor = metadata.get("visual_anchor")

    if not isinstance(anchor, dict):
        anchor = {
            "type": "visual",
            "source_pose": metadata.get("latest_source_pose"),
            "view_id": metadata.get("latest_view_id"),
            "bbox": metadata.get("latest_bbox"),
            "bbox_center_norm": metadata.get("latest_bbox_center_norm"),
            "frame_id": metadata.get("latest_frame_id"),
        }

    anchor_type = str(anchor.get("type") or anchor_type or "visual").lower()
    if anchor_type not in ("visual", "spatial"):
        anchor_type = "visual"

    # A malformed spatial anchor falls back to visual anchor. This keeps Step 14
    # robust without letting raw target_evidence.position_3d override Step 13.
    if anchor_type == "spatial":
        spatial_position = _svnav_approach_position(anchor.get("position_3d"))
        if spatial_position is None:
            visual_anchor = metadata.get("visual_anchor")
            if isinstance(visual_anchor, dict):
                anchor = visual_anchor
            anchor_type = "visual"

    return anchor_type, anchor, admission


def _svnav_build_spatial_approach_viewpoint(self, current_pose, target_evidence, anchor):
    target_pos = _svnav_approach_position(anchor.get("position_3d"))
    if target_pos is None:
        return None

    current_x = _svnav_approach_float(getattr(current_pose, "x", 0.0))
    current_y = _svnav_approach_float(getattr(current_pose, "y", 0.0))
    current_z = _svnav_approach_float(getattr(current_pose, "z", 0.0))
    current_yaw = _svnav_approach_float(getattr(current_pose, "yaw", 0.0))

    target_x, target_y, target_z = target_pos

    dx = target_x - current_x
    dy = target_y - current_y
    horizontal_dist = _svnav_approach_math.sqrt(dx * dx + dy * dy)

    approach_distance = _svnav_approach_float(
        getattr(getattr(self, "config", None), "approach_distance", 7.0),
        7.0,
    )
    min_approach_distance = _svnav_approach_float(
        getattr(getattr(self, "config", None), "min_approach_distance", 5.0),
        5.0,
    )
    final_check_distance = _svnav_approach_float(
        getattr(getattr(self, "config", None), "final_check_distance", 7.0),
        7.0,
    )
    approach_distance = max(min_approach_distance, approach_distance)

    if horizontal_dist > 1e-6:
        unit_x = dx / horizontal_dist
        unit_y = dy / horizontal_dist
    else:
        yaw_rad = _svnav_approach_math.radians(current_yaw)
        unit_x = _svnav_approach_math.cos(yaw_rad)
        unit_y = _svnav_approach_math.sin(yaw_rad)

    if horizontal_dist > approach_distance:
        vp_x = target_x - unit_x * approach_distance
        vp_y = target_y - unit_y * approach_distance
    else:
        vp_x = current_x
        vp_y = current_y

    desired_yaw = _svnav_approach_math.degrees(
        _svnav_approach_math.atan2(target_y - vp_y, target_x - vp_x)
    )

    return {
        "approach_kind": "spatial",
        "target_position": target_pos,
        "viewpoint": (float(vp_x), float(vp_y), float(current_z)),
        "observe_yaw": float(desired_yaw),
        "target_distance": float(horizontal_dist),
        "approach_distance": float(approach_distance),
        "final_check_distance": float(final_check_distance),
        "anchor": dict(anchor),
    }


def _svnav_build_visual_approach_viewpoint(self, current_pose, target_evidence, anchor):
    source_pose = anchor.get("source_pose") or {}
    bbox_center = anchor.get("bbox_center_norm")
    if bbox_center is None and isinstance(anchor.get("bbox"), dict):
        bbox = anchor.get("bbox") or {}
        center = bbox.get("center")
        image_width = _svnav_approach_float(bbox.get("image_width"), 0.0)
        image_height = _svnav_approach_float(bbox.get("image_height"), 0.0)
        if isinstance(center, (list, tuple)) and len(center) >= 2 and image_width > 1.0 and image_height > 1.0:
            bbox_center = [
                _svnav_approach_float(center[0]) / image_width,
                _svnav_approach_float(center[1]) / image_height,
            ]

    view_id = anchor.get("view_id") or ""

    current_x = _svnav_approach_float(getattr(current_pose, "x", 0.0))
    current_y = _svnav_approach_float(getattr(current_pose, "y", 0.0))
    current_z = _svnav_approach_float(getattr(current_pose, "z", 0.0))
    current_yaw = _svnav_approach_float(getattr(current_pose, "yaw", 0.0))

    source_x = _svnav_approach_float(source_pose.get("x"), current_x)
    source_y = _svnav_approach_float(source_pose.get("y"), current_y)
    source_z = _svnav_approach_float(source_pose.get("z"), current_z)
    source_yaw = _svnav_approach_float(source_pose.get("yaw"), current_yaw)

    observe_yaw = (
        source_yaw
        + _svnav_view_yaw_offset(view_id)
        + _svnav_bbox_yaw_offset(bbox_center)
    )

    inspect_distance = _svnav_approach_float(
        getattr(getattr(self, "config", None), "visual_inspect_distance", 6.0),
        6.0,
    )
    final_check_distance = _svnav_approach_float(
        getattr(getattr(self, "config", None), "final_check_distance", 7.0),
        7.0,
    )

    yaw_rad = _svnav_approach_math.radians(observe_yaw)
    ray_x = _svnav_approach_math.cos(yaw_rad)
    ray_y = _svnav_approach_math.sin(yaw_rad)

    vp_x = source_x + ray_x * inspect_distance
    vp_y = source_y + ray_y * inspect_distance

    dist_from_current = _svnav_approach_math.sqrt(
        (vp_x - current_x) ** 2 + (vp_y - current_y) ** 2
    )

    return {
        "approach_kind": "visual",
        "target_position": None,
        "viewpoint": (float(vp_x), float(vp_y), float(source_z)),
        "observe_yaw": float(observe_yaw),
        "target_distance": None,
        "dist_to_visual_viewpoint": float(dist_from_current),
        "approach_distance": float(inspect_distance),
        "final_check_distance": float(final_check_distance),
        "source_pose": source_pose,
        "source_view_id": view_id,
        "source_bbox_center_norm": bbox_center,
        "anchor": dict(anchor),
    }


def _svnav_build_approach_viewpoint(self, current_pose, target_evidence):
    anchor_type, anchor, admission = _svnav_get_approach_anchor(target_evidence)

    if anchor_type == "spatial":
        spatial = _svnav_build_spatial_approach_viewpoint(
            self=self,
            current_pose=current_pose,
            target_evidence=target_evidence,
            anchor=anchor,
        )
        if spatial is not None:
            spatial["task2_admission"] = admission
            spatial["anchor_type"] = "spatial"
            return spatial

    visual = _svnav_build_visual_approach_viewpoint(
        self=self,
        current_pose=current_pose,
        target_evidence=target_evidence,
        anchor=anchor,
    )
    visual["task2_admission"] = admission
    visual["anchor_type"] = "visual"
    return visual


def _svnav_build_approach_decision(
    self,
    episode_id,
    step_id,
    observation,
    target_evidence,
):
    current_pose = getattr(observation, "pose", None)
    if current_pose is None:
        return _SVNavDecision.hold(
            episode_id=episode_id,
            step_id=step_id,
            reason="approach missing current pose",
        )

    metadata = _svnav_get_metadata(target_evidence)

    # Defense in depth: Step 13 should already ensure this. Step 14 refuses to
    # approach a candidate that is not Task2-passed.
    if metadata.get("task2_admission") not in (None, "passed"):
        return _SVNavDecision.hold(
            episode_id=episode_id,
            step_id=step_id,
            reason="approach candidate not Task2-passed",
        )

    approach_viewpoint = _svnav_build_approach_viewpoint(
        self=self,
        current_pose=current_pose,
        target_evidence=target_evidence,
    )

    if approach_viewpoint is None:
        return _SVNavDecision.hold(
            episode_id=episode_id,
            step_id=step_id,
            reason="approach missing candidate anchor",
        )

    current_x = _svnav_approach_float(getattr(current_pose, "x", 0.0))
    current_y = _svnav_approach_float(getattr(current_pose, "y", 0.0))
    current_yaw = _svnav_approach_float(getattr(current_pose, "yaw", 0.0))

    vp_x, vp_y, vp_z = approach_viewpoint["viewpoint"]
    to_vp_x = vp_x - current_x
    to_vp_y = vp_y - current_y
    dist_to_viewpoint = _svnav_approach_math.sqrt(to_vp_x * to_vp_x + to_vp_y * to_vp_y)

    if dist_to_viewpoint > 1e-6:
        desired_move_yaw = _svnav_approach_math.degrees(
            _svnav_approach_math.atan2(to_vp_y, to_vp_x)
        )
    else:
        desired_move_yaw = _svnav_approach_float(approach_viewpoint.get("observe_yaw"), current_yaw)

    yaw_error = _svnav_approach_angle_diff_deg(desired_move_yaw, current_yaw)
    observe_yaw = _svnav_approach_float(approach_viewpoint.get("observe_yaw"), desired_move_yaw)
    observe_yaw_error = _svnav_approach_angle_diff_deg(observe_yaw, current_yaw)

    step_size = _svnav_approach_default_step_size()
    rotate_angle = _svnav_approach_default_rotate_angle()

    yaw_threshold = _svnav_approach_float(
        getattr(getattr(self, "config", None), "approach_yaw_threshold_deg", 18.0),
        18.0,
    )
    waypoint_tolerance = _svnav_approach_float(
        getattr(getattr(self, "config", None), "approach_waypoint_tolerance", 1.5),
        1.5,
    )

    approach_kind = approach_viewpoint.get("approach_kind", "visual")
    anchor_type = approach_viewpoint.get("anchor_type", approach_kind)
    final_check_distance = _svnav_approach_float(approach_viewpoint.get("final_check_distance"), 7.0)

    mode = _SVNavMode.APPROACH
    action = "forward"
    action_step = min(step_size, max(1.0, dist_to_viewpoint))
    reason = "move_to_approach_candidate_viewpoint"
    phase = "move_to_approach_viewpoint"

    target_distance = approach_viewpoint.get("target_distance")

    if approach_kind == "spatial" and target_distance is not None and target_distance <= final_check_distance:
        mode = _SVNavMode.FINAL_CHECK
        if abs(observe_yaw_error) > yaw_threshold:
            action = "rotr" if observe_yaw_error > 0.0 else "rotl"
            action_step = rotate_angle
            reason = "near_spatial_candidate_align"
            phase = "spatial_final_check_align"
        else:
            action = "rotl"
            action_step = rotate_angle
            reason = "near_spatial_candidate_observe"
            phase = "spatial_final_check_observe"

    elif dist_to_viewpoint <= waypoint_tolerance:
        if abs(observe_yaw_error) > yaw_threshold:
            action = "rotr" if observe_yaw_error > 0.0 else "rotl"
            action_step = rotate_angle
            reason = "{}_candidate_viewpoint_reached_align".format(approach_kind)
            phase = "{}_align_candidate".format(approach_kind)
        else:
            action = "forward"
            action_step = min(step_size, 3.0)
            reason = "{}_candidate_viewpoint_reached_inspect_forward".format(approach_kind)
            phase = "{}_inspect_forward".format(approach_kind)

    elif abs(yaw_error) > yaw_threshold:
        action = "rotr" if yaw_error > 0.0 else "rotl"
        action_step = rotate_angle
        reason = "turn_to_{}_approach_viewpoint".format(approach_kind)
        phase = "turn_to_{}_approach_viewpoint".format(approach_kind)

    else:
        action = "forward"
        action_step = min(step_size, max(1.0, dist_to_viewpoint))
        reason = "move_to_{}_candidate_viewpoint".format(approach_kind)
        phase = "move_to_{}_approach_viewpoint".format(approach_kind)

    debug_info = {
        "adapter_phase": phase,
        "approach_kind": approach_kind,
        "anchor_type": anchor_type,
        "task2_admission": approach_viewpoint.get("task2_admission"),
        "target_id": getattr(target_evidence, "target_id", None),
        "target_status": getattr(getattr(target_evidence, "status", None), "value", str(getattr(target_evidence, "status", ""))),
        "target_position": approach_viewpoint.get("target_position"),
        "approach_viewpoint": approach_viewpoint.get("viewpoint"),
        "approach_anchor": approach_viewpoint.get("anchor"),
        "target_distance": target_distance,
        "dist_to_approach_viewpoint": float(dist_to_viewpoint),
        "approach_distance": approach_viewpoint.get("approach_distance"),
        "final_check_distance": final_check_distance,
        "desired_yaw": float(desired_move_yaw),
        "yaw_to_viewpoint_deg": float(yaw_error),
        "observe_yaw": float(observe_yaw),
        "observe_yaw_error_deg": float(observe_yaw_error),
        "visual_score": metadata.get("visual_score"),
        "task2_score": metadata.get("task2_score"),
        "approach_score": metadata.get("approach_score"),
        "rank_score": metadata.get("rank_score"),
        "evidence_kind": metadata.get("evidence_kind"),
        "latest_view_id": metadata.get("latest_view_id"),
        "latest_bbox_center_norm": metadata.get("latest_bbox_center_norm"),
        "latest_source_pose": metadata.get("latest_source_pose"),
        "position_confidence": getattr(target_evidence, "position_confidence", 0.0),
        "position_stability": getattr(target_evidence, "position_stability", 0.0),
    }

    nav_target_position = approach_viewpoint.get("target_position")
    if nav_target_position is None:
        nav_target_position = approach_viewpoint.get("viewpoint")

    return _SVNavDecision(
        episode_id=episode_id,
        step_id=int(step_id),
        mode=mode,
        target_type="approach_candidate",
        target_position=nav_target_position,
        target_id=getattr(target_evidence, "target_id", None),
        candidate_id=getattr(target_evidence, "latest_candidate_id", None),
        action=action,
        step_size=float(action_step),
        action_source=_SVNavActionSource.VERIFIED_TARGET,
        stop_allowed=False,
        reason=reason,
        debug_info=debug_info,
    )


SearchNavigator.build_approach_decision = _svnav_build_approach_decision

