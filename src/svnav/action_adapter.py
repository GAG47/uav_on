from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from svnav.semantic_map import SemanticMap
from svnav.types import (
    ActionSource,
    NavDecision,
    NavMode,
    ObservationRecord,
    PoseRecord,
    ViewID,
)


@dataclass
class ActionAdapterConfig:
    translation_step_min: float = 2.0
    translation_step_max: float = 5.0

    rotation_step_min: float = 15.0
    rotation_step_max: float = 30.0

    position_tolerance: float = 5.0
    move_yaw_tolerance_deg: float = 45.0
    observe_yaw_tolerance_deg: float = 45.0

    forward_action: str = "forward"
    left_action: str = "left"
    right_action: str = "right"
    rotate_left_action: str = "rotl"
    rotate_right_action: str = "rotr"

    strafe_yaw_threshold_deg: float = 20.0

    min_safe_depth: float = 2.0
    depth_percentile: float = 20.0
    use_depth_safety: bool = True

    def __post_init__(self) -> None:
        self.translation_step_min = float(self.translation_step_min)
        self.translation_step_max = float(self.translation_step_max)
        self.rotation_step_min = float(self.rotation_step_min)
        self.rotation_step_max = float(self.rotation_step_max)

        self.position_tolerance = float(self.position_tolerance)
        self.move_yaw_tolerance_deg = float(self.move_yaw_tolerance_deg)
        self.observe_yaw_tolerance_deg = float(self.observe_yaw_tolerance_deg)
        self.strafe_yaw_threshold_deg = float(self.strafe_yaw_threshold_deg)

        self.min_safe_depth = float(self.min_safe_depth)
        self.depth_percentile = float(self.depth_percentile)

        if self.translation_step_min <= 0:
            self.translation_step_min = 1.0
        if self.translation_step_max < self.translation_step_min:
            self.translation_step_max = self.translation_step_min

        if self.rotation_step_min <= 0:
            self.rotation_step_min = 5.0
        if self.rotation_step_max < self.rotation_step_min:
            self.rotation_step_max = self.rotation_step_min


class ActionAdapter:
    """
    Convert a waypoint/viewpoint into a UAV-ON executable action.

    Search mode:
        Track the waypoint position. Once the position is reached, do not
        force precise view-yaw alignment. Search viewpoints are for obtaining
        new observations, not for final target-facing precision.

    Future Approach mode:
        Can use yaw alignment more strictly around verified target positions.
    """

    def __init__(self, config: Optional[ActionAdapterConfig] = None) -> None:
        self.config = config or ActionAdapterConfig()

    def follow_viewpoint(
        self,
        episode_id: str,
        step_id: int,
        current_pose: PoseRecord,
        viewpoint: Any,
        observation: Optional[ObservationRecord] = None,
        semantic_map: Optional[SemanticMap] = None,
        mode: NavMode = NavMode.SEARCH,
        target_type: str = "viewpoint",
        action_source: ActionSource = ActionSource.GEOMETRIC_EXPLORE,
        reason: str = "",
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> NavDecision:
        position = tuple(viewpoint.position)
        desired_yaw = float(viewpoint.yaw)

        dx = float(position[0]) - float(current_pose.x)
        dy = float(position[1]) - float(current_pose.y)
        distance_xy = math.hypot(dx, dy)

        yaw_rad = self._yaw_to_rad(current_pose.yaw)
        target_angle = math.atan2(dy, dx)
        yaw_to_waypoint = self._angle_diff(target_angle, yaw_rad)
        yaw_to_waypoint_deg = math.degrees(yaw_to_waypoint)

        desired_yaw_rad = self._yaw_to_rad(desired_yaw)
        observe_yaw_error = self._angle_diff(desired_yaw_rad, yaw_rad)
        observe_yaw_error_deg = math.degrees(observe_yaw_error)

        debug = dict(debug_info or {})
        debug.update(
            {
                "viewpoint_id": getattr(viewpoint, "viewpoint_id", None),
                "region_id": getattr(viewpoint, "region_id", None),
                "waypoint": list(position),
                "desired_yaw": desired_yaw,
                "distance_to_waypoint": distance_xy,
                "yaw_to_waypoint_deg": yaw_to_waypoint_deg,
                "observe_yaw_error_deg": observe_yaw_error_deg,
            }
        )

        if distance_xy <= self.config.position_tolerance:
            if self._is_search_mode(mode):
                debug["adapter_phase"] = "viewpoint_reached"
                return NavDecision(
                    episode_id=episode_id,
                    step_id=step_id,
                    mode=mode,
                    target_type=target_type,
                    target_position=position,
                    target_id=getattr(viewpoint, "viewpoint_id", None),
                    action=self.config.rotate_left_action,
                    step_size=self.config.rotation_step_min,
                    action_source=action_source,
                    reason=reason or "search viewpoint position reached",
                    debug_info=debug,
                )

            if abs(observe_yaw_error_deg) <= self.config.observe_yaw_tolerance_deg:
                debug["adapter_phase"] = "viewpoint_reached"
                return NavDecision(
                    episode_id=episode_id,
                    step_id=step_id,
                    mode=mode,
                    target_type=target_type,
                    target_position=position,
                    target_id=getattr(viewpoint, "viewpoint_id", None),
                    action=self.config.rotate_left_action,
                    step_size=self.config.rotation_step_min,
                    action_source=action_source,
                    reason=reason or "viewpoint reached",
                    debug_info=debug,
                )

            action = (
                self.config.rotate_right_action
                if observe_yaw_error_deg > 0.0
                else self.config.rotate_left_action
            )
            step_size = self._rotation_step(abs(observe_yaw_error_deg))
            debug["adapter_phase"] = "align_view"

            return NavDecision(
                episode_id=episode_id,
                step_id=step_id,
                mode=mode,
                target_type=target_type,
                target_position=position,
                target_id=getattr(viewpoint, "viewpoint_id", None),
                action=action,
                step_size=step_size,
                action_source=action_source,
                reason=reason or "align yaw to observe target region",
                debug_info=debug,
            )

        if abs(yaw_to_waypoint_deg) > self.config.move_yaw_tolerance_deg:
            action = (
                self.config.rotate_right_action
                if yaw_to_waypoint_deg > 0.0
                else self.config.rotate_left_action
            )
            step_size = self._rotation_step(abs(yaw_to_waypoint_deg))
            debug["adapter_phase"] = "turn_to_waypoint"

            return NavDecision(
                episode_id=episode_id,
                step_id=step_id,
                mode=mode,
                target_type=target_type,
                target_position=position,
                target_id=getattr(viewpoint, "viewpoint_id", None),
                action=action,
                step_size=step_size,
                action_source=action_source,
                reason=reason or "turn toward waypoint",
                debug_info=debug,
            )

        preferred_actions = self._translation_actions(yaw_to_waypoint_deg)
        for action in preferred_actions:
            if self._is_action_safe(
                action=action,
                current_pose=current_pose,
                observation=observation,
                semantic_map=semantic_map,
            ):
                step_size = self._translation_step(distance_xy)
                debug["adapter_phase"] = "move_to_waypoint"
                debug["preferred_actions"] = preferred_actions

                return NavDecision(
                    episode_id=episode_id,
                    step_id=step_id,
                    mode=mode,
                    target_type=target_type,
                    target_position=position,
                    target_id=getattr(viewpoint, "viewpoint_id", None),
                    action=action,
                    step_size=step_size,
                    action_source=action_source,
                    reason=reason or "move toward waypoint",
                    debug_info=debug,
                )

        action, step_size, safe_reason = self._safe_scan_action(
            current_pose=current_pose,
            observation=observation,
            semantic_map=semantic_map,
        )
        debug["adapter_phase"] = "translation_unsafe"
        debug["safe_reason"] = safe_reason
        debug["preferred_actions"] = preferred_actions

        return NavDecision(
            episode_id=episode_id,
            step_id=step_id,
            mode=mode,
            target_type=target_type,
            target_position=position,
            target_id=getattr(viewpoint, "viewpoint_id", None),
            action=action,
            step_size=step_size,
            action_source=ActionSource.SAFETY_HOLD,
            reason=safe_reason,
            debug_info=debug,
        )

    def target_to_action(
        self,
        episode_id: str,
        step_id: int,
        current_pose: PoseRecord,
        target_position: Tuple[float, float, float],
        observation: Optional[ObservationRecord] = None,
        semantic_map: Optional[SemanticMap] = None,
        mode: NavMode = NavMode.SEARCH,
        target_type: str = "region",
        target_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        action_source: ActionSource = ActionSource.GEOMETRIC_EXPLORE,
        reason: str = "",
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> NavDecision:
        class _TargetViewpoint:
            pass

        vp = _TargetViewpoint()
        vp.viewpoint_id = target_id
        vp.region_id = target_id
        vp.position = tuple(target_position)
        vp.yaw = current_pose.yaw

        return self.follow_viewpoint(
            episode_id=episode_id,
            step_id=step_id,
            current_pose=current_pose,
            viewpoint=vp,
            observation=observation,
            semantic_map=semantic_map,
            mode=mode,
            target_type=target_type,
            action_source=action_source,
            reason=reason,
            debug_info=debug_info,
        )

    def _translation_actions(self, yaw_error_deg: float) -> List[str]:
        if yaw_error_deg > self.config.strafe_yaw_threshold_deg:
            return [
                self.config.left_action,
                self.config.forward_action,
                self.config.rotate_left_action,
            ]

        if yaw_error_deg < -self.config.strafe_yaw_threshold_deg:
            return [
                self.config.right_action,
                self.config.forward_action,
                self.config.rotate_right_action,
            ]

        return [
            self.config.forward_action,
            self.config.left_action,
            self.config.right_action,
        ]

    def _safe_scan_action(
        self,
        current_pose: PoseRecord,
        observation: Optional[ObservationRecord],
        semantic_map: Optional[SemanticMap],
    ) -> Tuple[str, float, str]:
        candidates = [
            self.config.rotate_left_action,
            self.config.rotate_right_action,
            self.config.left_action,
            self.config.right_action,
            self.config.forward_action,
        ]

        for action in candidates:
            if self._is_action_safe(
                action=action,
                current_pose=current_pose,
                observation=observation,
                semantic_map=semantic_map,
            ):
                return action, self._step_size_for_action(action), "safe scan action selected"

        return (
            self.config.rotate_left_action,
            self.config.rotation_step_min,
            "no safe translation; rotate for observation",
        )

    def _translation_step(self, distance: float) -> float:
        return self._clamp(
            distance,
            self.config.translation_step_min,
            self.config.translation_step_max,
        )

    def _rotation_step(self, angle_deg: float) -> float:
        return self._clamp(
            angle_deg,
            self.config.rotation_step_min,
            self.config.rotation_step_max,
        )

    def _step_size_for_action(self, action: str) -> float:
        if action in (self.config.rotate_left_action, self.config.rotate_right_action):
            return self.config.rotation_step_min
        return self.config.translation_step_min

    def _is_action_safe(
        self,
        action: str,
        current_pose: PoseRecord,
        observation: Optional[ObservationRecord],
        semantic_map: Optional[SemanticMap],
    ) -> bool:
        if not self._is_boundary_safe(action, current_pose, semantic_map):
            return False

        if self.config.use_depth_safety:
            if not self._is_depth_safe(action, observation):
                return False

        return True

    def _is_boundary_safe(
        self,
        action: str,
        current_pose: PoseRecord,
        semantic_map: Optional[SemanticMap],
    ) -> bool:
        if semantic_map is None:
            return True

        next_pose = self._predict_next_pose(action, current_pose)
        return semantic_map.is_pose_in_bounds(next_pose)

    def _predict_next_pose(self, action: str, pose: PoseRecord) -> PoseRecord:
        yaw = self._yaw_to_rad(pose.yaw)
        step = self.config.translation_step_max

        dx = 0.0
        dy = 0.0

        if action == self.config.forward_action:
            dx = math.cos(yaw) * step
            dy = math.sin(yaw) * step
        elif action == self.config.left_action:
            dx = math.cos(yaw + math.pi / 2.0) * step
            dy = math.sin(yaw + math.pi / 2.0) * step
        elif action == self.config.right_action:
            dx = math.cos(yaw - math.pi / 2.0) * step
            dy = math.sin(yaw - math.pi / 2.0) * step

        return PoseRecord(
            x=pose.x + dx,
            y=pose.y + dy,
            z=pose.z,
            yaw=pose.yaw,
            pitch=pose.pitch,
            roll=pose.roll,
            quaternion=pose.quaternion,
        )

    def _is_depth_safe(
        self,
        action: str,
        observation: Optional[ObservationRecord],
    ) -> bool:
        if observation is None:
            return True

        view_id = self._action_to_view(action)
        if view_id is None:
            return True

        frame = observation.get_frame(view_id)
        if frame is None:
            return True

        values = self._extract_depth_values(frame.depth, frame.depth_grid3x3)
        if values.size == 0:
            return True

        depth_value = float(np.percentile(values, self.config.depth_percentile))
        return depth_value >= self.config.min_safe_depth

    def _action_to_view(self, action: str) -> Optional[ViewID]:
        if action == self.config.forward_action:
            return ViewID.FRONT
        if action == self.config.left_action:
            return ViewID.LEFT
        if action == self.config.right_action:
            return ViewID.RIGHT
        return None

    @staticmethod
    def _extract_depth_values(depth: Any, depth_grid3x3: Any) -> np.ndarray:
        arrays = []

        if depth is not None:
            try:
                arrays.append(np.asarray(depth, dtype=np.float32).reshape(-1))
            except Exception:
                pass

        if depth_grid3x3 is not None:
            try:
                arrays.append(np.asarray(depth_grid3x3, dtype=np.float32).reshape(-1))
            except Exception:
                pass

        if not arrays:
            return np.asarray([], dtype=np.float32)

        values = np.concatenate(arrays)
        values = values[np.isfinite(values)]
        values = values[values > 0.0]
        return values.astype(np.float32)

    @staticmethod
    def _is_search_mode(mode: Any) -> bool:
        if mode == NavMode.SEARCH:
            return True
        return str(mode).lower() == "search"

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

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        value = float(value)
        low = float(low)
        high = float(high)
        return max(low, min(high, value))


__all__ = [
    "ActionAdapter",
    "ActionAdapterConfig",
]
