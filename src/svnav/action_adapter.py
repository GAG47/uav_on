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
    horizontal_step_size: float = 5.0
    yaw_step_size: float = 30.0

    forward_action: str = "forward"
    left_action: str = "left"
    right_action: str = "right"
    rotate_left_action: str = "rotl"
    rotate_right_action: str = "rotr"

    rotate_threshold_deg: float = 45.0
    strafe_threshold_deg: float = 15.0
    arrive_distance: float = 4.0

    min_safe_depth: float = 2.0
    depth_percentile: float = 20.0
    use_depth_safety: bool = True

    def __post_init__(self) -> None:
        self.horizontal_step_size = float(self.horizontal_step_size)
        self.yaw_step_size = float(self.yaw_step_size)
        self.rotate_threshold_deg = float(self.rotate_threshold_deg)
        self.strafe_threshold_deg = float(self.strafe_threshold_deg)
        self.arrive_distance = float(self.arrive_distance)
        self.min_safe_depth = float(self.min_safe_depth)
        self.depth_percentile = float(self.depth_percentile)

        if self.horizontal_step_size <= 0:
            raise ValueError("horizontal_step_size must be positive")
        if self.yaw_step_size <= 0:
            raise ValueError("yaw_step_size must be positive")


class ActionAdapter:
    """
    Convert a selected world-space target position into a UAV-ON executable action.

    This class does not select semantic targets and does not call VLM.
    It only converts:
        current_pose + target_position -> action + step_size

    Safety handled here:
        - simple search-boundary check
        - simple depth check for obvious collision risk
    """

    def __init__(self, config: Optional[ActionAdapterConfig] = None) -> None:
        self.config = config or ActionAdapterConfig()

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
        dx = float(target_position[0]) - float(current_pose.x)
        dy = float(target_position[1]) - float(current_pose.y)
        distance_xy = math.hypot(dx, dy)

        debug = dict(debug_info or {})
        debug.update(
            {
                "target_position": list(target_position),
                "distance_xy": distance_xy,
            }
        )

        if distance_xy <= self.config.arrive_distance:
            action, step_size, safe_reason = self._safe_scan_action(
                current_pose=current_pose,
                observation=observation,
                semantic_map=semantic_map,
            )
            debug["adapter_reason"] = "target_region_reached_scan"
            debug["safe_reason"] = safe_reason

            return NavDecision(
                episode_id=episode_id,
                step_id=step_id,
                mode=mode,
                target_type=target_type,
                target_position=target_position,
                target_id=target_id,
                candidate_id=candidate_id,
                action=action,
                step_size=step_size,
                action_source=action_source,
                reason=reason or "target region reached; scan for new observation",
                debug_info=debug,
            )

        yaw = self._yaw_to_rad(current_pose.yaw)
        target_angle = math.atan2(dy, dx)
        angle_diff = self._angle_diff(target_angle, yaw)
        angle_diff_deg = math.degrees(angle_diff)

        debug["target_angle_deg"] = math.degrees(target_angle)
        debug["current_yaw_deg"] = math.degrees(yaw)
        debug["angle_diff_deg"] = angle_diff_deg

        preferred_actions = self._preferred_actions_from_angle(angle_diff_deg)

        for action in preferred_actions:
            if self._is_action_safe(
                action=action,
                current_pose=current_pose,
                observation=observation,
                semantic_map=semantic_map,
            ):
                step_size = self._step_size_for_action(action)
                debug["adapter_reason"] = "selected_preferred_action"
                debug["preferred_actions"] = preferred_actions

                return NavDecision(
                    episode_id=episode_id,
                    step_id=step_id,
                    mode=mode,
                    target_type=target_type,
                    target_position=target_position,
                    target_id=target_id,
                    candidate_id=candidate_id,
                    action=action,
                    step_size=step_size,
                    action_source=action_source,
                    reason=reason or "move toward selected search target",
                    debug_info=debug,
                )

        action, step_size, safe_reason = self._safe_scan_action(
            current_pose=current_pose,
            observation=observation,
            semantic_map=semantic_map,
        )
        debug["adapter_reason"] = "preferred_actions_unsafe"
        debug["preferred_actions"] = preferred_actions
        debug["safe_reason"] = safe_reason

        return NavDecision(
            episode_id=episode_id,
            step_id=step_id,
            mode=mode,
            target_type=target_type,
            target_position=target_position,
            target_id=target_id,
            candidate_id=candidate_id,
            action=action,
            step_size=step_size,
            action_source=ActionSource.SAFETY_HOLD,
            reason=safe_reason,
            debug_info=debug,
        )

    # ------------------------------------------------------------------
    # Action choice
    # ------------------------------------------------------------------

    def _preferred_actions_from_angle(self, angle_diff_deg: float) -> List[str]:
        if angle_diff_deg > self.config.rotate_threshold_deg:
            return [
                self.config.rotate_left_action,
                self.config.left_action,
                self.config.forward_action,
            ]

        if angle_diff_deg < -self.config.rotate_threshold_deg:
            return [
                self.config.rotate_right_action,
                self.config.right_action,
                self.config.forward_action,
            ]

        if angle_diff_deg > self.config.strafe_threshold_deg:
            return [
                self.config.left_action,
                self.config.forward_action,
                self.config.rotate_left_action,
            ]

        if angle_diff_deg < -self.config.strafe_threshold_deg:
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

        return self.config.rotate_left_action, self.config.yaw_step_size, "no safe translation; rotate for observation"

    def _step_size_for_action(self, action: str) -> float:
        if action in (self.config.rotate_left_action, self.config.rotate_right_action):
            return self.config.yaw_step_size
        return self.config.horizontal_step_size

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

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
        step = self.config.horizontal_step_size

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

    # ------------------------------------------------------------------
    # Angle helpers
    # ------------------------------------------------------------------

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
    "ActionAdapter",
    "ActionAdapterConfig",
]
