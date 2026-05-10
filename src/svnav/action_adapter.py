from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from svnav.semantic_map import SemanticMap
from svnav.types import (
    ActionSource,
    ControlCommand,
    ControllerFeedback,
    NavDecision,
    NavMode,
    NavigationWaypoint,
    ObservationRecord,
    PoseRecord,
    ViewID,
)


@dataclass
class ActionAdapterConfig:
    translation_step_min: float = 1.5
    translation_step_max: float = 5.0
    rotation_step_min: float = 6.0
    rotation_step_max: float = 25.0
    position_tolerance: float = 5.0

    hard_turn_yaw_threshold_deg: float = 105.0
    soft_turn_yaw_threshold_deg: float = 35.0
    strafe_yaw_threshold_deg: float = 18.0
    observe_yaw_tolerance_deg: float = 45.0

    forward_action: str = "forward"
    left_action: str = "left"
    right_action: str = "right"
    rotate_left_action: str = "rotl"
    rotate_right_action: str = "rotr"

    min_safe_depth: float = 2.0
    depth_percentile: float = 20.0
    use_depth_safety: bool = True

    def __post_init__(self) -> None:
        self.translation_step_min = float(self.translation_step_min)
        self.translation_step_max = float(self.translation_step_max)
        self.rotation_step_min = float(self.rotation_step_min)
        self.rotation_step_max = float(self.rotation_step_max)
        self.position_tolerance = float(self.position_tolerance)

        self.hard_turn_yaw_threshold_deg = float(self.hard_turn_yaw_threshold_deg)
        self.soft_turn_yaw_threshold_deg = float(self.soft_turn_yaw_threshold_deg)
        self.strafe_yaw_threshold_deg = float(self.strafe_yaw_threshold_deg)
        self.observe_yaw_tolerance_deg = float(self.observe_yaw_tolerance_deg)

        self.min_safe_depth = float(self.min_safe_depth)
        self.depth_percentile = float(self.depth_percentile)

        if self.translation_step_min <= 0.0:
            self.translation_step_min = 1.0
        if self.translation_step_max < self.translation_step_min:
            self.translation_step_max = self.translation_step_min

        if self.rotation_step_min <= 0.0:
            self.rotation_step_min = 5.0
        if self.rotation_step_max < self.rotation_step_min:
            self.rotation_step_max = self.rotation_step_min

        if self.soft_turn_yaw_threshold_deg <= 0.0:
            self.soft_turn_yaw_threshold_deg = 30.0
        if self.hard_turn_yaw_threshold_deg < self.soft_turn_yaw_threshold_deg:
            self.hard_turn_yaw_threshold_deg = self.soft_turn_yaw_threshold_deg


class ActionAdapter:
    """
    Waypoint controller for SVNav.

    The navigator selects a continuous waypoint. This adapter tracks that
    waypoint and converts it to UAV-ON's existing parameterized bottom-level
    action API.

    This file does not add Inspect, does not change TargetEvidence, and does
    not change StopGate.
    """

    def __init__(self, config: Optional[ActionAdapterConfig] = None) -> None:
        self.config = config or ActionAdapterConfig()

    def follow_waypoint(
        self,
        current_pose: PoseRecord,
        waypoint: NavigationWaypoint,
        observation: Optional[ObservationRecord] = None,
        semantic_map: Optional[SemanticMap] = None,
        last_feedback: Optional[ControllerFeedback] = None,
    ) -> Tuple[ControlCommand, ControllerFeedback]:
        position = tuple(waypoint.position)

        if waypoint.debug_info.get("fallback_action") == "rotate_observe":
            command = ControlCommand(
                action=self.config.rotate_left_action,
                step_size=self.config.rotation_step_min,
                source="waypoint_controller",
                phase="observe_rotate",
                reason=waypoint.reason or "no valid search waypoint; rotate to observe",
                waypoint_id=waypoint.waypoint_id,
                debug_info=dict(waypoint.debug_info),
            )
            feedback = ControllerFeedback(
                waypoint_id=waypoint.waypoint_id,
                reached=False,
                distance_to_waypoint=0.0,
                yaw_to_waypoint=0.0,
                observe_yaw_error=0.0,
                translation_safe=True,
                reason=command.reason,
                debug_info={"phase": command.phase},
            )
            return command, feedback

        dx = float(position[0]) - float(current_pose.x)
        dy = float(position[1]) - float(current_pose.y)
        distance_xy = math.hypot(dx, dy)

        yaw_rad = self._yaw_to_rad(current_pose.yaw)
        target_angle = math.atan2(dy, dx)
        yaw_to_waypoint = self._angle_diff(target_angle, yaw_rad)
        yaw_to_waypoint_deg = math.degrees(yaw_to_waypoint)

        desired_yaw = waypoint.desired_yaw
        if desired_yaw is None:
            desired_yaw = current_pose.yaw
        desired_yaw_rad = self._yaw_to_rad(desired_yaw)
        observe_yaw_error = self._angle_diff(desired_yaw_rad, yaw_rad)
        observe_yaw_error_deg = math.degrees(observe_yaw_error)

        debug = dict(waypoint.debug_info or {})
        debug.update(
            {
                "waypoint_id": waypoint.waypoint_id,
                "viewpoint_id": waypoint.debug_info.get("viewpoint_id"),
                "region_id": waypoint.region_id,
                "waypoint": list(position),
                "desired_yaw": desired_yaw,
                "distance_to_waypoint": distance_xy,
                "yaw_to_waypoint_deg": yaw_to_waypoint_deg,
                "observe_yaw_error_deg": observe_yaw_error_deg,
                "controller": "continuous_waypoint_controller",
                "hard_turn_yaw_threshold_deg": self.config.hard_turn_yaw_threshold_deg,
                "soft_turn_yaw_threshold_deg": self.config.soft_turn_yaw_threshold_deg,
            }
        )

        reached = distance_xy <= float(waypoint.arrive_radius)
        if reached:
            command = ControlCommand(
                action=self.config.rotate_left_action,
                step_size=0.0,
                source="waypoint_controller",
                phase="waypoint_reached",
                reason=waypoint.reason or "search waypoint reached; wait for navigator replan",
                waypoint_id=waypoint.waypoint_id,
                debug_info=debug,
            )
            feedback = ControllerFeedback(
                waypoint_id=waypoint.waypoint_id,
                reached=True,
                distance_to_waypoint=distance_xy,
                yaw_to_waypoint=yaw_to_waypoint_deg,
                observe_yaw_error=observe_yaw_error_deg,
                translation_safe=True,
                reason=command.reason,
                debug_info={"phase": command.phase},
            )
            return command, feedback

        abs_yaw = abs(yaw_to_waypoint_deg)

        if abs_yaw >= self.config.hard_turn_yaw_threshold_deg:
            action = self._turn_action(yaw_to_waypoint_deg)
            step_size = self._rotation_step(abs_yaw)
            command = ControlCommand(
                action=action,
                step_size=step_size,
                source="waypoint_controller",
                phase="continuous_turn_to_waypoint",
                reason=waypoint.reason or "continuous turn toward waypoint",
                waypoint_id=waypoint.waypoint_id,
                debug_info=debug,
            )
            feedback = self._build_feedback(
                waypoint=waypoint,
                distance_xy=distance_xy,
                yaw_to_waypoint_deg=yaw_to_waypoint_deg,
                observe_yaw_error_deg=observe_yaw_error_deg,
                command=command,
                last_feedback=last_feedback,
                translation_safe=True,
            )
            return command, feedback

        if abs_yaw >= self.config.soft_turn_yaw_threshold_deg:
            preferred_actions = self._soft_translation_actions(yaw_to_waypoint_deg)
            step_size = self._soft_translation_step(distance_xy, abs_yaw)
            phase = "turn_while_tracking_waypoint"
        else:
            preferred_actions = self._direct_translation_actions(yaw_to_waypoint_deg)
            step_size = self._translation_step(distance_xy)
            phase = "move_to_waypoint"

        for action in preferred_actions:
            if self._is_action_safe(
                action=action,
                current_pose=current_pose,
                observation=observation,
                semantic_map=semantic_map,
                step_size=step_size,
                waypoint_position=position,
            ):
                debug["preferred_actions"] = preferred_actions
                debug["selected_action"] = action
                debug["selected_step_size"] = step_size
                command = ControlCommand(
                    action=action,
                    step_size=step_size,
                    source="waypoint_controller",
                    phase=phase,
                    reason=waypoint.reason or "follow continuous waypoint",
                    waypoint_id=waypoint.waypoint_id,
                    debug_info=debug,
                )
                feedback = self._build_feedback(
                    waypoint=waypoint,
                    distance_xy=distance_xy,
                    yaw_to_waypoint_deg=yaw_to_waypoint_deg,
                    observe_yaw_error_deg=observe_yaw_error_deg,
                    command=command,
                    last_feedback=last_feedback,
                    translation_safe=True,
                )
                return command, feedback

        action, fallback_step, safe_reason = self._safe_scan_action(
            current_pose=current_pose,
            observation=observation,
            semantic_map=semantic_map,
            waypoint_position=position,
            preferred_actions=preferred_actions,
            yaw_error_deg=yaw_to_waypoint_deg,
        )
        debug["preferred_actions"] = preferred_actions
        debug["selected_action"] = action
        debug["selected_step_size"] = fallback_step
        debug["safe_reason"] = safe_reason

        command = ControlCommand(
            action=action,
            step_size=fallback_step,
            source="safety_hold",
            phase="translation_unsafe",
            reason=safe_reason,
            waypoint_id=waypoint.waypoint_id,
            debug_info=debug,
        )
        feedback = self._build_feedback(
            waypoint=waypoint,
            distance_xy=distance_xy,
            yaw_to_waypoint_deg=yaw_to_waypoint_deg,
            observe_yaw_error_deg=observe_yaw_error_deg,
            command=command,
            last_feedback=last_feedback,
            translation_safe=False,
        )
        return command, feedback

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
        waypoint = NavigationWaypoint(
            episode_id=episode_id,
            step_id=step_id,
            waypoint_id=str(getattr(viewpoint, "viewpoint_id", "viewpoint")),
            mode=getattr(mode, "value", str(mode)),
            position=tuple(viewpoint.position),
            desired_yaw=float(getattr(viewpoint, "yaw", current_pose.yaw)),
            source=getattr(action_source, "value", str(action_source)),
            target_type=target_type,
            target_id=getattr(viewpoint, "viewpoint_id", None),
            region_id=getattr(viewpoint, "region_id", None),
            arrive_radius=self.config.position_tolerance,
            yaw_tolerance=self.config.observe_yaw_tolerance_deg,
            reason=reason,
            debug_info=debug_info or {},
        )
        command, feedback = self.follow_waypoint(
            current_pose=current_pose,
            waypoint=waypoint,
            observation=observation,
            semantic_map=semantic_map,
        )
        debug = dict(command.debug_info or {})
        debug["adapter_phase"] = command.phase
        debug["control_command"] = command.to_log_dict()
        debug["controller_feedback"] = feedback.to_log_dict()
        return NavDecision(
            episode_id=episode_id,
            step_id=step_id,
            mode=mode,
            target_type=target_type,
            target_position=tuple(viewpoint.position),
            target_id=getattr(viewpoint, "viewpoint_id", None),
            action=command.action,
            step_size=command.step_size,
            action_source=action_source,
            reason=command.reason,
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
        waypoint = NavigationWaypoint(
            episode_id=episode_id,
            step_id=step_id,
            waypoint_id=str(target_id or "target_waypoint"),
            mode=getattr(mode, "value", str(mode)),
            position=tuple(target_position),
            desired_yaw=current_pose.yaw,
            source=getattr(action_source, "value", str(action_source)),
            target_type=target_type,
            target_id=target_id,
            region_id=target_id,
            arrive_radius=self.config.position_tolerance,
            reason=reason,
            debug_info=debug_info or {},
        )
        command, feedback = self.follow_waypoint(
            current_pose=current_pose,
            waypoint=waypoint,
            observation=observation,
            semantic_map=semantic_map,
        )
        debug = dict(command.debug_info or {})
        debug["adapter_phase"] = command.phase
        debug["control_command"] = command.to_log_dict()
        debug["controller_feedback"] = feedback.to_log_dict()
        decision = NavDecision(
            episode_id=episode_id,
            step_id=step_id,
            mode=mode,
            target_type=target_type,
            target_position=tuple(target_position),
            target_id=target_id,
            candidate_id=candidate_id,
            action=command.action,
            step_size=command.step_size,
            action_source=action_source,
            reason=command.reason,
            debug_info=debug,
        )
        return decision

    def _build_feedback(
        self,
        waypoint: NavigationWaypoint,
        distance_xy: float,
        yaw_to_waypoint_deg: float,
        observe_yaw_error_deg: float,
        command: ControlCommand,
        last_feedback: Optional[ControllerFeedback],
        translation_safe: bool,
    ) -> ControllerFeedback:
        no_progress_steps = 0
        rotate_loop_steps = 0
        progress = 0.0

        if last_feedback is not None and last_feedback.waypoint_id == waypoint.waypoint_id:
            progress = float(last_feedback.distance_to_waypoint) - float(distance_xy)
            if progress <= 0.2:
                no_progress_steps = int(last_feedback.no_progress_steps) + 1
            if command.action in (self.config.rotate_left_action, self.config.rotate_right_action):
                rotate_loop_steps = int(last_feedback.rotate_loop_steps) + 1

        replan_needed = no_progress_steps >= 5 or rotate_loop_steps >= 5
        reason = command.reason
        if replan_needed:
            if rotate_loop_steps >= 5:
                reason = "rotate_loop"
            else:
                reason = "no_progress"

        return ControllerFeedback(
            waypoint_id=waypoint.waypoint_id,
            reached=False,
            distance_to_waypoint=float(distance_xy),
            yaw_to_waypoint=float(yaw_to_waypoint_deg),
            observe_yaw_error=float(observe_yaw_error_deg),
            progress=float(progress),
            no_progress_steps=int(no_progress_steps),
            rotate_loop_steps=int(rotate_loop_steps),
            translation_safe=bool(translation_safe),
            replan_needed=bool(replan_needed),
            reason=reason,
            debug_info={"phase": command.phase},
        )

    def _soft_translation_actions(self, yaw_error_deg: float) -> List[str]:
        if yaw_error_deg > self.config.strafe_yaw_threshold_deg:
            return [self.config.left_action, self.config.forward_action, self.config.right_action]
        if yaw_error_deg < -self.config.strafe_yaw_threshold_deg:
            return [self.config.right_action, self.config.forward_action, self.config.left_action]
        return [self.config.forward_action, self.config.left_action, self.config.right_action]

    def _direct_translation_actions(self, yaw_error_deg: float) -> List[str]:
        if yaw_error_deg > self.config.strafe_yaw_threshold_deg:
            return [self.config.forward_action, self.config.left_action, self.config.right_action]
        if yaw_error_deg < -self.config.strafe_yaw_threshold_deg:
            return [self.config.forward_action, self.config.right_action, self.config.left_action]
        return [self.config.forward_action, self.config.left_action, self.config.right_action]

    def _safe_scan_action(
        self,
        current_pose: PoseRecord,
        observation: Optional[ObservationRecord],
        semantic_map: Optional[SemanticMap],
        waypoint_position: Tuple[float, float, float],
        preferred_actions: List[str],
        yaw_error_deg: float,
    ) -> Tuple[str, float, str]:
        translation_candidates = []
        for action in preferred_actions + [
            self.config.forward_action,
            self.config.left_action,
            self.config.right_action,
        ]:
            if action not in translation_candidates:
                translation_candidates.append(action)

        micro_steps = [
            self.config.translation_step_min,
            max(0.5, self.config.translation_step_min * 0.5),
        ]

        for step_size in micro_steps:
            for action in translation_candidates:
                if self._is_action_safe(
                    action=action,
                    current_pose=current_pose,
                    observation=observation,
                    semantic_map=semantic_map,
                    step_size=step_size,
                    waypoint_position=waypoint_position,
                ):
                    return action, step_size, "micro translation selected after safety check"

        action = self._turn_action(yaw_error_deg)
        return action, self.config.rotation_step_min, "no safe translation; minimal rotate for observation"

    def _translation_step(self, distance: float) -> float:
        return self._clamp(distance, self.config.translation_step_min, self.config.translation_step_max)

    def _soft_translation_step(self, distance: float, abs_yaw_error: float) -> float:
        yaw_factor = max(0.25, 1.0 - abs_yaw_error / 120.0)
        raw_step = min(distance, self.config.translation_step_max) * yaw_factor
        return self._clamp(raw_step, self.config.translation_step_min, min(3.5, self.config.translation_step_max))

    def _rotation_step(self, angle_deg: float) -> float:
        raw_step = max(self.config.rotation_step_min, float(angle_deg) * 0.35)
        return self._clamp(raw_step, self.config.rotation_step_min, self.config.rotation_step_max)

    def _turn_action(self, yaw_error_deg: float) -> str:
        if float(yaw_error_deg) > 0.0:
            return self.config.rotate_right_action
        return self.config.rotate_left_action

    def _is_action_safe(
        self,
        action: str,
        current_pose: PoseRecord,
        observation: Optional[ObservationRecord],
        semantic_map: Optional[SemanticMap],
        step_size: float,
        waypoint_position: Tuple[float, float, float],
    ) -> bool:
        if not self._is_boundary_safe(
            action=action,
            current_pose=current_pose,
            semantic_map=semantic_map,
            step_size=step_size,
            waypoint_position=waypoint_position,
        ):
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
        step_size: float,
        waypoint_position: Tuple[float, float, float],
    ) -> bool:
        if semantic_map is None:
            return True

        if action in (self.config.rotate_left_action, self.config.rotate_right_action):
            return True

        next_pose = self._predict_next_pose(action, current_pose, step_size)
        if semantic_map.is_pose_in_bounds(next_pose):
            return True

        try:
            current_in_bounds = semantic_map.is_pose_in_bounds(current_pose)
        except Exception:
            current_in_bounds = True

        if not current_in_bounds:
            cur_dist = self._distance_xy_pose_to_position(current_pose, waypoint_position)
            next_dist = self._distance_xy_pose_to_position(next_pose, waypoint_position)
            return next_dist < cur_dist - 1e-3

        return False

    def _predict_next_pose(
        self,
        action: str,
        pose: PoseRecord,
        step_size: float,
    ) -> PoseRecord:
        yaw = self._yaw_to_rad(pose.yaw)
        step = float(step_size)
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
            x=float(pose.x) + dx,
            y=float(pose.y) + dy,
            z=float(pose.z),
            yaw=float(pose.yaw),
            pitch=float(pose.pitch),
            roll=float(pose.roll),
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
    def _distance_xy_pose_to_position(
        pose: PoseRecord,
        position: Tuple[float, float, float],
    ) -> float:
        return math.hypot(float(position[0]) - float(pose.x), float(position[1]) - float(pose.y))

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
