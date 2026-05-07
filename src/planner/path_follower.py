import math

try:
    from src.planner.planning_types import PathPlan, PathReason
except Exception:
    from planner.planning_types import PathPlan, PathReason


class PathFollower:
    """
    Convert a planned path into the next UAV-ON executable action.

    This class is only an action adapter for the execution layer. It must not
    select semantic targets, verify objects, stop the episode, or decide planner
    fallback policies. If the path is invalid, it returns a structured invalid
    result and lets the planner/viewpoint layer decide what to do next.
    """

    ACTION_SOURCE = "path_follower"

    def __init__(
        self,
        max_translation_step=5.0,
        max_rotation_step=60.0,
        min_translation_step=0.5,
        arrival_distance=0.8,
        yaw_align_threshold=55.0,
        vertical_threshold=1.5,
    ):
        self.max_translation_step = float(max_translation_step)
        self.max_rotation_step = float(max_rotation_step)
        self.min_translation_step = float(min_translation_step)
        self.arrival_distance = float(arrival_distance)
        self.yaw_align_threshold = float(yaw_align_threshold)
        self.vertical_threshold = float(vertical_threshold)

    def follow(self, current_pose, path_plan):
        """
        Args:
            current_pose:
                [x, y, z, yaw_degree], consistent with ONAir.current_poses.
            path_plan:
                PathPlan or dict returned by LocalPlanner.

        Returns:
            dict with:
                valid: whether an executable action is produced
                action: UAV-ON action name when valid
                step_size: action magnitude when valid
                done: always False here. Stop must be decided by StopGate.
                reason: why action is valid/invalid
                action_source: path_follower
        """
        current_pose = self.normalize_current_pose(current_pose)
        if current_pose is None:
            return self.invalid_result("invalid_current_pose")

        path_plan = self.normalize_path_plan(path_plan)
        if path_plan is None:
            return self.invalid_result(PathReason.UNKNOWN)

        if not path_plan.valid:
            return self.invalid_result(path_plan.reason, path_plan=path_plan)

        waypoint = self.select_next_waypoint(current_pose, path_plan)
        if waypoint is None:
            return self.invalid_result("next_waypoint_is_none", path_plan=path_plan)

        return self.waypoint_to_action(
            current_pose=current_pose,
            waypoint=waypoint,
            path_plan=path_plan,
        )

    def follow_as_tuple(self, current_pose, path_plan):
        """
        Compatibility helper for ON_Air_2.py.

        Returns:
            action, step_size, done, info

        When valid is False, action is None. The caller must not silently convert
        this invalid result into rotate/hold/stop. It should feed the reason back
        to viewpoint reselection or replanning.
        """
        info = self.follow(current_pose=current_pose, path_plan=path_plan)
        if not info.get("valid", False):
            return None, 0.0, False, info

        return (
            info.get("action", None),
            info.get("step_size", 0.0),
            bool(info.get("done", False)),
            info,
        )

    def waypoint_to_action(self, current_pose, waypoint, path_plan=None):
        x, y, z, yaw = current_pose
        wx, wy, wz = waypoint

        dx = float(wx) - float(x)
        dy = float(wy) - float(y)
        dz = float(wz) - float(z)

        horizontal_distance = math.hypot(dx, dy)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance <= self.arrival_distance:
            return self.invalid_result(
                "arrived_at_next_waypoint",
                path_plan=path_plan,
                waypoint=waypoint,
                distance=distance,
            )

        if abs(dz) > self.vertical_threshold and abs(dz) > horizontal_distance:
            if dz < 0:
                action = "ascend"
            else:
                action = "descend"

            step_size = self.bound_translation_step(abs(dz))
            return self.valid_result(
                action=action,
                step_size=step_size,
                reason="vertical_tracking",
                waypoint=waypoint,
                distance=distance,
                horizontal_distance=horizontal_distance,
                yaw_error=0.0,
                path_plan=path_plan,
            )

        if horizontal_distance <= self.arrival_distance:
            return self.invalid_result(
                "arrived_at_horizontal_waypoint",
                path_plan=path_plan,
                waypoint=waypoint,
                distance=distance,
            )

        target_yaw = math.degrees(math.atan2(dy, dx))
        yaw_error = self.normalize_angle(target_yaw - yaw)

        forward_component, right_component = self.world_delta_to_body(
            dx=dx,
            dy=dy,
            yaw_degree=yaw,
        )

        if self.should_rotate_first(
            yaw_error=yaw_error,
            forward_component=forward_component,
            right_component=right_component,
        ):
            if yaw_error > 0:
                action = "rotr"
            else:
                action = "rotl"

            step_size = self.bound_rotation_step(abs(yaw_error))
            return self.valid_result(
                action=action,
                step_size=step_size,
                reason="align_yaw_to_path",
                waypoint=waypoint,
                distance=distance,
                horizontal_distance=horizontal_distance,
                yaw_error=yaw_error,
                path_plan=path_plan,
            )

        if abs(forward_component) >= abs(right_component):
            if forward_component >= 0:
                action = "forward"
                step_size = self.bound_translation_step(forward_component)
                reason = "track_path_forward"
            else:
                if yaw_error > 0:
                    action = "rotr"
                else:
                    action = "rotl"
                step_size = self.bound_rotation_step(abs(yaw_error))
                reason = "turn_to_backward_waypoint"
        else:
            if right_component >= 0:
                action = "right"
            else:
                action = "left"
            step_size = self.bound_translation_step(abs(right_component))
            reason = "track_path_lateral"

        return self.valid_result(
            action=action,
            step_size=step_size,
            reason=reason,
            waypoint=waypoint,
            distance=distance,
            horizontal_distance=horizontal_distance,
            yaw_error=yaw_error,
            path_plan=path_plan,
        )

    def select_next_waypoint(self, current_pose, path_plan):
        if path_plan.next_waypoint is not None:
            return path_plan.next_waypoint

        if path_plan.path is None or len(path_plan.path) == 0:
            return None

        if len(path_plan.path) == 1:
            return path_plan.path[0]

        current_xyz = current_pose[:3]
        for waypoint in path_plan.path[1:]:
            if self.distance_3d(current_xyz, waypoint) > self.arrival_distance:
                return waypoint

        return path_plan.path[-1]

    def should_rotate_first(self, yaw_error, forward_component, right_component):
        if abs(yaw_error) <= self.yaw_align_threshold:
            return False

        if forward_component < 0:
            return True

        if abs(forward_component) >= abs(right_component):
            return True

        return False

    def world_delta_to_body(self, dx, dy, yaw_degree):
        yaw = math.radians(float(yaw_degree))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        forward_component = dx * cos_yaw + dy * sin_yaw
        right_component = -dx * sin_yaw + dy * cos_yaw

        return forward_component, right_component

    def bound_translation_step(self, value):
        value = abs(float(value))
        if value < self.min_translation_step:
            return self.min_translation_step

        return min(value, self.max_translation_step)

    def bound_rotation_step(self, value):
        value = abs(float(value))
        if value < 1.0:
            return 1.0

        return min(value, self.max_rotation_step)

    def normalize_current_pose(self, current_pose):
        if current_pose is None:
            return None

        if not isinstance(current_pose, (list, tuple)):
            return None

        if len(current_pose) < 4:
            return None

        try:
            x = float(current_pose[0])
            y = float(current_pose[1])
            z = float(current_pose[2])
            yaw = float(current_pose[3])
        except (TypeError, ValueError):
            return None

        return (x, y, z, yaw)

    def normalize_path_plan(self, path_plan):
        if isinstance(path_plan, PathPlan):
            return path_plan

        if isinstance(path_plan, dict):
            return PathPlan.from_dict(path_plan)

        return None

    def normalize_angle(self, angle):
        angle = float(angle)
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def distance_3d(self, point_a, point_b):
        ax, ay, az = point_a[:3]
        bx, by, bz = point_b[:3]
        return math.sqrt(
            (float(ax) - float(bx)) ** 2
            + (float(ay) - float(by)) ** 2
            + (float(az) - float(bz)) ** 2
        )

    def valid_result(
        self,
        action,
        step_size,
        reason,
        waypoint,
        distance,
        horizontal_distance,
        yaw_error,
        path_plan=None,
    ):
        result = {
            "valid": True,
            "action": action,
            "step_size": round(float(step_size), 2),
            "done": False,
            "reason": reason,
            "action_source": self.ACTION_SOURCE,
            "next_waypoint": self.round_position(waypoint),
            "distance_to_waypoint": round(float(distance), 2),
            "horizontal_distance_to_waypoint": round(float(horizontal_distance), 2),
            "yaw_error": round(float(yaw_error), 2),
        }

        if path_plan is not None:
            result.update(
                {
                    "path_reason": path_plan.reason,
                    "path_len": int(path_plan.path_len),
                    "path_length": round(float(path_plan.path_length), 2),
                }
            )

        return result

    def invalid_result(
        self,
        reason,
        path_plan=None,
        waypoint=None,
        distance=None,
    ):
        result = {
            "valid": False,
            "action": None,
            "step_size": 0.0,
            "done": False,
            "reason": str(reason),
            "action_source": self.ACTION_SOURCE,
        }

        if waypoint is not None:
            result["next_waypoint"] = self.round_position(waypoint)

        if distance is not None:
            result["distance_to_waypoint"] = round(float(distance), 2)

        if path_plan is not None:
            result.update(
                {
                    "path_reason": path_plan.reason,
                    "path_len": int(path_plan.path_len),
                    "path_length": round(float(path_plan.path_length), 2),
                }
            )

        return result

    def round_position(self, position):
        if position is None:
            return None

        return (
            round(float(position[0]), 2),
            round(float(position[1]), 2),
            round(float(position[2]), 2),
        )
