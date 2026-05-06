import math


class FinalStopGate:
    def __init__(
        self,
        max_stop_distance=12.0,
        max_estimated_depth=12.0,
        min_verification_confidence=0.55,
        min_detection_score=0.25,
        min_navigate_count=1,
        max_lost_count=0,
        min_area_ratio=0.0005,
        max_area_ratio=0.65,
        max_position_jump=8.0,
        min_stable_count=1,
    ):
        self.max_stop_distance = max_stop_distance
        self.max_estimated_depth = max_estimated_depth
        self.min_verification_confidence = min_verification_confidence
        self.min_detection_score = min_detection_score
        self.min_navigate_count = min_navigate_count
        self.max_lost_count = max_lost_count
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_position_jump = max_position_jump
        self.min_stable_count = min_stable_count
        self.reset()

    def reset(self):
        self.last_target_position = None
        self.stable_count = 0
        self.last_result = {}

    def check(
        self,
        current_pose,
        navigation_info,
        tracker_info,
        verification_info,
        stop_target,
        step_num=None,
    ):
        reasons = []
        warnings = []

        if not isinstance(navigation_info, dict):
            navigation_info = {}

        if not isinstance(tracker_info, dict):
            tracker_info = {}

        if not isinstance(verification_info, dict):
            verification_info = {}

        if not isinstance(stop_target, dict):
            stop_target = {}

        observation = tracker_info.get("observation", {})
        if not isinstance(observation, dict):
            observation = {}

        target_position = self.get_target_position(
            tracker_info=tracker_info,
            stop_target=stop_target,
        )
        stop_distance = self.compute_stop_distance(
            current_pose=current_pose,
            target_position=target_position,
        )
        estimated_depth = self.safe_float(
            observation.get("estimated_depth", None),
            default=None,
        )

        position_stable = self.update_position_stability(target_position)

        if not tracker_info.get("verified", False):
            reasons.append("target tracker is not verified")

        if tracker_info.get("lost_count", 0) > self.max_lost_count:
            reasons.append(
                "verified target is lost: "
                f"lost_count={tracker_info.get('lost_count', 0)}"
            )

        if tracker_info.get("navigate_count", 0) < self.min_navigate_count:
            reasons.append(
                "target has not been approached enough: "
                f"navigate_count={tracker_info.get('navigate_count', 0)}"
            )

        if not self.check_latest_verification(verification_info):
            reasons.append(self.get_verification_reject_reason(verification_info))

        detection_score = self.safe_float(
            observation.get("score", 0.0),
            default=0.0,
        )
        if detection_score < self.min_detection_score:
            reasons.append(
                "latest detection score is too low: "
                f"{round(detection_score, 3)}"
            )

        area_ratio = self.safe_float(
            observation.get("area_ratio", 0.0),
            default=0.0,
        )
        if area_ratio < self.min_area_ratio:
            reasons.append(
                "latest bbox area is too small: "
                f"{round(area_ratio, 5)}"
            )

        if area_ratio > self.max_area_ratio:
            reasons.append(
                "latest bbox area is too large: "
                f"{round(area_ratio, 5)}"
            )

        if stop_distance is not None:
            if stop_distance > self.max_stop_distance:
                reasons.append(
                    "target is still too far: "
                    f"distance={round(stop_distance, 2)}"
                )
        elif estimated_depth is not None:
            if estimated_depth > self.max_estimated_depth:
                reasons.append(
                    "estimated target depth is still too far: "
                    f"depth={round(estimated_depth, 2)}"
                )
        else:
            warnings.append("no geometric distance or depth available")

        if not position_stable:
            reasons.append(
                "target position is not stable: "
                f"stable_count={self.stable_count}"
            )

        allow_stop = len(reasons) == 0

        result = {
            "allow_stop": allow_stop,
            "reasons": reasons,
            "warnings": warnings,
            "step_num": step_num,
            "target_position": target_position,
            "stop_distance": stop_distance,
            "estimated_depth": estimated_depth,
            "position_stable": position_stable,
            "stable_count": self.stable_count,
            "detection_score": detection_score,
            "area_ratio": area_ratio,
            "verification_checked": verification_info.get("checked", False),
            "verification_verified": verification_info.get("verified", False),
            "verification_confidence": verification_info.get("confidence", 0.0),
            "same_object": verification_info.get("same_object", False),
            "hard_reject": verification_info.get("hard_reject", False),
            "navigate_count": tracker_info.get("navigate_count", 0),
            "lost_count": tracker_info.get("lost_count", 0),
            "verified_candidate_id": tracker_info.get("verified_candidate_id", None),
            "selected_candidate_id": verification_info.get("selected_candidate_id", None),
        }

        self.last_result = result
        return result

    def check_latest_verification(self, verification_info):
        if verification_info.get("hard_reject", False):
            return False

        if not verification_info.get("checked", False):
            return False

        if not verification_info.get("verified", False):
            return False

        confidence = self.safe_float(
            verification_info.get("confidence", 0.0),
            default=0.0,
        )
        if confidence < self.min_verification_confidence:
            return False

        if not verification_info.get("same_object", False):
            return False

        return True

    def get_verification_reject_reason(self, verification_info):
        if verification_info.get("hard_reject", False):
            reject_reason = verification_info.get("reject_reason", "")
            if reject_reason:
                return f"latest verification hard rejected: {reject_reason}"
            return "latest verification hard rejected"

        if not verification_info.get("checked", False):
            return "latest verification was not checked"

        if not verification_info.get("verified", False):
            reject_reason = verification_info.get("reject_reason", "")
            if reject_reason:
                return f"latest verification rejected target: {reject_reason}"
            return "latest verification rejected target"

        confidence = self.safe_float(
            verification_info.get("confidence", 0.0),
            default=0.0,
        )
        if confidence < self.min_verification_confidence:
            return (
                "latest verification confidence is too low: "
                f"{round(confidence, 2)}"
            )

        if not verification_info.get("same_object", False):
            return "latest verification does not support same object"

        return "latest verification does not pass final gate"

    def get_target_position(self, tracker_info, stop_target):
        position = tracker_info.get("verified_target_position", None)
        parsed = self.parse_position(position)
        if parsed is not None:
            return parsed

        planner_target = tracker_info.get("planner_target", {})
        if isinstance(planner_target, dict):
            for key in [
                "target_world_position",
                "verified_target_position",
                "position",
                "viewpoint_position",
                "approach_viewpoint",
            ]:
                parsed = self.parse_position(planner_target.get(key, None))
                if parsed is not None:
                    return parsed

        for key in [
            "target_world_position",
            "verified_target_position",
            "position",
            "viewpoint_position",
            "approach_viewpoint",
        ]:
            parsed = self.parse_position(stop_target.get(key, None))
            if parsed is not None:
                return parsed

        return None

    def parse_position(self, position):
        if position is None:
            return None

        if not isinstance(position, (list, tuple)):
            return None

        if len(position) < 2:
            return None

        try:
            x = float(position[0])
            y = float(position[1])
            if len(position) >= 3:
                z = float(position[2])
            else:
                z = 0.0
        except Exception:
            return None

        if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
            return None

        return (x, y, z)

    def compute_stop_distance(self, current_pose, target_position):
        if target_position is None:
            return None

        if current_pose is None or len(current_pose) < 2:
            return None

        try:
            dx = float(current_pose[0]) - float(target_position[0])
            dy = float(current_pose[1]) - float(target_position[1])

            if len(current_pose) >= 3 and len(target_position) >= 3:
                dz = float(current_pose[2]) - float(target_position[2])
            else:
                dz = 0.0

            return math.sqrt(dx * dx + dy * dy + dz * dz)

        except Exception:
            return None

    def update_position_stability(self, target_position):
        if target_position is None:
            self.stable_count = 0
            return False

        if self.last_target_position is None:
            self.last_target_position = target_position
            self.stable_count = 1
            return self.stable_count >= self.min_stable_count

        distance = self.position_distance(
            self.last_target_position,
            target_position,
        )

        if distance <= self.max_position_jump:
            self.stable_count += 1
        else:
            self.stable_count = 1

        self.last_target_position = target_position

        return self.stable_count >= self.min_stable_count

    def position_distance(self, point_a, point_b):
        dx = float(point_a[0]) - float(point_b[0])
        dy = float(point_a[1]) - float(point_b[1])
        dz = float(point_a[2]) - float(point_b[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def safe_float(self, value, default=0.0):
        if value is None:
            return default

        try:
            value = float(value)
        except Exception:
            return default

        if not math.isfinite(value):
            return default

        return value
