import math


class FinalStopGate:
    """
    Final stop decision gate.

    This is the only module allowed to approve a stop action. It does not use
    detector-specific special cases such as bbox size, camera index, or score
    hacks. It only checks abstract navigation evidence:
        - verified target evidence
        - approach / final-check phase
        - latest observation still supports the target
        - distance satisfies UAV-ON success threshold
    """

    def __init__(
        self,
        success_distance=20.0,
        hard_far_distance=25.0,
        min_navigate_count=0,
    ):
        self.success_distance = float(success_distance)
        self.hard_far_distance = float(hard_far_distance)
        self.min_navigate_count = int(min_navigate_count)

    def evaluate(
        self,
        current_pose,
        navigation_info,
        tracker_info,
        stop_candidate,
    ):
        current_position = self.normalize_position(current_pose)
        if current_position is None:
            return self.reject("invalid_current_pose")

        if not isinstance(navigation_info, dict):
            navigation_info = {}

        if not isinstance(tracker_info, dict):
            tracker_info = {}

        if not isinstance(stop_candidate, dict):
            stop_candidate = {}

        mode = navigation_info.get("mode", "")
        target_evidence = navigation_info.get("target_evidence", {})
        if not isinstance(target_evidence, dict):
            target_evidence = {}

        if mode not in ["final_check", "stop"]:
            return self.reject(
                "not_in_final_check",
                mode=mode,
            )

        evidence_status = target_evidence.get("status", "none")
        evidence_verified = bool(target_evidence.get("verified", False))
        tracker_verified = bool(tracker_info.get("verified", False))
        tracker_confirmed = bool(tracker_info.get("confirmed", False))

        if evidence_status != "verified" and not evidence_verified and not tracker_verified:
            return self.reject(
                "target_not_verified",
                mode=mode,
                evidence_status=evidence_status,
            )

        if not self.has_latest_support(target_evidence, tracker_info):
            return self.reject(
                "latest_evidence_does_not_support_target",
                mode=mode,
                evidence_status=evidence_status,
                tracker_verified=tracker_verified,
                tracker_confirmed=tracker_confirmed,
            )

        navigate_count = int(tracker_info.get("navigate_count", 0) or 0)
        if navigate_count < self.min_navigate_count:
            return self.reject(
                "approach_progress_insufficient",
                mode=mode,
                navigate_count=navigate_count,
                min_navigate_count=self.min_navigate_count,
            )

        target_position = self.extract_target_position(
            stop_candidate=stop_candidate,
            target_evidence=target_evidence,
            tracker_info=tracker_info,
        )
        if target_position is None:
            return self.reject(
                "target_position_missing",
                mode=mode,
                evidence_status=evidence_status,
            )

        distance = self.distance_3d(current_position, target_position)

        if distance > self.hard_far_distance:
            return self.reject(
                "target_too_far",
                mode=mode,
                distance=round(distance, 2),
                threshold=self.hard_far_distance,
                target_position=target_position,
            )

        if distance > self.success_distance:
            return self.reject(
                "outside_success_threshold",
                mode=mode,
                distance=round(distance, 2),
                threshold=self.success_distance,
                target_position=target_position,
            )

        return {
            "pass": True,
            "reason": "final_stop_gate_passed",
            "mode": mode,
            "distance": round(distance, 2),
            "threshold": self.success_distance,
            "target_position": target_position,
            "evidence_status": evidence_status,
            "tracker_verified": tracker_verified,
            "tracker_confirmed": tracker_confirmed,
            "navigate_count": navigate_count,
        }

    def has_latest_support(self, target_evidence, tracker_info):
        evidence_verified = bool(target_evidence.get("verified", False))
        evidence_stop_ready = bool(target_evidence.get("stop_ready", False))
        tracker_verified = bool(tracker_info.get("verified", False))
        tracker_confirmed = bool(tracker_info.get("confirmed", False))
        tracker_stop_ready = bool(tracker_info.get("stop_ready", False))

        if evidence_verified and evidence_stop_ready:
            return True

        if tracker_verified and tracker_confirmed:
            return True

        if tracker_verified and tracker_stop_ready:
            return True

        return False

    def extract_target_position(self, stop_candidate, target_evidence, tracker_info):
        sources = [
            stop_candidate,
            target_evidence,
            tracker_info,
        ]

        keys = [
            "target_world_position",
            "verified_target_position",
            "target_position",
            "anchor_position",
        ]

        for source in sources:
            if not isinstance(source, dict):
                continue

            for key in keys:
                position = self.normalize_position(source.get(key, None))
                if position is not None:
                    return position

            planner_target = source.get("planner_target", None)
            if isinstance(planner_target, dict):
                for key in keys + ["position", "viewpoint_position"]:
                    position = self.normalize_position(planner_target.get(key, None))
                    if position is not None:
                        return position

        return None

    def normalize_position(self, position):
        if position is None:
            return None

        if not isinstance(position, (list, tuple)):
            return None

        if len(position) < 2:
            return None

        try:
            x = float(position[0])
            y = float(position[1])
            z = float(position[2]) if len(position) >= 3 else 0.0
        except Exception:
            return None

        return (round(x, 2), round(y, 2), round(z, 2))

    def distance_3d(self, point_a, point_b):
        return math.sqrt(
            (float(point_a[0]) - float(point_b[0])) ** 2
            + (float(point_a[1]) - float(point_b[1])) ** 2
            + (float(point_a[2]) - float(point_b[2])) ** 2
        )

    def reject(self, reason, **kwargs):
        result = {
            "pass": False,
            "reason": reason,
        }
        result.update(kwargs)
        return result
