class NavigationState:
    """
    Minimal AirHunt-localized navigation state.

    Navigation state should describe the high-level phase of the planner:
        SEARCH      : no verified target, use semantic memory/frontier/viewpoint.
        APPROACH    : verified target exists, navigate to approach viewpoint.
        FINAL_CHECK : target is close enough for final evidence checking.
        STOP        : terminal stop state.

    Object verification is not a navigation state. It is represented by
    target_evidence. Recovery is not a navigation state. It is represented by
    planner_policy / planner feedback.
    """

    MODE_SEARCH = "search"
    MODE_APPROACH = "approach"
    MODE_FINAL_CHECK = "final_check"
    MODE_STOP = "stop"

    EVIDENCE_NONE = "none"
    EVIDENCE_CANDIDATE = "candidate"
    EVIDENCE_NEEDS_VERIFICATION = "needs_verification"
    EVIDENCE_VERIFIED = "verified"
    EVIDENCE_LOST = "lost"
    EVIDENCE_REJECTED = "rejected"

    POLICY_NORMAL = "normal"
    POLICY_REPLAN = "replan"
    POLICY_RESELECT_VIEWPOINT = "reselect_viewpoint"
    POLICY_LOWER_TARGET_PRIORITY = "lower_target_priority"

    def __init__(self):
        self.mode = self.MODE_SEARCH
        self.prev_mode = self.MODE_SEARCH
        self.last_reason = "initialized"
        self.last_update_step = -1
        self.last_tracker_info = {}
        self.last_planner_target = {}
        self.last_target_evidence = self.default_target_evidence()
        self.last_planner_policy = self.default_planner_policy()

    def reset(self):
        self.mode = self.MODE_SEARCH
        self.prev_mode = self.MODE_SEARCH
        self.last_reason = "reset"
        self.last_update_step = -1
        self.last_tracker_info = {}
        self.last_planner_target = {}
        self.last_target_evidence = self.default_target_evidence()
        self.last_planner_policy = self.default_planner_policy()

    def update(self, tracker_info, step_num=0, planner_feedback=None):
        if tracker_info is None:
            tracker_info = {}

        if planner_feedback is None:
            planner_feedback = {}

        self.prev_mode = self.mode
        self.last_update_step = step_num
        self.last_tracker_info = tracker_info

        target_evidence = self.build_target_evidence(tracker_info)
        planner_policy = self.build_planner_policy(planner_feedback)

        planner_target = tracker_info.get("planner_target", None)
        has_valid_planner_target = self.is_valid_planner_target(planner_target)

        verified = target_evidence["status"] == self.EVIDENCE_VERIFIED
        final_check_ready = bool(target_evidence.get("stop_ready", False))
        target_lost = target_evidence["status"] == self.EVIDENCE_LOST

        if verified and final_check_ready and has_valid_planner_target:
            self.mode = self.MODE_FINAL_CHECK
            self.last_reason = tracker_info.get(
                "reason",
                "verified target is ready for final check"
            )
        elif verified and has_valid_planner_target:
            self.mode = self.MODE_APPROACH
            self.last_reason = tracker_info.get(
                "reason",
                "verified target object is active"
            )
        else:
            self.mode = self.MODE_SEARCH
            if target_lost:
                self.last_reason = tracker_info.get(
                    "reason",
                    "verified target evidence lost, continue search"
                )
            else:
                self.last_reason = tracker_info.get(
                    "reason",
                    "no verified target object"
                )

        if self.mode in [self.MODE_APPROACH, self.MODE_FINAL_CHECK, self.MODE_STOP]:
            self.last_planner_target = planner_target if planner_target is not None else {}
        else:
            self.last_planner_target = {}

        self.last_target_evidence = target_evidence
        self.last_planner_policy = planner_policy

        return self.get_info()

    def build_target_evidence(self, tracker_info):
        candidate = bool(tracker_info.get("candidate", False))
        confirmed = bool(tracker_info.get("confirmed", False))
        verified = bool(tracker_info.get("verified", False))
        verification_required = bool(tracker_info.get("verification_required", False))
        stop_ready = bool(tracker_info.get("stop_ready", False))
        lost_count = int(tracker_info.get("lost_count", 0))
        reject_reason = tracker_info.get("reject_reason", "")
        verified_target_position = tracker_info.get("verified_target_position", None)

        if verified and confirmed:
            status = self.EVIDENCE_VERIFIED
        elif lost_count > 0 and verified_target_position is not None:
            status = self.EVIDENCE_LOST
        elif verification_required and candidate:
            status = self.EVIDENCE_NEEDS_VERIFICATION
        elif candidate:
            status = self.EVIDENCE_CANDIDATE
        elif reject_reason:
            status = self.EVIDENCE_REJECTED
        else:
            status = self.EVIDENCE_NONE

        return {
            "status": status,
            "candidate": candidate,
            "confirmed": confirmed,
            "verified": verified,
            "verification_required": verification_required,
            "stop_ready": stop_ready,
            "lost_count": lost_count,
            "verified_target_position": verified_target_position,
            "verified_candidate_id": tracker_info.get("verified_candidate_id", None),
            "candidate_id": tracker_info.get("candidate_id", None),
            "confirm_count": tracker_info.get("confirm_count", 0),
            "required_count": tracker_info.get("required_count", 0),
            "navigate_count": tracker_info.get("navigate_count", 0),
            "reason": tracker_info.get("reason", ""),
            "reject_reason": reject_reason,
        }

    def build_planner_policy(self, planner_feedback):
        if not isinstance(planner_feedback, dict):
            planner_feedback = {}

        reason = planner_feedback.get("reason", "")
        replan_required = bool(planner_feedback.get("replan_required", False))
        should_reselect = bool(planner_feedback.get("should_reselect_viewpoint", False))
        should_update_priority = bool(planner_feedback.get("should_update_target_priority", False))

        if should_reselect:
            policy = self.POLICY_RESELECT_VIEWPOINT
        elif should_update_priority:
            policy = self.POLICY_LOWER_TARGET_PRIORITY
        elif replan_required:
            policy = self.POLICY_REPLAN
        else:
            policy = self.POLICY_NORMAL

        return {
            "policy": policy,
            "reason": reason,
            "replan_required": replan_required,
            "should_reselect_viewpoint": should_reselect,
            "should_update_target_priority": should_update_priority,
        }

    def is_valid_planner_target(self, planner_target):
        if not isinstance(planner_target, dict):
            return False

        if not planner_target.get("valid", False):
            return False

        if (
            planner_target.get("position", None) is None
            and planner_target.get("viewpoint_position", None) is None
            and planner_target.get("target_world_position", None) is None
            and planner_target.get("verified_target_position", None) is None
        ):
            return False

        return True

    def default_target_evidence(self):
        return {
            "status": self.EVIDENCE_NONE,
            "candidate": False,
            "confirmed": False,
            "verified": False,
            "verification_required": False,
            "stop_ready": False,
            "lost_count": 0,
            "verified_target_position": None,
            "verified_candidate_id": None,
            "candidate_id": None,
            "confirm_count": 0,
            "required_count": 0,
            "navigate_count": 0,
            "reason": "",
            "reject_reason": "",
        }

    def default_planner_policy(self):
        return {
            "policy": self.POLICY_NORMAL,
            "reason": "",
            "replan_required": False,
            "should_reselect_viewpoint": False,
            "should_update_target_priority": False,
        }

    def get_info(self):
        return {
            "mode": self.mode,
            "prev_mode": self.prev_mode,
            "changed": self.mode != self.prev_mode,
            "reason": self.last_reason,
            "step_num": self.last_update_step,
            "planner_target": self.last_planner_target,
            "tracker_info": self.last_tracker_info,
            "target_evidence": self.last_target_evidence,
            "planner_policy": self.last_planner_policy,
        }
