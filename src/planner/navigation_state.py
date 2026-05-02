class NavigationState:
    MODE_EXPLORE = "explore"
    MODE_CONFIRM = "confirm"
    MODE_VERIFY = "verify"
    MODE_NAVIGATE = "navigate"
    MODE_RECOVER = "recover"
    MODE_STOP = "stop"

    def __init__(self, recover_steps=3):
        self.mode = self.MODE_EXPLORE
        self.prev_mode = self.MODE_EXPLORE
        self.recover_steps = recover_steps
        self.last_reason = "initialized"
        self.last_update_step = -1
        self.last_tracker_info = {}
        self.last_planner_target = {}

    def reset(self):
        self.mode = self.MODE_EXPLORE
        self.prev_mode = self.MODE_EXPLORE
        self.last_reason = "reset"
        self.last_update_step = -1
        self.last_tracker_info = {}
        self.last_planner_target = {}

    def update(self, tracker_info, step_num=0):
        if tracker_info is None:
            tracker_info = {}

        self.prev_mode = self.mode
        self.last_update_step = step_num
        self.last_tracker_info = tracker_info

        candidate = bool(tracker_info.get("candidate", False))
        geometric_confirmed = bool(
            tracker_info.get("geometric_confirmed", tracker_info.get("confirmed", False))
        )
        verified = bool(tracker_info.get("verified", False))
        confirmed = bool(tracker_info.get("confirmed", False))
        stop_ready = bool(tracker_info.get("stop_ready", False))
        lost_count = int(tracker_info.get("lost_count", 0))
        planner_target = tracker_info.get("planner_target", None)

        if stop_ready and verified:
            self.mode = self.MODE_STOP
            self.last_reason = tracker_info.get("reason", "verified target stop condition satisfied")
        elif verified and confirmed and self.is_valid_planner_target(planner_target):
            self.mode = self.MODE_NAVIGATE
            self.last_reason = tracker_info.get("reason", "target verified")
        elif geometric_confirmed and not verified:
            self.mode = self.MODE_VERIFY
            self.last_reason = tracker_info.get("reason", "target candidate requires verification")
        elif candidate and self.is_valid_planner_target(planner_target):
            self.mode = self.MODE_CONFIRM
            self.last_reason = tracker_info.get("reason", "target candidate detected")
        elif self.prev_mode in [self.MODE_CONFIRM, self.MODE_VERIFY, self.MODE_NAVIGATE] and lost_count > 0 and lost_count <= self.recover_steps:
            self.mode = self.MODE_RECOVER
            self.last_reason = tracker_info.get("reason", "target temporarily lost")
        else:
            self.mode = self.MODE_EXPLORE
            self.last_reason = tracker_info.get("reason", "no reliable target candidate")

        if self.mode in [
            self.MODE_CONFIRM,
            self.MODE_VERIFY,
            self.MODE_NAVIGATE,
            self.MODE_RECOVER,
            self.MODE_STOP
        ]:
            self.last_planner_target = planner_target if planner_target is not None else {}
        else:
            self.last_planner_target = {}

        return self.get_info()

    def is_valid_planner_target(self, planner_target):
        if not isinstance(planner_target, dict):
            return False
        if not planner_target.get("valid", False):
            return False
        if planner_target.get("position", None) is None and planner_target.get("viewpoint_position", None) is None:
            return False
        return True

    def get_info(self):
        return {
            "mode": self.mode,
            "prev_mode": self.prev_mode,
            "changed": self.mode != self.prev_mode,
            "reason": self.last_reason,
            "step_num": self.last_update_step,
            "planner_target": self.last_planner_target,
            "tracker_info": self.last_tracker_info,
        }