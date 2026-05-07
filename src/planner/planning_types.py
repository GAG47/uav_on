from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


WorldPosition = Tuple[float, float, float]
GridPosition = Tuple[int, int]


class TargetType:
    FRONTIER = "frontier"
    MEMORY = "memory"
    VERIFIED_OBJECT = "verified_object"
    OBSERVE = "observe"


class ViewpointType:
    EXPLORE = "explore"
    APPROACH = "approach"
    OBSERVE = "observe"


class PathReason:
    OK = "ok"
    INVALID_TARGET = "invalid_target"
    INVALID_VIEWPOINT = "invalid_viewpoint"
    OUTSIDE_MAP = "outside_map"
    OCCUPIED_GOAL = "occupied_goal"
    UNSAFE_GOAL = "unsafe_goal"
    ASTAR_FAILED = "astar_failed"
    PATH_TOO_SHORT = "path_too_short"
    SAME_AS_CURRENT = "same_as_current"
    NO_LOCAL_MAP = "no_local_map"
    UNKNOWN = "unknown"


@dataclass
class NavigationTarget:
    """
    High-level navigation target selected from semantic memory or verified object evidence.

    This object is not an executable control command. It only describes what the planner
    should try to observe or approach. It must be converted into an ExecutableViewpoint
    before being sent to LocalPlanner.
    """
    valid: bool = False
    target_id: str = ""
    target_type: str = TargetType.MEMORY
    source: str = ""
    position: Optional[WorldPosition] = None
    anchor_position: Optional[WorldPosition] = None
    semantic_score: float = 0.0
    confidence: float = 0.0
    preferred_view_distance: float = 8.0
    require_line_of_sight: bool = False
    allow_stop: bool = False
    reason: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "source": self.source,
            "position": self.position,
            "anchor_position": self.anchor_position,
            "semantic_score": self.semantic_score,
            "confidence": self.confidence,
            "preferred_view_distance": self.preferred_view_distance,
            "require_line_of_sight": self.require_line_of_sight,
            "allow_stop": self.allow_stop,
            "reason": self.reason,
            "debug": self.debug,
        }

    @classmethod
    def invalid(cls, reason: str = "invalid_target", source: str = ""):
        return cls(
            valid=False,
            source=source,
            reason=reason,
        )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        if not isinstance(data, dict):
            return cls.invalid(reason="target is not dict")

        return cls(
            valid=bool(data.get("valid", False)),
            target_id=str(data.get("target_id", data.get("id", ""))),
            target_type=str(data.get("target_type", data.get("type", TargetType.MEMORY))),
            source=str(data.get("source", "")),
            position=_normalize_world_position(
                data.get("position", data.get("target_position", data.get("viewpoint_position")))
            ),
            anchor_position=_normalize_world_position(
                data.get("anchor_position", data.get("target_position"))
            ),
            semantic_score=float(data.get("semantic_score", data.get("score", 0.0)) or 0.0),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            preferred_view_distance=float(data.get("preferred_view_distance", 8.0) or 8.0),
            require_line_of_sight=bool(data.get("require_line_of_sight", False)),
            allow_stop=bool(data.get("allow_stop", False)),
            reason=str(data.get("reason", "")),
            debug=dict(data.get("debug", {})) if isinstance(data.get("debug", {}), dict) else {},
        )


@dataclass
class ExecutableViewpoint:
    """
    Concrete viewpoint that can be planned to.

    A NavigationTarget is a semantic anchor. An ExecutableViewpoint is the actual
    geometric goal passed into LocalPlanner.
    """
    valid: bool = False
    viewpoint_id: str = ""
    viewpoint_type: str = ViewpointType.EXPLORE
    position: Optional[WorldPosition] = None
    yaw: Optional[float] = None
    anchor_target_id: str = ""
    anchor_position: Optional[WorldPosition] = None
    expected_observation_direction: str = ""
    safety_score: float = 0.0
    information_gain: float = 0.0
    reason: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "viewpoint_id": self.viewpoint_id,
            "viewpoint_type": self.viewpoint_type,
            "position": self.position,
            "yaw": self.yaw,
            "anchor_target_id": self.anchor_target_id,
            "anchor_position": self.anchor_position,
            "expected_observation_direction": self.expected_observation_direction,
            "safety_score": self.safety_score,
            "information_gain": self.information_gain,
            "reason": self.reason,
            "debug": self.debug,
        }

    @classmethod
    def invalid(cls, reason: str = "invalid_viewpoint", anchor_target_id: str = ""):
        return cls(
            valid=False,
            anchor_target_id=anchor_target_id,
            reason=reason,
        )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        if not isinstance(data, dict):
            return cls.invalid(reason="viewpoint is not dict")

        return cls(
            valid=bool(data.get("valid", False)),
            viewpoint_id=str(data.get("viewpoint_id", data.get("id", ""))),
            viewpoint_type=str(data.get("viewpoint_type", data.get("type", ViewpointType.EXPLORE))),
            position=_normalize_world_position(
                data.get("position", data.get("viewpoint_position"))
            ),
            yaw=_normalize_float(data.get("yaw", data.get("viewpoint_yaw"))),
            anchor_target_id=str(data.get("anchor_target_id", data.get("target_id", ""))),
            anchor_position=_normalize_world_position(
                data.get("anchor_position", data.get("target_position"))
            ),
            expected_observation_direction=str(data.get("expected_observation_direction", "")),
            safety_score=float(data.get("safety_score", data.get("safety", 0.0)) or 0.0),
            information_gain=float(data.get("information_gain", data.get("viewpoint_score", 0.0)) or 0.0),
            reason=str(data.get("reason", "")),
            debug=dict(data.get("debug", {})) if isinstance(data.get("debug", {}), dict) else {},
        )


@dataclass
class PathPlan:
    """
    Structured local planning result.

    LocalPlanner should only describe whether a path exists and what the path is.
    It must not decide semantic policy, stop, or fallback actions.
    """
    valid: bool = False
    reason: str = PathReason.UNKNOWN
    path: List[WorldPosition] = field(default_factory=list)
    grid_path: List[GridPosition] = field(default_factory=list)
    path_len: int = 0
    raw_grid_path_len: int = 0
    path_length: float = 0.0
    start_grid: Optional[GridPosition] = None
    goal_grid: Optional[GridPosition] = None
    target_position: Optional[WorldPosition] = None
    next_waypoint: Optional[WorldPosition] = None
    viewpoint: Optional[ExecutableViewpoint] = None
    debug: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.path_len <= 0 and self.path is not None:
            self.path_len = len(self.path)

        if self.next_waypoint is None:
            self.next_waypoint = self.get_next_waypoint()

    def get_next_waypoint(self) -> Optional[WorldPosition]:
        if not self.valid or len(self.path) == 0:
            return None

        if len(self.path) >= 2:
            return self.path[1]

        return self.path[0]

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "valid": self.valid,
            "reason": self.reason,
            "path": self.path,
            "grid_path": self.grid_path,
            "path_len": self.path_len,
            "raw_grid_path_len": self.raw_grid_path_len,
            "path_length": self.path_length,
            "start_grid": self.start_grid,
            "goal_grid": self.goal_grid,
            "target_position": self.target_position,
            "next_waypoint": self.next_waypoint,
            "debug": self.debug,
        }

        if self.viewpoint is not None:
            data["viewpoint"] = self.viewpoint.to_dict()

        return data

    @classmethod
    def invalid(cls, reason: str = PathReason.UNKNOWN, debug: Optional[Dict[str, Any]] = None):
        return cls(
            valid=False,
            reason=reason,
            path=[],
            grid_path=[],
            path_len=0,
            raw_grid_path_len=0,
            path_length=0.0,
            debug=debug or {},
        )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        if not isinstance(data, dict):
            return cls.invalid(reason="plan is not dict")

        path = _normalize_world_path(data.get("path", []))
        grid_path = _normalize_grid_path(data.get("grid_path", []))
        viewpoint_data = data.get("viewpoint", None)

        return cls(
            valid=bool(data.get("valid", False)),
            reason=str(data.get("reason", PathReason.UNKNOWN)),
            path=path,
            grid_path=grid_path,
            path_len=int(data.get("path_len", len(path)) or 0),
            raw_grid_path_len=int(data.get("raw_grid_path_len", len(grid_path)) or 0),
            path_length=float(data.get("path_length", 0.0) or 0.0),
            start_grid=_normalize_grid_position(data.get("start_grid", None)),
            goal_grid=_normalize_grid_position(data.get("goal_grid", None)),
            target_position=_normalize_world_position(data.get("target_position", None)),
            next_waypoint=_normalize_world_position(data.get("next_waypoint", None)),
            viewpoint=ExecutableViewpoint.from_dict(viewpoint_data) if isinstance(viewpoint_data, dict) else None,
            debug=dict(data.get("debug", {})) if isinstance(data.get("debug", {}), dict) else {},
        )


@dataclass
class PlannerFeedback:
    """
    Feedback passed back to target/viewpoint selection.

    This feedback must not be directly converted into an action. It only describes why
    the current target/viewpoint/path is valid or invalid.
    """
    valid: bool = False
    reason: str = PathReason.UNKNOWN
    target_id: str = ""
    target_type: str = ""
    viewpoint_id: str = ""
    viewpoint_type: str = ""
    path_len: int = 0
    path_length: float = 0.0
    action_source: str = ""
    replan_required: bool = False
    should_reselect_viewpoint: bool = False
    should_update_target_priority: bool = False
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "viewpoint_id": self.viewpoint_id,
            "viewpoint_type": self.viewpoint_type,
            "path_len": self.path_len,
            "path_length": self.path_length,
            "action_source": self.action_source,
            "replan_required": self.replan_required,
            "should_reselect_viewpoint": self.should_reselect_viewpoint,
            "should_update_target_priority": self.should_update_target_priority,
            "debug": self.debug,
        }

    @classmethod
    def from_path_plan(
        cls,
        path_plan: Optional[PathPlan],
        target: Optional[NavigationTarget] = None,
        viewpoint: Optional[ExecutableViewpoint] = None,
        action_source: str = "",
    ):
        if path_plan is None:
            path_plan = PathPlan.invalid(reason=PathReason.UNKNOWN)

        reason = path_plan.reason
        valid = bool(path_plan.valid)

        reselect_reasons = {
            PathReason.INVALID_TARGET,
            PathReason.INVALID_VIEWPOINT,
            PathReason.OUTSIDE_MAP,
            PathReason.OCCUPIED_GOAL,
            PathReason.UNSAFE_GOAL,
            PathReason.ASTAR_FAILED,
            PathReason.PATH_TOO_SHORT,
            PathReason.SAME_AS_CURRENT,
            PathReason.NO_LOCAL_MAP,
        }

        target_id = target.target_id if target is not None else ""
        target_type = target.target_type if target is not None else ""
        viewpoint_id = viewpoint.viewpoint_id if viewpoint is not None else ""
        viewpoint_type = viewpoint.viewpoint_type if viewpoint is not None else ""

        return cls(
            valid=valid,
            reason=reason,
            target_id=target_id,
            target_type=target_type,
            viewpoint_id=viewpoint_id,
            viewpoint_type=viewpoint_type,
            path_len=path_plan.path_len,
            path_length=path_plan.path_length,
            action_source=action_source,
            replan_required=(not valid),
            should_reselect_viewpoint=reason in reselect_reasons,
            should_update_target_priority=reason in {
                PathReason.OUTSIDE_MAP,
                PathReason.OCCUPIED_GOAL,
                PathReason.UNSAFE_GOAL,
                PathReason.ASTAR_FAILED,
            },
            debug=dict(path_plan.debug),
        )


def _normalize_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_world_position(value: Any) -> Optional[WorldPosition]:
    if value is None:
        return None

    if not isinstance(value, (list, tuple)):
        return None

    if len(value) < 2:
        return None

    try:
        x = float(value[0])
        y = float(value[1])
        z = float(value[2]) if len(value) >= 3 else 0.0
    except (TypeError, ValueError):
        return None

    return (x, y, z)


def _normalize_grid_position(value: Any) -> Optional[GridPosition]:
    if value is None:
        return None

    if not isinstance(value, (list, tuple)):
        return None

    if len(value) < 2:
        return None

    try:
        x = int(value[0])
        y = int(value[1])
    except (TypeError, ValueError):
        return None

    return (x, y)


def _normalize_world_path(value: Any) -> List[WorldPosition]:
    if not isinstance(value, list):
        return []

    path = []
    for item in value:
        position = _normalize_world_position(item)
        if position is not None:
            path.append(position)

    return path


def _normalize_grid_path(value: Any) -> List[GridPosition]:
    if not isinstance(value, list):
        return []

    path = []
    for item in value:
        position = _normalize_grid_position(item)
        if position is not None:
            path.append(position)

    return path
