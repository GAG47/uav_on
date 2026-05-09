from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple


def new_id(prefix: str) -> str:
    return "{}_{}".format(prefix, uuid.uuid4().hex[:12])


def now_ts() -> float:
    return time.time()


def clamp01(value: float) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def strip_data_uri(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    if "," in value and value.strip().lower().startswith("data:"):
        return value.split(",", 1)[1]
    return value


def to_data_uri(b64_value: Optional[str], mime: str = "image/png") -> Optional[str]:
    if not b64_value:
        return None
    if b64_value.strip().lower().startswith("data:"):
        return b64_value
    return "data:{};base64,{}".format(mime, b64_value)


def bytes_to_b64(raw: Optional[bytes]) -> Optional[str]:
    if raw is None:
        return None
    return base64.b64encode(raw).decode("utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    """
    Convert runtime objects into log-safe values.

    Important:
    - bytes are not dumped.
    - numpy arrays / tensor-like objects are summarized, not serialized.
    - Enum values are exported as strings.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return "<bytes:{}>".format(len(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    shape = getattr(value, "shape", None)
    if shape is not None:
        return "<array shape={}>".format(tuple(shape))

    if hasattr(value, "to_log_dict"):
        return value.to_log_dict()

    if hasattr(value, "__dict__"):
        return {k: _json_safe(v) for k, v in value.__dict__.items()}

    return str(value)


class ViewID(str, Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    DOWN = "down"

    @staticmethod
    def from_any(value: Any) -> "ViewID":
        if isinstance(value, ViewID):
            return value

        if isinstance(value, int):
            mapping = {
                0: ViewID.FRONT,
                1: ViewID.LEFT,
                2: ViewID.RIGHT,
                3: ViewID.DOWN,
            }
            if value in mapping:
                return mapping[value]

        text = str(value).lower().strip()
        aliases = {
            "0": ViewID.FRONT,
            "front": ViewID.FRONT,
            "forward": ViewID.FRONT,
            "1": ViewID.LEFT,
            "left": ViewID.LEFT,
            "2": ViewID.RIGHT,
            "right": ViewID.RIGHT,
            "3": ViewID.DOWN,
            "down": ViewID.DOWN,
            "bottom": ViewID.DOWN,
        }
        if text not in aliases:
            raise ValueError("Unsupported view id: {}".format(value))
        return aliases[text]

    @property
    def index(self) -> int:
        return {
            ViewID.FRONT: 0,
            ViewID.LEFT: 1,
            ViewID.RIGHT: 2,
            ViewID.DOWN: 3,
        }[self]


class NavMode(str, Enum):
    SEARCH = "search"
    CHECK_CANDIDATE = "check_candidate"
    APPROACH = "approach"
    FINAL_CHECK = "final_check"
    STOP = "stop"


class CandidateStatus(str, Enum):
    NEW = "new"
    TENTATIVE = "tentative"
    VERIFIED = "verified"
    REJECTED = "rejected"
    LOST = "lost"


class Task2Decision(str, Enum):
    YES = "yes"
    MAYBE = "maybe"
    NO = "no"
    UNKNOWN = "unknown"

    @staticmethod
    def from_any(value: Any) -> "Task2Decision":
        if isinstance(value, Task2Decision):
            return value
        text = str(value).lower().strip()
        if text in ("yes", "true", "match", "matched", "target", "positive"):
            return Task2Decision.YES
        if text in ("maybe", "uncertain", "possible", "weak"):
            return Task2Decision.MAYBE
        if text in ("no", "false", "not", "negative", "reject", "rejected"):
            return Task2Decision.NO
        return Task2Decision.UNKNOWN


class ActionSource(str, Enum):
    SEMANTIC_MAP = "semantic_map"
    GEOMETRIC_EXPLORE = "geometric_explore"
    CANDIDATE_TARGET = "candidate_target"
    VERIFIED_TARGET = "verified_target"
    STOP_GATE = "stop_gate"
    SAFETY_HOLD = "safety_hold"
    BASELINE = "baseline"


class AsyncTaskType(str, Enum):
    TASK1 = "task1"
    GDINO = "gdino"
    TASK2 = "task2"


@dataclass
class PoseRecord:
    x: float
    y: float
    z: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    quaternion: Optional[Tuple[float, float, float, float]] = None
    timestamp: float = field(default_factory=now_ts)

    def xyz(self) -> Tuple[float, float, float]:
        return float(self.x), float(self.y), float(self.z)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "yaw": float(self.yaw),
            "pitch": float(self.pitch),
            "roll": float(self.roll),
            "quaternion": _json_safe(self.quaternion),
            "timestamp": float(self.timestamp),
        }


@dataclass
class TargetInfo:
    name: str
    size: Optional[str] = None
    description: Optional[str] = None
    instruction: Optional[str] = None
    search_radius: float = 50.0
    success_threshold: float = 20.0

    @property
    def text(self) -> str:
        parts = []
        if self.name:
            parts.append("Name: {}".format(self.name))
        if self.size:
            parts.append("Size: {}".format(self.size))
        if self.description:
            parts.append("Description: {}".format(self.description))
        if self.instruction:
            parts.append("Instruction: {}".format(self.instruction))
        return "; ".join(parts)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "description": self.description,
            "instruction": self.instruction,
            "search_radius": self.search_radius,
            "success_threshold": self.success_threshold,
        }


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int
    normalized: bool = False

    def __post_init__(self) -> None:
        self.x1 = safe_float(self.x1)
        self.y1 = safe_float(self.y1)
        self.x2 = safe_float(self.x2)
        self.y2 = safe_float(self.y2)
        self.image_width = int(self.image_width)
        self.image_height = int(self.image_height)

    def denormalized(self) -> "BBox":
        if not self.normalized:
            return self
        return BBox(
            x1=self.x1 * self.image_width,
            y1=self.y1 * self.image_height,
            x2=self.x2 * self.image_width,
            y2=self.y2 * self.image_height,
            image_width=self.image_width,
            image_height=self.image_height,
            normalized=False,
        )

    def clipped(self) -> "BBox":
        box = self.denormalized()
        x1 = max(0.0, min(float(box.image_width), box.x1))
        x2 = max(0.0, min(float(box.image_width), box.x2))
        y1 = max(0.0, min(float(box.image_height), box.y1))
        y2 = max(0.0, min(float(box.image_height), box.y2))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        return BBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            image_width=box.image_width,
            image_height=box.image_height,
            normalized=False,
        )

    @property
    def width(self) -> float:
        box = self.clipped()
        return max(0.0, box.x2 - box.x1)

    @property
    def height(self) -> float:
        box = self.clipped()
        return max(0.0, box.y2 - box.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def image_area(self) -> float:
        return max(1.0, float(self.image_width * self.image_height))

    @property
    def area_ratio(self) -> float:
        return float(self.area / self.image_area)

    @property
    def center(self) -> Tuple[float, float]:
        box = self.clipped()
        return (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0

    def to_xyxy_int(self) -> Tuple[int, int, int, int]:
        box = self.clipped()
        return int(round(box.x1)), int(round(box.y1)), int(round(box.x2)), int(round(box.y2))

    def is_valid(
        self,
        min_area_ratio: float = 1e-4,
        max_area_ratio: float = 0.75,
        min_side: int = 3,
    ) -> bool:
        if self.image_width <= 0 or self.image_height <= 0:
            return False
        if self.width < min_side or self.height < min_side:
            return False
        if self.area_ratio < min_area_ratio:
            return False
        if self.area_ratio > max_area_ratio:
            return False
        return True

    def is_near_border(self, margin_ratio: float = 0.02) -> bool:
        box = self.clipped()
        mx = self.image_width * margin_ratio
        my = self.image_height * margin_ratio
        return (
            box.x1 <= mx
            or box.y1 <= my
            or box.x2 >= self.image_width - mx
            or box.y2 >= self.image_height - my
        )

    def to_log_dict(self) -> Dict[str, Any]:
        box = self.clipped()
        return {
            "x1": box.x1,
            "y1": box.y1,
            "x2": box.x2,
            "y2": box.y2,
            "image_width": box.image_width,
            "image_height": box.image_height,
            "normalized": False,
            "area_ratio": box.area_ratio,
            "center": _json_safe(box.center),
        }


@dataclass
class FrameRecord:
    """
    One view at one step.

    UAV-ON baseline sends RGB images to DashScope/Qwen as base64 data URIs.
    This record keeps both runtime image bytes and base64 strings so Task1,
    GDINO and Task2 can share the same observation safely.
    """

    episode_id: str
    step_id: int
    view_id: ViewID
    pose: PoseRecord
    target_info: TargetInfo

    rgb_bytes: Optional[bytes] = None
    rgb_b64: Optional[str] = None
    rgb_mime: str = "image/png"
    rgb_caption: Optional[str] = None

    depth: Any = None
    depth_grid3x3: Optional[List[List[float]]] = None

    image_width: Optional[int] = None
    image_height: Optional[int] = None

    timestamp: float = field(default_factory=now_ts)
    frame_id: str = field(default_factory=lambda: new_id("frame"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.view_id = ViewID.from_any(self.view_id)
        self.rgb_b64 = strip_data_uri(self.rgb_b64)
        if self.rgb_b64 is None and self.rgb_bytes is not None:
            self.rgb_b64 = bytes_to_b64(self.rgb_bytes)

    @property
    def rgb_data_uri(self) -> Optional[str]:
        return to_data_uri(self.rgb_b64, self.rgb_mime)

    def has_image(self) -> bool:
        return bool(self.rgb_bytes or self.rgb_b64)

    def has_depth(self) -> bool:
        return self.depth is not None or self.depth_grid3x3 is not None

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "view_id": self.view_id.value,
            "pose": self.pose.to_log_dict(),
            "target_info": self.target_info.to_log_dict(),
            "has_rgb_bytes": self.rgb_bytes is not None,
            "has_rgb_b64": self.rgb_b64 is not None,
            "rgb_mime": self.rgb_mime,
            "rgb_caption": self.rgb_caption,
            "has_depth": self.has_depth(),
            "depth_grid3x3": _json_safe(self.depth_grid3x3),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "timestamp": self.timestamp,
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class ObservationRecord:
    """
    One complete UAV-ON step observation with synchronized multi-view frames.
    """

    episode_id: str
    step_id: int
    pose: PoseRecord
    target_info: TargetInfo
    frames: Dict[ViewID, FrameRecord] = field(default_factory=dict)
    timestamp: float = field(default_factory=now_ts)
    observation_id: str = field(default_factory=lambda: new_id("obs"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: Dict[ViewID, FrameRecord] = {}
        for key, frame in self.frames.items():
            view_id = ViewID.from_any(key)
            frame.view_id = view_id
            normalized[view_id] = frame
        self.frames = normalized

    def add_frame(self, frame: FrameRecord) -> None:
        self.frames[ViewID.from_any(frame.view_id)] = frame

    def get_frame(self, view_id: Any) -> Optional[FrameRecord]:
        return self.frames.get(ViewID.from_any(view_id))

    def iter_frames(self) -> Iterable[FrameRecord]:
        for view_id in (ViewID.FRONT, ViewID.LEFT, ViewID.RIGHT, ViewID.DOWN):
            frame = self.frames.get(view_id)
            if frame is not None:
                yield frame

    def frame_list(self) -> List[FrameRecord]:
        return list(self.iter_frames())

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "pose": self.pose.to_log_dict(),
            "target_info": self.target_info.to_log_dict(),
            "frames": {k.value: v.to_log_dict() for k, v in self.frames.items()},
            "timestamp": self.timestamp,
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class RegionScore:
    """
    Task1 output for one keyframe/view/visible region.
    """

    request_id: str
    episode_id: str
    step_id: int
    view_id: ViewID
    frame_id: str
    semantic_value: float
    confidence: float = 1.0
    reason: str = ""

    region_center_world: Optional[Tuple[float, float, float]] = None
    visible_grid_cells: Optional[List[Tuple[int, int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.view_id = ViewID.from_any(self.view_id)
        self.semantic_value = clamp01(self.semantic_value)
        self.confidence = clamp01(self.confidence)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "view_id": self.view_id.value,
            "frame_id": self.frame_id,
            "semantic_value": self.semantic_value,
            "confidence": self.confidence,
            "reason": self.reason,
            "region_center_world": _json_safe(self.region_center_world),
            "visible_grid_cells": _json_safe(self.visible_grid_cells),
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class Task1Request:
    request_id: str
    episode_id: str
    submit_step: int
    target_info: TargetInfo
    frames: List[FrameRecord]
    created_at: float = field(default_factory=now_ts)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        episode_id: str,
        submit_step: int,
        target_info: TargetInfo,
        frames: List[FrameRecord],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Task1Request":
        return Task1Request(
            request_id=new_id("task1"),
            episode_id=episode_id,
            submit_step=submit_step,
            target_info=target_info,
            frames=frames,
            metadata=metadata or {},
        )

    @property
    def frame_ids(self) -> List[str]:
        return [frame.frame_id for frame in self.frames]

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "submit_step": self.submit_step,
            "target_info": self.target_info.to_log_dict(),
            "frame_ids": self.frame_ids,
            "frames": [frame.to_log_dict() for frame in self.frames],
            "created_at": self.created_at,
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class Task1Result:
    request_id: str
    episode_id: str
    submit_step: int
    return_step: int
    scores: List[RegionScore] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    raw_response: Optional[str] = None
    latency_ms: Optional[float] = None
    returned_at: float = field(default_factory=now_ts)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "submit_step": self.submit_step,
            "return_step": self.return_step,
            "scores": [score.to_log_dict() for score in self.scores],
            "success": self.success,
            "error": self.error,
            "raw_response": self.raw_response,
            "latency_ms": self.latency_ms,
            "returned_at": self.returned_at,
            "metadata": _json_safe(self.metadata),
        }



@dataclass
class GDINORequest:
    """
    Request built from selected SVNav GDINO keyframes.

    A GDINORequest is not a target confirmation request. It only asks the
    GroundingDINO server to produce open-vocabulary detection candidates for
    selected FrameRecord images.

    Every request keeps the source frames and metadata snapshot so asynchronous
    GDINO results can be safely traced back to the original step, pose, view,
    semantic context, and navigation context.
    """

    request_id: str
    episode_id: str
    submit_step: int
    target_info: TargetInfo
    frames: List[FrameRecord]
    prompt: Optional[str] = None
    created_at: float = field(default_factory=now_ts)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        episode_id: str,
        submit_step: int,
        target_info: TargetInfo,
        frames: List[FrameRecord],
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "GDINORequest":
        return GDINORequest(
            request_id=new_id("gdino"),
            episode_id=str(episode_id),
            submit_step=int(submit_step),
            target_info=target_info,
            frames=list(frames or []),
            prompt=prompt,
            metadata=metadata or {},
        )

    @property
    def frame_ids(self) -> List[str]:
        return [frame.frame_id for frame in self.frames]

    @property
    def view_ids(self) -> List[str]:
        return [frame.view_id.value for frame in self.frames]

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "submit_step": int(self.submit_step),
            "target_info": self.target_info.to_log_dict(),
            "frame_ids": self.frame_ids,
            "view_ids": self.view_ids,
            "prompt": self.prompt,
            "frames": [frame.to_log_dict() for frame in self.frames],
            "created_at": float(self.created_at),
            "metadata": _json_safe(self.metadata),
        }

@dataclass
class GDINOCandidate:
    """
    One object candidate proposed by GroundingDINO.

    This is not a confirmed target.
    It can only be sent to Task2 or fused into TargetEvidence.
    """

    candidate_id: str
    episode_id: str
    step_id: int
    view_id: ViewID
    frame_id: str
    bbox: BBox
    label: str
    score: float

    prompt: Optional[str] = None
    frame: Optional[FrameRecord] = None

    crop_bytes: Optional[bytes] = None
    crop_b64: Optional[str] = None
    crop_mime: str = "image/png"
    crop_path: Optional[str] = None

    position_3d: Optional[Tuple[float, float, float]] = None
    position_confidence: float = 0.0
    depth_valid: bool = False
    quality_score: float = 0.0
    rejected_reason: Optional[str] = None

    timestamp: float = field(default_factory=now_ts)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.view_id = ViewID.from_any(self.view_id)
        self.score = clamp01(self.score)
        self.position_confidence = clamp01(self.position_confidence)
        self.quality_score = clamp01(self.quality_score)
        self.crop_b64 = strip_data_uri(self.crop_b64)
        if self.crop_b64 is None and self.crop_bytes is not None:
            self.crop_b64 = bytes_to_b64(self.crop_bytes)

    @property
    def crop_data_uri(self) -> Optional[str]:
        return to_data_uri(self.crop_b64, self.crop_mime)

    @property
    def full_image_data_uri(self) -> Optional[str]:
        if self.frame is None:
            return None
        return self.frame.rgb_data_uri

    def has_valid_position(self) -> bool:
        return self.position_3d is not None and self.position_confidence > 0.0

    def is_usable_for_task2(self) -> bool:
        if self.rejected_reason:
            return False
        if not self.bbox.is_valid():
            return False
        if self.score <= 0.0:
            return False
        if not self.crop_b64 and not self.crop_path:
            return False
        return True

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "view_id": self.view_id.value,
            "frame_id": self.frame_id,
            "bbox": self.bbox.to_log_dict(),
            "label": self.label,
            "score": self.score,
            "prompt": self.prompt,
            "has_frame": self.frame is not None,
            "has_crop_bytes": self.crop_bytes is not None,
            "has_crop_b64": self.crop_b64 is not None,
            "crop_path": self.crop_path,
            "position_3d": _json_safe(self.position_3d),
            "position_confidence": self.position_confidence,
            "depth_valid": self.depth_valid,
            "quality_score": self.quality_score,
            "rejected_reason": self.rejected_reason,
            "timestamp": self.timestamp,
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class GDINOResult:
    request_id: str
    episode_id: str
    submit_step: int
    return_step: int
    candidates: List[GDINOCandidate] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    returned_at: float = field(default_factory=now_ts)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "submit_step": self.submit_step,
            "return_step": self.return_step,
            "candidates": [c.to_log_dict() for c in self.candidates],
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "returned_at": self.returned_at,
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class Task2Request:
    request_id: str
    episode_id: str
    submit_step: int
    target_info: TargetInfo
    candidate: GDINOCandidate
    context_frames: List[FrameRecord] = field(default_factory=list)
    created_at: float = field(default_factory=now_ts)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        episode_id: str,
        submit_step: int,
        target_info: TargetInfo,
        candidate: GDINOCandidate,
        context_frames: Optional[List[FrameRecord]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Task2Request":
        return Task2Request(
            request_id=new_id("task2"),
            episode_id=episode_id,
            submit_step=submit_step,
            target_info=target_info,
            candidate=candidate,
            context_frames=context_frames or [],
            metadata=metadata or {},
        )

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "submit_step": self.submit_step,
            "target_info": self.target_info.to_log_dict(),
            "candidate": self.candidate.to_log_dict(),
            "context_frame_ids": [f.frame_id for f in self.context_frames],
            "context_frames": [f.to_log_dict() for f in self.context_frames],
            "created_at": self.created_at,
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class Task2Result:
    request_id: str
    candidate_id: str
    episode_id: str
    submit_step: int
    return_step: int
    decision: Task2Decision
    confidence: float
    reason: str = ""

    matched_attributes: List[str] = field(default_factory=list)
    failed_attributes: List[str] = field(default_factory=list)

    success: bool = True
    error: Optional[str] = None
    raw_response: Optional[str] = None
    latency_ms: Optional[float] = None
    returned_at: float = field(default_factory=now_ts)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.decision = Task2Decision.from_any(self.decision)
        self.confidence = clamp01(self.confidence)

    def is_positive(self, threshold: float = 0.6) -> bool:
        return self.decision == Task2Decision.YES and self.confidence >= threshold

    def is_negative(self, threshold: float = 0.5) -> bool:
        return self.decision == Task2Decision.NO and self.confidence >= threshold

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "episode_id": self.episode_id,
            "submit_step": self.submit_step,
            "return_step": self.return_step,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "matched_attributes": list(self.matched_attributes),
            "failed_attributes": list(self.failed_attributes),
            "success": self.success,
            "error": self.error,
            "raw_response": self.raw_response,
            "latency_ms": self.latency_ms,
            "returned_at": self.returned_at,
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class TargetEvidence:
    """
    Accumulated evidence for one possible target object.

    This is the safety layer between GDINO/Task2 and navigation.
    A single GDINO detection or one Task2 positive result is not enough to stop.
    """

    target_id: str
    episode_id: str
    status: CandidateStatus = CandidateStatus.NEW

    latest_candidate_id: Optional[str] = None
    position_3d: Optional[Tuple[float, float, float]] = None
    position_confidence: float = 0.0
    position_stability: float = 0.0

    verify_score: float = 0.0
    seen_count: int = 0
    positive_count: int = 0
    maybe_count: int = 0
    negative_count: int = 0
    rejected_count: int = 0

    first_seen_step: Optional[int] = None
    last_seen_step: Optional[int] = None
    last_verified_step: Optional[int] = None

    source_candidate_ids: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = CandidateStatus(self.status)
        self.position_confidence = clamp01(self.position_confidence)
        self.position_stability = clamp01(self.position_stability)
        self.verify_score = clamp01(self.verify_score)

    def mark_seen(self, candidate: GDINOCandidate) -> None:
        self.latest_candidate_id = candidate.candidate_id
        self.seen_count += 1

        if self.first_seen_step is None:
            self.first_seen_step = candidate.step_id
        self.last_seen_step = candidate.step_id

        if candidate.candidate_id not in self.source_candidate_ids:
            self.source_candidate_ids.append(candidate.candidate_id)

        if candidate.position_3d is not None:
            self.position_3d = candidate.position_3d
            self.position_confidence = max(
                self.position_confidence,
                clamp01(candidate.position_confidence),
            )

        if self.status == CandidateStatus.NEW:
            self.status = CandidateStatus.TENTATIVE

    def apply_task2_result(self, result: Task2Result) -> None:
        if result.candidate_id not in self.source_candidate_ids:
            self.source_candidate_ids.append(result.candidate_id)

        self.last_verified_step = result.return_step
        if result.reason:
            self.reasons.append(result.reason)

        if result.decision == Task2Decision.YES:
            self.positive_count += 1
            self.verify_score = max(self.verify_score, result.confidence)
            if self.status not in (CandidateStatus.REJECTED, CandidateStatus.LOST):
                self.status = CandidateStatus.TENTATIVE

        elif result.decision == Task2Decision.MAYBE:
            self.maybe_count += 1
            self.verify_score = max(self.verify_score, result.confidence * 0.5)
            if self.status == CandidateStatus.NEW:
                self.status = CandidateStatus.TENTATIVE

        elif result.decision == Task2Decision.NO:
            self.negative_count += 1
            self.verify_score = max(0.0, self.verify_score - result.confidence * 0.5)
            if self.negative_count >= max(2, self.positive_count + 1):
                self.status = CandidateStatus.REJECTED
                self.rejected_count += 1

    def set_verified(self, reason: str = "") -> None:
        self.status = CandidateStatus.VERIFIED
        if reason:
            self.reasons.append(reason)

    def set_lost(self, reason: str = "") -> None:
        self.status = CandidateStatus.LOST
        if reason:
            self.reasons.append(reason)

    def set_rejected(self, reason: str = "") -> None:
        self.status = CandidateStatus.REJECTED
        self.rejected_count += 1
        if reason:
            self.reasons.append(reason)

    def has_stable_position(self, threshold: float = 0.5) -> bool:
        return (
            self.position_3d is not None
            and self.position_confidence >= threshold
            and self.position_stability >= threshold
        )

    def is_verified(self) -> bool:
        return self.status == CandidateStatus.VERIFIED

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "episode_id": self.episode_id,
            "status": self.status.value,
            "latest_candidate_id": self.latest_candidate_id,
            "position_3d": _json_safe(self.position_3d),
            "position_confidence": self.position_confidence,
            "position_stability": self.position_stability,
            "verify_score": self.verify_score,
            "seen_count": self.seen_count,
            "positive_count": self.positive_count,
            "maybe_count": self.maybe_count,
            "negative_count": self.negative_count,
            "rejected_count": self.rejected_count,
            "first_seen_step": self.first_seen_step,
            "last_seen_step": self.last_seen_step,
            "last_verified_step": self.last_verified_step,
            "source_candidate_ids": list(self.source_candidate_ids),
            "reasons": list(self.reasons[-10:]),
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class NavDecision:
    episode_id: str
    step_id: int
    mode: NavMode

    target_type: str = "none"
    target_position: Optional[Tuple[float, float, float]] = None
    target_id: Optional[str] = None
    candidate_id: Optional[str] = None

    action: Optional[str] = None
    step_size: Optional[float] = None
    action_source: ActionSource = ActionSource.GEOMETRIC_EXPLORE

    stop_allowed: bool = False
    reason: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mode = NavMode(self.mode)
        self.action_source = ActionSource(self.action_source)

    @staticmethod
    def hold(
        episode_id: str,
        step_id: int,
        reason: str = "safety hold",
    ) -> "NavDecision":
        return NavDecision(
            episode_id=episode_id,
            step_id=step_id,
            mode=NavMode.SEARCH,
            action="rotl",
            step_size=0.0,
            action_source=ActionSource.SAFETY_HOLD,
            reason=reason,
        )

    @staticmethod
    def stop(
        episode_id: str,
        step_id: int,
        reason: str = "stop gate passed",
        target_id: Optional[str] = None,
    ) -> "NavDecision":
        return NavDecision(
            episode_id=episode_id,
            step_id=step_id,
            mode=NavMode.STOP,
            target_type="object",
            target_id=target_id,
            action="stop",
            step_size=0.0,
            action_source=ActionSource.STOP_GATE,
            stop_allowed=True,
            reason=reason,
        )

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "mode": self.mode.value,
            "target_type": self.target_type,
            "target_position": _json_safe(self.target_position),
            "target_id": self.target_id,
            "candidate_id": self.candidate_id,
            "action": self.action,
            "step_size": self.step_size,
            "action_source": self.action_source.value,
            "stop_allowed": self.stop_allowed,
            "reason": self.reason,
            "debug_info": _json_safe(self.debug_info),
        }


@dataclass
class AsyncTaskRecord:
    request_id: str
    task_type: AsyncTaskType
    episode_id: str
    submit_step: int
    payload: Any
    submitted_at: float = field(default_factory=now_ts)
    completed: bool = False
    cancelled: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task_type = AsyncTaskType(self.task_type)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_type": self.task_type.value,
            "episode_id": self.episode_id,
            "submit_step": self.submit_step,
            "payload": _json_safe(self.payload),
            "submitted_at": self.submitted_at,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "error": self.error,
            "metadata": _json_safe(self.metadata),
        }


__all__ = [
    "ActionSource",
    "AsyncTaskRecord",
    "AsyncTaskType",
    "BBox",
    "CandidateStatus",
    "FrameRecord",
    "GDINOCandidate",
    "GDINOResult",
    "NavDecision",
    "NavMode",
    "ObservationRecord",
    "PoseRecord",
    "RegionScore",
    "TargetEvidence",
    "TargetInfo",
    "Task1Request",
    "Task1Result",
    "Task2Decision",
    "Task2Request",
    "Task2Result",
    "ViewID",
    "bytes_to_b64",
    "clamp01",
    "new_id",
    "now_ts",
    "strip_data_uri",
    "to_data_uri",
]
