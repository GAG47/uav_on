from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from svnav.semantic_map import SemanticMap
from svnav.types import (
    FrameRecord,
    GDINORequest,
    NavMode,
    ObservationRecord,
    PoseRecord,
    Task1Request,
    Task1Result,
    TargetInfo,
    ViewID,
    now_ts,
)


class KeyframeRejectReason(str, Enum):
    EPISODE_MISMATCH = "episode_mismatch"
    DUPLICATED_FRAME = "duplicated_frame"
    NO_VISUAL_CONTENT = "no_visual_content"
    REDUNDANT_COVERAGE = "redundant_coverage"
    PENDING_FULL_LOW_PRIORITY = "pending_full_low_priority"


@dataclass
class KeyframeManagerConfig:
    task1_batch_size: int = 4
    min_task1_batch_size: int = 2
    max_task1_interval: int = 8
    max_inflight_task1: int = 2

    max_pending_keyframes: int = 8
    coverage_overlap_threshold: float = 0.65
    coverage_history_size: int = 64

    min_visible_cells: int = 1
    min_translation: float = 5.0
    min_yaw_change_deg: float = 45.0

    force_initial_keyframes: bool = True
    max_interval_refresh_keyframes: int = 2

    def __post_init__(self) -> None:
        self.task1_batch_size = int(self.task1_batch_size)
        self.min_task1_batch_size = int(self.min_task1_batch_size)
        self.max_task1_interval = int(self.max_task1_interval)
        self.max_inflight_task1 = int(self.max_inflight_task1)
        self.max_pending_keyframes = int(self.max_pending_keyframes)
        self.coverage_history_size = int(self.coverage_history_size)
        self.min_visible_cells = int(self.min_visible_cells)
        self.min_translation = float(self.min_translation)
        self.min_yaw_change_deg = float(self.min_yaw_change_deg)
        self.max_interval_refresh_keyframes = int(self.max_interval_refresh_keyframes)

        if self.task1_batch_size <= 0:
            raise ValueError("task1_batch_size must be positive")
        if self.min_task1_batch_size <= 0:
            raise ValueError("min_task1_batch_size must be positive")
        if self.min_task1_batch_size > self.task1_batch_size:
            self.min_task1_batch_size = self.task1_batch_size
        if self.max_inflight_task1 <= 0:
            raise ValueError("max_inflight_task1 must be positive")
        if self.max_pending_keyframes < self.task1_batch_size:
            self.max_pending_keyframes = self.task1_batch_size


@dataclass
class CoverageRecord:
    """
    Lightweight historical coverage record.

    This intentionally does not store full FrameRecord, rgb_bytes, rgb_b64,
    or depth. It only keeps the information needed for redundancy checking.
    """

    frame_id: str
    episode_id: str
    step_id: int
    view_id: ViewID
    pose: PoseRecord
    visible_cells: Set[Tuple[int, int]]
    reason: str
    coverage_score: float
    overlap_ratio: float
    created_at: float = field(default_factory=now_ts)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "view_id": self.view_id.value,
            "visible_cell_count": len(self.visible_cells),
            "reason": self.reason,
            "coverage_score": float(self.coverage_score),
            "overlap_ratio": float(self.overlap_ratio),
            "created_at": self.created_at,
        }


@dataclass
class CoverageKeyframe:
    """
    Full keyframe kept only while pending or inflight.

    This contains FrameRecord, so it may indirectly contain image bytes/base64.
    It should not be kept forever.
    """

    frame: FrameRecord
    visible_cells: Set[Tuple[int, int]]
    reason: str
    coverage_score: float
    overlap_ratio: float
    created_at: float = field(default_factory=now_ts)

    @property
    def frame_id(self) -> str:
        return self.frame.frame_id

    @property
    def episode_id(self) -> str:
        return self.frame.episode_id

    @property
    def step_id(self) -> int:
        return self.frame.step_id

    @property
    def view_id(self) -> ViewID:
        return self.frame.view_id

    @property
    def pose(self) -> PoseRecord:
        return self.frame.pose

    def to_record(self) -> CoverageRecord:
        return CoverageRecord(
            frame_id=self.frame_id,
            episode_id=self.episode_id,
            step_id=self.step_id,
            view_id=self.view_id,
            pose=self.pose,
            visible_cells=set(self.visible_cells),
            reason=self.reason,
            coverage_score=self.coverage_score,
            overlap_ratio=self.overlap_ratio,
            created_at=self.created_at,
        )

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "view_id": self.view_id.value,
            "visible_cell_count": len(self.visible_cells),
            "reason": self.reason,
            "coverage_score": float(self.coverage_score),
            "overlap_ratio": float(self.overlap_ratio),
            "created_at": self.created_at,
        }


@dataclass
class KeyframeUpdateResult:
    episode_id: str
    step_id: int
    accepted: List[CoverageKeyframe] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)
    replaced: Dict[str, str] = field(default_factory=dict)
    task1_request: Optional[Task1Request] = None
    reason: str = ""

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "accepted": [item.to_log_dict() for item in self.accepted],
            "rejected": dict(self.rejected),
            "replaced": dict(self.replaced),
            "task1_request_id": None if self.task1_request is None else self.task1_request.request_id,
            "task1_frame_ids": [] if self.task1_request is None else self.task1_request.frame_ids,
            "reason": self.reason,
        }


class Task1KeyframeManager:
    """
    Coverage-aware keyframe manager for Task1.

    Responsibilities:
        - select keyframes from UAV-ON multi-view observations
        - keep frame_id -> FrameRecord lookup for async Task1 results
        - build Task1Request when pending keyframes are ready
        - clean frame lookup after Task1Result has been applied to SemanticMap

    This class does not call VLM, does not update SemanticMap values,
    does not call GDINO, and does not output navigation actions.
    """

    def __init__(self, config: Optional[KeyframeManagerConfig] = None) -> None:
        self.config = config or KeyframeManagerConfig()
        self.current_episode_id: Optional[str] = None

        self.pending_keyframes: List[CoverageKeyframe] = []
        self.frame_lookup: Dict[str, FrameRecord] = {}

        self.submitted_requests: Dict[str, Task1Request] = {}
        self.request_frame_ids: Dict[str, Set[str]] = {}

        self.coverage_history: List[CoverageRecord] = []
        self.last_keyframe_by_view: Dict[ViewID, CoverageRecord] = {}

        self.last_submit_step: int = -1
        self.last_observe_step: int = -1
        self._last_reject_reason: KeyframeRejectReason = KeyframeRejectReason.REDUNDANT_COVERAGE
        self._last_replaced_frame_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self, episode_id: str) -> None:
        self.current_episode_id = str(episode_id)

        self.pending_keyframes.clear()
        self.frame_lookup.clear()
        self.submitted_requests.clear()
        self.request_frame_ids.clear()

        self.coverage_history.clear()
        self.last_keyframe_by_view.clear()

        self.last_submit_step = -1
        self.last_observe_step = -1
        self._last_reject_reason = KeyframeRejectReason.REDUNDANT_COVERAGE
        self._last_replaced_frame_id = None

    def reset(self, episode_id: str) -> None:
        self.reset_episode(episode_id)

    # ------------------------------------------------------------------
    # Main update entry
    # ------------------------------------------------------------------

    def observe(
        self,
        observation: ObservationRecord,
        semantic_map: SemanticMap,
        build_request: bool = True,
    ) -> KeyframeUpdateResult:
        """
        Observe one synchronized UAV-ON step and collect Task1 keyframes.

        If build_request is True, this method also tries to build a
        Task1Request after adding accepted keyframes.
        """
        if self.current_episode_id is None:
            self.reset_episode(observation.episode_id)

        if observation.episode_id != self.current_episode_id:
            self.reset_episode(observation.episode_id)

        result = KeyframeUpdateResult(
            episode_id=observation.episode_id,
            step_id=observation.step_id,
        )

        self.last_observe_step = int(observation.step_id)

        if build_request and self.pending_count >= self.config.task1_batch_size:
            request = self.maybe_build_task1_request(
                current_step=observation.step_id,
                target_info=observation.target_info,
            )
            if request is not None:
                result.task1_request = request
                result.reason = "built_task1_request_before_observe"

        accepted_this_step: List[CoverageKeyframe] = []
        frames = observation.frame_list()

        for frame in frames:
            self._last_replaced_frame_id = None

            accepted = self._try_add_frame(
                frame=frame,
                semantic_map=semantic_map,
                force_reason=None,
            )

            if accepted is not None:
                accepted_this_step.append(accepted)
                result.accepted.append(accepted)

                if self._last_replaced_frame_id is not None:
                    result.replaced[self._last_replaced_frame_id] = accepted.frame_id

                if (
                    build_request
                    and result.task1_request is None
                    and self.pending_count >= self.config.task1_batch_size
                ):
                    request = self.maybe_build_task1_request(
                        current_step=observation.step_id,
                        target_info=observation.target_info,
                    )
                    if request is not None:
                        result.task1_request = request
                        result.reason = "built_task1_request_after_accept"
            else:
                result.rejected[frame.frame_id] = self._last_reject_reason.value

        if (
            not accepted_this_step
            and self._should_force_interval_refresh(observation.step_id)
        ):
            forced = self._add_interval_refresh_frames(
                observation=observation,
                semantic_map=semantic_map,
            )
            result.accepted.extend(forced)

            if (
                build_request
                and result.task1_request is None
                and self.pending_count >= self.config.min_task1_batch_size
            ):
                request = self.maybe_build_task1_request(
                    current_step=observation.step_id,
                    target_info=observation.target_info,
                )
                if request is not None:
                    result.task1_request = request
                    result.reason = "built_task1_request_after_interval_refresh"

        if build_request and result.task1_request is None:
            request = self.maybe_build_task1_request(
                current_step=observation.step_id,
                target_info=observation.target_info,
            )
            result.task1_request = request
            if request is not None and not result.reason:
                result.reason = "built_task1_request_by_interval"

        return result

    # ------------------------------------------------------------------
    # Frame selection
    # ------------------------------------------------------------------

    def _try_add_frame(
        self,
        frame: FrameRecord,
        semantic_map: SemanticMap,
        force_reason: Optional[str] = None,
    ) -> Optional[CoverageKeyframe]:
        if frame.episode_id != self.current_episode_id:
            self._last_reject_reason = KeyframeRejectReason.EPISODE_MISMATCH
            return None

        if frame.frame_id in self.frame_lookup:
            self._last_reject_reason = KeyframeRejectReason.DUPLICATED_FRAME
            return None

        if not self._frame_has_task1_content(frame):
            self._last_reject_reason = KeyframeRejectReason.NO_VISUAL_CONTENT
            return None

        visible_cells = self._project_visible_cells(frame, semantic_map)

        reason = ""
        overlap = 1.0
        coverage_score = 0.0

        if force_reason is not None:
            reason = force_reason
            overlap = self._max_coverage_overlap(visible_cells)
            coverage_score = self._compute_coverage_score(
                visible_cells=visible_cells,
                overlap_ratio=overlap,
                reason=reason,
            )
            return self._add_pending_keyframe(
                frame=frame,
                visible_cells=visible_cells,
                reason=reason,
                coverage_score=coverage_score,
                overlap_ratio=overlap,
            )

        if self.config.force_initial_keyframes and not self.coverage_history:
            reason = "initial_observation"
            overlap = 0.0
            coverage_score = self._compute_coverage_score(
                visible_cells=visible_cells,
                overlap_ratio=overlap,
                reason=reason,
            )
            return self._add_pending_keyframe(
                frame=frame,
                visible_cells=visible_cells,
                reason=reason,
                coverage_score=coverage_score,
                overlap_ratio=overlap,
            )

        if len(visible_cells) >= self.config.min_visible_cells:
            overlap = self._max_coverage_overlap(visible_cells)
            if overlap < self.config.coverage_overlap_threshold:
                reason = "new_coverage:{:.3f}".format(overlap)
                coverage_score = self._compute_coverage_score(
                    visible_cells=visible_cells,
                    overlap_ratio=overlap,
                    reason=reason,
                )
                return self._add_pending_keyframe(
                    frame=frame,
                    visible_cells=visible_cells,
                    reason=reason,
                    coverage_score=coverage_score,
                    overlap_ratio=overlap,
                )

        if self._has_motion_change(frame):
            reason = "motion_change"
            overlap = self._max_coverage_overlap(visible_cells)
            coverage_score = self._compute_coverage_score(
                visible_cells=visible_cells,
                overlap_ratio=overlap,
                reason=reason,
            )
            return self._add_pending_keyframe(
                frame=frame,
                visible_cells=visible_cells,
                reason=reason,
                coverage_score=coverage_score,
                overlap_ratio=overlap,
            )

        self._last_reject_reason = KeyframeRejectReason.REDUNDANT_COVERAGE
        return None

    def _add_pending_keyframe(
        self,
        frame: FrameRecord,
        visible_cells: Set[Tuple[int, int]],
        reason: str,
        coverage_score: float,
        overlap_ratio: float,
    ) -> Optional[CoverageKeyframe]:
        keyframe = CoverageKeyframe(
            frame=frame,
            visible_cells=visible_cells,
            reason=reason,
            coverage_score=coverage_score,
            overlap_ratio=overlap_ratio,
        )

        if len(self.pending_keyframes) >= self.config.max_pending_keyframes:
            replaced = self._replace_weak_pending_if_better(keyframe)
            if replaced is None:
                self._last_reject_reason = KeyframeRejectReason.PENDING_FULL_LOW_PRIORITY
                return None
            return replaced

        self._append_pending_keyframe(keyframe)
        return keyframe

    def _append_pending_keyframe(self, keyframe: CoverageKeyframe) -> None:
        self.pending_keyframes.append(keyframe)
        self.frame_lookup[keyframe.frame_id] = keyframe.frame

        record = keyframe.to_record()
        self.coverage_history.append(record)
        if len(self.coverage_history) > self.config.coverage_history_size:
            self.coverage_history = self.coverage_history[-self.config.coverage_history_size:]

        self.last_keyframe_by_view[keyframe.view_id] = record

    def _replace_weak_pending_if_better(
        self,
        new_keyframe: CoverageKeyframe,
    ) -> Optional[CoverageKeyframe]:
        if not self.pending_keyframes:
            self._append_pending_keyframe(new_keyframe)
            return new_keyframe

        worst_index = 0
        worst_priority = self._pending_priority(self.pending_keyframes[0])

        for idx, item in enumerate(self.pending_keyframes[1:], start=1):
            priority = self._pending_priority(item)
            if priority < worst_priority:
                worst_index = idx
                worst_priority = priority

        new_priority = self._pending_priority(new_keyframe)

        if new_priority <= worst_priority + 1e-6:
            return None

        old = self.pending_keyframes.pop(worst_index)
        self._last_replaced_frame_id = old.frame_id

        self.frame_lookup.pop(old.frame_id, None)
        self._remove_coverage_record(old.frame_id)

        self._append_pending_keyframe(new_keyframe)
        return new_keyframe

    def _remove_coverage_record(self, frame_id: str) -> None:
        self.coverage_history = [
            item for item in self.coverage_history if item.frame_id != frame_id
        ]

        for view_id, record in list(self.last_keyframe_by_view.items()):
            if record.frame_id == frame_id:
                replacement = None
                for item in reversed(self.coverage_history):
                    if item.view_id == view_id:
                        replacement = item
                        break

                if replacement is None:
                    self.last_keyframe_by_view.pop(view_id, None)
                else:
                    self.last_keyframe_by_view[view_id] = replacement

    def _compute_coverage_score(
        self,
        visible_cells: Set[Tuple[int, int]],
        overlap_ratio: float,
        reason: str,
    ) -> float:
        novelty = max(0.0, 1.0 - float(overlap_ratio))
        coverage_size_score = min(1.0, float(len(visible_cells)) / 20.0)

        score = 0.75 * novelty + 0.25 * coverage_size_score

        if reason == "initial_observation":
            score += 0.20
        elif reason == "motion_change":
            score += 0.10
        elif reason == "interval_refresh":
            score -= 0.10

        return float(max(0.0, score))

    def _pending_priority(self, keyframe: CoverageKeyframe) -> float:
        priority = float(keyframe.coverage_score)

        if keyframe.reason == "initial_observation":
            priority += 0.05
        if keyframe.reason == "interval_refresh":
            priority -= 0.15

        priority += min(0.10, float(len(keyframe.visible_cells)) * 0.005)
        return priority

    def _project_visible_cells(
        self,
        frame: FrameRecord,
        semantic_map: SemanticMap,
    ) -> Set[Tuple[int, int]]:
        try:
            projected = semantic_map.project_frame_to_cells(frame)
        except Exception:
            projected = []

        cells: Set[Tuple[int, int]] = set()
        for gx, gy, _ in projected:
            cells.add((int(gx), int(gy)))
        return cells

    def _max_coverage_overlap(self, visible_cells: Set[Tuple[int, int]]) -> float:
        if not visible_cells:
            return 1.0

        max_overlap = 0.0
        for old_keyframe in self.coverage_history:
            old_cells = old_keyframe.visible_cells
            if not old_cells:
                continue

            inter = len(visible_cells.intersection(old_cells))
            ratio = float(inter) / float(max(1, len(visible_cells)))
            if ratio > max_overlap:
                max_overlap = ratio

        return max_overlap

    def _has_motion_change(self, frame: FrameRecord) -> bool:
        last = self.last_keyframe_by_view.get(frame.view_id)
        if last is None:
            return False

        dx = frame.pose.x - last.pose.x
        dy = frame.pose.y - last.pose.y
        dz = frame.pose.z - last.pose.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        yaw_diff = abs(self._angle_diff_deg(frame.pose.yaw, last.pose.yaw))

        return (
            dist >= self.config.min_translation
            or yaw_diff >= self.config.min_yaw_change_deg
        )

    def _should_force_interval_refresh(self, current_step: int) -> bool:
        if len(self.pending_keyframes) >= self.config.min_task1_batch_size:
            return False

        if self.last_submit_step < 0:
            return False

        return int(current_step) - int(self.last_submit_step) >= self.config.max_task1_interval

    def _add_interval_refresh_frames(
        self,
        observation: ObservationRecord,
        semantic_map: SemanticMap,
    ) -> List[CoverageKeyframe]:
        added: List[CoverageKeyframe] = []

        preferred_views = [
            ViewID.FRONT,
            ViewID.LEFT,
            ViewID.RIGHT,
            ViewID.DOWN,
        ]

        for view_id in preferred_views:
            if len(added) >= self.config.max_interval_refresh_keyframes:
                break

            frame = observation.get_frame(view_id)
            if frame is None:
                continue

            accepted = self._try_add_frame(
                frame=frame,
                semantic_map=semantic_map,
                force_reason="interval_refresh",
            )
            if accepted is not None:
                added.append(accepted)

        return added

    @staticmethod
    def _frame_has_task1_content(frame: FrameRecord) -> bool:
        # SVNav Task1 directly uses image input to VLM.
        # Caption fallback is intentionally not used here.
        return bool(frame.has_image())

    @staticmethod
    def _angle_diff_deg(a: float, b: float) -> float:
        a_rad = Task1KeyframeManager._yaw_to_rad(a)
        b_rad = Task1KeyframeManager._yaw_to_rad(b)

        diff = a_rad - b_rad
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff <= -math.pi:
            diff += 2.0 * math.pi

        return math.degrees(diff)

    @staticmethod
    def _yaw_to_rad(yaw: float) -> float:
        yaw = float(yaw)
        if abs(yaw) > 2.0 * math.pi + 1e-3:
            return math.radians(yaw)
        return yaw

    # ------------------------------------------------------------------
    # Task1 request lifecycle
    # ------------------------------------------------------------------

    def maybe_build_task1_request(
        self,
        current_step: int,
        target_info: TargetInfo,
        force: bool = False,
    ) -> Optional[Task1Request]:
        if self.current_episode_id is None:
            return None

        if not self.pending_keyframes:
            return None

        if self.inflight_count >= self.config.max_inflight_task1:
            return None

        current_step = int(current_step)

        ready_by_count = len(self.pending_keyframes) >= self.config.task1_batch_size

        if self.last_submit_step < 0:
            ready_by_interval = len(self.pending_keyframes) >= self.config.min_task1_batch_size
        else:
            ready_by_interval = (
                current_step - self.last_submit_step >= self.config.max_task1_interval
                and len(self.pending_keyframes) >= self.config.min_task1_batch_size
            )

        if not force and not ready_by_count and not ready_by_interval:
            return None

        if force and len(self.pending_keyframes) < 1:
            return None

        take_count = min(len(self.pending_keyframes), self.config.task1_batch_size)
        selected = self.pending_keyframes[:take_count]
        self.pending_keyframes = self.pending_keyframes[take_count:]

        frames = [item.frame for item in selected]
        request = Task1Request.create(
            episode_id=self.current_episode_id,
            submit_step=current_step,
            target_info=target_info,
            frames=frames,
            metadata={
                "keyframe_reasons": {
                    item.frame_id: item.reason for item in selected
                },
                "visible_cell_counts": {
                    item.frame_id: len(item.visible_cells) for item in selected
                },
                "coverage_scores": {
                    item.frame_id: item.coverage_score for item in selected
                },
                "overlap_ratios": {
                    item.frame_id: item.overlap_ratio for item in selected
                },
            },
        )

        self.submitted_requests[request.request_id] = request
        self.request_frame_ids[request.request_id] = set(request.frame_ids)
        self.last_submit_step = current_step

        return request

    def should_accept_task1_result(self, result: Task1Result) -> bool:
        if self.current_episode_id is None:
            return False
        if result.episode_id != self.current_episode_id:
            return False
        if result.request_id not in self.submitted_requests:
            return False
        return True

    def get_frame_lookup(self) -> Dict[str, FrameRecord]:
        return self.frame_lookup

    def get_request_frame_lookup(self, request_id: str) -> Dict[str, FrameRecord]:
        frame_ids = self.request_frame_ids.get(request_id, set())
        return {
            frame_id: self.frame_lookup[frame_id]
            for frame_id in frame_ids
            if frame_id in self.frame_lookup
        }

    def mark_request_completed(self, request_id: str) -> Dict[str, Any]:
        """
        Call this after Task1Result has already been applied to SemanticMap.

        It removes request records and deletes FrameRecord entries that are
        no longer referenced by pending keyframes or other inflight requests.
        """
        request_id = str(request_id)
        frame_ids = set(self.request_frame_ids.get(request_id, set()))

        self.submitted_requests.pop(request_id, None)
        self.request_frame_ids.pop(request_id, None)

        protected = self._protected_frame_ids()
        removed = []

        for frame_id in frame_ids:
            if frame_id in protected:
                continue
            if frame_id in self.frame_lookup:
                self.frame_lookup.pop(frame_id, None)
                removed.append(frame_id)

        return {
            "request_id": request_id,
            "removed_frame_ids": removed,
            "remaining_frame_lookup": len(self.frame_lookup),
            "inflight_count": self.inflight_count,
        }

    def cancel_request(self, request_id: str) -> Dict[str, Any]:
        return self.mark_request_completed(request_id)

    def _protected_frame_ids(self) -> Set[str]:
        protected: Set[str] = set()

        for keyframe in self.pending_keyframes:
            protected.add(keyframe.frame_id)

        for frame_ids in self.request_frame_ids.values():
            protected.update(frame_ids)

        return protected

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def inflight_count(self) -> int:
        return len(self.submitted_requests)

    @property
    def pending_count(self) -> int:
        return len(self.pending_keyframes)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "current_episode_id": self.current_episode_id,
            "pending_count": self.pending_count,
            "inflight_count": self.inflight_count,
            "frame_lookup_size": len(self.frame_lookup),
            "coverage_history_size": len(self.coverage_history),
            "last_submit_step": self.last_submit_step,
            "last_observe_step": self.last_observe_step,
            "pending_keyframes": [
                item.to_log_dict() for item in self.pending_keyframes
            ],
            "submitted_requests": {
                request_id: request.to_log_dict()
                for request_id, request in self.submitted_requests.items()
            },
            "coverage_history": [
                item.to_log_dict() for item in self.coverage_history[-10:]
            ],
        }



# ----------------------------------------------------------------------
# GDINO keyframe selection
# ----------------------------------------------------------------------

class GDINOKeyframeRejectReason(str, Enum):
    EPISODE_MISMATCH = "episode_mismatch"
    DUPLICATED_FRAME = "duplicated_frame"
    NO_VISUAL_CONTENT = "no_visual_content"
    OUTSIDE_SEARCH_MODE = "outside_search_mode"
    GDINO_DISABLED = "gdino_disabled"
    INVALID_VISIBLE_CELLS = "invalid_visible_cells"
    REDUNDANT_COVERAGE = "redundant_coverage"
    SAME_VIEW_TOO_FREQUENT = "same_view_too_frequent"
    LOW_DETECTION_PRIORITY = "low_detection_priority"
    PENDING_FULL_LOW_PRIORITY = "pending_full_low_priority"


@dataclass
class GDINOKeyframeManagerConfig:
    """
    Config for Search-stage GDINO keyframe selection.

    Pending is not a raw image pool. It only stores frames that have passed
    admission checks and are waiting to be consumed by GDINO.

    The GDINO server is consumed conservatively: one selected keyframe per
    GDINORequest by default, because the current HTTP server processes images
    sequentially and every detection may later trigger filtering, 3D estimation,
    and Task2 verification.
    """

    gdino_request_size: int = 1
    max_inflight_gdino: int = 1
    max_pending_keyframes: int = 12

    coverage_overlap_threshold: float = 0.70
    coverage_history_size: int = 96

    min_visible_cells: int = 1
    min_translation: float = 4.0
    min_yaw_change_deg: float = 35.0

    min_admission_score: float = 0.28
    replace_score_margin: float = 0.05

    same_view_min_step_gap: int = 2
    same_view_recent_penalty_steps: int = 5
    stale_view_bonus_after_steps: int = 8

    semantic_support_weight: float = 0.30
    coverage_novelty_weight: float = 0.25
    active_viewpoint_weight: float = 0.20
    motion_score_weight: float = 0.10
    view_diversity_weight: float = 0.10
    stale_view_weight: float = 0.05
    recent_same_view_penalty_weight: float = 0.15

    front_view_base: float = 1.00
    side_view_base: float = 0.85
    down_view_base: float = 0.65

    def __post_init__(self) -> None:
        self.gdino_request_size = int(self.gdino_request_size)
        self.max_inflight_gdino = int(self.max_inflight_gdino)
        self.max_pending_keyframes = int(self.max_pending_keyframes)
        self.coverage_history_size = int(self.coverage_history_size)
        self.min_visible_cells = int(self.min_visible_cells)
        self.min_translation = float(self.min_translation)
        self.min_yaw_change_deg = float(self.min_yaw_change_deg)
        self.min_admission_score = float(self.min_admission_score)
        self.replace_score_margin = float(self.replace_score_margin)
        self.same_view_min_step_gap = int(self.same_view_min_step_gap)
        self.same_view_recent_penalty_steps = int(self.same_view_recent_penalty_steps)
        self.stale_view_bonus_after_steps = int(self.stale_view_bonus_after_steps)

        if self.gdino_request_size <= 0:
            raise ValueError("gdino_request_size must be positive")
        if self.max_inflight_gdino <= 0:
            raise ValueError("max_inflight_gdino must be positive")
        if self.max_pending_keyframes < self.gdino_request_size:
            self.max_pending_keyframes = self.gdino_request_size


@dataclass
class GDINOKeyframeRecord:
    """
    Lightweight accepted GDINO keyframe history.

    It intentionally does not keep image bytes. Full FrameRecord is kept only
    while pending or inflight.
    """

    keyframe_id: str
    frame_id: str
    episode_id: str
    step_id: int
    view_id: ViewID
    pose: PoseRecord
    visible_cells: Set[Tuple[int, int]]
    reason: str
    admission_score: float
    semantic_support: float
    coverage_novelty: float
    active_viewpoint_relevance: float
    motion_score: float
    view_diversity_bonus: float
    stale_view_bonus: float
    recent_same_view_penalty: float
    overlap_ratio: float
    freshness_status: str
    created_at: float = field(default_factory=now_ts)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "frame_id": self.frame_id,
            "episode_id": self.episode_id,
            "step_id": int(self.step_id),
            "view_id": self.view_id.value,
            "visible_cell_count": len(self.visible_cells),
            "reason": self.reason,
            "admission_score": float(self.admission_score),
            "semantic_support": float(self.semantic_support),
            "coverage_novelty": float(self.coverage_novelty),
            "active_viewpoint_relevance": float(self.active_viewpoint_relevance),
            "motion_score": float(self.motion_score),
            "view_diversity_bonus": float(self.view_diversity_bonus),
            "stale_view_bonus": float(self.stale_view_bonus),
            "recent_same_view_penalty": float(self.recent_same_view_penalty),
            "overlap_ratio": float(self.overlap_ratio),
            "freshness_status": self.freshness_status,
            "pose": self.pose.to_log_dict(),
            "created_at": float(self.created_at),
        }


@dataclass
class GDINOKeyframe:
    """
    Full pending / inflight GDINO keyframe.

    This contains FrameRecord, so it may indirectly contain image bytes/base64.
    It should not be stored forever outside the keyframe lifecycle.
    """

    keyframe_id: str
    frame: FrameRecord
    visible_cells: Set[Tuple[int, int]]
    reason: str
    admission_score: float
    semantic_support: float
    coverage_novelty: float
    active_viewpoint_relevance: float
    motion_score: float
    view_diversity_bonus: float
    stale_view_bonus: float
    recent_same_view_penalty: float
    overlap_ratio: float
    freshness_status: str
    semantic_snapshot: Dict[str, Any] = field(default_factory=dict)
    navigation_snapshot: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=now_ts)

    @property
    def frame_id(self) -> str:
        return self.frame.frame_id

    @property
    def episode_id(self) -> str:
        return self.frame.episode_id

    @property
    def step_id(self) -> int:
        return self.frame.step_id

    @property
    def view_id(self) -> ViewID:
        return self.frame.view_id

    @property
    def pose(self) -> PoseRecord:
        return self.frame.pose

    def to_record(self) -> GDINOKeyframeRecord:
        return GDINOKeyframeRecord(
            keyframe_id=self.keyframe_id,
            frame_id=self.frame_id,
            episode_id=self.episode_id,
            step_id=self.step_id,
            view_id=self.view_id,
            pose=self.pose,
            visible_cells=set(self.visible_cells),
            reason=self.reason,
            admission_score=self.admission_score,
            semantic_support=self.semantic_support,
            coverage_novelty=self.coverage_novelty,
            active_viewpoint_relevance=self.active_viewpoint_relevance,
            motion_score=self.motion_score,
            view_diversity_bonus=self.view_diversity_bonus,
            stale_view_bonus=self.stale_view_bonus,
            recent_same_view_penalty=self.recent_same_view_penalty,
            overlap_ratio=self.overlap_ratio,
            freshness_status=self.freshness_status,
            created_at=self.created_at,
        )

    def to_log_dict(self) -> Dict[str, Any]:
        data = self.to_record().to_log_dict()
        data["semantic_snapshot"] = dict(self.semantic_snapshot)
        data["navigation_snapshot"] = dict(self.navigation_snapshot)
        return data


@dataclass
class GDINOUpdateResult:
    episode_id: str
    step_id: int
    accepted: List[GDINOKeyframe] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)
    replaced: Dict[str, str] = field(default_factory=dict)
    gdino_request: Optional[GDINORequest] = None
    reason: str = ""

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_id": int(self.step_id),
            "accepted": [item.to_log_dict() for item in self.accepted],
            "rejected": dict(self.rejected),
            "replaced": dict(self.replaced),
            "gdino_request_id": None if self.gdino_request is None else self.gdino_request.request_id,
            "gdino_frame_ids": [] if self.gdino_request is None else self.gdino_request.frame_ids,
            "reason": self.reason,
        }


class GDINOKeyframeManager:
    """
    Search-stage GDINO keyframe manager.

    Responsibilities:
        - select GDINO keyframes from UAV-ON multi-view observations
        - store only admitted keyframes in a pending queue
        - build one-frame GDINORequest from the best pending keyframe
        - keep frame_id -> FrameRecord lookup for asynchronous result tracing
        - clean request records after GDINOResult is handled

    This class does not call GDINO, does not call VLM, does not update
    SemanticMap values, and does not output navigation actions.
    """

    def __init__(self, config: Optional[GDINOKeyframeManagerConfig] = None) -> None:
        self.config = config or GDINOKeyframeManagerConfig()
        self.current_episode_id: Optional[str] = None

        self.pending_keyframes: List[GDINOKeyframe] = []
        self.frame_lookup: Dict[str, FrameRecord] = {}

        self.submitted_requests: Dict[str, GDINORequest] = {}
        self.request_frame_ids: Dict[str, Set[str]] = {}
        self.request_keyframe_ids: Dict[str, Set[str]] = {}

        self.coverage_history: List[GDINOKeyframeRecord] = []
        self.last_keyframe_by_view: Dict[ViewID, GDINOKeyframeRecord] = {}

        self.last_submit_step: int = -1
        self.last_observe_step: int = -1
        self._last_reject_reason: GDINOKeyframeRejectReason = GDINOKeyframeRejectReason.LOW_DETECTION_PRIORITY
        self._last_replaced_keyframe_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self, episode_id: str) -> None:
        self.current_episode_id = str(episode_id)

        self.pending_keyframes.clear()
        self.frame_lookup.clear()
        self.submitted_requests.clear()
        self.request_frame_ids.clear()
        self.request_keyframe_ids.clear()

        self.coverage_history.clear()
        self.last_keyframe_by_view.clear()

        self.last_submit_step = -1
        self.last_observe_step = -1
        self._last_reject_reason = GDINOKeyframeRejectReason.LOW_DETECTION_PRIORITY
        self._last_replaced_keyframe_id = None

    def reset(self, episode_id: str) -> None:
        self.reset_episode(episode_id)

    # ------------------------------------------------------------------
    # Main update entry
    # ------------------------------------------------------------------

    def observe(
        self,
        observation: ObservationRecord,
        semantic_map: SemanticMap,
        nav_mode: Any = NavMode.SEARCH,
        nav_decision: Optional[Any] = None,
        build_request: bool = True,
        gdino_enabled: bool = True,
    ) -> GDINOUpdateResult:
        if self.current_episode_id is None:
            self.reset_episode(observation.episode_id)

        if observation.episode_id != self.current_episode_id:
            self.reset_episode(observation.episode_id)

        result = GDINOUpdateResult(
            episode_id=observation.episode_id,
            step_id=observation.step_id,
        )
        self.last_observe_step = int(observation.step_id)

        mode = self._coerce_nav_mode(nav_mode)
        frames = observation.frame_list()

        for frame in frames:
            self._last_replaced_keyframe_id = None
            accepted = self._try_add_frame(
                frame=frame,
                semantic_map=semantic_map,
                nav_mode=mode,
                nav_decision=nav_decision,
                gdino_enabled=gdino_enabled,
            )

            if accepted is not None:
                result.accepted.append(accepted)

                if self._last_replaced_keyframe_id is not None:
                    result.replaced[self._last_replaced_keyframe_id] = accepted.keyframe_id
            else:
                result.rejected[frame.frame_id] = self._last_reject_reason.value

        if build_request:
            request = self.maybe_build_gdino_request(
                current_step=observation.step_id,
                target_info=observation.target_info,
            )
            result.gdino_request = request
            if request is not None:
                result.reason = "built_gdino_request"

        return result

    # ------------------------------------------------------------------
    # Frame selection
    # ------------------------------------------------------------------

    def _try_add_frame(
        self,
        frame: FrameRecord,
        semantic_map: SemanticMap,
        nav_mode: NavMode,
        nav_decision: Optional[Any],
        gdino_enabled: bool,
    ) -> Optional[GDINOKeyframe]:
        if not gdino_enabled:
            self._last_reject_reason = GDINOKeyframeRejectReason.GDINO_DISABLED
            return None

        if nav_mode != NavMode.SEARCH:
            self._last_reject_reason = GDINOKeyframeRejectReason.OUTSIDE_SEARCH_MODE
            return None

        if frame.episode_id != self.current_episode_id:
            self._last_reject_reason = GDINOKeyframeRejectReason.EPISODE_MISMATCH
            return None

        if frame.frame_id in self.frame_lookup:
            self._last_reject_reason = GDINOKeyframeRejectReason.DUPLICATED_FRAME
            return None

        if not frame.has_image():
            self._last_reject_reason = GDINOKeyframeRejectReason.NO_VISUAL_CONTENT
            return None

        visible_cells = self._project_visible_cells(frame, semantic_map)
        if len(visible_cells) < self.config.min_visible_cells:
            self._last_reject_reason = GDINOKeyframeRejectReason.INVALID_VISIBLE_CELLS
            return None

        overlap = self._max_coverage_overlap(visible_cells)
        has_motion_change = self._has_motion_change(frame)

        if (
            overlap >= self.config.coverage_overlap_threshold
            and not has_motion_change
        ):
            self._last_reject_reason = GDINOKeyframeRejectReason.REDUNDANT_COVERAGE
            return None

        if self._is_same_view_too_frequent(frame) and not has_motion_change:
            self._last_reject_reason = GDINOKeyframeRejectReason.SAME_VIEW_TOO_FREQUENT
            return None

        score_parts = self._compute_admission_score(
            frame=frame,
            visible_cells=visible_cells,
            overlap_ratio=overlap,
            has_motion_change=has_motion_change,
            semantic_map=semantic_map,
            nav_decision=nav_decision,
        )

        admission_score = score_parts["admission_score"]
        if admission_score < self.config.min_admission_score:
            self._last_reject_reason = GDINOKeyframeRejectReason.LOW_DETECTION_PRIORITY
            return None

        reason = self._score_reason(score_parts)

        semantic_snapshot = self._build_semantic_snapshot(
            visible_cells=visible_cells,
            semantic_map=semantic_map,
        )
        navigation_snapshot = self._build_navigation_snapshot(nav_decision)

        keyframe = GDINOKeyframe(
            keyframe_id=self._new_keyframe_id(frame),
            frame=frame,
            visible_cells=visible_cells,
            reason=reason,
            admission_score=admission_score,
            semantic_support=score_parts["semantic_support"],
            coverage_novelty=score_parts["coverage_novelty"],
            active_viewpoint_relevance=score_parts["active_viewpoint_relevance"],
            motion_score=score_parts["motion_score"],
            view_diversity_bonus=score_parts["view_diversity_bonus"],
            stale_view_bonus=score_parts["stale_view_bonus"],
            recent_same_view_penalty=score_parts["recent_same_view_penalty"],
            overlap_ratio=overlap,
            freshness_status="current",
            semantic_snapshot=semantic_snapshot,
            navigation_snapshot=navigation_snapshot,
        )

        return self._add_pending_keyframe(keyframe)

    def _add_pending_keyframe(
        self,
        keyframe: GDINOKeyframe,
    ) -> Optional[GDINOKeyframe]:
        if len(self.pending_keyframes) >= self.config.max_pending_keyframes:
            replaced = self._replace_weak_pending_if_better(keyframe)
            if replaced is None:
                self._last_reject_reason = GDINOKeyframeRejectReason.PENDING_FULL_LOW_PRIORITY
                return None
            return replaced

        self._append_pending_keyframe(keyframe)
        return keyframe

    def _append_pending_keyframe(self, keyframe: GDINOKeyframe) -> None:
        self.pending_keyframes.append(keyframe)
        self.frame_lookup[keyframe.frame_id] = keyframe.frame

        record = keyframe.to_record()
        self.coverage_history.append(record)
        if len(self.coverage_history) > self.config.coverage_history_size:
            self.coverage_history = self.coverage_history[-self.config.coverage_history_size:]

        self.last_keyframe_by_view[keyframe.view_id] = record

    def _replace_weak_pending_if_better(
        self,
        new_keyframe: GDINOKeyframe,
    ) -> Optional[GDINOKeyframe]:
        if not self.pending_keyframes:
            self._append_pending_keyframe(new_keyframe)
            return new_keyframe

        worst_index = 0
        worst_priority = self._pending_priority(self.pending_keyframes[0])

        for idx, item in enumerate(self.pending_keyframes[1:], start=1):
            priority = self._pending_priority(item)
            if priority < worst_priority:
                worst_index = idx
                worst_priority = priority

        new_priority = self._pending_priority(new_keyframe)

        if new_priority <= worst_priority + self.config.replace_score_margin:
            return None

        old = self.pending_keyframes.pop(worst_index)
        self._last_replaced_keyframe_id = old.keyframe_id

        self.frame_lookup.pop(old.frame_id, None)
        self._remove_coverage_record(old.keyframe_id)

        self._append_pending_keyframe(new_keyframe)
        return new_keyframe

    def _remove_coverage_record(self, keyframe_id: str) -> None:
        self.coverage_history = [
            item for item in self.coverage_history if item.keyframe_id != keyframe_id
        ]

        for view_id, record in list(self.last_keyframe_by_view.items()):
            if record.keyframe_id == keyframe_id:
                replacement = None
                for item in reversed(self.coverage_history):
                    if item.view_id == view_id:
                        replacement = item
                        break

                if replacement is None:
                    self.last_keyframe_by_view.pop(view_id, None)
                else:
                    self.last_keyframe_by_view[view_id] = replacement

    def _pending_priority(self, keyframe: GDINOKeyframe, current_step: Optional[int] = None) -> float:
        priority = float(keyframe.admission_score)

        if current_step is not None:
            age = max(0, int(current_step) - int(keyframe.step_id))
            # Small age bonus prevents medium-priority pending keyframes from
            # being permanently starved by newer frames.
            priority += min(0.10, 0.005 * float(age))

        return float(priority)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_admission_score(
        self,
        frame: FrameRecord,
        visible_cells: Set[Tuple[int, int]],
        overlap_ratio: float,
        has_motion_change: bool,
        semantic_map: SemanticMap,
        nav_decision: Optional[Any],
    ) -> Dict[str, float]:
        semantic_support = self._semantic_support(visible_cells, semantic_map)
        coverage_novelty = max(0.0, 1.0 - float(overlap_ratio))
        active_viewpoint_relevance = self._active_viewpoint_relevance(frame, nav_decision)
        motion_score = 1.0 if has_motion_change else 0.0
        view_diversity_bonus = self._view_diversity_bonus(frame)
        stale_view_bonus = self._stale_view_bonus(frame)
        recent_same_view_penalty = self._recent_same_view_penalty(frame)

        score = (
            self.config.semantic_support_weight * semantic_support
            + self.config.coverage_novelty_weight * coverage_novelty
            + self.config.active_viewpoint_weight * active_viewpoint_relevance
            + self.config.motion_score_weight * motion_score
            + self.config.view_diversity_weight * view_diversity_bonus
            + self.config.stale_view_weight * stale_view_bonus
            - self.config.recent_same_view_penalty_weight * recent_same_view_penalty
        )

        view_base = self._view_base_score(frame.view_id)
        score *= view_base

        return {
            "admission_score": float(max(0.0, min(1.0, score))),
            "semantic_support": float(max(0.0, min(1.0, semantic_support))),
            "coverage_novelty": float(max(0.0, min(1.0, coverage_novelty))),
            "active_viewpoint_relevance": float(max(0.0, min(1.0, active_viewpoint_relevance))),
            "motion_score": float(max(0.0, min(1.0, motion_score))),
            "view_diversity_bonus": float(max(0.0, min(1.0, view_diversity_bonus))),
            "stale_view_bonus": float(max(0.0, min(1.0, stale_view_bonus))),
            "recent_same_view_penalty": float(max(0.0, min(1.0, recent_same_view_penalty))),
        }

    def _score_reason(self, score_parts: Dict[str, float]) -> str:
        ranked = sorted(
            [
                ("semantic", score_parts.get("semantic_support", 0.0)),
                ("coverage", score_parts.get("coverage_novelty", 0.0)),
                ("viewpoint", score_parts.get("active_viewpoint_relevance", 0.0)),
                ("motion", score_parts.get("motion_score", 0.0)),
                ("diversity", score_parts.get("view_diversity_bonus", 0.0)),
                ("stale_view", score_parts.get("stale_view_bonus", 0.0)),
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        parts = [
            "{}:{:.3f}".format(name, value)
            for name, value in ranked[:3]
            if value > 0.0
        ]
        if not parts:
            return "admitted"
        return "admitted_" + ",".join(parts)

    def _semantic_support(
        self,
        visible_cells: Set[Tuple[int, int]],
        semantic_map: SemanticMap,
    ) -> float:
        if not visible_cells:
            return 0.0

        scores = []
        cell_lookup = self._semantic_cell_lookup(semantic_map)

        for gx, gy in visible_cells:
            cell = cell_lookup.get((int(gx), int(gy)))
            if cell is None:
                continue
            value = float(getattr(cell, "semantic_value", 0.0))
            conf = float(getattr(cell, "semantic_conf", 0.0))
            scores.append(max(0.0, min(1.0, value * conf)))

        if not scores:
            return 0.0

        scores.sort(reverse=True)
        top = scores[: min(3, len(scores))]
        return float(0.7 * top[0] + 0.3 * (sum(top) / float(len(top))))

    def _semantic_cell_lookup(self, semantic_map: SemanticMap) -> Dict[Tuple[int, int], Any]:
        lookup: Dict[Tuple[int, int], Any] = {}
        try:
            cells = semantic_map.iter_cells()
        except Exception:
            return lookup

        for cell in cells:
            try:
                lookup[(int(cell.gx), int(cell.gy))] = cell
            except Exception:
                continue
        return lookup

    def _active_viewpoint_relevance(
        self,
        frame: FrameRecord,
        nav_decision: Optional[Any],
    ) -> float:
        if nav_decision is None:
            return 0.0

        debug_info = getattr(nav_decision, "debug_info", None) or {}
        desired_yaw = debug_info.get("desired_yaw")

        if desired_yaw is None:
            action = getattr(nav_decision, "action", "") or ""
            if action == "rotl" and frame.view_id == ViewID.LEFT:
                return 1.0
            if action == "rotr" and frame.view_id == ViewID.RIGHT:
                return 1.0
            if action == "forward" and frame.view_id == ViewID.FRONT:
                return 1.0
            return 0.2 if frame.view_id == ViewID.FRONT else 0.0

        try:
            desired_yaw = float(desired_yaw)
        except Exception:
            return 0.0

        frame_yaw = float(frame.pose.yaw) + self._view_yaw_offset_deg(frame.view_id)
        diff = abs(self._angle_diff_deg(frame_yaw, desired_yaw))
        return max(0.0, 1.0 - diff / 90.0)

    def _view_yaw_offset_deg(self, view_id: ViewID) -> float:
        if view_id == ViewID.LEFT:
            return -90.0
        if view_id == ViewID.RIGHT:
            return 90.0
        return 0.0

    def _view_diversity_bonus(self, frame: FrameRecord) -> float:
        counts = {
            ViewID.FRONT: 0,
            ViewID.LEFT: 0,
            ViewID.RIGHT: 0,
            ViewID.DOWN: 0,
        }
        for record in self.coverage_history:
            counts[record.view_id] = counts.get(record.view_id, 0) + 1

        max_count = max(1, max(counts.values()))
        view_count = counts.get(frame.view_id, 0)
        return max(0.0, 1.0 - float(view_count) / float(max_count + 1))

    def _stale_view_bonus(self, frame: FrameRecord) -> float:
        last = self.last_keyframe_by_view.get(frame.view_id)
        if last is None:
            return 1.0

        gap = int(frame.step_id) - int(last.step_id)
        if gap <= 0:
            return 0.0
        return min(1.0, float(gap) / float(max(1, self.config.stale_view_bonus_after_steps)))

    def _recent_same_view_penalty(self, frame: FrameRecord) -> float:
        last = self.last_keyframe_by_view.get(frame.view_id)
        if last is None:
            return 0.0

        gap = int(frame.step_id) - int(last.step_id)
        if gap < 0:
            return 0.0
        if gap >= self.config.same_view_recent_penalty_steps:
            return 0.0

        return 1.0 - float(gap) / float(max(1, self.config.same_view_recent_penalty_steps))

    def _view_base_score(self, view_id: ViewID) -> float:
        if view_id == ViewID.FRONT:
            return float(self.config.front_view_base)
        if view_id in (ViewID.LEFT, ViewID.RIGHT):
            return float(self.config.side_view_base)
        if view_id == ViewID.DOWN:
            return float(self.config.down_view_base)
        return 1.0

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def maybe_build_gdino_request(
        self,
        current_step: int,
        target_info: TargetInfo,
        force: bool = False,
    ) -> Optional[GDINORequest]:
        if self.current_episode_id is None:
            return None

        if not self.pending_keyframes:
            return None

        if self.inflight_count >= self.config.max_inflight_gdino:
            return None

        current_step = int(current_step)

        if not force and not self.pending_keyframes:
            return None

        ranked = sorted(
            self.pending_keyframes,
            key=lambda item: self._pending_priority(item, current_step=current_step),
            reverse=True,
        )
        take_count = min(len(ranked), self.config.gdino_request_size)
        selected = ranked[:take_count]
        selected_ids = {item.keyframe_id for item in selected}

        self.pending_keyframes = [
            item for item in self.pending_keyframes if item.keyframe_id not in selected_ids
        ]

        frames = [item.frame for item in selected]
        request = GDINORequest.create(
            episode_id=self.current_episode_id,
            submit_step=current_step,
            target_info=target_info,
            frames=frames,
            metadata={
                "gdino_keyframe_ids": [item.keyframe_id for item in selected],
                "keyframe_reasons": {
                    item.keyframe_id: item.reason for item in selected
                },
                "admission_scores": {
                    item.keyframe_id: item.admission_score for item in selected
                },
                "semantic_support": {
                    item.keyframe_id: item.semantic_support for item in selected
                },
                "coverage_novelty": {
                    item.keyframe_id: item.coverage_novelty for item in selected
                },
                "view_ids": {
                    item.keyframe_id: item.view_id.value for item in selected
                },
                "observe_steps": {
                    item.keyframe_id: item.step_id for item in selected
                },
                "freshness_status": {
                    item.keyframe_id: self._freshness_status(item, current_step)
                    for item in selected
                },
                "semantic_snapshot": {
                    item.keyframe_id: item.semantic_snapshot for item in selected
                },
                "navigation_snapshot": {
                    item.keyframe_id: item.navigation_snapshot for item in selected
                },
            },
        )

        self.submitted_requests[request.request_id] = request
        self.request_frame_ids[request.request_id] = set(request.frame_ids)
        self.request_keyframe_ids[request.request_id] = selected_ids
        self.last_submit_step = current_step

        return request

    def should_accept_gdino_result(self, result: Any) -> bool:
        if self.current_episode_id is None:
            return False
        if getattr(result, "episode_id", None) != self.current_episode_id:
            return False
        if getattr(result, "request_id", None) not in self.submitted_requests:
            return False
        return True

    def get_frame_lookup(self) -> Dict[str, FrameRecord]:
        return self.frame_lookup

    def get_request_frame_lookup(self, request_id: str) -> Dict[str, FrameRecord]:
        frame_ids = self.request_frame_ids.get(str(request_id), set())
        return {
            frame_id: self.frame_lookup[frame_id]
            for frame_id in frame_ids
            if frame_id in self.frame_lookup
        }

    def mark_request_completed(self, request_id: str) -> Dict[str, Any]:
        request_id = str(request_id)
        frame_ids = set(self.request_frame_ids.get(request_id, set()))

        self.submitted_requests.pop(request_id, None)
        self.request_frame_ids.pop(request_id, None)
        self.request_keyframe_ids.pop(request_id, None)

        protected = self._protected_frame_ids()
        removed = []

        for frame_id in frame_ids:
            if frame_id in protected:
                continue
            if frame_id in self.frame_lookup:
                self.frame_lookup.pop(frame_id, None)
                removed.append(frame_id)

        return {
            "request_id": request_id,
            "removed_frame_ids": removed,
            "remaining_frame_lookup": len(self.frame_lookup),
            "inflight_count": self.inflight_count,
        }

    def cancel_request(self, request_id: str) -> Dict[str, Any]:
        return self.mark_request_completed(request_id)

    def _protected_frame_ids(self) -> Set[str]:
        protected: Set[str] = set()

        for keyframe in self.pending_keyframes:
            protected.add(keyframe.frame_id)

        for frame_ids in self.request_frame_ids.values():
            protected.update(frame_ids)

        return protected

    # ------------------------------------------------------------------
    # Geometry / coverage helpers
    # ------------------------------------------------------------------

    def _project_visible_cells(
        self,
        frame: FrameRecord,
        semantic_map: SemanticMap,
    ) -> Set[Tuple[int, int]]:
        try:
            projected = semantic_map.project_frame_to_cells(frame)
        except Exception:
            projected = []

        cells: Set[Tuple[int, int]] = set()
        for gx, gy, _ in projected:
            cells.add((int(gx), int(gy)))
        return cells

    def _max_coverage_overlap(self, visible_cells: Set[Tuple[int, int]]) -> float:
        if not visible_cells:
            return 1.0

        max_overlap = 0.0
        for old_keyframe in self.coverage_history:
            old_cells = old_keyframe.visible_cells
            if not old_cells:
                continue

            inter = len(visible_cells.intersection(old_cells))
            ratio = float(inter) / float(max(1, len(visible_cells)))
            if ratio > max_overlap:
                max_overlap = ratio

        return max_overlap

    def _has_motion_change(self, frame: FrameRecord) -> bool:
        last = self.last_keyframe_by_view.get(frame.view_id)
        if last is None:
            return True

        dx = frame.pose.x - last.pose.x
        dy = frame.pose.y - last.pose.y
        dz = frame.pose.z - last.pose.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        yaw_diff = abs(self._angle_diff_deg(frame.pose.yaw, last.pose.yaw))

        return (
            dist >= self.config.min_translation
            or yaw_diff >= self.config.min_yaw_change_deg
        )

    def _is_same_view_too_frequent(self, frame: FrameRecord) -> bool:
        last = self.last_keyframe_by_view.get(frame.view_id)
        if last is None:
            return False

        gap = int(frame.step_id) - int(last.step_id)
        return gap >= 0 and gap < self.config.same_view_min_step_gap

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def _build_semantic_snapshot(
        self,
        visible_cells: Set[Tuple[int, int]],
        semantic_map: SemanticMap,
    ) -> Dict[str, Any]:
        cell_lookup = self._semantic_cell_lookup(semantic_map)

        values = []
        confs = []
        top_cells = []

        for gx, gy in visible_cells:
            cell = cell_lookup.get((int(gx), int(gy)))
            if cell is None:
                continue

            value = float(getattr(cell, "semantic_value", 0.0))
            conf = float(getattr(cell, "semantic_conf", 0.0))
            effective = value * conf
            values.append(value)
            confs.append(conf)
            top_cells.append(
                {
                    "gx": int(gx),
                    "gy": int(gy),
                    "semantic_value": value,
                    "semantic_conf": conf,
                    "effective_value": effective,
                    "visited_count": int(getattr(cell, "visited_count", 0)),
                    "status": getattr(getattr(cell, "status", None), "value", str(getattr(cell, "status", ""))),
                }
            )

        top_cells.sort(key=lambda item: item["effective_value"], reverse=True)

        return {
            "visible_cell_count": len(visible_cells),
            "semantic_value_max": max(values) if values else 0.0,
            "semantic_value_mean": sum(values) / float(len(values)) if values else 0.0,
            "semantic_conf_mean": sum(confs) / float(len(confs)) if confs else 0.0,
            "top_cells": top_cells[:5],
        }

    def _build_navigation_snapshot(self, nav_decision: Optional[Any]) -> Dict[str, Any]:
        if nav_decision is None:
            return {}

        data = {
            "mode": str(getattr(nav_decision, "mode", "")),
            "action": getattr(nav_decision, "action", None),
            "step_size": getattr(nav_decision, "step_size", None),
            "reason": getattr(nav_decision, "reason", None),
            "action_source": str(getattr(nav_decision, "action_source", "")),
        }

        debug = getattr(nav_decision, "debug_info", None) or {}
        active = debug.get("active_viewpoint") or {}
        viewpoint = active.get("viewpoint") or {}
        region = viewpoint.get("region") or {}

        data.update(
            {
                "viewpoint_id": viewpoint.get("viewpoint_id") or debug.get("viewpoint_id"),
                "region_id": viewpoint.get("region_id") or debug.get("region_id"),
                "region_source": region.get("source") or viewpoint.get("region_source"),
                "desired_yaw": debug.get("desired_yaw"),
                "waypoint": debug.get("waypoint"),
            }
        )
        return data

    def _freshness_status(self, keyframe: GDINOKeyframe, current_step: int) -> str:
        age = max(0, int(current_step) - int(keyframe.step_id))
        if age <= 5:
            return "current"
        if age <= 25:
            return "recent_history"
        return "historical"

    def _new_keyframe_id(self, frame: FrameRecord) -> str:
        return "gkf_{}_{}_{}".format(
            int(frame.step_id),
            frame.view_id.value,
            frame.frame_id.split("_")[-1],
        )

    @staticmethod
    def _coerce_nav_mode(value: Any) -> NavMode:
        if isinstance(value, NavMode):
            return value
        text = str(value).lower().strip()
        if text == NavMode.SEARCH.value:
            return NavMode.SEARCH
        if text == NavMode.CHECK_CANDIDATE.value:
            return NavMode.CHECK_CANDIDATE
        if text == NavMode.APPROACH.value:
            return NavMode.APPROACH
        if text == NavMode.FINAL_CHECK.value:
            return NavMode.FINAL_CHECK
        if text == NavMode.STOP.value:
            return NavMode.STOP
        return NavMode.SEARCH

    @staticmethod
    def _angle_diff_deg(a: float, b: float) -> float:
        a_rad = GDINOKeyframeManager._yaw_to_rad(a)
        b_rad = GDINOKeyframeManager._yaw_to_rad(b)

        diff = a_rad - b_rad
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff <= -math.pi:
            diff += 2.0 * math.pi

        return math.degrees(diff)

    @staticmethod
    def _yaw_to_rad(yaw: float) -> float:
        yaw = float(yaw)
        if abs(yaw) > 2.0 * math.pi + 1e-3:
            return math.radians(yaw)
        return yaw

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def inflight_count(self) -> int:
        return len(self.submitted_requests)

    @property
    def pending_count(self) -> int:
        return len(self.pending_keyframes)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "current_episode_id": self.current_episode_id,
            "pending_count": self.pending_count,
            "inflight_count": self.inflight_count,
            "frame_lookup_size": len(self.frame_lookup),
            "coverage_history_size": len(self.coverage_history),
            "last_submit_step": self.last_submit_step,
            "last_observe_step": self.last_observe_step,
            "pending_keyframes": [
                item.to_log_dict() for item in self.pending_keyframes
            ],
            "submitted_requests": {
                request_id: request.to_log_dict()
                for request_id, request in self.submitted_requests.items()
            },
            "coverage_history": [
                item.to_log_dict() for item in self.coverage_history[-10:]
            ],
        }


__all__ = [
    "GDINOKeyframe",
    "GDINOKeyframeManager",
    "GDINOKeyframeManagerConfig",
    "GDINOKeyframeRecord",
    "GDINOKeyframeRejectReason",
    "GDINOUpdateResult",
    "CoverageKeyframe",
    "CoverageRecord",
    "KeyframeManagerConfig",
    "KeyframeRejectReason",
    "KeyframeUpdateResult",
    "Task1KeyframeManager",
]
