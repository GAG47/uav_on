from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    CandidateStatus,
    TargetEvidence,
    Task2Decision,
    Task2Result,
    clamp01,
    new_id,
    now_ts,
)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_position(value: Any) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (
                float(value[0]),
                float(value[1]),
                float(value[2]),
            )
        except Exception:
            return None

    return None


def _distance_2d(
    a: Optional[Tuple[float, float]],
    b: Optional[Tuple[float, float]],
) -> float:
    if a is None or b is None:
        return float("inf")

    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
    )


def _distance_3d(
    a: Optional[Tuple[float, float, float]],
    b: Optional[Tuple[float, float, float]],
) -> float:
    if a is None or b is None:
        return float("inf")

    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def _bbox_center_norm(bbox: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if not bbox:
        return None

    center = bbox.get("center")
    width = _safe_float(bbox.get("image_width"), 0.0)
    height = _safe_float(bbox.get("image_height"), 0.0)

    if isinstance(center, (list, tuple)) and len(center) >= 2 and width > 1.0 and height > 1.0:
        return (
            _safe_float(center[0]) / width,
            _safe_float(center[1]) / height,
        )

    x1 = _safe_float(bbox.get("x1"), 0.0)
    y1 = _safe_float(bbox.get("y1"), 0.0)
    x2 = _safe_float(bbox.get("x2"), 0.0)
    y2 = _safe_float(bbox.get("y2"), 0.0)

    if width <= 1.0 or height <= 1.0:
        return None

    return (
        ((x1 + x2) * 0.5) / width,
        ((y1 + y2) * 0.5) / height,
    )


def _bbox_area_ratio(bbox: Dict[str, Any]) -> float:
    if not bbox:
        return 0.0

    if "area_ratio" in bbox:
        return clamp01(bbox.get("area_ratio", 0.0))

    width = _safe_float(bbox.get("image_width"), 0.0)
    height = _safe_float(bbox.get("image_height"), 0.0)
    if width <= 1.0 or height <= 1.0:
        return 0.0

    x1 = _safe_float(bbox.get("x1"), 0.0)
    y1 = _safe_float(bbox.get("y1"), 0.0)
    x2 = _safe_float(bbox.get("x2"), 0.0)
    y2 = _safe_float(bbox.get("y2"), 0.0)

    return clamp01(
        max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, width * height)
    )


def _pose_xy(source_pose: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if not source_pose:
        return None

    if "x" not in source_pose or "y" not in source_pose:
        return None

    return (
        _safe_float(source_pose.get("x"), 0.0),
        _safe_float(source_pose.get("y"), 0.0),
    )


def _label_overlap(a: str, b: str) -> float:
    aw = {
        w.strip().lower()
        for w in str(a or "").replace(".", " ").replace(",", " ").split()
        if w.strip()
    }
    bw = {
        w.strip().lower()
        for w in str(b or "").replace(".", " ").replace(",", " ").split()
        if w.strip()
    }

    if not aw or not bw:
        return 0.0

    return float(len(aw & bw)) / float(max(1, len(aw | bw)))


# ----------------------------------------------------------------------
# Config / records
# ----------------------------------------------------------------------


@dataclass
class TargetEvidenceManagerConfig:
    """
    Task2-first target evidence manager.

    Core rule:
        Task2 decides whether a candidate is worth approaching.
        Depth / 3D is only an anchor for where to approach.

    This module does not output actions and does not switch modes.
    Step 14 should call get_best_approach_candidate().
    """

    max_tracks: int = 32

    # Visual matching for candidates without trusting noisy depth.
    visual_match_pose_distance: float = 18.0
    visual_match_bbox_distance: float = 0.35
    visual_match_step_gap: int = 24
    visual_label_overlap_threshold: float = 0.20

    # Spatial matching is secondary. It is used only after visual matching fails.
    spatial_match_distance: float = 8.0

    # Task2 admission.
    maybe_pass_count: int = 2
    reject_no_margin: int = 2
    stale_lost_steps: int = 45
    recent_support_steps: int = 35

    # Ranking among already Task2-passed candidates.
    spatial_anchor_bonus: float = 0.08
    recent_bonus_weight: float = 0.12
    bbox_bonus_weight: float = 0.06

    position_history_size: int = 6

    # Step 15: Approach rollback / failure feedback.
    # These fields do not create new navigation states. They only decide when a
    # currently active approach target should be demoted so that Search can resume.
    approach_feedback_cooldown_steps: int = 8
    approach_no_progress_limit: int = 6
    approach_rotate_loop_limit: int = 8
    approach_support_timeout_steps: int = 14
    approach_max_steps_without_support: int = 18
    approach_negative_reject_margin: int = 2

    def __post_init__(self) -> None:
        self.max_tracks = max(1, int(self.max_tracks))
        self.visual_match_pose_distance = float(self.visual_match_pose_distance)
        self.visual_match_bbox_distance = float(self.visual_match_bbox_distance)
        self.visual_match_step_gap = max(1, int(self.visual_match_step_gap))
        self.visual_label_overlap_threshold = float(self.visual_label_overlap_threshold)
        self.spatial_match_distance = float(self.spatial_match_distance)
        self.maybe_pass_count = max(1, int(self.maybe_pass_count))
        self.reject_no_margin = max(1, int(self.reject_no_margin))
        self.stale_lost_steps = max(1, int(self.stale_lost_steps))
        self.recent_support_steps = max(1, int(self.recent_support_steps))
        self.spatial_anchor_bonus = float(self.spatial_anchor_bonus)
        self.recent_bonus_weight = float(self.recent_bonus_weight)
        self.bbox_bonus_weight = float(self.bbox_bonus_weight)
        self.position_history_size = max(1, int(self.position_history_size))
        self.approach_feedback_cooldown_steps = max(1, int(self.approach_feedback_cooldown_steps))
        self.approach_no_progress_limit = max(1, int(self.approach_no_progress_limit))
        self.approach_rotate_loop_limit = max(1, int(self.approach_rotate_loop_limit))
        self.approach_support_timeout_steps = max(1, int(self.approach_support_timeout_steps))
        self.approach_max_steps_without_support = max(1, int(self.approach_max_steps_without_support))
        self.approach_negative_reject_margin = max(1, int(self.approach_negative_reject_margin))


@dataclass
class TargetObservation:
    candidate_id: str
    episode_id: str
    step_id: int
    view_id: str
    frame_id: str
    label: str

    decision: Task2Decision
    confidence: float
    reason: str = ""

    bbox: Dict[str, Any] = field(default_factory=dict)
    bbox_center_norm: Optional[Tuple[float, float]] = None
    bbox_area_ratio: float = 0.0

    source_pose: Dict[str, Any] = field(default_factory=dict)
    source_xy: Optional[Tuple[float, float]] = None

    position_3d: Optional[Tuple[float, float, float]] = None
    position_confidence: float = 0.0
    depth_valid: bool = False

    gdino_score: float = 0.0
    quality_score: float = 0.0
    geometry: Dict[str, Any] = field(default_factory=dict)

    task2_result: Optional[Task2Result] = None
    created_at: float = field(default_factory=now_ts)

    @staticmethod
    def from_task2_result(result: Task2Result) -> "TargetObservation":
        snapshot = (result.metadata or {}).get("candidate_snapshot", {}) or {}
        bbox = snapshot.get("bbox", {}) or {}
        source_pose = snapshot.get("source_pose", {}) or {}

        decision = Task2Decision.from_any(result.decision)

        return TargetObservation(
            candidate_id=result.candidate_id,
            episode_id=result.episode_id,
            step_id=int(result.return_step),
            view_id=str(snapshot.get("view_id", "")),
            frame_id=str(snapshot.get("frame_id", "")),
            label=str(snapshot.get("label", "")),
            decision=decision,
            confidence=clamp01(result.confidence),
            reason=result.reason or "",
            bbox=bbox,
            bbox_center_norm=_bbox_center_norm(bbox),
            bbox_area_ratio=_bbox_area_ratio(bbox),
            source_pose=source_pose,
            source_xy=_pose_xy(source_pose),
            position_3d=_as_position(snapshot.get("position_3d")),
            position_confidence=clamp01(snapshot.get("position_confidence", 0.0)),
            depth_valid=bool(snapshot.get("depth_valid", False)),
            gdino_score=clamp01(snapshot.get("score", 0.0)),
            quality_score=clamp01(snapshot.get("quality_score", 0.0)),
            geometry=snapshot.get("geometry", {}) or {},
            task2_result=result,
        )

    def is_positive_like(self) -> bool:
        return self.decision in (Task2Decision.YES, Task2Decision.MAYBE)

    def has_spatial_anchor(self) -> bool:
        return self.position_3d is not None and self.position_confidence > 0.0

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "episode_id": self.episode_id,
            "step_id": int(self.step_id),
            "view_id": self.view_id,
            "frame_id": self.frame_id,
            "label": self.label,
            "decision": self.decision.value,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "bbox": self.bbox,
            "bbox_center_norm": self.bbox_center_norm,
            "bbox_area_ratio": float(self.bbox_area_ratio),
            "source_pose": self.source_pose,
            "position_3d": self.position_3d,
            "position_confidence": float(self.position_confidence),
            "depth_valid": bool(self.depth_valid),
            "gdino_score": float(self.gdino_score),
            "quality_score": float(self.quality_score),
            "geometry": self.geometry,
        }


@dataclass
class EvidenceTrack:
    evidence: TargetEvidence
    observations: List[TargetObservation] = field(default_factory=list)

    position_history: List[Tuple[float, float, float]] = field(default_factory=list)
    position_confidence_history: List[float] = field(default_factory=list)

    task2_admission: str = "pending"  # pending / passed / rejected
    anchor_type: str = "visual"       # visual / spatial

    task2_score: float = 0.0
    rank_score: float = 0.0
    approach_ready: bool = False

    last_positive_step: Optional[int] = None
    last_update_step: int = 0
    created_at: float = field(default_factory=now_ts)

    def latest_observation(self) -> Optional[TargetObservation]:
        if not self.observations:
            return None
        return self.observations[-1]

    def to_log_dict(self) -> Dict[str, Any]:
        data = self.evidence.to_log_dict()
        data["task2_admission"] = self.task2_admission
        data["anchor_type"] = self.anchor_type
        data["task2_score"] = float(self.task2_score)
        data["rank_score"] = float(self.rank_score)
        data["approach_score"] = float(self.rank_score)
        data["approach_ready"] = bool(self.approach_ready)
        data["last_positive_step"] = self.last_positive_step
        data["last_update_step"] = int(self.last_update_step)
        data["position_history"] = list(self.position_history)
        data["position_confidence_history"] = list(self.position_confidence_history)
        data["latest_observation"] = None if self.latest_observation() is None else self.latest_observation().to_log_dict()
        return data


@dataclass
class TargetEvidenceUpdate:
    episode_id: str
    step_id: int

    updated_track_ids: List[str] = field(default_factory=list)
    created_track_ids: List[str] = field(default_factory=list)
    rejected_track_ids: List[str] = field(default_factory=list)
    tentative_track_ids: List[str] = field(default_factory=list)
    verified_track_ids: List[str] = field(default_factory=list)
    approach_candidate_ids: List[str] = field(default_factory=list)

    ignored_results: Dict[str, str] = field(default_factory=dict)

    best_evidence: Optional[TargetEvidence] = None
    best_approach_candidate: Optional[TargetEvidence] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_id": int(self.step_id),
            "updated_track_ids": list(self.updated_track_ids),
            "created_track_ids": list(self.created_track_ids),
            "rejected_track_ids": list(self.rejected_track_ids),
            "tentative_track_ids": list(self.tentative_track_ids),
            "verified_track_ids": list(self.verified_track_ids),
            "approach_candidate_ids": list(self.approach_candidate_ids),
            "ignored_results": dict(self.ignored_results),
            "best_evidence": None if self.best_evidence is None else self.best_evidence.to_log_dict(),
            "best_approach_candidate": None if self.best_approach_candidate is None else self.best_approach_candidate.to_log_dict(),
            "summary": dict(self.summary),
        }




@dataclass
class ApproachFeedback:
    """
    Feedback generated while a candidate is being approached.

    This is not a navigation state. It is only evidence feedback:
        keep   -> continue approaching the active target
        demote -> active approach attempt failed; return to Search
        reject -> strong negative evidence; do not approach this target again
    """

    target_id: str
    episode_id: str
    step_id: int

    action: str = ""
    mode: str = ""
    phase: str = ""

    distance_to_anchor: Optional[float] = None
    previous_distance_to_anchor: Optional[float] = None
    distance_progress: Optional[float] = None

    support_age_steps: Optional[int] = None
    steps_since_attempt: int = 0

    no_progress_count: int = 0
    rotate_loop_count: int = 0

    negative_count: int = 0
    positive_count: int = 0
    maybe_count: int = 0

    anchor_type: str = ""
    anchor_unreliable: bool = False

    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "episode_id": self.episode_id,
            "step_id": int(self.step_id),
            "action": self.action,
            "mode": self.mode,
            "phase": self.phase,
            "distance_to_anchor": self.distance_to_anchor,
            "previous_distance_to_anchor": self.previous_distance_to_anchor,
            "distance_progress": self.distance_progress,
            "support_age_steps": self.support_age_steps,
            "steps_since_attempt": int(self.steps_since_attempt),
            "no_progress_count": int(self.no_progress_count),
            "rotate_loop_count": int(self.rotate_loop_count),
            "negative_count": int(self.negative_count),
            "positive_count": int(self.positive_count),
            "maybe_count": int(self.maybe_count),
            "anchor_type": self.anchor_type,
            "anchor_unreliable": bool(self.anchor_unreliable),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass
class ApproachFeedbackResult:
    target_id: str
    episode_id: str
    step_id: int
    decision: str
    reason: str
    blocked_until_step: Optional[int] = None
    feedback: Optional[ApproachFeedback] = None

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "episode_id": self.episode_id,
            "step_id": int(self.step_id),
            "decision": self.decision,
            "reason": self.reason,
            "blocked_until_step": self.blocked_until_step,
            "feedback": None if self.feedback is None else self.feedback.to_log_dict(),
        }


# ----------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------


class TargetEvidenceManager:
    """
    TargetEvidenceManager, rewritten for Step 13.

    It is no longer a depth/geometry-based target manager.

    New contract:
        1. Task2 decides admission:
             yes -> passed
             repeated maybe -> passed
             no -> negative evidence / rejected
        2. Depth / geometry only provides an optional spatial anchor.
        3. get_best_approach_candidate() only returns Task2-passed tracks.
    """

    def __init__(self, config: Optional[TargetEvidenceManagerConfig] = None) -> None:
        self.config = config or TargetEvidenceManagerConfig()
        self.current_episode_id: Optional[str] = None
        self.tracks: Dict[str, EvidenceTrack] = {}
        self.candidate_to_track: Dict[str, str] = {}
        self.latest_update: Optional[TargetEvidenceUpdate] = None

    def reset_episode(self, episode_id: str) -> None:
        self.current_episode_id = str(episode_id)
        self.tracks.clear()
        self.candidate_to_track.clear()
        self.latest_update = None

    def update_from_task2_results(
        self,
        results: List[Task2Result],
        current_step: int,
    ) -> TargetEvidenceUpdate:
        episode_id = self._infer_episode_id(results)

        if self.current_episode_id is None:
            self.reset_episode(episode_id)

        if episode_id != self.current_episode_id:
            self.reset_episode(episode_id)

        update = TargetEvidenceUpdate(
            episode_id=episode_id,
            step_id=int(current_step),
        )

        for result in results or []:
            if result.episode_id != self.current_episode_id:
                update.ignored_results[result.candidate_id] = "episode_mismatch"
                continue

            if result.decision == Task2Decision.UNKNOWN and not result.success:
                update.ignored_results[result.candidate_id] = "task2_error"
                continue

            observation = TargetObservation.from_task2_result(result)
            track, created, reason = self._match_or_create_track(observation)

            if track is None:
                update.ignored_results[result.candidate_id] = reason
                continue

            self._update_track(track, observation, current_step=int(current_step))
            self.candidate_to_track[observation.candidate_id] = track.evidence.target_id

            update.updated_track_ids.append(track.evidence.target_id)
            if created:
                update.created_track_ids.append(track.evidence.target_id)

        self._refresh_all_tracks(current_step=int(current_step))
        self._prune_tracks()

        for track in self.tracks.values():
            status = track.evidence.status

            if status == CandidateStatus.VERIFIED:
                update.verified_track_ids.append(track.evidence.target_id)
            elif status == CandidateStatus.REJECTED:
                update.rejected_track_ids.append(track.evidence.target_id)
            elif status == CandidateStatus.TENTATIVE:
                update.tentative_track_ids.append(track.evidence.target_id)

            if track.approach_ready:
                update.approach_candidate_ids.append(track.evidence.target_id)

        update.best_approach_candidate = self.get_best_approach_candidate()
        update.best_evidence = self.get_best_evidence(allow_tentative=True)
        update.summary = self._build_summary()

        self.latest_update = update
        return update

    def _infer_episode_id(self, results: List[Task2Result]) -> str:
        for result in results or []:
            if result.episode_id:
                return result.episode_id
        return self.current_episode_id or "unknown_episode"

    # ------------------------------------------------------------------
    # Track matching
    # ------------------------------------------------------------------

    def _match_or_create_track(
        self,
        observation: TargetObservation,
    ) -> Tuple[Optional[EvidenceTrack], bool, str]:
        existing_id = self.candidate_to_track.get(observation.candidate_id)
        if existing_id and existing_id in self.tracks:
            return self.tracks[existing_id], False, "matched_candidate_id"

        track = self._match_visual_track(observation)
        if track is not None:
            return track, False, "matched_visual"

        track = self._match_spatial_track(observation)
        if track is not None:
            return track, False, "matched_spatial"

        # A pure no result without a matched track does not create a rejected
        # target. It only says this one unmatched candidate is not useful.
        if observation.decision == Task2Decision.NO:
            return None, False, "unmatched_negative"

        track = self._create_track(observation)
        return track, True, "created"

    def _match_visual_track(self, observation: TargetObservation) -> Optional[EvidenceTrack]:
        best_track = None
        best_score = -1.0

        for track in self.tracks.values():
            if track.evidence.status == CandidateStatus.REJECTED:
                continue

            latest = track.latest_observation()
            if latest is None:
                continue

            step_gap = abs(int(observation.step_id) - int(latest.step_id))
            if step_gap > self.config.visual_match_step_gap:
                continue

            bbox_dist = _distance_2d(observation.bbox_center_norm, latest.bbox_center_norm)
            bbox_score = 0.0
            if math.isfinite(bbox_dist):
                # If both have bbox and are far apart in image, they are not
                # the same visual candidate.
                if bbox_dist > self.config.visual_match_bbox_distance:
                    continue
                bbox_score = max(
                    0.0,
                    1.0 - bbox_dist / max(1e-6, self.config.visual_match_bbox_distance),
                )

            pose_dist = _distance_2d(observation.source_xy, latest.source_xy)
            if math.isfinite(pose_dist):
                pose_score = max(
                    0.0,
                    1.0 - pose_dist / max(1e-6, self.config.visual_match_pose_distance),
                )
            else:
                pose_score = 0.35

            view_score = 1.0 if observation.view_id == latest.view_id else 0.45

            label_score = _label_overlap(observation.label, latest.label)
            if label_score < self.config.visual_label_overlap_threshold and observation.label and latest.label:
                continue

            score = (
                0.34 * bbox_score
                + 0.28 * pose_score
                + 0.22 * view_score
                + 0.16 * max(label_score, 0.20)
            )

            if score > best_score:
                best_score = score
                best_track = track

        if best_track is not None and best_score >= 0.42:
            return best_track

        return None

    def _match_spatial_track(self, observation: TargetObservation) -> Optional[EvidenceTrack]:
        if not observation.has_spatial_anchor():
            return None

        best_track = None
        best_distance = float("inf")

        for track in self.tracks.values():
            if track.evidence.status == CandidateStatus.REJECTED:
                continue

            if track.evidence.position_3d is None:
                continue

            distance = _distance_3d(observation.position_3d, track.evidence.position_3d)
            if distance < best_distance:
                best_distance = distance
                best_track = track

        if best_track is not None and best_distance <= self.config.spatial_match_distance:
            return best_track

        return None

    def _create_track(self, observation: TargetObservation) -> EvidenceTrack:
        track_id = new_id("target")
        evidence = TargetEvidence(
            target_id=track_id,
            episode_id=observation.episode_id,
            status=CandidateStatus.NEW,
        )

        track = EvidenceTrack(
            evidence=evidence,
            last_update_step=int(observation.step_id),
            anchor_type="spatial" if observation.has_spatial_anchor() else "visual",
        )

        self.tracks[track_id] = track
        return track

    # ------------------------------------------------------------------
    # Track update
    # ------------------------------------------------------------------

    def _update_track(
        self,
        track: EvidenceTrack,
        observation: TargetObservation,
        current_step: int,
    ) -> None:
        track.observations.append(observation)
        track.last_update_step = int(current_step)

        evidence = track.evidence

        if observation.candidate_id not in evidence.source_candidate_ids:
            evidence.source_candidate_ids.append(observation.candidate_id)

        evidence.latest_candidate_id = observation.candidate_id
        evidence.seen_count += 1

        if evidence.first_seen_step is None:
            evidence.first_seen_step = observation.step_id
        evidence.last_seen_step = observation.step_id

        if observation.is_positive_like():
            evidence.last_verified_step = observation.step_id
            track.last_positive_step = observation.step_id

        if observation.reason:
            evidence.reasons.append(observation.reason)

        self._update_task2_counts(track, observation)
        self._update_anchor(track, observation)
        self._refresh_track(track, current_step=current_step)
        self._update_metadata(track, observation)

    def _update_task2_counts(
        self,
        track: EvidenceTrack,
        observation: TargetObservation,
    ) -> None:
        evidence = track.evidence

        if observation.decision == Task2Decision.YES:
            evidence.positive_count += 1
            if evidence.status == CandidateStatus.NEW:
                evidence.status = CandidateStatus.TENTATIVE

        elif observation.decision == Task2Decision.MAYBE:
            evidence.maybe_count += 1
            if evidence.status == CandidateStatus.NEW:
                evidence.status = CandidateStatus.TENTATIVE

        elif observation.decision == Task2Decision.NO:
            evidence.negative_count += 1

        evidence.verify_score = self._compute_task2_score(track)

    def _update_anchor(
        self,
        track: EvidenceTrack,
        observation: TargetObservation,
    ) -> None:
        evidence = track.evidence

        if observation.has_spatial_anchor():
            track.anchor_type = "spatial"
            track.position_history.append(observation.position_3d)
            track.position_confidence_history.append(clamp01(observation.position_confidence))

            if len(track.position_history) > self.config.position_history_size:
                track.position_history = track.position_history[-self.config.position_history_size :]
                track.position_confidence_history = track.position_confidence_history[-self.config.position_history_size :]

            evidence.position_3d = self._fuse_position(track)
            evidence.position_confidence = max(
                evidence.position_confidence,
                clamp01(observation.position_confidence),
            )
            evidence.position_stability = self._position_stability(track)

        # Always update visual anchor information. Even spatial candidates need
        # source_pose/view/bbox for inspection and debugging.
        evidence.metadata["latest_source_pose"] = observation.source_pose
        evidence.metadata["latest_view_id"] = observation.view_id
        evidence.metadata["latest_bbox"] = observation.bbox
        evidence.metadata["latest_bbox_center_norm"] = observation.bbox_center_norm
        evidence.metadata["latest_bbox_area_ratio"] = observation.bbox_area_ratio
        evidence.metadata["latest_frame_id"] = observation.frame_id
        evidence.metadata["latest_label"] = observation.label
        evidence.metadata["latest_geometry"] = observation.geometry

    def _fuse_position(self, track: EvidenceTrack) -> Optional[Tuple[float, float, float]]:
        if not track.position_history:
            return None

        weights = [
            max(0.05, clamp01(c))
            for c in track.position_confidence_history
        ]
        total = sum(weights)
        if total <= 1e-6:
            weights = [1.0 for _ in track.position_history]
            total = float(len(weights))

        x = sum(p[0] * w for p, w in zip(track.position_history, weights)) / total
        y = sum(p[1] * w for p, w in zip(track.position_history, weights)) / total
        z = sum(p[2] * w for p, w in zip(track.position_history, weights)) / total

        return (float(x), float(y), float(z))

    def _position_stability(self, track: EvidenceTrack) -> float:
        if len(track.position_history) <= 1:
            conf = max(track.position_confidence_history) if track.position_confidence_history else 0.0
            return clamp01(0.40 * conf)

        center = self._fuse_position(track)
        if center is None:
            return 0.0

        dists = [
            _distance_3d(position, center)
            for position in track.position_history
        ]
        mean_dist = sum(dists) / float(max(1, len(dists)))
        geometric_stability = 1.0 - min(1.0, mean_dist / max(1e-6, self.config.spatial_match_distance))
        mean_conf = sum(track.position_confidence_history) / float(max(1, len(track.position_confidence_history)))

        return clamp01(0.65 * geometric_stability + 0.35 * mean_conf)

    # ------------------------------------------------------------------
    # Admission / ranking
    # ------------------------------------------------------------------

    def _refresh_all_tracks(self, current_step: int) -> None:
        for track in self.tracks.values():
            self._refresh_track(track, current_step=current_step)

    def _refresh_track(
        self,
        track: EvidenceTrack,
        current_step: int,
    ) -> None:
        evidence = track.evidence

        if evidence.status == CandidateStatus.REJECTED:
            track.task2_admission = "rejected"
            track.approach_ready = False
            track.rank_score = 0.0
            return

        if evidence.last_seen_step is not None:
            age = int(current_step) - int(evidence.last_seen_step)
            if age > self.config.stale_lost_steps:
                if evidence.status == CandidateStatus.VERIFIED:
                    evidence.set_lost("target evidence became stale")
                track.approach_ready = False

        if self._should_reject(track):
            track.task2_admission = "rejected"
            track.approach_ready = False
            track.rank_score = 0.0
            evidence.set_rejected("Task2 negative evidence dominates")
            self._write_admission_metadata(track)
            return

        if evidence.status == CandidateStatus.NEW and evidence.seen_count > 0:
            evidence.status = CandidateStatus.TENTATIVE

        if self._is_task2_passed(track, current_step=current_step):
            track.task2_admission = "passed"
            track.approach_ready = True
        else:
            track.task2_admission = "pending"
            track.approach_ready = False

        track.task2_score = self._compute_task2_score(track)
        track.rank_score = self._compute_rank_score(track, current_step=current_step)

        self._write_admission_metadata(track)

    def _is_task2_passed(
        self,
        track: EvidenceTrack,
        current_step: int,
    ) -> bool:
        evidence = track.evidence

        if evidence.status in (CandidateStatus.REJECTED, CandidateStatus.LOST):
            return False

        if track.last_positive_step is None:
            return False

        recent_age = int(current_step) - int(track.last_positive_step)
        if recent_age > self.config.recent_support_steps:
            return False

        # Task2 yes is a direct admission signal.
        if evidence.positive_count >= 1:
            if evidence.negative_count >= evidence.positive_count + evidence.maybe_count + self.config.reject_no_margin:
                return False
            return True

        # Repeated maybe is allowed. This handles cautious Task2 outputs.
        if evidence.maybe_count >= self.config.maybe_pass_count:
            if evidence.negative_count >= evidence.maybe_count + self.config.reject_no_margin:
                return False
            return True

        return False

    def _should_reject(self, track: EvidenceTrack) -> bool:
        evidence = track.evidence

        positive_support = int(evidence.positive_count) + int(evidence.maybe_count)

        if positive_support <= 0:
            return evidence.negative_count >= self.config.reject_no_margin

        return evidence.negative_count >= positive_support + self.config.reject_no_margin

    def _compute_task2_score(self, track: EvidenceTrack) -> float:
        evidence = track.evidence

        # This is only a Task2 evidence summary. It is not allowed to use depth.
        score = (
            1.00 * float(evidence.positive_count)
            + 0.45 * float(evidence.maybe_count)
            - 0.75 * float(evidence.negative_count)
        )

        return clamp01(score / 2.0)

    def _compute_rank_score(
        self,
        track: EvidenceTrack,
        current_step: int,
    ) -> float:
        evidence = track.evidence

        if track.task2_admission != "passed":
            return 0.0

        task2_score = self._compute_task2_score(track)

        recent = 0.0
        if track.last_positive_step is not None:
            age = max(0, int(current_step) - int(track.last_positive_step))
            recent = max(0.0, 1.0 - age / float(max(1, self.config.recent_support_steps)))

        latest = track.latest_observation()
        bbox_bonus = 0.0
        if latest is not None:
            bbox_bonus = self._bbox_quality_bonus(latest)

        spatial_bonus = 0.0
        if track.anchor_type == "spatial":
            # This is deliberately small. Depth helps choose between Task2-
            # passed candidates, but it never creates admission by itself.
            spatial_bonus = min(
                self.config.spatial_anchor_bonus,
                self.config.spatial_anchor_bonus
                * (
                    0.55 * clamp01(evidence.position_confidence)
                    + 0.45 * clamp01(evidence.position_stability)
                ),
            )

        score = (
            task2_score
            + self.config.recent_bonus_weight * recent
            + self.config.bbox_bonus_weight * bbox_bonus
            + spatial_bonus
        )

        return clamp01(score)

    def _bbox_quality_bonus(self, observation: TargetObservation) -> float:
        area = clamp01(observation.bbox_area_ratio)

        if area <= 0.0:
            return 0.0
        if area < 0.002:
            return 0.15
        if area < 0.05:
            return 0.70
        if area < 0.35:
            return 1.00
        if area < 0.65:
            return 0.55
        return 0.20

    def _write_admission_metadata(self, track: EvidenceTrack) -> None:
        evidence = track.evidence

        visual_anchor = {
            "type": "visual",
            "source_pose": evidence.metadata.get("latest_source_pose"),
            "view_id": evidence.metadata.get("latest_view_id"),
            "bbox": evidence.metadata.get("latest_bbox"),
            "bbox_center_norm": evidence.metadata.get("latest_bbox_center_norm"),
            "frame_id": evidence.metadata.get("latest_frame_id"),
        }

        spatial_anchor = None
        if evidence.position_3d is not None:
            spatial_anchor = {
                "type": "spatial",
                "position_3d": evidence.position_3d,
                "position_confidence": float(evidence.position_confidence),
                "position_stability": float(evidence.position_stability),
            }

        anchor_type = "spatial" if spatial_anchor is not None else "visual"
        track.anchor_type = anchor_type

        evidence.metadata.update(
            {
                "task2_admission": track.task2_admission,
                "approach_ready": bool(track.approach_ready),
                "rank_score": float(track.rank_score),
                "approach_score": float(track.rank_score),
                "task2_score": float(track.task2_score),
                "anchor_type": anchor_type,
                "evidence_kind": anchor_type,
                "visual_score": float(track.task2_score),
                "negative_score": float(evidence.negative_count),
                "visual_anchor": visual_anchor,
                "spatial_anchor": spatial_anchor,
                "approach_anchor": spatial_anchor or visual_anchor,
                "last_positive_step": track.last_positive_step,
            }
        )

    def _update_metadata(
        self,
        track: EvidenceTrack,
        observation: TargetObservation,
    ) -> None:
        evidence = track.evidence

        evidence.metadata.update(
            {
                "latest_observation": observation.to_log_dict(),
                "latest_task2_decision": observation.decision.value,
                "latest_task2_confidence": float(observation.confidence),
                "latest_task2_reason": observation.reason,
            }
        )
        self._write_admission_metadata(track)

    # ------------------------------------------------------------------
    # Step 15: Approach feedback / rollback
    # ------------------------------------------------------------------

    def get_track_by_id(self, target_id: str) -> Optional[EvidenceTrack]:
        return self.tracks.get(str(target_id))

    def get_evidence_by_id(self, target_id: str) -> Optional[TargetEvidence]:
        track = self.get_track_by_id(str(target_id))
        if track is None:
            return None
        self._write_admission_metadata(track)
        return track.evidence

    def mark_approach_attempt(
        self,
        target_id: str,
        step_id: int,
        reason: str = "selected_for_approach",
    ) -> Dict[str, Any]:
        track = self.get_track_by_id(str(target_id))
        if track is None:
            return {
                "target_id": target_id,
                "step_id": int(step_id),
                "updated": False,
                "reason": "missing_target",
            }

        meta = track.evidence.metadata
        meta["active_approach"] = True
        meta["last_approach_attempt_step"] = int(step_id)
        meta["approach_attempt_count"] = int(meta.get("approach_attempt_count", 0)) + 1
        meta["last_approach_attempt_reason"] = reason

        self._write_admission_metadata(track)

        return {
            "target_id": target_id,
            "step_id": int(step_id),
            "updated": True,
            "reason": reason,
            "approach_attempt_count": meta.get("approach_attempt_count", 0),
        }

    def apply_approach_feedback(
        self,
        target_id: str,
        feedback: ApproachFeedback,
    ) -> ApproachFeedbackResult:
        track = self.get_track_by_id(str(target_id))
        if track is None:
            return ApproachFeedbackResult(
                target_id=str(target_id),
                episode_id=feedback.episode_id,
                step_id=int(feedback.step_id),
                decision="missing",
                reason="target_not_found",
                feedback=feedback,
            )

        evidence = track.evidence

        # Strong negative Task2 evidence can reject the target.
        positive_support = int(evidence.positive_count) + int(evidence.maybe_count)
        negative_support = int(evidence.negative_count)

        if negative_support >= positive_support + self.config.approach_negative_reject_margin:
            result = self._reject_after_approach(
                track=track,
                feedback=feedback,
                reason="negative_task2_evidence_dominates",
            )
            return result

        # Spatial anchor is only an anchor. If it becomes unreliable during
        # approach, demote to Search instead of forcing continued approach.
        if feedback.anchor_unreliable:
            result = self._demote_after_approach(
                track=track,
                feedback=feedback,
                reason="anchor_unreliable",
            )
            return result

        support_age = feedback.support_age_steps
        if support_age is None:
            support_age = 10 ** 6

        # No progress without fresh Task2 support: rollback, not rejection.
        if (
            support_age >= self.config.approach_support_timeout_steps
            and feedback.no_progress_count >= self.config.approach_no_progress_limit
        ):
            result = self._demote_after_approach(
                track=track,
                feedback=feedback,
                reason="no_progress_without_recent_task2_support",
            )
            return result

        # Rotation loop without fresh support: rollback.
        if (
            support_age >= self.config.approach_support_timeout_steps
            and feedback.rotate_loop_count >= self.config.approach_rotate_loop_limit
        ):
            result = self._demote_after_approach(
                track=track,
                feedback=feedback,
                reason="rotate_loop_without_recent_task2_support",
            )
            return result

        # Active target has been pursued for too long without new support.
        if (
            support_age >= self.config.approach_support_timeout_steps
            and feedback.steps_since_attempt >= self.config.approach_max_steps_without_support
        ):
            result = self._demote_after_approach(
                track=track,
                feedback=feedback,
                reason="approach_timeout_without_recent_task2_support",
            )
            return result

        evidence.metadata["last_approach_feedback"] = feedback.to_log_dict()
        evidence.metadata["last_approach_feedback_decision"] = "keep"
        evidence.metadata["last_approach_feedback_reason"] = "active_target_still_valid"
        self._write_admission_metadata(track)

        return ApproachFeedbackResult(
            target_id=str(target_id),
            episode_id=feedback.episode_id,
            step_id=int(feedback.step_id),
            decision="keep",
            reason="active_target_still_valid",
            feedback=feedback,
        )

    def _demote_after_approach(
        self,
        track: EvidenceTrack,
        feedback: ApproachFeedback,
        reason: str,
    ) -> ApproachFeedbackResult:
        evidence = track.evidence

        blocked_until = int(feedback.step_id) + self.config.approach_feedback_cooldown_steps

        track.task2_admission = "pending"
        track.approach_ready = False
        track.rank_score = 0.0

        # Keep the track tentative. Demotion means "this approach attempt failed",
        # not "this object is definitely wrong".
        if evidence.status != CandidateStatus.REJECTED:
            evidence.status = CandidateStatus.TENTATIVE

        meta = evidence.metadata
        meta["active_approach"] = False
        meta["approach_blocked_until_step"] = blocked_until
        meta["approach_failure_count"] = int(meta.get("approach_failure_count", 0)) + 1
        meta["last_approach_feedback"] = feedback.to_log_dict()
        meta["last_approach_feedback_decision"] = "demote"
        meta["last_approach_feedback_reason"] = reason

        self._write_admission_metadata(track)

        return ApproachFeedbackResult(
            target_id=evidence.target_id,
            episode_id=feedback.episode_id,
            step_id=int(feedback.step_id),
            decision="demote",
            reason=reason,
            blocked_until_step=blocked_until,
            feedback=feedback,
        )

    def _reject_after_approach(
        self,
        track: EvidenceTrack,
        feedback: ApproachFeedback,
        reason: str,
    ) -> ApproachFeedbackResult:
        evidence = track.evidence

        track.task2_admission = "rejected"
        track.approach_ready = False
        track.rank_score = 0.0

        evidence.set_rejected(reason)

        meta = evidence.metadata
        meta["active_approach"] = False
        meta["approach_blocked_until_step"] = None
        meta["approach_failure_count"] = int(meta.get("approach_failure_count", 0)) + 1
        meta["last_approach_feedback"] = feedback.to_log_dict()
        meta["last_approach_feedback_decision"] = "reject"
        meta["last_approach_feedback_reason"] = reason

        self._write_admission_metadata(track)

        return ApproachFeedbackResult(
            target_id=evidence.target_id,
            episode_id=feedback.episode_id,
            step_id=int(feedback.step_id),
            decision="reject",
            reason=reason,
            blocked_until_step=None,
            feedback=feedback,
        )

    def _is_approach_blocked(
        self,
        track: EvidenceTrack,
        current_step: Optional[int] = None,
    ) -> bool:
        blocked_until = track.evidence.metadata.get("approach_blocked_until_step")
        if blocked_until is None:
            return False

        try:
            blocked_until = int(blocked_until)
        except Exception:
            return False

        if current_step is None:
            return blocked_until > 0

        return int(current_step) < blocked_until


    # ------------------------------------------------------------------
    # Public selection
    # ------------------------------------------------------------------

    def get_best_approach_candidate(
        self,
        exclude_target_ids: Optional[List[str]] = None,
        current_step: Optional[int] = None,
    ) -> Optional[TargetEvidence]:
        exclude_set = set(exclude_target_ids or [])

        best_track = None
        best_score = -1.0

        for track in self.tracks.values():
            target_id = track.evidence.target_id

            if target_id in exclude_set:
                continue

            if not track.approach_ready:
                continue

            if track.task2_admission != "passed":
                continue

            if track.evidence.status in (CandidateStatus.REJECTED, CandidateStatus.LOST):
                continue

            if self._is_approach_blocked(track, current_step=current_step):
                continue

            score = float(track.rank_score)
            if score > best_score:
                best_score = score
                best_track = track

        if best_track is None:
            return None

        self._write_admission_metadata(best_track)
        return best_track.evidence

    def get_best_verified_evidence(self) -> Optional[TargetEvidence]:
        # Kept for compatibility. The new Step 13 does not require verified
        # spatial target to enter Approach. Step 14 should use
        # get_best_approach_candidate().
        return self.get_best_evidence(allow_tentative=False)

    def get_best_evidence(self, allow_tentative: bool = False) -> Optional[TargetEvidence]:
        best_track = None
        best_score = -1.0

        for track in self.tracks.values():
            status = track.evidence.status

            if status in (CandidateStatus.REJECTED, CandidateStatus.LOST):
                continue

            if status != CandidateStatus.VERIFIED and not allow_tentative:
                continue

            # For logging, allow best tentative evidence even if not approach
            # ready. This should not be used to trigger Approach.
            score = float(track.rank_score)
            if score <= 0.0:
                score = self._compute_rank_score(track, current_step=track.last_update_step)

            if score > best_score:
                best_score = score
                best_track = track

        if best_track is None:
            return None

        self._write_admission_metadata(best_track)
        return best_track.evidence

    def get_verified_evidence(self) -> List[TargetEvidence]:
        return [
            track.evidence
            for track in self.tracks.values()
            if track.evidence.status == CandidateStatus.VERIFIED
        ]

    def get_tentative_evidence(self) -> List[TargetEvidence]:
        return [
            track.evidence
            for track in self.tracks.values()
            if track.evidence.status == CandidateStatus.TENTATIVE
        ]

    def _prune_tracks(self) -> None:
        if len(self.tracks) <= self.config.max_tracks:
            return

        ranked = sorted(
            self.tracks.values(),
            key=lambda track: track.rank_score,
            reverse=True,
        )
        keep_ids = {
            track.evidence.target_id
            for track in ranked[: self.config.max_tracks]
        }

        for track_id in list(self.tracks.keys()):
            if track_id not in keep_ids:
                self.tracks.pop(track_id, None)

        self.candidate_to_track = {
            candidate_id: track_id
            for candidate_id, track_id in self.candidate_to_track.items()
            if track_id in self.tracks
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        status_counts = {
            "new": 0,
            "tentative": 0,
            "verified": 0,
            "rejected": 0,
            "lost": 0,
        }

        anchor_counts = {
            "visual": 0,
            "spatial": 0,
        }

        admission_counts = {
            "pending": 0,
            "passed": 0,
            "rejected": 0,
        }

        approach_ready_count = 0
        best_approach = self.get_best_approach_candidate()
        best_evidence = self.get_best_evidence(allow_tentative=True)

        for track in self.tracks.values():
            status = track.evidence.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            anchor_counts[track.anchor_type] = anchor_counts.get(track.anchor_type, 0) + 1
            admission_counts[track.task2_admission] = admission_counts.get(track.task2_admission, 0) + 1

            if track.approach_ready:
                approach_ready_count += 1

        return {
            "track_count": len(self.tracks),
            "status_counts": status_counts,
            "anchor_counts": anchor_counts,
            "kind_counts": anchor_counts,
            "admission_counts": admission_counts,
            "visual_count": anchor_counts.get("visual", 0),
            "spatial_count": anchor_counts.get("spatial", 0),
            "pending_count": admission_counts.get("pending", 0),
            "passed_count": admission_counts.get("passed", 0),
            "verified_count": status_counts.get("verified", 0),
            "tentative_count": status_counts.get("tentative", 0),
            "rejected_count": status_counts.get("rejected", 0),
            "approach_ready_count": approach_ready_count,
            "best_approach_candidate_id": None if best_approach is None else best_approach.target_id,
            "best_approach_candidate_status": None if best_approach is None else best_approach.status.value,
            "best_approach_candidate_score": None if best_approach is None else best_approach.metadata.get("approach_score"),
            "best_approach_candidate_kind": None if best_approach is None else best_approach.metadata.get("evidence_kind"),
            "best_approach_candidate_admission": None if best_approach is None else best_approach.metadata.get("task2_admission"),
            "best_evidence_id": None if best_evidence is None else best_evidence.target_id,
            "best_evidence_status": None if best_evidence is None else best_evidence.status.value,
            "best_evidence_score": None if best_evidence is None else best_evidence.metadata.get("approach_score"),
            "best_evidence_kind": None if best_evidence is None else best_evidence.metadata.get("evidence_kind"),
            "best_evidence_admission": None if best_evidence is None else best_evidence.metadata.get("task2_admission"),
        }

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "current_episode_id": self.current_episode_id,
            "summary": self._build_summary(),
            "tracks": {
                track_id: track.to_log_dict()
                for track_id, track in self.tracks.items()
            },
            "candidate_to_track": dict(self.candidate_to_track),
        }
