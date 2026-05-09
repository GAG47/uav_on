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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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


@dataclass
class TargetEvidenceManagerConfig:
    """
    Multi-frame target evidence manager.

    This module consumes Task2Result plus candidate_snapshot and maintains
    independent target evidence tracks.

    It does not call GDINO, does not call VLM, does not update semantic_map,
    does not output actions, and does not decide stop.
    """

    track_match_distance: float = 8.0
    max_tracks: int = 32
    position_history_size: int = 6

    positive_score_weight: float = 1.0
    maybe_score_weight: float = 0.45
    negative_score_weight: float = 0.80
    score_decay: float = 0.92

    verified_score_threshold: float = 0.65
    verified_position_confidence: float = 0.35
    verified_position_stability: float = 0.45
    verified_positive_count: int = 2
    verified_maybe_count: int = 3

    rejected_negative_count: int = 2
    strong_negative_confidence: float = 0.85

    stale_lost_steps: int = 35

    def __post_init__(self) -> None:
        self.track_match_distance = float(self.track_match_distance)
        self.max_tracks = int(self.max_tracks)
        self.position_history_size = int(self.position_history_size)
        self.verified_score_threshold = float(self.verified_score_threshold)
        self.verified_position_confidence = float(self.verified_position_confidence)
        self.verified_position_stability = float(self.verified_position_stability)
        self.verified_positive_count = int(self.verified_positive_count)
        self.verified_maybe_count = int(self.verified_maybe_count)
        self.rejected_negative_count = int(self.rejected_negative_count)
        self.strong_negative_confidence = float(self.strong_negative_confidence)
        self.stale_lost_steps = int(self.stale_lost_steps)


@dataclass
class TargetObservation:
    candidate_id: str
    episode_id: str
    step_id: int
    view_id: str
    decision: Task2Decision
    confidence: float
    reason: str = ""

    position_3d: Optional[Tuple[float, float, float]] = None
    position_confidence: float = 0.0
    depth_valid: bool = False

    score: float = 0.0
    quality_score: float = 0.0
    bbox: Dict[str, Any] = field(default_factory=dict)
    geometry: Dict[str, Any] = field(default_factory=dict)
    task2_result: Optional[Task2Result] = None

    created_at: float = field(default_factory=now_ts)

    @staticmethod
    def from_task2_result(result: Task2Result) -> "TargetObservation":
        snapshot = (result.metadata or {}).get("candidate_snapshot", {}) or {}

        return TargetObservation(
            candidate_id=result.candidate_id,
            episode_id=result.episode_id,
            step_id=int(result.return_step),
            view_id=str(snapshot.get("view_id", "")),
            decision=Task2Decision.from_any(result.decision),
            confidence=clamp01(result.confidence),
            reason=result.reason or "",
            position_3d=_as_position(snapshot.get("position_3d")),
            position_confidence=clamp01(snapshot.get("position_confidence", 0.0)),
            depth_valid=bool(snapshot.get("depth_valid", False)),
            score=clamp01(snapshot.get("score", 0.0)),
            quality_score=clamp01(snapshot.get("quality_score", 0.0)),
            bbox=snapshot.get("bbox", {}) or {},
            geometry=snapshot.get("geometry", {}) or {},
            task2_result=result,
        )

    def has_position(self) -> bool:
        return self.position_3d is not None and self.position_confidence > 0.0

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "episode_id": self.episode_id,
            "step_id": int(self.step_id),
            "view_id": self.view_id,
            "decision": self.decision.value,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "position_3d": self.position_3d,
            "position_confidence": float(self.position_confidence),
            "depth_valid": bool(self.depth_valid),
            "score": float(self.score),
            "quality_score": float(self.quality_score),
            "bbox": self.bbox,
            "geometry": self.geometry,
        }


@dataclass
class EvidenceTrack:
    evidence: TargetEvidence
    observations: List[TargetObservation] = field(default_factory=list)
    position_history: List[Tuple[float, float, float]] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    decision_history: List[str] = field(default_factory=list)
    last_update_step: int = 0
    created_at: float = field(default_factory=now_ts)

    def to_log_dict(self) -> Dict[str, Any]:
        data = self.evidence.to_log_dict()
        data["observation_count"] = len(self.observations)
        data["position_history"] = list(self.position_history)
        data["confidence_history"] = list(self.confidence_history)
        data["decision_history"] = list(self.decision_history[-10:])
        data["last_update_step"] = int(self.last_update_step)
        data["created_at"] = float(self.created_at)
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
    ignored_results: Dict[str, str] = field(default_factory=dict)
    best_evidence: Optional[TargetEvidence] = None
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
            "ignored_results": dict(self.ignored_results),
            "best_evidence": None if self.best_evidence is None else self.best_evidence.to_log_dict(),
            "summary": dict(self.summary),
        }


class TargetEvidenceManager:
    """
    Fuse Task2 verification results into multi-frame target evidence.

    The manager keeps separate evidence tracks for spatially different
    candidates. A single GDINO candidate or one Task2 positive result is not
    enough to control navigation. The output is only evidence state.
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

            observation = TargetObservation.from_task2_result(result)

            if observation.decision == Task2Decision.UNKNOWN and not result.success:
                update.ignored_results[result.candidate_id] = "task2_error"
                continue

            track, created = self._match_or_create_track(observation)

            if track is None:
                update.ignored_results[result.candidate_id] = "no_track_created"
                continue

            self._update_track(track, observation, current_step=int(current_step))
            self.candidate_to_track[observation.candidate_id] = track.evidence.target_id

            update.updated_track_ids.append(track.evidence.target_id)
            if created:
                update.created_track_ids.append(track.evidence.target_id)

        self._refresh_all_status(current_step=int(current_step))
        self._prune_tracks()

        for track in self.tracks.values():
            status = track.evidence.status
            if status == CandidateStatus.VERIFIED:
                update.verified_track_ids.append(track.evidence.target_id)
            elif status == CandidateStatus.REJECTED:
                update.rejected_track_ids.append(track.evidence.target_id)
            elif status == CandidateStatus.TENTATIVE:
                update.tentative_track_ids.append(track.evidence.target_id)

        update.best_evidence = self.get_best_evidence(allow_tentative=False)
        if update.best_evidence is None:
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
    # Track update
    # ------------------------------------------------------------------

    def _match_or_create_track(
        self,
        observation: TargetObservation,
    ) -> Tuple[Optional[EvidenceTrack], bool]:
        existing_id = self.candidate_to_track.get(observation.candidate_id)
        if existing_id and existing_id in self.tracks:
            return self.tracks[existing_id], False

        match = self._match_spatial_track(observation)
        if match is not None:
            return match, False

        if observation.decision == Task2Decision.NO and not observation.has_position():
            return None, False

        track = self._create_track(observation)
        return track, True

    def _match_spatial_track(self, observation: TargetObservation) -> Optional[EvidenceTrack]:
        if not observation.has_position():
            return None

        best_track = None
        best_distance = float("inf")

        for track in self.tracks.values():
            if track.evidence.status == CandidateStatus.REJECTED:
                continue
            if track.evidence.position_3d is None:
                continue

            distance = _distance_3d(
                observation.position_3d,
                track.evidence.position_3d,
            )
            if distance < best_distance:
                best_distance = distance
                best_track = track

        if best_track is not None and best_distance <= self.config.track_match_distance:
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
        )

        self.tracks[track_id] = track
        return track

    def _update_track(
        self,
        track: EvidenceTrack,
        observation: TargetObservation,
        current_step: int,
    ) -> None:
        track.observations.append(observation)
        track.last_update_step = int(current_step)

        if observation.candidate_id not in track.evidence.source_candidate_ids:
            track.evidence.source_candidate_ids.append(observation.candidate_id)

        track.evidence.latest_candidate_id = observation.candidate_id
        track.evidence.seen_count += 1

        if track.evidence.first_seen_step is None:
            track.evidence.first_seen_step = observation.step_id
        track.evidence.last_seen_step = observation.step_id
        track.evidence.last_verified_step = observation.step_id

        if observation.reason:
            track.evidence.reasons.append(observation.reason)

        self._update_position(track, observation)
        self._update_decision(track, observation)
        self._update_evidence_score(track, observation)
        self._update_track_metadata(track, observation)

    def _update_position(
        self,
        track: EvidenceTrack,
        observation: TargetObservation,
    ) -> None:
        if not observation.has_position():
            return

        position = observation.position_3d
        confidence = clamp01(observation.position_confidence)

        track.position_history.append(position)
        track.confidence_history.append(confidence)

        if len(track.position_history) > self.config.position_history_size:
            track.position_history = track.position_history[-self.config.position_history_size :]
            track.confidence_history = track.confidence_history[-self.config.position_history_size :]

        fused = self._weighted_average_position(
            track.position_history,
            track.confidence_history,
        )

        track.evidence.position_3d = fused
        track.evidence.position_confidence = max(
            track.evidence.position_confidence,
            confidence,
        )
        track.evidence.position_stability = self._position_stability(track)

    def _weighted_average_position(
        self,
        positions: List[Tuple[float, float, float]],
        confidences: List[float],
    ) -> Optional[Tuple[float, float, float]]:
        if not positions:
            return None

        weights = [max(0.05, clamp01(c)) for c in confidences]
        total = sum(weights)
        if total <= 1e-6:
            total = float(len(positions))
            weights = [1.0 for _ in positions]

        x = sum(float(p[0]) * w for p, w in zip(positions, weights)) / total
        y = sum(float(p[1]) * w for p, w in zip(positions, weights)) / total
        z = sum(float(p[2]) * w for p, w in zip(positions, weights)) / total

        return (float(x), float(y), float(z))

    def _position_stability(self, track: EvidenceTrack) -> float:
        if len(track.position_history) <= 1:
            # One valid 3D observation is useful but not fully stable.
            conf = max(track.confidence_history) if track.confidence_history else 0.0
            return clamp01(0.45 * conf)

        center = self._weighted_average_position(
            track.position_history,
            track.confidence_history,
        )
        if center is None:
            return 0.0

        distances = [
            _distance_3d(position, center)
            for position in track.position_history
        ]
        mean_dist = sum(distances) / float(max(1, len(distances)))

        # Distances below ~2m are very stable; distances near match threshold are weak.
        stability = 1.0 - min(1.0, mean_dist / max(1e-6, self.config.track_match_distance))
        confidence = sum(track.confidence_history) / float(max(1, len(track.confidence_history)))

        return clamp01(0.65 * stability + 0.35 * confidence)

    def _update_decision(
        self,
        track: EvidenceTrack,
        observation: TargetObservation,
    ) -> None:
        decision = observation.decision
        conf = clamp01(observation.confidence)
        track.decision_history.append(decision.value)

        if decision == Task2Decision.YES:
            track.evidence.positive_count += 1
            if track.evidence.status not in (CandidateStatus.REJECTED, CandidateStatus.LOST):
                track.evidence.status = CandidateStatus.TENTATIVE

        elif decision == Task2Decision.MAYBE:
            track.evidence.maybe_count += 1
            if track.evidence.status == CandidateStatus.NEW:
                track.evidence.status = CandidateStatus.TENTATIVE

        elif decision == Task2Decision.NO:
            track.evidence.negative_count += 1
            if conf >= self.config.strong_negative_confidence and track.evidence.positive_count == 0:
                # A single strong negative is not always enough to delete a spatial track,
                # but it should keep it from becoming verified.
                track.evidence.status = CandidateStatus.TENTATIVE

    def _update_evidence_score(
        self,
        track: EvidenceTrack,
        observation: TargetObservation,
    ) -> None:
        old_score = clamp01(track.evidence.verify_score) * self.config.score_decay
        conf = clamp01(observation.confidence)

        if observation.decision == Task2Decision.YES:
            delta = self.config.positive_score_weight * conf
        elif observation.decision == Task2Decision.MAYBE:
            delta = self.config.maybe_score_weight * conf
        elif observation.decision == Task2Decision.NO:
            delta = -self.config.negative_score_weight * conf
        else:
            delta = 0.0

        # Keep score in [0, 1].
        # Positive evidence should accumulate, and a strong positive result
        # should be able to lift the track above the verification threshold.
        if delta >= 0.0:
            accumulated = old_score + 0.35 * delta
            new_score = max(old_score, delta, accumulated)
        else:
            new_score = old_score + 0.45 * delta

        track.evidence.verify_score = clamp01(new_score)

    def _update_track_metadata(
        self,
        track: EvidenceTrack,
        observation: TargetObservation,
    ) -> None:
        score = self._track_selection_score(track)

        track.evidence.metadata.update(
            {
                "evidence_score": score,
                "last_observation": observation.to_log_dict(),
                "position_history_count": len(track.position_history),
                "decision_history": list(track.decision_history[-10:]),
                "track_match_distance": self.config.track_match_distance,
            }
        )

    # ------------------------------------------------------------------
    # Status refresh
    # ------------------------------------------------------------------

    def _refresh_all_status(self, current_step: int) -> None:
        for track in self.tracks.values():
            self._refresh_track_status(track, current_step=current_step)

    def _refresh_track_status(self, track: EvidenceTrack, current_step: int) -> None:
        evidence = track.evidence

        if evidence.status == CandidateStatus.REJECTED:
            return

        if evidence.last_seen_step is not None:
            age = int(current_step) - int(evidence.last_seen_step)
            if age > self.config.stale_lost_steps and evidence.status == CandidateStatus.VERIFIED:
                evidence.set_lost("verified evidence became stale")
                return

        if self._should_reject(track):
            evidence.set_rejected("negative evidence dominates")
            return

        if self._should_verify(track):
            evidence.set_verified("task2 and 3d position evidence stable")
            return

        if evidence.status == CandidateStatus.NEW and evidence.seen_count > 0:
            evidence.status = CandidateStatus.TENTATIVE

    def _should_reject(self, track: EvidenceTrack) -> bool:
        evidence = track.evidence

        if evidence.positive_count > 0:
            # Do not reject a track with positive support unless negative evidence dominates.
            return evidence.negative_count > evidence.positive_count + evidence.maybe_count + 1

        if evidence.negative_count >= self.config.rejected_negative_count:
            return True

        return False

    def _should_verify(self, track: EvidenceTrack) -> bool:
        evidence = track.evidence

        if evidence.position_3d is None:
            return False

        if evidence.position_confidence < self.config.verified_position_confidence:
            return False

        if evidence.position_stability < self.config.verified_position_stability:
            return False

        if evidence.verify_score < self.config.verified_score_threshold:
            return False

        if evidence.negative_count > evidence.positive_count + evidence.maybe_count:
            return False

        if evidence.positive_count >= self.config.verified_positive_count:
            return True

        if (
            evidence.positive_count >= 1
            and evidence.maybe_count >= 1
            and evidence.verify_score >= self.config.verified_score_threshold
        ):
            return True

        if (
            evidence.maybe_count >= self.config.verified_maybe_count
            and evidence.verify_score >= self.config.verified_score_threshold
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Selection / pruning
    # ------------------------------------------------------------------

    def _track_selection_score(self, track: EvidenceTrack) -> float:
        evidence = track.evidence

        verify = clamp01(evidence.verify_score)
        stability = clamp01(evidence.position_stability)
        position_conf = clamp01(evidence.position_confidence)

        recent_bonus = 0.0
        if evidence.last_seen_step is not None:
            recent_bonus = 1.0

        negative_penalty = min(1.0, 0.20 * float(evidence.negative_count))
        status_bonus = 0.0
        if evidence.status == CandidateStatus.VERIFIED:
            status_bonus = 0.20
        elif evidence.status == CandidateStatus.TENTATIVE:
            status_bonus = 0.05

        score = (
            0.45 * verify
            + 0.25 * stability
            + 0.20 * position_conf
            + 0.10 * recent_bonus
            + status_bonus
            - negative_penalty
        )
        return clamp01(score)

    def get_best_evidence(self, allow_tentative: bool = False) -> Optional[TargetEvidence]:
        best = None
        best_score = -1.0

        for track in self.tracks.values():
            status = track.evidence.status

            if status == CandidateStatus.REJECTED or status == CandidateStatus.LOST:
                continue

            if status != CandidateStatus.VERIFIED and not allow_tentative:
                continue

            if status == CandidateStatus.TENTATIVE and not allow_tentative:
                continue

            score = self._track_selection_score(track)
            track.evidence.metadata["evidence_score"] = score

            if score > best_score:
                best_score = score
                best = track.evidence

        return best

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
            key=lambda track: self._track_selection_score(track),
            reverse=True,
        )
        keep_ids = {track.evidence.target_id for track in ranked[: self.config.max_tracks]}

        for track_id in list(self.tracks.keys()):
            if track_id not in keep_ids:
                self.tracks.pop(track_id, None)

        self.candidate_to_track = {
            cand_id: track_id
            for cand_id, track_id in self.candidate_to_track.items()
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

        best_verified = self.get_best_evidence(allow_tentative=False)
        best_any = best_verified or self.get_best_evidence(allow_tentative=True)

        for track in self.tracks.values():
            status = track.evidence.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "track_count": len(self.tracks),
            "status_counts": status_counts,
            "verified_count": status_counts.get("verified", 0),
            "tentative_count": status_counts.get("tentative", 0),
            "rejected_count": status_counts.get("rejected", 0),
            "best_verified_id": None if best_verified is None else best_verified.target_id,
            "best_evidence_id": None if best_any is None else best_any.target_id,
            "best_evidence_status": None if best_any is None else best_any.status.value,
            "best_evidence_score": None if best_any is None else best_any.metadata.get("evidence_score"),
            "best_evidence_position": None if best_any is None else best_any.position_3d,
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
