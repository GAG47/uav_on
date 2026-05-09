from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .types import TargetEvidence, clamp01


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _enum_value(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


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


def _get_pose_xyz(pose: Any) -> Optional[Tuple[float, float, float]]:
    if pose is None:
        return None

    try:
        return (
            float(getattr(pose, "x")),
            float(getattr(pose, "y")),
            float(getattr(pose, "z")),
        )
    except Exception:
        pass

    if isinstance(pose, dict):
        if "x" in pose and "y" in pose and "z" in pose:
            return (
                _safe_float(pose.get("x")),
                _safe_float(pose.get("y")),
                _safe_float(pose.get("z")),
            )

    return None


def _distance_2d(
    a: Optional[Tuple[float, float, float]],
    b: Optional[Tuple[float, float, float]],
) -> Optional[float]:
    if a is None or b is None:
        return None

    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _distance_3d(
    a: Optional[Tuple[float, float, float]],
    b: Optional[Tuple[float, float, float]],
) -> Optional[float]:
    if a is None or b is None:
        return None

    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


@dataclass
class StopGateConfig:
    """
    Final stop gate for SVNav.

    This module is deliberately stricter than Approach:
        - Visual anchors can be approached, but cannot stop.
        - Spatial anchors can stop only when Task2 evidence is recent and
          strong enough, the 3D position is reliable enough, and the UAV has
          actually approached the target.
    """

    success_distance: float = 5.0

    min_position_confidence: float = 0.30
    min_position_stability: float = 0.20

    max_support_age_steps: int = 12
    min_stop_task2_score: float = 0.45

    min_approach_progress: float = 3.0
    near_count_required: int = 1

    allow_visual_only_stop: bool = False
    use_horizontal_distance: bool = True

    block_if_demoted: bool = True
    block_if_negative_dominates: bool = True
    negative_reject_margin: int = 1

    def __post_init__(self) -> None:
        self.success_distance = float(self.success_distance)
        self.min_position_confidence = float(self.min_position_confidence)
        self.min_position_stability = float(self.min_position_stability)
        self.max_support_age_steps = max(1, int(self.max_support_age_steps))
        self.min_stop_task2_score = float(self.min_stop_task2_score)
        self.min_approach_progress = float(self.min_approach_progress)
        self.near_count_required = max(1, int(self.near_count_required))
        self.negative_reject_margin = max(1, int(self.negative_reject_margin))


@dataclass
class StopGateResult:
    episode_id: str
    step_id: int
    allow_stop: bool
    reason: str

    target_id: Optional[str] = None
    candidate_id: Optional[str] = None

    mode: str = ""
    action: str = ""

    anchor_type: str = ""
    task2_admission: Optional[str] = None

    distance_to_target: Optional[float] = None
    horizontal_distance_to_target: Optional[float] = None

    position_confidence: float = 0.0
    position_stability: float = 0.0

    task2_score: float = 0.0
    support_age_steps: Optional[int] = None

    approach_progress: float = 0.0
    near_count: int = 0

    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_id": int(self.step_id),
            "allow_stop": bool(self.allow_stop),
            "reason": self.reason,
            "target_id": self.target_id,
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "action": self.action,
            "anchor_type": self.anchor_type,
            "task2_admission": self.task2_admission,
            "distance_to_target": self.distance_to_target,
            "horizontal_distance_to_target": self.horizontal_distance_to_target,
            "position_confidence": float(self.position_confidence),
            "position_stability": float(self.position_stability),
            "task2_score": float(self.task2_score),
            "support_age_steps": self.support_age_steps,
            "approach_progress": float(self.approach_progress),
            "near_count": int(self.near_count),
            "debug_info": dict(self.debug_info),
        }


class StopGate:
    """
    Unique stop decision interface for SVNav.

    It never calls Task1, GDINO, or Task2. It only reads current observation,
    current navigation decision, target evidence, and approach session.

    If allow_stop=True, ON_Air_SV.py may replace the current action with stop.
    """

    def __init__(self, config: Optional[StopGateConfig] = None) -> None:
        self.config = config or StopGateConfig()

    def evaluate(
        self,
        episode_id: str,
        step_id: int,
        observation: Any,
        nav_decision: Any,
        target_evidence: Optional[TargetEvidence],
        approach_session: Optional[Dict[str, Any]] = None,
    ) -> StopGateResult:
        session = approach_session if isinstance(approach_session, dict) else {}

        mode = _enum_value(getattr(nav_decision, "mode", ""))
        action = str(getattr(nav_decision, "action", "") or "")
        decision_target_id = getattr(nav_decision, "target_id", None)
        decision_candidate_id = getattr(nav_decision, "candidate_id", None)

        if target_evidence is None:
            return self._deny(
                episode_id=episode_id,
                step_id=step_id,
                reason="missing_target_evidence",
                mode=mode,
                action=action,
            )

        target_id = getattr(target_evidence, "target_id", None)
        candidate_id = getattr(target_evidence, "latest_candidate_id", None) or decision_candidate_id

        if decision_target_id is not None and target_id is not None:
            if str(decision_target_id) != str(target_id):
                return self._deny(
                    episode_id=episode_id,
                    step_id=step_id,
                    reason="decision_target_mismatch",
                    target_id=target_id,
                    candidate_id=candidate_id,
                    mode=mode,
                    action=action,
                )

        metadata = getattr(target_evidence, "metadata", {}) or {}

        if mode not in ("approach", "final_check"):
            return self._deny(
                episode_id=episode_id,
                step_id=step_id,
                reason="not_in_approach_mode",
                target_id=target_id,
                candidate_id=candidate_id,
                mode=mode,
                action=action,
            )

        status = _enum_value(getattr(target_evidence, "status", ""))
        if status in ("rejected", "lost"):
            return self._deny(
                episode_id=episode_id,
                step_id=step_id,
                reason="target_status_not_valid",
                target_id=target_id,
                candidate_id=candidate_id,
                mode=mode,
                action=action,
                debug_info={"status": status},
            )

        task2_admission = metadata.get("task2_admission")
        if task2_admission != "passed":
            return self._deny(
                episode_id=episode_id,
                step_id=step_id,
                reason="target_not_task2_passed",
                target_id=target_id,
                candidate_id=candidate_id,
                mode=mode,
                action=action,
                task2_admission=task2_admission,
            )

        if self.config.block_if_demoted:
            last_feedback_decision = metadata.get("last_approach_feedback_decision")
            if last_feedback_decision in ("demote", "reject"):
                return self._deny(
                    episode_id=episode_id,
                    step_id=step_id,
                    reason="target_recently_demoted",
                    target_id=target_id,
                    candidate_id=candidate_id,
                    mode=mode,
                    action=action,
                    task2_admission=task2_admission,
                    debug_info={"last_approach_feedback_decision": last_feedback_decision},
                )

            blocked_until = metadata.get("approach_blocked_until_step")
            if blocked_until is not None and _safe_int(blocked_until, -1) > int(step_id):
                return self._deny(
                    episode_id=episode_id,
                    step_id=step_id,
                    reason="target_in_approach_cooldown",
                    target_id=target_id,
                    candidate_id=candidate_id,
                    mode=mode,
                    action=action,
                    task2_admission=task2_admission,
                    debug_info={"approach_blocked_until_step": blocked_until},
                )

        anchor_type = self._get_anchor_type(target_evidence)
        if anchor_type != "spatial" and not self.config.allow_visual_only_stop:
            return self._deny(
                episode_id=episode_id,
                step_id=step_id,
                reason="visual_anchor_cannot_stop",
                target_id=target_id,
                candidate_id=candidate_id,
                mode=mode,
                action=action,
                anchor_type=anchor_type,
                task2_admission=task2_admission,
            )

        target_position = self._get_target_position(target_evidence)
        if target_position is None:
            return self._deny(
                episode_id=episode_id,
                step_id=step_id,
                reason="missing_target_position",
                target_id=target_id,
                candidate_id=candidate_id,
                mode=mode,
                action=action,
                anchor_type=anchor_type,
                task2_admission=task2_admission,
            )

        pose = _get_pose_xyz(getattr(observation, "pose", None))
        if pose is None:
            return self._deny(
                episode_id=episode_id,
                step_id=step_id,
                reason="missing_current_pose",
                target_id=target_id,
                candidate_id=candidate_id,
                mode=mode,
                action=action,
                anchor_type=anchor_type,
                task2_admission=task2_admission,
            )

        horizontal_distance = _distance_2d(pose, target_position)
        distance_3d = _distance_3d(pose, target_position)

        gate_distance = horizontal_distance if self.config.use_horizontal_distance else distance_3d
        if gate_distance is None:
            return self._deny(
                episode_id=episode_id,
                step_id=step_id,
                reason="cannot_compute_distance",
                target_id=target_id,
                candidate_id=candidate_id,
                mode=mode,
                action=action,
                anchor_type=anchor_type,
                task2_admission=task2_admission,
            )

        position_confidence = clamp01(getattr(target_evidence, "position_confidence", 0.0))
        position_stability = clamp01(getattr(target_evidence, "position_stability", 0.0))

        if position_confidence < self.config.min_position_confidence:
            return self._deny_with_metrics(
                episode_id=episode_id,
                step_id=step_id,
                reason="low_position_confidence",
                target_evidence=target_evidence,
                mode=mode,
                action=action,
                anchor_type=anchor_type,
                task2_admission=task2_admission,
                distance_3d=distance_3d,
                horizontal_distance=horizontal_distance,
                position_confidence=position_confidence,
                position_stability=position_stability,
                session=session,
            )

        if position_stability < self.config.min_position_stability:
            return self._deny_with_metrics(
                episode_id=episode_id,
                step_id=step_id,
                reason="low_position_stability",
                target_evidence=target_evidence,
                mode=mode,
                action=action,
                anchor_type=anchor_type,
                task2_admission=task2_admission,
                distance_3d=distance_3d,
                horizontal_distance=horizontal_distance,
                position_confidence=position_confidence,
                position_stability=position_stability,
                session=session,
            )

        task2_score = clamp01(metadata.get("task2_score", metadata.get("visual_score", 0.0)))
        if task2_score < self.config.min_stop_task2_score:
            return self._deny_with_metrics(
                episode_id=episode_id,
                step_id=step_id,
                reason="weak_task2_support_for_stop",
                target_evidence=target_evidence,
                mode=mode,
                action=action,
                anchor_type=anchor_type,
                task2_admission=task2_admission,
                distance_3d=distance_3d,
                horizontal_distance=horizontal_distance,
                position_confidence=position_confidence,
                position_stability=position_stability,
                session=session,
                extra={"task2_score": task2_score},
            )

        support_age = self._get_support_age(target_evidence, step_id)
        if support_age is None:
            return self._deny_with_metrics(
                episode_id=episode_id,
                step_id=step_id,
                reason="missing_recent_task2_support",
                target_evidence=target_evidence,
                mode=mode,
                action=action,
                anchor_type=anchor_type,
                task2_admission=task2_admission,
                distance_3d=distance_3d,
                horizontal_distance=horizontal_distance,
                position_confidence=position_confidence,
                position_stability=position_stability,
                session=session,
                extra={"task2_score": task2_score},
            )

        if support_age > self.config.max_support_age_steps:
            return self._deny_with_metrics(
                episode_id=episode_id,
                step_id=step_id,
                reason="stale_task2_support",
                target_evidence=target_evidence,
                mode=mode,
                action=action,
                anchor_type=anchor_type,
                task2_admission=task2_admission,
                distance_3d=distance_3d,
                horizontal_distance=horizontal_distance,
                position_confidence=position_confidence,
                position_stability=position_stability,
                session=session,
                extra={
                    "task2_score": task2_score,
                    "support_age_steps": support_age,
                },
            )

        if self.config.block_if_negative_dominates:
            positive_count = _safe_int(getattr(target_evidence, "positive_count", 0), 0)
            maybe_count = _safe_int(getattr(target_evidence, "maybe_count", 0), 0)
            negative_count = _safe_int(getattr(target_evidence, "negative_count", 0), 0)
            if negative_count >= positive_count + maybe_count + self.config.negative_reject_margin:
                return self._deny_with_metrics(
                    episode_id=episode_id,
                    step_id=step_id,
                    reason="negative_task2_evidence_dominates",
                    target_evidence=target_evidence,
                    mode=mode,
                    action=action,
                    anchor_type=anchor_type,
                    task2_admission=task2_admission,
                    distance_3d=distance_3d,
                    horizontal_distance=horizontal_distance,
                    position_confidence=position_confidence,
                    position_stability=position_stability,
                    session=session,
                    extra={
                        "task2_score": task2_score,
                        "support_age_steps": support_age,
                        "positive_count": positive_count,
                        "maybe_count": maybe_count,
                        "negative_count": negative_count,
                    },
                )

        if gate_distance > self.config.success_distance:
            near_count = self._update_near_count(session, target_id, is_near=False)
            return self._deny_with_metrics(
                episode_id=episode_id,
                step_id=step_id,
                reason="too_far_from_target",
                target_evidence=target_evidence,
                mode=mode,
                action=action,
                anchor_type=anchor_type,
                task2_admission=task2_admission,
                distance_3d=distance_3d,
                horizontal_distance=horizontal_distance,
                position_confidence=position_confidence,
                position_stability=position_stability,
                session=session,
                extra={
                    "task2_score": task2_score,
                    "support_age_steps": support_age,
                    "near_count": near_count,
                },
            )

        approach_progress = self._get_approach_progress(session)
        if approach_progress < self.config.min_approach_progress:
            near_count = self._update_near_count(session, target_id, is_near=True)
            return self._deny_with_metrics(
                episode_id=episode_id,
                step_id=step_id,
                reason="no_approach_progress",
                target_evidence=target_evidence,
                mode=mode,
                action=action,
                anchor_type=anchor_type,
                task2_admission=task2_admission,
                distance_3d=distance_3d,
                horizontal_distance=horizontal_distance,
                position_confidence=position_confidence,
                position_stability=position_stability,
                session=session,
                extra={
                    "task2_score": task2_score,
                    "support_age_steps": support_age,
                    "approach_progress": approach_progress,
                    "near_count": near_count,
                },
            )

        near_count = self._update_near_count(session, target_id, is_near=True)
        if near_count < self.config.near_count_required:
            return self._deny_with_metrics(
                episode_id=episode_id,
                step_id=step_id,
                reason="near_count_not_enough",
                target_evidence=target_evidence,
                mode=mode,
                action=action,
                anchor_type=anchor_type,
                task2_admission=task2_admission,
                distance_3d=distance_3d,
                horizontal_distance=horizontal_distance,
                position_confidence=position_confidence,
                position_stability=position_stability,
                session=session,
                extra={
                    "task2_score": task2_score,
                    "support_age_steps": support_age,
                    "approach_progress": approach_progress,
                    "near_count": near_count,
                },
            )

        return StopGateResult(
            episode_id=episode_id,
            step_id=int(step_id),
            allow_stop=True,
            reason="near_verified_spatial_target",
            target_id=target_id,
            candidate_id=candidate_id,
            mode=mode,
            action=action,
            anchor_type=anchor_type,
            task2_admission=task2_admission,
            distance_to_target=distance_3d,
            horizontal_distance_to_target=horizontal_distance,
            position_confidence=position_confidence,
            position_stability=position_stability,
            task2_score=task2_score,
            support_age_steps=support_age,
            approach_progress=approach_progress,
            near_count=near_count,
            debug_info={
                "target_position": target_position,
                "current_pose": pose,
                "success_distance": self.config.success_distance,
                "use_horizontal_distance": self.config.use_horizontal_distance,
                "approach_session": dict(session),
            },
        )

    def _get_anchor_type(self, target_evidence: TargetEvidence) -> str:
        metadata = getattr(target_evidence, "metadata", {}) or {}
        anchor_type = metadata.get("anchor_type") or metadata.get("evidence_kind")

        anchor = metadata.get("approach_anchor")
        if isinstance(anchor, dict):
            anchor_type = anchor.get("type") or anchor_type

        return str(anchor_type or "visual").lower()

    def _get_target_position(
        self,
        target_evidence: TargetEvidence,
    ) -> Optional[Tuple[float, float, float]]:
        pos = _as_position(getattr(target_evidence, "position_3d", None))
        if pos is not None:
            return pos

        metadata = getattr(target_evidence, "metadata", {}) or {}
        spatial_anchor = metadata.get("spatial_anchor")
        if isinstance(spatial_anchor, dict):
            pos = _as_position(spatial_anchor.get("position_3d"))
            if pos is not None:
                return pos

        approach_anchor = metadata.get("approach_anchor")
        if isinstance(approach_anchor, dict):
            pos = _as_position(approach_anchor.get("position_3d"))
            if pos is not None:
                return pos

        return None

    def _get_support_age(
        self,
        target_evidence: TargetEvidence,
        step_id: int,
    ) -> Optional[int]:
        metadata = getattr(target_evidence, "metadata", {}) or {}

        last_positive_step = metadata.get("last_positive_step")
        if last_positive_step is None:
            last_positive_step = getattr(target_evidence, "last_verified_step", None)

        if last_positive_step is None:
            return None

        try:
            return max(0, int(step_id) - int(last_positive_step))
        except Exception:
            return None

    def _get_approach_progress(self, session: Dict[str, Any]) -> float:
        start = session.get("start_distance_to_anchor")
        best = session.get("best_distance_to_anchor")

        if start is None or best is None:
            return 0.0

        try:
            return max(0.0, float(start) - float(best))
        except Exception:
            return 0.0

    def _update_near_count(
        self,
        session: Dict[str, Any],
        target_id: Optional[str],
        is_near: bool,
    ) -> int:
        key_target = session.get("stop_near_target_id")
        if key_target != target_id:
            session["stop_near_target_id"] = target_id
            session["stop_near_count"] = 0

        if is_near:
            session["stop_near_count"] = int(session.get("stop_near_count", 0)) + 1
        else:
            session["stop_near_count"] = 0

        return int(session.get("stop_near_count", 0))

    def _deny(
        self,
        episode_id: str,
        step_id: int,
        reason: str,
        target_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        mode: str = "",
        action: str = "",
        anchor_type: str = "",
        task2_admission: Optional[str] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> StopGateResult:
        return StopGateResult(
            episode_id=episode_id,
            step_id=int(step_id),
            allow_stop=False,
            reason=reason,
            target_id=target_id,
            candidate_id=candidate_id,
            mode=mode,
            action=action,
            anchor_type=anchor_type,
            task2_admission=task2_admission,
            debug_info=debug_info or {},
        )

    def _deny_with_metrics(
        self,
        episode_id: str,
        step_id: int,
        reason: str,
        target_evidence: TargetEvidence,
        mode: str,
        action: str,
        anchor_type: str,
        task2_admission: Optional[str],
        distance_3d: Optional[float],
        horizontal_distance: Optional[float],
        position_confidence: float,
        position_stability: float,
        session: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> StopGateResult:
        metadata = getattr(target_evidence, "metadata", {}) or {}
        task2_score = clamp01(metadata.get("task2_score", metadata.get("visual_score", 0.0)))
        support_age = self._get_support_age(target_evidence, step_id)
        approach_progress = self._get_approach_progress(session)

        debug = {
            "success_distance": self.config.success_distance,
            "min_position_confidence": self.config.min_position_confidence,
            "min_position_stability": self.config.min_position_stability,
            "min_stop_task2_score": self.config.min_stop_task2_score,
            "max_support_age_steps": self.config.max_support_age_steps,
            "min_approach_progress": self.config.min_approach_progress,
            "near_count_required": self.config.near_count_required,
            "approach_session": dict(session),
        }
        if extra:
            debug.update(extra)

        return StopGateResult(
            episode_id=episode_id,
            step_id=int(step_id),
            allow_stop=False,
            reason=reason,
            target_id=getattr(target_evidence, "target_id", None),
            candidate_id=getattr(target_evidence, "latest_candidate_id", None),
            mode=mode,
            action=action,
            anchor_type=anchor_type,
            task2_admission=task2_admission,
            distance_to_target=distance_3d,
            horizontal_distance_to_target=horizontal_distance,
            position_confidence=position_confidence,
            position_stability=position_stability,
            task2_score=task2_score,
            support_age_steps=support_age,
            approach_progress=approach_progress,
            near_count=int(session.get("stop_near_count", 0)),
            debug_info=debug,
        )
