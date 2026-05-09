from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .geometry import (
    bbox_basic_stats,
    bbox_border_stats,
    clamp01,
    depth_stats_in_bbox,
)
from .types import GDINOCandidate


class CandidateRejectReason(str, Enum):
    LOW_SCORE = "low_score"
    INVALID_BBOX = "invalid_bbox"
    TOO_SMALL = "too_small"
    TOO_LARGE = "too_large"
    FULL_IMAGE_BOX = "full_image_box"
    BAD_ASPECT_RATIO = "bad_aspect_ratio"
    BORDER_LARGE = "border_large"
    BORDER_LOW_SCORE = "border_low_score"
    NO_VALID_DEPTH = "no_valid_depth"


@dataclass
class GDINOCandidateFilterConfig:
    """
    Candidate quality filter before Task2.

    This filter does not decide whether a candidate is the target. It only
    rejects detections that are clearly unsuitable for downstream verification:
    full-image boxes, tiny boxes, extreme aspect-ratio boxes, border-dominated
    boxes, low score boxes, and optionally boxes without usable depth.
    """

    min_score: float = 0.25

    min_area_ratio: float = 0.0005
    max_area_ratio: float = 0.65
    full_image_area_ratio: float = 0.90

    min_box_width_px: float = 8.0
    min_box_height_px: float = 8.0
    max_aspect_ratio: float = 8.0

    border_margin_px: float = 2.0
    border_large_area_ratio: float = 0.30
    border_low_score: float = 0.35

    require_depth: bool = False
    min_depth_valid_ratio: float = 0.10

    # Quality score weights.
    score_weight: float = 0.45
    area_weight: float = 0.20
    shape_weight: float = 0.15
    center_weight: float = 0.10
    depth_weight: float = 0.10

    def __post_init__(self) -> None:
        self.min_score = float(self.min_score)
        self.min_area_ratio = float(self.min_area_ratio)
        self.max_area_ratio = float(self.max_area_ratio)
        self.full_image_area_ratio = float(self.full_image_area_ratio)
        self.min_box_width_px = float(self.min_box_width_px)
        self.min_box_height_px = float(self.min_box_height_px)
        self.max_aspect_ratio = float(self.max_aspect_ratio)
        self.border_margin_px = float(self.border_margin_px)
        self.border_large_area_ratio = float(self.border_large_area_ratio)
        self.border_low_score = float(self.border_low_score)
        self.min_depth_valid_ratio = float(self.min_depth_valid_ratio)


@dataclass
class CandidateFilterSummary:
    raw_count: int = 0
    kept_count: int = 0
    rejected_count: int = 0
    rejected_by_reason: Dict[str, int] = field(default_factory=dict)

    def add_reject(self, reason: str) -> None:
        reason = str(reason or "unknown")
        self.rejected_count += 1
        self.rejected_by_reason[reason] = self.rejected_by_reason.get(reason, 0) + 1

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "raw_count": int(self.raw_count),
            "kept_count": int(self.kept_count),
            "rejected_count": int(self.rejected_count),
            "rejected_by_reason": dict(self.rejected_by_reason),
        }


@dataclass
class CandidateFilterOutput:
    kept: List[GDINOCandidate] = field(default_factory=list)
    rejected: List[GDINOCandidate] = field(default_factory=list)
    summary: CandidateFilterSummary = field(default_factory=CandidateFilterSummary)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "kept": [item.to_log_dict() for item in self.kept],
            "rejected": [item.to_log_dict() for item in self.rejected],
            "summary": self.summary.to_log_dict(),
        }


class GDINOCandidateFilter:
    def __init__(self, config: Optional[GDINOCandidateFilterConfig] = None) -> None:
        self.config = config or GDINOCandidateFilterConfig()

    def filter_candidates(
        self,
        candidates: List[GDINOCandidate],
    ) -> CandidateFilterOutput:
        output = CandidateFilterOutput()
        output.summary.raw_count = len(candidates or [])

        for candidate in candidates or []:
            filtered = self.evaluate_candidate(candidate)

            if filtered.rejected_reason:
                output.rejected.append(filtered)
                output.summary.add_reject(filtered.rejected_reason)
            else:
                output.kept.append(filtered)

        output.summary.kept_count = len(output.kept)
        output.summary.rejected_count = len(output.rejected)
        return output

    def evaluate_candidate(self, candidate: GDINOCandidate) -> GDINOCandidate:
        stats = bbox_basic_stats(candidate.bbox)
        border = bbox_border_stats(
            candidate.bbox,
            margin_px=self.config.border_margin_px,
        )
        depth_stats = self._candidate_depth_stats(candidate)

        reason = self._reject_reason(
            candidate=candidate,
            stats=stats,
            border=border,
            depth_stats=depth_stats,
        )

        quality_score = self._quality_score(
            candidate=candidate,
            stats=stats,
            border=border,
            depth_stats=depth_stats,
            rejected_reason=reason,
        )

        candidate.rejected_reason = None if reason is None else reason.value
        candidate.quality_score = quality_score
        candidate.depth_valid = bool(depth_stats.get("depth_valid", False))

        metadata = dict(candidate.metadata or {})
        metadata["candidate_filter"] = {
            "bbox_stats": stats,
            "border_stats": border,
            "depth_stats": depth_stats,
            "quality_score": quality_score,
            "rejected_reason": candidate.rejected_reason,
            "config": {
                "min_score": self.config.min_score,
                "min_area_ratio": self.config.min_area_ratio,
                "max_area_ratio": self.config.max_area_ratio,
                "full_image_area_ratio": self.config.full_image_area_ratio,
                "min_box_width_px": self.config.min_box_width_px,
                "min_box_height_px": self.config.min_box_height_px,
                "max_aspect_ratio": self.config.max_aspect_ratio,
                "border_margin_px": self.config.border_margin_px,
                "border_large_area_ratio": self.config.border_large_area_ratio,
                "border_low_score": self.config.border_low_score,
                "require_depth": self.config.require_depth,
                "min_depth_valid_ratio": self.config.min_depth_valid_ratio,
            },
        }
        candidate.metadata = metadata

        return candidate

    def _reject_reason(
        self,
        candidate: GDINOCandidate,
        stats: Dict[str, Any],
        border: Dict[str, Any],
        depth_stats: Dict[str, Any],
    ) -> Optional[CandidateRejectReason]:
        if float(candidate.score) < self.config.min_score:
            return CandidateRejectReason.LOW_SCORE

        if stats["image_width"] <= 0 or stats["image_height"] <= 0:
            return CandidateRejectReason.INVALID_BBOX

        if stats["width"] < self.config.min_box_width_px:
            return CandidateRejectReason.TOO_SMALL

        if stats["height"] < self.config.min_box_height_px:
            return CandidateRejectReason.TOO_SMALL

        if stats["area_ratio"] < self.config.min_area_ratio:
            return CandidateRejectReason.TOO_SMALL

        if stats["area_ratio"] >= self.config.full_image_area_ratio:
            return CandidateRejectReason.FULL_IMAGE_BOX

        if stats["area_ratio"] > self.config.max_area_ratio:
            return CandidateRejectReason.TOO_LARGE

        if stats["aspect_ratio"] > self.config.max_aspect_ratio:
            return CandidateRejectReason.BAD_ASPECT_RATIO

        if border["touches_border"] and stats["area_ratio"] > self.config.border_large_area_ratio:
            return CandidateRejectReason.BORDER_LARGE

        if border["touches_border"] and float(candidate.score) < self.config.border_low_score:
            return CandidateRejectReason.BORDER_LOW_SCORE

        if self.config.require_depth:
            if not depth_stats.get("depth_valid", False):
                return CandidateRejectReason.NO_VALID_DEPTH
            if float(depth_stats.get("depth_valid_ratio", 0.0)) < self.config.min_depth_valid_ratio:
                return CandidateRejectReason.NO_VALID_DEPTH

        return None

    def _candidate_depth_stats(self, candidate: GDINOCandidate) -> Dict[str, Any]:
        frame = candidate.frame
        if frame is None:
            return depth_stats_in_bbox(None, candidate.bbox)

        return depth_stats_in_bbox(frame.depth, candidate.bbox)

    def _quality_score(
        self,
        candidate: GDINOCandidate,
        stats: Dict[str, Any],
        border: Dict[str, Any],
        depth_stats: Dict[str, Any],
        rejected_reason: Optional[CandidateRejectReason],
    ) -> float:
        if rejected_reason is not None:
            return 0.0

        score_part = clamp01(candidate.score)

        # Prefer moderate-sized object boxes. Very tiny and very large boxes
        # receive lower score even if they pass hard filters.
        area = float(stats["area_ratio"])
        if area <= 0.0:
            area_part = 0.0
        elif area < 0.02:
            area_part = clamp01(area / 0.02)
        elif area > 0.45:
            area_part = clamp01((self.config.max_area_ratio - area) / max(1e-6, self.config.max_area_ratio - 0.45))
        else:
            area_part = 1.0

        aspect = float(stats["aspect_ratio"])
        shape_part = clamp01(1.0 - max(0.0, aspect - 1.0) / max(1.0, self.config.max_aspect_ratio - 1.0))

        center_x, center_y = stats["center"]
        image_width = max(1.0, float(stats["image_width"]))
        image_height = max(1.0, float(stats["image_height"]))
        dx = abs(float(center_x) - image_width / 2.0) / (image_width / 2.0)
        dy = abs(float(center_y) - image_height / 2.0) / (image_height / 2.0)
        center_part = clamp01(1.0 - 0.5 * (dx + dy))

        if border.get("touches_border", False):
            center_part *= 0.75

        if depth_stats.get("depth_available", False):
            depth_part = clamp01(depth_stats.get("depth_valid_ratio", 0.0))
        else:
            # Depth may be unavailable in the current smoke tests / partial
            # pipeline. Do not punish too heavily unless require_depth=True.
            depth_part = 0.5

        quality = (
            self.config.score_weight * score_part
            + self.config.area_weight * area_part
            + self.config.shape_weight * shape_part
            + self.config.center_weight * center_part
            + self.config.depth_weight * depth_part
        )

        return clamp01(quality)
