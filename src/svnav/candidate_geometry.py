from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .geometry import (
    bbox_basic_stats,
    camera_point_from_depth,
    camera_point_to_world,
    clamp01,
    pixel_to_camera_ray,
    robust_depth_in_bbox,
)
from .types import GDINOCandidate, ViewID


@dataclass
class CandidateGeometryConfig:
    """
    Candidate-level rough 3D geometry estimation.

    This module estimates where a candidate bbox would be in 3D if the bbox is
    a real object. It does not verify the target, does not update target
    evidence, does not switch navigation mode, and does not decide stop.
    """

    horizontal_fov_deg: float = 90.0
    vertical_fov_deg: float = 90.0

    center_patch_ratio: float = 0.15
    min_depth: float = 0.05
    max_depth: float = 200.0
    min_depth_valid_ratio: float = 0.10

    down_view_mode: str = "down"

    front_view_reliability: float = 1.0
    side_view_reliability: float = 0.9
    down_view_reliability: float = 0.65

    min_position_confidence: float = 0.20

    @staticmethod
    def from_env() -> "CandidateGeometryConfig":
        return CandidateGeometryConfig(
            horizontal_fov_deg=float(os.getenv("SVNAV_CAMERA_HFOV_DEG", "90.0")),
            vertical_fov_deg=float(os.getenv("SVNAV_CAMERA_VFOV_DEG", "90.0")),
            center_patch_ratio=float(os.getenv("SVNAV_GEOMETRY_CENTER_PATCH_RATIO", "0.15")),
            min_depth=float(os.getenv("SVNAV_GEOMETRY_MIN_DEPTH", "0.05")),
            max_depth=float(os.getenv("SVNAV_GEOMETRY_MAX_DEPTH", "200.0")),
            min_depth_valid_ratio=float(os.getenv("SVNAV_GEOMETRY_MIN_DEPTH_VALID_RATIO", "0.10")),
            down_view_mode=os.getenv("SVNAV_DOWN_VIEW_MODE", "down"),
        )


@dataclass
class CandidateGeometrySummary:
    input_count: int = 0
    estimated_count: int = 0
    valid_position_count: int = 0
    invalid_position_count: int = 0
    invalid_by_reason: Dict[str, int] = field(default_factory=dict)

    def add_invalid(self, reason: str) -> None:
        reason = str(reason or "unknown")
        self.invalid_position_count += 1
        self.invalid_by_reason[reason] = self.invalid_by_reason.get(reason, 0) + 1

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "input_count": int(self.input_count),
            "estimated_count": int(self.estimated_count),
            "valid_position_count": int(self.valid_position_count),
            "invalid_position_count": int(self.invalid_position_count),
            "invalid_by_reason": dict(self.invalid_by_reason),
        }


@dataclass
class CandidateGeometryOutput:
    candidates: List[GDINOCandidate] = field(default_factory=list)
    summary: CandidateGeometrySummary = field(default_factory=CandidateGeometrySummary)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary.to_log_dict(),
            "candidates": [candidate.to_log_dict() for candidate in self.candidates],
        }


class CandidateGeometryEstimator:
    def __init__(self, config: Optional[CandidateGeometryConfig] = None) -> None:
        self.config = config or CandidateGeometryConfig.from_env()

    def estimate_candidates(
        self,
        candidates: List[GDINOCandidate],
    ) -> CandidateGeometryOutput:
        output = CandidateGeometryOutput()
        output.summary.input_count = len(candidates or [])

        for candidate in candidates or []:
            estimated = self.estimate_candidate(candidate)
            output.candidates.append(estimated)
            output.summary.estimated_count += 1

            if estimated.position_3d is not None and estimated.depth_valid:
                output.summary.valid_position_count += 1
            else:
                reason = (
                    estimated.metadata.get("geometry", {}).get("geometry_warning")
                    if estimated.metadata
                    else None
                )
                output.summary.add_invalid(reason or "invalid_geometry")

        return output

    def estimate_candidate(self, candidate: GDINOCandidate) -> GDINOCandidate:
        geometry = self._estimate_geometry(candidate)

        candidate.position_3d = geometry.get("world_position")
        candidate.position_confidence = float(geometry.get("position_confidence", 0.0))
        candidate.depth_valid = bool(geometry.get("depth_valid", False))

        metadata = dict(candidate.metadata or {})
        metadata["geometry"] = geometry
        candidate.metadata = metadata

        return candidate

    def _estimate_geometry(self, candidate: GDINOCandidate) -> Dict[str, Any]:
        frame = candidate.frame
        bbox = candidate.bbox.clipped()
        bbox_stats = bbox_basic_stats(bbox)

        base = {
            "candidate_id": candidate.candidate_id,
            "frame_id": candidate.frame_id,
            "episode_id": candidate.episode_id,
            "step_id": int(candidate.step_id),
            "view_id": getattr(candidate.view_id, "value", str(candidate.view_id)),
            "bbox_stats": bbox_stats,
            "world_position": None,
            "position_confidence": 0.0,
            "depth_valid": False,
            "geometry_warning": None,
        }

        if frame is None:
            base["geometry_warning"] = "missing_frame"
            return base

        depth_stats = robust_depth_in_bbox(
            depth=frame.depth,
            bbox=bbox,
            center_patch_ratio=self.config.center_patch_ratio,
            min_depth=self.config.min_depth,
            max_depth=self.config.max_depth,
        )

        base["depth"] = depth_stats

        used_depth = depth_stats.get("used_depth")
        if used_depth is None:
            base["geometry_warning"] = "no_valid_depth"
            return base

        depth_valid_ratio = float(depth_stats.get("depth_valid_ratio", 0.0))
        if depth_valid_ratio < self.config.min_depth_valid_ratio:
            base["geometry_warning"] = "low_depth_valid_ratio"
            return base

        cx, cy = bbox.center

        ray = pixel_to_camera_ray(
            u=cx,
            v=cy,
            image_width=bbox.image_width,
            image_height=bbox.image_height,
            horizontal_fov_deg=self.config.horizontal_fov_deg,
            vertical_fov_deg=self.config.vertical_fov_deg,
        )
        camera_point = camera_point_from_depth(
            ray=ray,
            depth=float(used_depth),
        )
        world = camera_point_to_world(
            camera_point=camera_point,
            pose=frame.pose,
            view_id=candidate.view_id,
            down_view_mode=self.config.down_view_mode,
        )

        confidence = self._position_confidence(
            candidate=candidate,
            depth_stats=depth_stats,
            view_id=candidate.view_id,
        )

        world_position = world.get("world_position")

        base.update(
            {
                "depth_valid": True,
                "pixel_center": [float(cx), float(cy)],
                "camera_ray": ray,
                "camera_point": camera_point,
                "world_position": world_position,
                "estimated_distance": float(used_depth),
                "estimate_method": depth_stats.get("used_depth_source"),
                "projection": world,
                "position_confidence": confidence,
                "geometry_warning": None if confidence >= self.config.min_position_confidence else "low_position_confidence",
            }
        )

        if confidence < self.config.min_position_confidence:
            # Keep the raw geometry in metadata, but do not expose it as a
            # usable candidate position yet.
            base["world_position_low_confidence"] = world_position
            base["world_position"] = None
            base["depth_valid"] = False

        return base

    def _position_confidence(
        self,
        candidate: GDINOCandidate,
        depth_stats: Dict[str, Any],
        view_id: ViewID,
    ) -> float:
        depth_valid_ratio = clamp01(depth_stats.get("depth_valid_ratio", 0.0))
        depth_consistency = clamp01(depth_stats.get("depth_consistency", 0.0))
        quality_score = clamp01(getattr(candidate, "quality_score", 0.0))
        view_reliability = clamp01(self._view_reliability(view_id))

        confidence = (
            0.40 * depth_valid_ratio
            + 0.25 * depth_consistency
            + 0.20 * quality_score
            + 0.15 * view_reliability
        )
        return clamp01(confidence)

    def _view_reliability(self, view_id: ViewID) -> float:
        view_text = getattr(view_id, "value", str(view_id)).lower()
        if view_text == "front":
            return self.config.front_view_reliability
        if view_text in ("left", "right"):
            return self.config.side_view_reliability
        if view_text == "down":
            return self.config.down_view_reliability
        return 0.7
