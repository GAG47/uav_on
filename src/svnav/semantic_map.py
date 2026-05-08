from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from svnav.types import (
    FrameRecord,
    PoseRecord,
    RegionScore,
    Task1Result,
    ViewID,
    clamp01,
)


class MapCellStatus(str, Enum):
    UNKNOWN = "unknown"
    VISITED = "visited"
    EXPLORED = "explored"
    HIGH_VALUE = "high_value"
    REJECTED = "rejected"


@dataclass
class SearchBounds:
    """
    Search boundary in UAV-ON world coordinates.

    This class is used to keep SemanticMap consistent with the UAV-ON
    episode-level search region. The map should not silently use another
    boundary that differs from the environment/wrapper.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        self.x_min = float(self.x_min)
        self.x_max = float(self.x_max)
        self.y_min = float(self.y_min)
        self.y_max = float(self.y_max)

        if self.x_max <= self.x_min:
            raise ValueError("SearchBounds requires x_max > x_min")
        if self.y_max <= self.y_min:
            raise ValueError("SearchBounds requires y_max > y_min")

    @classmethod
    def from_center_radius(
        cls,
        center_x: float,
        center_y: float,
        radius: float,
    ) -> "SearchBounds":
        radius = float(radius)
        if radius <= 0:
            raise ValueError("search radius must be positive")

        return cls(
            x_min=float(center_x) - radius,
            x_max=float(center_x) + radius,
            y_min=float(center_y) - radius,
            y_max=float(center_y) + radius,
        )

    @classmethod
    def from_uavon_start(
        cls,
        start_pose: Any,
        search_radius: float,
    ) -> "SearchBounds":
        """
        Build bounds from UAV-ON episode start pose.

        Supported start_pose formats:
            PoseRecord
            dict with x/y or position
            list/tuple like [x, y, z, ...]
        """
        x, y = cls._extract_xy(start_pose)
        return cls.from_center_radius(x, y, search_radius)

    @staticmethod
    def _extract_xy(pose_like: Any) -> Tuple[float, float]:
        if isinstance(pose_like, PoseRecord):
            return float(pose_like.x), float(pose_like.y)

        if isinstance(pose_like, dict):
            if "x" in pose_like and "y" in pose_like:
                return float(pose_like["x"]), float(pose_like["y"])
            if "position" in pose_like:
                return SearchBounds._extract_xy(pose_like["position"])
            if "start_position" in pose_like:
                return SearchBounds._extract_xy(pose_like["start_position"])

        if isinstance(pose_like, Sequence) and len(pose_like) >= 2:
            return float(pose_like[0]), float(pose_like[1])

        raise ValueError("Cannot extract x/y from start_pose: {}".format(type(pose_like)))

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    def contains_xy(self, x: float, y: float, eps: float = 1e-6) -> bool:
        return (
            self.x_min - eps <= float(x) <= self.x_max + eps
            and self.y_min - eps <= float(y) <= self.y_max + eps
        )

    def contains_pose(self, pose: PoseRecord, eps: float = 1e-6) -> bool:
        return self.contains_xy(pose.x, pose.y, eps=eps)

    def clamp_xy(self, x: float, y: float) -> Tuple[float, float]:
        return (
            min(max(float(x), self.x_min), self.x_max),
            min(max(float(y), self.y_min), self.y_max),
        )

    def nearest_in_bounds_position(
        self,
        pose: PoseRecord,
        keep_z: bool = True,
    ) -> Tuple[float, float, float]:
        x, y = self.clamp_xy(pose.x, pose.y)
        z = pose.z if keep_z else 0.0
        return float(x), float(y), float(z)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "width": self.width,
            "height": self.height,
            "center": list(self.center),
        }


@dataclass
class SemanticMapConfig:
    """
    Configuration for the 2.5D semantic map.

    search_bounds should normally be initialized from the UAV-ON episode
    start pose and search radius. search_radius is kept only as a convenient
    way to construct bounds when explicit x/y bounds are not provided.
    """

    search_bounds: Optional[SearchBounds] = None
    search_radius: Optional[float] = None
    cell_size: float = 5.0

    max_projection_distance: float = 30.0
    min_projection_distance: float = 3.0
    horizontal_fov_deg: float = 90.0
    down_projection_radius: float = 8.0

    depth_percentile: float = 70.0
    depth_safety_margin: float = 2.0
    min_valid_depth: float = 0.3

    min_update_confidence: float = 0.05
    high_value_threshold: float = 0.65
    high_conf_threshold: float = 0.35

    semantic_conf_decay_per_step: float = 0.995
    explored_value_decay: float = 0.95
    explored_visit_threshold: int = 4
    rejected_value: float = 0.05
    rejected_confidence: float = 0.8

    select_semantic_weight: float = 1.0
    select_exploration_weight: float = 0.35
    select_distance_weight: float = 0.25
    select_revisit_weight: float = 0.35
    select_stale_weight: float = 0.10

    stale_step_threshold: int = 20
    revisit_norm: int = 6

    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.cell_size = float(self.cell_size)
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")

        if self.search_radius is not None:
            self.search_radius = float(self.search_radius)
            if self.search_radius <= 0:
                raise ValueError("search_radius must be positive")

        if self.search_bounds is not None and not isinstance(self.search_bounds, SearchBounds):
            if isinstance(self.search_bounds, dict):
                self.search_bounds = SearchBounds(**self.search_bounds)
            else:
                raise ValueError("search_bounds must be SearchBounds or dict")

        self.max_projection_distance = float(self.max_projection_distance)
        self.min_projection_distance = float(self.min_projection_distance)
        self.horizontal_fov_deg = float(self.horizontal_fov_deg)
        self.down_projection_radius = float(self.down_projection_radius)

        if self.max_projection_distance <= 0:
            raise ValueError("max_projection_distance must be positive")

    @classmethod
    def from_uavon_start(
        cls,
        start_pose: Any,
        search_radius: float,
        cell_size: float = 5.0,
        **kwargs: Any,
    ) -> "SemanticMapConfig":
        bounds = SearchBounds.from_uavon_start(
            start_pose=start_pose,
            search_radius=search_radius,
        )
        return cls(
            search_bounds=bounds,
            search_radius=float(search_radius),
            cell_size=cell_size,
            **kwargs,
        )

    @classmethod
    def from_bounds(
        cls,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        cell_size: float = 5.0,
        **kwargs: Any,
    ) -> "SemanticMapConfig":
        bounds = SearchBounds(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        return cls(
            search_bounds=bounds,
            search_radius=None,
            cell_size=cell_size,
            **kwargs,
        )


@dataclass
class SemanticMapCell:
    gx: int
    gy: int

    semantic_value: float = 0.0
    semantic_conf: float = 0.0
    visited_count: int = 0

    last_update_step: int = -1
    last_visit_step: int = -1

    status: MapCellStatus = MapCellStatus.UNKNOWN
    update_count: int = 0

    def effective_value(self) -> float:
        return clamp01(self.semantic_value) * clamp01(self.semantic_conf)

    def mark_visited(self, step_id: int) -> None:
        self.visited_count += 1
        self.last_visit_step = int(step_id)
        if self.status == MapCellStatus.UNKNOWN:
            self.status = MapCellStatus.VISITED

    def mark_rejected(self, step_id: int, value: float, confidence: float) -> None:
        self.semantic_value = clamp01(value)
        self.semantic_conf = clamp01(confidence)
        self.last_update_step = int(step_id)
        self.status = MapCellStatus.REJECTED

    def refresh_status(self, config: SemanticMapConfig) -> None:
        if self.status == MapCellStatus.REJECTED:
            return

        if (
            self.semantic_value >= config.high_value_threshold
            and self.semantic_conf >= config.high_conf_threshold
        ):
            self.status = MapCellStatus.HIGH_VALUE
            return

        if self.visited_count >= config.explored_visit_threshold:
            self.status = MapCellStatus.EXPLORED
            return

        if self.visited_count > 0:
            self.status = MapCellStatus.VISITED
            return

        if self.semantic_conf <= 0.0:
            self.status = MapCellStatus.UNKNOWN

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "gx": self.gx,
            "gy": self.gy,
            "semantic_value": float(self.semantic_value),
            "semantic_conf": float(self.semantic_conf),
            "effective_value": float(self.effective_value()),
            "visited_count": int(self.visited_count),
            "last_update_step": int(self.last_update_step),
            "last_visit_step": int(self.last_visit_step),
            "status": self.status.value,
            "update_count": int(self.update_count),
        }


@dataclass
class SemanticRegionTarget:
    gx: int
    gy: int
    world_position: Tuple[float, float, float]
    score: float
    semantic_value: float
    semantic_conf: float
    visited_count: int
    status: MapCellStatus
    reason: str = ""

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "gx": self.gx,
            "gy": self.gy,
            "world_position": list(self.world_position),
            "score": float(self.score),
            "semantic_value": float(self.semantic_value),
            "semantic_conf": float(self.semantic_conf),
            "visited_count": int(self.visited_count),
            "status": self.status.value,
            "reason": self.reason,
        }


class SemanticMap:
    """
    2.5D semantic value map for Task1.

    This map only stores region-level semantic evidence:
        semantic_value
        semantic_conf
        visited_count
        last_update_step
        last_visit_step
        status

    It does not store GDINO candidates, Task2 verification scores,
    bbox information, or target evidence.
    """

    def __init__(
        self,
        origin_pose: PoseRecord,
        config: SemanticMapConfig,
    ) -> None:
        self.origin_pose = origin_pose
        self.config = config

        if self.config.search_bounds is None:
            if self.config.search_radius is None:
                raise ValueError(
                    "SemanticMap requires search_bounds or search_radius. "
                    "Use SemanticMapConfig.from_uavon_start(...) for UAV-ON episodes."
                )
            self.search_bounds = SearchBounds.from_center_radius(
                center_x=origin_pose.x,
                center_y=origin_pose.y,
                radius=self.config.search_radius,
            )
        else:
            self.search_bounds = self.config.search_bounds

        self.width = max(
            1,
            int(math.ceil(self.search_bounds.width / self.config.cell_size)),
        )
        self.height = max(
            1,
            int(math.ceil(self.search_bounds.height / self.config.cell_size)),
        )

        self.grid: List[List[SemanticMapCell]] = []
        for gx in range(self.width):
            col = []
            for gy in range(self.height):
                col.append(SemanticMapCell(gx=gx, gy=gy))
            self.grid.append(col)

        self.created_step: int = 0
        self.last_decay_step: int = 0
        self.last_update_step: int = -1
        self.last_visit_step: int = -1

    # ------------------------------------------------------------------
    # Boundary helpers
    # ------------------------------------------------------------------

    def is_pose_in_bounds(self, pose: PoseRecord) -> bool:
        return self.search_bounds.contains_pose(pose)

    def is_world_in_bounds(self, position: Tuple[float, float, float]) -> bool:
        return self.search_bounds.contains_xy(position[0], position[1])

    def nearest_in_bounds_position(self, pose: PoseRecord) -> Tuple[float, float, float]:
        return self.search_bounds.nearest_in_bounds_position(pose, keep_z=True)

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def world_to_grid(self, position: Tuple[float, float, float]) -> Optional[Tuple[int, int]]:
        x, y, _ = position

        if not self.search_bounds.contains_xy(x, y):
            return None

        gx = int(math.floor((float(x) - self.search_bounds.x_min) / self.config.cell_size))
        gy = int(math.floor((float(y) - self.search_bounds.y_min) / self.config.cell_size))

        if gx >= self.width:
            gx = self.width - 1
        if gy >= self.height:
            gy = self.height - 1

        if not self.in_bounds(gx, gy):
            return None
        return gx, gy

    def grid_to_world(self, gx: int, gy: int, z: Optional[float] = None) -> Tuple[float, float, float]:
        if z is None:
            z = self.origin_pose.z

        x = self.search_bounds.x_min + (float(gx) + 0.5) * self.config.cell_size
        y = self.search_bounds.y_min + (float(gy) + 0.5) * self.config.cell_size

        x = min(max(x, self.search_bounds.x_min), self.search_bounds.x_max)
        y = min(max(y, self.search_bounds.y_min), self.search_bounds.y_max)

        return float(x), float(y), float(z)

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= int(gx) < self.width and 0 <= int(gy) < self.height

    def get_cell(self, gx: int, gy: int) -> Optional[SemanticMapCell]:
        if not self.in_bounds(gx, gy):
            return None
        return self.grid[int(gx)][int(gy)]

    def get_cell_by_world(self, position: Tuple[float, float, float]) -> Optional[SemanticMapCell]:
        grid_pos = self.world_to_grid(position)
        if grid_pos is None:
            return None
        return self.get_cell(grid_pos[0], grid_pos[1])

    # ------------------------------------------------------------------
    # Visit update
    # ------------------------------------------------------------------

    def mark_visited(
        self,
        pose: PoseRecord,
        step_id: int,
        radius_cells: int = 0,
    ) -> List[Tuple[int, int]]:
        center = self.world_to_grid(pose.xyz())
        if center is None:
            return []

        updated: List[Tuple[int, int]] = []
        cx, cy = center
        radius_cells = max(0, int(radius_cells))

        for gx in range(cx - radius_cells, cx + radius_cells + 1):
            for gy in range(cy - radius_cells, cy + radius_cells + 1):
                if not self.in_bounds(gx, gy):
                    continue
                if math.hypot(gx - cx, gy - cy) > radius_cells + 0.01:
                    continue

                cell = self.grid[gx][gy]
                cell.mark_visited(step_id)
                cell.refresh_status(self.config)
                updated.append((gx, gy))

        self.last_visit_step = max(self.last_visit_step, int(step_id))
        return updated

    # ------------------------------------------------------------------
    # Task1 update
    # ------------------------------------------------------------------

    def update_from_task1_result(
        self,
        result: Task1Result,
        frame_lookup: Dict[str, FrameRecord],
    ) -> Dict[str, Any]:
        summary = {
            "request_id": result.request_id,
            "episode_id": result.episode_id,
            "success": result.success,
            "updated_cells": 0,
            "missing_frames": [],
            "score_count": len(result.scores),
        }

        if not result.success:
            summary["error"] = result.error
            return summary

        for score in result.scores:
            frame = frame_lookup.get(score.frame_id)
            if frame is None:
                summary["missing_frames"].append(score.frame_id)
                continue

            count = self.update_from_region_score(score, frame)
            summary["updated_cells"] += count

        return summary

    def update_from_region_score(self, score: RegionScore, frame: FrameRecord) -> int:
        if score.episode_id != frame.episode_id:
            return 0
        if score.step_id != frame.step_id:
            return 0
        if score.view_id != frame.view_id:
            return 0

        projected_cells = self.project_frame_to_cells(frame)

        updated_count = 0
        for gx, gy, spatial_conf in projected_cells:
            cell = self.get_cell(gx, gy)
            if cell is None:
                continue
            if cell.status == MapCellStatus.REJECTED:
                continue

            obs_conf = clamp01(score.confidence * spatial_conf)
            if obs_conf < self.config.min_update_confidence:
                continue

            self._fuse_cell(
                cell=cell,
                obs_value=score.semantic_value,
                obs_conf=obs_conf,
                update_step=frame.step_id,
            )
            updated_count += 1

        self.last_update_step = max(self.last_update_step, int(frame.step_id))
        return updated_count

    def _fuse_cell(
        self,
        cell: SemanticMapCell,
        obs_value: float,
        obs_conf: float,
        update_step: int,
    ) -> None:
        obs_value = clamp01(obs_value)
        obs_conf = clamp01(obs_conf)

        old_value = clamp01(cell.semantic_value)
        old_conf = clamp01(cell.semantic_conf)

        denom = old_conf + obs_conf + self.config.eps
        fused_value = (old_value * old_conf + obs_value * obs_conf) / denom

        if old_conf <= self.config.eps:
            fused_conf = obs_conf
        else:
            fused_conf = (old_conf * old_conf + obs_conf * obs_conf) / denom

        cell.semantic_value = clamp01(fused_value)
        cell.semantic_conf = clamp01(fused_conf)
        cell.last_update_step = int(update_step)
        cell.update_count += 1
        cell.refresh_status(self.config)

    # ------------------------------------------------------------------
    # Depth-aware projection
    # ------------------------------------------------------------------

    def project_frame_to_cells(self, frame: FrameRecord) -> List[Tuple[int, int, float]]:
        if frame.view_id == ViewID.DOWN:
            return self._project_down_view(frame)

        direction_rad = self._view_direction_rad(frame.pose, frame.view_id)
        max_dist = self._estimate_visible_distance(frame)
        max_dist = max(self.config.min_projection_distance, max_dist)
        max_dist = min(self.config.max_projection_distance, max_dist)

        fov_rad = math.radians(self.config.horizontal_fov_deg)
        pose_xy = (float(frame.pose.x), float(frame.pose.y))

        cells: List[Tuple[int, int, float]] = []
        for gx in range(self.width):
            for gy in range(self.height):
                wx, wy, _ = self.grid_to_world(gx, gy, frame.pose.z)
                vx = wx - pose_xy[0]
                vy = wy - pose_xy[1]
                dist = math.hypot(vx, vy)

                if dist < self.config.cell_size * 0.3:
                    continue
                if dist > max_dist + self.config.depth_safety_margin:
                    continue

                angle = math.atan2(vy, vx)
                diff = abs(self._angle_diff(angle, direction_rad))
                if diff > fov_rad / 2.0:
                    continue

                angular_conf = max(0.0, 1.0 - diff / (fov_rad / 2.0 + self.config.eps))
                distance_conf = max(0.0, 1.0 - dist / (max_dist + self.config.eps))

                if dist > max_dist:
                    depth_conf = max(
                        0.0,
                        1.0 - (dist - max_dist) / (self.config.depth_safety_margin + self.config.eps),
                    )
                else:
                    depth_conf = 1.0

                spatial_conf = 0.25 + 0.75 * angular_conf
                spatial_conf *= 0.35 + 0.65 * distance_conf
                spatial_conf *= depth_conf
                spatial_conf = clamp01(spatial_conf)

                if spatial_conf >= self.config.min_update_confidence:
                    cells.append((gx, gy, spatial_conf))

        return cells

    def _project_down_view(self, frame: FrameRecord) -> List[Tuple[int, int, float]]:
        center = self.world_to_grid(frame.pose.xyz())
        if center is None:
            return []

        depth_dist = self._estimate_visible_distance(frame)
        radius = min(self.config.down_projection_radius, max(self.config.cell_size, depth_dist))
        radius_cells = max(1, int(math.ceil(radius / self.config.cell_size)))

        cx, cy = center
        cells: List[Tuple[int, int, float]] = []

        for gx in range(cx - radius_cells, cx + radius_cells + 1):
            for gy in range(cy - radius_cells, cy + radius_cells + 1):
                if not self.in_bounds(gx, gy):
                    continue

                wx, wy, _ = self.grid_to_world(gx, gy, frame.pose.z)
                dist = math.hypot(wx - frame.pose.x, wy - frame.pose.y)
                if dist > radius:
                    continue

                distance_conf = max(0.0, 1.0 - dist / (radius + self.config.eps))
                spatial_conf = 0.4 + 0.6 * distance_conf
                spatial_conf = clamp01(spatial_conf)

                if spatial_conf >= self.config.min_update_confidence:
                    cells.append((gx, gy, spatial_conf))

        return cells

    def _estimate_visible_distance(self, frame: FrameRecord) -> float:
        values = self._extract_depth_values(frame)

        if values.size == 0:
            return self.config.max_projection_distance

        percentile = np.percentile(values, self.config.depth_percentile)
        if not np.isfinite(percentile):
            return self.config.max_projection_distance

        depth_dist = float(percentile)
        if depth_dist <= 0.0:
            return self.config.max_projection_distance

        depth_dist = min(depth_dist, self.config.max_projection_distance)
        depth_dist = max(depth_dist, self.config.min_projection_distance)
        return depth_dist

    def _extract_depth_values(self, frame: FrameRecord) -> np.ndarray:
        arrays = []

        if frame.depth is not None:
            try:
                arr = np.asarray(frame.depth, dtype=np.float32).reshape(-1)
                arrays.append(arr)
            except Exception:
                pass

        if frame.depth_grid3x3 is not None:
            try:
                arr = np.asarray(frame.depth_grid3x3, dtype=np.float32).reshape(-1)
                arrays.append(arr)
            except Exception:
                pass

        if not arrays:
            return np.asarray([], dtype=np.float32)

        values = np.concatenate(arrays)
        values = values[np.isfinite(values)]
        values = values[values >= self.config.min_valid_depth]

        if values.size == 0:
            return np.asarray([], dtype=np.float32)

        values = values[values <= self.config.max_projection_distance * 5.0]
        return values.astype(np.float32)

    def _view_direction_rad(self, pose: PoseRecord, view_id: ViewID) -> float:
        yaw = self._yaw_to_rad(pose.yaw)

        if view_id == ViewID.FRONT:
            offset = 0.0
        elif view_id == ViewID.LEFT:
            offset = math.pi / 2.0
        elif view_id == ViewID.RIGHT:
            offset = -math.pi / 2.0
        else:
            offset = 0.0

        return self._normalize_angle(yaw + offset)

    @staticmethod
    def _yaw_to_rad(yaw: float) -> float:
        yaw = float(yaw)
        if abs(yaw) > 2.0 * math.pi + 1e-3:
            return math.radians(yaw)
        return yaw

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle <= -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = a - b
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff <= -math.pi:
            diff += 2.0 * math.pi
        return diff

    # ------------------------------------------------------------------
    # Decay and status
    # ------------------------------------------------------------------

    def decay(self, current_step: int) -> None:
        current_step = int(current_step)
        if current_step <= self.last_decay_step:
            return

        delta = current_step - self.last_decay_step
        conf_decay = self.config.semantic_conf_decay_per_step ** float(delta)

        for cell in self.iter_cells():
            if cell.status == MapCellStatus.REJECTED:
                continue

            if cell.semantic_conf > 0.0:
                cell.semantic_conf = clamp01(cell.semantic_conf * conf_decay)

            if (
                cell.visited_count >= self.config.explored_visit_threshold
                and cell.status != MapCellStatus.HIGH_VALUE
            ):
                cell.semantic_value = clamp01(cell.semantic_value * self.config.explored_value_decay)

            cell.refresh_status(self.config)

        self.last_decay_step = current_step

    def mark_rejected_region(
        self,
        gx: int,
        gy: int,
        step_id: int,
        radius_cells: int = 0,
    ) -> List[Tuple[int, int]]:
        updated: List[Tuple[int, int]] = []
        radius_cells = max(0, int(radius_cells))

        for x in range(gx - radius_cells, gx + radius_cells + 1):
            for y in range(gy - radius_cells, gy + radius_cells + 1):
                if not self.in_bounds(x, y):
                    continue
                if math.hypot(x - gx, y - gy) > radius_cells + 0.01:
                    continue

                cell = self.grid[x][y]
                cell.mark_rejected(
                    step_id=step_id,
                    value=self.config.rejected_value,
                    confidence=self.config.rejected_confidence,
                )
                updated.append((x, y))

        return updated

    # ------------------------------------------------------------------
    # Selection support
    # ------------------------------------------------------------------

    def select_best_region(
        self,
        current_pose: PoseRecord,
        current_step: int,
        include_unknown: bool = True,
    ) -> Optional[SemanticRegionTarget]:
        best: Optional[SemanticRegionTarget] = None

        for cell in self.iter_cells():
            if cell.status == MapCellStatus.REJECTED:
                continue
            if not include_unknown and cell.status == MapCellStatus.UNKNOWN:
                continue

            score, reason = self._score_cell_for_selection(cell, current_pose, current_step)
            if best is None or score > best.score:
                wx, wy, wz = self.grid_to_world(cell.gx, cell.gy, current_pose.z)
                best = SemanticRegionTarget(
                    gx=cell.gx,
                    gy=cell.gy,
                    world_position=(wx, wy, wz),
                    score=score,
                    semantic_value=cell.semantic_value,
                    semantic_conf=cell.semantic_conf,
                    visited_count=cell.visited_count,
                    status=cell.status,
                    reason=reason,
                )

        return best

    def _score_cell_for_selection(
        self,
        cell: SemanticMapCell,
        current_pose: PoseRecord,
        current_step: int,
    ) -> Tuple[float, str]:
        wx, wy, _ = self.grid_to_world(cell.gx, cell.gy, current_pose.z)
        dist = math.hypot(wx - current_pose.x, wy - current_pose.y)
        max_dist = max(self.search_bounds.width, self.search_bounds.height)
        dist_norm = min(1.0, dist / (max_dist + self.config.eps))

        semantic_score = cell.semantic_value * cell.semantic_conf

        exploration_gain = 1.0 / (1.0 + float(cell.visited_count))
        if cell.status == MapCellStatus.UNKNOWN:
            exploration_gain += 0.25

        revisit_penalty = min(1.0, float(cell.visited_count) / float(self.config.revisit_norm))

        if cell.last_update_step < 0:
            stale_penalty = 0.0
        else:
            stale_steps = max(0, int(current_step) - int(cell.last_update_step))
            stale_penalty = min(1.0, stale_steps / float(self.config.stale_step_threshold))

        score = (
            self.config.select_semantic_weight * semantic_score
            + self.config.select_exploration_weight * exploration_gain
            - self.config.select_distance_weight * dist_norm
            - self.config.select_revisit_weight * revisit_penalty
            - self.config.select_stale_weight * stale_penalty
        )

        reason = (
            "semantic={:.3f}, explore={:.3f}, dist={:.3f}, revisit={:.3f}, stale={:.3f}".format(
                semantic_score,
                exploration_gain,
                dist_norm,
                revisit_penalty,
                stale_penalty,
            )
        )
        return float(score), reason

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def iter_cells(self) -> Iterable[SemanticMapCell]:
        for gx in range(self.width):
            for gy in range(self.height):
                yield self.grid[gx][gy]

    def get_high_value_cells(self) -> List[SemanticMapCell]:
        cells = []
        for cell in self.iter_cells():
            if cell.status == MapCellStatus.HIGH_VALUE:
                cells.append(cell)
        cells.sort(key=lambda c: c.effective_value(), reverse=True)
        return cells

    def get_summary(self, top_k: int = 5) -> Dict[str, Any]:
        counts = {status.value: 0 for status in MapCellStatus}
        for cell in self.iter_cells():
            counts[cell.status.value] += 1

        high_cells = self.get_high_value_cells()[:top_k]
        return {
            "origin": self.origin_pose.to_log_dict(),
            "search_bounds": self.search_bounds.to_log_dict(),
            "width": self.width,
            "height": self.height,
            "cell_size": self.config.cell_size,
            "last_update_step": self.last_update_step,
            "last_visit_step": self.last_visit_step,
            "last_decay_step": self.last_decay_step,
            "status_counts": counts,
            "top_high_value_cells": [cell.to_log_dict() for cell in high_cells],
        }

    def to_log_dict(self, include_cells: bool = False) -> Dict[str, Any]:
        data = self.get_summary()
        if include_cells:
            data["cells"] = [cell.to_log_dict() for cell in self.iter_cells()]
        return data


__all__ = [
    "MapCellStatus",
    "SearchBounds",
    "SemanticMap",
    "SemanticMapCell",
    "SemanticMapConfig",
    "SemanticRegionTarget",
]
