from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from svnav.semantic_map import MapCellStatus, SemanticMap, SemanticMapCell
from svnav.types import PoseRecord, new_id


@dataclass
class SearchRegion:
    """
    A temporary search target region derived from the current SemanticMap.

    This is not a final object target. It is a frontier-like region that is
    worth exploring or observing. The planner uses it to generate viewpoints.
    """

    region_id: str
    center: Tuple[float, float, float]
    source: str
    score: float
    semantic_score: float
    exploration_score: float
    distance_cost: float
    boundary_risk: float
    cells: List[Tuple[int, int]] = field(default_factory=list)
    reason: str = ""

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "center": list(self.center),
            "source": self.source,
            "score": float(self.score),
            "semantic_score": float(self.semantic_score),
            "exploration_score": float(self.exploration_score),
            "distance_cost": float(self.distance_cost),
            "boundary_risk": float(self.boundary_risk),
            "cells": [list(c) for c in self.cells],
            "reason": self.reason,
        }


@dataclass
class SearchViewpoint:
    """
    A viewpoint for observing a SearchRegion.

    waypoint = position
    yaw      = desired viewing yaw at the waypoint
    look_at  = region center that this viewpoint observes
    """

    viewpoint_id: str
    region: SearchRegion
    position: Tuple[float, float, float]
    yaw: float
    look_at: Tuple[float, float, float]
    score: float
    distance_cost: float
    yaw_cost: float
    boundary_risk: float
    reason: str = ""
    created_step: int = -1

    @property
    def region_id(self) -> str:
        return self.region.region_id

    @property
    def target_position(self) -> Tuple[float, float, float]:
        return self.position

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint_id": self.viewpoint_id,
            "region_id": self.region.region_id,
            "region_source": self.region.source,
            "position": list(self.position),
            "yaw": float(self.yaw),
            "look_at": list(self.look_at),
            "score": float(self.score),
            "distance_cost": float(self.distance_cost),
            "yaw_cost": float(self.yaw_cost),
            "boundary_risk": float(self.boundary_risk),
            "created_step": int(self.created_step),
            "reason": self.reason,
            "region": self.region.to_log_dict(),
        }


@dataclass
class SearchViewpointPlannerConfig:
    max_regions: int = 12
    max_viewpoints_per_region: int = 8

    semantic_threshold: float = 0.20
    semantic_activation_threshold: float = 0.25
    semantic_topk: int = 3

    min_region_distance: float = 6.0
    region_cluster_radius: float = 10.0

    view_distance: float = 10.0
    min_view_distance: float = 8.0
    max_view_distance: float = 16.0
    min_viewpoint_distance_from_uav: float = 8.0

    semantic_weight: float = 1.80
    semantic_region_boost: float = 0.25
    semantic_region_exploration_weight: float = 0.45

    geometric_exploration_weight: float = 0.90
    geometric_exploration_weight_when_semantic_active: float = 0.35

    distance_weight: float = 0.35
    yaw_weight: float = 0.30
    boundary_weight: float = 0.25

    unknown_bonus: float = 0.30
    high_value_bonus: float = 0.20
    explored_penalty: float = 0.25

    angle_offsets_deg: Tuple[float, ...] = (
        0.0,
        45.0,
        -45.0,
        90.0,
        -90.0,
        135.0,
        -135.0,
        180.0,
    )

    def __post_init__(self) -> None:
        self.max_regions = int(self.max_regions)
        self.max_viewpoints_per_region = int(self.max_viewpoints_per_region)

        self.semantic_threshold = float(self.semantic_threshold)
        self.semantic_activation_threshold = float(self.semantic_activation_threshold)
        self.semantic_topk = int(self.semantic_topk)

        self.min_region_distance = float(self.min_region_distance)
        self.region_cluster_radius = float(self.region_cluster_radius)

        self.view_distance = float(self.view_distance)
        self.min_view_distance = float(self.min_view_distance)
        self.max_view_distance = float(self.max_view_distance)
        self.min_viewpoint_distance_from_uav = float(self.min_viewpoint_distance_from_uav)

        self.semantic_weight = float(self.semantic_weight)
        self.semantic_region_boost = float(self.semantic_region_boost)
        self.semantic_region_exploration_weight = float(self.semantic_region_exploration_weight)

        self.geometric_exploration_weight = float(self.geometric_exploration_weight)
        self.geometric_exploration_weight_when_semantic_active = float(
            self.geometric_exploration_weight_when_semantic_active
        )

        self.distance_weight = float(self.distance_weight)
        self.yaw_weight = float(self.yaw_weight)
        self.boundary_weight = float(self.boundary_weight)

        self.unknown_bonus = float(self.unknown_bonus)
        self.high_value_bonus = float(self.high_value_bonus)
        self.explored_penalty = float(self.explored_penalty)


class SearchViewpointPlanner:
    """
    Build SearchRegion and SearchViewpoint candidates from the current map.

    Search regions are built in two separate groups:
        1. semantic regions from high-value / high-semantic cells
        2. geometric regions from unknown / low-visited cells

    The planner first selects a SearchRegion, then selects the best viewpoint
    for that region. This avoids letting a nearby geometric viewpoint override
    an already meaningful semantic region.
    """

    def __init__(
        self,
        config: Optional[SearchViewpointPlannerConfig] = None,
    ) -> None:
        self.config = config or SearchViewpointPlannerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_best_viewpoint(
        self,
        current_pose: PoseRecord,
        semantic_map: SemanticMap,
        current_step: int,
    ) -> Optional[SearchViewpoint]:
        regions = self.build_search_regions(
            current_pose=current_pose,
            semantic_map=semantic_map,
        )

        for region in regions:
            candidates = self.build_viewpoints_for_region(
                region=region,
                current_pose=current_pose,
                semantic_map=semantic_map,
                current_step=current_step,
            )

            if not candidates:
                continue

            candidates.sort(key=lambda item: item.score, reverse=True)
            return candidates[0]

        return None

    def best_region_score(
        self,
        current_pose: PoseRecord,
        semantic_map: SemanticMap,
    ) -> float:
        regions = self.build_search_regions(
            current_pose=current_pose,
            semantic_map=semantic_map,
        )
        if not regions:
            return 0.0
        return float(regions[0].score)

    def build_search_regions(
        self,
        current_pose: PoseRecord,
        semantic_map: SemanticMap,
    ) -> List[SearchRegion]:
        semantic_candidates = self._score_semantic_cells(
            current_pose=current_pose,
            semantic_map=semantic_map,
        )

        semantic_active = self._has_active_semantics(
            semantic_candidates=semantic_candidates,
            semantic_map=semantic_map,
        )

        geometric_candidates = self._score_geometric_cells(
            current_pose=current_pose,
            semantic_map=semantic_map,
            semantic_active=semantic_active,
        )

        regions: List[SearchRegion] = []

        regions.extend(
            self._build_regions_from_candidates(
                candidates=semantic_candidates,
                current_pose=current_pose,
                semantic_map=semantic_map,
                source="semantic",
                semantic_active=semantic_active,
            )
        )

        regions.extend(
            self._build_regions_from_candidates(
                candidates=geometric_candidates,
                current_pose=current_pose,
                semantic_map=semantic_map,
                source="geometric",
                semantic_active=semantic_active,
            )
        )

        regions.sort(key=lambda item: item.score, reverse=True)
        return regions[: self.config.max_regions]

    def build_viewpoints_for_region(
        self,
        region: SearchRegion,
        current_pose: PoseRecord,
        semantic_map: SemanticMap,
        current_step: int,
    ) -> List[SearchViewpoint]:
        rx, ry, rz = region.center

        base_angle = math.atan2(
            float(current_pose.y) - float(ry),
            float(current_pose.x) - float(rx),
        )

        candidates: List[SearchViewpoint] = []
        distances = self._view_distances()

        for dist in distances:
            for offset_deg in self.config.angle_offsets_deg:
                theta = base_angle + math.radians(float(offset_deg))
                vx = float(rx) + math.cos(theta) * dist
                vy = float(ry) + math.sin(theta) * dist
                vz = float(current_pose.z)

                distance_from_uav = math.hypot(
                    vx - float(current_pose.x),
                    vy - float(current_pose.y),
                )
                if distance_from_uav < self.config.min_viewpoint_distance_from_uav:
                    continue

                pose = PoseRecord(x=vx, y=vy, z=vz, yaw=0.0)
                if not semantic_map.is_pose_in_bounds(pose):
                    continue

                yaw = math.degrees(math.atan2(float(ry) - vy, float(rx) - vx))

                viewpoint = self._make_viewpoint(
                    region=region,
                    position=(vx, vy, vz),
                    yaw=yaw,
                    look_at=region.center,
                    current_pose=current_pose,
                    semantic_map=semantic_map,
                    current_step=current_step,
                )
                candidates.append(viewpoint)

        candidates.sort(key=lambda v: v.score, reverse=True)
        return candidates[: self.config.max_viewpoints_per_region]

    # ------------------------------------------------------------------
    # Cell candidate construction
    # ------------------------------------------------------------------

    def _score_semantic_cells(
        self,
        current_pose: PoseRecord,
        semantic_map: SemanticMap,
    ) -> List[Dict[str, Any]]:
        items = []

        for cell in semantic_map.iter_cells():
            if cell.status == MapCellStatus.REJECTED:
                continue

            effective = float(cell.semantic_value * cell.semantic_conf)

            if not self._is_semantic_cell(cell, effective):
                continue

            wx, wy, wz = semantic_map.grid_to_world(
                cell.gx,
                cell.gy,
                current_pose.z,
            )
            dist = math.hypot(wx - current_pose.x, wy - current_pose.y)

            if dist < self.config.min_region_distance:
                continue

            distance_cost = self._distance_cost(dist, semantic_map)
            boundary_risk = self._boundary_risk(wx, wy, semantic_map)
            exploration_score = self._exploration_score(cell)

            score = (
                effective
                + self.config.high_value_bonus
                + 0.20 * exploration_score
                - self.config.distance_weight * distance_cost
                - self.config.boundary_weight * boundary_risk
            )

            items.append(
                {
                    "cell": cell,
                    "world": (wx, wy, wz),
                    "score": float(score),
                    "semantic_score": float(effective),
                    "exploration_score": float(exploration_score),
                    "distance_cost": float(distance_cost),
                    "boundary_risk": float(boundary_risk),
                }
            )

        items.sort(key=lambda item: item["score"], reverse=True)
        return items

    def _score_geometric_cells(
        self,
        current_pose: PoseRecord,
        semantic_map: SemanticMap,
        semantic_active: bool,
    ) -> List[Dict[str, Any]]:
        items = []

        for cell in semantic_map.iter_cells():
            if cell.status == MapCellStatus.REJECTED:
                continue

            effective = float(cell.semantic_value * cell.semantic_conf)

            if self._is_semantic_cell(cell, effective):
                continue

            wx, wy, wz = semantic_map.grid_to_world(
                cell.gx,
                cell.gy,
                current_pose.z,
            )
            dist = math.hypot(wx - current_pose.x, wy - current_pose.y)

            if dist < self.config.min_region_distance:
                continue

            distance_cost = self._distance_cost(dist, semantic_map)
            boundary_risk = self._boundary_risk(wx, wy, semantic_map)
            exploration_score = self._exploration_score(cell)

            exploration_weight = (
                self.config.geometric_exploration_weight_when_semantic_active
                if semantic_active
                else self.config.geometric_exploration_weight
            )

            score = (
                exploration_weight * exploration_score
                - self.config.distance_weight * distance_cost
                - self.config.boundary_weight * boundary_risk
            )

            items.append(
                {
                    "cell": cell,
                    "world": (wx, wy, wz),
                    "score": float(score),
                    "semantic_score": float(effective),
                    "exploration_score": float(exploration_score),
                    "distance_cost": float(distance_cost),
                    "boundary_risk": float(boundary_risk),
                }
            )

        items.sort(key=lambda item: item["score"], reverse=True)
        return items

    def _is_semantic_cell(
        self,
        cell: SemanticMapCell,
        effective_value: float,
    ) -> bool:
        if cell.status == MapCellStatus.HIGH_VALUE:
            return True
        return float(effective_value) >= self.config.semantic_threshold

    def _has_active_semantics(
        self,
        semantic_candidates: List[Dict[str, Any]],
        semantic_map: SemanticMap,
    ) -> bool:
        if semantic_candidates:
            return True

        best = 0.0
        high_value_count = 0

        for cell in semantic_map.iter_cells():
            effective = float(cell.semantic_value * cell.semantic_conf)
            best = max(best, effective)

            if cell.status == MapCellStatus.HIGH_VALUE:
                high_value_count += 1

        if high_value_count > 0:
            return True

        return best >= self.config.semantic_activation_threshold

    # ------------------------------------------------------------------
    # Region construction
    # ------------------------------------------------------------------

    def _build_regions_from_candidates(
        self,
        candidates: List[Dict[str, Any]],
        current_pose: PoseRecord,
        semantic_map: SemanticMap,
        source: str,
        semantic_active: bool,
    ) -> List[SearchRegion]:
        if not candidates:
            return []

        regions: List[SearchRegion] = []
        used = set()

        for item in candidates:
            cell = item["cell"]
            key = (cell.gx, cell.gy)

            if key in used:
                continue

            cluster = self._collect_cluster(
                seed=cell,
                candidates=candidates,
                used=used,
                semantic_map=semantic_map,
            )

            if not cluster:
                continue

            region = self._make_region(
                cluster=cluster,
                current_pose=current_pose,
                semantic_map=semantic_map,
                source=source,
                semantic_active=semantic_active,
            )
            regions.append(region)

            for cluster_cell in cluster:
                used.add((cluster_cell.gx, cluster_cell.gy))

        regions.sort(key=lambda item: item.score, reverse=True)
        return regions

    def _collect_cluster(
        self,
        seed: SemanticMapCell,
        candidates: List[Dict[str, Any]],
        used: set,
        semantic_map: SemanticMap,
    ) -> List[SemanticMapCell]:
        sx, sy, _ = semantic_map.grid_to_world(seed.gx, seed.gy, 0.0)
        cluster = []

        for item in candidates:
            cell = item["cell"]
            key = (cell.gx, cell.gy)

            if key in used:
                continue

            wx, wy, _ = semantic_map.grid_to_world(cell.gx, cell.gy, 0.0)
            if math.hypot(wx - sx, wy - sy) <= self.config.region_cluster_radius:
                cluster.append(cell)

        return cluster

    def _make_region(
        self,
        cluster: List[SemanticMapCell],
        current_pose: PoseRecord,
        semantic_map: SemanticMap,
        source: str,
        semantic_active: bool,
    ) -> SearchRegion:
        xs = []
        ys = []
        zs = []
        semantic_scores = []
        exploration_scores = []
        cells = []

        for cell in cluster:
            wx, wy, wz = semantic_map.grid_to_world(
                cell.gx,
                cell.gy,
                current_pose.z,
            )
            xs.append(wx)
            ys.append(wy)
            zs.append(wz)
            cells.append((cell.gx, cell.gy))

            semantic_score = float(cell.semantic_value * cell.semantic_conf)
            exploration_score = self._exploration_score(cell)

            semantic_scores.append(semantic_score)
            exploration_scores.append(exploration_score)

        cx = sum(xs) / max(1, len(xs))
        cy = sum(ys) / max(1, len(ys))
        cz = sum(zs) / max(1, len(zs))

        if source == "semantic":
            semantic_score = self._topk_mean(
                semantic_scores,
                k=self.config.semantic_topk,
            )
            exploration_score = self._topk_mean(
                exploration_scores,
                k=self.config.semantic_topk,
            )
        else:
            semantic_score = sum(semantic_scores) / max(1, len(semantic_scores))
            exploration_score = sum(exploration_scores) / max(1, len(exploration_scores))

        dist = math.hypot(cx - current_pose.x, cy - current_pose.y)
        distance_cost = self._distance_cost(dist, semantic_map)
        boundary_risk = self._boundary_risk(cx, cy, semantic_map)

        if source == "semantic":
            score = (
                self.config.semantic_weight * semantic_score
                + self.config.semantic_region_boost
                + self.config.semantic_region_exploration_weight * exploration_score
                - self.config.distance_weight * distance_cost
                - self.config.boundary_weight * boundary_risk
            )
        else:
            exploration_weight = (
                self.config.geometric_exploration_weight_when_semantic_active
                if semantic_active
                else self.config.geometric_exploration_weight
            )
            score = (
                exploration_weight * exploration_score
                - self.config.distance_weight * distance_cost
                - self.config.boundary_weight * boundary_risk
            )

        reason = (
            "source={}, cells={}, semantic={:.3f}, explore={:.3f}, "
            "dist={:.3f}, boundary={:.3f}, semantic_active={}"
        ).format(
            source,
            len(cluster),
            semantic_score,
            exploration_score,
            distance_cost,
            boundary_risk,
            semantic_active,
        )

        return SearchRegion(
            region_id=new_id("region"),
            center=(cx, cy, cz),
            source=source,
            score=float(score),
            semantic_score=float(semantic_score),
            exploration_score=float(exploration_score),
            distance_cost=float(distance_cost),
            boundary_risk=float(boundary_risk),
            cells=cells,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Viewpoint scoring
    # ------------------------------------------------------------------

    def _make_viewpoint(
        self,
        region: SearchRegion,
        position: Tuple[float, float, float],
        yaw: float,
        look_at: Tuple[float, float, float],
        current_pose: PoseRecord,
        semantic_map: SemanticMap,
        current_step: int,
    ) -> SearchViewpoint:
        px, py, pz = position
        dist = math.hypot(px - current_pose.x, py - current_pose.y)

        distance_cost = self._distance_cost(dist, semantic_map)

        target_angle = math.atan2(py - current_pose.y, px - current_pose.x)
        yaw_to_waypoint = abs(
            self._angle_diff(target_angle, self._yaw_to_rad(current_pose.yaw))
        )
        yaw_cost = min(1.0, yaw_to_waypoint / math.pi)

        boundary_risk = self._boundary_risk(px, py, semantic_map)

        score = (
            region.score
            - self.config.distance_weight * distance_cost
            - self.config.yaw_weight * yaw_cost
            - self.config.boundary_weight * boundary_risk
        )

        reason = (
            "region_score={:.3f}, dist={:.3f}, yaw={:.3f}, boundary={:.3f}"
        ).format(region.score, distance_cost, yaw_cost, boundary_risk)

        return SearchViewpoint(
            viewpoint_id=new_id("vp"),
            region=region,
            position=position,
            yaw=float(yaw),
            look_at=look_at,
            score=float(score),
            distance_cost=float(distance_cost),
            yaw_cost=float(yaw_cost),
            boundary_risk=float(boundary_risk),
            reason=reason,
            created_step=int(current_step),
        )

    # ------------------------------------------------------------------
    # Score helpers
    # ------------------------------------------------------------------

    def _exploration_score(self, cell: SemanticMapCell) -> float:
        score = 1.0 / (1.0 + float(cell.visited_count))

        if cell.status == MapCellStatus.UNKNOWN:
            score += self.config.unknown_bonus

        if cell.status == MapCellStatus.HIGH_VALUE:
            score += self.config.high_value_bonus

        if cell.status == MapCellStatus.EXPLORED:
            score -= self.config.explored_penalty

        return max(0.0, float(score))

    def _distance_cost(
        self,
        distance: float,
        semantic_map: SemanticMap,
    ) -> float:
        max_dist = max(
            semantic_map.search_bounds.width,
            semantic_map.search_bounds.height,
            semantic_map.config.eps,
        )
        return min(1.0, float(distance) / max_dist)

    def _boundary_risk(
        self,
        x: float,
        y: float,
        semantic_map: SemanticMap,
    ) -> float:
        bounds = semantic_map.search_bounds

        dist_left = abs(float(x) - bounds.x_min)
        dist_right = abs(bounds.x_max - float(x))
        dist_bottom = abs(float(y) - bounds.y_min)
        dist_top = abs(bounds.y_max - float(y))

        min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
        norm = min_dist / max(
            semantic_map.config.cell_size * 2.0,
            semantic_map.config.eps,
        )

        return max(0.0, 1.0 - min(1.0, norm))

    def _view_distances(self) -> List[float]:
        values = [
            self.config.view_distance,
            self.config.min_view_distance,
            self.config.max_view_distance,
        ]
        clean = []

        for value in values:
            value = float(value)
            if value <= 0:
                continue
            if value not in clean:
                clean.append(value)

        return clean

    @staticmethod
    def _topk_mean(values: List[float], k: int) -> float:
        if not values:
            return 0.0

        k = max(1, int(k))
        sorted_values = sorted([float(v) for v in values], reverse=True)
        selected = sorted_values[:k]
        return sum(selected) / max(1, len(selected))

    @staticmethod
    def _yaw_to_rad(yaw: float) -> float:
        yaw = float(yaw)
        if abs(yaw) > 2.0 * math.pi + 1e-3:
            return math.radians(yaw)
        return yaw

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = a - b
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff <= -math.pi:
            diff += 2.0 * math.pi
        return diff


__all__ = [
    "SearchRegion",
    "SearchViewpoint",
    "SearchViewpointPlanner",
    "SearchViewpointPlannerConfig",
]
