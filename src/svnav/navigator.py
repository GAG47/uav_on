from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from svnav.action_adapter import ActionAdapter, ActionAdapterConfig
from svnav.semantic_map import MapCellStatus, SemanticMap, SemanticMapCell
from svnav.types import (
    ActionSource,
    NavDecision,
    NavMode,
    ObservationRecord,
    PoseRecord,
)


@dataclass
class SearchNavigatorConfig:
    semantic_clear_threshold: float = 0.28
    semantic_margin: float = 0.08

    semantic_weight_clear: float = 1.20
    exploration_weight_clear: float = 0.25
    heading_weight_clear: float = 0.10

    semantic_weight_unclear: float = 0.35
    exploration_weight_unclear: float = 1.00
    heading_weight_unclear: float = 0.45

    distance_weight: float = 0.30
    revisit_weight: float = 0.45
    boundary_weight: float = 0.10

    revisit_norm: int = 6
    min_target_distance: float = 5.0

    unknown_bonus: float = 0.25
    high_value_bonus: float = 0.15
    explored_penalty: float = 0.20

    def __post_init__(self) -> None:
        self.semantic_clear_threshold = float(self.semantic_clear_threshold)
        self.semantic_margin = float(self.semantic_margin)

        self.semantic_weight_clear = float(self.semantic_weight_clear)
        self.exploration_weight_clear = float(self.exploration_weight_clear)
        self.heading_weight_clear = float(self.heading_weight_clear)

        self.semantic_weight_unclear = float(self.semantic_weight_unclear)
        self.exploration_weight_unclear = float(self.exploration_weight_unclear)
        self.heading_weight_unclear = float(self.heading_weight_unclear)

        self.distance_weight = float(self.distance_weight)
        self.revisit_weight = float(self.revisit_weight)
        self.boundary_weight = float(self.boundary_weight)

        self.revisit_norm = int(self.revisit_norm)
        self.min_target_distance = float(self.min_target_distance)


@dataclass
class SearchTarget:
    gx: int
    gy: int
    world_position: Tuple[float, float, float]
    score: float
    semantic_score: float
    exploration_gain: float
    heading_alignment: float
    distance_cost: float
    revisit_penalty: float
    boundary_risk: float
    status: MapCellStatus
    visited_count: int
    semantic_clear: bool
    reason: str

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "gx": self.gx,
            "gy": self.gy,
            "world_position": list(self.world_position),
            "score": float(self.score),
            "semantic_score": float(self.semantic_score),
            "exploration_gain": float(self.exploration_gain),
            "heading_alignment": float(self.heading_alignment),
            "distance_cost": float(self.distance_cost),
            "revisit_penalty": float(self.revisit_penalty),
            "boundary_risk": float(self.boundary_risk),
            "status": self.status.value,
            "visited_count": int(self.visited_count),
            "semantic_clear": bool(self.semantic_clear),
            "reason": self.reason,
        }


class SearchNavigator:
    """
    Search-stage navigator.

    It selects a region target from SemanticMap and asks ActionAdapter to
    convert the selected target into a UAV-ON executable action.

    It does not call VLM, does not call GDINO, does not verify targets,
    and does not decide stop.
    """

    def __init__(
        self,
        config: Optional[SearchNavigatorConfig] = None,
        action_adapter: Optional[ActionAdapter] = None,
        action_adapter_config: Optional[ActionAdapterConfig] = None,
    ) -> None:
        self.config = config or SearchNavigatorConfig()
        self.action_adapter = action_adapter or ActionAdapter(action_adapter_config)

    def decide_search(
        self,
        episode_id: str,
        step_id: int,
        observation: ObservationRecord,
        semantic_map: SemanticMap,
    ) -> NavDecision:
        current_pose = observation.pose

        if not semantic_map.is_pose_in_bounds(current_pose):
            target_position = semantic_map.nearest_in_bounds_position(current_pose)
            return self.action_adapter.target_to_action(
                episode_id=episode_id,
                step_id=step_id,
                current_pose=current_pose,
                target_position=target_position,
                observation=observation,
                semantic_map=semantic_map,
                mode=NavMode.SEARCH,
                target_type="in_bounds_region",
                action_source=ActionSource.GEOMETRIC_EXPLORE,
                reason="uav outside semantic map; return to nearest in-bounds position",
                debug_info={
                    "navigator_reason": "out_of_bounds",
                    "target_position": list(target_position),
                },
            )

        target = self.select_search_target(
            current_pose=current_pose,
            current_step=step_id,
            semantic_map=semantic_map,
        )

        if target is None:
            return NavDecision(
                episode_id=episode_id,
                step_id=step_id,
                mode=NavMode.SEARCH,
                target_type="none",
                action="rotl",
                step_size=self.action_adapter.config.yaw_step_size,
                action_source=ActionSource.GEOMETRIC_EXPLORE,
                reason="no valid search target; rotate to observe",
                debug_info={"navigator_reason": "no_valid_target"},
            )

        source = (
            ActionSource.SEMANTIC_MAP
            if target.semantic_clear
            else ActionSource.GEOMETRIC_EXPLORE
        )

        return self.action_adapter.target_to_action(
            episode_id=episode_id,
            step_id=step_id,
            current_pose=current_pose,
            target_position=target.world_position,
            observation=observation,
            semantic_map=semantic_map,
            mode=NavMode.SEARCH,
            target_type="region",
            action_source=source,
            reason=target.reason,
            debug_info={
                "navigator_reason": "semantic_search" if target.semantic_clear else "geometric_explore",
                "search_target": target.to_log_dict(),
            },
        )

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def select_search_target(
        self,
        current_pose: PoseRecord,
        current_step: int,
        semantic_map: SemanticMap,
    ) -> Optional[SearchTarget]:
        semantic_clear, semantic_stats = self._is_semantic_preference_clear(
            semantic_map=semantic_map
        )

        candidates: List[SearchTarget] = []
        for cell in semantic_map.iter_cells():
            if cell.status == MapCellStatus.REJECTED:
                continue

            target = self._score_cell(
                cell=cell,
                current_pose=current_pose,
                current_step=current_step,
                semantic_map=semantic_map,
                semantic_clear=semantic_clear,
                allow_close_target=False,
            )

            if target is None:
                continue

            candidates.append(target)

        if not candidates:
            for cell in semantic_map.iter_cells():
                if cell.status == MapCellStatus.REJECTED:
                    continue

                target = self._score_cell(
                    cell=cell,
                    current_pose=current_pose,
                    current_step=current_step,
                    semantic_map=semantic_map,
                    semantic_clear=semantic_clear,
                    allow_close_target=True,
                )

                if target is None:
                    continue

                candidates.append(target)

        if not candidates:
            return None

        candidates.sort(key=lambda item: item.score, reverse=True)
        best = candidates[0]
        best.reason = "{}; {}".format(
            best.reason,
            "semantic_stats={}".format(semantic_stats),
        )
        return best

    def _is_semantic_preference_clear(
        self,
        semantic_map: SemanticMap,
    ) -> Tuple[bool, Dict[str, Any]]:
        values = []
        high_value_count = 0

        for cell in semantic_map.iter_cells():
            if cell.status == MapCellStatus.REJECTED:
                continue

            effective = cell.semantic_value * cell.semantic_conf
            values.append(effective)

            if (
                cell.status == MapCellStatus.HIGH_VALUE
                and effective >= self.config.semantic_clear_threshold
            ):
                high_value_count += 1

        if not values:
            return False, {
                "best": 0.0,
                "second": 0.0,
                "margin": 0.0,
                "high_value_count": 0,
                "clear": False,
            }

        values.sort(reverse=True)
        best = float(values[0])
        second = float(values[1]) if len(values) > 1 else 0.0
        margin = best - second

        clear = False

        if best >= self.config.semantic_clear_threshold:
            if high_value_count > 0:
                clear = True
            elif margin >= self.config.semantic_margin:
                clear = True

        return clear, {
            "best": best,
            "second": second,
            "margin": margin,
            "high_value_count": high_value_count,
            "clear": clear,
        }

    def _score_cell(
        self,
        cell: SemanticMapCell,
        current_pose: PoseRecord,
        current_step: int,
        semantic_map: SemanticMap,
        semantic_clear: bool,
        allow_close_target: bool,
    ) -> Optional[SearchTarget]:
        wx, wy, wz = semantic_map.grid_to_world(cell.gx, cell.gy, current_pose.z)
        dist = math.hypot(wx - current_pose.x, wy - current_pose.y)

        if not allow_close_target and dist < self.config.min_target_distance:
            return None

        max_dist = max(
            semantic_map.search_bounds.width,
            semantic_map.search_bounds.height,
            semantic_map.config.eps,
        )
        distance_cost = min(1.0, dist / max_dist)

        semantic_score = cell.semantic_value * cell.semantic_conf
        exploration_gain = self._exploration_gain(cell)
        heading_alignment = self._heading_alignment(
            current_pose=current_pose,
            target_x=wx,
            target_y=wy,
        )
        revisit_penalty = min(
            1.0,
            float(cell.visited_count) / float(max(1, self.config.revisit_norm)),
        )
        boundary_risk = self._boundary_risk(wx, wy, semantic_map)

        if semantic_clear:
            semantic_weight = self.config.semantic_weight_clear
            exploration_weight = self.config.exploration_weight_clear
            heading_weight = self.config.heading_weight_clear
        else:
            semantic_weight = self.config.semantic_weight_unclear
            exploration_weight = self.config.exploration_weight_unclear
            heading_weight = self.config.heading_weight_unclear

        score = (
            semantic_weight * semantic_score
            + exploration_weight * exploration_gain
            + heading_weight * heading_alignment
            - self.config.distance_weight * distance_cost
            - self.config.revisit_weight * revisit_penalty
            - self.config.boundary_weight * boundary_risk
        )

        reason = (
            "semantic_clear={}, semantic={:.3f}, explore={:.3f}, "
            "heading={:.3f}, dist={:.3f}, revisit={:.3f}, boundary={:.3f}"
        ).format(
            semantic_clear,
            semantic_score,
            exploration_gain,
            heading_alignment,
            distance_cost,
            revisit_penalty,
            boundary_risk,
        )

        if allow_close_target:
            reason = "{}; allow_close_target=True".format(reason)

        return SearchTarget(
            gx=cell.gx,
            gy=cell.gy,
            world_position=(wx, wy, wz),
            score=float(score),
            semantic_score=float(semantic_score),
            exploration_gain=float(exploration_gain),
            heading_alignment=float(heading_alignment),
            distance_cost=float(distance_cost),
            revisit_penalty=float(revisit_penalty),
            boundary_risk=float(boundary_risk),
            status=cell.status,
            visited_count=cell.visited_count,
            semantic_clear=semantic_clear,
            reason=reason,
        )

    def _exploration_gain(self, cell: SemanticMapCell) -> float:
        gain = 1.0 / (1.0 + float(cell.visited_count))

        if cell.status == MapCellStatus.UNKNOWN:
            gain += self.config.unknown_bonus

        if cell.status == MapCellStatus.HIGH_VALUE:
            gain += self.config.high_value_bonus

        if cell.status == MapCellStatus.EXPLORED:
            gain -= self.config.explored_penalty

        return max(0.0, gain)

    def _heading_alignment(
        self,
        current_pose: PoseRecord,
        target_x: float,
        target_y: float,
    ) -> float:
        dx = float(target_x) - float(current_pose.x)
        dy = float(target_y) - float(current_pose.y)

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        yaw = self._yaw_to_rad(current_pose.yaw)
        target_angle = math.atan2(dy, dx)
        diff = abs(self._angle_diff(target_angle, yaw))

        # front: 1.0, side: 0.5, back: 0.0
        return max(0.0, (math.cos(diff) + 1.0) * 0.5)

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
        norm = min_dist / max(semantic_map.config.cell_size * 2.0, semantic_map.config.eps)

        return max(0.0, 1.0 - min(1.0, norm))

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
    "SearchNavigator",
    "SearchNavigatorConfig",
    "SearchTarget",
]
