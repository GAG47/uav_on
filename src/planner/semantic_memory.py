import math
import numpy as np

try:
    from src.planner.semantic_frontier import SemanticFrontierBuilder
    from src.planner.viewpoint_planner import ViewpointPlanner
    from src.planner.sgcp_planner import SGCPPlanner
except Exception:
    from planner.semantic_frontier import SemanticFrontierBuilder
    from planner.viewpoint_planner import ViewpointPlanner
    from planner.sgcp_planner import SGCPPlanner


class SemanticMemory:
    def __init__(
        self,
        origin,
        resolution=2.0,
        map_size=100.0,
        max_sensing_range=25.0,
        sector_angle=70.0,
        visited_radius=3.0,
    ):
        self.origin = np.array(origin[:3], dtype=float)
        self.resolution = resolution
        self.map_size = map_size
        self.max_sensing_range = max_sensing_range
        self.sector_angle = sector_angle
        self.visited_radius = visited_radius

        self.grid_size = int(math.ceil(self.map_size / self.resolution))
        self.center_idx = self.grid_size // 2

        self.semantic_value = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.confidence = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.safety_value = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.novelty_value = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        self.observe_count = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.last_update_step = -np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=np.bool_)

        self.last_region_update = {}
        self.last_frontiers = []

        self.last_selected_frontier = None
        self.frontier_selection_count = {}
        self.frontier_switch_margin = 0.08
        self.frontier_match_distance = 6.0

        self.frontier_builder = SemanticFrontierBuilder(self)
        self.viewpoint_planner = ViewpointPlanner(self)
        self.sgcp_planner = SGCPPlanner(self)


    def reset(self, origin):
        self.origin = np.array(origin[:3], dtype=float)

        self.semantic_value.fill(0.0)
        self.confidence.fill(0.0)
        self.safety_value.fill(0.0)
        self.novelty_value.fill(0.0)

        self.observe_count.fill(0)
        self.last_update_step.fill(-1)
        self.visited.fill(False)

        self.last_region_update = {}
        self.last_frontiers = []

        self.last_selected_frontier = None
        self.frontier_selection_count = {}


    def same_origin(self, origin, threshold=1e-3):
        origin = np.array(origin[:3], dtype=float)
        return np.linalg.norm(self.origin[:2] - origin[:2]) < threshold


    def update(self, semantic_result, current_pose, depth_info=None, step_num=0):
        self.mark_visited(current_pose)

        region_scores = semantic_result.get("region_scores", {})
        safety_scores = semantic_result.get("safety_scores", {})
        novelty_scores = semantic_result.get("novelty_scores", {})

        for region in ["front", "left", "right"]:
            region_score = self._get_score(region_scores, region, 0.0)
            safety_score = self._get_score(safety_scores, region, 0.5)
            novelty_score = self._get_score(novelty_scores, region, 0.5)

            region_depth = None
            if depth_info is not None:
                region_depth = self.get_region_depth(depth_info, region)

            cells = self.get_region_cells(
                current_pose=current_pose,
                region=region,
                depth_grid=region_depth
            )

            self.update_region(
                cells=cells,
                region=region,
                region_score=region_score,
                safety_score=safety_score,
                novelty_score=novelty_score,
                step_num=step_num
            )

        return self.get_summary()


    def update_region(self, cells, region, region_score, safety_score, novelty_score, step_num):
        if len(cells) == 0:
            self.last_region_update[region] = {
                "num_cells": 0,
                "score": region_score,
                "safety": safety_score,
                "novelty": novelty_score
            }
            return

        for gx, gy, distance_weight in cells:
            new_conf = self.compute_update_confidence(
                distance_weight=distance_weight,
                safety_score=safety_score,
                novelty_score=novelty_score
            )

            old_conf = float(self.confidence[gx, gy])
            old_value = float(self.semantic_value[gx, gy])
            old_safety = float(self.safety_value[gx, gy])
            old_novelty = float(self.novelty_value[gx, gy])

            denom = old_conf + new_conf
            if denom <= 1e-6:
                continue

            self.semantic_value[gx, gy] = (
                old_conf * old_value + new_conf * region_score
            ) / denom

            self.safety_value[gx, gy] = (
                old_conf * old_safety + new_conf * safety_score
            ) / denom

            self.novelty_value[gx, gy] = (
                old_conf * old_novelty + new_conf * novelty_score
            ) / denom

            self.confidence[gx, gy] = (
                old_conf * old_conf + new_conf * new_conf
            ) / denom

            self.observe_count[gx, gy] += 1
            self.last_update_step[gx, gy] = step_num

        self.last_region_update[region] = {
            "num_cells": len(cells),
            "score": region_score,
            "safety": safety_score,
            "novelty": novelty_score
        }


    def compute_update_confidence(self, distance_weight, safety_score, novelty_score):
        base_conf = 0.4 * safety_score + 0.3 * novelty_score + 0.3
        update_conf = base_conf * distance_weight
        update_conf = max(0.05, min(1.0, update_conf))
        return update_conf


    def mark_visited(self, current_pose):
        x, y, z, yaw = current_pose
        center = self.world_to_grid(x, y)

        if center is None:
            return

        radius_cell = int(math.ceil(self.visited_radius / self.resolution))
        cx, cy = center

        for gx in range(cx - radius_cell, cx + radius_cell + 1):
            for gy in range(cy - radius_cell, cy + radius_cell + 1):
                if not self.in_bounds(gx, gy):
                    continue

                wx, wy = self.grid_to_world(gx, gy)
                dist = math.hypot(wx - x, wy - y)

                if dist <= self.visited_radius:
                    self.visited[gx, gy] = True


    def get_region_cells(self, current_pose, region, depth_grid=None):
        x, y, z, yaw = current_pose

        region_yaw = self.get_region_yaw(yaw, region)
        visible_range = self.estimate_visible_range(depth_grid)

        center = self.world_to_grid(x, y)
        if center is None:
            return []

        radius_cell = int(math.ceil(visible_range / self.resolution))
        cx, cy = center
        cells = []

        for gx in range(cx - radius_cell, cx + radius_cell + 1):
            for gy in range(cy - radius_cell, cy + radius_cell + 1):
                if not self.in_bounds(gx, gy):
                    continue

                wx, wy = self.grid_to_world(gx, gy)
                dx = wx - x
                dy = wy - y
                dist = math.hypot(dx, dy)

                if dist < self.resolution * 0.5:
                    continue

                if dist > visible_range:
                    continue

                cell_yaw = math.degrees(math.atan2(dy, dx))
                yaw_diff = abs(self.normalize_angle(cell_yaw - region_yaw))

                if yaw_diff <= self.sector_angle / 2.0:
                    distance_weight = max(0.0, 1.0 - dist / max(visible_range, 1e-6))
                    cells.append((gx, gy, distance_weight))

        return cells


    def get_region_yaw(self, yaw, region):
        if region == "front":
            return yaw
        elif region == "left":
            return yaw - 90.0
        elif region == "right":
            return yaw + 90.0
        else:
            return yaw


    def estimate_visible_range(self, depth_grid):
        if depth_grid is None:
            return self.max_sensing_range * 0.7

        try:
            depth_array = np.array(depth_grid, dtype=np.float32)
            depth_array = depth_array[np.isfinite(depth_array)]

            if depth_array.size == 0:
                return self.max_sensing_range * 0.7

            depth_array = depth_array[depth_array > 0]
            if depth_array.size == 0:
                return self.max_sensing_range * 0.7

            mean_depth = float(np.mean(depth_array))
            low_depth = float(np.percentile(depth_array, 30))
            visible_range = 0.5 * mean_depth + 0.5 * low_depth

            visible_range = max(4.0, min(self.max_sensing_range, visible_range))
            return visible_range

        except Exception:
            return self.max_sensing_range * 0.7


    def get_region_depth(self, depth_info, region):
        if depth_info is None or len(depth_info) < 3:
            return None

        if region == "front":
            return depth_info[0]
        elif region == "left":
            return depth_info[1]
        elif region == "right":
            return depth_info[2]
        else:
            return None


    def get_best_memory_target(
        self,
        current_pose,
        min_confidence=0.05,
        min_distance=3.0,
        max_distance=None,
    ):
        if max_distance is None:
            max_distance = self.map_size / 2.0

        score_map = self.compute_planning_score_map()

        frontiers = self.frontier_builder.get_semantic_frontiers(
            current_pose=current_pose,
            score_map=score_map,
            min_confidence=min_confidence,
            min_distance=min_distance,
            max_distance=max_distance
        )

        frontiers = self.viewpoint_planner.generate_viewpoints_for_frontiers(
            frontiers=frontiers,
            score_map=score_map,
            current_pose=current_pose
        )

        self.last_frontiers = frontiers

        if len(frontiers) > 0:
            target = self.sgcp_planner.select_next_frontier(
                frontiers=frontiers,
                current_pose=current_pose
            )

            if target is not None:
                target = self.apply_frontier_hysteresis(
                    selected_target=target,
                    frontiers=frontiers
                )
                target = self.add_relative_info_to_target(target, current_pose)
                self.update_frontier_selection_history(target)
                return target

        return self.get_best_memory_cell_target(
            current_pose=current_pose,
            min_confidence=min_confidence,
            min_distance=min_distance,
            max_distance=max_distance
        )


    def apply_frontier_hysteresis(self, selected_target, frontiers):
        if selected_target is None or not selected_target.get("valid", False):
            return selected_target

        if self.last_selected_frontier is None:
            selected_target["hysteresis_kept"] = False
            selected_target["hysteresis_reason"] = "no previous frontier"
            return selected_target

        matched_frontier = self.find_matching_frontier(
            frontiers=frontiers,
            last_frontier=self.last_selected_frontier
        )

        if matched_frontier is None:
            selected_target["hysteresis_kept"] = False
            selected_target["hysteresis_reason"] = "previous frontier disappeared"
            return selected_target

        selected_score = float(selected_target.get("score", 0.0))
        matched_score = float(matched_frontier.get("score", 0.0))

        if selected_score <= matched_score + self.frontier_switch_margin:
            target = dict(matched_frontier)
            target["target_type"] = selected_target.get("target_type", "sgcp_frontier")
            target["sgcp_score"] = selected_target.get("sgcp_score", selected_score)
            target["sgcp_candidate_count"] = selected_target.get("sgcp_candidate_count", 0)
            target["sgcp_total_frontier_count"] = selected_target.get("sgcp_total_frontier_count", len(frontiers))
            target["sgcp_constraint_count"] = selected_target.get("sgcp_constraint_count", 0)
            target["semantic_gap_rho"] = selected_target.get("semantic_gap_rho", 0.0)
            target["hysteresis_kept"] = True
            target["hysteresis_reason"] = "keep previous frontier"
            return target

        selected_target["hysteresis_kept"] = False
        selected_target["hysteresis_reason"] = "new frontier is significantly better"
        return selected_target


    def find_matching_frontier(self, frontiers, last_frontier):
        if last_frontier is None:
            return None

        last_pos = last_frontier.get("frontier_position", None)
        if last_pos is None:
            last_pos = last_frontier.get("position", None)

        if last_pos is None:
            return None

        best_frontier = None
        best_dist = None

        for frontier in frontiers:
            pos = frontier.get("frontier_position", None)
            if pos is None:
                pos = frontier.get("position", None)

            if pos is None:
                continue

            dist = math.hypot(pos[0] - last_pos[0], pos[1] - last_pos[1])

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_frontier = frontier

        if best_frontier is None:
            return None

        if best_dist is not None and best_dist <= self.frontier_match_distance:
            return best_frontier

        return None


    def update_frontier_selection_history(self, target):
        if target is None or not target.get("valid", False):
            return

        self.last_selected_frontier = dict(target)

        key = self.get_frontier_key(
            target.get("frontier_position", target.get("position", None))
        )

        if key is None:
            return

        if key not in self.frontier_selection_count:
            self.frontier_selection_count[key] = 0

        self.frontier_selection_count[key] += 1


    def get_frontier_selection_count(self, position):
        key = self.get_frontier_key(position)

        if key is None:
            return 0

        return int(self.frontier_selection_count.get(key, 0))


    def get_frontier_key(self, position):
        if position is None:
            return None

        x, y = position
        cell_size = max(self.resolution * 2.0, 1e-6)

        key_x = int(round(float(x) / cell_size))
        key_y = int(round(float(y) / cell_size))

        return f"{key_x}_{key_y}"


    def get_best_memory_cell_target(
        self,
        current_pose,
        min_confidence=0.05,
        min_distance=3.0,
        max_distance=None,
    ):
        if max_distance is None:
            max_distance = self.map_size / 2.0

        observed_mask = self.confidence >= min_confidence

        if not np.any(observed_mask):
            return self.default_memory_target()

        score_map = self.compute_planning_score_map()
        score_map = np.where(observed_mask, score_map, -1.0)

        self.suppress_nearby_cells(
            score_map=score_map,
            current_pose=current_pose,
            min_distance=min_distance,
            max_distance=max_distance
        )

        if np.max(score_map) <= 0.0:
            return self.default_memory_target()

        max_index = np.unravel_index(np.argmax(score_map), score_map.shape)
        gx, gy = int(max_index[0]), int(max_index[1])
        wx, wy = self.grid_to_world(gx, gy)

        target = {
            "valid": True,
            "target_type": "cell",
            "grid": (gx, gy),
            "position": (round(wx, 2), round(wy, 2)),
            "frontier_position": (round(wx, 2), round(wy, 2)),
            "viewpoint_position": (round(wx, 2), round(wy, 2)),
            "viewpoint_yaw": 0.0,
            "score": float(score_map[gx, gy]),
            "semantic_value": float(self.semantic_value[gx, gy]),
            "confidence": float(self.confidence[gx, gy]),
            "safety_value": float(self.safety_value[gx, gy]),
            "novelty_value": float(self.novelty_value[gx, gy]),
            "unknown_gain": 0.0,
            "boundary_ratio": 0.0,
            "history_penalty": 0.0,
            "hysteresis_kept": False,
            "visited": bool(self.visited[gx, gy]),
            "observe_count": int(self.observe_count[gx, gy]),
            "cluster_size": 1,
        }

        target = self.add_relative_info_to_target(target, current_pose)
        return target


    def select_best_frontier(self, frontiers):
        if len(frontiers) == 0:
            return self.default_memory_target()

        best_frontier = max(frontiers, key=lambda item: item.get("score", 0.0))
        return best_frontier


    def add_relative_info_to_target(self, target, current_pose):
        if not target.get("valid", False):
            return target

        x, y, z, yaw = current_pose
        pos = target.get("position", None)

        if pos is None:
            return target

        wx, wy = pos
        dx = wx - x
        dy = wy - y

        distance = math.hypot(dx, dy)
        target_yaw = math.degrees(math.atan2(dy, dx))
        relative_angle = self.normalize_angle(target_yaw - yaw)
        relative_region = self.get_relative_region(relative_angle)

        target["distance"] = round(distance, 2)
        target["target_yaw"] = round(target_yaw, 2)
        target["relative_angle"] = round(relative_angle, 2)
        target["relative_region"] = relative_region

        return target


    def compute_planning_score_map(self):
        semantic_score = np.clip(self.semantic_value, 0.0, 1.0)
        confidence_score = np.clip(self.confidence, 0.0, 1.0)
        safety_score = np.clip(self.safety_value, 0.0, 1.0)
        novelty_score = np.clip(self.novelty_value, 0.0, 1.0)

        # 语义是主项；安全、新颖性、置信度作为调制项，避免单帧低置信噪声主导规划。
        score_map = semantic_score
        score_map = score_map * (0.4 + 0.6 * confidence_score)
        score_map = score_map * (0.5 + 0.5 * safety_score)
        score_map = score_map * (0.5 + 0.5 * novelty_score)

        visited_penalty = np.where(self.visited, 0.55, 1.0)
        score_map = score_map * visited_penalty

        return score_map


    def suppress_nearby_cells(self, score_map, current_pose, min_distance, max_distance):
        x, y, z, yaw = current_pose

        for gx in range(self.grid_size):
            for gy in range(self.grid_size):
                if score_map[gx, gy] < 0.0:
                    continue

                wx, wy = self.grid_to_world(gx, gy)
                dist = math.hypot(wx - x, wy - y)

                if dist < min_distance or dist > max_distance:
                    score_map[gx, gy] = -1.0


    def get_relative_region(self, relative_angle):
        if abs(relative_angle) <= 35.0:
            return "front"

        if relative_angle > 35.0 and relative_angle <= 135.0:
            return "right"

        if relative_angle < -35.0 and relative_angle >= -135.0:
            return "left"

        if relative_angle > 135.0:
            return "back_right"

        return "back_left"


    def default_memory_target(self):
        target = {
            "valid": False,
            "target_type": "none",
            "grid": None,
            "position": None,
            "frontier_position": None,
            "viewpoint_position": None,
            "viewpoint_yaw": 0.0,
            "score": 0.0,
            "semantic_value": 0.0,
            "confidence": 0.0,
            "safety_value": 0.0,
            "novelty_value": 0.0,
            "unknown_gain": 0.0,
            "boundary_ratio": 0.0,
            "history_penalty": 0.0,
            "hysteresis_kept": False,
            "visited": False,
            "observe_count": 0,
            "cluster_size": 0,
            "distance": 0.0,
            "target_yaw": 0.0,
            "relative_angle": 0.0,
            "relative_region": "front"
        }

        return target


    def get_neighbors(self, gx, gy, eight_connected=True):
        if eight_connected:
            offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
        else:
            offsets = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1)
            ]

        neighbors = []
        for dx, dy in offsets:
            nx = gx + dx
            ny = gy + dy

            if self.in_bounds(nx, ny):
                neighbors.append((nx, ny))

        return neighbors


    def world_to_grid(self, x, y):
        gx = int(round((x - self.origin[0]) / self.resolution + self.center_idx))
        gy = int(round((y - self.origin[1]) / self.resolution + self.center_idx))

        if not self.in_bounds(gx, gy):
            return None

        return gx, gy


    def grid_to_world(self, gx, gy):
        x = (gx - self.center_idx) * self.resolution + self.origin[0]
        y = (gy - self.center_idx) * self.resolution + self.origin[1]
        return x, y


    def in_bounds(self, gx, gy):
        return 0 <= gx < self.grid_size and 0 <= gy < self.grid_size


    def get_summary(self):
        observed_mask = self.confidence > 0.0
        observed_cells = int(np.sum(observed_mask))
        visited_cells = int(np.sum(self.visited))

        if observed_cells == 0:
            return {
                "observed_cells": observed_cells,
                "visited_cells": visited_cells,
                "max_value": 0.0,
                "max_confidence": 0.0,
                "max_position": None,
                "mean_value": 0.0,
                "frontier_count": 0,
                "last_region_update": self.last_region_update
            }

        value_map = self.semantic_value * np.maximum(self.confidence, 1e-6)
        value_map = np.where(observed_mask, value_map, -1.0)

        max_index = np.unravel_index(np.argmax(value_map), value_map.shape)
        max_x, max_y = self.grid_to_world(max_index[0], max_index[1])

        observed_values = self.semantic_value[observed_mask]

        return {
            "observed_cells": observed_cells,
            "visited_cells": visited_cells,
            "max_value": float(self.semantic_value[max_index]),
            "max_confidence": float(self.confidence[max_index]),
            "max_position": (round(max_x, 2), round(max_y, 2)),
            "mean_value": float(np.mean(observed_values)),
            "frontier_count": len(self.last_frontiers),
            "last_region_update": self.last_region_update
        }


    def _get_score(self, score_dict, key, default):
        try:
            if not isinstance(score_dict, dict):
                return default

            value = float(score_dict.get(key, default))
            value = max(0.0, min(1.0, value))
            return value

        except Exception:
            return default


    def normalize_angle(self, angle):
        while angle > 180.0:
            angle -= 360.0

        while angle < -180.0:
            angle += 360.0

        return angle