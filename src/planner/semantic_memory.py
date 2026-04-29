import math
import numpy as np


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