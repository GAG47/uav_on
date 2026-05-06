import math
import heapq

import numpy as np


class LocalPlanner:
    def __init__(
        self,
        memory,
        unknown_cost=2.0,
        safety_weight=3.0,
        semantic_weight=0.3,
        visited_weight=0.2,
        soft_safety_weight=2.0,
        blocked_safety_threshold=0.12,
        soft_safety_threshold=0.35,
        blocked_confidence_threshold=0.05,
        blocked_inflation_radius=1,
        boundary_margin_cells=1,
        max_path_points=30,
    ):
        self.memory = memory
        self.unknown_cost = unknown_cost
        self.safety_weight = safety_weight
        self.semantic_weight = semantic_weight
        self.visited_weight = visited_weight
        self.soft_safety_weight = soft_safety_weight
        self.blocked_safety_threshold = blocked_safety_threshold
        self.soft_safety_threshold = soft_safety_threshold
        self.blocked_confidence_threshold = blocked_confidence_threshold
        self.blocked_inflation_radius = blocked_inflation_radius
        self.boundary_margin_cells = boundary_margin_cells
        self.max_path_points = max_path_points

    def plan_path(self, current_pose, memory_target):
        if memory_target is None or not memory_target.get("valid", False):
            return self.default_plan(reason="invalid memory target")

        target_position = self.get_target_position(memory_target)
        if target_position is None:
            return self.default_plan(reason="target position is None")

        start_grid = self.memory.world_to_grid(current_pose[0], current_pose[1])
        goal_grid = self.memory.world_to_grid(target_position[0], target_position[1])

        if start_grid is None:
            return self.default_plan(reason="start is outside memory map")

        if goal_grid is None:
            return self.default_plan(reason="goal is outside memory map")

        cost_map, blocked_map, raw_blocked_map = self.build_cost_map()

        # 起点是无人机当前位置，不能因为局部安全值低就把自己封死。
        blocked_map[start_grid[0], start_grid[1]] = False

        if blocked_map[goal_grid[0], goal_grid[1]]:
            new_goal = self.find_nearest_free_cell(
                grid=goal_grid,
                blocked_map=blocked_map,
                cost_map=cost_map,
            )
            if new_goal is None:
                return self.default_plan(
                    reason="goal is blocked and no nearby free cell found"
                )
            goal_grid = new_goal

        grid_path = self.astar_search(
            start=start_grid,
            goal=goal_grid,
            cost_map=cost_map,
            blocked_map=blocked_map,
        )

        if len(grid_path) == 0:
            return self.default_plan(reason="astar failed to find path")

        if not self.validate_grid_path(grid_path, blocked_map):
            return self.default_plan(reason="raw astar path is not safe")

        smooth_grid_path = self.smooth_path(grid_path, blocked_map)

        if not self.validate_grid_path(smooth_grid_path, blocked_map):
            smooth_grid_path = grid_path

        smooth_grid_path = self.downsample_path(
            grid_path=smooth_grid_path,
            max_points=self.max_path_points,
            blocked_map=blocked_map,
        )

        if not self.validate_grid_path(smooth_grid_path, blocked_map):
            return self.default_plan(reason="downsampled path is not safe")

        world_path = self.grid_path_to_world_path(
            grid_path=smooth_grid_path,
            z=current_pose[2],
        )
        path_length = self.compute_world_path_length(world_path)
        safety_summary = self.compute_path_safety_summary(
            grid_path=smooth_grid_path,
            raw_blocked_map=raw_blocked_map,
        )

        plan = {
            "valid": True,
            "reason": "ok",
            "start_grid": start_grid,
            "goal_grid": goal_grid,
            "target_position": target_position,
            "path": world_path,
            "grid_path": smooth_grid_path,
            "raw_grid_path_len": len(grid_path),
            "path_len": len(world_path),
            "path_length": round(path_length, 2),
            "safety_valid": True,
            "min_safety_value": round(safety_summary["min_safety_value"], 4),
            "min_confidence_value": round(safety_summary["min_confidence_value"], 4),
            "raw_blocked_cells_on_path": safety_summary["raw_blocked_cells_on_path"],
            "blocked_inflation_radius": self.blocked_inflation_radius,
            "boundary_margin_cells": self.boundary_margin_cells,
        }

        return plan

    def get_target_position(self, memory_target):
        if memory_target.get("approach_viewpoint", None) is not None:
            return memory_target["approach_viewpoint"]

        if memory_target.get("viewpoint_position", None) is not None:
            return memory_target["viewpoint_position"]

        if memory_target.get("position", None) is not None:
            return memory_target["position"]

        if memory_target.get("frontier_position", None) is not None:
            return memory_target["frontier_position"]

        return None

    def build_cost_map(self):
        confidence = np.clip(self.memory.confidence, 0.0, 1.0)
        safety = np.clip(self.memory.safety_value, 0.0, 1.0)
        semantic = np.clip(self.memory.semantic_value, 0.0, 1.0)

        observed_mask = confidence > 0.0

        cost_map = np.ones(
            (self.memory.grid_size, self.memory.grid_size),
            dtype=np.float32,
        )
        cost_map = np.where(observed_mask, cost_map, self.unknown_cost)

        safety_cost = self.safety_weight * (1.0 - safety)
        semantic_bonus = self.semantic_weight * semantic
        visited_cost = np.where(self.memory.visited, self.visited_weight, 0.0)

        soft_risk = np.zeros_like(cost_map, dtype=np.float32)
        risky_mask = (
            (confidence >= self.blocked_confidence_threshold)
            & (safety < self.soft_safety_threshold)
        )
        soft_risk[risky_mask] = (
            self.soft_safety_weight
            * (self.soft_safety_threshold - safety[risky_mask])
            / max(self.soft_safety_threshold, 1e-6)
        )

        cost_map = cost_map + safety_cost + soft_risk + visited_cost - semantic_bonus
        cost_map = np.maximum(cost_map, 0.1)

        raw_blocked_map = (
            (safety <= self.blocked_safety_threshold)
            & (confidence >= self.blocked_confidence_threshold)
        )

        blocked_map = self.inflate_blocked_map(
            blocked_map=raw_blocked_map,
            radius=self.blocked_inflation_radius,
        )
        blocked_map = self.apply_boundary_margin(blocked_map)

        return cost_map, blocked_map, raw_blocked_map

    def inflate_blocked_map(self, blocked_map, radius):
        if radius <= 0:
            return blocked_map.copy()

        inflated = blocked_map.copy()
        blocked_indices = np.argwhere(blocked_map)

        for gx, gy in blocked_indices:
            for nx in range(gx - radius, gx + radius + 1):
                for ny in range(gy - radius, gy + radius + 1):
                    if not self.memory.in_bounds(nx, ny):
                        continue

                    dist = math.hypot(nx - gx, ny - gy)
                    if dist <= radius:
                        inflated[nx, ny] = True

        return inflated

    def apply_boundary_margin(self, blocked_map):
        margin = int(self.boundary_margin_cells)
        if margin <= 0:
            return blocked_map

        blocked = blocked_map.copy()
        grid_size = self.memory.grid_size

        blocked[:margin, :] = True
        blocked[grid_size - margin:, :] = True
        blocked[:, :margin] = True
        blocked[:, grid_size - margin:] = True

        return blocked

    def astar_search(self, start, goal, cost_map, blocked_map):
        open_heap = []
        heapq.heappush(open_heap, (0.0, start))

        came_from = {}
        g_score = {
            start: 0.0
        }
        closed_set = set()

        while len(open_heap) > 0:
            current_f, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue

            if current == goal:
                return self.reconstruct_path(came_from, current)

            closed_set.add(current)

            for neighbor in self.get_neighbors(current, blocked_map):
                nx, ny = neighbor

                if blocked_map[nx, ny]:
                    continue

                move_cost = self.get_move_cost(current, neighbor)
                tentative_g = g_score[current] + move_cost * float(cost_map[nx, ny])

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))

        return []

    def reconstruct_path(self, came_from, current):
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def get_neighbors(self, grid, blocked_map=None):
        gx, gy = grid
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        neighbors = []

        for dx, dy in offsets:
            nx = gx + dx
            ny = gy + dy

            if not self.memory.in_bounds(nx, ny):
                continue

            if blocked_map is not None and dx != 0 and dy != 0:
                side_a = (gx + dx, gy)
                side_b = (gx, gy + dy)

                if not self.memory.in_bounds(side_a[0], side_a[1]):
                    continue

                if not self.memory.in_bounds(side_b[0], side_b[1]):
                    continue

                # 禁止斜向从两个障碍格之间“切角”穿过去。
                if blocked_map[side_a[0], side_a[1]] or blocked_map[side_b[0], side_b[1]]:
                    continue

            neighbors.append((nx, ny))

        return neighbors

    def get_move_cost(self, grid_a, grid_b):
        ax, ay = grid_a
        bx, by = grid_b

        if ax != bx and ay != by:
            return math.sqrt(2.0)

        return 1.0

    def heuristic(self, grid, goal):
        gx, gy = grid
        tx, ty = goal
        return math.hypot(tx - gx, ty - gy)

    def smooth_path(self, grid_path, blocked_map):
        if len(grid_path) <= 2:
            return grid_path

        smooth_path = [grid_path[0]]
        anchor_idx = 0

        while anchor_idx < len(grid_path) - 1:
            next_idx = len(grid_path) - 1

            while next_idx > anchor_idx + 1:
                if self.has_line_of_sight(
                    grid_path[anchor_idx],
                    grid_path[next_idx],
                    blocked_map,
                ):
                    break

                next_idx -= 1

            smooth_path.append(grid_path[next_idx])
            anchor_idx = next_idx

        return smooth_path

    def has_line_of_sight(self, start, end, blocked_map):
        cells = self.bresenham_line(start, end)

        for gx, gy in cells:
            if not self.memory.in_bounds(gx, gy):
                return False

            if blocked_map[gx, gy]:
                return False

        return True

    def validate_grid_path(self, grid_path, blocked_map):
        if len(grid_path) == 0:
            return False

        for gx, gy in grid_path:
            if not self.memory.in_bounds(gx, gy):
                return False

            if blocked_map[gx, gy]:
                return False

        for idx in range(1, len(grid_path)):
            if not self.has_line_of_sight(
                grid_path[idx - 1],
                grid_path[idx],
                blocked_map,
            ):
                return False

        return True

    def bresenham_line(self, start, end):
        x0, y0 = start
        x1, y1 = end

        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x = x0
        y = y0

        while True:
            cells.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return cells

    def downsample_path(self, grid_path, max_points, blocked_map):
        if len(grid_path) <= max_points:
            return grid_path

        if max_points <= 2:
            return [grid_path[0], grid_path[-1]]

        new_path = [grid_path[0]]
        current_idx = 0

        while current_idx < len(grid_path) - 1:
            remaining_slots = max_points - len(new_path)

            if remaining_slots <= 1:
                if self.has_line_of_sight(
                    new_path[-1],
                    grid_path[-1],
                    blocked_map,
                ):
                    new_path.append(grid_path[-1])
                else:
                    new_path.append(grid_path[current_idx + 1])
                break

            remaining_points = len(grid_path) - 1 - current_idx
            jump = max(1, int(math.ceil(float(remaining_points) / remaining_slots)))
            candidate_idx = min(len(grid_path) - 1, current_idx + jump)

            while candidate_idx > current_idx + 1:
                if self.has_line_of_sight(
                    new_path[-1],
                    grid_path[candidate_idx],
                    blocked_map,
                ):
                    break

                candidate_idx -= 1

            if candidate_idx <= current_idx:
                candidate_idx = current_idx + 1

            new_path.append(grid_path[candidate_idx])
            current_idx = candidate_idx

            if len(new_path) >= max_points and new_path[-1] != grid_path[-1]:
                break

        if new_path[-1] != grid_path[-1]:
            if len(new_path) < max_points and self.has_line_of_sight(
                new_path[-1],
                grid_path[-1],
                blocked_map,
            ):
                new_path.append(grid_path[-1])

        return new_path

    def grid_path_to_world_path(self, grid_path, z):
        world_path = []

        for gx, gy in grid_path:
            wx, wy = self.memory.grid_to_world(gx, gy)
            world_path.append((round(wx, 2), round(wy, 2), round(z, 2)))

        return world_path

    def compute_world_path_length(self, world_path):
        if len(world_path) <= 1:
            return 0.0

        total_length = 0.0

        for i in range(1, len(world_path)):
            x0, y0, z0 = world_path[i - 1]
            x1, y1, z1 = world_path[i]
            total_length += math.sqrt(
                (x1 - x0) ** 2
                + (y1 - y0) ** 2
                + (z1 - z0) ** 2
            )

        return total_length

    def compute_path_safety_summary(self, grid_path, raw_blocked_map):
        min_safety_value = 1.0
        min_confidence_value = 1.0
        raw_blocked_cells_on_path = 0

        for gx, gy in grid_path:
            min_safety_value = min(
                min_safety_value,
                float(self.memory.safety_value[gx, gy]),
            )
            min_confidence_value = min(
                min_confidence_value,
                float(self.memory.confidence[gx, gy]),
            )

            if raw_blocked_map[gx, gy]:
                raw_blocked_cells_on_path += 1

        return {
            "min_safety_value": min_safety_value,
            "min_confidence_value": min_confidence_value,
            "raw_blocked_cells_on_path": raw_blocked_cells_on_path,
        }

    def find_nearest_free_cell(self, grid, blocked_map, cost_map, max_radius=8):
        gx, gy = grid

        for radius in range(1, max_radius + 1):
            candidates = []

            for nx in range(gx - radius, gx + radius + 1):
                for ny in range(gy - radius, gy + radius + 1):
                    if not self.memory.in_bounds(nx, ny):
                        continue

                    if blocked_map[nx, ny]:
                        continue

                    dist = math.hypot(nx - gx, ny - gy)
                    cost = float(cost_map[nx, ny])
                    candidates.append((cost + 0.2 * dist, (nx, ny)))

            if len(candidates) > 0:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]

        return None

    def default_plan(self, reason="unknown"):
        plan = {
            "valid": False,
            "reason": reason,
            "start_grid": None,
            "goal_grid": None,
            "target_position": None,
            "path": [],
            "grid_path": [],
            "raw_grid_path_len": 0,
            "path_len": 0,
            "path_length": 0.0,
            "safety_valid": False,
            "min_safety_value": 0.0,
            "min_confidence_value": 0.0,
            "raw_blocked_cells_on_path": 0,
            "blocked_inflation_radius": self.blocked_inflation_radius,
            "boundary_margin_cells": self.boundary_margin_cells,
        }

        return plan
