import math
import heapq
import numpy as np


class OccupancyGeometryPlanner:
    def __init__(
        self,
        memory,
        unknown_cost=6.0,
        safety_weight=3.0,
        semantic_weight=0.3,
        visited_weight=0.2,
        blocked_safety_threshold=0.18,
        blocked_confidence_threshold=0.05,
        free_confidence_threshold=0.03,
        free_safety_threshold=0.28,
        obstacle_confidence_threshold=0.45,
        obstacle_inflation_radius=3.0,
        risk_inflation_radius=5.0,
        risk_safety_threshold=0.45,
        risk_weight=4.0,
        block_unknown=True,
        min_movement_distance=1.0,
        min_goal_distance=1.5,
        max_path_points=30,
    ):
        self.memory = memory
        self.unknown_cost = unknown_cost
        self.safety_weight = safety_weight
        self.semantic_weight = semantic_weight
        self.visited_weight = visited_weight
        self.blocked_safety_threshold = blocked_safety_threshold
        self.blocked_confidence_threshold = blocked_confidence_threshold
        self.free_confidence_threshold = free_confidence_threshold
        self.free_safety_threshold = free_safety_threshold
        self.obstacle_confidence_threshold = obstacle_confidence_threshold
        self.obstacle_inflation_radius = obstacle_inflation_radius
        self.risk_inflation_radius = risk_inflation_radius
        self.risk_safety_threshold = risk_safety_threshold
        self.risk_weight = risk_weight
        self.block_unknown = block_unknown
        self.min_movement_distance = min_movement_distance
        self.min_goal_distance = min_goal_distance
        self.max_path_points = max_path_points

    def evaluate_path(
        self,
        current_pose,
        target_position,
        z=None,
        geometry=None,
        start_grid=None,
        allow_goal_adjustment=True,
        allow_arrived=False,
    ):
        if target_position is None:
            return self.default_result(reason="target position is None")

        if start_grid is None:
            start_grid = self.memory.world_to_grid(current_pose[0], current_pose[1])
        if start_grid is None:
            return self.default_result(reason="start is outside memory map")

        goal_grid = self.memory.world_to_grid(target_position[0], target_position[1])
        if goal_grid is None:
            return self.default_result(
                reason="goal is outside memory map",
                start_grid=start_grid,
                target_position=target_position
            )

        if geometry is None:
            geometry = self.build_geometry_maps()

        self.release_start_area(
            start=start_grid,
            blocked_map=geometry["blocked_map"],
            free_map=geometry["free_map"]
        )

        original_goal_grid = goal_grid
        straight_distance = self.compute_world_distance(
            (current_pose[0], current_pose[1]),
            target_position
        )

        if start_grid == goal_grid:
            if allow_arrived or straight_distance <= self.min_goal_distance:
                return self.build_arrived_result(
                    current_pose=current_pose,
                    target_position=target_position,
                    start_grid=start_grid,
                    goal_grid=goal_grid,
                    original_goal_grid=original_goal_grid,
                    geometry=geometry,
                    reason="already at target cell"
                )
            return self.default_result(
                reason="goal is current cell but target is not reached",
                start_grid=start_grid,
                goal_grid=goal_grid,
                original_goal_grid=original_goal_grid,
                target_position=target_position,
                geometry=geometry,
                straight_distance=straight_distance
            )

        goal_adjusted = False
        if (
            geometry["blocked_map"][goal_grid[0], goal_grid[1]]
            or not geometry["free_map"][goal_grid[0], goal_grid[1]]
        ):
            if not allow_goal_adjustment:
                return self.default_result(
                    reason="goal is not reachable in occupancy map",
                    start_grid=start_grid,
                    goal_grid=goal_grid,
                    original_goal_grid=original_goal_grid,
                    target_position=target_position,
                    geometry=geometry,
                    straight_distance=straight_distance
                )

            adjusted_goal = self.find_nearest_free_cell(
                grid=goal_grid,
                start=start_grid,
                blocked_map=geometry["blocked_map"],
                free_map=geometry["free_map"],
                cost_map=geometry["cost_map"]
            )
            if adjusted_goal is None:
                return self.default_result(
                    reason="goal is blocked and no nearby free cell found",
                    start_grid=start_grid,
                    goal_grid=goal_grid,
                    original_goal_grid=original_goal_grid,
                    target_position=target_position,
                    geometry=geometry,
                    straight_distance=straight_distance
                )

            goal_grid = adjusted_goal
            goal_adjusted = True

            if start_grid == goal_grid:
                if allow_arrived:
                    return self.build_arrived_result(
                        current_pose=current_pose,
                        target_position=target_position,
                        start_grid=start_grid,
                        goal_grid=goal_grid,
                        original_goal_grid=original_goal_grid,
                        geometry=geometry,
                        reason="adjusted goal is current cell"
                    )
                return self.default_result(
                    reason="adjusted goal is current cell",
                    start_grid=start_grid,
                    goal_grid=goal_grid,
                    original_goal_grid=original_goal_grid,
                    target_position=target_position,
                    geometry=geometry,
                    straight_distance=straight_distance
                )

        raw_grid_path = self.astar_search(
            start=start_grid,
            goal=goal_grid,
            cost_map=geometry["cost_map"],
            blocked_map=geometry["blocked_map"]
        )
        if len(raw_grid_path) == 0:
            return self.default_result(
                reason="astar failed to find path on occupancy map",
                start_grid=start_grid,
                goal_grid=goal_grid,
                original_goal_grid=original_goal_grid,
                target_position=target_position,
                geometry=geometry,
                straight_distance=straight_distance
            )

        refined_grid_path = self.refine_path(
            grid_path=raw_grid_path,
            blocked_map=geometry["blocked_map"],
            free_map=geometry["free_map"]
        )
        refined_grid_path = self.downsample_path(
            grid_path=refined_grid_path,
            max_points=self.max_path_points
        )

        if z is None:
            z = current_pose[2]

        world_path = self.grid_path_to_world_path(
            grid_path=refined_grid_path,
            z=z,
            current_pose=current_pose
        )
        path_length = self.compute_world_path_length(world_path)

        if len(world_path) <= 1 or path_length < self.min_movement_distance:
            if allow_arrived or straight_distance <= self.min_goal_distance:
                return self.build_arrived_result(
                    current_pose=current_pose,
                    target_position=target_position,
                    start_grid=start_grid,
                    goal_grid=goal_grid,
                    original_goal_grid=original_goal_grid,
                    geometry=geometry,
                    reason="path reached target cell"
                )
            return self.default_result(
                reason="path is too short for movement",
                start_grid=start_grid,
                goal_grid=goal_grid,
                original_goal_grid=original_goal_grid,
                target_position=target_position,
                geometry=geometry,
                straight_distance=straight_distance,
                raw_grid_path=raw_grid_path,
                grid_path=refined_grid_path,
                world_path=world_path,
                path_length=path_length
            )

        path_to_straight_ratio = self.compute_path_ratio(
            path_length=path_length,
            straight_distance=straight_distance
        )
        geometry_cost = self.compute_geometry_cost(path_length)
        geometry_score = 1.0 - geometry_cost

        return {
            "reachable": True,
            "arrived": False,
            "reason": "ok",
            "start_grid": start_grid,
            "goal_grid": goal_grid,
            "original_goal_grid": original_goal_grid,
            "goal_adjusted": goal_adjusted,
            "target_position": target_position,
            "path": world_path,
            "grid_path": refined_grid_path,
            "raw_grid_path": raw_grid_path,
            "raw_grid_path_len": len(raw_grid_path),
            "path_len": len(world_path),
            "path_length": round(float(path_length), 2),
            "straight_distance": round(float(straight_distance), 2),
            "path_to_straight_ratio": round(float(path_to_straight_ratio), 3),
            "geometry_cost": float(geometry_cost),
            "geometry_score": float(geometry_score),
            "blocked_cells": geometry.get("blocked_cells", 0),
            "obstacle_cells": geometry.get("obstacle_cells", 0),
            "unknown_blocked_cells": geometry.get("unknown_blocked_cells", 0),
            "inflation_cells": geometry.get("inflation_cells", 0),
        }

    def build_geometry_maps(self):
        confidence = np.clip(self.memory.confidence, 0.0, 1.0)
        safety = np.clip(self.memory.safety_value, 0.0, 1.0)
        semantic = np.clip(self.memory.semantic_value, 0.0, 1.0)

        free_confidence = self.get_memory_map("free_confidence")
        obstacle_confidence = self.get_memory_map("obstacle_confidence")

        semantic_free = (
            (confidence >= self.free_confidence_threshold)
            & (safety >= self.free_safety_threshold)
        )
        geometric_free = free_confidence >= self.free_confidence_threshold
        free_map = semantic_free | geometric_free | self.memory.visited

        semantic_obstacle = (
            (safety <= self.blocked_safety_threshold)
            & (confidence >= self.blocked_confidence_threshold)
        )
        depth_obstacle = obstacle_confidence >= self.obstacle_confidence_threshold
        obstacle_seed = semantic_obstacle | depth_obstacle

        obstacle_inflation_cells = self.meters_to_cells(self.obstacle_inflation_radius)
        risk_inflation_cells = self.meters_to_cells(self.risk_inflation_radius)

        inflated_obstacle = self.inflate_mask(
            mask=obstacle_seed,
            radius_cells=obstacle_inflation_cells
        )
        risk_seed = (
            (
                (safety <= self.risk_safety_threshold)
                & (confidence >= self.blocked_confidence_threshold)
            )
            | (
                obstacle_confidence
                >= max(0.25, self.obstacle_confidence_threshold * 0.7)
            )
        )
        inflated_risk = self.inflate_mask(
            mask=risk_seed,
            radius_cells=risk_inflation_cells
        )

        blocked_map = inflated_obstacle.copy()
        unknown_blocked = np.zeros_like(blocked_map, dtype=np.bool_)
        if self.block_unknown:
            unknown_blocked = ~free_map
            blocked_map = blocked_map | unknown_blocked

        observed_mask = confidence > 0.0
        cost_map = np.ones(
            (self.memory.grid_size, self.memory.grid_size),
            dtype=np.float32
        )
        if self.block_unknown:
            cost_map = np.where(free_map, cost_map, self.unknown_cost)
        else:
            cost_map = np.where(observed_mask | free_map, cost_map, self.unknown_cost)

        safety_cost = self.safety_weight * (1.0 - safety) * np.maximum(confidence, 0.2)
        semantic_bonus = self.semantic_weight * semantic
        visited_cost = np.where(self.memory.visited, self.visited_weight, 0.0)
        risk_cost = np.where(inflated_risk, self.risk_weight, 0.0)
        obstacle_cost = self.risk_weight * np.clip(obstacle_confidence, 0.0, 1.0)

        cost_map = (
            cost_map
            + safety_cost
            + visited_cost
            + risk_cost
            + obstacle_cost
            - semantic_bonus
        )
        cost_map = np.maximum(cost_map, 0.1)
        cost_map = np.where(blocked_map, np.inf, cost_map)

        return {
            "cost_map": cost_map,
            "blocked_map": blocked_map,
            "free_map": free_map,
            "obstacle_seed": obstacle_seed,
            "risk_map": inflated_risk,
            "blocked_cells": int(np.sum(blocked_map)),
            "obstacle_cells": int(np.sum(obstacle_seed)),
            "unknown_blocked_cells": int(np.sum(unknown_blocked)),
            "inflation_cells": int(obstacle_inflation_cells),
        }

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
            for neighbor in self.get_neighbors(current):
                nx, ny = neighbor
                if blocked_map[nx, ny]:
                    continue
                if self.is_diagonal_blocked(current, neighbor, blocked_map):
                    continue

                move_cost = self.get_move_cost(current, neighbor)
                cell_cost = float(cost_map[nx, ny])
                if not math.isfinite(cell_cost):
                    continue

                tentative_g = g_score[current] + move_cost * cell_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))

        return []

    def refine_path(self, grid_path, blocked_map, free_map):
        if len(grid_path) <= 2:
            return grid_path

        refined_path = [grid_path[0]]
        anchor_idx = 0
        while anchor_idx < len(grid_path) - 1:
            next_idx = len(grid_path) - 1
            while next_idx > anchor_idx + 1:
                if self.has_line_of_sight(
                    start=grid_path[anchor_idx],
                    end=grid_path[next_idx],
                    blocked_map=blocked_map,
                    free_map=free_map
                ):
                    break
                next_idx -= 1
            refined_path.append(grid_path[next_idx])
            anchor_idx = next_idx

        return refined_path

    def has_line_of_sight(self, start, end, blocked_map, free_map):
        cells = self.bresenham_line(start, end)
        for gx, gy in cells:
            if not self.memory.in_bounds(gx, gy):
                return False
            if blocked_map[gx, gy]:
                return False
            if self.block_unknown and not free_map[gx, gy]:
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

    def find_nearest_free_cell(
        self,
        grid,
        start,
        blocked_map,
        free_map,
        cost_map,
        max_radius=8
    ):
        gx, gy = grid
        candidates = []

        min_start_cell_distance = max(
            1.0,
            self.min_movement_distance / max(self.memory.resolution, 1e-6)
        )

        for radius in range(1, max_radius + 1):
            for nx in range(gx - radius, gx + radius + 1):
                for ny in range(gy - radius, gy + radius + 1):
                    if not self.memory.in_bounds(nx, ny):
                        continue
                    if blocked_map[nx, ny]:
                        continue
                    if not free_map[nx, ny]:
                        continue
                    if not math.isfinite(float(cost_map[nx, ny])):
                        continue

                    dist_to_start = math.hypot(nx - start[0], ny - start[1])
                    if dist_to_start < min_start_cell_distance:
                        continue

                    dist_to_goal = math.hypot(nx - gx, ny - gy)
                    score = dist_to_goal + 0.1 * float(cost_map[nx, ny])
                    candidates.append((score, (nx, ny)))

            if len(candidates) > 0:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]

        return None

    def release_start_area(self, start, blocked_map, free_map):
        sx, sy = start
        radius = max(1, self.meters_to_cells(self.memory.resolution * 1.5))
        for gx in range(sx - radius, sx + radius + 1):
            for gy in range(sy - radius, sy + radius + 1):
                if not self.memory.in_bounds(gx, gy):
                    continue
                if math.hypot(gx - sx, gy - sy) <= radius:
                    blocked_map[gx, gy] = False
                    free_map[gx, gy] = True

    def inflate_mask(self, mask, radius_cells):
        if radius_cells <= 0:
            return mask.copy()

        inflated = mask.copy()
        obstacle_indices = np.argwhere(mask)
        for gx, gy in obstacle_indices:
            for nx in range(gx - radius_cells, gx + radius_cells + 1):
                for ny in range(gy - radius_cells, gy + radius_cells + 1):
                    if not self.memory.in_bounds(nx, ny):
                        continue
                    if math.hypot(nx - gx, ny - gy) <= radius_cells:
                        inflated[nx, ny] = True
        return inflated

    def get_memory_map(self, name):
        if hasattr(self.memory, name):
            value = getattr(self.memory, name)
            try:
                return np.clip(value, 0.0, 1.0)
            except Exception:
                pass

        return np.zeros(
            (self.memory.grid_size, self.memory.grid_size),
            dtype=np.float32
        )

    def get_neighbors(self, grid):
        gx, gy = grid
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        neighbors = []
        for dx, dy in offsets:
            nx = gx + dx
            ny = gy + dy
            if self.memory.in_bounds(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def is_diagonal_blocked(self, grid_a, grid_b, blocked_map):
        ax, ay = grid_a
        bx, by = grid_b
        if ax == bx or ay == by:
            return False
        if blocked_map[ax, by]:
            return True
        if blocked_map[bx, ay]:
            return True
        return False

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

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def downsample_path(self, grid_path, max_points):
        if len(grid_path) <= max_points:
            return grid_path

        indices = np.linspace(0, len(grid_path) - 1, max_points)
        indices = np.round(indices).astype(int).tolist()

        new_path = []
        last_idx = None
        for idx in indices:
            if idx == last_idx:
                continue
            new_path.append(grid_path[idx])
            last_idx = idx

        if new_path[-1] != grid_path[-1]:
            new_path.append(grid_path[-1])
        return new_path

    def grid_path_to_world_path(self, grid_path, z, current_pose=None):
        world_path = []
        for idx, (gx, gy) in enumerate(grid_path):
            if idx == 0 and current_pose is not None:
                world_path.append((
                    round(float(current_pose[0]), 2),
                    round(float(current_pose[1]), 2),
                    round(float(z), 2)
                ))
                continue

            wx, wy = self.memory.grid_to_world(gx, gy)
            world_path.append((round(wx, 2), round(wy, 2), round(z, 2)))
        return world_path

    def compute_world_path_length(self, world_path):
        if len(world_path) <= 1:
            return 0.0

        total_length = 0.0
        for idx in range(1, len(world_path)):
            x0, y0, z0 = world_path[idx - 1]
            x1, y1, z1 = world_path[idx]
            total_length += math.sqrt(
                (x1 - x0) ** 2
                + (y1 - y0) ** 2
                + (z1 - z0) ** 2
            )
        return total_length

    def compute_world_distance(self, start_position, target_position):
        sx, sy = start_position
        tx, ty = target_position
        return math.hypot(float(tx) - float(sx), float(ty) - float(sy))

    def compute_path_ratio(self, path_length, straight_distance):
        if straight_distance <= 1e-6:
            return 1.0
        return float(path_length) / max(float(straight_distance), 1e-6)

    def compute_geometry_cost(self, path_length):
        max_distance = max(self.memory.map_size / 2.0, 1e-6)
        return float(max(0.0, min(1.0, path_length / max_distance)))

    def meters_to_cells(self, distance):
        return int(math.ceil(float(distance) / max(float(self.memory.resolution), 1e-6)))

    def build_arrived_result(
        self,
        current_pose,
        target_position,
        start_grid,
        goal_grid,
        original_goal_grid,
        geometry,
        reason
    ):
        z = current_pose[2]
        return {
            "reachable": True,
            "arrived": True,
            "reason": reason,
            "start_grid": start_grid,
            "goal_grid": goal_grid,
            "original_goal_grid": original_goal_grid,
            "goal_adjusted": goal_grid != original_goal_grid,
            "target_position": target_position,
            "path": [(
                round(float(current_pose[0]), 2),
                round(float(current_pose[1]), 2),
                round(float(z), 2)
            )],
            "grid_path": [start_grid],
            "raw_grid_path": [start_grid],
            "raw_grid_path_len": 1,
            "path_len": 1,
            "path_length": 0.0,
            "straight_distance": 0.0,
            "path_to_straight_ratio": 1.0,
            "geometry_cost": 0.0,
            "geometry_score": 1.0,
            "blocked_cells": geometry.get("blocked_cells", 0),
            "obstacle_cells": geometry.get("obstacle_cells", 0),
            "unknown_blocked_cells": geometry.get("unknown_blocked_cells", 0),
            "inflation_cells": geometry.get("inflation_cells", 0),
        }

    def default_result(
        self,
        reason="unknown",
        start_grid=None,
        goal_grid=None,
        original_goal_grid=None,
        target_position=None,
        geometry=None,
        straight_distance=0.0,
        raw_grid_path=None,
        grid_path=None,
        world_path=None,
        path_length=0.0
    ):
        if geometry is None:
            geometry = {}
        if raw_grid_path is None:
            raw_grid_path = []
        if grid_path is None:
            grid_path = []
        if world_path is None:
            world_path = []

        path_to_straight_ratio = self.compute_path_ratio(
            path_length=path_length,
            straight_distance=straight_distance
        )
        geometry_cost = self.compute_geometry_cost(path_length)

        return {
            "reachable": False,
            "arrived": False,
            "reason": reason,
            "start_grid": start_grid,
            "goal_grid": goal_grid,
            "original_goal_grid": original_goal_grid,
            "goal_adjusted": False,
            "target_position": target_position,
            "path": world_path,
            "grid_path": grid_path,
            "raw_grid_path": raw_grid_path,
            "raw_grid_path_len": len(raw_grid_path),
            "path_len": len(world_path),
            "path_length": round(float(path_length), 2),
            "straight_distance": round(float(straight_distance), 2),
            "path_to_straight_ratio": round(float(path_to_straight_ratio), 3),
            "geometry_cost": float(geometry_cost),
            "geometry_score": 0.0,
            "blocked_cells": geometry.get("blocked_cells", 0),
            "obstacle_cells": geometry.get("obstacle_cells", 0),
            "unknown_blocked_cells": geometry.get("unknown_blocked_cells", 0),
            "inflation_cells": geometry.get("inflation_cells", 0),
        }