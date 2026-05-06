import copy
import math

import numpy as np


class TargetViewpointPlanner:
    def __init__(
        self,
        memory,
        approach_radii=None,
        ray_distances=None,
        yaw_offsets=None,
        blocked_safety_threshold=0.12,
        blocked_confidence_threshold=0.05,
        preferred_target_distance=8.0,
        min_target_distance=4.0,
        max_goal_distance=35.0,
        max_nearest_free_radius=4,
    ):
        self.memory = memory
        self.approach_radii = approach_radii or [6.0, 8.0, 10.0, 12.0, 15.0]
        self.ray_distances = ray_distances or [4.0, 8.0, 12.0, 16.0, 20.0, 25.0]
        self.yaw_offsets = yaw_offsets or [0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0]
        self.blocked_safety_threshold = blocked_safety_threshold
        self.blocked_confidence_threshold = blocked_confidence_threshold
        self.preferred_target_distance = preferred_target_distance
        self.min_target_distance = min_target_distance
        self.max_goal_distance = max_goal_distance
        self.max_nearest_free_radius = max_nearest_free_radius

    def build_navigation_target(self, current_pose, planner_target):
        if not isinstance(planner_target, dict):
            return self.default_navigation_target("planner target is not a dict")

        if not planner_target.get("valid", False):
            return self.default_navigation_target("planner target is invalid")

        target_position = self.extract_target_position(planner_target)
        if target_position is None:
            return self.default_navigation_target("target position is missing")

        if current_pose is None or len(current_pose) < 3:
            return self.default_navigation_target("current pose is invalid")

        candidates = self.generate_candidates(
            current_pose=current_pose,
            target_position=target_position,
            planner_target=planner_target,
        )
        best_candidate = self.select_best_candidate(
            current_pose=current_pose,
            target_position=target_position,
            candidates=candidates,
        )

        if best_candidate is None:
            result = self.default_navigation_target("no feasible approach viewpoint")
            result["source_target"] = copy.deepcopy(planner_target)
            result["target_world_position"] = target_position
            return result

        return self.make_navigation_target(
            current_pose=current_pose,
            target_position=target_position,
            planner_target=planner_target,
            candidate=best_candidate,
        )

    def extract_target_position(self, planner_target):
        keys = [
            "target_world_position",
            "verified_target_position",
            "position",
            "viewpoint_position",
            "frontier_position",
        ]

        for key in keys:
            position = planner_target.get(key, None)
            parsed = self.parse_position(position)
            if parsed is not None:
                return parsed

        return None

    def parse_position(self, position):
        if position is None:
            return None

        if not isinstance(position, (list, tuple)):
            return None

        if len(position) < 2:
            return None

        try:
            x = float(position[0])
            y = float(position[1])
            if len(position) >= 3:
                z = float(position[2])
            else:
                z = 0.0
        except Exception:
            return None

        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
            return None

        return (x, y, z)

    def generate_candidates(self, current_pose, target_position, planner_target):
        candidates = []
        current_xy = (float(current_pose[0]), float(current_pose[1]))
        current_z = float(current_pose[2])
        target_xy = (float(target_position[0]), float(target_position[1]))

        current_to_target_yaw = math.atan2(
            target_xy[1] - current_xy[1],
            target_xy[0] - current_xy[0],
        )
        target_to_current_yaw = current_to_target_yaw + math.pi

        candidates.extend(
            self.generate_target_ring_candidates(
                target_xy=target_xy,
                target_to_current_yaw=target_to_current_yaw,
                z=current_z,
            )
        )

        candidates.extend(
            self.generate_ray_candidates(
                current_xy=current_xy,
                target_xy=target_xy,
                current_to_target_yaw=current_to_target_yaw,
                z=current_z,
            )
        )

        candidates.extend(
            self.generate_existing_goal_candidates(
                planner_target=planner_target,
                z=current_z,
            )
        )

        return candidates

    def generate_target_ring_candidates(self, target_xy, target_to_current_yaw, z):
        candidates = []

        for radius in self.approach_radii:
            for offset_deg in self.yaw_offsets:
                yaw = target_to_current_yaw + math.radians(offset_deg)
                x = target_xy[0] + radius * math.cos(yaw)
                y = target_xy[1] + radius * math.sin(yaw)
                candidates.append(
                    {
                        "position": (x, y, z),
                        "candidate_type": "target_ring_viewpoint",
                        "candidate_radius": radius,
                        "candidate_yaw_offset": offset_deg,
                    }
                )

        return candidates

    def generate_ray_candidates(self, current_xy, target_xy, current_to_target_yaw, z):
        candidates = []

        for distance in self.ray_distances:
            x = current_xy[0] + distance * math.cos(current_to_target_yaw)
            y = current_xy[1] + distance * math.sin(current_to_target_yaw)
            candidates.append(
                {
                    "position": (x, y, z),
                    "candidate_type": "target_direction_viewpoint",
                    "candidate_distance": distance,
                    "candidate_yaw_offset": 0.0,
                }
            )

        return candidates

    def generate_existing_goal_candidates(self, planner_target, z):
        candidates = []

        for key in ["viewpoint_position", "frontier_position", "position"]:
            position = self.parse_position(planner_target.get(key, None))
            if position is None:
                continue

            candidates.append(
                {
                    "position": (position[0], position[1], z),
                    "candidate_type": f"existing_{key}",
                    "candidate_distance": 0.0,
                    "candidate_yaw_offset": 0.0,
                }
            )

        return candidates

    def select_best_candidate(self, current_pose, target_position, candidates):
        if len(candidates) == 0:
            return None

        cost_map, blocked_map = self.build_map_status()
        scored_candidates = []

        for candidate in candidates:
            feasible_candidate = self.make_feasible_candidate(
                candidate=candidate,
                current_pose=current_pose,
                target_position=target_position,
                blocked_map=blocked_map,
            )
            if feasible_candidate is None:
                continue

            score = self.score_candidate(
                candidate=feasible_candidate,
                current_pose=current_pose,
                target_position=target_position,
                cost_map=cost_map,
                blocked_map=blocked_map,
            )
            feasible_candidate["score"] = score
            scored_candidates.append(feasible_candidate)

        if len(scored_candidates) == 0:
            return None

        scored_candidates.sort(key=lambda item: item["score"], reverse=True)
        return scored_candidates[0]

    def make_feasible_candidate(self, candidate, current_pose, target_position, blocked_map):
        position = candidate.get("position", None)
        parsed = self.parse_position(position)
        if parsed is None:
            return None

        grid = self.memory.world_to_grid(parsed[0], parsed[1])
        if grid is None:
            return None

        if blocked_map[grid[0], grid[1]]:
            free_grid = self.find_nearest_free_grid(grid, blocked_map)
            if free_grid is None:
                return None

            wx, wy = self.memory.grid_to_world(free_grid[0], free_grid[1])
            parsed = (wx, wy, parsed[2])
            grid = free_grid

        current_distance = self.xy_distance(
            (current_pose[0], current_pose[1]),
            (parsed[0], parsed[1]),
        )
        if current_distance > self.max_goal_distance:
            return None

        target_distance = self.xy_distance(
            (parsed[0], parsed[1]),
            (target_position[0], target_position[1]),
        )
        if target_distance < self.min_target_distance:
            return None

        feasible_candidate = copy.deepcopy(candidate)
        feasible_candidate["position"] = parsed
        feasible_candidate["grid"] = grid
        feasible_candidate["current_distance"] = current_distance
        feasible_candidate["target_distance"] = target_distance
        return feasible_candidate

    def score_candidate(self, candidate, current_pose, target_position, cost_map, blocked_map):
        grid = candidate["grid"]
        gx, gy = grid

        current_xy = (float(current_pose[0]), float(current_pose[1]))
        candidate_xy = (candidate["position"][0], candidate["position"][1])
        target_xy = (target_position[0], target_position[1])

        current_target_distance = self.xy_distance(current_xy, target_xy)
        candidate_target_distance = candidate["target_distance"]
        candidate_current_distance = candidate["current_distance"]

        progress = current_target_distance - candidate_target_distance
        progress_score = self.safe_div(progress, max(current_target_distance, 1.0))

        distance_penalty = self.safe_div(candidate_current_distance, self.max_goal_distance)
        standoff_error = abs(candidate_target_distance - self.preferred_target_distance)
        standoff_score = max(
            0.0,
            1.0 - self.safe_div(standoff_error, self.preferred_target_distance),
        )

        safety_score = float(np.clip(self.memory.safety_value[gx, gy], 0.0, 1.0))
        confidence_score = float(np.clip(self.memory.confidence[gx, gy], 0.0, 1.0))
        semantic_score = float(np.clip(self.memory.semantic_value[gx, gy], 0.0, 1.0))

        current_grid = self.memory.world_to_grid(current_xy[0], current_xy[1])
        target_grid = self.memory.world_to_grid(target_xy[0], target_xy[1])

        path_clear_score = 0.0
        if current_grid is not None:
            path_clear_score = self.line_clear_ratio(current_grid, grid, blocked_map)

        visibility_score = 0.0
        if target_grid is not None:
            visibility_score = self.line_clear_ratio(grid, target_grid, blocked_map)

        score = (
            2.0 * progress_score
            + 1.5 * standoff_score
            + 1.2 * safety_score
            + 0.8 * path_clear_score
            + 0.8 * visibility_score
            + 0.4 * semantic_score
            + 0.2 * confidence_score
            - 0.8 * distance_penalty
            - 0.3 * float(cost_map[gx, gy])
        )

        return float(score)

    def make_navigation_target(self, current_pose, target_position, planner_target, candidate):
        result = copy.deepcopy(planner_target)
        viewpoint_position = candidate["position"]

        yaw_to_target = math.degrees(
            math.atan2(
                target_position[1] - viewpoint_position[1],
                target_position[0] - viewpoint_position[0],
            )
        )

        relative_angle = self.compute_relative_angle(
            current_pose=current_pose,
            goal_position=viewpoint_position,
        )

        result["valid"] = True
        result["source_target_type"] = planner_target.get("target_type", "unknown")
        result["target_type"] = "target_approach_viewpoint"
        result["planner_goal_type"] = "approach_viewpoint"
        result["position"] = viewpoint_position
        result["viewpoint_position"] = viewpoint_position
        result["approach_viewpoint"] = viewpoint_position
        result["target_world_position"] = target_position
        result["target_yaw"] = round(yaw_to_target, 2)
        result["relative_angle"] = round(relative_angle, 2)
        result["relative_region"] = self.classify_relative_region(relative_angle)
        result["distance"] = round(candidate["current_distance"], 2)
        result["target_distance"] = round(candidate["target_distance"], 2)
        result["score"] = round(float(candidate["score"]), 4)
        result["approach_reason"] = candidate.get("candidate_type", "selected approach viewpoint")
        result["candidate_type"] = candidate.get("candidate_type", "unknown")
        result["candidate_grid"] = candidate.get("grid", None)
        result["stop_reason"] = ""

        return result

    def build_map_status(self):
        confidence = np.clip(self.memory.confidence, 0.0, 1.0)
        safety = np.clip(self.memory.safety_value, 0.0, 1.0)

        cost_map = np.ones((self.memory.grid_size, self.memory.grid_size), dtype=np.float32)
        safety_cost = 1.0 - safety
        unknown_cost = np.where(confidence > 0.0, 0.0, 0.5)
        cost_map = cost_map + safety_cost + unknown_cost

        blocked_map = (
            (safety <= self.blocked_safety_threshold)
            & (confidence >= self.blocked_confidence_threshold)
        )

        return cost_map, blocked_map

    def find_nearest_free_grid(self, grid, blocked_map):
        gx, gy = grid

        for radius in range(1, self.max_nearest_free_radius + 1):
            candidates = []

            for nx in range(gx - radius, gx + radius + 1):
                for ny in range(gy - radius, gy + radius + 1):
                    if not self.memory.in_bounds(nx, ny):
                        continue
                    if blocked_map[nx, ny]:
                        continue

                    dist = math.hypot(nx - gx, ny - gy)
                    candidates.append((dist, (nx, ny)))

            if len(candidates) > 0:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]

        return None

    def line_clear_ratio(self, start_grid, end_grid, blocked_map):
        cells = self.bresenham_line(start_grid, end_grid)
        if len(cells) == 0:
            return 0.0

        clear_count = 0
        total_count = 0

        for gx, gy in cells:
            if not self.memory.in_bounds(gx, gy):
                continue

            total_count += 1
            if not blocked_map[gx, gy]:
                clear_count += 1

        if total_count == 0:
            return 0.0

        return float(clear_count) / float(total_count)

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

    def compute_relative_angle(self, current_pose, goal_position):
        dx = goal_position[0] - current_pose[0]
        dy = goal_position[1] - current_pose[1]
        goal_yaw = math.degrees(math.atan2(dy, dx))

        if len(current_pose) >= 4:
            current_yaw = float(current_pose[3])
        else:
            current_yaw = 0.0

        return self.normalize_angle(goal_yaw - current_yaw)

    def classify_relative_region(self, relative_angle):
        if -45.0 <= relative_angle <= 45.0:
            return "front"

        if 45.0 < relative_angle <= 135.0:
            return "left"

        if -135.0 <= relative_angle < -45.0:
            return "right"

        if relative_angle > 135.0:
            return "back_left"

        return "back_right"

    def normalize_angle(self, angle):
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def xy_distance(self, point_a, point_b):
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    def safe_div(self, numerator, denominator):
        if abs(denominator) < 1e-6:
            return 0.0
        return float(numerator) / float(denominator)

    def default_navigation_target(self, reason):
        return {
            "valid": False,
            "target_type": "target_approach_viewpoint",
            "planner_goal_type": "none",
            "position": None,
            "viewpoint_position": None,
            "approach_viewpoint": None,
            "target_world_position": None,
            "score": 0.0,
            "confidence": 0.0,
            "distance": 0.0,
            "target_distance": 0.0,
            "target_yaw": 0.0,
            "relative_angle": 0.0,
            "relative_region": "front",
            "approach_reason": reason,
            "stop_reason": "",
        }