import math
import numpy as np


class ViewpointPlanner:
    def __init__(
        self,
        memory,
        obstacle_confidence_threshold=0.45,
        free_confidence_threshold=0.03,
        min_viewpoint_safety=0.25,
        min_los_ratio=0.25,
    ):
        self.memory = memory
        self.obstacle_confidence_threshold = obstacle_confidence_threshold
        self.free_confidence_threshold = free_confidence_threshold
        self.min_viewpoint_safety = min_viewpoint_safety
        self.min_los_ratio = min_los_ratio

    def generate_viewpoints_for_frontiers(self, frontiers, score_map, current_pose):
        new_frontiers = []
        for frontier in frontiers:
            frontier = self.generate_viewpoint_for_frontier(
                frontier=frontier,
                score_map=score_map,
                current_pose=current_pose
            )
            new_frontiers.append(frontier)
        return new_frontiers

    def generate_viewpoint_for_frontier(self, frontier, score_map, current_pose):
        candidates = self.generate_candidate_viewpoints(
            frontier=frontier,
            current_pose=current_pose
        )
        if len(candidates) == 0:
            frontier["viewpoint_valid"] = False
            frontier["viewpoint_reason"] = "no candidate viewpoint"
            return frontier

        best_candidate = None
        best_score = -1.0
        for candidate in candidates:
            candidate_score = self.evaluate_candidate_viewpoint(
                candidate=candidate,
                frontier=frontier,
                score_map=score_map,
                current_pose=current_pose
            )
            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate = candidate

        if best_candidate is None or best_score <= 0.0:
            frontier["viewpoint_valid"] = False
            frontier["viewpoint_reason"] = "no feasible candidate viewpoint"
            return frontier

        frontier["viewpoint_position"] = (
            round(best_candidate["x"], 2),
            round(best_candidate["y"], 2)
        )
        frontier["viewpoint_yaw"] = round(best_candidate["yaw"], 2)
        frontier["position"] = frontier["viewpoint_position"]
        frontier["score"] = 0.7 * frontier["score"] + 0.3 * best_score
        frontier["viewpoint_score"] = float(best_score)
        frontier["viewpoint_valid"] = True
        frontier["viewpoint_reason"] = "ok"
        frontier["viewpoint_los_ratio"] = float(best_candidate.get("los_ratio", 0.0))
        frontier["viewpoint_free_confidence"] = float(
            best_candidate.get("free_confidence", 0.0)
        )
        frontier["viewpoint_obstacle_confidence"] = float(
            best_candidate.get("obstacle_confidence", 0.0)
        )
        return frontier

    def generate_candidate_viewpoints(self, frontier, current_pose):
        x, y, z, yaw = current_pose
        center_x, center_y = frontier["frontier_position"]

        angle_from_frontier_to_current = math.degrees(
            math.atan2(y - center_y, x - center_x)
        )
        radius_list = [6.0, 8.0, 10.0, 12.0]
        angle_offsets = [-75.0, -45.0, -20.0, 0.0, 20.0, 45.0, 75.0]

        candidates = []
        for radius in radius_list:
            for offset in angle_offsets:
                candidate_angle = angle_from_frontier_to_current + offset
                cx = center_x + radius * math.cos(math.radians(candidate_angle))
                cy = center_y + radius * math.sin(math.radians(candidate_angle))
                grid = self.memory.world_to_grid(cx, cy)
                if grid is None:
                    continue

                dist_to_current = math.hypot(cx - x, cy - y)
                if dist_to_current < self.memory.resolution:
                    continue

                yaw_to_frontier = math.degrees(
                    math.atan2(center_y - cy, center_x - cx)
                )
                candidates.append({
                    "x": cx,
                    "y": cy,
                    "yaw": yaw_to_frontier,
                    "distance_to_current": dist_to_current,
                    "grid": grid
                })

        return candidates

    def evaluate_candidate_viewpoint(self, candidate, frontier, score_map, current_pose):
        cx = candidate["x"]
        cy = candidate["y"]
        candidate_yaw = candidate["yaw"]
        cgx, cgy = candidate["grid"]

        viewpoint_quality = self.evaluate_viewpoint_geometry(
            grid=(cgx, cgy),
            candidate=candidate,
            frontier=frontier
        )
        if not viewpoint_quality["valid"]:
            return 0.0

        visible_score = 0.0
        visible_count = 0
        los_visible_count = 0

        for gx, gy in frontier["cells"]:
            wx, wy = self.memory.grid_to_world(gx, gy)
            dx = wx - cx
            dy = wy - cy
            dist = math.hypot(dx, dy)
            if dist > self.memory.max_sensing_range:
                continue

            cell_yaw = math.degrees(math.atan2(dy, dx))
            yaw_diff = abs(self.memory.normalize_angle(cell_yaw - candidate_yaw))
            if yaw_diff > self.memory.sector_angle / 2.0:
                continue

            visible_count += 1
            if self.has_line_of_sight(
                start=(cgx, cgy),
                end=(gx, gy)
            ):
                visible_score += float(score_map[gx, gy])
                los_visible_count += 1

        if visible_count == 0:
            return 0.0

        los_ratio = float(los_visible_count / max(visible_count, 1))
        candidate["los_ratio"] = los_ratio
        if los_ratio < self.min_los_ratio:
            return 0.0

        information_gain = visible_score / max(len(frontier["cells"]), 1)
        candidate_safety = viewpoint_quality["safety"]
        candidate_free = viewpoint_quality["free_confidence"]
        candidate_obstacle = viewpoint_quality["obstacle_confidence"]
        candidate_visited = bool(self.memory.visited[cgx, cgy])

        candidate["free_confidence"] = candidate_free
        candidate["obstacle_confidence"] = candidate_obstacle

        distance_to_current = float(candidate["distance_to_current"])
        distance_cost = min(
            1.0,
            distance_to_current / max(self.memory.map_size / 2.0, 1e-6)
        )
        visited_penalty = 0.75 if candidate_visited else 1.0

        candidate_score = (
            0.45 * information_gain
            + 0.22 * frontier["score"]
            + 0.13 * candidate_safety
            + 0.10 * candidate_free
            + 0.06 * los_ratio
            + 0.04 * (1.0 - distance_cost)
        )
        candidate_score = candidate_score * visited_penalty
        return float(max(0.0, candidate_score))

    def evaluate_viewpoint_geometry(self, grid, candidate, frontier):
        gx, gy = grid
        safety = float(self.memory.safety_value[gx, gy])
        confidence = float(self.memory.confidence[gx, gy])
        free_confidence = self.get_memory_value("free_confidence", gx, gy)
        obstacle_confidence = self.get_memory_value("obstacle_confidence", gx, gy)

        if obstacle_confidence >= self.obstacle_confidence_threshold:
            return {
                "valid": False,
                "reason": "viewpoint is occupied",
                "safety": safety,
                "free_confidence": free_confidence,
                "obstacle_confidence": obstacle_confidence,
            }

        if confidence > 0.0 and safety < self.min_viewpoint_safety:
            return {
                "valid": False,
                "reason": "viewpoint has low safety",
                "safety": safety,
                "free_confidence": free_confidence,
                "obstacle_confidence": obstacle_confidence,
            }

        if (
            free_confidence < self.free_confidence_threshold
            and not bool(self.memory.visited[gx, gy])
            and confidence <= 0.0
        ):
            return {
                "valid": False,
                "reason": "viewpoint is not in known free space",
                "safety": safety,
                "free_confidence": free_confidence,
                "obstacle_confidence": obstacle_confidence,
            }

        if confidence <= 0.0:
            safety = 0.55

        return {
            "valid": True,
            "reason": "ok",
            "safety": safety,
            "free_confidence": free_confidence,
            "obstacle_confidence": obstacle_confidence,
        }

    def has_line_of_sight(self, start, end):
        cells = self.bresenham_line(start, end)
        for gx, gy in cells:
            if not self.memory.in_bounds(gx, gy):
                return False

            obstacle_confidence = self.get_memory_value(
                "obstacle_confidence", gx, gy
            )
            if obstacle_confidence >= self.obstacle_confidence_threshold:
                return False

            confidence = float(self.memory.confidence[gx, gy])
            safety = float(self.memory.safety_value[gx, gy])
            if confidence > 0.0 and safety < self.min_viewpoint_safety:
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

    def get_memory_value(self, name, gx, gy):
        if not hasattr(self.memory, name):
            return 0.0
        value_map = getattr(self.memory, name)
        try:
            return float(value_map[gx, gy])
        except Exception:
            return 0.0