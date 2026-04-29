import math


class ViewpointPlanner:
    def __init__(self, memory):
        self.memory = memory


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

        if best_candidate is None:
            return frontier

        frontier["viewpoint_position"] = (
            round(best_candidate["x"], 2),
            round(best_candidate["y"], 2)
        )
        frontier["viewpoint_yaw"] = round(best_candidate["yaw"], 2)
        frontier["position"] = frontier["viewpoint_position"]
        frontier["score"] = 0.7 * frontier["score"] + 0.3 * best_score
        frontier["viewpoint_score"] = float(best_score)

        return frontier


    def generate_candidate_viewpoints(self, frontier, current_pose):
        x, y, z, yaw = current_pose
        center_x, center_y = frontier["frontier_position"]

        angle_from_frontier_to_current = math.degrees(math.atan2(y - center_y, x - center_x))

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

                yaw_to_frontier = math.degrees(math.atan2(center_y - cy, center_x - cx))

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

        visible_score = 0.0
        visible_count = 0

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

            visible_score += float(score_map[gx, gy])
            visible_count += 1

        if visible_count == 0:
            return 0.0

        information_gain = visible_score / max(len(frontier["cells"]), 1)

        cgx, cgy = candidate["grid"]
        candidate_safety = float(self.memory.safety_value[cgx, cgy])
        candidate_confidence = float(self.memory.confidence[cgx, cgy])
        candidate_visited = bool(self.memory.visited[cgx, cgy])

        if candidate_confidence <= 0.0:
            candidate_safety = 0.55

        distance_to_current = float(candidate["distance_to_current"])
        distance_cost = min(1.0, distance_to_current / max(self.memory.map_size / 2.0, 1e-6))

        visited_penalty = 0.75 if candidate_visited else 1.0

        candidate_score = (
            0.55 * information_gain
            + 0.25 * frontier["score"]
            + 0.15 * candidate_safety
            + 0.05 * (1.0 - distance_cost)
        )

        candidate_score = candidate_score * visited_penalty
        return float(max(0.0, candidate_score))