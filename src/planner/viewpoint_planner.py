import copy
import math

try:
    from src.planner.planning_types import ExecutableViewpoint
except Exception:
    from planner.planning_types import ExecutableViewpoint


class ViewpointPlanner:
    """
    Build executable viewpoints from semantic navigation targets.

    NavigationTarget / memory target / verified target are semantic anchors.
    LocalPlanner should receive an executable viewpoint, not a raw object
    position. This planner keeps the existing frontier viewpoint API while
    adding approach-viewpoint generation for verified targets.
    """

    def __init__(
        self,
        memory,
        approach_radius_list=None,
        explore_radius_list=None,
        min_approach_distance=5.0,
        max_approach_distance=16.0,
        min_safety_score=0.20,
    ):
        self.memory = memory
        self.approach_radius_list = approach_radius_list or [6.0, 8.0, 10.0, 12.0, 15.0]
        self.explore_radius_list = explore_radius_list or [6.0, 8.0, 10.0, 12.0]
        self.min_approach_distance = float(min_approach_distance)
        self.max_approach_distance = float(max_approach_distance)
        self.min_safety_score = float(min_safety_score)

    def build_executable_viewpoint(self, navigation_target, current_pose, navigation_info=None, rejected_viewpoint_ids=None):
        """
        Convert a selected semantic target into an executable viewpoint.

        Args:
            navigation_target: dict selected by ON_Air_2.
            current_pose: [x, y, z, yaw_degree].
            navigation_info: optional NavigationState output.

        Returns:
            dict compatible with the current LocalPlanner interface. It includes:
                valid
                viewpoint_type
                viewpoint_position
                position
                viewpoint_yaw
                anchor_position
                executable_viewpoint
        """
        if not isinstance(navigation_target, dict):
            return self.invalid_viewpoint("navigation target is not dict")

        if not navigation_target.get("valid", False):
            return self.invalid_viewpoint(
                navigation_target.get("reason", "navigation target is invalid"),
                navigation_target=navigation_target
            )

        current_pose = self.normalize_current_pose(current_pose)
        if current_pose is None:
            return self.invalid_viewpoint(
                "current pose is invalid",
                navigation_target=navigation_target
            )

        if self.is_approach_target(navigation_target, navigation_info):
            return self.build_approach_viewpoint(
                navigation_target=navigation_target,
                current_pose=current_pose,
                rejected_viewpoint_ids=rejected_viewpoint_ids
            )

        return self.build_explore_viewpoint(
            navigation_target=navigation_target,
            current_pose=current_pose
        )

    def build_explore_viewpoint(self, navigation_target, current_pose):
        """
        Exploration targets are already viewpoint-like in the current branch:
        semantic frontiers and SGCP-lite memory targets usually carry
        viewpoint_position. This method normalizes them into the same
        executable viewpoint format used by approach targets.
        """
        anchor = self.extract_anchor_position(navigation_target, current_pose)
        viewpoint_position = self.extract_viewpoint_position(navigation_target, current_pose)

        if viewpoint_position is None and anchor is not None:
            viewpoint_position = anchor

        if viewpoint_position is None:
            return self.invalid_viewpoint(
                "explore target has no viewpoint position",
                navigation_target=navigation_target
            )

        viewpoint_position = self.keep_current_altitude(
            position=viewpoint_position,
            current_pose=current_pose
        )

        if not self.is_inside_memory(viewpoint_position):
            return self.invalid_viewpoint(
                "explore viewpoint is outside memory map",
                navigation_target=navigation_target,
                viewpoint_position=viewpoint_position
            )

        viewpoint_yaw = self.extract_viewpoint_yaw(
            navigation_target=navigation_target,
            viewpoint_position=viewpoint_position,
            anchor_position=anchor,
            current_pose=current_pose
        )

        safety_score = self.get_safety_score(viewpoint_position)
        information_gain = float(navigation_target.get("viewpoint_score", navigation_target.get("score", 0.0)) or 0.0)

        return self.make_viewpoint_target(
            navigation_target=navigation_target,
            viewpoint_type="explore",
            viewpoint_position=viewpoint_position,
            viewpoint_yaw=viewpoint_yaw,
            anchor_position=anchor,
            safety_score=safety_score,
            information_gain=information_gain,
            reason="explore viewpoint from semantic memory"
        )

    def build_approach_viewpoint(self, navigation_target, current_pose, rejected_viewpoint_ids=None):
        """
        Verified target position is only a semantic anchor. Generate an approach
        viewpoint around it and face the target from a safe, reachable position.
        """
        anchor = self.extract_anchor_position(navigation_target, current_pose)
        if anchor is None:
            return self.invalid_viewpoint(
                "verified target has no anchor position",
                navigation_target=navigation_target
            )

        anchor = self.keep_current_altitude(anchor, current_pose)
        candidates = self.generate_approach_candidates(
            anchor_position=anchor,
            current_pose=current_pose
        )

        rejected_viewpoint_ids = set(rejected_viewpoint_ids or [])
        if len(rejected_viewpoint_ids) > 0:
            candidates = [
                candidate for candidate in candidates
                if candidate.get("viewpoint_id", "") not in rejected_viewpoint_ids
            ]

        if len(candidates) == 0:
            return self.invalid_viewpoint(
                "no valid approach viewpoint candidate",
                navigation_target=navigation_target,
                anchor_position=anchor
            )

        best_candidate = None
        best_score = -1.0

        for candidate in candidates:
            score = self.evaluate_approach_candidate(
                candidate=candidate,
                anchor_position=anchor,
                current_pose=current_pose
            )
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            return self.invalid_viewpoint(
                "failed to select approach viewpoint",
                navigation_target=navigation_target,
                anchor_position=anchor
            )

        viewpoint_position = (
            round(best_candidate["x"], 2),
            round(best_candidate["y"], 2),
            round(current_pose[2], 2),
        )
        viewpoint_yaw = self.yaw_to_point(
            from_position=viewpoint_position,
            to_position=anchor
        )

        return self.make_viewpoint_target(
            navigation_target=navigation_target,
            viewpoint_type="approach",
            viewpoint_position=viewpoint_position,
            viewpoint_yaw=viewpoint_yaw,
            anchor_position=anchor,
            safety_score=best_candidate.get("safety_score", 0.0),
            information_gain=best_score,
            reason="approach viewpoint around verified target"
        )

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
            round(best_candidate["y"], 2),
        )
        frontier["viewpoint_yaw"] = round(best_candidate["yaw"], 2)
        frontier["position"] = frontier["viewpoint_position"]
        frontier["score"] = 0.7 * frontier["score"] + 0.3 * best_score
        frontier["viewpoint_score"] = float(best_score)
        frontier["viewpoint_type"] = "explore"

        return frontier

    def generate_candidate_viewpoints(self, frontier, current_pose):
        x, y, z, yaw = current_pose
        center_x, center_y = frontier["frontier_position"][:2]

        angle_from_frontier_to_current = math.degrees(
            math.atan2(y - center_y, x - center_x)
        )

        angle_offsets = [-75.0, -45.0, -20.0, 0.0, 20.0, 45.0, 75.0]
        candidates = []

        for radius in self.explore_radius_list:
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

    def generate_approach_candidates(self, anchor_position, current_pose):
        x, y, z, yaw = current_pose
        ax, ay, az = anchor_position

        angle_from_anchor_to_current = math.degrees(math.atan2(y - ay, x - ax))
        angle_offsets = [-90.0, -60.0, -35.0, -15.0, 0.0, 15.0, 35.0, 60.0, 90.0]

        candidates = []
        for radius in self.approach_radius_list:
            if radius < self.min_approach_distance or radius > self.max_approach_distance:
                continue

            for offset in angle_offsets:
                candidate_angle = angle_from_anchor_to_current + offset
                cx = ax + radius * math.cos(math.radians(candidate_angle))
                cy = ay + radius * math.sin(math.radians(candidate_angle))
                cz = z

                grid = self.memory.world_to_grid(cx, cy)
                if grid is None:
                    continue

                safety_score = self.get_safety_score((cx, cy, cz))
                if safety_score < self.min_safety_score:
                    continue

                distance_to_current = math.hypot(cx - x, cy - y)
                yaw_to_anchor = math.degrees(math.atan2(ay - cy, ax - cx))
                line_of_sight = self.approximate_line_of_sight(
                    from_position=(cx, cy, cz),
                    to_position=anchor_position
                )

                viewpoint_id = self.make_viewpoint_id(
                    "approach",
                    (cx, cy, cz)
                )

                candidates.append({
                    "x": cx,
                    "y": cy,
                    "z": cz,
                    "yaw": yaw_to_anchor,
                    "viewpoint_id": viewpoint_id,
                    "grid": grid,
                    "radius": radius,
                    "distance_to_current": distance_to_current,
                    "safety_score": safety_score,
                    "line_of_sight": line_of_sight,
                })

        return candidates

    def evaluate_approach_candidate(self, candidate, anchor_position, current_pose):
        distance_to_current = float(candidate.get("distance_to_current", 0.0))
        radius = float(candidate.get("radius", 0.0))
        safety_score = float(candidate.get("safety_score", 0.0))
        line_of_sight = bool(candidate.get("line_of_sight", False))

        distance_cost = min(1.0, distance_to_current / max(self.memory.map_size / 2.0, 1e-6))

        preferred_radius = 8.0
        radius_cost = min(1.0, abs(radius - preferred_radius) / max(preferred_radius, 1e-6))

        line_of_sight_score = 1.0 if line_of_sight else 0.3

        score = (
            0.40 * safety_score
            + 0.25 * line_of_sight_score
            + 0.20 * (1.0 - radius_cost)
            + 0.15 * (1.0 - distance_cost)
        )

        return float(max(0.0, score))

    def make_viewpoint_target(
        self,
        navigation_target,
        viewpoint_type,
        viewpoint_position,
        viewpoint_yaw,
        anchor_position,
        safety_score,
        information_gain,
        reason,
    ):
        target = copy.deepcopy(navigation_target)

        viewpoint_position = self.round_position(viewpoint_position)
        anchor_position = self.round_position(anchor_position) if anchor_position is not None else None

        executable_viewpoint = ExecutableViewpoint(
            valid=True,
            viewpoint_id=self.make_viewpoint_id(viewpoint_type, viewpoint_position),
            viewpoint_type=viewpoint_type,
            position=viewpoint_position,
            yaw=round(float(viewpoint_yaw), 2) if viewpoint_yaw is not None else None,
            anchor_target_id=str(target.get("target_id", target.get("id", ""))),
            anchor_position=anchor_position,
            expected_observation_direction=target.get("observation_region", ""),
            safety_score=float(safety_score),
            information_gain=float(information_gain),
            reason=reason,
            debug={
                "original_target_type": target.get("target_type", ""),
                "original_source": target.get("source", ""),
            }
        )

        target["valid"] = True
        target["viewpoint_type"] = viewpoint_type
        target["viewpoint_id"] = executable_viewpoint.viewpoint_id
        target["viewpoint_position"] = viewpoint_position
        target["position"] = viewpoint_position
        target["viewpoint_yaw"] = executable_viewpoint.yaw
        target["target_yaw"] = executable_viewpoint.yaw
        target["anchor_position"] = anchor_position
        target["safety_score"] = float(safety_score)
        target["information_gain"] = float(information_gain)
        target["viewpoint_reason"] = reason
        target["executable_viewpoint"] = executable_viewpoint.to_dict()

        return target

    def invalid_viewpoint(
        self,
        reason,
        navigation_target=None,
        viewpoint_position=None,
        anchor_position=None,
    ):
        target = copy.deepcopy(navigation_target) if isinstance(navigation_target, dict) else {}
        target["valid"] = False
        target["viewpoint_type"] = target.get("viewpoint_type", "invalid")
        target["viewpoint_reason"] = str(reason)
        target["reason"] = str(reason)

        if viewpoint_position is not None:
            target["viewpoint_position"] = self.round_position(viewpoint_position)
            target["position"] = self.round_position(viewpoint_position)

        if anchor_position is not None:
            target["anchor_position"] = self.round_position(anchor_position)

        executable_viewpoint = ExecutableViewpoint.invalid(
            reason=str(reason),
            anchor_target_id=str(target.get("target_id", target.get("id", "")))
        )
        target["executable_viewpoint"] = executable_viewpoint.to_dict()

        return target

    def is_approach_target(self, navigation_target, navigation_info=None):
        target_type = str(navigation_target.get("target_type", "")).lower()
        source = str(navigation_target.get("source", "")).lower()

        if "verified" in target_type:
            return True

        if "gdino" in target_type and self.extract_anchor_position(navigation_target, None) is not None:
            return True

        if navigation_target.get("target_world_position", None) is not None:
            return True

        if navigation_target.get("verified_target_position", None) is not None:
            return True

        if isinstance(navigation_info, dict):
            mode = str(navigation_info.get("mode", "")).lower()
            if mode in ["navigate", "approach", "final_check"]:
                return True

        if "navigation_state" in source and self.extract_anchor_position(navigation_target, None) is not None:
            return True

        return False

    def extract_anchor_position(self, target, current_pose=None):
        if not isinstance(target, dict):
            return None

        keys = [
            "target_world_position",
            "verified_target_position",
            "anchor_position",
            "target_position",
            "frontier_position",
            "viewpoint_position",
            "position",
        ]

        default_z = 0.0
        if current_pose is not None and len(current_pose) >= 3:
            default_z = float(current_pose[2])

        for key in keys:
            position = self.normalize_position(target.get(key, None), default_z=default_z)
            if position is not None:
                return position

        return None

    def extract_viewpoint_position(self, target, current_pose):
        default_z = float(current_pose[2])
        for key in ["viewpoint_position", "position"]:
            position = self.normalize_position(target.get(key, None), default_z=default_z)
            if position is not None:
                return position
        return None

    def extract_viewpoint_yaw(
        self,
        navigation_target,
        viewpoint_position,
        anchor_position,
        current_pose,
    ):
        yaw = navigation_target.get("viewpoint_yaw", navigation_target.get("target_yaw", None))
        if yaw is not None:
            try:
                return float(yaw)
            except Exception:
                pass

        if anchor_position is not None:
            return self.yaw_to_point(
                from_position=viewpoint_position,
                to_position=anchor_position
            )

        return float(current_pose[3])

    def keep_current_altitude(self, position, current_pose):
        if position is None:
            return None

        return (
            float(position[0]),
            float(position[1]),
            float(current_pose[2]),
        )

    def is_inside_memory(self, position):
        if position is None:
            return False

        grid = self.memory.world_to_grid(float(position[0]), float(position[1]))
        return grid is not None

    def get_safety_score(self, position):
        try:
            grid = self.memory.world_to_grid(float(position[0]), float(position[1]))
            if grid is None:
                return 0.0

            gx, gy = grid
            safety = float(self.memory.safety_value[gx, gy])
            confidence = float(self.memory.confidence[gx, gy])

            if confidence <= 0.0:
                return 0.55

            return safety
        except Exception:
            return 0.0

    def approximate_line_of_sight(self, from_position, to_position):
        if from_position is None or to_position is None:
            return False

        x0, y0 = float(from_position[0]), float(from_position[1])
        x1, y1 = float(to_position[0]), float(to_position[1])

        distance = math.hypot(x1 - x0, y1 - y0)
        if distance <= 1e-6:
            return True

        steps = max(2, int(distance / max(self.memory.resolution, 1e-6)))
        unsafe_count = 0

        for i in range(1, steps):
            ratio = i / float(steps)
            x = x0 + (x1 - x0) * ratio
            y = y0 + (y1 - y0) * ratio

            grid = self.memory.world_to_grid(x, y)
            if grid is None:
                continue

            gx, gy = grid
            confidence = float(self.memory.confidence[gx, gy])
            if confidence <= 0.0:
                continue

            safety = float(self.memory.safety_value[gx, gy])
            if safety < self.min_safety_score:
                unsafe_count += 1

        return unsafe_count == 0

    def yaw_to_point(self, from_position, to_position):
        dx = float(to_position[0]) - float(from_position[0])
        dy = float(to_position[1]) - float(from_position[1])
        return round(math.degrees(math.atan2(dy, dx)), 2)

    def normalize_position(self, position, default_z=0.0):
        if position is None:
            return None

        if not isinstance(position, (list, tuple)):
            return None

        if len(position) < 2:
            return None

        try:
            x = float(position[0])
            y = float(position[1])
            z = float(position[2]) if len(position) >= 3 else float(default_z)
        except Exception:
            return None

        return (x, y, z)

    def normalize_current_pose(self, current_pose):
        if current_pose is None:
            return None

        if not isinstance(current_pose, (list, tuple)):
            return None

        if len(current_pose) < 4:
            return None

        try:
            return (
                float(current_pose[0]),
                float(current_pose[1]),
                float(current_pose[2]),
                float(current_pose[3]),
            )
        except Exception:
            return None

    def make_viewpoint_id(self, viewpoint_type, viewpoint_position):
        if viewpoint_position is None:
            return f"{viewpoint_type}_invalid"

        return (
            f"{viewpoint_type}_"
            f"{round(float(viewpoint_position[0]), 1)}_"
            f"{round(float(viewpoint_position[1]), 1)}_"
            f"{round(float(viewpoint_position[2]), 1)}"
        )

    def round_position(self, position):
        if position is None:
            return None

        return (
            round(float(position[0]), 2),
            round(float(position[1]), 2),
            round(float(position[2]), 2) if len(position) >= 3 else 0.0,
        )
