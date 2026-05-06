try:
    from src.planner.geometry_planner import OccupancyGeometryPlanner
except Exception:
    from planner.geometry_planner import OccupancyGeometryPlanner


class SGCPPlanner:
    def __init__(
        self,
        memory,
        semantic_gap_rho=0.12,
        max_geometry_cost=0.75,
        max_path_to_straight_ratio=4.0,
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
    ):
        self.memory = memory
        self.semantic_gap_rho = semantic_gap_rho
        self.max_geometry_cost = max_geometry_cost
        self.max_path_to_straight_ratio = max_path_to_straight_ratio
        self.geometry_planner = OccupancyGeometryPlanner(
            memory=memory,
            unknown_cost=unknown_cost,
            safety_weight=safety_weight,
            semantic_weight=semantic_weight,
            visited_weight=visited_weight,
            blocked_safety_threshold=blocked_safety_threshold,
            blocked_confidence_threshold=blocked_confidence_threshold,
            free_confidence_threshold=free_confidence_threshold,
            free_safety_threshold=free_safety_threshold,
            obstacle_confidence_threshold=obstacle_confidence_threshold,
            obstacle_inflation_radius=obstacle_inflation_radius,
            risk_inflation_radius=risk_inflation_radius,
            risk_safety_threshold=risk_safety_threshold,
            risk_weight=risk_weight,
            block_unknown=block_unknown,
        )

    def select_next_frontier(self, frontiers, current_pose):
        valid_frontiers = self.get_valid_frontiers(frontiers)
        if len(valid_frontiers) == 0:
            return None

        geometry = self.geometry_planner.build_geometry_maps()
        start_grid = self.memory.world_to_grid(current_pose[0], current_pose[1])
        if start_grid is None:
            return None

        self.geometry_planner.release_start_area(
            start=start_grid,
            blocked_map=geometry["blocked_map"],
            free_map=geometry["free_map"]
        )

        annotated_frontiers = self.annotate_frontier_geometry(
            frontiers=valid_frontiers,
            current_pose=current_pose,
            start_grid=start_grid,
            geometry=geometry
        )

        reachable_frontiers = [
            frontier for frontier in annotated_frontiers
            if frontier.get("geometry_reachable", False)
        ]
        if len(reachable_frontiers) == 0:
            return None

        reasonable_frontiers = self.filter_reasonable_geometry(reachable_frontiers)
        if len(reasonable_frontiers) == 0:
            reasonable_frontiers = reachable_frontiers

        constraints = self.build_precedence_constraints(
            frontiers=reasonable_frontiers,
            semantic_gap_rho=self.semantic_gap_rho
        )
        candidate_indices = self.get_feasible_frontier_indices(
            frontiers=reasonable_frontiers,
            constraints=constraints
        )
        if len(candidate_indices) == 0:
            candidate_indices = list(range(len(reasonable_frontiers)))

        best_index = self.select_by_geometric_objective(
            frontiers=reasonable_frontiers,
            candidate_indices=candidate_indices
        )
        if best_index is None:
            return None

        best_frontier = dict(reasonable_frontiers[best_index])
        best_frontier["target_type"] = "sgcp_frontier"
        best_frontier["sgcp_score"] = self.compute_sgcp_score(best_frontier)
        best_frontier["sgcp_candidate_count"] = len(candidate_indices)
        best_frontier["sgcp_total_frontier_count"] = len(valid_frontiers)
        best_frontier["sgcp_reachable_frontier_count"] = len(reachable_frontiers)
        best_frontier["sgcp_reasonable_frontier_count"] = len(reasonable_frontiers)
        best_frontier["sgcp_constraint_count"] = len(constraints)
        best_frontier["semantic_gap_rho"] = self.semantic_gap_rho
        return best_frontier

    def get_valid_frontiers(self, frontiers):
        valid_frontiers = []
        for frontier in frontiers:
            if not isinstance(frontier, dict):
                continue
            if not frontier.get("valid", False):
                continue
            if frontier.get("viewpoint_valid", True) is False:
                continue
            if self.get_frontier_position(frontier) is None:
                continue
            valid_frontiers.append(frontier)
        return valid_frontiers

    def annotate_frontier_geometry(self, frontiers, current_pose, start_grid, geometry):
        annotated_frontiers = []
        for frontier in frontiers:
            frontier = dict(frontier)
            target_position = self.get_frontier_position(frontier)
            geometry_result = self.geometry_planner.evaluate_path(
                current_pose=current_pose,
                target_position=target_position,
                z=current_pose[2],
                geometry=geometry,
                start_grid=start_grid,
                allow_goal_adjustment=True,
                allow_arrived=False,
            )

            frontier["geometry_reachable"] = geometry_result.get("reachable", False)
            frontier["geometry_arrived"] = geometry_result.get("arrived", False)
            frontier["geometry_reason"] = geometry_result.get("reason", "unknown")
            frontier["geometry_path_length"] = geometry_result.get("path_length", 0.0)
            frontier["geometry_path_len"] = geometry_result.get("path_len", 0)
            frontier["geometry_raw_path_len"] = geometry_result.get("raw_grid_path_len", 0)
            frontier["geometry_cost"] = geometry_result.get("geometry_cost", 1.0)
            frontier["geometry_score"] = geometry_result.get("geometry_score", 0.0)
            frontier["geometry_straight_distance"] = geometry_result.get("straight_distance", 0.0)
            frontier["geometry_path_to_straight_ratio"] = geometry_result.get(
                "path_to_straight_ratio", 1.0
            )
            frontier["geometry_goal_adjusted"] = geometry_result.get("goal_adjusted", False)
            frontier["geometry_blocked_cells"] = geometry_result.get("blocked_cells", 0)
            frontier["geometry_obstacle_cells"] = geometry_result.get("obstacle_cells", 0)
            frontier["geometry_unknown_blocked_cells"] = geometry_result.get(
                "unknown_blocked_cells", 0
            )

            if geometry_result.get("reachable", False):
                target_position = geometry_result.get("target_position", target_position)
                frontier["position"] = (
                    round(float(target_position[0]), 2),
                    round(float(target_position[1]), 2)
                )
                if frontier.get("viewpoint_position", None) is not None:
                    frontier["viewpoint_position"] = frontier["position"]

            annotated_frontiers.append(frontier)

        return annotated_frontiers

    def filter_reasonable_geometry(self, frontiers):
        filtered = []
        for frontier in frontiers:
            geometry_cost = float(frontier.get("geometry_cost", 1.0))
            ratio = float(frontier.get("geometry_path_to_straight_ratio", 1.0))
            if geometry_cost > self.max_geometry_cost:
                continue
            if ratio > self.max_path_to_straight_ratio:
                continue
            filtered.append(frontier)
        return filtered

    def build_precedence_constraints(self, frontiers, semantic_gap_rho):
        constraints = []
        for i in range(len(frontiers)):
            semantic_i = self.get_semantic_priority(frontiers[i])
            for j in range(len(frontiers)):
                if i == j:
                    continue
                semantic_j = self.get_semantic_priority(frontiers[j])
                if semantic_i - semantic_j > semantic_gap_rho:
                    constraints.append((i, j))
        return constraints

    def get_feasible_frontier_indices(self, frontiers, constraints):
        blocked_indices = set()
        for prior_idx, later_idx in constraints:
            blocked_indices.add(later_idx)

        feasible_indices = []
        for idx in range(len(frontiers)):
            if idx not in blocked_indices:
                feasible_indices.append(idx)
        return feasible_indices

    def select_by_geometric_objective(self, frontiers, candidate_indices):
        best_index = None
        best_key = None

        for idx in candidate_indices:
            frontier = frontiers[idx]
            geometry_cost = float(frontier.get("geometry_cost", 1.0))
            path_length = float(frontier.get("geometry_path_length", 0.0))
            semantic_priority = self.get_semantic_priority(frontier)
            viewpoint_score = self.normalize_score(
                frontier.get("viewpoint_score", frontier.get("score", 0.0))
            )
            unknown_gain = self.normalize_score(frontier.get("unknown_gain", 0.0))

            key = (
                geometry_cost,
                path_length,
                -semantic_priority,
                -viewpoint_score,
                -unknown_gain,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_index = idx

        return best_index

    def compute_sgcp_score(self, frontier):
        planning_score = self.normalize_score(frontier.get("score", 0.0))
        semantic_score = self.normalize_score(frontier.get("semantic_value", planning_score))
        viewpoint_score = self.normalize_score(frontier.get("viewpoint_score", planning_score))
        unknown_gain = self.normalize_score(frontier.get("unknown_gain", 0.0))
        geometry_score = self.normalize_score(frontier.get("geometry_score", 0.0))
        history_penalty = self.normalize_score(frontier.get("history_penalty", 0.0))

        score = (
            0.20 * planning_score
            + 0.20 * semantic_score
            + 0.15 * viewpoint_score
            + 0.35 * geometry_score
            + 0.10 * unknown_gain
            - 0.08 * history_penalty
        )
        return float(max(0.0, min(1.0, score)))

    def get_frontier_position(self, frontier):
        if frontier.get("viewpoint_position", None) is not None:
            return frontier["viewpoint_position"]
        if frontier.get("position", None) is not None:
            return frontier["position"]
        if frontier.get("frontier_position", None) is not None:
            return frontier["frontier_position"]
        return None

    def get_semantic_priority(self, frontier):
        semantic_value = frontier.get("semantic_value", None)
        if semantic_value is None:
            semantic_value = frontier.get("score", 0.0)
        unknown_gain = frontier.get("unknown_gain", 0.0)

        semantic_priority = (
            0.85 * self.normalize_score(semantic_value)
            + 0.15 * self.normalize_score(unknown_gain)
        )
        return float(max(0.0, min(1.0, semantic_priority)))

    def normalize_score(self, score):
        try:
            score = float(score)
        except Exception:
            score = 0.0
        return max(0.0, min(1.0, score))