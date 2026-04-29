import math


class SGCPPlanner:
    def __init__(
        self,
        memory,
        semantic_gap_rho=0.12,
    ):
        self.memory = memory
        self.semantic_gap_rho = semantic_gap_rho


    def select_next_frontier(self, frontiers, current_pose):
        valid_frontiers = self.get_valid_frontiers(frontiers)

        if len(valid_frontiers) == 0:
            return None

        constraints = self.build_precedence_constraints(
            frontiers=valid_frontiers,
            semantic_gap_rho=self.semantic_gap_rho
        )

        candidate_indices = self.get_feasible_frontier_indices(
            frontiers=valid_frontiers,
            constraints=constraints
        )

        if len(candidate_indices) == 0:
            candidate_indices = list(range(len(valid_frontiers)))

        best_frontier = None
        best_score = -1.0

        for index in candidate_indices:
            frontier = valid_frontiers[index]
            sgcp_score = self.compute_sgcp_score(
                frontier=frontier,
                current_pose=current_pose
            )

            if sgcp_score > best_score:
                best_score = sgcp_score
                best_frontier = frontier

        if best_frontier is None:
            return None

        target = dict(best_frontier)
        target["target_type"] = "sgcp_frontier"
        target["sgcp_score"] = float(best_score)
        target["sgcp_candidate_count"] = len(candidate_indices)
        target["sgcp_total_frontier_count"] = len(valid_frontiers)
        target["sgcp_constraint_count"] = len(constraints)
        target["semantic_gap_rho"] = self.semantic_gap_rho

        return target


    def get_valid_frontiers(self, frontiers):
        valid_frontiers = []

        for frontier in frontiers:
            if not isinstance(frontier, dict):
                continue

            if not frontier.get("valid", False):
                continue

            if frontier.get("position", None) is None:
                continue

            valid_frontiers.append(frontier)

        return valid_frontiers


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


    def compute_sgcp_score(self, frontier, current_pose):
        planning_score = self.normalize_score(frontier.get("score", 0.0))
        semantic_score = self.normalize_score(frontier.get("semantic_value", 0.0))
        safety_score = self.normalize_score(frontier.get("safety_value", 0.5))
        novelty_score = self.normalize_score(frontier.get("novelty_value", 0.5))
        viewpoint_score = self.normalize_score(frontier.get("viewpoint_score", planning_score))
        unknown_gain = self.normalize_score(frontier.get("unknown_gain", 0.0))
        history_penalty = self.normalize_score(frontier.get("history_penalty", 0.0))

        distance_cost = self.compute_frontier_travel_cost(
            frontier=frontier,
            current_pose=current_pose
        )
        geometry_score = 1.0 - distance_cost

        sgcp_score = (
            0.26 * planning_score
            + 0.18 * semantic_score
            + 0.14 * safety_score
            + 0.08 * novelty_score
            + 0.10 * viewpoint_score
            + 0.14 * geometry_score
            + 0.10 * unknown_gain
        )

        sgcp_score = sgcp_score - 0.08 * history_penalty

        return float(max(0.0, min(1.0, sgcp_score)))


    def compute_frontier_travel_cost(self, frontier, current_pose):
        x, y, z, yaw = current_pose

        pos = frontier.get("position", None)
        if pos is None:
            return 1.0

        tx, ty = pos
        distance = math.hypot(tx - x, ty - y)

        max_distance = max(self.memory.map_size / 2.0, 1e-6)
        distance_cost = min(1.0, distance / max_distance)

        return float(distance_cost)


    def get_semantic_priority(self, frontier):
        semantic_value = frontier.get("semantic_value", None)

        if semantic_value is None:
            semantic_value = frontier.get("score", 0.0)

        unknown_gain = frontier.get("unknown_gain", 0.0)

        semantic_priority = (
            0.85 * self.normalize_score(semantic_value)
            + 0.15 * self.normalize_score(unknown_gain)
        )

        return semantic_priority


    def normalize_score(self, score):
        try:
            score = float(score)
        except Exception:
            score = 0.0

        score = max(0.0, min(1.0, score))
        return score