import math
import numpy as np
from collections import deque


class SemanticFrontierBuilder:
    def __init__(self, memory):
        self.memory = memory


    def get_semantic_frontiers(
        self,
        current_pose,
        score_map=None,
        min_confidence=0.05,
        min_distance=3.0,
        max_distance=45.0,
        min_frontier_score=0.02,
        min_cluster_size=2,
        semantic_similarity_threshold=0.18,
    ):
        if score_map is None:
            score_map = self.memory.compute_planning_score_map()

        candidate_mask = self.build_frontier_candidate_mask(
            score_map=score_map,
            current_pose=current_pose,
            min_confidence=min_confidence,
            min_distance=min_distance,
            max_distance=max_distance,
            min_frontier_score=min_frontier_score
        )

        clusters = self.cluster_frontier_cells(
            candidate_mask=candidate_mask,
            score_map=score_map,
            min_cluster_size=min_cluster_size,
            semantic_similarity_threshold=semantic_similarity_threshold
        )

        frontiers = []
        for idx, cluster_cells in enumerate(clusters):
            frontier = self.build_frontier_info(
                frontier_id=idx,
                cells=cluster_cells,
                score_map=score_map,
                current_pose=current_pose
            )
            frontiers.append(frontier)

        return frontiers


    def build_frontier_candidate_mask(
        self,
        score_map,
        current_pose,
        min_confidence,
        min_distance,
        max_distance,
        min_frontier_score
    ):
        x, y, z, yaw = current_pose

        observed_mask = self.memory.confidence >= min_confidence
        score_mask = score_map >= min_frontier_score
        unvisited_mask = np.logical_not(self.memory.visited)
        boundary_mask = self.get_observed_boundary_mask(observed_mask)

        candidate_mask = observed_mask & score_mask & unvisited_mask & boundary_mask

        for gx in range(self.memory.grid_size):
            for gy in range(self.memory.grid_size):
                if not candidate_mask[gx, gy]:
                    continue

                wx, wy = self.memory.grid_to_world(gx, gy)
                dist = math.hypot(wx - x, wy - y)

                if dist < min_distance or dist > max_distance:
                    candidate_mask[gx, gy] = False

        if not np.any(candidate_mask):
            candidate_mask = observed_mask & score_mask & unvisited_mask

            for gx in range(self.memory.grid_size):
                for gy in range(self.memory.grid_size):
                    if not candidate_mask[gx, gy]:
                        continue

                    wx, wy = self.memory.grid_to_world(gx, gy)
                    dist = math.hypot(wx - x, wy - y)

                    if dist < min_distance or dist > max_distance:
                        candidate_mask[gx, gy] = False

        return candidate_mask


    def get_observed_boundary_mask(self, observed_mask):
        boundary_mask = np.zeros_like(observed_mask, dtype=np.bool_)

        for gx in range(self.memory.grid_size):
            for gy in range(self.memory.grid_size):
                if not observed_mask[gx, gy]:
                    continue

                has_unknown_neighbor = False
                for nx, ny in self.memory.get_neighbors(gx, gy, eight_connected=False):
                    if not observed_mask[nx, ny]:
                        has_unknown_neighbor = True
                        break

                if has_unknown_neighbor:
                    boundary_mask[gx, gy] = True

        return boundary_mask


    def cluster_frontier_cells(
        self,
        candidate_mask,
        score_map,
        min_cluster_size=2,
        semantic_similarity_threshold=0.18,
    ):
        visited_mask = np.zeros_like(candidate_mask, dtype=np.bool_)
        clusters = []

        for gx in range(self.memory.grid_size):
            for gy in range(self.memory.grid_size):
                if not candidate_mask[gx, gy]:
                    continue

                if visited_mask[gx, gy]:
                    continue

                cluster = []
                seed_semantic = float(self.memory.semantic_value[gx, gy])

                queue = deque()
                queue.append((gx, gy))
                visited_mask[gx, gy] = True

                while len(queue) > 0:
                    cx, cy = queue.popleft()
                    cluster.append((cx, cy))

                    for nx, ny in self.memory.get_neighbors(cx, cy, eight_connected=True):
                        if visited_mask[nx, ny]:
                            continue

                        if not candidate_mask[nx, ny]:
                            continue

                        semantic_diff = abs(float(self.memory.semantic_value[nx, ny]) - seed_semantic)
                        if semantic_diff > semantic_similarity_threshold:
                            continue

                        visited_mask[nx, ny] = True
                        queue.append((nx, ny))

                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)

        return clusters


    def build_frontier_info(self, frontier_id, cells, score_map, current_pose):
        x, y, z, yaw = current_pose

        world_positions = []
        planning_scores = []
        semantic_values = []
        confidence_values = []
        safety_values = []
        novelty_values = []
        observe_counts = []

        for gx, gy in cells:
            wx, wy = self.memory.grid_to_world(gx, gy)
            world_positions.append((wx, wy))

            planning_scores.append(float(score_map[gx, gy]))
            semantic_values.append(float(self.memory.semantic_value[gx, gy]))
            confidence_values.append(float(self.memory.confidence[gx, gy]))
            safety_values.append(float(self.memory.safety_value[gx, gy]))
            novelty_values.append(float(self.memory.novelty_value[gx, gy]))
            observe_counts.append(int(self.memory.observe_count[gx, gy]))

        world_positions = np.array(world_positions, dtype=np.float32)

        center_x = float(np.mean(world_positions[:, 0]))
        center_y = float(np.mean(world_positions[:, 1]))

        distance = math.hypot(center_x - x, center_y - y)
        mean_score = float(np.mean(planning_scores))
        max_score = float(np.max(planning_scores))

        mean_semantic = float(np.mean(semantic_values))
        mean_confidence = float(np.mean(confidence_values))
        mean_safety = float(np.mean(safety_values))
        mean_novelty = float(np.mean(novelty_values))
        mean_observe = float(np.mean(observe_counts))

        cluster_size = len(cells)
        size_bonus = min(1.0, math.log(cluster_size + 1.0) / math.log(12.0))
        distance_penalty = min(0.35, distance / max(self.memory.map_size, 1e-6))

        frontier_score = 0.65 * mean_score + 0.35 * max_score
        frontier_score = frontier_score * (0.85 + 0.15 * size_bonus)
        frontier_score = frontier_score - distance_penalty * 0.08
        frontier_score = max(0.0, frontier_score)

        frontier = {
            "valid": True,
            "target_type": "frontier",
            "frontier_id": frontier_id,
            "cells": cells,
            "grid": None,
            "frontier_position": (round(center_x, 2), round(center_y, 2)),
            "position": (round(center_x, 2), round(center_y, 2)),
            "viewpoint_position": (round(center_x, 2), round(center_y, 2)),
            "viewpoint_yaw": 0.0,
            "score": frontier_score,
            "mean_score": mean_score,
            "max_score": max_score,
            "semantic_value": mean_semantic,
            "confidence": mean_confidence,
            "safety_value": mean_safety,
            "novelty_value": mean_novelty,
            "visited": False,
            "observe_count": int(round(mean_observe)),
            "cluster_size": cluster_size,
            "distance": round(distance, 2),
        }

        return frontier