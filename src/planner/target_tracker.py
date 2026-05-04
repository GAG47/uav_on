import math
import numpy as np


class TargetTracker:
    def __init__(
        self,
        candidate_score=0.30,
        confirm_score=0.40,
        stop_score=0.55,
        max_lost_count=3,
        min_area_ratio=0.002,
        max_reasonable_area=0.45,
        min_box_width_ratio=0.025,
        min_box_height_ratio=0.025,
        max_edge_area_ratio=0.035,
        min_candidate_quality=0.14,
        horizontal_fov=90.0,
        candidate_ttl=6,
        max_candidate_count=12,
        max_candidates_for_verification=4,
        verified_target_ttl=12,
        stop_position_distance=5.0,
        navigate_stop_distance=5.0,
        min_navigate_steps_before_stop=1,
        confirm_step_distance=3.0,
        max_association_distance=10.0,
        track_ttl=8,
        max_track_count=12,
        min_track_hits=2,
        max_track_lost_count=3,
        min_track_match_score=0.45,
        stable_track_score=0.42,
        track_position_gate=12.0,
        track_spread_gate=8.0,
    ):
        self.candidate_score = candidate_score
        self.confirm_score = confirm_score
        self.stop_score = stop_score
        self.max_lost_count = max_lost_count

        self.min_area_ratio = min_area_ratio
        self.max_reasonable_area = max_reasonable_area
        self.min_box_width_ratio = min_box_width_ratio
        self.min_box_height_ratio = min_box_height_ratio
        self.max_edge_area_ratio = max_edge_area_ratio
        self.min_candidate_quality = min_candidate_quality

        self.horizontal_fov = horizontal_fov
        self.candidate_ttl = candidate_ttl
        self.max_candidate_count = max_candidate_count
        self.max_candidates_for_verification = max_candidates_for_verification

        self.verified_target_ttl = verified_target_ttl
        self.stop_position_distance = stop_position_distance
        self.navigate_stop_distance = navigate_stop_distance
        self.min_navigate_steps_before_stop = min_navigate_steps_before_stop
        self.confirm_step_distance = confirm_step_distance
        self.max_association_distance = max_association_distance

        self.track_ttl = track_ttl
        self.max_track_count = max_track_count
        self.min_track_hits = min_track_hits
        self.max_track_lost_count = max_track_lost_count
        self.min_track_match_score = min_track_match_score
        self.stable_track_score = stable_track_score
        self.track_position_gate = track_position_gate
        self.track_spread_gate = track_spread_gate

        self.candidate_buffer = []
        self.candidate_counter = 0

        self.target_tracks = []
        self.track_counter = 0
        self.verified_track_id = None

        self.confirm_count = 0
        self.lost_count = 0
        self.navigate_count = 0

        self.last_observation = None
        self.last_horizontal_observation = None
        self.last_planner_target = self.default_planner_target()

        self.verified_target = None
        self.verified_target_position = None
        self.verified_target_score = 0.0
        self.verified_target_step = -1
        self.verified_candidate_id = None

    def reset(self):
        self.candidate_buffer = []
        self.candidate_counter = 0

        self.target_tracks = []
        self.track_counter = 0
        self.verified_track_id = None

        self.confirm_count = 0
        self.lost_count = 0
        self.navigate_count = 0

        self.last_observation = None
        self.last_horizontal_observation = None
        self.last_planner_target = self.default_planner_target()

        self.verified_target = None
        self.verified_target_position = None
        self.verified_target_score = 0.0
        self.verified_target_step = -1
        self.verified_candidate_id = None

    def update(self, grounding_result, current_pose, depth_info=None, step_num=0):
        observations = self.parse_grounding_result(
            grounding_result=grounding_result,
            current_pose=current_pose,
            depth_info=depth_info,
            step_num=step_num
        )

        self.prune_candidate_buffer(step_num=step_num)
        self.mark_tracks_lost(step_num=step_num)

        added_candidates = self.update_candidate_buffer(
            observations=observations,
            current_pose=current_pose,
            step_num=step_num
        )

        self.prune_target_tracks(step_num=step_num)
        self.refresh_verified_target_from_track(step_num=step_num)

        best_observation = self.select_best_observation(observations)
        best_track = self.select_best_track(stable_only=False, step_num=step_num)
        best_stable_track = self.select_best_track(stable_only=True, step_num=step_num)

        verification_candidates = self.get_verification_candidates(
            step_num=step_num
        )

        candidate = len(self.get_recent_candidates(step_num=step_num)) > 0
        stable_track = best_stable_track is not None
        verification_required = (
            stable_track
            and len(verification_candidates) > 0
            and self.verified_target is None
        )

        confirmed = self.has_valid_verified_target(step_num=step_num)

        if confirmed:
            self.lost_count = 0
            self.confirm_count = max(1, self.confirm_count)
            self.navigate_count += 1

            planner_target = self.build_verified_target(
                current_pose=current_pose,
                step_num=step_num
            )

            stop_ready = self.is_stop_ready(
                current_pose=current_pose,
                step_num=step_num
            )

            if stop_ready:
                planner_target = self.build_stop_target(
                    current_pose=current_pose,
                    step_num=step_num
                )

            reason = "verified target object is active"
        else:
            self.navigate_count = 0
            stop_ready = False

            if stable_track:
                self.lost_count = 0
                self.confirm_count = int(best_stable_track.get("stable_step_count", 0))
                planner_target = self.default_planner_target()
                reason = "stable target track collected, waiting for verification"
            elif candidate:
                self.lost_count = 0
                self.confirm_count = 0
                planner_target = self.default_planner_target()
                reason = "task-aware object candidates collected, waiting for stable track"
            else:
                self.lost_count += 1
                self.confirm_count = 0
                planner_target = self.default_planner_target()
                reason = "no reliable task-aware object candidate"

        self.last_observation = best_observation
        if best_observation.get("is_horizontal", False):
            self.last_horizontal_observation = best_observation

        self.last_planner_target = planner_target

        filter_summary = self.build_filter_summary(observations)
        track_summary = self.build_track_summary(step_num=step_num)

        info = {
            "candidate": candidate,
            "horizontal_candidate": self.has_horizontal_candidate(verification_candidates),
            "down_candidate": self.has_down_candidate(observations),
            "consistent": stable_track,
            "same_object": stable_track,
            "stable_track": stable_track,
            "geometric_confirmed": stable_track,
            "confirmed": confirmed,
            "verified": confirmed,
            "verification_required": verification_required,
            "verification": {},
            "stop_ready": stop_ready,
            "geometric_stop_ready": stop_ready,
            "confirm_count": self.confirm_count,
            "required_count": self.min_track_hits,
            "lost_count": self.lost_count,
            "navigate_count": self.navigate_count,
            "stable_target_position": self.verified_target_position,
            "stable_target_score": round(self.verified_target_score, 3),
            "verified_target": self.verified_target,
            "verified_target_position": self.verified_target_position,
            "verified_candidate_id": self.verified_candidate_id,
            "verified_track_id": self.verified_track_id,
            "candidate_buffer_size": len(self.candidate_buffer),
            "added_candidate_count": added_candidates,
            "verification_candidates": verification_candidates,
            "filter_summary": filter_summary,
            "observation_count": len(observations),
            "track_count": len(self.target_tracks),
            "best_track": self.public_track_info(best_track),
            "best_stable_track": self.public_track_info(best_stable_track),
            "track_summary": track_summary,
            "observation": best_observation,
            "planner_target": planner_target,
            "reason": reason,
        }

        return info

    def apply_verification(self, tracker_info, verification_info):
        if tracker_info is None:
            tracker_info = {}
        if verification_info is None:
            verification_info = {}

        self.update_candidate_cache_from_verification(
            verification_info=verification_info
        )

        new_info = dict(tracker_info)
        new_info["verification"] = verification_info

        checked = bool(verification_info.get("checked", False))
        verified = bool(verification_info.get("verified", False))
        selected_candidate_id = verification_info.get("selected_candidate_id", None)

        current_confirmed = bool(tracker_info.get("confirmed", False))
        current_verified = bool(tracker_info.get("verified", False))
        current_stop_ready = bool(tracker_info.get("stop_ready", False))

        planner_target = tracker_info.get("planner_target", None)
        if isinstance(planner_target, dict):
            if planner_target.get("target_type", "") in [
                "verified_target_stop",
                "gdino_stop"
            ]:
                current_stop_ready = True

        if checked and verified and selected_candidate_id is not None:
            candidate = self.find_candidate(selected_candidate_id)
            if candidate is not None:
                track = self.find_track(candidate.get("track_id", None))
                self.set_verified_target(
                    candidate=candidate,
                    verification_info=verification_info,
                    track=track
                )

                current_pose = verification_info.get("current_pose", None)
                if current_pose is None:
                    current_pose = candidate.get("current_pose", None)

                planner_target = self.default_planner_target()
                if current_pose is not None:
                    stop_ready = self.is_stop_ready(
                        current_pose=current_pose,
                        step_num=candidate.get("step_num", 0)
                    )
                    if stop_ready:
                        planner_target = self.build_stop_target(
                            current_pose=current_pose,
                            step_num=candidate.get("step_num", 0)
                        )
                    else:
                        planner_target = self.build_verified_target(
                            current_pose=current_pose,
                            step_num=candidate.get("step_num", 0)
                        )
                else:
                    stop_ready = False

                new_info["candidate"] = True
                new_info["confirmed"] = True
                new_info["verified"] = True
                new_info["stable_track"] = True
                new_info["same_object"] = True
                new_info["geometric_confirmed"] = True
                new_info["stop_ready"] = stop_ready
                new_info["geometric_stop_ready"] = stop_ready
                new_info["planner_target"] = planner_target
                new_info["stable_target_position"] = self.verified_target_position
                new_info["stable_target_score"] = round(self.verified_target_score, 3)
                new_info["verified_target"] = self.verified_target
                new_info["verified_target_position"] = self.verified_target_position
                new_info["verified_candidate_id"] = self.verified_candidate_id
                new_info["verified_track_id"] = self.verified_track_id
                new_info["best_track"] = self.public_track_info(track)
                new_info["reason"] = (
                    "target object verified by task-aware track selection: "
                    + str(verification_info.get("reason", ""))
                )
                return new_info

        if checked and not verified:
            self.mark_verified_candidates_rejected(
                verification_info=verification_info
            )
            self.clear_unverified_state()

            new_info["confirmed"] = False
            new_info["verified"] = False
            new_info["stop_ready"] = False
            new_info["geometric_stop_ready"] = False
            new_info["planner_target"] = self.default_planner_target()
            new_info["stable_target_position"] = self.verified_target_position
            new_info["verified_target"] = self.verified_target
            new_info["verified_target_position"] = self.verified_target_position
            new_info["verified_candidate_id"] = self.verified_candidate_id
            new_info["verified_track_id"] = self.verified_track_id
            new_info["track_summary"] = self.build_track_summary(
                step_num=tracker_info.get("observation", {}).get("step_num", 0)
            )
            new_info["reason"] = (
                "no target object selected by verifier: "
                + str(verification_info.get("reason", ""))
            )
            return new_info

        new_info["confirmed"] = current_confirmed or self.verified_target is not None
        new_info["verified"] = current_verified or self.verified_target is not None
        new_info["stop_ready"] = current_stop_ready
        new_info["geometric_stop_ready"] = current_stop_ready

        if current_stop_ready:
            new_info["reason"] = "reached verified target object position"

        return new_info

    def update_candidate_cache_from_verification(self, verification_info):
        if not isinstance(verification_info, dict):
            return

        candidates = verification_info.get("candidates", [])
        if not isinstance(candidates, list):
            return

        cache_fields = [
            "crop_caption",
            "crop_debug",
            "crop_path",
            "caption_ready",
            "caption_step",
            "caption_source",
        ]

        for verified_candidate in candidates:
            if not isinstance(verified_candidate, dict):
                continue

            candidate_id = verified_candidate.get("candidate_id", None)
            if candidate_id is None:
                continue

            cached_candidate = self.find_candidate(candidate_id)
            if cached_candidate is None:
                continue

            for field in cache_fields:
                if field in verified_candidate:
                    cached_candidate[field] = verified_candidate.get(field)

            crop_caption = cached_candidate.get("crop_caption", "")
            if isinstance(crop_caption, str) and len(crop_caption.strip()) > 0:
                cached_candidate["caption_ready"] = True
                cached_candidate["caption_step"] = verified_candidate.get(
                    "caption_step",
                    cached_candidate.get("step_num", 0)
                )

            crop_debug = cached_candidate.get("crop_debug", {})
            if not isinstance(crop_debug, dict):
                crop_debug = {}

            if self.is_candidate_crop_failed(cached_candidate):
                cached_candidate["verification_reject_count"] = (
                    int(cached_candidate.get("verification_reject_count", 0)) + 1
                )
                cached_candidate["last_reject_reason"] = crop_debug.get(
                    "reason",
                    "invalid or empty crop"
                )
                cached_candidate["last_rejected_step"] = cached_candidate.get(
                    "step_num",
                    0
                )
                self.reject_candidate_track(
                    candidate=cached_candidate,
                    reason=cached_candidate["last_reject_reason"]
                )

    def mark_verified_candidates_rejected(self, verification_info):
        if not isinstance(verification_info, dict):
            return

        candidates = verification_info.get("candidates", [])
        reason = verification_info.get("reason", "")
        reject_reason = verification_info.get("reject_reason", "")

        if not isinstance(candidates, list):
            return

        rejected_track_ids = set()

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            candidate_id = candidate.get("candidate_id", None)
            if candidate_id is None:
                continue

            cached_candidate = self.find_candidate(candidate_id)
            if cached_candidate is None:
                continue

            final_reason = reject_reason if reject_reason else reason

            cached_candidate["verification_reject_count"] = (
                int(cached_candidate.get("verification_reject_count", 0)) + 1
            )
            cached_candidate["last_reject_reason"] = final_reason
            cached_candidate["last_rejected_step"] = candidate.get(
                "step_num",
                cached_candidate.get("step_num", 0)
            )

            track_id = cached_candidate.get("track_id", None)
            if track_id in rejected_track_ids:
                continue

            rejected_track_ids.add(track_id)
            self.reject_candidate_track(
                candidate=cached_candidate,
                reason=final_reason
            )

    def reject_candidate_track(self, candidate, reason=""):
        if not isinstance(candidate, dict):
            return

        track_id = candidate.get("track_id", None)
        track = self.find_track(track_id)
        if track is None:
            return

        track["reject_count"] = int(track.get("reject_count", 0)) + 1
        track["score"] = round(float(track.get("score", 0.0)) * 0.55, 4)
        track["stable"] = False
        track["last_reject_reason"] = reason

    def set_verified_target(self, candidate, verification_info, track=None):
        if track is None:
            track = self.find_track(candidate.get("track_id", None))

        if track is not None:
            candidate_position = track.get("position", None)
        else:
            candidate_position = candidate.get("target_world_position", None)

        if candidate_position is None:
            return

        confidence = float(
            verification_info.get("confidence", candidate.get("score", 0.0))
        )

        if track is not None:
            track["verified"] = True
            track["verified_step"] = int(candidate.get("step_num", 0))
            track["verified_candidate_id"] = candidate.get("candidate_id", None)
            track["verification_confidence"] = confidence
            track["verification_reason"] = verification_info.get("reason", "")
            track["score"] = max(float(track.get("score", 0.0)), confidence)
            self.verified_track_id = track.get("track_id", None)

        if self.verified_target_position is None:
            verified_position = candidate_position
        else:
            distance = self.xy_distance(
                p1=self.verified_target_position,
                p2=candidate_position
            )
            if distance <= self.max_association_distance:
                alpha = max(0.30, min(0.70, confidence))
                vx, vy = self.verified_target_position
                cx, cy = candidate_position
                verified_position = (
                    round(vx * (1.0 - alpha) + cx * alpha, 2),
                    round(vy * (1.0 - alpha) + cy * alpha, 2)
                )
            else:
                verified_position = candidate_position

        self.verified_target_position = verified_position
        self.verified_target_score = max(
            self.verified_target_score * 0.85,
            confidence
        )
        self.verified_target_step = int(candidate.get("step_num", 0))
        self.verified_candidate_id = candidate.get("candidate_id", None)

        self.verified_target = dict(candidate)
        self.verified_target["target_world_position"] = self.verified_target_position
        self.verified_target["verification_confidence"] = confidence
        self.verified_target["verification_reason"] = verification_info.get("reason", "")
        self.verified_target["track_id"] = self.verified_track_id

        if track is not None:
            self.verified_target["track_position"] = track.get("position", None)
            self.verified_target["track_score"] = track.get("score", 0.0)
            self.verified_target["track_hit_count"] = track.get("hit_count", 0)
            self.verified_target["stable_step_count"] = track.get("stable_step_count", 0)

        self.confirm_count = 1
        self.lost_count = 0
        self.navigate_count = 0

    def clear_unverified_state(self):
        self.confirm_count = 0
        self.navigate_count = 0
        self.last_planner_target = self.default_planner_target()

    def has_valid_verified_target(self, step_num):
        if self.verified_target is None:
            return False
        if self.verified_target_position is None:
            return False
        if self.verified_target_step < 0:
            return False

        if step_num - self.verified_target_step > self.verified_target_ttl:
            self.clear_verified_target()
            return False

        return True

    def clear_verified_target(self):
        if self.verified_track_id is not None:
            track = self.find_track(self.verified_track_id)
            if track is not None:
                track["verified"] = False

        self.verified_target = None
        self.verified_target_position = None
        self.verified_target_score = 0.0
        self.verified_target_step = -1
        self.verified_candidate_id = None
        self.verified_track_id = None
        self.navigate_count = 0

    def refresh_verified_target_from_track(self, step_num):
        if self.verified_track_id is None:
            return

        track = self.find_track(self.verified_track_id)
        if track is None:
            return

        if step_num - int(track.get("last_step", 0)) > self.verified_target_ttl:
            self.clear_verified_target()
            return

        track_position = track.get("position", None)
        if track_position is not None:
            self.verified_target_position = track_position

        self.verified_target_score = max(
            float(self.verified_target_score),
            float(track.get("score", 0.0))
        )
        self.verified_target_step = max(
            int(self.verified_target_step),
            int(track.get("last_step", self.verified_target_step))
        )

        if self.verified_target is not None:
            self.verified_target["target_world_position"] = self.verified_target_position
            self.verified_target["track_position"] = track_position
            self.verified_target["track_score"] = track.get("score", 0.0)
            self.verified_target["track_hit_count"] = track.get("hit_count", 0)
            self.verified_target["stable_step_count"] = track.get("stable_step_count", 0)

    def is_stop_ready(self, current_pose, step_num):
        if not self.has_valid_verified_target(step_num=step_num):
            return False

        if self.navigate_count < self.min_navigate_steps_before_stop:
            return False

        current_xy = (float(current_pose[0]), float(current_pose[1]))
        distance = self.xy_distance(
            p1=current_xy,
            p2=self.verified_target_position
        )

        if distance <= self.stop_position_distance:
            return True

        return False

    def build_verified_target(self, current_pose, step_num=0):
        if self.verified_target_position is None:
            return self.default_planner_target()

        current_xy = (float(current_pose[0]), float(current_pose[1]))
        target_xy = self.verified_target_position

        distance = self.xy_distance(
            p1=current_xy,
            p2=target_xy
        )

        target_angle_world = math.degrees(
            math.atan2(target_xy[1] - current_xy[1], target_xy[0] - current_xy[0])
        )
        relative_angle = self.normalize_angle(target_angle_world - float(current_pose[3]))

        move_distance = max(
            2.0,
            min(8.0, distance - self.navigate_stop_distance)
        )
        if distance <= self.navigate_stop_distance:
            move_distance = 2.0

        target_position = self.forward_position(
            current_pose=current_pose,
            target_yaw=target_angle_world,
            move_distance=move_distance
        )

        return {
            "valid": True,
            "target_type": "verified_target_navigate",
            "position": target_position,
            "viewpoint_position": target_position,
            "target_world_position": target_xy,
            "score": float(self.verified_target_score),
            "semantic_value": float(self.verified_target_score),
            "confidence": float(self.verified_target_score),
            "safety_value": 0.8,
            "novelty_value": 0.5,
            "distance": round(move_distance, 2),
            "relative_angle": round(relative_angle, 2),
            "relative_region": self.get_relative_region(relative_angle),
            "bbox": self.verified_target.get("bbox", None) if self.verified_target else None,
            "area_ratio": self.verified_target.get("area_ratio", 0.0) if self.verified_target else 0.0,
            "estimated_depth": self.verified_target.get("estimated_depth", None) if self.verified_target else None,
            "candidate_id": self.verified_candidate_id,
            "track_id": self.verified_track_id,
            "stop_reason": "",
        }

    def build_stop_target(self, current_pose, step_num=0):
        current_xy = (round(float(current_pose[0]), 2), round(float(current_pose[1]), 2))

        return {
            "valid": True,
            "target_type": "verified_target_stop",
            "position": current_xy,
            "viewpoint_position": current_xy,
            "target_world_position": self.verified_target_position,
            "score": float(self.verified_target_score),
            "semantic_value": float(self.verified_target_score),
            "confidence": float(self.verified_target_score),
            "safety_value": 0.8,
            "novelty_value": 0.5,
            "distance": 0.0,
            "relative_angle": 0.0,
            "relative_region": "front",
            "bbox": self.verified_target.get("bbox", None) if self.verified_target else None,
            "area_ratio": self.verified_target.get("area_ratio", 0.0) if self.verified_target else 0.0,
            "estimated_depth": self.verified_target.get("estimated_depth", None) if self.verified_target else None,
            "candidate_id": self.verified_candidate_id,
            "track_id": self.verified_track_id,
            "stop_reason": "reached verified target object position",
        }

    def parse_grounding_result(self, grounding_result, current_pose, depth_info=None, step_num=0):
        if grounding_result is None:
            return []

        if not grounding_result.get("available", False):
            return []

        detections = grounding_result.get("detections", [])
        if not isinstance(detections, list) or len(detections) == 0:
            best_detection = grounding_result.get("best_detection", None)
            if best_detection is not None:
                detections = [best_detection]

        if not isinstance(detections, list) or len(detections) == 0:
            return []

        observations = []
        for detection_index, detection in enumerate(detections):
            observation = self.parse_detection(
                detection=detection,
                current_pose=current_pose,
                depth_info=depth_info,
                step_num=step_num,
                detection_index=detection_index
            )
            if observation.get("valid", False):
                observations.append(observation)

        observations.sort(
            key=lambda item: item.get("quality_score", 0.0),
            reverse=True
        )

        return observations

    def parse_detection(self, detection, current_pose, depth_info=None, step_num=0, detection_index=0):
        observation = self.default_observation(step_num=step_num)

        if detection is None:
            observation["reason"] = "detection is None"
            return observation

        bbox = detection.get("bbox", None)
        image_width = int(detection.get("image_width", 0))
        image_height = int(detection.get("image_height", 0))

        if bbox is None or len(bbox) != 4 or image_width <= 0 or image_height <= 0:
            observation["reason"] = "invalid bbox"
            return observation

        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0.0, min(float(image_width), x1))
        x2 = max(0.0, min(float(image_width), x2))
        y1 = max(0.0, min(float(image_height), y1))
        y2 = max(0.0, min(float(image_height), y2))

        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)

        if box_width <= 1.0 or box_height <= 1.0:
            observation["reason"] = "empty bbox"
            return observation

        area_ratio = (box_width * box_height) / max(1.0, float(image_width * image_height))
        width_ratio = box_width / max(1.0, float(image_width))
        height_ratio = box_height / max(1.0, float(image_height))

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        center_offset_x = (cx - image_width / 2.0) / max(1.0, image_width / 2.0)
        center_offset_y = (cy - image_height / 2.0) / max(1.0, image_height / 2.0)

        image_index = int(detection.get("image_index", 0))
        camera_region = self.get_camera_region(image_index)

        is_horizontal = camera_region in ["front", "left", "right"]
        is_downward = camera_region == "down"

        score = float(detection.get("score", 0.0))

        border_info = self.get_border_info(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            width=image_width,
            height=image_height
        )
        touches_border_count = border_info["count"]

        abnormal_large_box = area_ratio >= self.max_reasonable_area
        full_frame_like_box = area_ratio >= 0.75 or touches_border_count >= 3
        edge_like_box = self.is_edge_like_box(
            area_ratio=area_ratio,
            width_ratio=width_ratio,
            height_ratio=height_ratio,
            border_info=border_info
        )

        estimated_depth = self.estimate_depth(
            depth_info=depth_info,
            image_index=image_index
        )

        if is_horizontal:
            camera_yaw_offset = self.get_camera_yaw_offset(camera_region)
            bbox_yaw_offset = center_offset_x * (self.horizontal_fov / 2.0)
            relative_angle = self.normalize_angle(camera_yaw_offset + bbox_yaw_offset)
            relative_region = self.get_relative_region(relative_angle)
            target_world_position = self.estimate_world_position(
                current_pose=current_pose,
                relative_angle=relative_angle,
                estimated_depth=estimated_depth
            )
        else:
            camera_yaw_offset = 0.0
            bbox_yaw_offset = 0.0
            relative_angle = 0.0
            relative_region = "down"
            target_world_position = None

        quality_score = self.compute_detection_quality(
            score=score,
            area_ratio=area_ratio,
            width_ratio=width_ratio,
            height_ratio=height_ratio,
            is_horizontal=is_horizontal,
            is_downward=is_downward,
            full_frame_like_box=full_frame_like_box,
            abnormal_large_box=abnormal_large_box,
            edge_like_box=edge_like_box,
            center_offset_x=center_offset_x,
            center_offset_y=center_offset_y,
        )

        observation.update({
            "valid": True,
            "step_num": step_num,
            "detection_index": detection_index,
            "score": score,
            "phrase": detection.get("phrase", ""),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "bbox_norm_cxcywh": detection.get("bbox_norm_cxcywh", None),
            "image_index": image_index,
            "image_width": image_width,
            "image_height": image_height,
            "area_ratio": round(area_ratio, 4),
            "width_ratio": round(width_ratio, 4),
            "height_ratio": round(height_ratio, 4),
            "center": (round(cx, 2), round(cy, 2)),
            "center_offset_x": round(center_offset_x, 4),
            "center_offset_y": round(center_offset_y, 4),
            "camera_region": camera_region,
            "is_horizontal": is_horizontal,
            "is_downward": is_downward,
            "camera_yaw_offset": round(camera_yaw_offset, 2),
            "bbox_yaw_offset": round(bbox_yaw_offset, 2),
            "relative_angle": round(relative_angle, 2),
            "relative_region": relative_region,
            "estimated_depth": None if estimated_depth is None else round(estimated_depth, 2),
            "target_world_position": target_world_position,
            "quality_score": round(quality_score, 4),
            "touches_border_count": touches_border_count,
            "touches_left": border_info["left"],
            "touches_top": border_info["top"],
            "touches_right": border_info["right"],
            "touches_bottom": border_info["bottom"],
            "abnormal_large_box": abnormal_large_box,
            "full_frame_like_box": full_frame_like_box,
            "edge_like_box": edge_like_box,
            "candidate_filter_reason": "",
            "candidate_valid": False,
            "reason": "ok"
        })

        filter_reason = self.get_candidate_filter_reason(observation)
        observation["candidate_filter_reason"] = filter_reason
        observation["candidate_valid"] = filter_reason == "ok"

        return observation

    def update_candidate_buffer(self, observations, current_pose, step_num):
        added_count = 0

        for observation in observations:
            if not self.is_task_aware_candidate(observation):
                continue

            candidate = self.build_candidate_from_observation(
                observation=observation,
                current_pose=current_pose
            )

            track = self.associate_candidate_to_track(
                candidate=candidate,
                step_num=step_num
            )
            if track is not None:
                candidate["track_id"] = track.get("track_id", None)
                candidate["track_score"] = track.get("score", 0.0)
                candidate["track_hit_count"] = track.get("hit_count", 0)
                candidate["stable_step_count"] = track.get("stable_step_count", 0)
                candidate["track_stable"] = track.get("stable", False)

            self.candidate_buffer.append(candidate)
            added_count += 1

        self.candidate_buffer.sort(
            key=lambda item: item.get("candidate_quality", 0.0),
            reverse=True
        )

        if len(self.candidate_buffer) > self.max_candidate_count:
            self.candidate_buffer = self.candidate_buffer[:self.max_candidate_count]

        return added_count

    def is_task_aware_candidate(self, observation):
        return self.get_candidate_filter_reason(observation) == "ok"

    def get_candidate_filter_reason(self, observation):
        if not observation.get("valid", False):
            return "invalid_detection"

        if not observation.get("is_horizontal", False):
            return "non_horizontal_view"

        if observation.get("score", 0.0) < self.candidate_score:
            return "low_gdino_score"

        if observation.get("area_ratio", 0.0) < self.min_area_ratio:
            return "box_area_too_small"

        if observation.get("width_ratio", 0.0) < self.min_box_width_ratio:
            return "box_width_too_small"

        if observation.get("height_ratio", 0.0) < self.min_box_height_ratio:
            return "box_height_too_small"

        if observation.get("full_frame_like_box", False):
            return "full_frame_like_box"

        if observation.get("abnormal_large_box", False):
            return "abnormal_large_box"

        if observation.get("edge_like_box", False):
            return "edge_like_box"

        if observation.get("target_world_position", None) is None:
            return "no_target_world_position"

        if observation.get("quality_score", 0.0) < self.min_candidate_quality:
            return "low_candidate_quality"

        return "ok"

    def build_candidate_from_observation(self, observation, current_pose):
        self.candidate_counter += 1
        candidate_id = "cand_{:06d}".format(self.candidate_counter)

        candidate = {
            "candidate_id": candidate_id,
            "track_id": None,
            "step_num": int(observation.get("step_num", 0)),
            "detection_index": int(observation.get("detection_index", 0)),
            "image_index": int(observation.get("image_index", -1)),
            "camera_region": observation.get("camera_region", "unknown"),
            "phrase": observation.get("phrase", ""),
            "score": float(observation.get("score", 0.0)),
            "bbox": observation.get("bbox", None),
            "bbox_norm_cxcywh": observation.get("bbox_norm_cxcywh", None),
            "image_width": int(observation.get("image_width", 0)),
            "image_height": int(observation.get("image_height", 0)),
            "area_ratio": float(observation.get("area_ratio", 0.0)),
            "width_ratio": float(observation.get("width_ratio", 0.0)),
            "height_ratio": float(observation.get("height_ratio", 0.0)),
            "center": observation.get("center", None),
            "center_offset_x": float(observation.get("center_offset_x", 0.0)),
            "center_offset_y": float(observation.get("center_offset_y", 0.0)),
            "estimated_depth": observation.get("estimated_depth", None),
            "target_world_position": observation.get("target_world_position", None),
            "relative_angle": float(observation.get("relative_angle", 0.0)),
            "relative_region": observation.get("relative_region", "front"),
            "quality_score": float(observation.get("quality_score", 0.0)),
            "candidate_quality": self.compute_candidate_quality(observation),
            "full_frame_like_box": bool(observation.get("full_frame_like_box", False)),
            "abnormal_large_box": bool(observation.get("abnormal_large_box", False)),
            "edge_like_box": bool(observation.get("edge_like_box", False)),
            "touches_border_count": int(observation.get("touches_border_count", 0)),
            "touches_left": bool(observation.get("touches_left", False)),
            "touches_top": bool(observation.get("touches_top", False)),
            "touches_right": bool(observation.get("touches_right", False)),
            "touches_bottom": bool(observation.get("touches_bottom", False)),
            "candidate_filter_reason": observation.get("candidate_filter_reason", "ok"),
            "current_pose": list(current_pose),
            "crop_caption": "",
            "crop_debug": {},
            "crop_path": "",
            "caption_ready": False,
            "caption_step": -1,
            "caption_source": "",
            "verification_reject_count": 0,
            "last_reject_reason": "",
            "last_rejected_step": -1,
        }

        return candidate

    def compute_candidate_quality(self, observation):
        quality = float(observation.get("quality_score", 0.0))
        score = float(observation.get("score", 0.0))
        area_ratio = float(observation.get("area_ratio", 0.0))
        width_ratio = float(observation.get("width_ratio", 0.0))
        height_ratio = float(observation.get("height_ratio", 0.0))
        center_offset_x = abs(float(observation.get("center_offset_x", 0.0)))

        if observation.get("camera_region", "unknown") == "front":
            quality += 0.10

        if center_offset_x <= 0.50:
            quality += 0.05

        if 0.006 <= area_ratio <= 0.20:
            quality += 0.05

        if width_ratio < self.min_box_width_ratio or height_ratio < self.min_box_height_ratio:
            quality *= 0.20

        if area_ratio > 0.25 and score < 0.60:
            quality *= 0.45

        if observation.get("edge_like_box", False):
            quality *= 0.20

        if observation.get("abnormal_large_box", False):
            quality *= 0.10

        if observation.get("full_frame_like_box", False):
            quality *= 0.03

        return round(max(0.0, quality), 4)

    def associate_candidate_to_track(self, candidate, step_num):
        best_track = None
        best_score = 0.0

        for track in self.target_tracks:
            if step_num - int(track.get("last_step", 0)) > self.track_ttl:
                continue
            if int(track.get("reject_count", 0)) >= 2:
                continue

            match_score = self.compute_track_match_score(
                track=track,
                candidate=candidate,
                step_num=step_num
            )

            if match_score > best_score:
                best_score = match_score
                best_track = track

        if best_track is not None and best_score >= self.min_track_match_score:
            self.update_track(
                track=best_track,
                candidate=candidate,
                match_score=best_score,
                step_num=step_num
            )
            return best_track

        return self.create_track(
            candidate=candidate,
            step_num=step_num
        )

    def compute_track_match_score(self, track, candidate, step_num):
        track_position = track.get("position", None)
        candidate_position = candidate.get("target_world_position", None)

        if track_position is None or candidate_position is None:
            return 0.0

        distance = self.xy_distance(track_position, candidate_position)
        if distance > self.track_position_gate:
            position_score = 0.0
        else:
            position_score = max(0.0, 1.0 - distance / self.track_position_gate)

        last_angle = float(track.get("last_relative_angle", 0.0))
        candidate_angle = float(candidate.get("relative_angle", 0.0))
        angle_diff = abs(self.normalize_angle(candidate_angle - last_angle))
        angle_score = max(0.0, 1.0 - angle_diff / 120.0)

        bbox_score = self.compute_bbox_center_score(
            track=track,
            candidate=candidate
        )

        region_score = self.compute_region_score(
            track=track,
            candidate=candidate
        )

        phrase_score = self.compute_phrase_score(
            track=track,
            candidate=candidate
        )

        age = max(0, step_num - int(track.get("last_step", 0)))
        age_penalty = min(0.25, 0.04 * age)

        score = (
            0.55 * position_score
            + 0.15 * angle_score
            + 0.15 * bbox_score
            + 0.10 * region_score
            + 0.05 * phrase_score
            - age_penalty
        )

        return round(max(0.0, score), 4)

    def compute_bbox_center_score(self, track, candidate):
        track_center = track.get("last_bbox_center", None)
        candidate_center = candidate.get("center", None)

        if track_center is None or candidate_center is None:
            return 0.5

        track_width = max(1.0, float(track.get("image_width", 1)))
        track_height = max(1.0, float(track.get("image_height", 1)))
        candidate_width = max(1.0, float(candidate.get("image_width", 1)))
        candidate_height = max(1.0, float(candidate.get("image_height", 1)))

        tx = float(track_center[0]) / track_width
        ty = float(track_center[1]) / track_height
        cx = float(candidate_center[0]) / candidate_width
        cy = float(candidate_center[1]) / candidate_height

        distance = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
        return max(0.0, 1.0 - distance / 0.75)

    def compute_region_score(self, track, candidate):
        track_region = track.get("last_camera_region", "unknown")
        candidate_region = candidate.get("camera_region", "unknown")

        if track_region == candidate_region:
            return 1.0

        horizontal_regions = ["front", "left", "right"]
        if track_region in horizontal_regions and candidate_region in horizontal_regions:
            return 0.65

        return 0.2

    def compute_phrase_score(self, track, candidate):
        track_phrase = str(track.get("last_phrase", "")).lower().strip()
        candidate_phrase = str(candidate.get("phrase", "")).lower().strip()

        if len(track_phrase) == 0 or len(candidate_phrase) == 0:
            return 0.5

        if track_phrase == candidate_phrase:
            return 1.0

        if track_phrase in candidate_phrase or candidate_phrase in track_phrase:
            return 0.8

        return 0.3

    def create_track(self, candidate, step_num):
        self.track_counter += 1
        track_id = "trk_{:06d}".format(self.track_counter)

        position = candidate.get("target_world_position", None)
        candidate_quality = float(candidate.get("candidate_quality", 0.0))

        track = {
            "track_id": track_id,
            "first_step": int(step_num),
            "last_step": int(step_num),
            "observed_steps": [int(step_num)],
            "stable_step_count": 1,
            "hit_count": 1,
            "same_step_hit_count": 1,
            "last_update_step": int(step_num),
            "lost_count": 0,
            "score": round(candidate_quality, 4),
            "stable": False,
            "verified": False,
            "verified_step": -1,
            "verified_candidate_id": None,
            "verification_confidence": 0.0,
            "verification_reason": "",
            "reject_count": 0,
            "last_reject_reason": "",
            "position": position,
            "last_position": position,
            "position_history": [position] if position is not None else [],
            "position_spread": 0.0,
            "candidate_ids": [candidate.get("candidate_id", None)],
            "best_candidate_id": candidate.get("candidate_id", None),
            "best_candidate_quality": candidate_quality,
            "last_candidate_id": candidate.get("candidate_id", None),
            "last_bbox_center": candidate.get("center", None),
            "bbox_centers": [candidate.get("center", None)],
            "image_width": int(candidate.get("image_width", 0)),
            "image_height": int(candidate.get("image_height", 0)),
            "last_relative_angle": float(candidate.get("relative_angle", 0.0)),
            "relative_angles": [float(candidate.get("relative_angle", 0.0))],
            "last_camera_region": candidate.get("camera_region", "unknown"),
            "camera_regions": [candidate.get("camera_region", "unknown")],
            "last_phrase": candidate.get("phrase", ""),
            "phrases": [candidate.get("phrase", "")],
            "last_score": float(candidate.get("score", 0.0)),
            "last_area_ratio": float(candidate.get("area_ratio", 0.0)),
        }

        self.target_tracks.append(track)
        candidate["track_id"] = track_id

        return track

    def update_track(self, track, candidate, match_score, step_num):
        old_position = track.get("position", None)
        new_position = candidate.get("target_world_position", None)
        candidate_quality = float(candidate.get("candidate_quality", 0.0))

        is_new_step = int(track.get("last_update_step", -1)) != int(step_num)

        if old_position is None:
            fused_position = new_position
        elif new_position is None:
            fused_position = old_position
        elif is_new_step:
            alpha = 0.30 + 0.35 * min(1.0, max(0.0, candidate_quality))
            ox, oy = old_position
            nx, ny = new_position
            fused_position = (
                round(ox * (1.0 - alpha) + nx * alpha, 2),
                round(oy * (1.0 - alpha) + ny * alpha, 2)
            )
        else:
            old_quality = float(track.get("best_candidate_quality", 0.0))
            if candidate_quality > old_quality:
                alpha = 0.20
                ox, oy = old_position
                nx, ny = new_position
                fused_position = (
                    round(ox * (1.0 - alpha) + nx * alpha, 2),
                    round(oy * (1.0 - alpha) + ny * alpha, 2)
                )
            else:
                fused_position = old_position

        observed_steps = track.get("observed_steps", [])
        if int(step_num) not in observed_steps:
            observed_steps.append(int(step_num))
        observed_steps = sorted(list(set(observed_steps)))[-8:]

        track["observed_steps"] = observed_steps
        track["stable_step_count"] = len(observed_steps)

        if is_new_step:
            track["same_step_hit_count"] = 1
        else:
            track["same_step_hit_count"] = int(track.get("same_step_hit_count", 0)) + 1

        track["last_update_step"] = int(step_num)
        track["last_position"] = old_position
        track["position"] = fused_position
        track["last_step"] = int(step_num)
        track["hit_count"] = int(track.get("hit_count", 0)) + 1
        track["lost_count"] = 0
        track["last_candidate_id"] = candidate.get("candidate_id", None)
        track["last_bbox_center"] = candidate.get("center", None)
        track["image_width"] = int(candidate.get("image_width", 0))
        track["image_height"] = int(candidate.get("image_height", 0))
        track["last_relative_angle"] = float(candidate.get("relative_angle", 0.0))
        track["last_camera_region"] = candidate.get("camera_region", "unknown")
        track["last_phrase"] = candidate.get("phrase", "")
        track["last_score"] = float(candidate.get("score", 0.0))
        track["last_area_ratio"] = float(candidate.get("area_ratio", 0.0))

        candidate_ids = track.get("candidate_ids", [])
        candidate_ids.append(candidate.get("candidate_id", None))
        track["candidate_ids"] = candidate_ids[-8:]

        bbox_centers = track.get("bbox_centers", [])
        bbox_centers.append(candidate.get("center", None))
        track["bbox_centers"] = bbox_centers[-8:]

        camera_regions = track.get("camera_regions", [])
        camera_regions.append(candidate.get("camera_region", "unknown"))
        track["camera_regions"] = camera_regions[-8:]

        phrases = track.get("phrases", [])
        phrases.append(candidate.get("phrase", ""))
        track["phrases"] = phrases[-8:]

        relative_angles = track.get("relative_angles", [])
        relative_angles.append(float(candidate.get("relative_angle", 0.0)))
        track["relative_angles"] = relative_angles[-8:]

        position_history = track.get("position_history", [])
        if fused_position is not None and is_new_step:
            position_history.append(fused_position)
        elif fused_position is not None and len(position_history) == 0:
            position_history.append(fused_position)
        track["position_history"] = position_history[-8:]
        track["position_spread"] = self.compute_position_spread(
            position_history=track["position_history"]
        )

        old_score = float(track.get("score", 0.0))
        stable_step_count = int(track.get("stable_step_count", 1))
        hit_bonus = min(0.12, 0.04 * stable_step_count)
        same_step_penalty = min(0.10, 0.015 * max(0, int(track.get("same_step_hit_count", 1)) - 1))

        track["score"] = round(
            min(
                1.0,
                max(
                    old_score * 0.88 + candidate_quality * 0.20 + hit_bonus - same_step_penalty,
                    candidate_quality * 0.85
                )
            ),
            4
        )

        if candidate_quality >= float(track.get("best_candidate_quality", 0.0)):
            track["best_candidate_id"] = candidate.get("candidate_id", None)
            track["best_candidate_quality"] = candidate_quality

        track["stable"] = self.is_track_stable(track)
        candidate["track_id"] = track.get("track_id", None)
        candidate["track_match_score"] = match_score

    def mark_tracks_lost(self, step_num):
        for track in self.target_tracks:
            if int(track.get("last_step", -1)) < int(step_num):
                track["lost_count"] = int(track.get("lost_count", 0)) + 1

    def prune_target_tracks(self, step_num):
        new_tracks = []
        for track in self.target_tracks:
            age = step_num - int(track.get("last_step", 0))
            if age > self.track_ttl:
                continue

            if int(track.get("lost_count", 0)) > self.max_track_lost_count:
                continue

            if int(track.get("reject_count", 0)) >= 3:
                continue

            new_tracks.append(track)

        new_tracks.sort(
            key=lambda item: (
                bool(item.get("verified", False)),
                bool(item.get("stable", False)),
                int(item.get("stable_step_count", 0)),
                float(item.get("score", 0.0)),
                int(item.get("hit_count", 0))
            ),
            reverse=True
        )

        if len(new_tracks) > self.max_track_count:
            new_tracks = new_tracks[:self.max_track_count]

        self.target_tracks = new_tracks

    def is_track_stable(self, track):
        if track is None:
            return False

        if int(track.get("reject_count", 0)) >= 2:
            return False

        stable_step_count = int(track.get("stable_step_count", 0))
        if stable_step_count < self.min_track_hits:
            return False

        first_step = int(track.get("first_step", 0))
        last_step = int(track.get("last_step", 0))
        if last_step <= first_step:
            return False

        if int(track.get("lost_count", 0)) > 1:
            return False

        if float(track.get("score", 0.0)) < self.stable_track_score:
            return False

        if track.get("position", None) is None:
            return False

        if float(track.get("position_spread", 0.0)) > self.track_spread_gate:
            return False

        return True

    def compute_position_spread(self, position_history):
        if not isinstance(position_history, list):
            return 0.0

        valid_positions = []
        for position in position_history:
            if position is None:
                continue
            valid_positions.append(position)

        if len(valid_positions) <= 1:
            return 0.0

        xs = [float(position[0]) for position in valid_positions]
        ys = [float(position[1]) for position in valid_positions]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        max_distance = 0.0
        for position in valid_positions:
            distance = self.xy_distance(
                p1=(cx, cy),
                p2=position
            )
            max_distance = max(max_distance, distance)

        return round(max_distance, 3)

    def select_best_track(self, stable_only=False, step_num=0):
        candidates = []
        for track in self.target_tracks:
            if stable_only and not bool(track.get("stable", False)):
                continue

            if step_num > 0:
                if step_num - int(track.get("last_step", 0)) > self.track_ttl:
                    continue

            if int(track.get("reject_count", 0)) >= 2:
                continue

            candidates.append(track)

        if len(candidates) == 0:
            return None

        candidates.sort(
            key=lambda item: (
                bool(item.get("verified", False)),
                bool(item.get("stable", False)),
                int(item.get("stable_step_count", 0)),
                float(item.get("score", 0.0)),
                -int(item.get("lost_count", 0))
            ),
            reverse=True
        )

        return candidates[0]

    def find_track(self, track_id):
        if track_id is None:
            return None

        for track in self.target_tracks:
            if track.get("track_id", None) == track_id:
                return track

        return None

    def find_track_by_candidate_id(self, candidate_id):
        if candidate_id is None:
            return None

        for track in self.target_tracks:
            candidate_ids = track.get("candidate_ids", [])
            if candidate_id in candidate_ids:
                return track

        return None

    def public_track_info(self, track):
        if track is None:
            return None

        return {
            "track_id": track.get("track_id", None),
            "stable": bool(track.get("stable", False)),
            "verified": bool(track.get("verified", False)),
            "score": round(float(track.get("score", 0.0)), 3),
            "hit_count": int(track.get("hit_count", 0)),
            "stable_step_count": int(track.get("stable_step_count", 0)),
            "same_step_hit_count": int(track.get("same_step_hit_count", 0)),
            "lost_count": int(track.get("lost_count", 0)),
            "reject_count": int(track.get("reject_count", 0)),
            "position": track.get("position", None),
            "position_spread": round(float(track.get("position_spread", 0.0)), 3),
            "observed_steps": track.get("observed_steps", []),
            "best_candidate_id": track.get("best_candidate_id", None),
            "last_candidate_id": track.get("last_candidate_id", None),
            "last_camera_region": track.get("last_camera_region", "unknown"),
            "last_phrase": track.get("last_phrase", ""),
        }

    def build_track_summary(self, step_num=0):
        summary = {
            "total": len(self.target_tracks),
            "stable": 0,
            "verified": 0,
            "rejected": 0,
            "best": None,
        }

        best_track = self.select_best_track(stable_only=False, step_num=step_num)
        summary["best"] = self.public_track_info(best_track)

        for track in self.target_tracks:
            if bool(track.get("stable", False)):
                summary["stable"] += 1
            if bool(track.get("verified", False)):
                summary["verified"] += 1
            if int(track.get("reject_count", 0)) > 0:
                summary["rejected"] += 1

        return summary

    def get_recent_candidates(self, step_num):
        candidates = []
        for candidate in self.candidate_buffer:
            if step_num - int(candidate.get("step_num", 0)) <= self.candidate_ttl:
                candidates.append(candidate)
        return candidates

    def get_verification_candidates(self, step_num):
        candidates = self.get_recent_candidates(step_num=step_num)
        representative_candidates = []

        stable_tracks = []
        for track in self.target_tracks:
            if not bool(track.get("stable", False)):
                continue
            if int(track.get("reject_count", 0)) >= 2:
                continue
            if step_num - int(track.get("last_step", 0)) > self.track_ttl:
                continue
            stable_tracks.append(track)

        stable_tracks.sort(
            key=lambda item: (
                bool(item.get("verified", False)),
                int(item.get("stable_step_count", 0)),
                float(item.get("score", 0.0))
            ),
            reverse=True
        )

        used_candidate_ids = set()
        used_track_ids = set()

        for track in stable_tracks:
            track_id = track.get("track_id", None)
            if track_id in used_track_ids:
                continue

            representative = self.select_representative_candidate_for_track(
                track=track,
                candidates=candidates,
                step_num=step_num
            )
            if representative is None:
                continue

            candidate_id = representative.get("candidate_id", None)
            if candidate_id in used_candidate_ids:
                continue

            used_candidate_ids.add(candidate_id)
            used_track_ids.add(track_id)
            representative_candidates.append(representative)

        representative_candidates.sort(
            key=lambda item: item.get("verification_priority", 0.0),
            reverse=True
        )

        return representative_candidates[:self.max_candidates_for_verification]

    def select_representative_candidate_for_track(self, track, candidates, step_num):
        track_id = track.get("track_id", None)
        track_candidates = []

        for candidate in candidates:
            if candidate.get("track_id", None) != track_id:
                continue

            age = max(0, step_num - int(candidate.get("step_num", 0)))
            reject_count = int(candidate.get("verification_reject_count", 0))
            caption_ready = bool(candidate.get("caption_ready", False))

            if self.is_candidate_crop_failed(candidate):
                continue

            if age > 0 and not caption_ready:
                continue

            if reject_count >= 2 and age > 0:
                continue

            if reject_count >= 3:
                continue

            if candidate.get("edge_like_box", False):
                continue

            priority = self.compute_verification_priority(
                candidate=candidate,
                step_num=step_num
            )
            priority += 0.20 * min(1.0, float(track.get("score", 0.0)))
            priority += 0.08 * min(3, int(track.get("stable_step_count", 0)))

            candidate = dict(candidate)
            candidate["verification_priority"] = round(priority, 4)
            candidate["track_id"] = track_id
            candidate["track_score"] = round(float(track.get("score", 0.0)), 4)
            candidate["track_hit_count"] = int(track.get("hit_count", 0))
            candidate["stable_step_count"] = int(track.get("stable_step_count", 0))
            candidate["track_position"] = track.get("position", None)
            candidate["track_stable"] = bool(track.get("stable", False))
            candidate["track_verified"] = bool(track.get("verified", False))
            track_candidates.append(candidate)

        if len(track_candidates) == 0:
            return None

        track_candidates.sort(
            key=lambda item: item.get("verification_priority", 0.0),
            reverse=True
        )

        return track_candidates[0]

    def compute_verification_priority(self, candidate, step_num):
        priority = float(candidate.get("candidate_quality", 0.0))
        age = max(0, step_num - int(candidate.get("step_num", 0)))
        reject_count = int(candidate.get("verification_reject_count", 0))

        if int(candidate.get("step_num", -1)) == step_num:
            priority += 0.25

        if bool(candidate.get("caption_ready", False)):
            priority += 0.12

        if candidate.get("camera_region", "unknown") == "front":
            priority += 0.08

        if abs(float(candidate.get("center_offset_x", 0.0))) <= 0.50:
            priority += 0.04

        if candidate.get("edge_like_box", False):
            priority -= 0.30

        priority -= 0.05 * age
        priority -= 0.35 * reject_count

        return round(max(0.0, priority), 4)

    def prune_candidate_buffer(self, step_num):
        new_buffer = []
        for candidate in self.candidate_buffer:
            age = step_num - int(candidate.get("step_num", 0))
            if age > self.candidate_ttl:
                continue

            reject_count = int(candidate.get("verification_reject_count", 0))
            if reject_count >= 3:
                continue

            if self.is_candidate_crop_failed(candidate):
                continue

            if candidate.get("edge_like_box", False):
                continue

            track_id = candidate.get("track_id", None)
            if track_id is not None and self.find_track(track_id) is None:
                continue

            new_buffer.append(candidate)

        self.candidate_buffer = new_buffer

    def is_candidate_crop_failed(self, candidate):
        if not isinstance(candidate, dict):
            return False

        crop_debug = candidate.get("crop_debug", {})
        if not isinstance(crop_debug, dict) or len(crop_debug) == 0:
            return False

        if crop_debug.get("ok", True) is False:
            return True

        crop_caption = candidate.get("crop_caption", "")
        caption_ready = bool(candidate.get("caption_ready", False))

        if caption_ready:
            if not isinstance(crop_caption, str):
                return True
            if len(crop_caption.strip()) == 0:
                return True

        return False

    def find_candidate(self, candidate_id):
        for candidate in self.candidate_buffer:
            if candidate.get("candidate_id", None) == candidate_id:
                return candidate

        if self.verified_target is not None:
            if self.verified_target.get("candidate_id", None) == candidate_id:
                return self.verified_target

        return None

    def has_horizontal_candidate(self, candidates):
        for candidate in candidates:
            if candidate.get("camera_region", "unknown") in ["front", "left", "right"]:
                return True
        return False

    def has_down_candidate(self, observations):
        for observation in observations:
            if observation.get("camera_region", "unknown") == "down":
                if observation.get("score", 0.0) >= self.confirm_score:
                    return True
        return False

    def select_best_observation(self, observations):
        if len(observations) == 0:
            return self.default_observation()

        valid_candidates = []
        for observation in observations:
            if observation.get("candidate_valid", False):
                valid_candidates.append(observation)

        if len(valid_candidates) > 0:
            valid_candidates.sort(
                key=lambda item: item.get("quality_score", 0.0),
                reverse=True
            )
            return valid_candidates[0]

        return observations[0]

    def compute_detection_quality(
        self,
        score,
        area_ratio,
        width_ratio,
        height_ratio,
        is_horizontal,
        is_downward,
        full_frame_like_box,
        abnormal_large_box,
        edge_like_box,
        center_offset_x,
        center_offset_y,
    ):
        quality = float(score)

        if area_ratio < self.min_area_ratio:
            quality *= 0.10

        if width_ratio < self.min_box_width_ratio:
            quality *= 0.20

        if height_ratio < self.min_box_height_ratio:
            quality *= 0.20

        if abnormal_large_box:
            quality *= 0.10

        if full_frame_like_box:
            quality *= 0.03

        if edge_like_box:
            quality *= 0.20

        if is_downward:
            quality *= 0.30

        if not is_horizontal and not is_downward:
            quality *= 0.20

        if abs(center_offset_x) > 0.90:
            quality *= 0.70

        if abs(center_offset_y) > 0.95:
            quality *= 0.70

        return quality

    def build_filter_summary(self, observations):
        summary = {
            "total": len(observations),
            "accepted": 0,
            "rejected": 0,
            "reasons": {},
        }

        for observation in observations:
            reason = observation.get("candidate_filter_reason", "unknown")
            if reason == "ok":
                summary["accepted"] += 1
            else:
                summary["rejected"] += 1
            summary["reasons"][reason] = summary["reasons"].get(reason, 0) + 1

        return summary

    def get_border_info(self, x1, y1, x2, y2, width, height):
        eps_x = max(2.0, float(width) * 0.02)
        eps_y = max(2.0, float(height) * 0.02)

        touches_left = x1 <= eps_x
        touches_top = y1 <= eps_y
        touches_right = x2 >= width - 1 - eps_x
        touches_bottom = y2 >= height - 1 - eps_y

        count = 0
        if touches_left:
            count += 1
        if touches_top:
            count += 1
        if touches_right:
            count += 1
        if touches_bottom:
            count += 1

        return {
            "left": touches_left,
            "top": touches_top,
            "right": touches_right,
            "bottom": touches_bottom,
            "count": count,
        }

    def is_edge_like_box(self, area_ratio, width_ratio, height_ratio, border_info):
        touches_border_count = int(border_info.get("count", 0))

        if touches_border_count >= 3:
            return True

        if touches_border_count >= 2 and area_ratio >= self.max_edge_area_ratio:
            return True

        if touches_border_count >= 1 and area_ratio >= 0.12:
            return True

        touches_vertical_edge = bool(border_info.get("left", False)) or bool(border_info.get("right", False))
        touches_horizontal_edge = bool(border_info.get("top", False)) or bool(border_info.get("bottom", False))

        if touches_vertical_edge and area_ratio >= 0.06 and height_ratio >= 0.25:
            return True

        if touches_horizontal_edge and area_ratio >= 0.06 and width_ratio >= 0.25:
            return True

        if touches_border_count >= 1 and width_ratio >= 0.90:
            return True

        if touches_border_count >= 1 and height_ratio >= 0.90:
            return True

        return False

    def forward_position(self, current_pose, target_yaw, move_distance):
        x, y, z, yaw = current_pose
        tx = x + math.cos(math.radians(target_yaw)) * move_distance
        ty = y + math.sin(math.radians(target_yaw)) * move_distance
        return (round(tx, 2), round(ty, 2))

    def estimate_world_position(self, current_pose, relative_angle, estimated_depth):
        if estimated_depth is None:
            return None

        x, y, z, yaw = current_pose
        target_yaw = yaw + relative_angle

        tx = x + math.cos(math.radians(target_yaw)) * estimated_depth
        ty = y + math.sin(math.radians(target_yaw)) * estimated_depth

        return (round(tx, 2), round(ty, 2))

    def estimate_depth(self, depth_info, image_index):
        if depth_info is None:
            return None

        if image_index < 0 or image_index >= len(depth_info):
            return None

        try:
            depth_grid = np.array(depth_info[image_index], dtype=np.float32)
            depth_grid = depth_grid[np.isfinite(depth_grid)]
            depth_grid = depth_grid[depth_grid > 0]

            if depth_grid.size == 0:
                return None

            median_depth = float(np.median(depth_grid))
            low_depth = float(np.percentile(depth_grid, 30))
            estimated_depth = 0.5 * median_depth + 0.5 * low_depth
            estimated_depth = max(1.0, min(40.0, estimated_depth))
            return estimated_depth
        except Exception:
            return None

    def count_touched_borders(self, x1, y1, x2, y2, width, height):
        return self.get_border_info(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            width=width,
            height=height
        )["count"]

    def xy_distance(self, p1, p2):
        if p1 is None or p2 is None:
            return float("inf")

        return math.sqrt(
            (float(p1[0]) - float(p2[0])) ** 2
            + (float(p1[1]) - float(p2[1])) ** 2
        )

    def get_camera_region(self, image_index):
        if image_index == 0:
            return "front"
        if image_index == 1:
            return "left"
        if image_index == 2:
            return "right"
        if image_index == 3:
            return "down"
        return "unknown"

    def get_camera_yaw_offset(self, camera_region):
        if camera_region == "front":
            return 0.0
        if camera_region == "left":
            return -90.0
        if camera_region == "right":
            return 90.0
        return 0.0

    def get_relative_region(self, relative_angle):
        relative_angle = self.normalize_angle(relative_angle)

        if abs(relative_angle) <= 35.0:
            return "front"

        if relative_angle > 35.0 and relative_angle <= 135.0:
            return "right"

        if relative_angle < -35.0 and relative_angle >= -135.0:
            return "left"

        if relative_angle > 135.0:
            return "back_right"

        return "back_left"

    def normalize_angle(self, angle):
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def default_observation(self, step_num=0):
        return {
            "valid": False,
            "step_num": step_num,
            "detection_index": -1,
            "score": 0.0,
            "phrase": "",
            "bbox": None,
            "bbox_norm_cxcywh": None,
            "image_index": -1,
            "image_width": 0,
            "image_height": 0,
            "area_ratio": 0.0,
            "width_ratio": 0.0,
            "height_ratio": 0.0,
            "center": None,
            "center_offset_x": 0.0,
            "center_offset_y": 0.0,
            "camera_region": "none",
            "is_horizontal": False,
            "is_downward": False,
            "camera_yaw_offset": 0.0,
            "bbox_yaw_offset": 0.0,
            "relative_angle": 0.0,
            "relative_region": "front",
            "estimated_depth": None,
            "target_world_position": None,
            "quality_score": 0.0,
            "touches_border_count": 0,
            "touches_left": False,
            "touches_top": False,
            "touches_right": False,
            "touches_bottom": False,
            "abnormal_large_box": False,
            "full_frame_like_box": False,
            "edge_like_box": False,
            "candidate_filter_reason": "",
            "candidate_valid": False,
            "reason": "",
        }

    def default_planner_target(self):
        return {
            "valid": False,
            "target_type": "none",
            "position": None,
            "viewpoint_position": None,
            "target_world_position": None,
            "score": 0.0,
            "semantic_value": 0.0,
            "confidence": 0.0,
            "safety_value": 0.0,
            "novelty_value": 0.0,
            "distance": 0.0,
            "relative_angle": 0.0,
            "relative_region": "front",
            "bbox": None,
            "area_ratio": 0.0,
            "estimated_depth": None,
            "candidate_id": None,
            "track_id": None,
            "stop_reason": "",
        }