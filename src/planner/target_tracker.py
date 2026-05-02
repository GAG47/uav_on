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
        max_reasonable_area=0.65,
        horizontal_fov=90.0,
        candidate_ttl=8,
        max_candidate_count=24,
        max_candidates_for_verification=6,
        verified_target_ttl=12,
        stop_position_distance=5.0,
        navigate_stop_distance=5.0,
        min_navigate_steps_before_stop=1,
        confirm_step_distance=3.0,
        max_association_distance=10.0,
    ):
        self.candidate_score = candidate_score
        self.confirm_score = confirm_score
        self.stop_score = stop_score
        self.max_lost_count = max_lost_count

        self.min_area_ratio = min_area_ratio
        self.max_reasonable_area = max_reasonable_area
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

        self.candidate_buffer = []
        self.candidate_counter = 0

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

        added_candidates = self.update_candidate_buffer(
            observations=observations,
            current_pose=current_pose,
            step_num=step_num
        )

        best_observation = self.select_best_observation(observations)

        candidate = len(self.get_recent_candidates(step_num=step_num)) > 0
        verification_required = candidate and self.verified_target is None

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

            if candidate:
                self.lost_count = 0
                self.confirm_count = min(1, self.confirm_count + 1)
                planner_target = self.default_planner_target()
                reason = "task-aware object candidates collected, waiting for verification"
            else:
                self.lost_count += 1
                self.confirm_count = 0
                planner_target = self.default_planner_target()
                reason = "no reliable task-aware object candidate"

        self.last_observation = best_observation
        if best_observation.get("is_horizontal", False):
            self.last_horizontal_observation = best_observation

        self.last_planner_target = planner_target

        verification_candidates = self.get_verification_candidates(
            step_num=step_num
        )

        info = {
            "candidate": candidate,
            "horizontal_candidate": self.has_horizontal_candidate(verification_candidates),
            "down_candidate": self.has_down_candidate(observations),
            "consistent": confirmed,
            "geometric_confirmed": False,
            "confirmed": confirmed,
            "verified": confirmed,
            "verification_required": verification_required,
            "verification": {},
            "stop_ready": stop_ready,
            "geometric_stop_ready": stop_ready,
            "confirm_count": self.confirm_count,
            "required_count": 1,
            "lost_count": self.lost_count,
            "navigate_count": self.navigate_count,
            "stable_target_position": self.verified_target_position,
            "stable_target_score": round(self.verified_target_score, 3),
            "verified_target": self.verified_target,
            "verified_target_position": self.verified_target_position,
            "verified_candidate_id": self.verified_candidate_id,
            "candidate_buffer_size": len(self.candidate_buffer),
            "added_candidate_count": added_candidates,
            "verification_candidates": verification_candidates,
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
                self.set_verified_target(
                    candidate=candidate,
                    verification_info=verification_info
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
                new_info["geometric_confirmed"] = False
                new_info["stop_ready"] = stop_ready
                new_info["geometric_stop_ready"] = stop_ready
                new_info["planner_target"] = planner_target
                new_info["stable_target_position"] = self.verified_target_position
                new_info["stable_target_score"] = round(self.verified_target_score, 3)
                new_info["verified_target"] = self.verified_target
                new_info["verified_target_position"] = self.verified_target_position
                new_info["verified_candidate_id"] = self.verified_candidate_id
                new_info["reason"] = (
                    "target object verified by task-aware candidate selection: "
                    + str(verification_info.get("reason", ""))
                )

                return new_info

        if checked and not verified:
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
            new_info["reason"] = (
                "no target object selected by verifier: "
                + str(verification_info.get("reason", ""))
            )

            return new_info

        # Important:
        # When verification is not required, preserve the stop decision
        # already made by update(). Otherwise a verified_target_stop target
        # can be kept while stop_ready is overwritten to False, causing
        # the agent to keep moving forward instead of stopping.
        new_info["confirmed"] = current_confirmed or self.verified_target is not None
        new_info["verified"] = current_verified or self.verified_target is not None
        new_info["stop_ready"] = current_stop_ready
        new_info["geometric_stop_ready"] = current_stop_ready

        if current_stop_ready:
            new_info["reason"] = "reached verified target object position"

        return new_info

    def set_verified_target(self, candidate, verification_info):
        candidate_position = candidate.get("target_world_position", None)

        if candidate_position is None:
            return

        confidence = float(verification_info.get("confidence", candidate.get("score", 0.0)))

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
        self.verified_target = None
        self.verified_target_position = None
        self.verified_target_score = 0.0
        self.verified_target_step = -1
        self.verified_candidate_id = None
        self.navigate_count = 0

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
        area_ratio = (box_width * box_height) / max(1.0, float(image_width * image_height))

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        center_offset_x = (cx - image_width / 2.0) / max(1.0, image_width / 2.0)
        center_offset_y = (cy - image_height / 2.0) / max(1.0, image_height / 2.0)

        image_index = int(detection.get("image_index", 0))
        camera_region = self.get_camera_region(image_index)
        is_horizontal = camera_region in ["front", "left", "right"]
        is_downward = camera_region == "down"

        score = float(detection.get("score", 0.0))

        touches_border_count = self.count_touched_borders(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            width=image_width,
            height=image_height
        )

        abnormal_large_box = area_ratio >= self.max_reasonable_area
        full_frame_like_box = area_ratio >= 0.80 or touches_border_count >= 3

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
            is_horizontal=is_horizontal,
            is_downward=is_downward,
            full_frame_like_box=full_frame_like_box,
            abnormal_large_box=abnormal_large_box
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
            "abnormal_large_box": abnormal_large_box,
            "full_frame_like_box": full_frame_like_box,
            "reason": "ok"
        })

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
        if not observation.get("valid", False):
            return False

        if not observation.get("is_horizontal", False):
            return False

        if observation.get("score", 0.0) < self.candidate_score:
            return False

        if observation.get("area_ratio", 0.0) < self.min_area_ratio:
            return False

        if observation.get("target_world_position", None) is None:
            return False

        if observation.get("full_frame_like_box", False):
            return False

        return True

    def build_candidate_from_observation(self, observation, current_pose):
        self.candidate_counter += 1

        candidate_id = "cand_{:06d}".format(self.candidate_counter)

        candidate = {
            "candidate_id": candidate_id,
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
            "current_pose": list(current_pose),
            "crop_caption": "",
        }

        return candidate

    def compute_candidate_quality(self, observation):
        quality = float(observation.get("quality_score", 0.0))
        score = float(observation.get("score", 0.0))
        area_ratio = float(observation.get("area_ratio", 0.0))

        if observation.get("camera_region", "unknown") == "front":
            quality += 0.10

        if abs(observation.get("center_offset_x", 0.0)) <= 0.50:
            quality += 0.05

        if area_ratio > 0.25 and score < 0.60:
            quality *= 0.55

        return round(max(0.0, quality), 4)

    def get_recent_candidates(self, step_num):
        candidates = []

        for candidate in self.candidate_buffer:
            if step_num - int(candidate.get("step_num", 0)) <= self.candidate_ttl:
                candidates.append(candidate)

        return candidates

    def get_verification_candidates(self, step_num):
        candidates = self.get_recent_candidates(step_num=step_num)

        candidates.sort(
            key=lambda item: item.get("candidate_quality", 0.0),
            reverse=True
        )

        return candidates[:self.max_candidates_for_verification]

    def prune_candidate_buffer(self, step_num):
        new_buffer = []

        for candidate in self.candidate_buffer:
            age = step_num - int(candidate.get("step_num", 0))
            if age <= self.candidate_ttl:
                new_buffer.append(candidate)

        self.candidate_buffer = new_buffer

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

        return observations[0]

    def compute_detection_quality(
        self,
        score,
        area_ratio,
        is_horizontal,
        is_downward,
        full_frame_like_box,
        abnormal_large_box
    ):
        quality = float(score)

        if area_ratio < self.min_area_ratio:
            quality *= 0.1

        if abnormal_large_box:
            quality *= 0.55

        if full_frame_like_box:
            quality *= 0.10

        if is_downward:
            quality *= 0.30

        if not is_horizontal and not is_downward:
            quality *= 0.2

        return quality

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
        eps = 2.0
        count = 0

        if x1 <= eps:
            count += 1
        if y1 <= eps:
            count += 1
        if x2 >= width - 1 - eps:
            count += 1
        if y2 >= height - 1 - eps:
            count += 1

        return count

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
            "abnormal_large_box": False,
            "full_frame_like_box": False,
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
            "stop_reason": "",
        }