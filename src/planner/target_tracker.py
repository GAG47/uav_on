import math
import numpy as np


class TargetTracker:
    def __init__(
        self,
        candidate_score=0.30,
        confirm_score=0.40,
        stop_score=0.55,
        confirm_required_count=2,
        max_lost_count=3,
        min_area_ratio=0.002,
        max_reasonable_area=0.65,
        max_stop_area_ratio=0.70,
        stop_area_ratio=0.045,
        stop_depth=8.0,
        stop_position_distance=8.0,
        horizontal_fov=90.0,
        confirm_step_distance=5.0,
        navigate_stop_distance=6.0,
        max_consistency_distance=12.0,
        max_consistency_angle=45.0,
        min_navigate_steps_before_stop=1,
        history_size=5,
    ):
        self.candidate_score = candidate_score
        self.confirm_score = confirm_score
        self.stop_score = stop_score
        self.confirm_required_count = confirm_required_count
        self.max_lost_count = max_lost_count

        self.min_area_ratio = min_area_ratio
        self.max_reasonable_area = max_reasonable_area
        self.max_stop_area_ratio = max_stop_area_ratio
        self.stop_area_ratio = stop_area_ratio

        self.stop_depth = stop_depth
        self.stop_position_distance = stop_position_distance
        self.horizontal_fov = horizontal_fov
        self.confirm_step_distance = confirm_step_distance
        self.navigate_stop_distance = navigate_stop_distance

        self.max_consistency_distance = max_consistency_distance
        self.max_consistency_angle = max_consistency_angle
        self.min_navigate_steps_before_stop = min_navigate_steps_before_stop
        self.history_size = history_size

        self.confirm_count = 0
        self.lost_count = 0
        self.navigate_count = 0

        self.last_observation = None
        self.last_horizontal_observation = None
        self.last_planner_target = None

        self.stable_target_position = None
        self.stable_target_score = 0.0
        self.horizontal_history = []

    def reset(self):
        self.confirm_count = 0
        self.lost_count = 0
        self.navigate_count = 0

        self.last_observation = None
        self.last_horizontal_observation = None
        self.last_planner_target = None

        self.stable_target_position = None
        self.stable_target_score = 0.0
        self.horizontal_history = []

    def update(self, grounding_result, current_pose, depth_info=None, step_num=0):
        observation = self.parse_grounding_result(
            grounding_result=grounding_result,
            current_pose=current_pose,
            depth_info=depth_info,
            step_num=step_num
        )

        horizontal_candidate = self.is_horizontal_candidate(observation)
        down_candidate = self.is_down_candidate(observation)
        consistent = False

        if horizontal_candidate:
            consistent = self.is_consistent_with_track(observation)

            if consistent:
                self.lost_count = 0
                self.last_observation = observation
                self.last_horizontal_observation = observation
                self.update_stable_target(observation)

                if observation["score"] >= self.confirm_score:
                    self.confirm_count += 1
                else:
                    self.confirm_count = max(0, self.confirm_count - 1)
            else:
                self.lost_count = 0
                self.last_observation = observation
                self.last_horizontal_observation = observation
                self.reset_stable_target(observation)

                if observation["score"] >= self.confirm_score:
                    self.confirm_count = 1
                else:
                    self.confirm_count = 0

        elif down_candidate:
            self.lost_count = 0
            self.last_observation = observation
            self.confirm_count = max(0, self.confirm_count - 1)

        else:
            self.lost_count += 1
            self.confirm_count = max(0, self.confirm_count - 1)

        candidate = horizontal_candidate or down_candidate

        geometric_confirmed = (
            horizontal_candidate
            and consistent
            and self.confirm_count >= self.confirm_required_count
            and self.stable_target_position is not None
        )

        if geometric_confirmed:
            self.navigate_count += 1
        else:
            self.navigate_count = 0

        geometric_stop_ready = self.is_stop_ready(
            observation=observation,
            confirmed=geometric_confirmed,
            current_pose=current_pose
        )

        if horizontal_candidate:
            planner_target = self.build_planner_target(
                observation=observation,
                current_pose=current_pose,
                confirmed=geometric_confirmed,
                stop_ready=geometric_stop_ready
            )
        elif self.last_horizontal_observation is not None and self.lost_count <= self.max_lost_count:
            planner_target = self.build_recover_target(
                observation=self.last_horizontal_observation,
                current_pose=current_pose
            )
        else:
            planner_target = self.default_planner_target()

        self.last_planner_target = planner_target

        if geometric_stop_ready:
            reason = "geometric target track satisfies stop condition, waiting for verification"
        elif geometric_confirmed:
            reason = "stable horizontal gdino target geometrically confirmed"
        elif horizontal_candidate and consistent:
            reason = "horizontal gdino candidate is consistent"
        elif horizontal_candidate and not consistent:
            reason = "horizontal gdino candidate starts a new track"
        elif down_candidate:
            reason = "downward gdino candidate kept as nearby evidence only"
        elif self.last_horizontal_observation is not None and self.lost_count <= self.max_lost_count:
            reason = "horizontal target temporarily lost, recover with last horizontal observation"
        else:
            reason = observation.get("reason", "no reliable gdino candidate")

        info = {
            "candidate": candidate,
            "horizontal_candidate": horizontal_candidate,
            "down_candidate": down_candidate,
            "consistent": consistent,
            "geometric_confirmed": geometric_confirmed,
            "confirmed": geometric_confirmed,
            "verified": False,
            "verification": {},
            "stop_ready": geometric_stop_ready,
            "geometric_stop_ready": geometric_stop_ready,
            "confirm_count": self.confirm_count,
            "required_count": self.confirm_required_count,
            "lost_count": self.lost_count,
            "navigate_count": self.navigate_count,
            "stable_target_position": self.stable_target_position,
            "stable_target_score": round(self.stable_target_score, 3),
            "observation": observation,
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

        verification_checked = bool(verification_info.get("checked", False))
        verified = bool(verification_info.get("verified", False))
        geometric_confirmed = bool(new_info.get("geometric_confirmed", new_info.get("confirmed", False)))
        geometric_stop_ready = bool(new_info.get("geometric_stop_ready", new_info.get("stop_ready", False)))

        new_info["verified"] = verified

        if verified and geometric_confirmed:
            new_info["confirmed"] = True
            new_info["stop_ready"] = geometric_stop_ready
            new_info["reason"] = (
                "verified target, "
                + str(verification_info.get("reason", new_info.get("reason", "")))
            )
            return new_info

        if verification_checked and geometric_confirmed and not verified:
            self.reject_current_track(
                reason=verification_info.get("reason", "target rejected by verifier")
            )
            new_info["confirmed"] = False
            new_info["stop_ready"] = False
            new_info["planner_target"] = self.default_planner_target()
            new_info["candidate"] = False
            new_info["reason"] = (
                "target rejected by verifier: "
                + str(verification_info.get("reason", "not verified"))
            )
            return new_info

        new_info["confirmed"] = False
        new_info["stop_ready"] = False
        return new_info

    def reject_current_track(self, reason=""):
        self.confirm_count = 0
        self.navigate_count = 0
        self.lost_count = 0
        self.last_observation = None
        self.last_horizontal_observation = None
        self.last_planner_target = None
        self.stable_target_position = None
        self.stable_target_score = 0.0
        self.horizontal_history = []

    def parse_grounding_result(self, grounding_result, current_pose, depth_info=None, step_num=0):
        default_observation = self.default_observation(step_num=step_num)

        if grounding_result is None:
            default_observation["reason"] = "grounding result is None"
            return default_observation

        if not grounding_result.get("available", False):
            default_observation["reason"] = grounding_result.get("error", "grounding dino not available")
            return default_observation

        detections = grounding_result.get("detections", [])
        if not isinstance(detections, list) or len(detections) == 0:
            default_observation["reason"] = "no detection"
            return default_observation

        observations = []

        for detection in detections:
            observation = self.parse_detection(
                detection=detection,
                current_pose=current_pose,
                depth_info=depth_info,
                step_num=step_num
            )
            if observation.get("valid", False):
                observations.append(observation)

        if len(observations) == 0:
            default_observation["reason"] = "no valid detection after parsing"
            return default_observation

        observations.sort(
            key=lambda item: item.get("quality_score", 0.0),
            reverse=True
        )

        best_observation = observations[0]
        best_observation["all_observation_count"] = len(observations)

        return best_observation

    def parse_detection(self, detection, current_pose, depth_info=None, step_num=0):
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
            image_index=image_index,
            center_offset_y=center_offset_y
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
            quality *= 0.35

        if full_frame_like_box:
            quality *= 0.15

        if is_downward:
            quality *= 0.45

        if not is_horizontal and not is_downward:
            quality *= 0.2

        return quality

    def is_horizontal_candidate(self, observation):
        if not observation.get("valid", False):
            return False

        if not observation.get("is_horizontal", False):
            return False

        if observation.get("score", 0.0) < self.candidate_score:
            return False

        if observation.get("area_ratio", 0.0) < self.min_area_ratio:
            return False

        if observation.get("full_frame_like_box", False):
            return False

        if observation.get("abnormal_large_box", False) and observation.get("score", 0.0) < 0.75:
            return False

        if observation.get("target_world_position", None) is None:
            return False

        return True

    def is_down_candidate(self, observation):
        if not observation.get("valid", False):
            return False

        if not observation.get("is_downward", False):
            return False

        if observation.get("score", 0.0) < self.confirm_score:
            return False

        if observation.get("area_ratio", 0.0) < self.min_area_ratio:
            return False

        if observation.get("full_frame_like_box", False):
            return False

        return True

    def is_consistent_with_track(self, observation):
        if self.stable_target_position is None:
            return True

        current_position = observation.get("target_world_position", None)
        if current_position is None:
            return False

        position_distance = self.xy_distance(
            p1=current_position,
            p2=self.stable_target_position
        )

        last_angle = 0.0
        if self.last_horizontal_observation is not None:
            last_angle = float(self.last_horizontal_observation.get("relative_angle", 0.0))

        current_angle = float(observation.get("relative_angle", 0.0))
        angle_diff = abs(self.normalize_angle(current_angle - last_angle))

        if position_distance > self.max_consistency_distance:
            return False

        if angle_diff > self.max_consistency_angle and self.confirm_count > 0:
            return False

        return True

    def update_stable_target(self, observation):
        position = observation.get("target_world_position", None)
        if position is None:
            return

        score = float(observation.get("score", 0.0))

        if self.stable_target_position is None:
            self.stable_target_position = position
            self.stable_target_score = score
        else:
            alpha = max(0.25, min(0.75, score))
            sx, sy = self.stable_target_position
            px, py = position

            new_x = sx * (1.0 - alpha) + px * alpha
            new_y = sy * (1.0 - alpha) + py * alpha

            self.stable_target_position = (round(new_x, 2), round(new_y, 2))
            self.stable_target_score = max(self.stable_target_score * 0.9, score)

        self.horizontal_history.append({
            "position": position,
            "score": score,
            "camera_region": observation.get("camera_region", "front"),
            "relative_angle": observation.get("relative_angle", 0.0),
            "step_num": observation.get("step_num", 0),
            "area_ratio": observation.get("area_ratio", 0.0),
            "estimated_depth": observation.get("estimated_depth", None),
        })

        if len(self.horizontal_history) > self.history_size:
            self.horizontal_history = self.horizontal_history[-self.history_size:]

    def reset_stable_target(self, observation):
        position = observation.get("target_world_position", None)
        score = float(observation.get("score", 0.0))

        self.stable_target_position = position
        self.stable_target_score = score
        self.horizontal_history = []

        if position is not None:
            self.horizontal_history.append({
                "position": position,
                "score": score,
                "camera_region": observation.get("camera_region", "front"),
                "relative_angle": observation.get("relative_angle", 0.0),
                "step_num": observation.get("step_num", 0),
                "area_ratio": observation.get("area_ratio", 0.0),
                "estimated_depth": observation.get("estimated_depth", None),
            })

    def is_stop_ready(self, observation, confirmed, current_pose):
        if not confirmed:
            return False

        if self.navigate_count < self.min_navigate_steps_before_stop:
            return False

        if self.stable_target_position is None:
            return False

        if observation.get("camera_region", "unknown") != "front":
            return False

        if observation.get("score", 0.0) < self.stop_score:
            return False

        if observation.get("full_frame_like_box", False):
            return False

        if observation.get("area_ratio", 0.0) > self.max_stop_area_ratio:
            return False

        if abs(observation.get("center_offset_x", 0.0)) > 0.45:
            return False

        current_xy = (float(current_pose[0]), float(current_pose[1]))
        distance_to_stable_target = self.xy_distance(
            p1=current_xy,
            p2=self.stable_target_position
        )

        if distance_to_stable_target <= self.stop_position_distance:
            return True

        estimated_depth = observation.get("estimated_depth", None)
        if estimated_depth is not None and estimated_depth <= self.stop_depth:
            if observation.get("area_ratio", 0.0) >= self.stop_area_ratio:
                return True

        return False

    def build_planner_target(self, observation, current_pose, confirmed=False, stop_ready=False):
        if not observation.get("valid", False):
            return self.default_planner_target()

        if not observation.get("is_horizontal", False):
            return self.default_planner_target()

        if stop_ready:
            current_xy = (round(float(current_pose[0]), 2), round(float(current_pose[1]), 2))
            return {
                "valid": True,
                "target_type": "gdino_stop",
                "position": current_xy,
                "viewpoint_position": current_xy,
                "target_world_position": self.stable_target_position,
                "score": float(observation.get("score", 0.0)),
                "semantic_value": float(observation.get("score", 0.0)),
                "confidence": float(observation.get("score", 0.0)),
                "safety_value": 0.8,
                "novelty_value": 0.8,
                "distance": 0.0,
                "relative_angle": 0.0,
                "relative_region": "front",
                "bbox": observation.get("bbox", None),
                "area_ratio": observation.get("area_ratio", 0.0),
                "estimated_depth": observation.get("estimated_depth", None),
                "stop_reason": "verified gdino target stop",
            }

        if confirmed and self.stable_target_position is not None:
            planner_target = self.build_target_from_stable_position(
                observation=observation,
                current_pose=current_pose,
                target_type="gdino_navigate"
            )
        else:
            planner_target = self.build_target_from_observation(
                observation=observation,
                current_pose=current_pose,
                target_type="gdino_confirm"
            )

        return planner_target

    def build_target_from_observation(self, observation, current_pose, target_type):
        relative_angle = float(observation.get("relative_angle", 0.0))
        target_yaw = float(current_pose[3]) + relative_angle

        estimated_depth = observation.get("estimated_depth", None)
        if estimated_depth is None:
            move_distance = self.confirm_step_distance
        else:
            move_distance = max(2.0, min(self.confirm_step_distance, float(estimated_depth) * 0.45))

        target_position = self.forward_position(
            current_pose=current_pose,
            target_yaw=target_yaw,
            move_distance=move_distance
        )

        return {
            "valid": True,
            "target_type": target_type,
            "position": target_position,
            "viewpoint_position": target_position,
            "target_world_position": observation.get("target_world_position", None),
            "score": float(observation.get("score", 0.0)),
            "semantic_value": float(observation.get("score", 0.0)),
            "confidence": float(observation.get("score", 0.0)),
            "safety_value": 0.8,
            "novelty_value": 0.8,
            "distance": round(move_distance, 2),
            "relative_angle": round(relative_angle, 2),
            "relative_region": observation.get("relative_region", "front"),
            "bbox": observation.get("bbox", None),
            "area_ratio": observation.get("area_ratio", 0.0),
            "estimated_depth": observation.get("estimated_depth", None),
            "stop_reason": "",
        }

    def build_target_from_stable_position(self, observation, current_pose, target_type):
        current_xy = (float(current_pose[0]), float(current_pose[1]))
        target_xy = self.stable_target_position

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
            "target_type": target_type,
            "position": target_position,
            "viewpoint_position": target_position,
            "target_world_position": target_xy,
            "score": float(observation.get("score", 0.0)),
            "semantic_value": float(observation.get("score", 0.0)),
            "confidence": float(observation.get("score", 0.0)),
            "safety_value": 0.8,
            "novelty_value": 0.8,
            "distance": round(move_distance, 2),
            "relative_angle": round(relative_angle, 2),
            "relative_region": self.get_relative_region(relative_angle),
            "bbox": observation.get("bbox", None),
            "area_ratio": observation.get("area_ratio", 0.0),
            "estimated_depth": observation.get("estimated_depth", None),
            "stop_reason": "",
        }

    def build_recover_target(self, observation, current_pose):
        if observation is None:
            return self.default_planner_target()

        if self.stable_target_position is not None:
            current_xy = (float(current_pose[0]), float(current_pose[1]))
            target_xy = self.stable_target_position

            target_angle_world = math.degrees(
                math.atan2(target_xy[1] - current_xy[1], target_xy[0] - current_xy[0])
            )
            relative_angle = self.normalize_angle(target_angle_world - float(current_pose[3]))

            target_position = self.forward_position(
                current_pose=current_pose,
                target_yaw=target_angle_world,
                move_distance=4.0
            )
        else:
            relative_angle = float(observation.get("relative_angle", 0.0))
            target_yaw = float(current_pose[3]) + relative_angle

            target_position = self.forward_position(
                current_pose=current_pose,
                target_yaw=target_yaw,
                move_distance=4.0
            )

        return {
            "valid": True,
            "target_type": "gdino_recover",
            "position": target_position,
            "viewpoint_position": target_position,
            "target_world_position": self.stable_target_position,
            "score": float(observation.get("score", 0.0)),
            "semantic_value": float(observation.get("score", 0.0)),
            "confidence": max(0.1, float(observation.get("score", 0.0)) * 0.6),
            "safety_value": 0.6,
            "novelty_value": 0.5,
            "distance": 4.0,
            "relative_angle": round(relative_angle, 2),
            "relative_region": self.get_relative_region(relative_angle),
            "bbox": observation.get("bbox", None),
            "area_ratio": observation.get("area_ratio", 0.0),
            "estimated_depth": observation.get("estimated_depth", None),
            "stop_reason": "",
        }

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

    def estimate_depth(self, depth_info, image_index, center_offset_y=0.0):
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
            "all_observation_count": 0,
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
            "stop_reason": "",
        }