from model_wrapper.base_model import BaseModelWrapper
from airsim_plugin.airsim_settings import AirsimActionSettings
#from model_wrapper.Qwen_api_captions_2 import generate_caption, encode_image
from model_wrapper.Qwen_api_captions import generate_caption, encode_image
from openai import AsyncClient
from io import BytesIO
from src.common.param import args
from common.prompts import fixed_system_prompt, fixed_user_prompt_template, unfixed_system_prompt, unfixed_user_prompt_template

try:
    from src.planner.semantic_memory import SemanticMemory
    from src.planner.local_planner import LocalPlanner
    from src.planner.navigation_state import NavigationState
    from src.planner.target_tracker import TargetTracker
    from src.planner.target_verifier import TargetVerifier
    from src.model_wrapper.grounding_dino_client import GroundingDINOClient
    from src.planner.path_follower import PathFollower
    from src.planner.planning_types import NavigationTarget, PathPlan, PlannerFeedback, PathReason
except Exception:
    from planner.semantic_memory import SemanticMemory
    from planner.local_planner import LocalPlanner
    from planner.navigation_state import NavigationState
    from planner.target_tracker import TargetTracker
    from planner.target_verifier import TargetVerifier
    from model_wrapper.grounding_dino_client import GroundingDINOClient
    from planner.path_follower import PathFollower
    from planner.planning_types import NavigationTarget, PathPlan, PlannerFeedback, PathReason

import numpy as np
import asyncio
import math
import airsim
import copy
import time
import torch
import torch.nn.functional as F
import os
import json


class ONAir(BaseModelWrapper):
    def __init__(self, fixed, batch_size):
        super().__init__()

        self.fixed = fixed
        self.gpt_client = AsyncClient()

        self.start_position = [[] for _ in range(batch_size)]
        self.start_yaw = [0 for _ in range(batch_size)]
        self.current_poses = [[] for _ in range(batch_size)]

        self.semantic_results = [{} for _ in range(batch_size)]
        self.semantic_memories = [None for _ in range(batch_size)]
        self.memory_update_infos = [{} for _ in range(batch_size)]
        self.memory_targets = [{} for _ in range(batch_size)]

        self.local_planners = [None for _ in range(batch_size)]
        self.planned_paths = [{} for _ in range(batch_size)]
        self.path_followers = [PathFollower() for _ in range(batch_size)]
        self.path_follower_infos = [{} for _ in range(batch_size)]
        self.planner_feedback_infos = [{} for _ in range(batch_size)]

        self.grounding_dino_client = GroundingDINOClient()
        self.grounding_dino_results = [{} for _ in range(batch_size)]

        self.target_trackers = [TargetTracker() for _ in range(batch_size)]
        self.target_tracker_infos = [{} for _ in range(batch_size)]

        self.target_verifier = TargetVerifier(client=self.gpt_client)
        self.target_verification_infos = [{} for _ in range(batch_size)]

        self.navigation_states = [NavigationState() for _ in range(batch_size)]
        self.navigation_infos = [{} for _ in range(batch_size)]

        self.unfixed_system_prompt = unfixed_system_prompt
        self.fixed_system_prompt = fixed_system_prompt

    def prepare_inputs(self, episodes, fixed):
        inputs = []
        user_prompts = []
        images = []
        depth_images = []
        episode_rgb_images = []

        for i in range(len(episodes)):
            sources = episodes[i]
            latest_rgb_images = []

            for src in sources[::-1]:
                if 'rgb' in src and 'depth' in src:
                    latest_rgb_images = src['rgb']
                    for img in src['rgb']:
                        images.append(img)
                    depth_images.extend(src['depth'])
                    break

            episode_rgb_images.append(latest_rgb_images)

        b64_imgs = encode_image(images)

        GROUP = 4
        GROUP_PER_BATCH = 2
        BATCH_IMG = GROUP * GROUP_PER_BATCH

        def iterate_batches(img_list):
            n = len(img_list)
            full_batches = n // BATCH_IMG
            tail = n % BATCH_IMG

            for b in range(full_batches):
                yield img_list[b*BATCH_IMG : (b+1)*BATCH_IMG]

            if tail:
                yield img_list[-tail:]

        captions = []
        verbose_eval = os.environ.get("AIRHUNT_VERBOSE_EVAL", "0") == "1"

        if verbose_eval:
            print("start generate caption")

        start = time.time()
        for imgs in iterate_batches(b64_imgs):
            raw = generate_caption(imgs)
            if len(raw) != len(imgs):
                raise ValueError(f"Expected {len(imgs)} captions, got {len(raw)}")
            captions.extend(raw)

        if verbose_eval:
            print("generation captions time:", time.time() - start)

        depth_info_all = self.process_depth(depth_images=depth_images)

        for i in range(len(episodes)):
            captions4 = captions[4*i:4*i+4]
            depth_info = depth_info_all[4*i:4*i+4]

            self.start_position[i] = episodes[i][-1]['start_position']

            quaternionr = airsim.Quaternionr(
                x_val=episodes[i][-1]['start_quaternionr'][0],
                y_val=episodes[i][-1]['start_quaternionr'][1],
                z_val=episodes[i][-1]['start_quaternionr'][2],
                w_val=episodes[i][-1]['start_quaternionr'][3]
            )
            pitch, roll, yaw = airsim.to_eularian_angles(quaternionr)
            self.start_yaw[i] = math.degrees(yaw)

            step_num = episodes[i][-1]['step']
            description = episodes[i][-1]['description']
            object_name = episodes[i][-1]['object_name']
            object_size = episodes[i][-1]['object_size']

            previous_position = episodes[i][-1]['pre_poses']
            move_distance = episodes[i][-1]['move_distance']
            AvgHeadingChange = episodes[i][-1]['avg_heading_changes']
            raw_poses = self.process_poses(poses=previous_position)

            if len(raw_poses) < 10 and len(raw_poses) > 0:
                last_pose = raw_poses[-1]
                raw_poses += [last_pose] * (10 - len(raw_poses))
            elif len(raw_poses) == 0:
                last_pose = [
                    (self.start_position[i][0], self.start_position[i][1], self.start_position[i][2]),
                    self.start_yaw[i]
                ]
                raw_poses = [last_pose] * 10

            format_previous_position = "{\n" + "\n".join([f"    {p}," for p in raw_poses]) + "\n}"

            if len(raw_poses) > 0:
                last_pose = raw_poses[-1]
                xyz = last_pose[0]
                yaw = last_pose[1]
                self.current_poses[i] = [xyz[0], xyz[1], xyz[2], yaw]
            else:
                self.current_poses[i] = [
                    self.start_position[i][0],
                    self.start_position[i][1],
                    self.start_position[i][2],
                    self.start_yaw[i]
                ]

            self.init_planning_modules(i, step_num)

            self.memory_update_infos[i] = {
                "current_pose": self.current_poses[i],
                "depth_info": depth_info,
                "step_num": step_num,
                "start_position": self.start_position[i]
            }

            grounding_result = self.run_grounding_dino_detection(
                index=i,
                rgb_images=episode_rgb_images[i],
                object_name=object_name,
                description=description,
                step_num=step_num
            )
            self.grounding_dino_results[i] = grounding_result
            self.print_grounding_dino_result(i, grounding_result)

            tracker_info = self.target_trackers[i].update(
                grounding_result=grounding_result,
                current_pose=self.current_poses[i],
                depth_info=depth_info,
                step_num=step_num
            )

            verification_info = self.target_verifier.verify_sync(
                object_name=object_name,
                object_size=object_size,
                description=description,
                captions4=captions4,
                rgb_images=episode_rgb_images[i],
                tracker_info=tracker_info,
                semantic_result=None,
                encode_image_fn=encode_image,
                generate_caption_fn=generate_caption
            )

            tracker_info = self.target_trackers[i].apply_verification(
                tracker_info=tracker_info,
                verification_info=verification_info
            )

            self.target_tracker_infos[i] = tracker_info
            self.target_verification_infos[i] = verification_info

            navigation_info = self.navigation_states[i].update(
                tracker_info=tracker_info,
                step_num=step_num
            )
            self.navigation_infos[i] = navigation_info

            self.print_target_tracker(i, tracker_info)
            self.print_target_verification(i, verification_info)
            self.print_navigation_state(i, navigation_info)

            x_min = int(math.floor(self.start_position[i][0] - 50))
            x_max = int(math.ceil(self.start_position[i][0] + 50))
            y_min = int(math.floor(self.start_position[i][1] - 50))
            y_max = int(math.ceil(self.start_position[i][1] + 50))

            if not fixed:
                conversation = [
                    {"role": "system", "content": self.unfixed_system_prompt},
                    {
                        "role": "user",
                        "content": unfixed_user_prompt_template.format(
                            object_name=object_name,
                            object_size=object_size,
                            description=description,
                            x_min=x_min,
                            x_max=x_max,
                            y_min=y_min,
                            y_max=y_max,
                            captions4=captions4,
                            depth_info=depth_info,
                            format_previous_position=format_previous_position,
                            step_num=step_num,
                            move_distance=move_distance,
                            AvgHeadingChange=AvgHeadingChange
                        )
                    }
                ]
            else:
                conversation = [
                    {"role": "system", "content": self.fixed_system_prompt},
                    {
                        "role": "user",
                        "content": fixed_user_prompt_template.format(
                            object_name=object_name,
                            object_size=object_size,
                            description=description,
                            x_min=x_min,
                            x_max=x_max,
                            y_min=y_min,
                            y_max=y_max,
                            captions4=captions4,
                            depth_info=depth_info,
                            format_previous_position=format_previous_position,
                            step_num=step_num,
                            move_distance=move_distance,
                            AvgHeadingChange=AvgHeadingChange
                        )
                    }
                ]

            prompt_info = conversation[1]["content"]
            user_prompts.append(prompt_info)
            inputs.append((i, conversation))

        return inputs, user_prompts

    def init_planning_modules(self, index, step_num):
        need_reset = False

        if self.semantic_memories[index] is None:
            need_reset = True
        elif step_num == 0:
            need_reset = True
        elif not self.semantic_memories[index].same_origin(self.start_position[index]):
            need_reset = True

        if need_reset:
            self.semantic_memories[index] = SemanticMemory(
                origin=self.start_position[index],
                resolution=2.0,
                map_size=100.0,
                max_sensing_range=25.0,
                sector_angle=70.0,
                visited_radius=3.0
            )

            self.local_planners[index] = LocalPlanner(
                memory=self.semantic_memories[index]
            )

            self.planned_paths[index] = {}
            self.path_follower_infos[index] = {}
            self.planner_feedback_infos[index] = {}
            self.memory_targets[index] = {}

            self.target_trackers[index].reset()
            self.target_tracker_infos[index] = {}
            self.target_verification_infos[index] = {}

            self.navigation_states[index].reset()
            self.navigation_infos[index] = {}

            self.grounding_dino_results[index] = {}

            if os.environ.get("AIRHUNT_VERBOSE_EVAL", "0") == "1":
                print(f"[Semantic Memory] Episode {index}: initialized")

    def run_grounding_dino_detection(self, index, rgb_images, object_name, description, step_num):
        try:
            if self.grounding_dino_client is None:
                return {
                    "available": False,
                    "error": "grounding dino client is None",
                    "detections": [],
                    "best_detection": None,
                    "best_score": 0.0,
                    "num_detections": 0
                }

            result = self.grounding_dino_client.detect_episode(
                rgb_images=rgb_images,
                object_name=object_name,
                description=description,
                episode_index=index,
                step_num=step_num
            )
            return result

        except Exception as e:
            print(f"[GroundingDINO] Episode {index}: detection failed: {e}")
            return {
                "available": False,
                "error": str(e),
                "detections": [],
                "best_detection": None,
                "best_score": 0.0,
                "num_detections": 0
            }

    async def unfixed_single_call(self, index, conversation):
        resp = await self.gpt_client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=conversation
        )
        text = resp.choices[0].message.content.strip()

        semantic_result = self.parse_semantic_result(text)
        memory_summary = self.update_semantic_memory(index, semantic_result)

        planner_step = self.planner_driven_step(
            index=index,
            semantic_result=semantic_result,
            fixed=False
        )

        action = planner_step["action"]
        value = planner_step["value"]
        done = planner_step["done"]
        selected_target = planner_step["selected_target"]
        planned_path = planner_step["planned_path"]

        if os.environ.get("AIRHUNT_VERBOSE_EVAL", "0") == "1":
            self.print_semantic_result(semantic_result, action, value)
            self.print_memory_summary(index, memory_summary)
            self.print_selected_target(index, selected_target)
            self.print_local_plan(index, planned_path)
            self.print_path_follower(index, planner_step.get("path_follower_info", {}))
            self.print_planner_feedback(index, planner_step.get("planner_feedback", {}))

        return action, value, done, semantic_result

    async def fixed_single_call(self, index, conversation):
        resp = await self.gpt_client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=conversation
        )
        text = resp.choices[0].message.content.strip()

        semantic_result = self.parse_semantic_result(text)
        memory_summary = self.update_semantic_memory(index, semantic_result)

        planner_step = self.planner_driven_step(
            index=index,
            semantic_result=semantic_result,
            fixed=True
        )

        action = planner_step["action"]
        value = planner_step["value"]
        done = planner_step["done"]
        selected_target = planner_step["selected_target"]
        planned_path = planner_step["planned_path"]

        if os.environ.get("AIRHUNT_VERBOSE_EVAL", "0") == "1":
            self.print_semantic_result(semantic_result, action, value)
            self.print_memory_summary(index, memory_summary)
            self.print_selected_target(index, selected_target)
            self.print_local_plan(index, planned_path)
            self.print_path_follower(index, planner_step.get("path_follower_info", {}))
            self.print_planner_feedback(index, planner_step.get("planner_feedback", {}))

        return action, value, done, semantic_result

    def planner_driven_step(self, index, semantic_result, fixed):
        """
        AirHunt-localized planner-driven step.

        Semantic outputs and GDINO verification only select a navigation target.
        The final UAV-ON action must be produced from LocalPlanner path through
        PathFollower. This function intentionally does not call
        select_navigation_action().
        """
        selected_target = self.select_navigation_target(
            index=index,
            semantic_result=semantic_result,
            fixed=fixed
        )

        if self.is_stop_candidate(selected_target):
            planned_path = self.default_local_plan("stop candidate selected")
            path_follower_info = {
                "valid": True,
                "action": "stop",
                "step_size": 0,
                "done": True,
                "reason": selected_target.get("stop_reason", "stop candidate selected"),
                "action_source": "navigation_stop_candidate",
            }
            planner_feedback = self.build_planner_feedback(
                index=index,
                selected_target=selected_target,
                planned_path=planned_path,
                path_follower_info=path_follower_info
            )

            self.memory_targets[index] = selected_target
            self.planned_paths[index] = planned_path
            self.path_follower_infos[index] = path_follower_info
            self.planner_feedback_infos[index] = planner_feedback

            return {
                "action": "stop",
                "value": 0,
                "done": True,
                "selected_target": selected_target,
                "planned_path": planned_path,
                "path_follower_info": path_follower_info,
                "planner_feedback": planner_feedback,
            }

        planned_path = self.plan_local_path(index, selected_target)
        action, value, done, path_follower_info = self.follow_local_path(
            index=index,
            planned_path=planned_path,
            fixed=fixed
        )

        if action is None:
            semantic_target = self.build_semantic_region_target(
                index=index,
                semantic_result=semantic_result,
                fixed=fixed,
                reason="primary target path is not executable"
            )
            if semantic_target.get("valid", False):
                retry_path = self.plan_local_path(index, semantic_target)
                retry_action, retry_value, retry_done, retry_follower_info = self.follow_local_path(
                    index=index,
                    planned_path=retry_path,
                    fixed=fixed
                )
                if retry_action is not None:
                    selected_target = semantic_target
                    planned_path = retry_path
                    action = retry_action
                    value = retry_value
                    done = retry_done
                    path_follower_info = retry_follower_info

        if action is None:
            action, value, done, path_follower_info = self.no_executable_path_action(
                index=index,
                fixed=fixed,
                selected_target=selected_target,
                planned_path=planned_path
            )

        planner_feedback = self.build_planner_feedback(
            index=index,
            selected_target=selected_target,
            planned_path=planned_path,
            path_follower_info=path_follower_info
        )

        self.memory_targets[index] = selected_target
        self.planned_paths[index] = planned_path
        self.path_follower_infos[index] = path_follower_info
        self.planner_feedback_infos[index] = planner_feedback

        return {
            "action": action,
            "value": value,
            "done": done,
            "selected_target": selected_target,
            "planned_path": planned_path,
            "path_follower_info": path_follower_info,
            "planner_feedback": planner_feedback,
        }

    def select_navigation_target(self, index, semantic_result, fixed):
        """
        Select a semantic navigation target, not an action.

        The returned target is still a planning target compatible with the current
        LocalPlanner interface. It must be planned into a path before action output.
        """
        navigation_info = self.navigation_infos[index]
        mode = navigation_info.get("mode", "explore")
        planner_target = navigation_info.get("planner_target", None)

        if isinstance(planner_target, dict):
            if self.is_stop_candidate(planner_target):
                return self.prepare_target_for_planning(
                    target=planner_target,
                    source="navigation_state_stop_candidate"
                )

        if mode == NavigationState.MODE_STOP:
            return self.make_stop_candidate(index, navigation_info)

        if mode in [NavigationState.MODE_NAVIGATE, NavigationState.MODE_RECOVER]:
            if isinstance(planner_target, dict) and planner_target.get("valid", False):
                return self.prepare_target_for_planning(
                    target=planner_target,
                    source="navigation_state_target"
                )

        memory_target = self.select_memory_navigation_target(index)
        if memory_target.get("valid", False):
            return memory_target

        return self.build_semantic_region_target(
            index=index,
            semantic_result=semantic_result,
            fixed=fixed,
            reason="no valid memory target"
        )

    def select_memory_navigation_target(self, index):
        try:
            memory = self.semantic_memories[index]
            current_pose = self.current_poses[index]
            memory_target = memory.get_best_memory_target(
                current_pose=current_pose,
                min_confidence=0.05,
                min_distance=3.0,
                max_distance=45.0
            )
            return self.prepare_target_for_planning(
                target=memory_target,
                source="semantic_memory"
            )
        except Exception as e:
            print(f"[WARNING] failed to select memory target for episode {index}: {e}")
            target = self.default_memory_target()
            target["reason"] = str(e)
            return target

    def prepare_target_for_planning(self, target, source=""):
        if not isinstance(target, dict):
            return self.default_memory_target()

        new_target = copy.deepcopy(target)
        if source:
            new_target["source"] = source

        if "target_type" not in new_target:
            new_target["target_type"] = new_target.get("type", "memory")

        target_position = self.extract_target_position(new_target)
        if target_position is not None:
            if new_target.get("position", None) is None:
                new_target["position"] = target_position
            if new_target.get("viewpoint_position", None) is None:
                new_target["viewpoint_position"] = target_position

        if "score" not in new_target:
            new_target["score"] = new_target.get("semantic_value", 0.0)

        if "relative_region" not in new_target:
            new_target["relative_region"] = "front"

        if "relative_angle" not in new_target:
            new_target["relative_angle"] = 0.0

        if "stop_reason" not in new_target:
            new_target["stop_reason"] = ""

        return new_target

    def extract_target_position(self, target):
        for key in [
            "viewpoint_position",
            "position",
            "frontier_position",
            "target_world_position",
            "verified_target_position",
        ]:
            value = target.get(key, None)
            if value is None:
                continue
            if not isinstance(value, (list, tuple)):
                continue
            if len(value) < 2:
                continue
            try:
                x = float(value[0])
                y = float(value[1])
                z = float(value[2]) if len(value) >= 3 else 0.0
                return (round(x, 2), round(y, 2), round(z, 2))
            except Exception:
                continue
        return None

    def build_semantic_region_target(self, index, semantic_result, fixed, reason=""):
        try:
            current_pose = self.current_poses[index]
            x, y, z, yaw = current_pose

            region_scores = semantic_result.get("region_scores", {})
            safety_scores = semantic_result.get("safety_scores", {})
            novelty_scores = semantic_result.get("novelty_scores", {})

            best_region = semantic_result.get("best_region", "front")
            if best_region not in ["front", "left", "right"]:
                best_region = max(region_scores, key=region_scores.get)

            region_score = float(region_scores.get(best_region, 0.0))
            safety_score = float(safety_scores.get(best_region, 0.5))
            novelty_score = float(novelty_scores.get(best_region, 0.5))

            yaw_offset = self.get_region_yaw_offset(best_region)
            target_yaw = yaw + yaw_offset

            if fixed:
                distance = 8.0
            else:
                distance = self.estimate_unfixed_step_size(
                    region_score=region_score,
                    safety_score=safety_score,
                    target_visible=False,
                    target_confidence=0.0
                )
                distance = max(5.0, min(12.0, float(distance) * 1.8))

            target_x = x + math.cos(math.radians(target_yaw)) * distance
            target_y = y + math.sin(math.radians(target_yaw)) * distance
            target_x, target_y = self.clamp_to_search_bounds(index, target_x, target_y)

            relative_angle = self.normalize_relative_angle(target_yaw - yaw)

            target = self.default_memory_target()
            target.update({
                "valid": True,
                "target_type": "semantic_region",
                "source": "semantic_result",
                "position": (round(target_x, 2), round(target_y, 2), round(z, 2)),
                "viewpoint_position": (round(target_x, 2), round(target_y, 2), round(z, 2)),
                "frontier_position": None,
                "score": round(region_score, 3),
                "semantic_value": round(region_score, 3),
                "confidence": 0.1,
                "safety_value": round(safety_score, 3),
                "novelty_value": round(novelty_score, 3),
                "distance": round(distance, 2),
                "target_yaw": round(target_yaw, 2),
                "relative_angle": round(relative_angle, 2),
                "relative_region": best_region,
                "stop_reason": "",
                "reason": reason,
            })
            return target
        except Exception as e:
            print(f"[WARNING] failed to build semantic region target for episode {index}: {e}")
            target = self.default_memory_target()
            target["reason"] = str(e)
            return target

    def get_region_yaw_offset(self, region):
        if region == "left":
            return -90.0
        if region == "right":
            return 90.0
        return 0.0

    def clamp_to_search_bounds(self, index, x, y):
        try:
            start_position = self.start_position[index]
            x_min = start_position[0] - 48.0
            x_max = start_position[0] + 48.0
            y_min = start_position[1] - 48.0
            y_max = start_position[1] + 48.0
            x = min(max(float(x), x_min), x_max)
            y = min(max(float(y), y_min), y_max)
        except Exception:
            pass
        return x, y

    def normalize_relative_angle(self, angle):
        angle = float(angle)
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def make_stop_candidate(self, index, navigation_info):
        stop_target = self.default_memory_target()
        stop_target["target_type"] = "verified_target_stop"
        stop_target["source"] = "navigation_state"
        stop_target["valid"] = True
        stop_target["position"] = self.get_current_xy(index)
        stop_target["viewpoint_position"] = self.get_current_xy(index)
        stop_target["target_world_position"] = None

        tracker_info = navigation_info.get("tracker_info", {})
        if isinstance(tracker_info, dict):
            stop_target["target_world_position"] = tracker_info.get(
                "verified_target_position", None
            )

        stop_target["stop_reason"] = navigation_info.get(
            "reason", "reached verified target object position"
        )
        return stop_target

    def is_stop_candidate(self, target):
        if not isinstance(target, dict):
            return False

        target_type = target.get("target_type", "")
        if target_type in [
            "verified_target_stop",
            "gdino_stop",
            "gdino_verified_stop",
            "gdino_position_stop"
        ]:
            return True

        if target.get("stop_reason", ""):
            return True

        return False

    def follow_local_path(self, index, planned_path, fixed):
        try:
            path_follower = self.path_followers[index]
            current_pose = self.current_poses[index]

            action, step_size, done, info = path_follower.follow_as_tuple(
                current_pose=current_pose,
                path_plan=planned_path
            )

            if action is None:
                return None, 0.0, False, info

            if fixed and action != "stop":
                step_size = 0

            return action, step_size, done, info
        except Exception as e:
            print(f"[WARNING] failed to follow local path for episode {index}: {e}")
            info = {
                "valid": False,
                "action": None,
                "step_size": 0.0,
                "done": False,
                "reason": str(e),
                "action_source": "path_follower_error",
            }
            return None, 0.0, False, info

    def no_executable_path_action(self, index, fixed, selected_target, planned_path):
        """
        Compatibility guard for the current UAV-ON evaluation interface.

        This is not a semantic fallback. It is only used when no executable path
        action can be produced after trying the selected target and semantic region
        target. PlannerFeedback records the failure so later steps can feed it back
        into viewpoint reselection instead of keeping this guard as policy.
        """
        if fixed:
            value = 0
        else:
            value = 15

        info = {
            "valid": False,
            "action": "rotl",
            "step_size": value,
            "done": False,
            "reason": "no_executable_path_after_replanning",
            "action_source": "planner_interface_guard",
            "path_reason": planned_path.get("reason", "unknown") if isinstance(planned_path, dict) else "unknown",
            "path_len": planned_path.get("path_len", 0) if isinstance(planned_path, dict) else 0,
            "path_length": planned_path.get("path_length", 0.0) if isinstance(planned_path, dict) else 0.0,
        }

        return "rotl", value, False, info

    def build_planner_feedback(self, index, selected_target, planned_path, path_follower_info):
        try:
            path_plan = PathPlan.from_dict(planned_path)
            target = NavigationTarget.from_dict(selected_target)
            feedback = PlannerFeedback.from_path_plan(
                path_plan=path_plan,
                target=target,
                viewpoint=None,
                action_source=path_follower_info.get("action_source", "")
            ).to_dict()

            feedback["path_follower_valid"] = bool(path_follower_info.get("valid", False))
            feedback["path_follower_reason"] = path_follower_info.get("reason", "")
            feedback["episode_index"] = index
            return feedback
        except Exception as e:
            return {
                "valid": False,
                "reason": str(e),
                "episode_index": index,
                "action_source": path_follower_info.get("action_source", "")
                if isinstance(path_follower_info, dict) else "",
            }

    def print_path_follower(self, index, path_follower_info):
        if not isinstance(path_follower_info, dict):
            return

        print(
            "[PathFollower] "
            f"Episode {index}: "
            f"valid={path_follower_info.get('valid', False)}, "
            f"action={path_follower_info.get('action', None)}, "
            f"step={path_follower_info.get('step_size', 0.0)}, "
            f"reason={path_follower_info.get('reason', '')}, "
            f"source={path_follower_info.get('action_source', '')}, "
            f"next={path_follower_info.get('next_waypoint', None)}, "
            f"dist={path_follower_info.get('distance_to_waypoint', None)}, "
            f"yaw_error={path_follower_info.get('yaw_error', None)}"
        )

    def print_planner_feedback(self, index, planner_feedback):
        if not isinstance(planner_feedback, dict):
            return

        print(
            "[PlannerFeedback] "
            f"Episode {index}: "
            f"valid={planner_feedback.get('valid', False)}, "
            f"reason={planner_feedback.get('reason', '')}, "
            f"target={planner_feedback.get('target_type', '')}, "
            f"path_len={planner_feedback.get('path_len', 0)}, "
            f"replan={planner_feedback.get('replan_required', False)}, "
            f"reselect={planner_feedback.get('should_reselect_viewpoint', False)}, "
            f"source={planner_feedback.get('action_source', '')}"
        )

    def parse_semantic_result(self, text):
        try:
            raw_text = text.strip()

            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                raw_text = raw_text.replace("json", "", 1).strip()

            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                raw_text = raw_text[start_idx:end_idx+1]

            semantic_result = json.loads(raw_text)
            semantic_result = self.normalize_semantic_result(semantic_result)

            return semantic_result

        except Exception as e:
            print(f"[WARNING] failed to parse semantic JSON: {e}")
            print(f"[WARNING] raw response: {text}")
            return self.default_semantic_result()

    def normalize_semantic_result(self, semantic_result):
        region_scores = semantic_result.get("region_scores", {})
        safety_scores = semantic_result.get("safety_scores", {})
        novelty_scores = semantic_result.get("novelty_scores", {})

        region_scores = self.normalize_score_dict(region_scores, default_score=0.0)
        safety_scores = self.normalize_score_dict(safety_scores, default_score=0.5)
        novelty_scores = self.normalize_score_dict(novelty_scores, default_score=0.5)

        best_region = semantic_result.get("best_region", None)
        if best_region not in ["front", "left", "right"]:
            best_region = max(region_scores, key=region_scores.get)

        target_visible = semantic_result.get("target_visible", False)
        target_visible = self.normalize_bool(target_visible)

        target_confidence = semantic_result.get("target_confidence", 0.0)
        target_confidence = self.normalize_score(target_confidence, default_score=0.0)

        altitude_assessment = semantic_result.get("altitude_assessment", {})
        if not isinstance(altitude_assessment, dict):
            altitude_assessment = {}

        altitude_assessment.setdefault("target_size_level", "unknown")
        altitude_assessment.setdefault("height_suitability", "unknown")
        altitude_assessment.setdefault("comment", "")

        reason = semantic_result.get("reason", "")
        evidence = semantic_result.get("evidence", [])

        if not isinstance(reason, str):
            reason = str(reason)

        if not isinstance(evidence, list):
            evidence = [str(evidence)]

        semantic_result = {
            "region_scores": region_scores,
            "safety_scores": safety_scores,
            "novelty_scores": novelty_scores,
            "best_region": best_region,
            "target_visible": target_visible,
            "target_confidence": target_confidence,
            "altitude_assessment": altitude_assessment,
            "reason": reason,
            "evidence": evidence
        }

        return semantic_result

    def normalize_score_dict(self, score_dict, default_score=0.0):
        valid_regions = ["front", "left", "right"]
        new_score_dict = {}

        if not isinstance(score_dict, dict):
            score_dict = {}

        for region in valid_regions:
            score = score_dict.get(region, default_score)
            new_score_dict[region] = self.normalize_score(score, default_score)

        return new_score_dict

    def normalize_score(self, score, default_score=0.0):
        try:
            score = float(score)
        except Exception:
            score = default_score

        score = max(0.0, min(1.0, score))
        return score

    def normalize_bool(self, value):
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            value = value.strip().lower()
            if value in ["true", "yes", "1"]:
                return True
            if value in ["false", "no", "0"]:
                return False

        return bool(value)

    def default_semantic_result(self):
        semantic_result = {
            "region_scores": {
                "front": 0.34,
                "left": 0.33,
                "right": 0.33
            },
            "safety_scores": {
                "front": 0.5,
                "left": 0.5,
                "right": 0.5
            },
            "novelty_scores": {
                "front": 0.5,
                "left": 0.5,
                "right": 0.5
            },
            "best_region": "front",
            "target_visible": False,
            "target_confidence": 0.0,
            "altitude_assessment": {
                "target_size_level": "unknown",
                "height_suitability": "unknown",
                "comment": ""
            },
            "reason": "fallback semantic result because JSON parsing failed",
            "evidence": []
        }

        return semantic_result

    def semantic_to_legacy_action(self, semantic_result, fixed):
        region_scores = semantic_result["region_scores"]
        safety_scores = semantic_result["safety_scores"]
        best_region = semantic_result["best_region"]

        max_safety = max(safety_scores.values())

        if max_safety < 0.25:
            if fixed:
                return "rotl", 0, False
            else:
                return "rotl", 30, False

        if best_region not in ["front", "left", "right"]:
            best_region = max(region_scores, key=region_scores.get)

        if best_region == "front":
            action = "forward"
        elif best_region == "left":
            action = "left"
        elif best_region == "right":
            action = "right"
        else:
            action = "forward"

        if fixed:
            value = 0
        else:
            value = self.estimate_unfixed_step_size(
                region_score=region_scores[best_region],
                safety_score=safety_scores[best_region],
                target_visible=False,
                target_confidence=0.0
            )

        done = False
        return action, value, done

    def estimate_unfixed_step_size(self, region_score, safety_score, target_visible, target_confidence):
        if target_visible or target_confidence >= 0.6:
            step_size = 2.0
        elif safety_score < 0.35:
            step_size = 2.0
        elif safety_score < 0.6:
            step_size = 3.0
        elif region_score >= 0.75 and safety_score >= 0.75:
            step_size = 6.0
        elif region_score >= 0.55 and safety_score >= 0.6:
            step_size = 5.0
        else:
            step_size = 4.0

        return step_size

    def default_memory_target(self):
        return {
            "valid": False,
            "target_type": "none",
            "grid": None,
            "position": None,
            "viewpoint_position": None,
            "frontier_position": None,
            "score": 0.0,
            "semantic_value": 0.0,
            "confidence": 0.0,
            "safety_value": 0.0,
            "novelty_value": 0.0,
            "visited": False,
            "observe_count": 0,
            "distance": 0.0,
            "target_yaw": 0.0,
            "relative_angle": 0.0,
            "relative_region": "front",
            "stop_reason": ""
        }

    def default_local_plan(self, reason):
        plan = {
            "valid": False,
            "reason": reason,
            "target_position": None,
            "path": [],
            "path_len": 0,
            "path_length": 0.0
        }
        return plan

    def get_current_xy(self, index):
        try:
            current_pose = self.current_poses[index]
            return (round(current_pose[0], 2), round(current_pose[1], 2))
        except Exception:
            return None

    def print_grounding_dino_result(self, index, grounding_result):
        try:
            if grounding_result is None:
                return

            summary = self.grounding_dino_client.summarize_result(grounding_result)
            print(
                "[GroundingDINO] "
                f"Episode {index}: {summary}"
            )
        except Exception as e:
            print(f"[GroundingDINO] Episode {index}: failed to print result: {e}")

    def print_target_tracker(self, index, tracker_info):
        if tracker_info is None:
            return

        observation = tracker_info.get("observation", {})
        planner_target = tracker_info.get("planner_target", {})

        print(
            "[TargetTracker] "
            f"Episode {index}: "
            f"candidate={tracker_info.get('candidate', False)}, "
            f"verify_required={tracker_info.get('verification_required', False)}, "
            f"confirmed={tracker_info.get('confirmed', False)}, "
            f"verified={tracker_info.get('verified', False)}, "
            f"stop={tracker_info.get('stop_ready', False)}, "
            f"score={observation.get('score', 0.0):.3f}, "
            f"count={tracker_info.get('confirm_count', 0)}/{tracker_info.get('required_count', 0)}, "
            f"lost={tracker_info.get('lost_count', 0)}, "
            f"nav_count={tracker_info.get('navigate_count', 0)}, "
            f"image={observation.get('image_index', -1)}, "
            f"region={observation.get('camera_region', 'none')}, "
            f"rel_region={observation.get('relative_region', 'front')}, "
            f"angle={observation.get('relative_angle', 0.0)}, "
            f"area={observation.get('area_ratio', 0.0)}, "
            f"depth={observation.get('estimated_depth', None)}, "
            f"buffer={tracker_info.get('candidate_buffer_size', 0)}, "
            f"verified_id={tracker_info.get('verified_candidate_id', None)}, "
            f"target={planner_target.get('position', None)}, "
            f"verified_pos={tracker_info.get('verified_target_position', None)}"
        )

    def print_target_verification(self, index, verification_info):
        if verification_info is None:
            return
        if not isinstance(verification_info, dict):
            return

        verbose_eval = os.environ.get("AIRHUNT_VERBOSE_EVAL", "0") == "1"
        verbose_stop = os.environ.get("AIRHUNT_VERBOSE_STOP", "0") == "1"

        if not verification_info.get("checked", False):
            reason = verification_info.get("reason", "")
            if reason is None:
                reason = ""
            if len(reason) > 220:
                reason = reason[:220] + "..."

            print(
                "[TargetVerifier] "
                f"Episode {index}: checked=False, "
                f"reason={reason}"
            )
            return

        selected_candidate_id = verification_info.get("selected_candidate_id", None)
        reason = verification_info.get("reason", "")
        reject_reason = verification_info.get("reject_reason", "")
        crop_caption = verification_info.get("crop_caption", "")

        if reason is None:
            reason = ""
        if reject_reason is None:
            reject_reason = ""
        if crop_caption is None:
            crop_caption = ""

        if len(reason) > 220:
            reason = reason[:220] + "..."
        if len(reject_reason) > 220:
            reject_reason = reject_reason[:220] + "..."
        if len(crop_caption) > 140:
            crop_caption = crop_caption[:140] + "..."

        print(
            "[TargetVerifier] "
            f"Episode {index}: "
            f"verified={verification_info.get('verified', False)}, "
            f"conf={verification_info.get('confidence', 0.0):.2f}, "
            f"same_object={verification_info.get('same_object', False)}, "
            f"hard_reject={verification_info.get('hard_reject', False)}, "
            f"selected={selected_candidate_id}, "
            f"candidate_count={verification_info.get('candidate_count', 0)}, "
            f"reason={reason}, "
            f"reject={reject_reason}, "
            f"crop_caption={crop_caption}"
        )

        if not verbose_eval and not verbose_stop:
            return

        candidate_debug = verification_info.get("candidate_debug", [])
        if not isinstance(candidate_debug, list):
            candidate_debug = []

        for candidate in candidate_debug[:6]:
            if not isinstance(candidate, dict):
                continue

            candidate_id = candidate.get("candidate_id", None)
            candidate_caption = candidate.get("crop_caption", "")

            if candidate_caption is None:
                candidate_caption = ""
            if len(candidate_caption) > 120:
                candidate_caption = candidate_caption[:120] + "..."

            crop_debug = candidate.get("crop_debug", {})
            if not isinstance(crop_debug, dict):
                crop_debug = {}

            print(
                "[TargetVerifierCandidate] "
                f"Episode {index}: "
                f"id={candidate_id}, "
                f"step={candidate.get('step_num', None)}, "
                f"img={candidate.get('image_index', None)}, "
                f"region={candidate.get('camera_region', None)}, "
                f"score={float(candidate.get('score', 0.0)):.3f}, "
                f"area={float(candidate.get('area_ratio', 0.0)):.4f}, "
                f"bbox={candidate.get('bbox', None)}, "
                f"target={candidate.get('target_world_position', None)}, "
                f"caption_len={candidate.get('crop_caption_len', 0)}, "
                f"caption={candidate_caption}"
            )

            print(
                "[TargetVerifierCrop] "
                f"Episode {index}: "
                f"id={candidate_id}, "
                f"ok={crop_debug.get('ok', False)}, "
                f"stage={crop_debug.get('stage', '')}, "
                f"debug_reason={crop_debug.get('reason', '')}, "
                f"rgb_count={crop_debug.get('rgb_count', 0)}, "
                f"image_type={crop_debug.get('image_type', '')}, "
                f"image_size={crop_debug.get('image_size', None)}, "
                f"crop_box={crop_debug.get('crop_box', None)}, "
                f"crop_size={crop_debug.get('crop_size', None)}, "
                f"caption_type={crop_debug.get('caption_type', '')}, "
                f"caption_len={crop_debug.get('caption_len', 0)}, "
                f"crop_path={crop_debug.get('crop_path', '')}"
            )

    def print_navigation_state(self, index, navigation_info):
        if navigation_info is None:
            return
        if not isinstance(navigation_info, dict):
            return

        verbose_eval = os.environ.get("AIRHUNT_VERBOSE_EVAL", "0") == "1"
        stop_trace = os.environ.get("AIRHUNT_STOP_TRACE", "1") != "0"

        changed = navigation_info.get("changed", False)
        prev_mode = navigation_info.get("prev_mode", "none")
        mode = navigation_info.get("mode", "none")
        reason = navigation_info.get("reason", "")

        if reason is None:
            reason = ""
        if len(reason) > 240:
            reason = reason[:240] + "..."

        should_print = (
            verbose_eval
            or stop_trace
            or changed
            or mode != "explore"
        )

        if not should_print:
            return

        if changed:
            print(
                "[NavMode] "
                f"Episode {index}: "
                f"{prev_mode} -> {mode}, "
                f"reason={reason}"
            )
        else:
            print(
                "[NavMode] "
                f"Episode {index}: "
                f"mode={mode}, "
                f"reason={reason}"
            )

    def print_semantic_result(self, semantic_result, action, value):
        region_scores = semantic_result["region_scores"]
        safety_scores = semantic_result["safety_scores"]
        novelty_scores = semantic_result["novelty_scores"]

        print(
            "[Semantic] "
            f"region=({region_scores['front']:.2f}, {region_scores['left']:.2f}, {region_scores['right']:.2f}), "
            f"safety=({safety_scores['front']:.2f}, {safety_scores['left']:.2f}, {safety_scores['right']:.2f}), "
            f"novelty=({novelty_scores['front']:.2f}, {novelty_scores['left']:.2f}, {novelty_scores['right']:.2f}), "
            f"best={semantic_result['best_region']}, "
            f"llm_visible={semantic_result['target_visible']}, "
            f"llm_conf={semantic_result['target_confidence']:.2f}, "
            f"action=[{action}, {value}]"
        )

    def print_selected_target(self, index, selected_target):
        if selected_target is None:
            return

        if not selected_target.get("valid", False):
            print(
                "[Selected Target] "
                f"Episode {index}: invalid, "
                f"type={selected_target.get('target_type', 'none')}, "
                f"reason={selected_target.get('stop_reason', 'no valid target')}"
            )
            return

        print(
            "[Selected Target] "
            f"Episode {index}: "
            f"type={selected_target.get('target_type', 'unknown')}, "
            f"pos={selected_target.get('position', None)}, "
            f"score={selected_target.get('score', 0.0):.3f}, "
            f"conf={selected_target.get('confidence', 0.0):.2f}, "
            f"dist={selected_target.get('distance', 0.0)}, "
            f"rel_angle={selected_target.get('relative_angle', 0.0)}, "
            f"rel_region={selected_target.get('relative_region', 'front')}"
        )

    def print_memory_summary(self, index, memory_summary):
        if memory_summary is None:
            return

        print(
            "[Semantic Memory] "
            f"Episode {index}: "
            f"observed={memory_summary['observed_cells']}, "
            f"visited={memory_summary['visited_cells']}, "
            f"max_value={memory_summary['max_value']:.2f}, "
            f"max_conf={memory_summary['max_confidence']:.2f}, "
            f"max_pos={memory_summary['max_position']}"
        )

    def print_local_plan(self, index, planned_path):
        if planned_path is None:
            return

        if not planned_path.get("valid", False):
            print(
                "[Local Planner] "
                f"Episode {index}: invalid, "
                f"reason={planned_path.get('reason', 'unknown')}"
            )
            return

        path = planned_path.get("path", [])
        preview_path = path[:3]

        print(
            "[Local Planner] "
            f"Episode {index}: "
            f"valid=True, "
            f"path_len={planned_path.get('path_len', 0)}, "
            f"path_length={planned_path.get('path_length', 0.0)}, "
            f"target={planned_path.get('target_position', None)}, "
            f"preview={preview_path}"
        )

    def process_depth(self, depth_images):
        depth_info = []

        for depth_image in depth_images:
            distance_image = np.array(depth_image) / 255.0 * 100
            x = torch.from_numpy(distance_image).unsqueeze(0).unsqueeze(0).float()
            y = -F.adaptive_max_pool2d(-x, (3, 3))
            y_np = y.squeeze().cpu().numpy()
            y_int = np.round(y_np).astype(int).tolist()
            depth_info.append(y_int)

        return depth_info

    def process_poses(self, poses):
        pre_poses_xyzYaw = []

        for pose in poses:
            pos = pose['position']
            raw_quaternionr = pose['quaternionr']
            quaternionr = airsim.Quaternionr(
                x_val=raw_quaternionr[0],
                y_val=raw_quaternionr[1],
                z_val=raw_quaternionr[2],
                w_val=raw_quaternionr[3]
            )
            pitch, roll, yaw = airsim.to_eularian_angles(quaternionr)
            yaw_degree = round(math.degrees(yaw), 2)

            formatted = [
                (round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)),
                yaw_degree
            ]
            pre_poses_xyzYaw.append(formatted)

        return pre_poses_xyzYaw

    def redirect_action(self, actions, step_size, fixed):
        new_actions = [None] * len(actions)
        new_step_size = list(step_size)

        for i, action in enumerate(actions):
            new_actions[i] = action

            try:
                start_position = self.start_position[i]
                x_min = round(start_position[0] - 50, 2)
                x_max = round(start_position[0] + 50, 2)
                y_min = round(start_position[1] - 50, 2)
                y_max = round(start_position[1] + 50, 2)

                current_pose = self.current_poses[i]
                x, y, z, yaw = current_pose
                current_step_size = new_step_size[i]

                if action == 'forward':
                    dx = math.cos(math.radians(yaw))
                    dy = math.sin(math.radians(yaw))
                    dz = 0
                    vector = np.array([dx, dy, dz])
                    norm = np.linalg.norm(vector)

                    if norm > 1e-6:
                        unit_vector = vector / norm
                    else:
                        unit_vector = np.array([0, 0, 0])

                    if fixed:
                        new_position = np.array([x, y, z]) + unit_vector * AirsimActionSettings.FORWARD_STEP_SIZE
                    else:
                        new_position = np.array([x, y, z]) + unit_vector * current_step_size

                    if new_position[0] > x_max or new_position[0] < x_min or new_position[1] > y_max or new_position[1] < y_min:
                        new_actions[i] = 'rotl'
                        new_step_size[i] = 15
                        print(f"[INFO] Episode {i}: '{action}' would go out of bounds → replaced with '{new_actions[i]}'")

                elif action == "left":
                    unit_x = 1.0 * math.cos(math.radians(yaw + 90))
                    unit_y = 1.0 * math.sin(math.radians(yaw + 90))
                    vector = np.array([unit_x, unit_y, 0])
                    norm = np.linalg.norm(vector)

                    if norm > 1e-6:
                        unit_vector = vector / norm
                    else:
                        unit_vector = np.array([0, 0, 0])

                    if fixed:
                        new_position = np.array([x, y, z]) - unit_vector * AirsimActionSettings.LEFT_RIGHT_STEP_SIZE
                    else:
                        new_position = np.array([x, y, z]) - unit_vector * current_step_size

                    if new_position[0] > x_max or new_position[0] < x_min or new_position[1] > y_max or new_position[1] < y_min:
                        new_actions[i] = 'rotl'
                        new_step_size[i] = 15
                        print(f"[INFO] Episode {i}: '{action}' would go out of bounds → replaced with '{new_actions[i]}'")

                elif action == "right":
                    unit_x = 1.0 * math.cos(math.radians(yaw + 90))
                    unit_y = 1.0 * math.sin(math.radians(yaw + 90))
                    vector = np.array([unit_x, unit_y, 0])
                    norm = np.linalg.norm(vector)

                    if norm > 1e-6:
                        unit_vector = vector / norm
                    else:
                        unit_vector = np.array([0, 0, 0])

                    if fixed:
                        new_position = np.array([x, y, z]) + unit_vector * AirsimActionSettings.LEFT_RIGHT_STEP_SIZE
                    else:
                        new_position = np.array([x, y, z]) + unit_vector * current_step_size

                    if new_position[0] > x_max or new_position[0] < x_min or new_position[1] > y_max or new_position[1] < y_min:
                        new_actions[i] = 'rotl'
                        new_step_size[i] = 15
                        print(f"[INFO] Episode {i}: '{action}' would go out of bounds → replaced with '{new_actions[i]}'")

                else:
                    new_actions[i] = action
                    new_step_size[i] = step_size[i]
                    continue

            except Exception as e:
                print(f"[WARNING] run() failed to check bounds for episode {i}: {e}")
                new_actions[i] = actions[i]
                new_step_size[i] = step_size[i]

        return new_actions, new_step_size