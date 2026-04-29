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
except Exception:
    from planner.semantic_memory import SemanticMemory
    from planner.local_planner import LocalPlanner

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

        self.unfixed_system_prompt = unfixed_system_prompt
        self.fixed_system_prompt = fixed_system_prompt


    def prepare_inputs(self, episodes, fixed):
        inputs = []
        user_prompts = []
        images = []
        depth_images = []

        for i in range(len(episodes)):
            sources = episodes[i]
            for src in sources[::-1]:
                if 'rgb' in src and 'depth' in src:
                    for img in src['rgb']:
                        images.append(img)
                    depth_images.extend(src['depth'])
                    break
  
        b64_imgs = encode_image(images)

        GROUP = 4
        GROUP_PER_BATCH = 2 
        BATCH_IMG = GROUP * GROUP_PER_BATCH
        
        def iterate_batches(img_list):
            n = len(img_list)
            full_batches = n // BATCH_IMG          # 完整批次数
            tail        = n %  BATCH_IMG           # 残余张数

            for b in range(full_batches):
                yield img_list[b*BATCH_IMG : (b+1)*BATCH_IMG]

            if tail:                               # 处理最后不足 8 张
                yield img_list[-tail:] 

        captions = []
        print("start generate caption")
        start = time.time()

        for imgs in iterate_batches(b64_imgs):
            # raw = generate_caption_qwen_api(imgs)
            raw = generate_caption(imgs)
            
            if len(raw) != len(imgs):
                raise ValueError(f"Expected {len(imgs)} captions, got {len(raw)}")
            captions.extend(raw)

        print("generation captions time:", time.time() - start)

        depth_info_all = self.process_depth(depth_images=depth_images)
      

        for i in range(len(episodes)):
            
            captions4 = captions[4*i:4*i+4]
            depth_info = depth_info_all[4*i:4*i+4]
            
            self.start_position[i] = episodes[i][-1]['start_position']
            
            quaternionr = airsim.Quaternionr(x_val=episodes[i][-1]['start_quaternionr'][0],
                                             y_val=episodes[i][-1]['start_quaternionr'][1],
                                             z_val=episodes[i][-1]['start_quaternionr'][2],
                                             w_val=episodes[i][-1]['start_quaternionr'][3])
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
                last_pose = [(self.start_position[i][0], self.start_position[i][1], self.start_position[i][2]), self.start_yaw[i]]
                raw_poses = [last_pose] * 10

            # 格式化为 prompt 字符串
            format_previous_position = "{\n" + "\n".join([f"    {p}," for p in raw_poses]) + "\n}"
            
            # 提取最后一个 pose 并存为结构化数据，便于后续使用
            if len(raw_poses) > 0:
                last_pose = raw_poses[-1]
                xyz = last_pose[0]  # (x, y, z)
                yaw = last_pose[1]
                self.current_poses[i] = [xyz[0], xyz[1], xyz[2], yaw]
            else:
                # fallback
                self.current_poses[i] = [self.start_position[i][0], self.start_position[i][1], 
                                         self.start_position[i][2], self.start_yaw[i]]

            self.init_semantic_memory(i, step_num)

            self.memory_update_infos[i] = {
                "current_pose": self.current_poses[i],
                "depth_info": depth_info,
                "step_num": step_num,
                "start_position": self.start_position[i]
            }

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
                            object_name=object_name, object_size=object_size, description=description,
                            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                            captions4=captions4, depth_info=depth_info,
                            format_previous_position=format_previous_position, step_num=step_num,
                            move_distance=move_distance, AvgHeadingChange=AvgHeadingChange
                        )
                    }
                ]
            else:
                conversation = [
                    {"role": "system", "content": self.fixed_system_prompt},
                    {
                        "role": "user", 
                        "content": fixed_user_prompt_template.format(
                            object_name=object_name, object_size=object_size, description=description,
                            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                            captions4=captions4, depth_info=depth_info,
                            format_previous_position=format_previous_position, step_num=step_num,
                            move_distance=move_distance, AvgHeadingChange=AvgHeadingChange
                        ) 
                    }
                ]

            prompt_info = conversation[1]["content"]
            user_prompts.append(prompt_info)
            inputs.append((i, conversation))

        return inputs, user_prompts
    

    def init_semantic_memory(self, index, step_num):
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

            print(f"[Semantic Memory] Episode {index}: initialized")


    async def unfixed_single_call(self, index, conversation):
        resp = await self.gpt_client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=conversation
        )
        text = resp.choices[0].message.content.strip()
        semantic_result = self.parse_semantic_result(text)

        memory_summary = self.update_semantic_memory(index, semantic_result)
        action, value, done, memory_target = self.memory_to_legacy_action(
            index=index,
            semantic_result=semantic_result,
            fixed=False
        )

        planned_path = self.plan_local_path(index, memory_target)

        self.print_semantic_result(semantic_result, action, value)
        self.print_memory_summary(index, memory_summary)
        self.print_memory_target(index, memory_target)
        self.print_local_plan(index, planned_path)

        return action, value, done, semantic_result 
    

    async def fixed_single_call(self, index, conversation):
        
        resp = await self.gpt_client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=conversation
        )
        text = resp.choices[0].message.content.strip()
        semantic_result = self.parse_semantic_result(text)

        memory_summary = self.update_semantic_memory(index, semantic_result)
        action, value, done, memory_target = self.memory_to_legacy_action(
            index=index,
            semantic_result=semantic_result,
            fixed=True
        )

        planned_path = self.plan_local_path(index, memory_target)

        self.print_semantic_result(semantic_result, action, value)
        self.print_memory_summary(index, memory_summary)
        self.print_memory_target(index, memory_target)
        self.print_local_plan(index, planned_path)

        return action, value, done, semantic_result              


    async def batch_calls(self, conversations, fixed):
        if fixed:
            tasks = [self.fixed_single_call(index, conv) for index, conv in conversations]
        else:
            tasks = [self.unfixed_single_call(index, conv) for index, conv in conversations]
        return await asyncio.gather(*tasks)


    def run(self, inputs, fixed, prompt_info_list=None):
        results = asyncio.run(self.batch_calls(inputs, fixed))
        actions, steps_size, predict_dones, semantic_results = zip(*results)

        self.semantic_results = list(semantic_results)

        new_actions, new_step_size = self.redirect_action(actions, steps_size, fixed)

        return list(new_actions), list(new_step_size), list(predict_dones)


    def update_semantic_memory(self, index, semantic_result):
        try:
            memory = self.semantic_memories[index]
            update_info = self.memory_update_infos[index]

            memory_summary = memory.update(
                semantic_result=semantic_result,
                current_pose=update_info["current_pose"],
                depth_info=update_info["depth_info"],
                step_num=update_info["step_num"]
            )

            return memory_summary

        except Exception as e:
            print(f"[WARNING] failed to update semantic memory for episode {index}: {e}")
            return None


    def plan_local_path(self, index, memory_target):
        try:
            local_planner = self.local_planners[index]
            current_pose = self.current_poses[index]

            if local_planner is None:
                return self.default_local_plan("local planner is None")

            planned_path = local_planner.plan_path(
                current_pose=current_pose,
                memory_target=memory_target
            )

            self.planned_paths[index] = planned_path
            return planned_path

        except Exception as e:
            print(f"[WARNING] failed to plan local path for episode {index}: {e}")
            return self.default_local_plan(str(e))


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


    def memory_to_legacy_action(self, index, semantic_result, fixed):
        target_visible = semantic_result["target_visible"]
        target_confidence = semantic_result["target_confidence"]

        if target_visible and target_confidence >= 0.75:
            return "stop", 0, True, self.default_memory_target()

        try:
            memory = self.semantic_memories[index]
            current_pose = self.current_poses[index]

            memory_target = memory.get_best_memory_target(
                current_pose=current_pose,
                min_confidence=0.05,
                min_distance=3.0,
                max_distance=45.0
            )
            self.memory_targets[index] = memory_target

            if memory_target.get("valid", False):
                action, value, done = self.memory_target_to_legacy_action(
                    memory_target=memory_target,
                    semantic_result=semantic_result,
                    fixed=fixed
                )
                return action, value, done, memory_target

        except Exception as e:
            print(f"[WARNING] failed to use semantic memory for episode {index}: {e}")

        action, value, done = self.semantic_to_legacy_action(semantic_result, fixed)
        return action, value, done, self.default_memory_target()


    def memory_target_to_legacy_action(self, memory_target, semantic_result, fixed):
        relative_region = memory_target.get("relative_region", "front")
        relative_angle = memory_target.get("relative_angle", 0.0)
        safety_value = memory_target.get("safety_value", 0.5)
        target_score = memory_target.get("score", 0.0)

        if safety_value < 0.20:
            if fixed:
                return "rotl", 0, False
            else:
                return "rotl", 30, False

        if relative_region == "front":
            action = "forward"
        elif relative_region == "left":
            action = "left"
        elif relative_region == "right":
            action = "right"
        elif relative_region == "back_left":
            action = "rotl"
        elif relative_region == "back_right":
            action = "rotr"
        else:
            action = "forward"

        if fixed:
            value = 0
        else:
            if action in ["rotl", "rotr"]:
                value = min(60, max(15, abs(relative_angle)))
            else:
                value = self.estimate_unfixed_step_size(
                    region_score=target_score,
                    safety_score=safety_value,
                    target_visible=semantic_result["target_visible"],
                    target_confidence=semantic_result["target_confidence"]
                )

        done = (action == "stop")
        return action, value, done


    def default_memory_target(self):
        return {
            "valid": False,
            "target_type": "none",
            "position": None,
            "frontier_position": None,
            "viewpoint_position": None,
            "viewpoint_yaw": 0.0,
            "score": 0.0,
            "semantic_value": 0.0,
            "confidence": 0.0,
            "safety_value": 0.0,
            "novelty_value": 0.0,
            "distance": 0.0,
            "relative_angle": 0.0,
            "relative_region": "front",
            "cluster_size": 0
        }


    def semantic_to_legacy_action(self, semantic_result, fixed):
        region_scores = semantic_result["region_scores"]
        safety_scores = semantic_result["safety_scores"]
        best_region = semantic_result["best_region"]
        target_visible = semantic_result["target_visible"]
        target_confidence = semantic_result["target_confidence"]

        if target_visible and target_confidence >= 0.75:
            return "stop", 0, True

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
                target_visible=target_visible,
                target_confidence=target_confidence
            )

        done = (action == "stop")
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
            f"visible={semantic_result['target_visible']}, "
            f"conf={semantic_result['target_confidence']:.2f}, "
            f"action=[{action}, {value}]"
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


    def print_memory_target(self, index, memory_target):
        if memory_target is None:
            return

        if not memory_target.get("valid", False):
            print(f"[Memory Target] Episode {index}: no valid target, fallback to current semantic result")
            return

        print(
            "[Memory Target] "
            f"Episode {index}: "
            f"type={memory_target.get('target_type', 'unknown')}, "
            f"cluster={memory_target.get('cluster_size', 0)}, "
            f"pos={memory_target['position']}, "
            f"score={memory_target['score']:.3f}, "
            f"sem={memory_target['semantic_value']:.2f}, "
            f"conf={memory_target['confidence']:.2f}, "
            f"safety={memory_target['safety_value']:.2f}, "
            f"novelty={memory_target['novelty_value']:.2f}, "
            f"dist={memory_target['distance']}, "
            f"rel_angle={memory_target['relative_angle']}, "
            f"rel_region={memory_target['relative_region']}"
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
                x_val=raw_quaternionr[0], y_val=raw_quaternionr[1], 
                z_val=raw_quaternionr[2], w_val=raw_quaternionr[3]
            )
            pitch, roll, yaw = airsim.to_eularian_angles(quaternionr)
            yaw_degree = round(math.degrees(yaw), 2)

            # 结构化格式 [(x, y, z), yaw]
            formatted = [
                (round(pos[0], 2), round(pos[1], 2)),
                yaw_degree
            ]

            # 保持旧数据格式 fallback 的兼容性
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
                    continue  # 跳过不检查 ascend/descend/rotl/rotr/stop

            except Exception as e:
                print(f"[WARNING] run() failed to check bounds for episode {i}: {e}")
                # 不变更动作
                new_actions[i] = actions[i]
                new_step_size[i] = step_size[i]
        
        return new_actions, new_step_size
           
    
    # def turn_to_nearest_axis(self, dx, dy, yaw):
    #     def closest_signed_xy_axis(dx: float, dy: float):
    #         """
    #         找出 (dx,dy) 在 XY 平面里最接近的有向轴方向，并返回该方向和向该轴的最小夹角（度）。
    #         """
    #         L = math.hypot(dx, dy)
    #         if L == 0:
    #             raise ValueError("零向量没有方向")
    #         # 计算与四个方向的夹角（弧度）
    #         angles = {
    #             '+X':   math.acos( dx / L),
    #             '-X':   math.acos(-dx / L),
    #             '+Y':   math.acos( dy / L),
    #             '-Y':   math.acos(-dy / L),
    #         }
    #         # 选最小的
    #         axis, angle_rad = min(angles.items(), key=lambda kv: kv[1])
    #         return axis, math.degrees(angle_rad)
        
    #     axis, _ = closest_signed_xy_axis(dx, dy)
    #     target_yaws = { '+X':   0,
    #                     '+Y':  90,
    #                     '-X': 180,
    #                     '-Y': 270}
    #     target = target_yaws[axis]

    #     delta_r = (target - yaw + 360) % 360
    #     delta_l = (yaw - target + 360) % 360

    #     if delta_r <= delta_l:
    #         return 'rotr'
    #     else:
    #         return 'rotl'