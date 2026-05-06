from collections import OrderedDict
import copy
import random
import sys
import time
import numpy as np
import math
import os
import json
from pathlib import Path
import airsim
from typing import List, Optional
import tqdm

from src.common.param import args
from utils.logger import logger

sys.path.append(str(Path(str(os.getcwd())).resolve()))

from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool
from utils.env_utils_uav import SimState
from utils.env_vector_uav import VectorEnvUtil


class AirVLNENV:
    def __init__(
        self,
        batch_size=8,
        dataset_path=None,
        save_path=None,
        seed=1,
        activate_maps=[],
    ):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.epoch_done = False
        self.seed = seed
        self.collected_keys = set()
        self.activate_maps = set(activate_maps)
        self.exist_save_path = save_path

        load_data = self.load_my_datasets()
        self.data = load_data
        logger.info('Loaded dataset {}.'.format(len(self.data)))

        self.index_data = 0
        self.dataset_group_by_scene = True
        self.data = self._group_scenes()
        logger.info('dataset grouped by scene, ')

        scenes = [item['map_name'] for item in self.data]
        self.scenes = set(scenes)
        self.sim_states: Optional[List[SimState]] = [None for _ in range(batch_size)]
        self.last_using_map_list = []
        self.one_scene_could_use_num = 5e3
        self.this_scene_used_cnt = 0

        # 连续执行时的最大单步移动距离。
        # planned_path 可以很长，但每次只执行一小段，然后重新感知和重新规划。
        self.max_continuous_step = 5.0
        self.min_continuous_advance = 0.5

        self.init_VectorEnvUtil()

    def load_my_datasets(self):
        """
        load object location json file, reconstruct a json file with every infomation

        return:
            object_info (contains position, rotation, scale, object name, instruction )
        """
        data = []
        trajectory_path = os.path.join(self.dataset_path)
        data_file = json.load(open(self.dataset_path, 'r'))

        for index, item in enumerate(tqdm.tqdm(data_file, desc="Loading")):
            traj_info = {}
            traj_info['map_name'] = item['map_name']
            traj_info['object_name'] = item['true_name']
            traj_info['object_size'] = item['size']
            traj_info['object_position'] = item['pose']
            traj_info['start_pose'] = item['start_pose']
            traj_info['description'] = item['description']
            traj_info['distance_to_target'] = item['info']['euclidean_distance']
            traj_info['trajectory_dir'] = trajectory_path
            traj_info['size'] = item['size']
            traj_info['task_id'] = item['episode_id']
            data.append(traj_info)

        return data

    def _group_scenes(self):
        """
        group all objects with their scene name, choose objects which
        """
        scene_sort_keys = OrderedDict()

        for item in self.data:
            if str(item['map_name']) not in scene_sort_keys:
                scene_sort_keys[str(item['map_name'])] = len(scene_sort_keys)

        return sorted(self.data, key=lambda e: (scene_sort_keys[str(e['map_name'])]))

    def init_VectorEnvUtil(self):
        self.delete_VectorEnvUtil()
        self.VectorEnvUtil = VectorEnvUtil(self.scenes, self.batch_size)

    def delete_VectorEnvUtil(self):
        if hasattr(self, 'VectorEnvUtil'):
            del self.VectorEnvUtil

        import gc
        gc.collect()

    def next_minibatch(self, skip_scenes=[], data_it=0):
        batch = []

        if self.epoch_done and self.index_data >= len(self.data):
            self.batch = None
            return

        while True:
            if self.index_data >= len(self.data):
                self.epoch_done = True
                random.shuffle(self.data)
                logger.warning('random shuffle data and pad to batch size')

                if self.dataset_group_by_scene:
                    self.data = self._group_scenes()
                    logger.warning('dataset grouped by scene')

                if len(batch) == 0:
                    self.index_data = 0
                    self.batch = None
                    return

                self.index_data = self.batch_size - len(batch)
                batch += self.data[:self.index_data]
                self.index_data = len(self.data) + 1
                break

            task = self.data[self.index_data]

            if task['map_name'] in skip_scenes:
                self.index_data += 1
                continue

            batch.append(task)
            self.index_data += 1

            if len(batch) == self.batch_size:
                break

        self.batch = copy.deepcopy(batch)

        assert len(self.batch) == self.batch_size, 'next_minibatch error'
        self.VectorEnvUtil.set_batch(self.batch)

        return self.batch

    def changeToNewTask(self):
        self._changeEnv(need_change=False)
        self._setDrone()
        self.update_measurements()

    def _setDrone(self):
        drone_position_info = [item['start_pose']['start_position'] for item in self.batch]
        drone_quaternior_info = [item['start_pose']['start_quaternionr'] for item in self.batch]

        poses = []
        cnt = 0

        for index_1, item in enumerate(self.machines_info):
            poses.append([])

            for index_2, _ in enumerate(item['open_scenes']):
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        x_val=drone_position_info[cnt][0],
                        y_val=drone_position_info[cnt][1],
                        z_val=drone_position_info[cnt][2],
                    ),
                    orientation_val=airsim.Quaternionr(
                        x_val=drone_quaternior_info[cnt][0],
                        y_val=drone_quaternior_info[cnt][1],
                        z_val=drone_quaternior_info[cnt][2],
                        w_val=drone_quaternior_info[cnt][3],
                    ),
                )
                poses[index_1].append(pose)
                cnt += 1

        self.simulator_tool.setPoses(poses=poses)

        state_info_results = self.simulator_tool.getSensorInfo()
        cnt = 0

        for index_1, item in enumerate(self.machines_info):
            for index_2, _ in enumerate(item['open_scenes']):
                self.sim_states[cnt] = SimState(
                    index=cnt,
                    step=0,
                    task_info=self.batch[cnt],
                )
                self.sim_states[cnt].sensorInfo = [state_info_results[index_1][index_2]]
                cnt += 1

    def _changeEnv(self, need_change: bool = True):
        using_map_list = [item['map_name'] for item in self.batch]
        assert len(using_map_list) == self.batch_size, '错误'

        machines_info_template = copy.deepcopy(args.machines_info)

        total_max_scene_num = 0
        for item in machines_info_template:
            total_max_scene_num += item['MAX_SCENE_NUM']

        assert self.batch_size <= total_max_scene_num, 'error args param: batch_size'

        machines_info = []
        ix = 0

        for index, item in enumerate(machines_info_template):
            machines_info.append(item)
            delta = min(
                self.batch_size,
                item['MAX_SCENE_NUM'],
                len(using_map_list) - ix,
            )
            machines_info[index]['open_scenes'] = using_map_list[ix: ix + delta]
            machines_info[index]['gpus'] = [args.gpu_id] * len(machines_info[index]['open_scenes'])
            ix += delta

        cnt = 0
        for item in machines_info:
            cnt += len(item['open_scenes'])

        assert self.batch_size == cnt, 'error create machines_info'

        if self.this_scene_used_cnt < self.one_scene_could_use_num and \
                len(set(using_map_list)) == 1 and len(set(self.last_using_map_list)) == 1 and \
                using_map_list[0] is not None and self.last_using_map_list[0] is not None and \
                using_map_list[0] == self.last_using_map_list[0] and \
                need_change == False:
            self.this_scene_used_cnt += 1
            logger.warning('no need to change env: {}'.format(using_map_list))
            return

        logger.warning('to change env: {}'.format(using_map_list))

        while True:
            try:
                self.machines_info = copy.deepcopy(machines_info)
                print('machines_info:', self.machines_info)
                self.simulator_tool = AirVLNSimulatorClientTool(machines_info=self.machines_info)
                self.simulator_tool.run_call()
                break

            except Exception as e:
                logger.error("启动场景失败 {}".format(e))
                time.sleep(3)

            except:
                logger.error('启动场景失败')
                time.sleep(3)

        self.last_using_map_list = using_map_list.copy()
        self.this_scene_used_cnt = 1

    def get_obs(self):
        obs_states = self._getStates()
        obs, states = self.VectorEnvUtil.get_obs(obs_states)
        self.sim_states = states

        return obs

    def _getStates(self):
        responses = self.simulator_tool.getImageResponses()

        cnt = 0
        for item in responses:
            cnt += len(item)

        assert len(responses) == len(self.machines_info), 'error'
        assert cnt == self.batch_size, 'error'

        states = [None for _ in range(self.batch_size)]
        cnt = 0

        for index_1, item in enumerate(self.machines_info):
            for index_2 in range(len(item['open_scenes'])):
                rgb_images = responses[index_1][index_2][0]
                depth_images = responses[index_1][index_2][1]
                state = self.sim_states[cnt]
                states[cnt] = (rgb_images, depth_images, state)
                cnt += 1

        return states

    def _get_current_state(self) -> list:
        states = []
        cnt = 0

        for index_1, item in enumerate(self.machines_info):
            states.append([])

            for index_2, _ in enumerate(item['open_scenes']):
                s = self.sim_states[cnt].state
                state = airsim.KinematicsState()

                state.position = airsim.Vector3r(*s['position'])
                orientation = s.get('orientation', s.get('quaternionr', [0.0, 0.0, 0.0, 1.0]))
                state.orientation = airsim.Quaternionr(*orientation)

                state.linear_velocity = airsim.Vector3r(*s.get('linear_velocity', [0.0, 0.0, 0.0]))
                state.angular_velocity = airsim.Vector3r(*s.get('angular_velocity', [0.0, 0.0, 0.0]))

                states[index_1].append(state)
                cnt += 1

        return states

    def _get_current_pose(self) -> list:
        poses = []
        cnt = 0

        for index_1, item in enumerate(self.machines_info):
            poses.append([])

            for index_2, _ in enumerate(item['open_scenes']):
                poses[index_1].append(self.sim_states[cnt].pose)
                cnt += 1

        return poses

    def reset(self):
        self.changeToNewTask()
        return self.get_obs()

    def makeActions(self, action_list, steps_size, is_fixed, planned_paths=None):
        poses = []
        fly_types = []
        current_airsim_poses = []
        executed_actions = []
        execute_infos = []
        already_ended = []

        for index, action in enumerate(action_list):
            already_ended.append(bool(self.sim_states[index].is_end))

            if self.sim_states[index].is_end == True:
                action = 'stop'

            if action == 'stop' or self.sim_states[index].step >= int(args.maxActions):
                self.sim_states[index].is_end = True

            airsim_pose = self.to_airsim_pose(self.sim_states[index].pose)
            current_airsim_poses.append(self.copy_airsim_pose(airsim_pose))

            use_continuous, planned_path, plan_reason = self.get_planned_path_for_episode(
                planned_paths=planned_paths,
                index=index,
            )

            if use_continuous and action != 'stop' and self.sim_states[index].is_end == False:
                try:
                    new_pose, fly_type, execute_info = self.get_continuous_next_pose(
                        airsim_pose=airsim_pose,
                        planned_path=planned_path,
                        index=index,
                    )
                    executed_actions.append('continuous')
                    execute_infos.append(execute_info)

                except Exception as e:
                    new_pose = self.copy_airsim_pose(airsim_pose)
                    fly_type = "move"
                    executed_actions.append('continuous_hold')
                    execute_infos.append(
                        self.build_hold_execute_info(
                            reason="continuous next pose failed: {}".format(e),
                            planned_path=planned_path,
                        )
                    )
                    print(
                        "[Continuous Hold] "
                        f"Episode {index}: "
                        f"reason=continuous next pose failed: {e}"
                    )

            else:
                new_pose = self.copy_airsim_pose(airsim_pose)
                fly_type = "move"

                if action == 'stop':
                    hold_reason = "stop action"
                    executed_action = "stop"
                elif self.sim_states[index].is_end == True:
                    hold_reason = "episode ended"
                    executed_action = "stop"
                else:
                    hold_reason = plan_reason
                    executed_action = "continuous_hold"

                executed_actions.append(executed_action)
                execute_infos.append(
                    self.build_hold_execute_info(
                        reason=hold_reason,
                        planned_path=planned_path,
                    )
                )

                if executed_action == "continuous_hold":
                    print(
                        "[Continuous Hold] "
                        f"Episode {index}: "
                        f"reason={hold_reason}"
                    )

            prev_pitch, prev_roll, prev_yaw = airsim.to_eularian_angles(airsim_pose.orientation)
            curr_pitch, curr_roll, curr_yaw = airsim.to_eularian_angles(new_pose.orientation)
            delta_yaw = abs((math.degrees(curr_yaw - prev_yaw) + 180) % 360 - 180)
            self.sim_states[index].heading_changes.append(delta_yaw)

            self.update_oracle_success_from_pose(index, new_pose)

            poses.append(new_pose)
            fly_types.append(fly_type)

        format_pose, format_fly_type = self.format_actions_for_machines(
            poses=poses,
            fly_types=fly_types,
        )

        raw_result, move_ok, move_reason = self.safe_move_to_next_pose(
            poses_list=format_pose,
            fly_types=format_fly_type,
        )

        result, result_ok, result_reason = self.normalize_move_result(
            result=raw_result,
            move_ok=move_ok,
            move_reason=move_reason,
        )

        execution_result_infos = self.apply_move_result_to_states(
            result=result,
            result_ok=result_ok,
            result_reason=result_reason,
            already_ended=already_ended,
        )

        if not result_ok:
            poses = [
                self.copy_airsim_pose(current_airsim_poses[index])
                for index in range(self.batch_size)
            ]

        for index, action in enumerate(action_list):
            if already_ended[index]:
                continue

            if action == 'stop':
                continue

            if self.sim_states[index].step >= int(args.maxActions):
                self.sim_states[index].is_end = True

            self.sim_states[index].step += 1

            traj = self.sim_states[index].trajectory

            if len(traj) >= 1:
                p_prev = np.array(traj[-1]['sensors']['state']['position'])
            else:
                p_prev = np.array([
                    current_airsim_poses[index].position.x_val,
                    current_airsim_poses[index].position.y_val,
                    current_airsim_poses[index].position.z_val,
                ])

            p_curr = np.array([
                poses[index].position.x_val,
                poses[index].position.y_val,
                poses[index].position.z_val,
            ])

            step_dist = np.linalg.norm(p_curr - p_prev)
            self.sim_states[index].move_distance += step_dist

            target = np.array(self.sim_states[index].target_position)
            distance_to_target = float(np.linalg.norm(p_curr - target))

            trajectory_info = {
                'sensors': {
                    'state': {
                        'position': [
                            poses[index].position.x_val,
                            poses[index].position.y_val,
                            poses[index].position.z_val,
                        ],
                        'quaternionr': [
                            poses[index].orientation.x_val,
                            poses[index].orientation.y_val,
                            poses[index].orientation.z_val,
                            poses[index].orientation.w_val,
                        ],
                    }
                },
                'move_distance': round(self.sim_states[index].move_distance, 2),
                'distance_to_target': round(distance_to_target, 2),
                'execute_type': executed_actions[index],
            }

            trajectory_info.update(execute_infos[index])
            trajectory_info.update(execution_result_infos[index])
            self.sim_states[index].trajectory.append(trajectory_info)

    def to_airsim_pose(self, current_pose):
        if isinstance(current_pose, list):
            pos = current_pose[:3]
            quat = current_pose[3:]

            return airsim.Pose(
                airsim.Vector3r(*pos),
                airsim.Quaternionr(
                    x_val=quat[0],
                    y_val=quat[1],
                    z_val=quat[2],
                    w_val=quat[3],
                ),
            )

        return current_pose

    def copy_airsim_pose(self, airsim_pose):
        return airsim.Pose(
            airsim.Vector3r(
                x_val=airsim_pose.position.x_val,
                y_val=airsim_pose.position.y_val,
                z_val=airsim_pose.position.z_val,
            ),
            airsim.Quaternionr(
                x_val=airsim_pose.orientation.x_val,
                y_val=airsim_pose.orientation.y_val,
                z_val=airsim_pose.orientation.z_val,
                w_val=airsim_pose.orientation.w_val,
            ),
        )

    def update_oracle_success_from_pose(self, index, pose):
        pos = pose.position
        curr = np.array([pos.x_val, pos.y_val, pos.z_val])
        coords = np.array(self.batch[index]["object_position"])

        if coords.ndim == 2 and coords.shape[1] == 3:
            dists = np.linalg.norm(coords - curr[None, :], axis=1)
            min_dist = dists.min()
        else:
            min_dist = np.linalg.norm(curr - coords)

        if min_dist < self.sim_states[index].SUCCESS_DISTANCE:
            self.sim_states[index].oracle_success = True

    def build_hold_execute_info(self, reason, planned_path=None):
        if isinstance(planned_path, dict):
            planned_path_len = int(planned_path.get("path_len", 0))
            planned_path_length = float(planned_path.get("path_length", 0.0))
        else:
            planned_path_len = 0
            planned_path_length = 0.0

        return {
            "execute_type": "continuous_hold",
            "next_waypoint": None,
            "step_distance": 0.0,
            "max_continuous_step": self.max_continuous_step,
            "is_step_limited": False,
            "planned_path_len": planned_path_len,
            "planned_path_length": planned_path_length,
            "hold_reason": reason,
        }

    def get_planned_path_for_episode(self, planned_paths, index):
        if planned_paths is None:
            return False, None, "planned_paths is None"

        try:
            if isinstance(planned_paths, list):
                planned_path = planned_paths[index]
            elif isinstance(planned_paths, dict):
                planned_path = planned_paths.get(index, None)
            else:
                return False, None, "planned_paths has invalid type"

            if planned_path is None:
                return False, None, "planned_path is None"

            if not isinstance(planned_path, dict):
                return False, None, "planned_path is not a dict"

            if not planned_path.get("valid", False):
                return False, planned_path, planned_path.get("reason", "planned_path is invalid")

            path = planned_path.get("path", [])

            if path is None or len(path) < 2:
                return False, planned_path, "planned_path has fewer than 2 points"

            return True, planned_path, "valid planned_path"

        except Exception as e:
            print(f"[WARNING] failed to get planned path for episode {index}: {e}")
            return False, None, str(e)

    def format_actions_for_machines(self, poses, fly_types):
        format_pose = []
        format_fly_type = []
        cnt = 0

        for index1, item in enumerate(self.machines_info):
            format_pose.append([])
            format_fly_type.append([])

            for index2, _ in enumerate(item['open_scenes']):
                format_pose[index1].append(poses[cnt])
                format_fly_type[index1].append(fly_types[cnt])
                cnt += 1

        return format_pose, format_fly_type

    def safe_move_to_next_pose(self, poses_list, fly_types):
        try:
            result = self.simulator_tool.move_to_next_pose(
                poses_list=poses_list,
                fly_types=fly_types,
            )

            if not result:
                logger.error('move_to_next_pose error: empty result')
                return result, False, "move_to_next_pose returned empty result"

            return result, True, "ok"

        except Exception as e:
            logger.error("move_to_next_pose exception: {}".format(e))
            print(f"[Execution Failure] move_to_next_pose exception: {e}")
            return None, False, "move_to_next_pose exception: {}".format(e)

        except:
            logger.error("move_to_next_pose unknown exception")
            print("[Execution Failure] move_to_next_pose unknown exception")
            return None, False, "move_to_next_pose unknown exception"

    def normalize_move_result(self, result, move_ok, move_reason):
        if not move_ok:
            return self.build_failed_move_result(move_reason), False, move_reason

        if not isinstance(result, list):
            return (
                self.build_failed_move_result("move result is not a list"),
                False,
                "move result is not a list",
            )

        if len(result) != len(self.machines_info):
            return (
                self.build_failed_move_result("move result machine count mismatch"),
                False,
                "move result machine count mismatch",
            )

        normalized_result = []
        global_valid = True
        invalid_reason = "ok"

        for machine_index, machine_info in enumerate(self.machines_info):
            machine_result = result[machine_index]
            expected_scene_count = len(machine_info['open_scenes'])

            if not isinstance(machine_result, list):
                global_valid = False
                invalid_reason = "move result scene result is not a list"
                normalized_result.append(
                    self.build_failed_machine_result(
                        count=expected_scene_count,
                        reason=invalid_reason,
                    )
                )
                continue

            if len(machine_result) != expected_scene_count:
                global_valid = False
                invalid_reason = "move result scene count mismatch"
                normalized_result.append(
                    self.build_failed_machine_result(
                        count=expected_scene_count,
                        reason=invalid_reason,
                    )
                )
                continue

            normalized_machine_result = []

            for scene_index in range(expected_scene_count):
                item = machine_result[scene_index]

                if isinstance(item, dict):
                    normalized_machine_result.append({
                        "collision": bool(item.get("collision", False)),
                        "execution_failed": bool(item.get("execution_failed", False)),
                        "execution_reason": item.get("execution_reason", "ok"),
                    })
                else:
                    global_valid = False
                    invalid_reason = "move result item is not a dict"
                    normalized_machine_result.append({
                        "collision": True,
                        "execution_failed": True,
                        "execution_reason": invalid_reason,
                    })

            normalized_result.append(normalized_machine_result)

        return normalized_result, global_valid, invalid_reason

    def build_failed_move_result(self, reason):
        result = []

        for machine_info in self.machines_info:
            result.append(
                self.build_failed_machine_result(
                    count=len(machine_info['open_scenes']),
                    reason=reason,
                )
            )

        return result

    def build_failed_machine_result(self, count, reason):
        return [
            {
                "collision": True,
                "execution_failed": True,
                "execution_reason": reason,
            }
            for _ in range(count)
        ]

    def apply_move_result_to_states(self, result, result_ok, result_reason, already_ended):
        execution_infos = [
            {
                "execution_failed": False,
                "execution_reason": "ok",
                "move_result_valid": bool(result_ok),
            }
            for _ in range(self.batch_size)
        ]

        cnt = 0

        for index1, item in enumerate(self.machines_info):
            for index2, _ in enumerate(item['open_scenes']):
                result_item = result[index1][index2]
                collision = bool(result_item.get('collision', False))
                execution_failed = bool(result_item.get('execution_failed', False))
                execution_reason = result_item.get('execution_reason', result_reason)

                self.sim_states[cnt].is_collisioned = collision

                if execution_failed:
                    self.sim_states[cnt].is_collisioned = True

                    if not already_ended[cnt]:
                        self.sim_states[cnt].is_end = True

                    print(
                        "[Execution Failure] "
                        f"Episode {cnt}: "
                        f"reason={execution_reason}"
                    )

                elif collision:
                    if not already_ended[cnt]:
                        self.sim_states[cnt].is_end = True

                    print(
                        "[Execution Collision] "
                        f"Episode {cnt}: collision=True"
                    )

                execution_infos[cnt] = {
                    "execution_failed": execution_failed,
                    "execution_reason": execution_reason,
                    "move_result_valid": bool(result_ok),
                    "move_result_reason": result_reason,
                    "collision_after_execute": bool(self.sim_states[cnt].is_collisioned),
                }

                cnt += 1

        return execution_infos

    def get_continuous_next_pose(self, airsim_pose, planned_path, index):
        path = planned_path.get("path", [])
        current_position = np.array([
            airsim_pose.position.x_val,
            airsim_pose.position.y_val,
            airsim_pose.position.z_val,
        ], dtype=float)

        waypoint, step_distance, is_step_limited = self.select_next_waypoint(
            current_position=current_position,
            path=path,
            max_step_distance=self.max_continuous_step,
            min_advance_distance=self.min_continuous_advance,
        )

        target_position = np.array([
            waypoint[0],
            waypoint[1],
            waypoint[2],
        ], dtype=float)

        current_orientation = airsim_pose.orientation

        new_pose = airsim.Pose(
            airsim.Vector3r(
                float(target_position[0]),
                float(target_position[1]),
                float(target_position[2]),
            ),
            airsim.Quaternionr(
                x_val=current_orientation.x_val,
                y_val=current_orientation.y_val,
                z_val=current_orientation.z_val,
                w_val=current_orientation.w_val,
            ),
        )

        print(
            "[Continuous Execute] "
            f"Episode {index}: "
            f"next_waypoint=({round(float(target_position[0]), 2)}, "
            f"{round(float(target_position[1]), 2)}, "
            f"{round(float(target_position[2]), 2)}), "
            f"step_distance={round(float(step_distance), 2)}, "
            f"max_step={self.max_continuous_step}, "
            f"limited={is_step_limited}, "
            f"path_len={planned_path.get('path_len', len(path))}, "
            f"path_length={planned_path.get('path_length', 0.0)}"
        )

        execute_info = {
            "execute_type": "continuous",
            "next_waypoint": [
                round(float(target_position[0]), 2),
                round(float(target_position[1]), 2),
                round(float(target_position[2]), 2),
            ],
            "step_distance": round(float(step_distance), 2),
            "max_continuous_step": self.max_continuous_step,
            "is_step_limited": bool(is_step_limited),
            "planned_path_len": int(planned_path.get('path_len', len(path))),
            "planned_path_length": float(planned_path.get('path_length', 0.0)),
        }

        return new_pose, "move", execute_info

    def select_next_waypoint(
        self,
        current_position,
        path,
        max_step_distance=5.0,
        min_advance_distance=0.5,
    ):
        if len(path) == 0:
            waypoint = (
                float(current_position[0]),
                float(current_position[1]),
                float(current_position[2]),
            )
            return waypoint, 0.0, False

        if len(path) == 1:
            waypoint_np = np.array(path[0], dtype=float)
            dist = np.linalg.norm(waypoint_np[:3] - current_position[:3])

            if dist <= max_step_distance:
                return path[0], float(dist), False

            direction = (waypoint_np[:3] - current_position[:3]) / max(dist, 1e-6)
            bounded_waypoint = current_position[:3] + direction * max_step_distance

            return (
                float(bounded_waypoint[0]),
                float(bounded_waypoint[1]),
                float(bounded_waypoint[2]),
            ), float(max_step_distance), True

        current = current_position[:3].astype(float)
        remain_distance = float(max_step_distance)
        total_advance = 0.0

        for waypoint in path[1:]:
            waypoint_np = np.array(waypoint, dtype=float)
            segment = waypoint_np[:3] - current[:3]
            segment_length = float(np.linalg.norm(segment))

            if segment_length < 1e-6:
                continue

            if segment_length <= remain_distance:
                current = waypoint_np[:3]
                remain_distance -= segment_length
                total_advance += segment_length

                if remain_distance <= 1e-6:
                    return (
                        float(current[0]),
                        float(current[1]),
                        float(current[2]),
                    ), float(total_advance), True

                continue

            direction = segment / max(segment_length, 1e-6)
            bounded_waypoint = current[:3] + direction * remain_distance
            total_advance += remain_distance

            return (
                float(bounded_waypoint[0]),
                float(bounded_waypoint[1]),
                float(bounded_waypoint[2]),
            ), float(total_advance), True

        final_waypoint = np.array(path[-1], dtype=float)
        final_dist = float(np.linalg.norm(final_waypoint[:3] - current_position[:3]))

        if final_dist < min_advance_distance and len(path) >= 2:
            fallback_waypoint = np.array(path[1], dtype=float)
            fallback_dist = float(np.linalg.norm(fallback_waypoint[:3] - current_position[:3]))

            if fallback_dist > max_step_distance:
                direction = (fallback_waypoint[:3] - current_position[:3]) / max(fallback_dist, 1e-6)
                bounded_waypoint = current_position[:3] + direction * max_step_distance

                return (
                    float(bounded_waypoint[0]),
                    float(bounded_waypoint[1]),
                    float(bounded_waypoint[2]),
                ), float(max_step_distance), True

            return path[1], float(fallback_dist), False

        is_step_limited = final_dist > max_step_distance

        if is_step_limited:
            direction = (final_waypoint[:3] - current_position[:3]) / max(final_dist, 1e-6)
            bounded_waypoint = current_position[:3] + direction * max_step_distance

            return (
                float(bounded_waypoint[0]),
                float(bounded_waypoint[1]),
                float(bounded_waypoint[2]),
            ), float(max_step_distance), True

        return path[-1], float(final_dist), False

    def update_measurements(self):
        self._update_distance_to_target()

    def _update_distance_to_target(self):
        target_positions = [item['object_position'] for item in self.batch]

        for idx, target_position in enumerate(target_positions):
            curr = np.array(self.sim_states[idx].pose[0:3])
            coords = np.array(target_position)

            if coords.ndim == 2 and coords.shape[1] == 3:
                dists = np.linalg.norm(coords - curr[None, :], axis=1)
                distance = float(dists.min())
            else:
                distance = float(np.linalg.norm(curr - coords))

            print(
                f'batch[{idx}/{len(self.batch)}]| '
                f'distance: {round(distance, 2)}, '
                f'position: {curr[0]}, {curr[1]}, {curr[2]}, '
                f'target: {coords}'
            )
