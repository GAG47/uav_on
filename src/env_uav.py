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
        activate_maps=[]
    ):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.epoch_done = False
        self.seed = seed
        self.collected_keys = set()
        # self.dataset_group_by_scene = dataset_group_by_scene
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
            # traj_info['instruction'] = item['instruction']
            data.append(traj_info)

        return data


    def _group_scenes(self):
        """
        group all objects with their scene name, choose objects which
        """
        scene_sort_keys: OrderedDict[str, int] = {}

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
                self.sim_states[cnt] = SimState(index=cnt, step=0, task_info=self.batch[cnt])
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
            delta = min(self.batch_size, item['MAX_SCENE_NUM'], len(using_map_list) - ix)
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
            # use the current environments
            return

        else:
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
                state.orientation = airsim.Quaternionr(*s['orientation'])
                state.linear_velocity = airsim.Vector3r(*s['linear_velocity'])
                state.angular_velocity = airsim.Vector3r(*s['angular_velocity'])
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
        executed_actions = []
        execute_infos = []

        for index, action in enumerate(action_list):
            if self.sim_states[index].is_end == True:
                action = 'stop'

            if action == 'stop' or self.sim_states[index].step >= int(args.maxActions):
                self.sim_states[index].is_end = True

            current_pose = self.sim_states[index].pose
            if isinstance(current_pose, list):
                pos = current_pose[:3]
                quat = current_pose[3:]
                airsim_pose = airsim.Pose(
                    airsim.Vector3r(*pos),
                    airsim.Quaternionr(
                        x_val=quat[0],
                        y_val=quat[1],
                        z_val=quat[2],
                        w_val=quat[3]
                    )
                )
            else:
                airsim_pose = current_pose

            use_continuous, planned_path = self.get_planned_path_for_episode(
                planned_paths=planned_paths,
                index=index
            )

            if action == 'stop' or self.sim_states[index].is_end == True:
                new_pose = copy.deepcopy(airsim_pose)
                fly_type = "move"
                executed_actions.append('stop')
                execute_infos.append({
                    "execute_type": "stop_hold",
                    "next_waypoint": None,
                    "step_distance": 0.0,
                    "max_continuous_step": self.max_continuous_step,
                    "is_step_limited": False,
                    "planned_path_len": 0,
                    "planned_path_length": 0.0,
                    "hold_reason": "episode stopped"
                })

            elif use_continuous:
                new_pose, fly_type, execute_info = self.get_continuous_next_pose(
                    airsim_pose=airsim_pose,
                    planned_path=planned_path,
                    index=index
                )
                executed_actions.append('continuous')
                execute_infos.append(execute_info)

            else:
                new_pose = copy.deepcopy(airsim_pose)
                fly_type = "move"
                executed_actions.append('continuous_hold')
                execute_infos.append({
                    "execute_type": "continuous_hold",
                    "next_waypoint": None,
                    "step_distance": 0.0,
                    "max_continuous_step": self.max_continuous_step,
                    "is_step_limited": False,
                    "planned_path_len": 0,
                    "planned_path_length": 0.0,
                    "hold_reason": "no valid continuous planned path"
                })
                print(
                    "[Continuous Execute] "
                    f"Episode {index}: no valid planned path; "
                    "holding position instead of using legacy fallback."
                )

            prev_pitch, prev_roll, prev_yaw = airsim.to_eularian_angles(airsim_pose.orientation)
            curr_pitch, curr_roll, curr_yaw = airsim.to_eularian_angles(new_pose.orientation)
            delta_yaw = abs((math.degrees(curr_yaw - prev_yaw) + 180) % 360 - 180)
            self.sim_states[index].heading_changes.append(delta_yaw)

            pos = new_pose.position
            curr = np.array([pos.x_val, pos.y_val, pos.z_val])
            coords = np.array(self.batch[index]["object_position"])
            if coords.ndim == 2 and coords.shape[1] == 3:
                dists = np.linalg.norm(coords - curr[None, :], axis=1)
                min_dist = dists.min()
            else:
                min_dist = np.linalg.norm(curr - coords)

            if min_dist < self.sim_states[index].SUCCESS_DISTANCE:
                self.sim_states[index].oracle_success = True

            poses.append(new_pose)
            fly_types.append(fly_type)

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

        result = self.simulator_tool.move_to_next_pose(
            poses_list=format_pose,
            fly_types=format_fly_type
        )
        if not result:
            logger.error('move_to_next_pose error')

        cnt = 0
        for index1, item in enumerate(self.machines_info):
            for index2, _ in enumerate(item['open_scenes']):
                self.sim_states[cnt].is_collisioned = result[index1][index2]['collision']
                cnt += 1

        for index, action in enumerate(action_list):
            if self.sim_states[index].is_end == True:
                continue

            if action == 'stop' or self.sim_states[index].step >= int(args.maxActions):
                self.sim_states[index].is_end = True

            self.sim_states[index].step += 1

            traj = self.sim_states[index].trajectory
            if len(traj) >= 1:
                p_prev = np.array(traj[-1]['sensors']['state']['position'])
            else:
                p_prev = np.array([
                    poses[index].position.x_val,
                    poses[index].position.y_val,
                    poses[index].position.z_val
                ])

            p_curr = np.array([
                poses[index].position.x_val,
                poses[index].position.y_val,
                poses[index].position.z_val
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
                            poses[index].position.z_val
                        ],
                        'quaternionr': [
                            poses[index].orientation.x_val,
                            poses[index].orientation.y_val,
                            poses[index].orientation.z_val,
                            poses[index].orientation.w_val
                        ]
                    }
                },
                'move_distance': round(self.sim_states[index].move_distance, 2),
                'distance_to_target': round(distance_to_target, 2),
                'execute_type': executed_actions[index],
            }
            trajectory_info.update(execute_infos[index])
            self.sim_states[index].trajectory.append(trajectory_info)

    def get_planned_path_for_episode(self, planned_paths, index):
        if planned_paths is None:
            return False, None

        try:
            if isinstance(planned_paths, list):
                planned_path = planned_paths[index]
            elif isinstance(planned_paths, dict):
                planned_path = planned_paths.get(index, None)
            else:
                return False, None

            if planned_path is None:
                return False, None

            if not isinstance(planned_path, dict):
                return False, None

            if not planned_path.get("valid", False):
                return False, None

            path = planned_path.get("path", [])
            if path is None or len(path) < 2:
                return False, None

            return True, planned_path

        except Exception as e:
            print(f"[WARNING] failed to get planned path for episode {index}: {e}")
            return False, None


    def get_continuous_next_pose(self, airsim_pose, planned_path, index):
        path = planned_path.get("path", [])

        current_position = np.array([
            airsim_pose.position.x_val,
            airsim_pose.position.y_val,
            airsim_pose.position.z_val
        ], dtype=float)

        waypoint, step_distance, is_step_limited = self.select_next_waypoint(
            current_position=current_position,
            path=path,
            max_step_distance=self.max_continuous_step,
            min_advance_distance=self.min_continuous_advance
        )

        target_position = np.array([
            waypoint[0],
            waypoint[1],
            waypoint[2]
        ], dtype=float)

        current_orientation = airsim_pose.orientation

        new_pose = airsim.Pose(
            airsim.Vector3r(
                float(target_position[0]),
                float(target_position[1]),
                float(target_position[2])
            ),
            airsim.Quaternionr(
                x_val=current_orientation.x_val,
                y_val=current_orientation.y_val,
                z_val=current_orientation.z_val,
                w_val=current_orientation.w_val
            )
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
                round(float(target_position[2]), 2)
            ],
            "step_distance": round(float(step_distance), 2),
            "max_continuous_step": self.max_continuous_step,
            "is_step_limited": bool(is_step_limited),
            "planned_path_len": int(planned_path.get('path_len', len(path))),
            "planned_path_length": float(planned_path.get('path_length', 0.0))
        }

        return new_pose, "move", execute_info


    def select_next_waypoint(
        self,
        current_position,
        path,
        max_step_distance=5.0,
        min_advance_distance=0.5
    ):
        if len(path) == 0:
            waypoint = (
                float(current_position[0]),
                float(current_position[1]),
                float(current_position[2])
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
                float(bounded_waypoint[2])
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
                        float(current[2])
                    ), float(total_advance), True

                continue

            direction = segment / max(segment_length, 1e-6)
            bounded_waypoint = current[:3] + direction * remain_distance
            total_advance += remain_distance

            return (
                float(bounded_waypoint[0]),
                float(bounded_waypoint[1]),
                float(bounded_waypoint[2])
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
                    float(bounded_waypoint[2])
                ), float(max_step_distance), True

            return path[1], float(fallback_dist), False

        is_step_limited = final_dist > max_step_distance

        if is_step_limited:
            direction = (final_waypoint[:3] - current_position[:3]) / max(final_dist, 1e-6)
            bounded_waypoint = current_position[:3] + direction * max_step_distance

            return (
                float(bounded_waypoint[0]),
                float(bounded_waypoint[1]),
                float(bounded_waypoint[2])
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