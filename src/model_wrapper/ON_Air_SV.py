from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import airsim
import numpy as np

from common.param import args
from model_wrapper.ON_Air_2 import ONAir

from svnav.keyframe_manager import KeyframeManagerConfig, Task1KeyframeManager
from svnav.navigator import SearchNavigator, SearchNavigatorConfig
from svnav.semantic_map import SemanticMap, SemanticMapConfig
from svnav.task1_reasoner import Task1Reasoner, Task1ReasonerConfig
from svnav.types import (
    FrameRecord,
    ObservationRecord,
    PoseRecord,
    TargetInfo,
    Task1Result,
    ViewID,
    now_ts,
)


@dataclass
class SVNAVEpisodeState:
    episode_id: str
    origin_pose: PoseRecord
    target_info: TargetInfo
    semantic_map: SemanticMap
    keyframe_manager: Task1KeyframeManager
    navigator: SearchNavigator

    created_at: float = field(default_factory=now_ts)
    last_step_id: int = -1
    last_observation_id: Optional[str] = None
    last_task1_request_id: Optional[str] = None
    last_task1_result_id: Optional[str] = None
    last_task1_update_summary: Optional[Dict[str, Any]] = None
    last_task1_cleanup_summary: Optional[Dict[str, Any]] = None
    last_nav_decision: Optional[Dict[str, Any]] = None

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "origin_pose": self.origin_pose.to_log_dict(),
            "target_info": self.target_info.to_log_dict(),
            "created_at": self.created_at,
            "last_step_id": self.last_step_id,
            "last_observation_id": self.last_observation_id,
            "last_task1_request_id": self.last_task1_request_id,
            "last_task1_result_id": self.last_task1_result_id,
            "last_task1_update_summary": self.last_task1_update_summary,
            "last_task1_cleanup_summary": self.last_task1_cleanup_summary,
            "last_nav_decision": self.last_nav_decision,
            "semantic_map": self.semantic_map.get_summary(),
            "keyframe_manager": self.keyframe_manager.to_log_dict(),
        }


class ONAirSV(ONAir):
    """
    Semantic-Value Navigation wrapper.

    Current stage:
        Task1 semantic map + Search navigation.

    This wrapper no longer asks the LLM to output navigation actions.
    It uses:
        Task1 keyframe selection
        Task1 VLM semantic scoring
        SemanticMap update
        SearchNavigator + ActionAdapter
    """

    def __init__(self, fixed, batch_size):
        super().__init__(fixed=fixed, batch_size=batch_size)
        self.method_name = "SVNav"

        self.batch_size = batch_size
        self.svnav_states: Dict[str, SVNAVEpisodeState] = {}
        self.batch_episode_ids: List[Optional[str]] = [None for _ in range(batch_size)]

        self.task1_reasoner = Task1Reasoner(
            Task1ReasonerConfig(
                model_name="qwen-vl-max",
                max_retries=1,
                retry_sleep=0.5,
            )
        )

    # ------------------------------------------------------------------
    # Main eval interface
    # ------------------------------------------------------------------

    def prepare_inputs(self, episodes, fixed):
        """
        Build SVNav runtime inputs from UAV-ON episodes.

        Return value keeps the same external shape as baseline:
            inputs, user_prompts

        But inputs are no longer GPT conversations. Each input is a dict
        containing the current ObservationRecord and its SVNav episode state.
        """
        sv_inputs = []
        user_prompts = []

        for batch_index in range(len(episodes)):
            try:
                sources = episodes[batch_index]
                latest = self._latest_source_with_observation(sources)
                episode_meta = sources[-1]

                episode_id = self._build_episode_id(
                    source=episode_meta,
                    batch_index=batch_index,
                )
                target_info = self._extract_target_info(episode_meta)
                origin_pose = self._extract_origin_pose(episode_meta)
                current_pose = self._extract_current_pose(episode_meta, origin_pose)

                self.start_position[batch_index] = [
                    origin_pose.x,
                    origin_pose.y,
                    origin_pose.z,
                ]
                self.start_yaw[batch_index] = origin_pose.yaw
                self.current_poses[batch_index] = [
                    current_pose.x,
                    current_pose.y,
                    current_pose.z,
                    current_pose.yaw,
                ]
                self.batch_episode_ids[batch_index] = episode_id

                state = self._ensure_episode_state(
                    episode_id=episode_id,
                    origin_pose=origin_pose,
                    target_info=target_info,
                )

                step_id = int(episode_meta.get("step", state.last_step_id + 1))
                observation = self._build_observation_record(
                    episode_id=episode_id,
                    step_id=step_id,
                    pose=current_pose,
                    target_info=target_info,
                    latest_source=latest,
                )

                state.last_step_id = step_id
                state.last_observation_id = observation.observation_id

                self._update_semantic_map_from_observation(
                    state=state,
                    observation=observation,
                    step_id=step_id,
                )

                prompt_info = self._build_prompt_info(
                    state=state,
                    observation=observation,
                )

                sv_inputs.append(
                    {
                        "episode_id": episode_id,
                        "batch_index": batch_index,
                        "step_id": step_id,
                        "observation": observation,
                        "state": state,
                        "prompt_info": prompt_info,
                    }
                )
                user_prompts.append(prompt_info)

            except Exception as exc:
                episode_id = "batch_{}_error".format(batch_index)
                sv_inputs.append(
                    {
                        "episode_id": episode_id,
                        "batch_index": batch_index,
                        "step_id": -1,
                        "observation": None,
                        "state": None,
                        "error": str(exc),
                        "prompt_info": "[SVNav] prepare_inputs error: {}".format(exc),
                    }
                )
                user_prompts.append("[SVNav] prepare_inputs error: {}".format(exc))

        return sv_inputs, user_prompts

    def run(self, inputs, fixed, prompt_info_list=None):
        actions = []
        steps_size = []
        predict_dones = []

        for item in inputs:
            try:
                if item.get("error"):
                    actions.append("rotl")
                    steps_size.append(float(getattr(args, "rotateAngle", 15)))
                    predict_dones.append(False)
                    continue

                state: SVNAVEpisodeState = item["state"]
                observation: ObservationRecord = item["observation"]
                episode_id = item["episode_id"]
                step_id = item["step_id"]

                decision = state.navigator.decide_search(
                    episode_id=episode_id,
                    step_id=step_id,
                    observation=observation,
                    semantic_map=state.semantic_map,
                )

                state.last_nav_decision = decision.to_log_dict()

                action = decision.action or "rotl"
                step_size = decision.step_size
                if step_size is None:
                    step_size = self._default_step_size_for_action(action)

                actions.append(action)
                steps_size.append(float(step_size))
                predict_dones.append(False)

                print(
                    "[SVNavSearch] episode={} step={} action={} step_size={} source={} reason={}".format(
                        episode_id,
                        step_id,
                        action,
                        step_size,
                        decision.action_source.value,
                        decision.reason,
                    )
                )

            except Exception as exc:
                actions.append("rotl")
                steps_size.append(float(getattr(args, "rotateAngle", 15)))
                predict_dones.append(False)
                print("[SVNavSearch] run error: {}".format(exc))

        return actions, steps_size, predict_dones

    # ------------------------------------------------------------------
    # SVNav step update
    # ------------------------------------------------------------------

    def _update_semantic_map_from_observation(
        self,
        state: SVNAVEpisodeState,
        observation: ObservationRecord,
        step_id: int,
    ) -> None:
        state.semantic_map.mark_visited(observation.pose, step_id=step_id)
        state.semantic_map.decay(step_id)

        keyframe_result = state.keyframe_manager.observe(
            observation=observation,
            semantic_map=state.semantic_map,
            build_request=True,
        )

        request = keyframe_result.task1_request
        if request is None:
            return

        state.last_task1_request_id = request.request_id

        task1_result = self.task1_reasoner.run(
            request=request,
            return_step=step_id,
        )

        apply_summary = self.apply_task1_result_to_map(
            episode_id=state.episode_id,
            task1_result=task1_result,
        )

        if apply_summary.get("accepted"):
            print(
                "[SVNavTask1] episode={} step={} request={} success={} updated_cells={}".format(
                    state.episode_id,
                    step_id,
                    request.request_id,
                    task1_result.success,
                    apply_summary.get("update_summary", {}).get("updated_cells", 0),
                )
            )
        else:
            print(
                "[SVNavTask1] episode={} step={} request={} rejected={}".format(
                    state.episode_id,
                    step_id,
                    request.request_id,
                    apply_summary.get("reason"),
                )
            )

    def apply_task1_result_to_map(
        self,
        episode_id: str,
        task1_result: Task1Result,
    ) -> Dict[str, Any]:
        episode_id = str(episode_id)
        state = self.svnav_states.get(episode_id)

        if state is None:
            return {
                "accepted": False,
                "reason": "missing_svnav_episode_state",
                "episode_id": episode_id,
                "request_id": task1_result.request_id,
            }

        if not state.keyframe_manager.should_accept_task1_result(task1_result):
            return {
                "accepted": False,
                "reason": "stale_or_unknown_task1_result",
                "episode_id": episode_id,
                "request_id": task1_result.request_id,
                "result_episode_id": task1_result.episode_id,
                "current_episode_id": state.episode_id,
            }

        frame_lookup = state.keyframe_manager.get_request_frame_lookup(
            task1_result.request_id
        )

        update_summary = state.semantic_map.update_from_task1_result(
            task1_result,
            frame_lookup,
        )

        cleanup_summary = state.keyframe_manager.mark_request_completed(
            task1_result.request_id
        )

        state.last_task1_result_id = task1_result.request_id
        state.last_task1_update_summary = update_summary
        state.last_task1_cleanup_summary = cleanup_summary

        return {
            "accepted": True,
            "episode_id": episode_id,
            "request_id": task1_result.request_id,
            "update_summary": update_summary,
            "cleanup_summary": cleanup_summary,
            "map_summary": state.semantic_map.get_summary(),
        }

    def apply_task1_result(
        self,
        episode_id: str,
        task1_result: Task1Result,
    ) -> Dict[str, Any]:
        return self.apply_task1_result_to_map(
            episode_id=episode_id,
            task1_result=task1_result,
        )

    # ------------------------------------------------------------------
    # Episode state
    # ------------------------------------------------------------------

    def _ensure_episode_state(
        self,
        episode_id: str,
        origin_pose: PoseRecord,
        target_info: TargetInfo,
    ) -> SVNAVEpisodeState:
        episode_id = str(episode_id)

        state = self.svnav_states.get(episode_id)
        if state is not None:
            return state

        return self.init_svnav_episode(
            episode_id=episode_id,
            origin_pose=origin_pose,
            target_info=target_info,
            search_radius=self._default_search_radius(target_info),
            cell_size=self._default_cell_size(),
        )

    def init_svnav_episode(
        self,
        episode_id: str,
        origin_pose: Any,
        target_info: Any,
        search_radius: Optional[float] = None,
        cell_size: Optional[float] = None,
        semantic_map_kwargs: Optional[Dict[str, Any]] = None,
        keyframe_config: Optional[KeyframeManagerConfig] = None,
        navigator_config: Optional[SearchNavigatorConfig] = None,
    ) -> SVNAVEpisodeState:
        episode_id = str(episode_id)
        origin = self._coerce_pose_record(origin_pose)
        target = self._coerce_target_info(target_info)

        if search_radius is None:
            search_radius = self._default_search_radius(target)
        if cell_size is None:
            cell_size = self._default_cell_size()

        kwargs = semantic_map_kwargs or {}
        map_config = SemanticMapConfig.from_uavon_start(
            start_pose=origin,
            search_radius=search_radius,
            cell_size=cell_size,
            **kwargs,
        )
        semantic_map = SemanticMap(
            origin_pose=origin,
            config=map_config,
        )

        keyframe_manager = Task1KeyframeManager(
            config=keyframe_config or KeyframeManagerConfig()
        )
        keyframe_manager.reset_episode(episode_id)

        navigator = SearchNavigator(
            config=navigator_config or SearchNavigatorConfig()
        )

        state = SVNAVEpisodeState(
            episode_id=episode_id,
            origin_pose=origin,
            target_info=target,
            semantic_map=semantic_map,
            keyframe_manager=keyframe_manager,
            navigator=navigator,
        )

        self.svnav_states[episode_id] = state
        return state

    def reset_svnav_episode(
        self,
        episode_id: str,
        origin_pose: Any,
        target_info: Any,
        search_radius: Optional[float] = None,
        cell_size: Optional[float] = None,
        semantic_map_kwargs: Optional[Dict[str, Any]] = None,
        keyframe_config: Optional[KeyframeManagerConfig] = None,
        navigator_config: Optional[SearchNavigatorConfig] = None,
    ) -> SVNAVEpisodeState:
        return self.init_svnav_episode(
            episode_id=episode_id,
            origin_pose=origin_pose,
            target_info=target_info,
            search_radius=search_radius,
            cell_size=cell_size,
            semantic_map_kwargs=semantic_map_kwargs,
            keyframe_config=keyframe_config,
            navigator_config=navigator_config,
        )

    def get_svnav_state(self, episode_id: str) -> Optional[SVNAVEpisodeState]:
        return self.svnav_states.get(str(episode_id))

    def get_semantic_map_summary(
        self,
        episode_id: str,
        top_k: int = 5,
    ) -> Optional[Dict[str, Any]]:
        state = self.get_svnav_state(episode_id)
        if state is None:
            return None
        return state.semantic_map.get_summary(top_k=top_k)

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _latest_source_with_observation(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        for src in sources[::-1]:
            if "rgb" in src and "depth" in src:
                return src
        return sources[-1]

    def _build_observation_record(
        self,
        episode_id: str,
        step_id: int,
        pose: PoseRecord,
        target_info: TargetInfo,
        latest_source: Dict[str, Any],
    ) -> ObservationRecord:
        rgb_images = latest_source.get("rgb", [])
        depth_images = latest_source.get("depth", [])

        depth_grid3x3 = self._process_depth_safe(depth_images)

        frames: Dict[ViewID, FrameRecord] = {}
        views = [ViewID.FRONT, ViewID.LEFT, ViewID.RIGHT, ViewID.DOWN]

        for idx, view_id in enumerate(views):
            rgb_item = rgb_images[idx] if idx < len(rgb_images) else None
            depth_item = depth_images[idx] if idx < len(depth_images) else None
            depth_grid = depth_grid3x3[idx] if idx < len(depth_grid3x3) else None

            frame = FrameRecord(
                episode_id=episode_id,
                step_id=step_id,
                view_id=view_id,
                pose=pose,
                target_info=target_info,
                rgb_bytes=self._image_to_bytes(rgb_item),
                rgb_b64=self._image_to_b64(rgb_item),
                depth=depth_item,
                depth_grid3x3=depth_grid,
                image_width=None,
                image_height=None,
                metadata={
                    "source": "uavon_observation",
                    "view_index": idx,
                },
            )
            frames[view_id] = frame

        return ObservationRecord(
            episode_id=episode_id,
            step_id=step_id,
            pose=pose,
            target_info=target_info,
            frames=frames,
            metadata={
                "source": "uavon_episode",
            },
        )

    def _process_depth_safe(self, depth_images) -> List[Any]:
        try:
            return self.process_depth(depth_images=depth_images)
        except Exception:
            return [None for _ in range(len(depth_images))]

    @staticmethod
    def _image_to_bytes(value: Any) -> Optional[bytes]:
        if value is None:
            return None

        if isinstance(value, bytes):
            return value

        if isinstance(value, bytearray):
            return bytes(value)

        return None

    @staticmethod
    def _image_to_b64(value: Any) -> Optional[str]:
        if value is None:
            return None

        if isinstance(value, str):
            if value.startswith("data:"):
                return value.split(",", 1)[1] if "," in value else value
            return value

        if isinstance(value, bytes):
            return base64.b64encode(value).decode("utf-8")

        if isinstance(value, bytearray):
            return base64.b64encode(bytes(value)).decode("utf-8")

        return None

    # ------------------------------------------------------------------
    # Episode metadata extraction
    # ------------------------------------------------------------------

    def _build_episode_id(
        self,
        source: Dict[str, Any],
        batch_index: int,
    ) -> str:
        map_name = source.get("map_name") or source.get("scene_name") or source.get("scene") or "scene"
        task_id = source.get("task_id") or source.get("episode_id") or source.get("trajectory_id") or "task"
        object_name = source.get("object_name") or source.get("true_name") or source.get("target") or "object"
        start_position = source.get("start_position")

        if start_position is None and "start_pose" in source:
            start_position = source["start_pose"].get("start_position")

        start_text = "unknown_start"
        if isinstance(start_position, (list, tuple)) and len(start_position) >= 3:
            start_text = "{:.2f}_{:.2f}_{:.2f}".format(
                float(start_position[0]),
                float(start_position[1]),
                float(start_position[2]),
            )

        return "{}:{}:{}:{}:batch{}".format(
            map_name,
            task_id,
            object_name,
            start_text,
            batch_index,
        )

    def _extract_target_info(self, source: Dict[str, Any]) -> TargetInfo:
        name = (
            source.get("object_name")
            or source.get("true_name")
            or source.get("target_name")
            or source.get("target")
            or "target"
        )
        size = source.get("object_size") or source.get("size")
        description = source.get("description")
        instruction = source.get("instruction") or description

        return TargetInfo(
            name=str(name),
            size=None if size is None else str(size),
            description=None if description is None else str(description),
            instruction=None if instruction is None else str(instruction),
            search_radius=float(source.get("search_radius", 100.0)),
            success_threshold=float(source.get("success_threshold", 20.0)),
        )

    def _extract_origin_pose(self, source: Dict[str, Any]) -> PoseRecord:
        start_position = source.get("start_position")
        start_quaternion = source.get("start_quaternionr")

        if start_position is None and "start_pose" in source:
            start_pose = source["start_pose"]
            start_position = start_pose.get("start_position")
            start_quaternion = start_pose.get("start_quaternionr")

        if start_position is None:
            start_position = [0.0, 0.0, 0.0]

        yaw = 0.0
        if start_quaternion is not None and len(start_quaternion) >= 4:
            quaternionr = airsim.Quaternionr(
                x_val=start_quaternion[0],
                y_val=start_quaternion[1],
                z_val=start_quaternion[2],
                w_val=start_quaternion[3],
            )
            _, _, yaw_rad = airsim.to_eularian_angles(quaternionr)
            yaw = math.degrees(yaw_rad)

        return PoseRecord(
            x=float(start_position[0]),
            y=float(start_position[1]),
            z=float(start_position[2]),
            yaw=float(yaw),
            quaternion=None if start_quaternion is None else tuple(start_quaternion),
        )

    def _extract_current_pose(
        self,
        source: Dict[str, Any],
        origin_pose: PoseRecord,
    ) -> PoseRecord:
        pre_poses = source.get("pre_poses", [])

        try:
            raw_poses = self.process_poses(pre_poses)
        except Exception:
            raw_poses = []

        if raw_poses:
            last_pose = raw_poses[-1]
            xyz = last_pose[0]
            yaw = last_pose[1]
            return PoseRecord(
                x=float(xyz[0]),
                y=float(xyz[1]),
                z=float(xyz[2]),
                yaw=float(yaw),
            )

        return PoseRecord(
            x=origin_pose.x,
            y=origin_pose.y,
            z=origin_pose.z,
            yaw=origin_pose.yaw,
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _build_prompt_info(
        self,
        state: SVNAVEpisodeState,
        observation: ObservationRecord,
    ) -> str:
        map_summary = state.semantic_map.get_summary(top_k=3)
        manager_summary = state.keyframe_manager.to_log_dict()

        lines = []
        lines.append("[SVNav]")
        lines.append("episode_id: {}".format(state.episode_id))
        lines.append("step_id: {}".format(observation.step_id))
        lines.append("target: {}".format(state.target_info.text))
        lines.append("map_status_counts: {}".format(map_summary.get("status_counts")))
        lines.append("top_high_value_cells: {}".format(map_summary.get("top_high_value_cells")))
        lines.append("pending_keyframes: {}".format(manager_summary.get("pending_count")))
        lines.append("inflight_task1: {}".format(manager_summary.get("inflight_count")))
        lines.append("last_task1_request_id: {}".format(state.last_task1_request_id))
        lines.append("last_task1_update_summary: {}".format(state.last_task1_update_summary))
        lines.append("last_nav_decision: {}".format(state.last_nav_decision))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Data conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_pose_record(value: Any) -> PoseRecord:
        if isinstance(value, PoseRecord):
            return value

        if isinstance(value, dict):
            if "x" in value and "y" in value and "z" in value:
                return PoseRecord(
                    x=float(value["x"]),
                    y=float(value["y"]),
                    z=float(value["z"]),
                    yaw=float(value.get("yaw", 0.0)),
                    pitch=float(value.get("pitch", 0.0)),
                    roll=float(value.get("roll", 0.0)),
                    quaternion=value.get("quaternion"),
                )

            if "position" in value:
                position = value["position"]
                yaw = value.get("yaw", 0.0)
                return ONAirSV._coerce_pose_record(
                    {
                        "x": position[0],
                        "y": position[1],
                        "z": position[2] if len(position) > 2 else 0.0,
                        "yaw": yaw,
                    }
                )

        if isinstance(value, (list, tuple)) and len(value) >= 3:
            yaw = value[3] if len(value) >= 4 else 0.0
            return PoseRecord(
                x=float(value[0]),
                y=float(value[1]),
                z=float(value[2]),
                yaw=float(yaw),
            )

        raise ValueError("Cannot convert pose to PoseRecord: {}".format(type(value)))

    @staticmethod
    def _coerce_target_info(value: Any) -> TargetInfo:
        if isinstance(value, TargetInfo):
            return value

        if isinstance(value, dict):
            name = (
                value.get("name")
                or value.get("target_name")
                or value.get("object_name")
                or value.get("target")
                or "target"
            )
            return TargetInfo(
                name=str(name),
                size=value.get("size"),
                description=value.get("description"),
                instruction=value.get("instruction") or value.get("prompt"),
                search_radius=float(value.get("search_radius", 50.0)),
                success_threshold=float(value.get("success_threshold", 20.0)),
            )

        text = str(value)
        return TargetInfo(
            name=text,
            instruction=text,
        )

    @staticmethod
    def _default_cell_size() -> float:
        return float(getattr(args, "xOy_step_size", 5.0))

    @staticmethod
    def _default_search_radius(target_info: TargetInfo) -> float:
        if target_info.search_radius is not None:
            return float(target_info.search_radius)
        return 50.0

    @staticmethod
    def _default_step_size_for_action(action: str) -> float:
        if action in ("rotl", "rotr"):
            return float(getattr(args, "rotateAngle", 15))
        return float(getattr(args, "xOy_step_size", 5.0))


__all__ = [
    "ONAirSV",
    "SVNAVEpisodeState",
]
