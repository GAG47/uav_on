from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from model_wrapper.ON_Air_2 import ONAir

from svnav.keyframe_manager import KeyframeManagerConfig, Task1KeyframeManager
from svnav.semantic_map import SemanticMap, SemanticMapConfig
from svnav.types import PoseRecord, TargetInfo, Task1Result, now_ts


@dataclass
class SVNAVEpisodeState:
    episode_id: str
    origin_pose: PoseRecord
    target_info: TargetInfo
    semantic_map: SemanticMap
    keyframe_manager: Task1KeyframeManager

    created_at: float = field(default_factory=now_ts)
    last_task1_result_id: Optional[str] = None
    last_task1_update_summary: Optional[Dict[str, Any]] = None
    last_task1_cleanup_summary: Optional[Dict[str, Any]] = None

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "origin_pose": self.origin_pose.to_log_dict(),
            "target_info": self.target_info.to_log_dict(),
            "created_at": self.created_at,
            "last_task1_result_id": self.last_task1_result_id,
            "last_task1_update_summary": self.last_task1_update_summary,
            "last_task1_cleanup_summary": self.last_task1_cleanup_summary,
            "semantic_map": self.semantic_map.get_summary(),
            "keyframe_manager": self.keyframe_manager.to_log_dict(),
        }


class ONAirSV(ONAir):
    """
    Semantic-Value Navigation wrapper.

    Current step:
        Add Task1Result -> SemanticMap write-back interface.

    This class still preserves the original UAV-ON baseline behavior in
    prepare_inputs() and run(). The SVNav semantic map is prepared as an
    independent state and does not control actions yet.
    """

    def __init__(self, fixed, batch_size):
        super().__init__(fixed=fixed, batch_size=batch_size)
        self.method_name = "SVNav"
        self._ensure_svnav_storage()

    # ------------------------------------------------------------------
    # Baseline-compatible entry
    # ------------------------------------------------------------------

    def prepare_inputs(self, episodes, fixed):
        return super().prepare_inputs(episodes, fixed)

    def run(self, inputs, fixed, prompt_info_list=None):
        return super().run(inputs, fixed, prompt_info_list)

    # ------------------------------------------------------------------
    # SVNav episode state
    # ------------------------------------------------------------------

    def _ensure_svnav_storage(self) -> None:
        if not hasattr(self, "svnav_states"):
            self.svnav_states: Dict[str, SVNAVEpisodeState] = {}

    def init_svnav_episode(
        self,
        episode_id: str,
        origin_pose: Any,
        target_info: Any,
        search_radius: Optional[float] = None,
        cell_size: Optional[float] = None,
        semantic_map_kwargs: Optional[Dict[str, Any]] = None,
        keyframe_config: Optional[KeyframeManagerConfig] = None,
    ) -> SVNAVEpisodeState:
        """
        Initialize one independent SVNav state for one episode.

        This should be called at the beginning of each UAV-ON episode.
        It creates:
            - one SemanticMap
            - one Task1KeyframeManager

        Old state with the same episode_id will be replaced.
        """
        self._ensure_svnav_storage()

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

        state = SVNAVEpisodeState(
            episode_id=episode_id,
            origin_pose=origin,
            target_info=target,
            semantic_map=semantic_map,
            keyframe_manager=keyframe_manager,
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
    ) -> SVNAVEpisodeState:
        return self.init_svnav_episode(
            episode_id=episode_id,
            origin_pose=origin_pose,
            target_info=target_info,
            search_radius=search_radius,
            cell_size=cell_size,
            semantic_map_kwargs=semantic_map_kwargs,
            keyframe_config=keyframe_config,
        )

    def get_svnav_state(self, episode_id: str) -> Optional[SVNAVEpisodeState]:
        self._ensure_svnav_storage()
        return self.svnav_states.get(str(episode_id))

    # ------------------------------------------------------------------
    # Task1Result -> SemanticMap write-back
    # ------------------------------------------------------------------

    def apply_task1_result_to_map(
        self,
        episode_id: str,
        task1_result: Task1Result,
    ) -> Dict[str, Any]:
        """
        Apply one Task1Result to the episode semantic map.

        Correct order:
            1. Check episode_id and request_id.
            2. Get historical FrameRecord lookup from keyframe_manager.
            3. Update semantic_map using historical pose/view/depth.
            4. Clean frame_lookup after map update.

        This method does not call VLM, does not output actions,
        and does not touch GDINO/Task2/StopGate.
        """
        self._ensure_svnav_storage()

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

        raise ValueError("Cannot convert origin_pose to PoseRecord: {}".format(type(value)))

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
        try:
            from common.param import args

            return float(getattr(args, "xOy_step_size", 5.0))
        except Exception:
            return 5.0

    @staticmethod
    def _default_search_radius(target_info: TargetInfo) -> float:
        if target_info.search_radius is not None:
            return float(target_info.search_radius)
        return 50.0


__all__ = [
    "ONAirSV",
    "SVNAVEpisodeState",
]
