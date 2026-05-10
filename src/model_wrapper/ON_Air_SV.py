from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import airsim
import numpy as np

from common.param import args
from model_wrapper.ON_Air_2 import ONAir
from svnav.async_manager import AsyncManager, AsyncManagerConfig

from svnav.gdino_client import GDINOClientConfig, SVNavGDINOClient
from svnav.keyframe_manager import (
    GDINOKeyframeManager,
    GDINOKeyframeManagerConfig,
    KeyframeManagerConfig,
    Task1KeyframeManager,
)
from svnav.navigator import SearchNavigator, SearchNavigatorConfig
from svnav.semantic_map import SemanticMap, SemanticMapConfig
from svnav.task1_reasoner import Task1Reasoner, Task1ReasonerConfig
from svnav.target_verifier import TargetVerifier, TargetVerifierConfig
from svnav.target_evidence import TargetEvidenceManager, TargetEvidenceManagerConfig
from svnav.types import (
    AsyncTaskType,
    FrameRecord,
    GDINOResult,
    NavMode,
    ObservationRecord,
    PoseRecord,
    TargetInfo,
    Task1Result,
    Task2Result,
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
    gdino_keyframe_manager: GDINOKeyframeManager
    target_verifier: TargetVerifier
    target_evidence_manager: TargetEvidenceManager
    navigator: SearchNavigator
    async_manager: AsyncManager

    created_at: float = field(default_factory=now_ts)
    last_step_id: int = -1
    last_observation_id: Optional[str] = None
    frame_lookup: Dict[str, FrameRecord] = field(default_factory=dict)
    last_task1_request_id: Optional[str] = None
    last_task1_result_id: Optional[str] = None
    last_task1_update_summary: Optional[Dict[str, Any]] = None
    last_task1_cleanup_summary: Optional[Dict[str, Any]] = None
    last_nav_decision: Optional[Dict[str, Any]] = None
    last_gdino_request_id: Optional[str] = None
    last_gdino_result_id: Optional[str] = None
    last_gdino_summary: Optional[Dict[str, Any]] = None
    last_gdino_cleanup_summary: Optional[Dict[str, Any]] = None
    latest_gdino_result: Optional[GDINOResult] = None
    gdino_history: List[GDINOResult] = field(default_factory=list)
    latest_task2_results: List[Task2Result] = field(default_factory=list)
    task2_history: List[Task2Result] = field(default_factory=list)
    last_task2_batch_id: Optional[str] = None
    last_task2_summary: Optional[Dict[str, Any]] = None
    last_task2_cleanup_summary: Optional[Dict[str, Any]] = None
    last_target_evidence_summary: Optional[Dict[str, Any]] = None
    last_target_cue_summary: Optional[Dict[str, Any]] = None
    best_target_evidence: Optional[Dict[str, Any]] = None

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
            "last_gdino_request_id": self.last_gdino_request_id,
            "last_gdino_result_id": self.last_gdino_result_id,
            "last_gdino_summary": self.last_gdino_summary,
            "last_gdino_cleanup_summary": self.last_gdino_cleanup_summary,
            "semantic_map": self.semantic_map.get_summary(),
            "keyframe_manager": self.keyframe_manager.to_log_dict(),
            "gdino_keyframe_manager": self.gdino_keyframe_manager.to_log_dict(),
            "target_verifier": self.target_verifier.to_log_dict(),
            "target_evidence_manager": self.target_evidence_manager.to_log_dict(),
            "async_manager": self.async_manager.to_log_dict(),
            "last_task2_batch_id": self.last_task2_batch_id,
            "last_task2_summary": self.last_task2_summary,
            "last_task2_cleanup_summary": self.last_task2_cleanup_summary,
            "last_target_evidence_summary": self.last_target_evidence_summary,
            "best_target_evidence": self.best_target_evidence,
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
        self.gdino_client = SVNavGDINOClient(GDINOClientConfig.from_env())

    def _cleanup_svnav_episode_async(self, episode_id: str) -> None:
        state = self.svnav_states.get(episode_id)
        if state is None:
            return

        manager = getattr(state, "async_manager", None)
        if manager is None:
            return

        try:
            summary = manager.cleanup_episode(episode_id)
            manager.shutdown()
            print(
                "[SVNavAsync] cleanup episode={} removed_inflight={}".format(
                    episode_id,
                    summary.get("removed_inflight", 0),
                )
            )
        except Exception as exc:
            print(
                "[SVNavAsync] cleanup episode={} error={}".format(
                    episode_id,
                    exc,
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
                previous_episode_id = self.batch_episode_ids[batch_index]
                if previous_episode_id and previous_episode_id != episode_id:
                    self._cleanup_svnav_episode_async(previous_episode_id)

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
                if not hasattr(state, "frame_lookup") or state.frame_lookup is None:
                    state.frame_lookup = {}
                for frame in observation.iter_frames():
                    state.frame_lookup[frame.frame_id] = frame
                if len(state.frame_lookup) > 160:
                    items = sorted(
                        state.frame_lookup.items(),
                        key=lambda item: getattr(item[1], "step_id", 0),
                    )
                    state.frame_lookup = dict(items[-120:])

                self._observe_semantic_map_and_submit_task1_async(
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

                self._poll_svnav_async_results(
                    state=state,
                    episode_id=episode_id,
                    step_id=step_id,
                )

                self._update_target_cues_to_semantic_map(
                    state=state,
                    step_id=step_id,
                )

                decision = self._select_svnav_navigation_decision(
                    state=state,
                    episode_id=episode_id,
                    step_id=step_id,
                    observation=observation,
                )

                decision = self._svnav_apply_pre_action_safety(
                    state=state,
                    episode_id=episode_id,
                    step_id=step_id,
                    observation=observation,
                    decision=decision,
                )

                state.last_nav_decision = decision.to_log_dict()

                self._update_gdino_from_observation(
                    state=state,
                    observation=observation,
                    step_id=step_id,
                    nav_decision=decision,
                )
                self._update_target_cues_to_semantic_map(
                    state=state,
                    step_id=step_id,
                )

                action = decision.action or "rotl"
                step_size = decision.step_size
                if step_size is None:
                    step_size = self._default_step_size_for_action(action)

                actions.append(action)
                steps_size.append(float(step_size))
                predict_dones.append(False)

                if decision.mode in (NavMode.APPROACH, NavMode.FINAL_CHECK):
                    self._print_svnav_approach_debug(decision)
                else:
                    self._print_svnav_search_debug(decision)

            except Exception as exc:
                actions.append("rotl")
                steps_size.append(float(getattr(args, "rotateAngle", 15)))
                predict_dones.append(False)
                print("[SVNavSearch] run error: {}".format(exc))

        return actions, steps_size, predict_dones




    # ------------------------------------------------------------------
    # Async result application
    # ------------------------------------------------------------------

    def _poll_svnav_async_results(
        self,
        state: SVNAVEpisodeState,
        episode_id: str,
        step_id: int,
    ) -> None:
        if state is None or getattr(state, "async_manager", None) is None:
            return

        results = state.async_manager.poll_results(
            current_step=step_id,
            episode_id=episode_id,
        )

        for async_record in results:
            if async_record.episode_id != state.episode_id:
                print(
                    "[SVNavAsync] drop episode={} step={} type={} request={} reason=episode_mismatch".format(
                        episode_id,
                        step_id,
                        async_record.task_type.value,
                        async_record.request_id,
                    )
                )
                continue

            if async_record.is_stale(state.async_manager.config):
                self._drop_stale_async_result(
                    state=state,
                    async_record=async_record,
                    step_id=step_id,
                )
                continue

            if not async_record.success:
                self._apply_failed_async_result(
                    state=state,
                    async_record=async_record,
                    step_id=step_id,
                )
                continue

            if async_record.task_type == AsyncTaskType.TASK1:
                self._apply_async_task1_result(
                    state=state,
                    async_record=async_record,
                    step_id=step_id,
                )
            elif async_record.task_type == AsyncTaskType.GDINO:
                self._apply_async_gdino_result(
                    state=state,
                    async_record=async_record,
                    step_id=step_id,
                )
            elif async_record.task_type == AsyncTaskType.TASK2:
                self._apply_async_task2_result(
                    state=state,
                    async_record=async_record,
                    step_id=step_id,
                )

    def _drop_stale_async_result(
        self,
        state: SVNAVEpisodeState,
        async_record,
        step_id: int,
    ) -> None:
        if async_record.task_type == AsyncTaskType.TASK1:
            try:
                state.keyframe_manager.mark_request_completed(async_record.request_id)
            except Exception:
                pass
        elif async_record.task_type == AsyncTaskType.GDINO:
            try:
                state.gdino_keyframe_manager.mark_request_completed(async_record.request_id)
            except Exception:
                pass
        elif async_record.task_type == AsyncTaskType.TASK2:
            try:
                state.target_verifier.mark_batch_completed(async_record.request_id)
            except Exception:
                pass

        print(
            "[SVNavAsync] drop episode={} step={} type={} request={} reason=stale age_steps={}".format(
                state.episode_id,
                step_id,
                async_record.task_type.value,
                async_record.request_id,
                async_record.age_steps,
            )
        )

    def _apply_failed_async_result(
        self,
        state: SVNAVEpisodeState,
        async_record,
        step_id: int,
    ) -> None:
        if async_record.task_type == AsyncTaskType.TASK1:
            try:
                cleanup = state.keyframe_manager.mark_request_completed(
                    async_record.request_id
                )
            except Exception:
                cleanup = None
            state.last_task1_cleanup_summary = cleanup
            state.last_task1_update_summary = {
                "accepted": False,
                "reason": "async_task1_failed",
                "request_id": async_record.request_id,
                "error": async_record.error,
            }
        elif async_record.task_type == AsyncTaskType.GDINO:
            try:
                cleanup = state.gdino_keyframe_manager.mark_request_completed(
                    async_record.request_id
                )
            except Exception:
                cleanup = None
            state.last_gdino_cleanup_summary = cleanup
            state.last_gdino_summary = {
                "request_id": async_record.request_id,
                "success": False,
                "candidate_count": 0,
                "error": async_record.error,
            }
        elif async_record.task_type == AsyncTaskType.TASK2:
            try:
                cleanup = state.target_verifier.mark_batch_completed(
                    async_record.request_id
                )
            except Exception:
                cleanup = None
            state.last_task2_cleanup_summary = cleanup
            state.last_task2_summary = {
                "batch_id": async_record.request_id,
                "result_count": 0,
                "error": async_record.error,
            }

        shorten = getattr(
            self,
            "_svnav_debug_shorten",
            lambda value, limit=180: str(value)[:limit],
        )
        print(
            "[SVNavAsync] failed episode={} step={} type={} request={} error={}".format(
                state.episode_id,
                step_id,
                async_record.task_type.value,
                async_record.request_id,
                shorten(async_record.error, limit=180),
            )
        )

    def _apply_async_task1_result(
        self,
        state: SVNAVEpisodeState,
        async_record,
        step_id: int,
    ) -> None:
        task1_result = async_record.result
        try:
            task1_result.return_step = int(step_id)
        except Exception:
            pass

        apply_summary = self.apply_task1_result_to_map(
            episode_id=state.episode_id,
            task1_result=task1_result,
        )

        if apply_summary.get("accepted"):
            print(
                "[SVNavTask1] episode={} submit_step={} return_step={} request={} "
                "success={} updated_cells={} latency_ms={} age_steps={}".format(
                    state.episode_id,
                    async_record.submit_step,
                    step_id,
                    async_record.request_id,
                    getattr(task1_result, "success", None),
                    apply_summary.get("update_summary", {}).get("updated_cells", 0),
                    self._svnav_debug_fmt(async_record.latency_ms)
                    if hasattr(self, "_svnav_debug_fmt")
                    else async_record.latency_ms,
                    async_record.age_steps,
                )
            )
        else:
            print(
                "[SVNavTask1] episode={} submit_step={} return_step={} request={} rejected={}".format(
                    state.episode_id,
                    async_record.submit_step,
                    step_id,
                    async_record.request_id,
                    apply_summary.get("reason"),
                )
            )

    def _apply_async_gdino_result(
        self,
        state: SVNAVEpisodeState,
        async_record,
        step_id: int,
    ) -> None:
        gdino_result = async_record.result
        request = async_record.request
        update = (async_record.metadata or {}).get("update")

        try:
            gdino_result.return_step = int(step_id)
        except Exception:
            pass

        self._update_task2_from_gdino_result(
            state=state,
            gdino_result=gdino_result,
            step_id=step_id,
        )

        cleanup_summary = state.gdino_keyframe_manager.mark_request_completed(
            async_record.request_id
        )

        state.latest_gdino_result = gdino_result
        state.gdino_history.append(gdino_result)
        if len(state.gdino_history) > 50:
            state.gdino_history = state.gdino_history[-50:]

        state.last_gdino_result_id = gdino_result.request_id
        state.last_gdino_cleanup_summary = cleanup_summary

        if update is not None and request is not None:
            state.last_gdino_summary = self._build_gdino_summary(
                request=request,
                result=gdino_result,
                update=update,
            )
            self._print_svnav_gdino_summary(
                state=state,
                request=request,
                result=gdino_result,
                update=update,
            )
        else:
            candidates = gdino_result.candidates or []
            state.last_gdino_summary = {
                "request_id": gdino_result.request_id,
                "success": bool(gdino_result.success),
                "candidate_count": len(candidates),
                "error": gdino_result.error,
                "latency_ms": gdino_result.latency_ms,
            }

    def _apply_async_task2_result(
        self,
        state: SVNAVEpisodeState,
        async_record,
        step_id: int,
    ) -> None:
        results = async_record.result or []
        batch = async_record.request
        update = (async_record.metadata or {}).get("update")

        for result in results:
            try:
                result.return_step = int(step_id)
            except Exception:
                pass

        cleanup = state.target_verifier.mark_batch_completed(async_record.request_id)

        state.latest_task2_results = results
        state.task2_history.extend(results)
        if len(state.task2_history) > 80:
            state.task2_history = state.task2_history[-80:]

        state.last_task2_cleanup_summary = cleanup

        if batch is not None and update is not None:
            state.last_task2_summary = self._build_task2_summary(
                batch=batch,
                results=results,
                update=update,
            )
            self._update_target_evidence_from_task2_results(
                state=state,
                results=results,
                step_id=step_id,
            )
            self._print_svnav_task2_summary(
                state=state,
                batch=batch,
                results=results,
                update=update,
            )
        else:
            state.last_task2_summary = {
                "batch_id": async_record.request_id,
                "result_count": len(results),
                "error": None,
            }
            self._update_target_evidence_from_task2_results(
                state=state,
                results=results,
                step_id=step_id,
            )

    # ------------------------------------------------------------------
    # Search / Approach mode selection
    # ------------------------------------------------------------------


    def _update_target_cues_to_semantic_map(
        self,
        state: SVNAVEpisodeState,
        step_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Feed GDINO/Task2 target cues back into the target-aware SemanticMap.

        This makes visual/spatial candidates influence Search through the map,
        while Approach still requires a stable 3D target_position.
        """
        evidence_manager = getattr(state, "target_evidence_manager", None)
        semantic_map = getattr(state, "semantic_map", None)

        if evidence_manager is None or semantic_map is None:
            return None
        if not hasattr(evidence_manager, "build_target_cue_result"):
            return None
        if not hasattr(semantic_map, "update_from_target_cue_result"):
            return None

        frame_lookup = getattr(state, "frame_lookup", None)
        if frame_lookup is None:
            frame_lookup = {}

        cue_result = evidence_manager.build_target_cue_result(current_step=int(step_id))
        summary = semantic_map.update_from_target_cue_result(
            result=cue_result,
            frame_lookup=frame_lookup,
        )

        state.last_target_cue_summary = summary

        if summary and int(summary.get("updated_cells", 0)) > 0:
            print(
                "[SVNavTargetCue] episode={} step={} cues={} visual={} spatial={} updated_cells={}".format(
                    state.episode_id,
                    step_id,
                    summary.get("cue_count", 0),
                    summary.get("visual_cue_count", 0),
                    summary.get("spatial_cue_count", 0),
                    summary.get("updated_cells", 0),
                )
            )

        return summary

    def _svnav_evidence_can_approach(self, evidence, step_id: int) -> bool:
        if evidence is None:
            return False

        metadata = getattr(evidence, "metadata", {}) or {}
        status = getattr(getattr(evidence, "status", None), "value", str(getattr(evidence, "status", "")))
        if status in ("rejected", "lost"):
            return False

        if metadata.get("task2_admission") != "passed":
            return False
        if metadata.get("approach_ready") is False:
            return False

        position = getattr(evidence, "position_3d", None)
        if position is None:
            return False

        try:
            pos_conf = float(getattr(evidence, "position_confidence", 0.0))
            pos_stability = float(getattr(evidence, "position_stability", 0.0))
        except Exception:
            return False

        if pos_conf <= 0.0 or pos_stability <= 0.0:
            return False

        blocked_until = metadata.get("approach_blocked_until_step")
        if blocked_until is not None:
            try:
                if int(step_id) < int(blocked_until):
                    return False
            except Exception:
                pass

        return True


    # ------------------------------------------------------------------
    # SVNav Step15 approach session helpers
    # ------------------------------------------------------------------

    def _svnav_get_active_approach_candidate(
        self,
        state,
        evidence_manager,
        step_id,
    ):
        active_target_id = getattr(state, "active_approach_target_id", None)
        if not active_target_id:
            return None

        if evidence_manager is None or not hasattr(evidence_manager, "get_evidence_by_id"):
            setattr(state, "active_approach_target_id", None)
            return None

        evidence = evidence_manager.get_evidence_by_id(active_target_id)
        if evidence is None:
            setattr(state, "active_approach_target_id", None)
            return None

        metadata = getattr(evidence, "metadata", {}) or {}
        status = getattr(getattr(evidence, "status", None), "value", str(getattr(evidence, "status", "")))

        if status in ("rejected", "lost"):
            setattr(state, "active_approach_target_id", None)
            return None

        if not self._svnav_evidence_can_approach(evidence, step_id):
            setattr(state, "active_approach_target_id", None)
            return None

        blocked_until = metadata.get("approach_blocked_until_step")
        if blocked_until is not None:
            try:
                if int(step_id) < int(blocked_until):
                    setattr(state, "active_approach_target_id", None)
                    return None
            except Exception:
                pass

        return evidence

    def _svnav_select_new_approach_candidate(
        self,
        state,
        evidence_manager,
        step_id,
    ):
        if evidence_manager is None:
            return None

        if not hasattr(evidence_manager, "get_best_approach_candidate"):
            return None

        candidate = evidence_manager.get_best_approach_candidate(current_step=step_id)
        if candidate is None:
            return None
        if not self._svnav_evidence_can_approach(candidate, step_id):
            return None

        setattr(state, "active_approach_target_id", candidate.target_id)

        if hasattr(evidence_manager, "mark_approach_attempt"):
            evidence_manager.mark_approach_attempt(
                target_id=candidate.target_id,
                step_id=step_id,
                reason="selected_as_active_approach_target",
            )

        setattr(
            state,
            "svnav_approach_session",
            {
                "target_id": candidate.target_id,
                "start_step": int(step_id),
                "last_distance_to_anchor": None,
                "last_action": None,
                "no_progress_count": 0,
                "rotate_loop_count": 0,
                "last_feedback_decision": "start",
            },
        )

        return candidate

    def _svnav_update_approach_feedback(
        self,
        state,
        evidence_manager,
        target_evidence,
        decision,
        step_id,
    ):
        if evidence_manager is None or target_evidence is None:
            return None

        if not hasattr(evidence_manager, "apply_approach_feedback"):
            return None

        try:
            from svnav.target_evidence import ApproachFeedback
        except Exception:
            return None

        metadata = getattr(target_evidence, "metadata", {}) or {}
        debug = getattr(decision, "debug_info", {}) or {}

        target_id = getattr(target_evidence, "target_id", None)
        if not target_id:
            return None

        session = getattr(state, "svnav_approach_session", None)
        if not isinstance(session, dict) or session.get("target_id") != target_id:
            session = {
                "target_id": target_id,
                "start_step": int(step_id),
                "last_distance_to_anchor": None,
                "last_action": None,
                "no_progress_count": 0,
                "rotate_loop_count": 0,
                "last_feedback_decision": "reset",
            }

        current_distance = debug.get("distance_to_anchor")
        if current_distance is None:
            current_distance = debug.get("target_distance")
        if current_distance is None:
            current_distance = debug.get("dist_to_approach_viewpoint")
        try:
            current_distance = None if current_distance is None else float(current_distance)
        except Exception:
            current_distance = None

        previous_distance = session.get("last_distance_to_anchor")
        try:
            previous_distance = None if previous_distance is None else float(previous_distance)
        except Exception:
            previous_distance = None

        distance_progress = None
        if previous_distance is not None and current_distance is not None:
            distance_progress = previous_distance - current_distance

        action = getattr(decision, "action", "") or ""
        mode = getattr(getattr(decision, "mode", None), "value", str(getattr(decision, "mode", "")))
        phase = debug.get("adapter_phase") or debug.get("phase") or ""

        last_action = session.get("last_action")
        no_progress_count = int(session.get("no_progress_count", 0))
        rotate_loop_count = int(session.get("rotate_loop_count", 0))

        # Progress is evaluated conservatively. Small numerical changes are not progress.
        if distance_progress is not None and distance_progress > 0.50:
            no_progress_count = 0
        else:
            # Rotation-only steps near the same anchor count as no progress.
            if action in ("rotl", "rotr") or current_distance is not None:
                no_progress_count += 1

        if action in ("rotl", "rotr") and last_action in ("rotl", "rotr"):
            # Alternating rotl/rotr or long rotation near anchor both indicate an
            # approach attempt that is not producing useful new evidence.
            rotate_loop_count += 1
        elif action in ("rotl", "rotr") and "final_check" in str(phase):
            rotate_loop_count += 1
        else:
            rotate_loop_count = 0

        last_positive_step = metadata.get("last_positive_step")
        if last_positive_step is None:
            last_positive_step = getattr(target_evidence, "last_verified_step", None)

        support_age = None
        try:
            if last_positive_step is not None:
                support_age = int(step_id) - int(last_positive_step)
        except Exception:
            support_age = None

        start_step = int(session.get("start_step", int(step_id)))
        steps_since_attempt = int(step_id) - start_step

        anchor_type = debug.get("anchor_type") or metadata.get("anchor_type") or metadata.get("evidence_kind") or ""
        anchor_unreliable = False

        # Do not reject spatial anchors merely because 3D is noisy. Demote only
        # when the approach attempt itself becomes unproductive.
        if anchor_type == "spatial":
            pos_conf = getattr(target_evidence, "position_confidence", 0.0)
            pos_stability = getattr(target_evidence, "position_stability", 0.0)
            try:
                if float(pos_conf) <= 0.05 and float(pos_stability) <= 0.05:
                    anchor_unreliable = True
            except Exception:
                pass

        feedback = ApproachFeedback(
            target_id=target_id,
            episode_id=getattr(target_evidence, "episode_id", getattr(state, "episode_id", "")),
            step_id=int(step_id),
            action=action,
            mode=mode,
            phase=str(phase),
            distance_to_anchor=current_distance,
            previous_distance_to_anchor=previous_distance,
            distance_progress=distance_progress,
            support_age_steps=support_age,
            steps_since_attempt=steps_since_attempt,
            no_progress_count=no_progress_count,
            rotate_loop_count=rotate_loop_count,
            negative_count=int(getattr(target_evidence, "negative_count", 0)),
            positive_count=int(getattr(target_evidence, "positive_count", 0)),
            maybe_count=int(getattr(target_evidence, "maybe_count", 0)),
            anchor_type=str(anchor_type),
            anchor_unreliable=anchor_unreliable,
            reason="approach_step_feedback",
            metadata={
                "approach_score": metadata.get("approach_score"),
                "task2_score": metadata.get("task2_score"),
                "visual_score": metadata.get("visual_score"),
                "latest_view_id": metadata.get("latest_view_id"),
                "latest_bbox_center_norm": metadata.get("latest_bbox_center_norm"),
            },
        )

        result = evidence_manager.apply_approach_feedback(
            target_id=target_id,
            feedback=feedback,
        )

        if current_distance is not None:
            if session.get("start_distance_to_anchor") is None:
                session["start_distance_to_anchor"] = current_distance

            best_distance = session.get("best_distance_to_anchor")
            try:
                if best_distance is None:
                    session["best_distance_to_anchor"] = current_distance
                else:
                    session["best_distance_to_anchor"] = min(float(best_distance), float(current_distance))
            except Exception:
                session["best_distance_to_anchor"] = current_distance

        session["last_distance_to_anchor"] = current_distance
        session["last_action"] = action
        session["no_progress_count"] = no_progress_count
        session["rotate_loop_count"] = rotate_loop_count
        session["last_feedback_decision"] = getattr(result, "decision", None)
        setattr(state, "svnav_approach_session", session)

        decision_text = getattr(result, "decision", "unknown")
        reason_text = getattr(result, "reason", "")
        if decision_text != "keep" or no_progress_count >= 3 or rotate_loop_count >= 3:
            print(
                "[SVNavApproachFeedback] episode={} step={} target={} decision={} "
                "reason={} action={} mode={} phase={} dist={} progress={} "
                "support_age={} no_progress={} rotate_loop={}".format(
                    getattr(state, "episode_id", ""),
                    step_id,
                    target_id,
                    decision_text,
                    reason_text,
                    action,
                    mode,
                    phase,
                    self._svnav_debug_fmt(current_distance) if hasattr(self, "_svnav_debug_fmt") else current_distance,
                    self._svnav_debug_fmt(distance_progress) if hasattr(self, "_svnav_debug_fmt") else distance_progress,
                    support_age,
                    no_progress_count,
                    rotate_loop_count,
                )
            )

        if decision_text in ("demote", "reject", "missing"):
            setattr(state, "active_approach_target_id", None)
            session["target_id"] = None
            setattr(state, "svnav_approach_session", session)

        return result

    # ------------------------------------------------------------------
    # End SVNav Step15 approach session helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # SVNav StopGate helpers
    # ------------------------------------------------------------------

    def _svnav_get_stop_gate(self, state):
        stop_gate = getattr(state, "stop_gate", None)
        if stop_gate is not None:
            return stop_gate

        try:
            from svnav.stop_gate import StopGate, StopGateConfig
        except Exception:
            return None

        config = StopGateConfig()
        stop_gate = StopGate(config)
        setattr(state, "stop_gate", stop_gate)
        return stop_gate

    def _svnav_get_current_active_target_evidence(self, state):
        target_id = getattr(state, "active_approach_target_id", None)
        if not target_id:
            return None

        evidence_manager = getattr(state, "target_evidence_manager", None)
        if evidence_manager is None:
            return None

        if not hasattr(evidence_manager, "get_evidence_by_id"):
            return None

        return evidence_manager.get_evidence_by_id(target_id)

    def _svnav_apply_stop_gate(
        self,
        state,
        episode_id,
        step_id,
        observation,
        decision,
    ):
        stop_gate = self._svnav_get_stop_gate(state)
        if stop_gate is None:
            return decision

        target_evidence = self._svnav_get_current_active_target_evidence(state)
        approach_session = getattr(state, "svnav_approach_session", None)

        result = stop_gate.evaluate(
            episode_id=episode_id,
            step_id=step_id,
            observation=observation,
            nav_decision=decision,
            target_evidence=target_evidence,
            approach_session=approach_session,
        )

        setattr(state, "last_stop_gate_result", result)

        should_print = bool(result.allow_stop)
        if result.reason not in (
            "not_in_approach_mode",
            "missing_target_evidence",
        ):
            should_print = True

        if should_print:
            print(
                "[SVNavStopGate] episode={} step={} allow={} reason={} "
                "target={} mode={} action={} anchor={} dist={} hdist={} "
                "task2_score={} support_age={} progress={} near_count={}".format(
                    episode_id,
                    step_id,
                    result.allow_stop,
                    result.reason,
                    result.target_id,
                    result.mode,
                    result.action,
                    result.anchor_type,
                    self._svnav_debug_fmt(result.distance_to_target) if hasattr(self, "_svnav_debug_fmt") else result.distance_to_target,
                    self._svnav_debug_fmt(result.horizontal_distance_to_target) if hasattr(self, "_svnav_debug_fmt") else result.horizontal_distance_to_target,
                    self._svnav_debug_fmt(result.task2_score) if hasattr(self, "_svnav_debug_fmt") else result.task2_score,
                    result.support_age_steps,
                    self._svnav_debug_fmt(result.approach_progress) if hasattr(self, "_svnav_debug_fmt") else result.approach_progress,
                    result.near_count,
                )
            )

        if not result.allow_stop:
            return decision

        try:
            decision.action = "stop"
            decision.step_size = 0.0
            decision.stop_allowed = True
            decision.reason = "stop_gate_allowed:{}".format(result.reason)
            debug_info = getattr(decision, "debug_info", None)
            if isinstance(debug_info, dict):
                debug_info["stop_gate"] = result.to_log_dict()
                debug_info["adapter_phase"] = "stop_gate"
        except Exception:
            pass

        print(
            "[SVNavStopAction] episode={} step={} target={} reason={} hdist={} "
            "support_age={} progress={}".format(
                episode_id,
                step_id,
                result.target_id,
                result.reason,
                self._svnav_debug_fmt(result.horizontal_distance_to_target) if hasattr(self, "_svnav_debug_fmt") else result.horizontal_distance_to_target,
                result.support_age_steps,
                self._svnav_debug_fmt(result.approach_progress) if hasattr(self, "_svnav_debug_fmt") else result.approach_progress,
            )
        )

        return decision

    # ------------------------------------------------------------------
    # End SVNav StopGate helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # SVNav PreActionSafety helpers
    # ------------------------------------------------------------------

    def _svnav_get_pre_action_safety_gate(self, state):
        gate = getattr(state, "pre_action_safety_gate", None)
        if gate is not None:
            return gate

        try:
            from svnav.pre_action_safety import (
                PreActionSafetyConfig,
                PreActionSafetyGate,
            )
        except Exception:
            return None

        gate = PreActionSafetyGate(PreActionSafetyConfig())
        setattr(state, "pre_action_safety_gate", gate)
        return gate

    def _svnav_apply_pre_action_safety(
        self,
        state,
        episode_id,
        step_id,
        observation,
        decision,
    ):
        gate = self._svnav_get_pre_action_safety_gate(state)
        if gate is None:
            return decision

        semantic_map = getattr(state, "semantic_map", None)

        filtered_decision, result = gate.filter_decision(
            decision=decision,
            observation=observation,
            semantic_map=semantic_map,
        )

        setattr(state, "last_pre_action_safety_result", result.to_log_dict())

        if result.changed or not result.safe:
            print(
                "[SVNavPreActionSafety] episode={} step={} changed={} safe={} "
                "reason={} original={}:{} final={}:{}".format(
                    episode_id,
                    step_id,
                    result.changed,
                    result.safe,
                    result.reason,
                    result.original_action,
                    self._svnav_debug_fmt(result.original_step_size)
                    if hasattr(self, "_svnav_debug_fmt") else result.original_step_size,
                    result.final_action,
                    self._svnav_debug_fmt(result.final_step_size)
                    if hasattr(self, "_svnav_debug_fmt") else result.final_step_size,
                )
            )

            original = result.original_check
            selected = result.selected_check
            if original is not None:
                print(
                    "[SVNavPreActionSafetyDebug] original action={} safe={} "
                    "reason={} depth={} min_depth={} view={} boundary={}".format(
                        original.action,
                        original.safe,
                        original.reason,
                        self._svnav_debug_fmt(original.depth_value)
                        if hasattr(self, "_svnav_debug_fmt") else original.depth_value,
                        self._svnav_debug_fmt(original.min_depth)
                        if hasattr(self, "_svnav_debug_fmt") else original.min_depth,
                        original.view_id,
                        original.boundary_safe,
                    )
                )

            if selected is not None and selected is not original:
                print(
                    "[SVNavPreActionSafetyDebug] selected action={} safe={} "
                    "reason={} depth={} min_depth={} view={} boundary={}".format(
                        selected.action,
                        selected.safe,
                        selected.reason,
                        self._svnav_debug_fmt(selected.depth_value)
                        if hasattr(self, "_svnav_debug_fmt") else selected.depth_value,
                        self._svnav_debug_fmt(selected.min_depth)
                        if hasattr(self, "_svnav_debug_fmt") else selected.min_depth,
                        selected.view_id,
                        selected.boundary_safe,
                    )
                )

        return filtered_decision

    # ------------------------------------------------------------------
    # End SVNav PreActionSafety helpers
    # ------------------------------------------------------------------

    def _select_svnav_navigation_decision(
        self,
        state: SVNAVEpisodeState,
        episode_id: str,
        step_id: int,
        observation: ObservationRecord,
    ):
        """
        Step 15 selection policy.

        The active approach target is sticky:
            - A new candidate cannot directly steal control.
            - The active target is kept until approach feedback demotes/rejects it.
            - After demotion, this step returns Search. A new candidate can be
              selected in a later step through the normal evidence pool.
        """
        evidence_manager = getattr(state, "target_evidence_manager", None)

        approach_candidate = self._svnav_get_active_approach_candidate(
            state=state,
            evidence_manager=evidence_manager,
            step_id=step_id,
        )

        if approach_candidate is None:
            approach_candidate = self._svnav_select_new_approach_candidate(
                state=state,
                evidence_manager=evidence_manager,
                step_id=step_id,
            )

        if approach_candidate is not None:
            decision = state.navigator.build_approach_decision(
                episode_id=episode_id,
                step_id=step_id,
                observation=observation,
                target_evidence=approach_candidate,
            )

            feedback_result = self._svnav_update_approach_feedback(
                state=state,
                evidence_manager=evidence_manager,
                target_evidence=approach_candidate,
                decision=decision,
                step_id=step_id,
            )

            # If feedback says this approach attempt failed, return to Search
            # immediately instead of executing one more stale approach action.
            if feedback_result is not None and getattr(feedback_result, "decision", None) in ("demote", "reject", "missing"):
                return state.navigator.decide_search(
                    episode_id=episode_id,
                    step_id=step_id,
                    observation=observation,
                    semantic_map=state.semantic_map,
                )

            decision = self._svnav_apply_stop_gate(
                state=state,
                episode_id=episode_id,
                step_id=step_id,
                observation=observation,
                decision=decision,
            )

            return decision

        return state.navigator.decide_search(
            episode_id=episode_id,
            step_id=step_id,
            observation=observation,
            semantic_map=state.semantic_map,
        )

    def _print_svnav_approach_debug(self, decision) -> None:
        debug = decision.debug_info or {}
        action_source = getattr(decision.action_source, "value", decision.action_source)

        print(
            "[SVNavApproach] episode={} step={} mode={} target={} action={} "
            "step_size={} source={} phase={} reason={}".format(
                decision.episode_id,
                decision.step_id,
                decision.mode.value,
                decision.target_id,
                decision.action,
                decision.step_size,
                action_source,
                debug.get("adapter_phase"),
                decision.reason,
            )
        )

        print(
            "[SVNavApproachDebug] kind={} target_pos={} approach_vp={} "
            "target_dist={} dist_to_vp={} yaw_to_vp={} observe_yaw_err={} "
            "final_check_dist={}".format(
                debug.get("approach_kind"),
                self._svnav_debug_fmt(debug.get("target_position")),
                self._svnav_debug_fmt(debug.get("approach_viewpoint")),
                self._svnav_debug_fmt(debug.get("target_distance")),
                self._svnav_debug_fmt(debug.get("dist_to_approach_viewpoint")),
                self._svnav_debug_fmt(debug.get("yaw_to_viewpoint_deg")),
                self._svnav_debug_fmt(debug.get("observe_yaw_error_deg")),
                self._svnav_debug_fmt(debug.get("final_check_distance")),
            )
        )

        print(
            "[SVNavApproachDebug] approach_score={} visual_score={} evidence_kind={} "
            "pos_conf={} pos_stability={} latest_view={} bbox_center={}".format(
                self._svnav_debug_fmt(debug.get("approach_score")),
                self._svnav_debug_fmt(debug.get("visual_score")),
                debug.get("evidence_kind"),
                self._svnav_debug_fmt(debug.get("position_confidence")),
                self._svnav_debug_fmt(debug.get("position_stability")),
                debug.get("latest_view_id"),
                self._svnav_debug_fmt(debug.get("latest_bbox_center_norm")),
            )
        )


    def _print_svnav_search_debug(self, decision) -> None:
        debug = decision.debug_info or {}

        active = debug.get("active_viewpoint") or {}
        viewpoint = active.get("viewpoint") or {}
        region = viewpoint.get("region") or {}

        action_source = getattr(decision.action_source, "value", decision.action_source)

        adapter_phase = debug.get("adapter_phase")
        viewpoint_id = debug.get("viewpoint_id") or viewpoint.get("viewpoint_id")
        region_id = debug.get("region_id") or viewpoint.get("region_id")
        region_source = viewpoint.get("region_source") or region.get("source")

        distance_to_waypoint = debug.get("distance_to_waypoint")
        yaw_to_waypoint_deg = debug.get("yaw_to_waypoint_deg")
        observe_yaw_error_deg = debug.get("observe_yaw_error_deg")
        viewpoint_yaw_error = debug.get("viewpoint_yaw_error")

        desired_yaw = debug.get("desired_yaw")
        waypoint = debug.get("waypoint")

        active_start_step = active.get("start_step")
        no_progress_count = active.get("no_progress_count")
        last_replan_reason = active.get("last_replan_reason")

        region_score = region.get("score")
        region_semantic = region.get("semantic_score")
        region_explore = region.get("exploration_score")
        viewpoint_score = viewpoint.get("score")

        print(
            "[SVNavSearch] episode={} step={} action={} step_size={} source={} "
            "phase={} reason={}".format(
                decision.episode_id,
                decision.step_id,
                decision.action,
                decision.step_size,
                action_source,
                adapter_phase,
                decision.reason,
            )
        )

        print(
            "[SVNavSearchDebug] vp={} region={} region_source={} "
            "active_start={} no_progress={} replan_reason={}".format(
                viewpoint_id,
                region_id,
                region_source,
                active_start_step,
                no_progress_count,
                last_replan_reason,
            )
        )

        print(
            "[SVNavSearchDebug] waypoint={} desired_yaw={} "
            "dist_to_wp={} yaw_to_wp={} observe_yaw_err={} view_yaw_err={}".format(
                self._svnav_debug_fmt(waypoint),
                self._svnav_debug_fmt(desired_yaw),
                self._svnav_debug_fmt(distance_to_waypoint),
                self._svnav_debug_fmt(yaw_to_waypoint_deg),
                self._svnav_debug_fmt(observe_yaw_error_deg),
                self._svnav_debug_fmt(viewpoint_yaw_error),
            )
        )

        print(
            "[SVNavSearchDebug] region_score={} semantic={} explore={} "
            "viewpoint_score={}".format(
                self._svnav_debug_fmt(region_score),
                self._svnav_debug_fmt(region_semantic),
                self._svnav_debug_fmt(region_explore),
                self._svnav_debug_fmt(viewpoint_score),
            )
        )



        map_debug = debug.get("map_debug") or {}
        if map_debug:
            top_cells = map_debug.get("top_semantic_cells") or []
            top_cell = top_cells[0] if top_cells else {}

            print(
                "[SVNavMapDebug] best={} second={} margin={} high_value={} "
                "top_cell=({}, {}, eff={}, status={})".format(
                    self._svnav_debug_fmt(map_debug.get("best_semantic")),
                    self._svnav_debug_fmt(map_debug.get("second_semantic")),
                    self._svnav_debug_fmt(map_debug.get("semantic_margin")),
                    map_debug.get("high_value_count"),
                    top_cell.get("gx"),
                    top_cell.get("gy"),
                    self._svnav_debug_fmt(top_cell.get("effective_value")),
                    top_cell.get("status"),
                )
            )

        region_debug = debug.get("region_debug") or {}
        if region_debug:
            top_regions = region_debug.get("top_regions") or []
            top_semantic_regions = region_debug.get("top_semantic_regions") or []

            best_region = top_regions[0] if top_regions else {}
            best_semantic_region = top_semantic_regions[0] if top_semantic_regions else {}

            best_region_center = best_region.get("center")
            best_semantic_center = best_semantic_region.get("center")

            print(
                "[SVNavRegionDebug] count={} best_source={} best_score={} "
                "best_sem={} best_exp={} best_center={} semantic_region_count={} "
                "best_semantic_score={} best_semantic_center={}".format(
                    region_debug.get("region_count"),
                    best_region.get("source"),
                    self._svnav_debug_fmt(best_region.get("score")),
                    self._svnav_debug_fmt(best_region.get("semantic_score")),
                    self._svnav_debug_fmt(best_region.get("exploration_score")),
                    self._svnav_debug_fmt(best_region_center),
                    len(top_semantic_regions),
                    self._svnav_debug_fmt(best_semantic_region.get("score")),
                    self._svnav_debug_fmt(best_semantic_center),
                )
            )


    @staticmethod
    def _svnav_debug_shorten(value, limit: int = 800) -> str:
        if value is None:
            return "None"

        text = str(value)
        limit = int(limit)

        if len(text) <= limit:
            return text

        return text[:limit] + "...<truncated>"


    @staticmethod
    def _svnav_debug_fmt(value) -> str:
        if value is None:
            return "None"

        if isinstance(value, float):
            return "{:.3f}".format(value)

        if isinstance(value, (list, tuple)):
            parts = []
            for item in value:
                if isinstance(item, float):
                    parts.append("{:.2f}".format(item))
                else:
                    parts.append(str(item))
            return "[" + ", ".join(parts) + "]"

        return str(value)



    # ------------------------------------------------------------------
    # GDINO Search keyframe update
    # ------------------------------------------------------------------



    def _update_gdino_from_observation(
        self,
        state: SVNAVEpisodeState,
        observation: ObservationRecord,
        step_id: int,
        nav_decision: Any,
    ) -> None:
        """
        Search / Approach GDINO keyframe selection and async candidate generation.

        This method only submits a GDINO request. It does not wait for GDINO,
        does not update SemanticMap, does not verify Task2, does not switch
        navigation mode, and does not decide stop.
        """
        try:
            update = state.gdino_keyframe_manager.observe(
                observation=observation,
                semantic_map=state.semantic_map,
                nav_mode=getattr(nav_decision, "mode", NavMode.SEARCH),
                nav_decision=nav_decision,
                build_request=True,
                gdino_enabled=self.gdino_client.enabled,
            )
        except Exception as exc:
            print(
                "[SVNavGDINO] episode={} step={} keyframe_update_error={}".format(
                    state.episode_id,
                    step_id,
                    exc,
                )
            )
            return

        request = update.gdino_request
        if request is None:
            return

        state.last_gdino_request_id = request.request_id

        submitted = state.async_manager.submit_gdino(
            request=request,
            gdino_client=self.gdino_client,
            metadata={
                "update": update,
                "source": "svnav_gdino_keyframe",
            },
        )

        if submitted is None:
            cleanup_summary = state.gdino_keyframe_manager.mark_request_completed(
                request.request_id
            )
            state.last_gdino_cleanup_summary = cleanup_summary
            state.last_gdino_summary = {
                "request_id": request.request_id,
                "success": False,
                "candidate_count": 0,
                "error": "async_gdino_submit_rejected_or_disabled",
                "cleanup_summary": cleanup_summary,
            }
            print(
                "[SVNavGDINO] episode={} step={} request={} async_submit=rejected".format(
                    state.episode_id,
                    step_id,
                    request.request_id,
                )
            )
            return

        print(
            "[SVNavAsync] submit type=gdino episode={} step={} request={}".format(
                state.episode_id,
                step_id,
                request.request_id,
            )
        )

    def _build_gdino_summary(self, request, result, update) -> Dict[str, Any]:
        candidates = result.candidates or []
        best_score = 0.0
        best_label = ""
        if candidates:
            best = max(candidates, key=lambda item: float(item.score))
            best_score = float(best.score)
            best_label = best.label

        filter_meta = (result.metadata or {}).get("candidate_filter", {})
        geometry_meta = (result.metadata or {}).get("candidate_geometry", {})
        return {
            "request_id": request.request_id,
            "success": bool(result.success),
            "candidate_count": len(candidates),
            "raw_candidate_count": (result.metadata or {}).get("raw_candidate_count"),
            "kept_candidate_count": (result.metadata or {}).get("kept_candidate_count"),
            "rejected_candidate_count": (result.metadata or {}).get("rejected_candidate_count"),
            "candidate_filter": filter_meta,
            "candidate_geometry": geometry_meta,
            "geometry_valid_count": (result.metadata or {}).get("geometry_valid_count"),
            "geometry_invalid_count": (result.metadata or {}).get("geometry_invalid_count"),
            "best_score": best_score,
            "best_label": best_label,
            "latency_ms": result.latency_ms,
            "frame_ids": request.frame_ids,
            "view_ids": request.view_ids,
            "accepted_count": len(update.accepted),
            "rejected_count": len(update.rejected),
            "pending_count": update.gdino_request.metadata.get("pending_count")
            if update.gdino_request is not None
            else None,
            "error": result.error,
        }

    def _print_svnav_gdino_summary(self, state, request, result, update) -> None:
        candidates = result.candidates or []
        best_score = 0.0
        best_label = ""
        if candidates:
            best = max(candidates, key=lambda item: float(item.score))
            best_score = float(best.score)
            best_label = best.label

        metadata = request.metadata or {}
        keyframe_ids = metadata.get("gdino_keyframe_ids", [])
        reasons = metadata.get("keyframe_reasons", {})
        scores = metadata.get("admission_scores", {})

        keyframe_id = keyframe_ids[0] if keyframe_ids else ""
        reason = reasons.get(keyframe_id, "")
        admission_score = scores.get(keyframe_id, None)

        print(
            "[SVNavGDINO] episode={} step={} request={} views={} "
            "keyframe={} reason={} admission={} raw={} kept={} rejected={} "
            "geo_valid={} geo_invalid={} best={:.3f} label={} success={} "
            "latency_ms={} error={}".format(
                state.episode_id,
                request.submit_step,
                request.request_id,
                ",".join(request.view_ids),
                keyframe_id,
                reason,
                self._svnav_debug_fmt(admission_score),
                (result.metadata or {}).get("raw_candidate_count", len(candidates)),
                len(candidates),
                (result.metadata or {}).get("rejected_candidate_count", 0),
                (result.metadata or {}).get("geometry_valid_count", 0),
                (result.metadata or {}).get("geometry_invalid_count", 0),
                best_score,
                self._svnav_debug_shorten(best_label, limit=80),
                result.success,
                self._svnav_debug_fmt(result.latency_ms),
                result.error,
            )
        )



    # ------------------------------------------------------------------
    # Target evidence update
    # ------------------------------------------------------------------

    def _update_target_evidence_from_task2_results(
        self,
        state: SVNAVEpisodeState,
        results: List[Task2Result],
        step_id: int,
    ) -> None:
        """
        Fuse Task2Result into multi-frame TargetEvidence.

        This method only updates evidence state. It does not change action,
        semantic_map, navigation mode, Approach state, or Stop decision.
        """
        if not results:
            return

        try:
            update = state.target_evidence_manager.update_from_task2_results(
                results=results,
                current_step=step_id,
            )
        except Exception as exc:
            print(
                "[SVNavEvidence] episode={} step={} update_error={}".format(
                    state.episode_id,
                    step_id,
                    exc,
                )
            )
            return

        state.last_target_evidence_summary = update.to_log_dict()
        state.best_target_evidence = (
            None if update.best_evidence is None else update.best_evidence.to_log_dict()
        )

        self._print_svnav_evidence_summary(
            state=state,
            update=update,
        )

    def _print_svnav_evidence_summary(self, state, update) -> None:
        summary = update.summary or {}
        status_counts = summary.get("status_counts", {}) or {}
        best = update.best_evidence

        if best is None:
            best_text = "none"
        else:
            best_text = "{}:{}:{:.3f}".format(
                best.target_id,
                best.status.value,
                float(best.metadata.get("evidence_score", 0.0)),
            )

        print(
            "[SVNavEvidence] episode={} step={} updated={} tentative={} "
            "verified={} rejected={} approach_ready={} visual={} spatial={} "
            "best={} approach_best={} created={} ignored={}".format(
                state.episode_id,
                update.step_id,
                len(update.updated_track_ids),
                status_counts.get("tentative", 0),
                status_counts.get("verified", 0),
                status_counts.get("rejected", 0),
                summary.get("approach_ready_count", 0),
                summary.get("visual_count", 0),
                summary.get("spatial_count", 0),
                best_text,
                summary.get("best_approach_candidate_id"),
                len(update.created_track_ids),
                len(update.ignored_results),
            )
        )

    # ------------------------------------------------------------------
    # Task2 target candidate verification
    # ------------------------------------------------------------------



    def _update_task2_from_gdino_result(
        self,
        state: SVNAVEpisodeState,
        gdino_result: GDINOResult,
        step_id: int,
    ) -> None:
        """
        Add filtered/geometrically-completed GDINO candidates into Task2 pending
        and submit a Task2 verification batch asynchronously.

        Task2 does not change action, SemanticMap, navigation mode, Approach,
        or Stop. TargetEvidence consumes completed Task2Result only when
        poll_results() returns it in the main thread.
        """
        try:
            update = state.target_verifier.observe_gdino_result(
                gdino_result=gdino_result,
                current_step=step_id,
                target_info=state.target_info,
                build_batch=True,
            )
        except Exception as exc:
            print(
                "[SVNavTask2] episode={} step={} update_error={}".format(
                    state.episode_id,
                    step_id,
                    exc,
                )
            )
            return

        batch = update.task2_batch
        if batch is None:
            return

        state.last_task2_batch_id = batch.batch_id

        submitted = state.async_manager.submit_task2(
            batch=batch,
            target_verifier=state.target_verifier,
            metadata={
                "update": update,
                "source": "svnav_task2_candidate_verification",
            },
        )

        if submitted is None:
            cleanup = state.target_verifier.mark_batch_completed(batch.batch_id)
            state.last_task2_cleanup_summary = cleanup
            state.last_task2_summary = {
                "batch_id": batch.batch_id,
                "candidate_ids": batch.candidate_ids,
                "result_count": 0,
                "decision_counts": {
                    "match": 0,
                    "maybe": 0,
                    "no": 0,
                    "unknown": 0,
                },
                "accepted_count": len(update.accepted),
                "rejected_count": len(update.rejected),
                "pending_count": len(update.accepted),
                "error": "async_task2_submit_rejected_or_disabled",
            }
            print(
                "[SVNavTask2] episode={} step={} batch={} async_submit=rejected".format(
                    state.episode_id,
                    step_id,
                    batch.batch_id,
                )
            )
            return

        print(
            "[SVNavAsync] submit type=task2 episode={} step={} batch={} candidates={}".format(
                state.episode_id,
                step_id,
                batch.batch_id,
                ",".join(batch.candidate_ids),
            )
        )

    def _build_task2_summary(self, batch, results, update) -> Dict[str, Any]:
        counts = {
            "match": 0,
            "maybe": 0,
            "no": 0,
            "unknown": 0,
        }

        for result in results:
            decision = getattr(result.decision, "value", str(result.decision))
            if decision == "yes":
                counts["match"] += 1
            elif decision == "maybe":
                counts["maybe"] += 1
            elif decision == "no":
                counts["no"] += 1
            else:
                counts["unknown"] += 1

        return {
            "batch_id": batch.batch_id,
            "candidate_ids": batch.candidate_ids,
            "result_count": len(results),
            "decision_counts": counts,
            "accepted_count": len(update.accepted),
            "rejected_count": len(update.rejected),
            "pending_count": len(update.accepted),
        }

    def _print_svnav_task2_summary(self, state, batch, results, update) -> None:
        parts = []
        for result in results:
            decision = getattr(result.decision, "value", str(result.decision))
            if decision == "yes":
                decision = "match"
            parts.append(
                "{}:{}:{:.2f}".format(
                    result.candidate_id,
                    decision,
                    float(result.confidence),
                )
            )

        print(
            "[SVNavTask2] episode={} step={} batch={} candidates={} results={} "
            "accepted={} rejected={} pending={} inflight={}".format(
                state.episode_id,
                batch.submit_step,
                batch.batch_id,
                ",".join(batch.candidate_ids),
                ";".join(parts),
                len(update.accepted),
                len(update.rejected),
                state.target_verifier.pending_count,
                state.target_verifier.inflight_count,
            )
        )

    # ------------------------------------------------------------------
    # SVNav step update
    # ------------------------------------------------------------------


    def _observe_semantic_map_and_submit_task1_async(
        self,
        state: SVNAVEpisodeState,
        observation: ObservationRecord,
        step_id: int,
    ) -> None:
        """
        Mark visited area synchronously, then submit Task1 asynchronously.

        Task1Result is applied later in _poll_svnav_async_results().
        The result must be projected using request-time frames and poses,
        not the current pose when the result returns.
        """
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

        submitted = state.async_manager.submit_task1(
            request=request,
            reasoner=self.task1_reasoner,
            metadata={
                "source": "svnav_task1_keyframe",
            },
        )

        if submitted is None:
            cleanup_summary = state.keyframe_manager.mark_request_completed(
                request.request_id
            )
            state.last_task1_cleanup_summary = cleanup_summary
            state.last_task1_update_summary = {
                "accepted": False,
                "reason": "async_task1_submit_rejected_or_disabled",
                "request_id": request.request_id,
            }
            print(
                "[SVNavTask1] episode={} step={} request={} async_submit=rejected".format(
                    state.episode_id,
                    step_id,
                    request.request_id,
                )
            )
            return

        print(
            "[SVNavAsync] submit type=task1 episode={} step={} request={}".format(
                state.episode_id,
                step_id,
                request.request_id,
            )
        )

    def _update_semantic_map_from_observation(
        self,
        state: SVNAVEpisodeState,
        observation: ObservationRecord,
        step_id: int,
    ) -> None:
        # Backward compatible alias. The main SVNav loop now uses async Task1.
        self._observe_semantic_map_and_submit_task1_async(
            state=state,
            observation=observation,
            step_id=step_id,
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

        gdino_keyframe_manager = GDINOKeyframeManager(
            config=GDINOKeyframeManagerConfig()
        )
        gdino_keyframe_manager.reset_episode(episode_id)

        target_verifier = TargetVerifier(
            config=TargetVerifierConfig.from_env()
        )
        target_verifier.reset_episode(episode_id)

        target_evidence_manager = TargetEvidenceManager(
            config=TargetEvidenceManagerConfig()
        )
        target_evidence_manager.reset_episode(episode_id)

        navigator = SearchNavigator(
            config=navigator_config or SearchNavigatorConfig()
        )
        async_manager = AsyncManager(
            AsyncManagerConfig(
                verbose=True,
            )
        )

        state = SVNAVEpisodeState(
            episode_id=episode_id,
            origin_pose=origin,
            target_info=target,
            semantic_map=semantic_map,
            keyframe_manager=keyframe_manager,
            gdino_keyframe_manager=gdino_keyframe_manager,
            target_verifier=target_verifier,
            target_evidence_manager=target_evidence_manager,
            navigator=navigator,
            async_manager=async_manager,
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
        gdino_summary = state.gdino_keyframe_manager.to_log_dict()
        lines.append("pending_gdino_keyframes: {}".format(gdino_summary.get("pending_count")))
        lines.append("inflight_gdino: {}".format(gdino_summary.get("inflight_count")))
        lines.append("last_gdino_summary: {}".format(state.last_gdino_summary))
        task2_summary = state.target_verifier.to_log_dict()
        lines.append("pending_task2_candidates: {}".format(task2_summary.get("pending_count")))
        lines.append("inflight_task2: {}".format(task2_summary.get("inflight_count")))
        lines.append("last_task2_summary: {}".format(state.last_task2_summary))
        evidence_summary = state.target_evidence_manager.to_log_dict().get("summary", {})
        lines.append("target_evidence: {}".format(evidence_summary))
        lines.append("best_target_evidence: {}".format(state.best_target_evidence))
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
                search_radius=float(value.get("search_radius", 100.0)),
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
