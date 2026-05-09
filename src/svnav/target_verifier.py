from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from PIL import Image

from .geometry import bbox_crop_slices
from .prompts import build_task2_batch_prompt
from .types import (
    BBox,
    GDINOCandidate,
    GDINOResult,
    TargetInfo,
    Task2Decision,
    Task2Request,
    Task2Result,
    bytes_to_b64,
    clamp01,
    new_id,
    now_ts,
    strip_data_uri,
    to_data_uri,
)


@dataclass
class TargetVerifierConfig:
    enabled: bool = True
    provider: str = "dashscope"
    model_name: str = "qwen-vl-max"
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    batch_size: int = 2
    min_batch_size: int = 1
    max_pending: int = 24
    max_inflight: int = 1
    max_wait_steps: int = 4
    high_priority_threshold: float = 0.70
    replace_score_margin: float = 0.05

    crop_padding: float = 0.20
    max_history: int = 80
    request_timeout: float = 60.0

    @staticmethod
    def from_env() -> "TargetVerifierConfig":
        use_env = os.getenv("USE_SVNAV_TASK2", "").strip().lower()
        explicit_enabled = use_env not in ("0", "false", "no", "off")

        provider = os.getenv("SVNAV_TASK2_PROVIDER", "dashscope").strip().lower()

        if provider == "openai":
            api_key = (
                os.getenv("SVNAV_TASK2_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("SVNAV_TASK2_BASE_URL")
        else:
            provider = "dashscope"
            api_key = (
                os.getenv("SVNAV_TASK2_API_KEY")
                or os.getenv("DASHSCOPE_API_KEY")
            )
            base_url = os.getenv("SVNAV_TASK2_BASE_URL")

        enabled = explicit_enabled and bool(api_key)

        return TargetVerifierConfig(
            enabled=enabled,
            provider=provider,
            model_name=os.getenv("SVNAV_TASK2_MODEL", "qwen-vl-max"),
            api_key=api_key,
            base_url=base_url,
            batch_size=int(os.getenv("SVNAV_TASK2_BATCH_SIZE", "2")),
            min_batch_size=int(os.getenv("SVNAV_TASK2_MIN_BATCH_SIZE", "1")),
            max_pending=int(os.getenv("SVNAV_TASK2_MAX_PENDING", "24")),
            max_inflight=int(os.getenv("SVNAV_TASK2_MAX_INFLIGHT", "1")),
            max_wait_steps=int(os.getenv("SVNAV_TASK2_MAX_WAIT_STEPS", "4")),
            high_priority_threshold=float(os.getenv("SVNAV_TASK2_HIGH_PRIORITY", "0.70")),
            crop_padding=float(os.getenv("SVNAV_TASK2_CROP_PADDING", "0.20")),
            request_timeout=float(os.getenv("SVNAV_TASK2_TIMEOUT", "60.0")),
        )

    def __post_init__(self) -> None:
        self.batch_size = max(1, int(self.batch_size))
        self.min_batch_size = max(1, int(self.min_batch_size))
        self.max_pending = max(self.batch_size, int(self.max_pending))
        self.max_inflight = max(1, int(self.max_inflight))
        self.max_wait_steps = max(0, int(self.max_wait_steps))
        self.high_priority_threshold = float(self.high_priority_threshold)
        self.replace_score_margin = float(self.replace_score_margin)
        self.crop_padding = float(self.crop_padding)
        self.max_history = max(1, int(self.max_history))
        self.request_timeout = float(self.request_timeout)


@dataclass
class Task2PendingItem:
    request: Task2Request
    priority: float
    context_b64: str
    context_mime: str
    crop_b64: str
    crop_mime: str
    candidate_snapshot: Dict[str, Any]
    created_step: int
    created_at: float = field(default_factory=now_ts)
    status: str = "pending"

    @property
    def candidate_id(self) -> str:
        return self.request.candidate_id

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def episode_id(self) -> str:
        return self.request.episode_id

    def to_prompt_dict(self) -> Dict[str, Any]:
        candidate = self.request.candidate
        bbox = candidate.bbox.to_log_dict()
        geometry = (candidate.metadata or {}).get("geometry", {})

        return {
            "candidate_id": candidate.candidate_id,
            "view_id": candidate.view_id.value,
            "step_id": candidate.step_id,
            "label": candidate.label,
            "score": round(float(candidate.score), 4),
            "bbox": bbox,
            "area_ratio": bbox.get("area_ratio"),
            "quality_score": round(float(candidate.quality_score), 4),
            "position_3d": candidate.position_3d,
            "position_confidence": round(float(candidate.position_confidence), 4),
            "depth_valid": bool(candidate.depth_valid),
            "geometry_summary": {
                "estimate_method": geometry.get("estimate_method"),
                "estimated_distance": geometry.get("estimated_distance"),
                "position_confidence": geometry.get("position_confidence"),
                "geometry_warning": geometry.get("geometry_warning"),
            },
        }

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "episode_id": self.episode_id,
            "priority": float(self.priority),
            "created_step": int(self.created_step),
            "status": self.status,
            "candidate_snapshot": self.candidate_snapshot,
            "has_context_b64": bool(self.context_b64),
            "has_crop_b64": bool(self.crop_b64),
        }

    def clear_image_payload(self) -> None:
        self.context_b64 = ""
        self.crop_b64 = ""
        self.request.candidate.crop_b64 = None
        self.request.candidate.crop_bytes = None
        self.request.candidate.frame = None


@dataclass
class Task2Batch:
    batch_id: str
    episode_id: str
    submit_step: int
    items: List[Task2PendingItem]
    prompt: str
    created_at: float = field(default_factory=now_ts)

    @property
    def candidate_ids(self) -> List[str]:
        return [item.candidate_id for item in self.items]

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "episode_id": self.episode_id,
            "submit_step": int(self.submit_step),
            "candidate_ids": self.candidate_ids,
            "items": [item.to_log_dict() for item in self.items],
            "created_at": float(self.created_at),
        }


@dataclass
class Task2UpdateResult:
    episode_id: str
    step_id: int
    accepted: List[str] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)
    replaced: Dict[str, str] = field(default_factory=dict)
    task2_batch: Optional[Task2Batch] = None
    reason: str = ""

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_id": int(self.step_id),
            "accepted": list(self.accepted),
            "rejected": dict(self.rejected),
            "replaced": dict(self.replaced),
            "task2_batch_id": None if self.task2_batch is None else self.task2_batch.batch_id,
            "task2_candidate_ids": [] if self.task2_batch is None else self.task2_batch.candidate_ids,
            "reason": self.reason,
        }


class TargetVerifier:
    """
    Task2 candidate verifier.

    It manages a pending lifecycle for candidate verification:
        GDINOResult candidates -> Task2 pending -> batch VLM verification
        -> Task2Result list -> cleanup image payload.

    It does not output actions, does not update SemanticMap, does not switch
    Approach, and does not decide Stop. TargetEvidence will consume Task2Result
    in the next step.
    """

    def __init__(self, config: Optional[TargetVerifierConfig] = None) -> None:
        self.config = config or TargetVerifierConfig.from_env()
        self.current_episode_id: Optional[str] = None

        self.pending_items: List[Task2PendingItem] = []
        self.submitted_batches: Dict[str, Task2Batch] = {}
        self.completed_candidate_ids: Set[str] = set()
        self.seen_candidate_ids: Set[str] = set()

        self.history: List[Task2Result] = []
        self.last_submit_step: int = -1
        self.last_update_step: int = -1

    def reset_episode(self, episode_id: str) -> None:
        self.current_episode_id = str(episode_id)
        self.pending_items.clear()
        self.submitted_batches.clear()
        self.completed_candidate_ids.clear()
        self.seen_candidate_ids.clear()
        self.history.clear()
        self.last_submit_step = -1
        self.last_update_step = -1

    def observe_gdino_result(
        self,
        gdino_result: GDINOResult,
        current_step: int,
        target_info: TargetInfo,
        build_batch: bool = True,
    ) -> Task2UpdateResult:
        if self.current_episode_id is None:
            self.reset_episode(gdino_result.episode_id)

        if gdino_result.episode_id != self.current_episode_id:
            self.reset_episode(gdino_result.episode_id)

        result = Task2UpdateResult(
            episode_id=gdino_result.episode_id,
            step_id=int(current_step),
        )
        self.last_update_step = int(current_step)

        if not self.config.enabled:
            result.reason = "task2_disabled_or_missing_api_key"
            return result

        for candidate in gdino_result.candidates or []:
            item = self._prepare_pending_item(
                candidate=candidate,
                target_info=target_info,
                current_step=current_step,
            )

            if item is None:
                result.rejected[candidate.candidate_id] = "not_usable_for_task2"
                continue

            accepted = self._add_pending_item(item)
            if accepted is None:
                result.rejected[candidate.candidate_id] = "pending_full_low_priority"
                continue

            result.accepted.append(accepted.candidate_id)

        if build_batch:
            batch = self.maybe_build_batch(
                current_step=current_step,
                target_info=target_info,
            )
            result.task2_batch = batch
            if batch is not None:
                result.reason = "built_task2_batch"

        return result

    def maybe_build_batch(
        self,
        current_step: int,
        target_info: TargetInfo,
        force: bool = False,
    ) -> Optional[Task2Batch]:
        if not self.config.enabled:
            return None

        if not self.pending_items:
            return None

        if len(self.submitted_batches) >= self.config.max_inflight:
            return None

        current_step = int(current_step)
        oldest_wait = max(
            current_step - int(item.created_step)
            for item in self.pending_items
        )
        high_priority = max(item.priority for item in self.pending_items)

        should_build = (
            force
            or len(self.pending_items) >= self.config.batch_size
            or (
                len(self.pending_items) >= self.config.min_batch_size
                and oldest_wait >= self.config.max_wait_steps
            )
            or high_priority >= self.config.high_priority_threshold
        )

        if not should_build:
            return None

        ranked = sorted(
            self.pending_items,
            key=lambda item: item.priority,
            reverse=True,
        )
        selected = ranked[: self.config.batch_size]
        selected_ids = {item.candidate_id for item in selected}

        self.pending_items = [
            item for item in self.pending_items if item.candidate_id not in selected_ids
        ]

        prompt = build_task2_batch_prompt(
            target_info=target_info,
            candidates=[item.to_prompt_dict() for item in selected],
        )

        batch = Task2Batch(
            batch_id=new_id("task2batch"),
            episode_id=self.current_episode_id or selected[0].episode_id,
            submit_step=current_step,
            items=selected,
            prompt=prompt,
        )

        for item in selected:
            item.status = "submitted"

        self.submitted_batches[batch.batch_id] = batch
        self.last_submit_step = current_step
        return batch

    def verify_batch(
        self,
        batch: Task2Batch,
        return_step: int,
    ) -> List[Task2Result]:
        started_at = time.time()

        try:
            raw_text = self._call_vlm(batch)
            parsed = self._parse_task2_response(raw_text)
            results = self._results_from_parsed(
                batch=batch,
                parsed=parsed,
                return_step=return_step,
                raw_text=raw_text,
                latency_ms=(time.time() - started_at) * 1000.0,
            )
        except Exception as exc:
            results = self._error_results(
                batch=batch,
                return_step=return_step,
                error=str(exc),
                latency_ms=(time.time() - started_at) * 1000.0,
            )

        self.history.extend(results)
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history :]

        for result in results:
            self.completed_candidate_ids.add(result.candidate_id)

        return results

    def mark_batch_completed(self, batch_id: str) -> Dict[str, Any]:
        batch_id = str(batch_id)
        batch = self.submitted_batches.pop(batch_id, None)

        if batch is None:
            return {
                "batch_id": batch_id,
                "removed": False,
                "reason": "missing_batch",
            }

        candidate_ids = []
        for item in batch.items:
            candidate_ids.append(item.candidate_id)
            item.clear_image_payload()
            item.status = "completed"

        return {
            "batch_id": batch_id,
            "removed": True,
            "candidate_ids": candidate_ids,
            "pending_count": len(self.pending_items),
            "inflight_count": len(self.submitted_batches),
        }

    # ------------------------------------------------------------------
    # Pending item preparation
    # ------------------------------------------------------------------

    def _prepare_pending_item(
        self,
        candidate: GDINOCandidate,
        target_info: TargetInfo,
        current_step: int,
    ) -> Optional[Task2PendingItem]:
        if candidate.candidate_id in self.seen_candidate_ids:
            return None
        if candidate.candidate_id in self.completed_candidate_ids:
            return None
        if candidate.rejected_reason:
            return None
        if candidate.frame is None:
            return None

        context_b64 = self._frame_context_b64(candidate)
        if not context_b64:
            return None

        crop_b64 = self._candidate_crop_b64(candidate)
        if not crop_b64:
            return None

        light_candidate = self._lightweight_candidate(
            candidate=candidate,
            crop_b64=crop_b64,
        )

        priority = self._candidate_priority(light_candidate)
        snapshot = self._candidate_snapshot(light_candidate)

        request = Task2Request.create(
            episode_id=candidate.episode_id,
            submit_step=int(current_step),
            target_info=target_info,
            candidate=light_candidate,
            context_frames=[],
            metadata={
                "source": "gdino_result",
                "candidate_snapshot": snapshot,
                "task2_priority": priority,
                "context_mime": candidate.frame.rgb_mime,
                "has_context_b64": True,
                "has_crop_b64": True,
            },
        )

        item = Task2PendingItem(
            request=request,
            priority=priority,
            context_b64=context_b64,
            context_mime=candidate.frame.rgb_mime,
            crop_b64=crop_b64,
            crop_mime="image/png",
            candidate_snapshot=snapshot,
            created_step=int(current_step),
        )

        self.seen_candidate_ids.add(candidate.candidate_id)
        return item

    def _add_pending_item(self, item: Task2PendingItem) -> Optional[Task2PendingItem]:
        if len(self.pending_items) < self.config.max_pending:
            self.pending_items.append(item)
            return item

        worst_idx = 0
        worst_priority = self.pending_items[0].priority

        for idx, old in enumerate(self.pending_items[1:], start=1):
            if old.priority < worst_priority:
                worst_idx = idx
                worst_priority = old.priority

        if item.priority <= worst_priority + self.config.replace_score_margin:
            return None

        old = self.pending_items.pop(worst_idx)
        old.clear_image_payload()
        self.pending_items.append(item)
        return item

    def _candidate_priority(self, candidate: GDINOCandidate) -> float:
        score = clamp01(candidate.score)
        quality = clamp01(candidate.quality_score)
        position = clamp01(candidate.position_confidence)
        depth = 1.0 if candidate.depth_valid else 0.0

        priority = (
            0.35 * score
            + 0.25 * quality
            + 0.20 * position
            + 0.10 * depth
            + 0.10
        )
        return clamp01(priority)

    def _frame_context_b64(self, candidate: GDINOCandidate) -> Optional[str]:
        frame = candidate.frame
        if frame is None:
            return None
        if frame.rgb_b64:
            return strip_data_uri(frame.rgb_b64)
        if frame.rgb_bytes:
            return bytes_to_b64(frame.rgb_bytes)
        return None

    def _candidate_crop_b64(self, candidate: GDINOCandidate) -> Optional[str]:
        frame = candidate.frame
        if frame is None:
            return candidate.crop_b64

        image = self._frame_to_pil(frame)
        if image is None:
            return candidate.crop_b64

        box = candidate.bbox.clipped()
        y_slice, x_slice = bbox_crop_slices(
            bbox=box,
            image_shape=(image.height, image.width),
            padding_ratio=self.config.crop_padding,
        )

        crop = image.crop(
            (
                int(x_slice.start),
                int(y_slice.start),
                int(x_slice.stop),
                int(y_slice.stop),
            )
        )

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _frame_to_pil(self, frame) -> Optional[Image.Image]:
        try:
            if frame.rgb_bytes:
                img = Image.open(io.BytesIO(frame.rgb_bytes))
                return img.convert("RGB")

            if frame.rgb_b64:
                raw = base64.b64decode(strip_data_uri(frame.rgb_b64))
                img = Image.open(io.BytesIO(raw))
                return img.convert("RGB")
        except Exception:
            return None
        return None

    def _lightweight_candidate(
        self,
        candidate: GDINOCandidate,
        crop_b64: str,
    ) -> GDINOCandidate:
        metadata = dict(candidate.metadata or {})
        metadata["task2_prepared"] = {
            "has_context_image": True,
            "has_crop_image": True,
            "frame_payload_removed": True,
        }

        return GDINOCandidate(
            candidate_id=candidate.candidate_id,
            episode_id=candidate.episode_id,
            step_id=candidate.step_id,
            view_id=candidate.view_id,
            frame_id=candidate.frame_id,
            bbox=candidate.bbox,
            label=candidate.label,
            score=candidate.score,
            prompt=candidate.prompt,
            frame=None,
            crop_b64=crop_b64,
            crop_mime="image/png",
            position_3d=candidate.position_3d,
            position_confidence=candidate.position_confidence,
            depth_valid=candidate.depth_valid,
            quality_score=candidate.quality_score,
            rejected_reason=candidate.rejected_reason,
            metadata=metadata,
        )

    def _candidate_snapshot(self, candidate: GDINOCandidate) -> Dict[str, Any]:
        frame = getattr(candidate, "frame", None)
        pose = getattr(frame, "pose", None) if frame is not None else None

        source_pose = {}
        if pose is not None:
            source_pose = {
                "x": float(getattr(pose, "x", 0.0)),
                "y": float(getattr(pose, "y", 0.0)),
                "z": float(getattr(pose, "z", 0.0)),
                "yaw": float(getattr(pose, "yaw", 0.0)),
            }

        frame_step_id = getattr(frame, "step_id", None) if frame is not None else None
        frame_view_id = getattr(frame, "view_id", None) if frame is not None else None

        return {
            "candidate_id": candidate.candidate_id,
            "episode_id": candidate.episode_id,
            "step_id": int(candidate.step_id),
            "view_id": candidate.view_id.value,
            "frame_id": candidate.frame_id,
            "frame_step_id": frame_step_id,
            "frame_view_id": None if frame_view_id is None else getattr(frame_view_id, "value", str(frame_view_id)),
            "source_pose": source_pose,
            "bbox": candidate.bbox.to_log_dict(),
            "label": candidate.label,
            "score": float(candidate.score),
            "quality_score": float(candidate.quality_score),
            "position_3d": candidate.position_3d,
            "position_confidence": float(candidate.position_confidence),
            "depth_valid": bool(candidate.depth_valid),
            "geometry": (candidate.metadata or {}).get("geometry", {}),
        }


    # ------------------------------------------------------------------
    # VLM call and parsing
    # ------------------------------------------------------------------

    def _call_vlm(self, batch: Task2Batch) -> str:
        provider = self._select_provider()

        if provider == "openai":
            return self._call_openai(batch)

        return self._call_dashscope(batch)

    def _select_provider(self) -> str:
        provider = str(self.config.provider or "dashscope").lower()
        if provider == "openai":
            return "openai"
        return "dashscope"

    def _call_openai(self, batch: Task2Batch) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.config.base_url,
        )

        content = []
        content.append({"type": "text", "text": batch.prompt})

        for idx, item in enumerate(batch.items, start=1):
            content.append({"type": "text", "text": "Candidate {} context image:".format(idx)})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": to_data_uri(item.context_b64, item.context_mime),
                    },
                }
            )
            content.append({"type": "text", "text": "Candidate {} crop image:".format(idx)})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": to_data_uri(item.crop_b64, item.crop_mime),
                    },
                }
            )

        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            timeout=self.config.request_timeout,
        )

        return response.choices[0].message.content or ""

    def _call_dashscope(self, batch: Task2Batch) -> str:
        from dashscope import MultiModalConversation

        content = []
        content.append({"text": batch.prompt})

        for idx, item in enumerate(batch.items, start=1):
            content.append({"text": "Candidate {} context image:".format(idx)})
            content.append({"image": to_data_uri(item.context_b64, item.context_mime)})
            content.append({"text": "Candidate {} crop image:".format(idx)})
            content.append({"image": to_data_uri(item.crop_b64, item.crop_mime)})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        response = MultiModalConversation.call(
            api_key=self.config.api_key or os.getenv("DASHSCOPE_API_KEY"),
            model=self.config.model_name,
            messages=messages,
        )

        return self._extract_dashscope_text(response)

    def _extract_dashscope_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response

        if hasattr(response, "output"):
            response = response.output

        if isinstance(response, dict):
            choices = response.get("choices")
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                return self._extract_content_text(content)

            output = response.get("output")
            if output is not None:
                return self._extract_dashscope_text(output)

        return str(response)

    def _extract_content_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                    elif "content" in item:
                        parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    def _parse_task2_response(self, raw_text: str) -> Dict[str, Any]:
        text = self._strip_code_fence(raw_text)

        try:
            data = json.loads(text)
        except Exception:
            data = self._extract_json_object(text)

        if isinstance(data, list):
            return {"results": data}

        if not isinstance(data, dict):
            return {"results": []}

        if "results" not in data:
            # Accept a single-result object.
            if "candidate_id" in data:
                return {"results": [data]}
            return {"results": []}

        return data

    def _strip_code_fence(self, text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
        return text

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        text = str(text or "")

        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj >= 0 and end_obj > start_obj:
            try:
                return json.loads(text[start_obj : end_obj + 1])
            except Exception:
                pass

        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_arr >= 0 and end_arr > start_arr:
            try:
                return {"results": json.loads(text[start_arr : end_arr + 1])}
            except Exception:
                pass

        return {"results": []}

    def _results_from_parsed(
        self,
        batch: Task2Batch,
        parsed: Dict[str, Any],
        return_step: int,
        raw_text: str,
        latency_ms: float,
    ) -> List[Task2Result]:
        result_items = parsed.get("results", []) or []
        by_candidate: Dict[str, Dict[str, Any]] = {}

        for item in result_items:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id", "")).strip()
            if candidate_id:
                by_candidate[candidate_id] = item

        results: List[Task2Result] = []

        for item in batch.items:
            parsed_item = by_candidate.get(item.candidate_id, {})
            decision = Task2Decision.from_any(parsed_item.get("verdict") or parsed_item.get("decision"))
            confidence = clamp01(parsed_item.get("confidence", 0.0))
            reason = str(parsed_item.get("reason", ""))

            if decision == Task2Decision.UNKNOWN:
                reason = reason or "missing_or_unparseable_candidate_result"

            result = Task2Result(
                request_id=item.request_id,
                candidate_id=item.candidate_id,
                episode_id=item.episode_id,
                submit_step=batch.submit_step,
                return_step=int(return_step),
                decision=decision,
                confidence=confidence,
                reason=reason,
                matched_attributes=list(parsed_item.get("matched_attributes", []) or []),
                failed_attributes=list(parsed_item.get("failed_attributes", []) or []),
                success=True,
                error=None,
                raw_response=raw_text,
                latency_ms=latency_ms,
                metadata={
                    "batch_id": batch.batch_id,
                    "candidate_snapshot": item.candidate_snapshot,
                    "task2_priority": item.priority,
                },
            )
            results.append(result)

        return results

    def _error_results(
        self,
        batch: Task2Batch,
        return_step: int,
        error: str,
        latency_ms: float,
    ) -> List[Task2Result]:
        results = []
        for item in batch.items:
            results.append(
                Task2Result(
                    request_id=item.request_id,
                    candidate_id=item.candidate_id,
                    episode_id=item.episode_id,
                    submit_step=batch.submit_step,
                    return_step=int(return_step),
                    decision=Task2Decision.UNKNOWN,
                    confidence=0.0,
                    reason="task2_verifier_error",
                    success=False,
                    error=error,
                    latency_ms=latency_ms,
                    metadata={
                        "batch_id": batch.batch_id,
                        "candidate_snapshot": item.candidate_snapshot,
                        "task2_priority": item.priority,
                    },
                )
            )
        return results

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        return len(self.pending_items)

    @property
    def inflight_count(self) -> int:
        return len(self.submitted_batches)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "current_episode_id": self.current_episode_id,
            "pending_count": self.pending_count,
            "inflight_count": self.inflight_count,
            "last_submit_step": self.last_submit_step,
            "last_update_step": self.last_update_step,
            "pending_items": [item.to_log_dict() for item in self.pending_items],
            "submitted_batches": {
                batch_id: batch.to_log_dict()
                for batch_id, batch in self.submitted_batches.items()
            },
            "history_count": len(self.history),
        }
