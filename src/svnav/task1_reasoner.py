from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from svnav.prompts import (
    SVNAV_TASK1_SYSTEM_PROMPT,
    build_task1_user_prompt,
)
from svnav.types import (
    FrameRecord,
    RegionScore,
    Task1Request,
    Task1Result,
    clamp01,
)


@dataclass
class Task1ReasonerConfig:
    model_name: str = "qwen-vl-max"
    max_retries: int = 2
    retry_sleep: float = 1.0
    require_image: bool = True
    store_raw_response: bool = True

    def __post_init__(self) -> None:
        self.max_retries = int(self.max_retries)
        self.retry_sleep = float(self.retry_sleep)
        if self.max_retries < 0:
            self.max_retries = 0
        if self.retry_sleep < 0:
            self.retry_sleep = 0.0


class Task1Reasoner:
    """
    SVNav Task1 reasoner.

    Input:
        Task1Request with keyframe images.

    Output:
        Task1Result with one RegionScore per frame_id.

    This class only calls the VLM and parses the result.
    It does not update SemanticMap, does not output navigation actions,
    does not call GDINO, and does not decide stop.
    """

    def __init__(self, config: Optional[Task1ReasonerConfig] = None) -> None:
        self.config = config or Task1ReasonerConfig()

    def run(self, request: Task1Request, return_step: int) -> Task1Result:
        start_time = time.time()

        valid_frames = self._valid_image_frames(request.frames)
        if not valid_frames:
            return Task1Result(
                request_id=request.request_id,
                episode_id=request.episode_id,
                submit_step=request.submit_step,
                return_step=return_step,
                scores=[],
                success=False,
                error="Task1Request contains no image keyframes.",
                raw_response=None,
                latency_ms=0.0,
            )

        messages = self.build_messages(request, valid_frames=valid_frames)

        last_error = None
        raw_text = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.call_dashscope(messages)
                raw_text = self.extract_response_text(response)
                latency_ms = (time.time() - start_time) * 1000.0
                return self.parse_response(
                    raw_text=raw_text,
                    request=request,
                    return_step=return_step,
                    latency_ms=latency_ms,
                )
            except Exception as exc:
                last_error = "{}".format(exc)
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_sleep)

        latency_ms = (time.time() - start_time) * 1000.0
        return Task1Result(
            request_id=request.request_id,
            episode_id=request.episode_id,
            submit_step=request.submit_step,
            return_step=return_step,
            scores=[],
            success=False,
            error="Task1 DashScope call failed: {}\n{}".format(
                last_error,
                traceback.format_exc(),
            ),
            raw_response=raw_text if self.config.store_raw_response else None,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def build_messages(
        self,
        request: Task1Request,
        valid_frames: Optional[List[FrameRecord]] = None,
    ) -> List[Dict[str, Any]]:
        frames = valid_frames if valid_frames is not None else self._valid_image_frames(request.frames)

        frame_items = []
        for frame in frames:
            frame_items.append(
                {
                    "frame_id": frame.frame_id,
                    "step_id": frame.step_id,
                    "view_id": frame.view_id.value,
                    "pose_text": self._pose_text(frame),
                }
            )

        user_prompt = build_task1_user_prompt(
            target_text=request.target_info.text,
            frame_items=frame_items,
        )

        user_content: List[Dict[str, Any]] = [{"text": user_prompt}]

        for frame in frames:
            user_content.append(
                {
                    "text": (
                        "Keyframe image for frame_id={}, step_id={}, view_id={}."
                    ).format(
                        frame.frame_id,
                        frame.step_id,
                        frame.view_id.value,
                    )
                }
            )
            user_content.append({"image": frame.rgb_data_uri})

        messages = [
            {
                "role": "system",
                "content": [SVNAV_TASK1_SYSTEM_PROMPT],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        return messages

    def _valid_image_frames(self, frames: List[FrameRecord]) -> List[FrameRecord]:
        valid = []
        for frame in frames:
            if frame.rgb_data_uri:
                valid.append(frame)
        return valid

    @staticmethod
    def _pose_text(frame: FrameRecord) -> str:
        pose = frame.pose
        return "x={:.2f}, y={:.2f}, z={:.2f}, yaw={:.2f}".format(
            float(pose.x),
            float(pose.y),
            float(pose.z),
            float(pose.yaw),
        )

    # ------------------------------------------------------------------
    # DashScope call
    # ------------------------------------------------------------------

    def call_dashscope(self, messages: List[Dict[str, Any]]) -> Any:
        from dashscope import MultiModalConversation

        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
        }

        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key

        return MultiModalConversation.call(**kwargs)

    @staticmethod
    def extract_response_text(response: Any) -> str:
        """
        Extract text from DashScope MultiModalConversation response.

        Supports the dict-like response used by UAV-ON baseline:
            response["output"]["choices"][0]["message"].content[0]["text"]
        """
        if response is None:
            raise RuntimeError("DashScope response is None")

        if isinstance(response, dict):
            if "output" not in response:
                raise RuntimeError("DashScope response missing output: {}".format(response))

            choices = response["output"].get("choices", [])
            if not choices:
                raise RuntimeError("DashScope response missing choices: {}".format(response))

            message = choices[0].get("message")
            if message is None:
                raise RuntimeError("DashScope response missing message: {}".format(response))

            content = message.get("content", None)
            if content is None and hasattr(message, "content"):
                content = message.content

            return Task1Reasoner._extract_text_from_content(content)

        output = getattr(response, "output", None)
        if output is not None:
            choices = output.get("choices", []) if isinstance(output, dict) else getattr(output, "choices", [])
            if choices:
                first = choices[0]
                message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
                content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
                return Task1Reasoner._extract_text_from_content(content)

        raise RuntimeError("Unsupported DashScope response format: {}".format(type(response)))

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        if content is None:
            raise RuntimeError("DashScope message content is None")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    texts.append(str(item["text"]))
                else:
                    text = getattr(item, "text", None)
                    if text is not None:
                        texts.append(str(text))

            text = "\n".join(texts).strip()
            if text:
                return text

        text = str(content).strip()
        if text:
            return text

        raise RuntimeError("No text found in DashScope message content")

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_response(
        self,
        raw_text: str,
        request: Task1Request,
        return_step: int,
        latency_ms: Optional[float] = None,
    ) -> Task1Result:
        try:
            parsed = self._load_json_from_text(raw_text)
        except Exception as exc:
            return Task1Result(
                request_id=request.request_id,
                episode_id=request.episode_id,
                submit_step=request.submit_step,
                return_step=return_step,
                scores=[],
                success=False,
                error="Failed to parse Task1 JSON: {}".format(exc),
                raw_response=raw_text if self.config.store_raw_response else None,
                latency_ms=latency_ms,
            )

        scores_data = self._extract_scores_data(parsed)
        frame_by_id = {frame.frame_id: frame for frame in request.frames}

        scores: List[RegionScore] = []
        ignored_items = []
        seen_frame_ids = set()

        for item in scores_data:
            if not isinstance(item, dict):
                ignored_items.append(item)
                continue

            frame_id = str(item.get("frame_id", "")).strip()
            if not frame_id:
                ignored_items.append(item)
                continue

            if frame_id not in frame_by_id:
                ignored_items.append(item)
                continue

            if frame_id in seen_frame_ids:
                continue

            frame = frame_by_id[frame_id]
            semantic_value = clamp01(item.get("semantic_value", 0.0))
            confidence = clamp01(item.get("confidence", 0.0))
            reason = str(item.get("reason", "")).strip()

            scores.append(
                RegionScore(
                    request_id=request.request_id,
                    episode_id=request.episode_id,
                    step_id=frame.step_id,
                    view_id=frame.view_id,
                    frame_id=frame.frame_id,
                    semantic_value=semantic_value,
                    confidence=confidence,
                    reason=reason,
                    metadata={
                        "raw_item": item,
                    },
                )
            )
            seen_frame_ids.add(frame_id)

        missing_frame_ids = [
            frame.frame_id for frame in request.frames if frame.frame_id not in seen_frame_ids
        ]

        success = len(scores) > 0
        error = None if success else "No valid Task1 scores returned."

        return Task1Result(
            request_id=request.request_id,
            episode_id=request.episode_id,
            submit_step=request.submit_step,
            return_step=return_step,
            scores=scores,
            success=success,
            error=error,
            raw_response=raw_text if self.config.store_raw_response else None,
            latency_ms=latency_ms,
            metadata={
                "missing_frame_ids": missing_frame_ids,
                "ignored_items": ignored_items,
            },
        )

    @staticmethod
    def _extract_scores_data(parsed: Any) -> List[Any]:
        if isinstance(parsed, list):
            return parsed

        if not isinstance(parsed, dict):
            return []

        for key in ("scores", "frame_scores", "results", "items"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value

        return []

    @staticmethod
    def _load_json_from_text(text: str) -> Any:
        cleaned = Task1Reasoner._strip_code_fence(text)

        try:
            return json.loads(cleaned)
        except Exception:
            pass

        block = Task1Reasoner._find_first_json_block(cleaned)
        if block is None:
            raise ValueError("No JSON object or array found in response")

        return json.loads(block)

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = str(text).strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        return text

    @staticmethod
    def _find_first_json_block(text: str) -> Optional[str]:
        starts = []
        for idx, ch in enumerate(text):
            if ch in ("{", "["):
                starts.append(idx)

        for start in starts:
            opening = text[start]
            closing = "}" if opening == "{" else "]"
            stack = []

            for idx in range(start, len(text)):
                ch = text[idx]
                if ch == opening:
                    stack.append(ch)
                elif ch == closing:
                    if stack:
                        stack.pop()
                    if not stack:
                        return text[start: idx + 1]

        return None


__all__ = [
    "Task1Reasoner",
    "Task1ReasonerConfig",
]
