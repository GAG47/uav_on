from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .types import (
    BBox,
    FrameRecord,
    GDINOCandidate,
    GDINORequest,
    GDINOResult,
    PoseRecord,
    TargetInfo,
    ViewID,
    new_id,
    now_ts,
    strip_data_uri,
)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if value == "":
        return default

    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _camel_to_words(text: str) -> str:
    if not text:
        return ""

    chars = []
    previous = ""
    for ch in str(text):
        if previous and ch.isupper() and (previous.islower() or previous.isdigit()):
            chars.append(" ")
        chars.append(ch)
        previous = ch

    return "".join(chars).replace("_", " ").replace("-", " ").strip().lower()


def _deduplicate_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    output = []

    for item in items:
        text = str(item).strip().lower()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        output.append(text)

    return output


@dataclass
class GDINOClientConfig:
    enabled: bool = True
    server_url: str = "http://127.0.0.1:8008/detect"
    timeout: float = 20.0
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    max_images: int = 4

    @staticmethod
    def from_env() -> "GDINOClientConfig":
        return GDINOClientConfig(
            enabled=_env_bool("USE_SVNAV_GDINO", default=True),
            server_url=os.getenv(
                "SVNAV_GDINO_SERVER_URL",
                os.getenv("GDINO_SERVER_URL", "http://127.0.0.1:8008/detect"),
            ),
            timeout=float(
                os.getenv(
                    "SVNAV_GDINO_TIMEOUT",
                    os.getenv("GDINO_TIMEOUT", "20"),
                )
            ),
            box_threshold=float(
                os.getenv(
                    "SVNAV_GDINO_BOX_THRESHOLD",
                    os.getenv("GROUNDINGDINO_BOX_THRESHOLD", "0.25"),
                )
            ),
            text_threshold=float(
                os.getenv(
                    "SVNAV_GDINO_TEXT_THRESHOLD",
                    os.getenv("GROUNDINGDINO_TEXT_THRESHOLD", "0.20"),
                )
            ),
            max_images=int(
                os.getenv(
                    "SVNAV_GDINO_MAX_IMAGES",
                    os.getenv("GROUNDINGDINO_MAX_IMAGES", "4"),
                )
            ),
        )


class SVNavGDINOClient:
    """
    GroundingDINO HTTP client for SVNav.

    This class only does candidate generation:
    - send FrameRecord images to the GDINO server
    - parse raw detections
    - convert them into GDINOResult / GDINOCandidate

    It does not:
    - update SemanticMap
    - call VLM / Task2
    - switch navigation mode
    - output UAV actions
    - decide stop
    """

    def __init__(self, config: Optional[GDINOClientConfig] = None) -> None:
        self.config = config or GDINOClientConfig.from_env()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def health_url(self) -> str:
        url = self.config.server_url.rstrip("/")
        if url.endswith("/detect"):
            return url[: -len("/detect")] + "/health"
        return url + "/health"

    def health(self) -> Dict[str, Any]:
        try:
            request = urllib.request.Request(
                self.health_url(),
                method="GET",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except Exception as e:
            return {
                "ok": False,
                "available": False,
                "model_loaded": False,
                "error": str(e),
            }

    def build_prompt(self, target_info: TargetInfo) -> str:
        name = target_info.name or ""
        name_words = _camel_to_words(name)

        phrases = [
            name_words,
            name.lower(),
        ]

        aliases = {
            "woodenbox": [
                "wooden box",
                "wood box",
                "wooden crate",
                "crate",
                "storage box",
                "box",
            ],
            "wooden box": [
                "wooden box",
                "wood box",
                "wooden crate",
                "crate",
                "storage box",
                "box",
            ],
            "barrel": [
                "barrel",
                "oil barrel",
                "drum",
            ],
            "trashcan": [
                "trash can",
                "garbage can",
                "bin",
            ],
            "trash can": [
                "trash can",
                "garbage can",
                "bin",
            ],
        }

        key_candidates = [
            name.strip().lower(),
            name_words.strip().lower(),
            name.strip().lower().replace(" ", ""),
        ]

        for key in key_candidates:
            if key in aliases:
                phrases.extend(aliases[key])

        if target_info.description:
            description_words = _camel_to_words(target_info.description)
            if 0 < len(description_words.split()) <= 5:
                phrases.append(description_words)

        prompt_phrases = _deduplicate_keep_order(phrases)
        if not prompt_phrases:
            prompt_phrases = ["object"]

        return ". ".join(prompt_phrases) + "."

    def detect_frame(
        self,
        frame: FrameRecord,
        prompt: Optional[str] = None,
        target_info: Optional[TargetInfo] = None,
        submit_step: Optional[int] = None,
        return_step: Optional[int] = None,
        trigger_reason: str = "manual",
    ) -> GDINOResult:
        return self.detect_frames(
            frames=[frame],
            prompt=prompt,
            target_info=target_info,
            submit_step=submit_step,
            return_step=return_step,
            trigger_reason=trigger_reason,
        )

    def detect_frames(
        self,
        frames: Sequence[FrameRecord],
        prompt: Optional[str] = None,
        target_info: Optional[TargetInfo] = None,
        submit_step: Optional[int] = None,
        return_step: Optional[int] = None,
        trigger_reason: str = "manual",
    ) -> GDINOResult:
        request_id = new_id("gdino")
        frames = list(frames or [])

        if frames:
            first_frame = frames[0]
            episode_id = first_frame.episode_id
            if submit_step is None:
                submit_step = first_frame.step_id
            if return_step is None:
                return_step = submit_step
            if target_info is None:
                target_info = first_frame.target_info
        else:
            episode_id = ""
            if submit_step is None:
                submit_step = 0
            if return_step is None:
                return_step = submit_step

        started_at = time.time()

        if not self.enabled:
            return self._failure_result(
                request_id=request_id,
                episode_id=episode_id,
                submit_step=submit_step,
                return_step=return_step,
                error="SVNav GDINO client is disabled",
                started_at=started_at,
                metadata={"trigger_reason": trigger_reason},
            )

        valid_frames = []
        for frame in frames:
            if frame is None:
                continue
            if not frame.has_image():
                continue
            valid_frames.append(frame)

        if not valid_frames:
            return self._failure_result(
                request_id=request_id,
                episode_id=episode_id,
                submit_step=submit_step,
                return_step=return_step,
                error="no valid frame image for GDINO",
                started_at=started_at,
                metadata={"trigger_reason": trigger_reason},
            )

        valid_frames = valid_frames[: max(1, int(self.config.max_images))]

        if prompt is None:
            if target_info is None:
                prompt = "object."
            else:
                prompt = self.build_prompt(target_info)

        image_items = []
        image_index_to_frame = {}

        for image_index, frame in enumerate(valid_frames):
            image_b64 = frame.rgb_b64
            if image_b64 is None and frame.rgb_bytes is not None:
                image_b64 = base64.b64encode(frame.rgb_bytes).decode("utf-8")

            image_b64 = strip_data_uri(image_b64)
            if not image_b64:
                continue

            image_index_to_frame[image_index] = frame
            image_items.append(
                {
                    "image_index": image_index,
                    "frame_id": frame.frame_id,
                    "view_id": frame.view_id.value,
                    "step_id": frame.step_id,
                    "data": image_b64,
                }
            )

        if not image_items:
            return self._failure_result(
                request_id=request_id,
                episode_id=episode_id,
                submit_step=submit_step,
                return_step=return_step,
                error="all frame images are empty",
                started_at=started_at,
                metadata={"trigger_reason": trigger_reason},
            )

        payload = {
            "request_id": request_id,
            "prompt": prompt,
            "text": prompt,
            "images": image_items,
            "box_threshold": self.config.box_threshold,
            "text_threshold": self.config.text_threshold,
        }

        try:
            response = self._post_json(self.config.server_url, payload)
        except Exception as e:
            return self._failure_result(
                request_id=request_id,
                episode_id=episode_id,
                submit_step=submit_step,
                return_step=return_step,
                error=str(e),
                started_at=started_at,
                metadata={
                    "trigger_reason": trigger_reason,
                    "server_url": self.config.server_url,
                    "prompt": prompt,
                },
            )

        if not response.get("available", False):
            return self._failure_result(
                request_id=request_id,
                episode_id=episode_id,
                submit_step=submit_step,
                return_step=return_step,
                error=response.get("error", "GDINO server unavailable"),
                started_at=started_at,
                metadata={
                    "trigger_reason": trigger_reason,
                    "server_url": self.config.server_url,
                    "prompt": prompt,
                    "raw_response": response,
                },
            )

        candidates = []
        detections = response.get("detections", []) or []

        for detection in detections:
            candidate = self._candidate_from_detection(
                detection=detection,
                request_id=request_id,
                episode_id=episode_id,
                prompt=prompt,
                image_index_to_frame=image_index_to_frame,
                trigger_reason=trigger_reason,
            )
            if candidate is not None:
                candidates.append(candidate)

        latency_ms = (time.time() - started_at) * 1000.0

        return GDINOResult(
            request_id=request_id,
            episode_id=episode_id,
            submit_step=submit_step,
            return_step=return_step,
            candidates=candidates,
            success=True,
            error=response.get("error") or None,
            latency_ms=latency_ms,
            returned_at=now_ts(),
            metadata={
                "trigger_reason": trigger_reason,
                "server_url": self.config.server_url,
                "prompt": prompt,
                "box_threshold": self.config.box_threshold,
                "text_threshold": self.config.text_threshold,
                "raw_num_detections": response.get("num_detections", len(detections)),
                "best_score": response.get("best_score", 0.0),
            },
        )


    def detect_request(self, request: GDINORequest) -> GDINOResult:
        """
        Run a structured SVNav GDINORequest.

        This keeps request_id and request metadata stable across asynchronous
        scheduling. The underlying detect_frames() still sends FrameRecord images
        to the HTTP server and converts raw detections into GDINOCandidate.
        """
        result = self.detect_frames(
            frames=request.frames,
            prompt=request.prompt,
            target_info=request.target_info,
            submit_step=request.submit_step,
            return_step=request.submit_step,
            trigger_reason="gdino_keyframe",
        )

        old_request_id = result.request_id
        result.request_id = request.request_id
        result.episode_id = request.episode_id
        result.submit_step = request.submit_step

        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "request_id": request.request_id,
                "internal_client_request_id": old_request_id,
                "source": "GDINORequest",
                "gdino_request_metadata": request.metadata,
                "gdino_frame_ids": request.frame_ids,
                "gdino_view_ids": request.view_ids,
            }
        )
        result.metadata = metadata

        for candidate in result.candidates:
            candidate.metadata["request_id"] = request.request_id
            candidate.metadata["internal_client_request_id"] = old_request_id
            candidate.metadata["gdino_request_metadata"] = request.metadata

        return result

    def _candidate_from_detection(
        self,
        detection: Dict[str, Any],
        request_id: str,
        episode_id: str,
        prompt: str,
        image_index_to_frame: Dict[int, FrameRecord],
        trigger_reason: str,
    ) -> Optional[GDINOCandidate]:
        image_index = _safe_int(detection.get("image_index", 0), 0)
        frame = image_index_to_frame.get(image_index)
        if frame is None:
            return None

        bbox_values = detection.get("bbox", None)
        if not bbox_values or len(bbox_values) != 4:
            return None

        image_width = _safe_int(
            detection.get("image_width", frame.image_width or 0),
            frame.image_width or 0,
        )
        image_height = _safe_int(
            detection.get("image_height", frame.image_height or 0),
            frame.image_height or 0,
        )

        bbox = BBox(
            x1=_safe_float(bbox_values[0]),
            y1=_safe_float(bbox_values[1]),
            x2=_safe_float(bbox_values[2]),
            y2=_safe_float(bbox_values[3]),
            image_width=image_width,
            image_height=image_height,
            normalized=False,
        )

        score = _safe_float(detection.get("score", 0.0), 0.0)
        label = str(detection.get("phrase", "") or detection.get("label", "") or "")

        return GDINOCandidate(
            candidate_id=new_id("cand"),
            episode_id=episode_id or frame.episode_id,
            step_id=frame.step_id,
            view_id=ViewID.from_any(frame.view_id),
            frame_id=frame.frame_id,
            bbox=bbox,
            label=label,
            score=score,
            prompt=prompt,
            frame=frame,
            crop_bytes=None,
            crop_b64=None,
            crop_mime="image/png",
            crop_path=None,
            position_3d=None,
            position_confidence=0.0,
            depth_valid=False,
            quality_score=score,
            rejected_reason=None,
            timestamp=now_ts(),
            metadata={
                "request_id": request_id,
                "trigger_reason": trigger_reason,
                "image_index": image_index,
                "bbox_norm_cxcywh": detection.get("bbox_norm_cxcywh"),
                "raw_detection": detection,
            },
        )

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError("GDINO HTTP error {}: {}".format(e.code, error_body))
        except urllib.error.URLError as e:
            raise RuntimeError("GDINO URL error: {}".format(e))
        except Exception as e:
            raise RuntimeError("GDINO request failed: {}".format(e))

    def _failure_result(
        self,
        request_id: str,
        episode_id: str,
        submit_step: int,
        return_step: int,
        error: str,
        started_at: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GDINOResult:
        latency_ms = (time.time() - started_at) * 1000.0

        return GDINOResult(
            request_id=request_id,
            episode_id=episode_id,
            submit_step=submit_step,
            return_step=return_step,
            candidates=[],
            success=False,
            error=error,
            latency_ms=latency_ms,
            returned_at=now_ts(),
            metadata=metadata or {},
        )


def _read_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("SVNAV_GDINO_SERVER_URL", "http://127.0.0.1:8008/detect"),
    )
    parser.add_argument("--view", type=str, default="front")
    parser.add_argument("--episode_id", type=str, default="manual_gdino_test")
    parser.add_argument("--step_id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    args = parser.parse_args()

    target_info = TargetInfo(name=args.target)
    frame = FrameRecord(
        episode_id=args.episode_id,
        step_id=args.step_id,
        view_id=ViewID.from_any(args.view),
        pose=PoseRecord(x=0.0, y=0.0, z=0.0),
        target_info=target_info,
        rgb_bytes=_read_image_bytes(args.image),
        rgb_mime="image/png",
        metadata={
            "source": "manual_gdino_client_test",
            "image_path": args.image,
        },
    )

    client = SVNavGDINOClient(
        GDINOClientConfig(
            enabled=True,
            server_url=args.url,
            timeout=args.timeout,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_images=1,
        )
    )

    result = client.detect_frame(
        frame=frame,
        target_info=target_info,
        trigger_reason="manual_test",
    )

    print(json.dumps(result.to_log_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
