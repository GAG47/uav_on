import asyncio
import json
import os
from io import BytesIO

import numpy as np
from PIL import Image


class TargetVerifier:
    def __init__(
        self,
        client=None,
        model="gpt-4.1-mini",
        verification_threshold=0.58,
        max_candidates=4,
        hard_reject_large_area=0.85,
        small_object_large_area=0.18,
        require_stable_track=True,
        min_stable_step_count=2,
        debug_dir="debug/target_verifier",
        save_debug_crops=True,
    ):
        self.client = client
        self.model = model
        self.verification_threshold = verification_threshold
        self.max_candidates = max_candidates
        self.hard_reject_large_area = hard_reject_large_area
        self.small_object_large_area = small_object_large_area
        self.require_stable_track = require_stable_track
        self.min_stable_step_count = min_stable_step_count
        self.debug_dir = debug_dir
        self.save_debug_crops = save_debug_crops

    def verify_sync(
        self,
        object_name,
        object_size,
        description,
        captions4,
        rgb_images,
        tracker_info,
        semantic_result=None,
        encode_image_fn=None,
        generate_caption_fn=None,
    ):
        return asyncio.run(
            self.verify(
                object_name=object_name,
                object_size=object_size,
                description=description,
                captions4=captions4,
                rgb_images=rgb_images,
                tracker_info=tracker_info,
                semantic_result=semantic_result,
                encode_image_fn=encode_image_fn,
                generate_caption_fn=generate_caption_fn,
            )
        )

    async def verify(
        self,
        object_name,
        object_size,
        description,
        captions4,
        rgb_images,
        tracker_info,
        semantic_result=None,
        encode_image_fn=None,
        generate_caption_fn=None,
    ):
        if tracker_info is None:
            tracker_info = {}

        if not tracker_info.get("verification_required", False):
            return self.default_result(
                checked=False,
                verified=False,
                reason="verification not required"
            )

        current_step_num = self.get_current_step_num(tracker_info)
        candidates = tracker_info.get("verification_candidates", [])
        candidates = self.filter_candidates(
            candidates=candidates,
            object_size=object_size
        )

        if len(candidates) == 0:
            return self.default_result(
                checked=True,
                verified=False,
                reason="no stable task-aware object candidates",
                reject_reason="tracker did not provide a valid stable candidate"
            )

        candidates = candidates[:self.max_candidates]
        candidates = self.attach_crop_captions(
            candidates=candidates,
            rgb_images=rgb_images,
            current_step_num=current_step_num,
            encode_image_fn=encode_image_fn,
            generate_caption_fn=generate_caption_fn
        )

        candidates = self.filter_caption_ready_candidates(candidates)
        if len(candidates) == 0:
            return self.default_result(
                checked=True,
                verified=False,
                reason="no candidate has a valid crop caption for object verification",
                reject_reason="empty crop caption or failed crop generation",
                candidates=candidates
            )

        if self.client is None:
            return self.default_result(
                checked=True,
                verified=False,
                reason="target verifier client is None",
                candidates=candidates
            )

        prompt = self.build_prompt(
            object_name=object_name,
            object_size=object_size,
            description=description,
            captions4=captions4,
            candidates=candidates,
            semantic_result=semantic_result,
            tracker_info=tracker_info
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0
            )

            text = response.choices[0].message.content.strip()
            parsed = self.parse_json_result(text)

            selected_candidate_id = parsed.get("selected_candidate_id", None)
            selected_track_id = parsed.get("selected_track_id", None)
            confidence = self.normalize_score(parsed.get("confidence", 0.0))
            reason = parsed.get("reason", "")
            reject_reason = parsed.get("reject_reason", "")
            decision = str(parsed.get("decision", "")).lower().strip()

            selected_candidate = self.find_candidate(
                candidates=candidates,
                candidate_id=selected_candidate_id
            )

            valid, validation_reason = self.validate_selected_candidate(
                selected_candidate=selected_candidate,
                selected_track_id=selected_track_id,
                confidence=confidence
            )

            verified = (
                decision in ["select", "verify", "verified", "true"]
                and valid
                and confidence >= self.verification_threshold
            )

            if not verified:
                if len(validation_reason) > 0:
                    reason = (str(reason) + " " + validation_reason).strip()
                selected_candidate_id = None
                selected_track_id = None
                selected_candidate = None

            crop_caption = self.get_selected_or_best_crop_caption(
                candidates=candidates,
                selected_candidate=selected_candidate
            )

            return {
                "checked": True,
                "verified": verified,
                "confidence": confidence,
                "same_object": verified,
                "selected_candidate_id": selected_candidate_id,
                "selected_track_id": selected_track_id,
                "selected_candidate": selected_candidate,
                "reason": reason,
                "reject_reason": reject_reason,
                "hard_reject": False,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "candidate_debug": self.build_candidate_debug(candidates),
                "crop_caption": crop_caption,
                "raw_response": text,
                "parsed_response": parsed,
                "verification_threshold": self.verification_threshold,
            }

        except Exception as e:
            return self.default_result(
                checked=True,
                verified=False,
                reason=f"target verification failed: {e}",
                candidates=candidates
            )

    def get_current_step_num(self, tracker_info):
        observation = tracker_info.get("observation", {})
        if isinstance(observation, dict):
            try:
                return int(observation.get("step_num", -1))
            except Exception:
                return -1
        return -1

    def filter_candidates(self, candidates, object_size):
        if not isinstance(candidates, list):
            return []

        size_level = self.parse_size_level(object_size)
        valid_candidates = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            if candidate.get("target_world_position", None) is None:
                continue

            if candidate.get("camera_region", "unknown") not in ["front", "left", "right"]:
                continue

            if bool(candidate.get("full_frame_like_box", False)):
                continue

            if bool(candidate.get("edge_like_box", False)):
                continue

            area_ratio = float(candidate.get("area_ratio", 0.0))
            if area_ratio >= self.hard_reject_large_area:
                continue

            if size_level == "small" and area_ratio >= self.small_object_large_area:
                continue

            if self.require_stable_track:
                if not bool(candidate.get("track_stable", False)):
                    continue

                stable_step_count = int(candidate.get("stable_step_count", 0))
                if stable_step_count < self.min_stable_step_count:
                    continue

                if candidate.get("track_id", None) is None:
                    continue

            valid_candidates.append(dict(candidate))

        valid_candidates.sort(
            key=lambda item: item.get("verification_priority", item.get("candidate_quality", 0.0)),
            reverse=True
        )
        return valid_candidates

    def filter_caption_ready_candidates(self, candidates):
        valid_candidates = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            crop_caption = candidate.get("crop_caption", "")
            if not isinstance(crop_caption, str):
                continue

            if len(crop_caption.strip()) == 0:
                continue

            crop_debug = candidate.get("crop_debug", {})
            if isinstance(crop_debug, dict):
                if crop_debug.get("ok", True) is False:
                    continue

            candidate["caption_ready"] = True
            valid_candidates.append(candidate)

        valid_candidates.sort(
            key=lambda item: item.get("verification_priority", item.get("candidate_quality", 0.0)),
            reverse=True
        )
        return valid_candidates

    def validate_selected_candidate(self, selected_candidate, selected_track_id, confidence):
        if selected_candidate is None:
            return False, "No valid candidate was selected."

        crop_caption = selected_candidate.get("crop_caption", "")
        if not isinstance(crop_caption, str) or len(crop_caption.strip()) == 0:
            return False, "Selected candidate has no crop caption."

        if selected_candidate.get("camera_region", "unknown") not in ["front", "left", "right"]:
            return False, "Selected candidate is not from a horizontal camera view."

        if self.require_stable_track:
            if not bool(selected_candidate.get("track_stable", False)):
                return False, "Selected candidate is not supported by a stable track."

            stable_step_count = int(selected_candidate.get("stable_step_count", 0))
            if stable_step_count < self.min_stable_step_count:
                return False, "Selected candidate has insufficient cross-step support."

            candidate_track_id = selected_candidate.get("track_id", None)
            if candidate_track_id is None:
                return False, "Selected candidate has no track id."

            if selected_track_id is not None and selected_track_id != candidate_track_id:
                return False, "Selected track id does not match selected candidate track id."

        if bool(selected_candidate.get("full_frame_like_box", False)):
            return False, "Selected candidate is a full-frame-like detection."

        if bool(selected_candidate.get("edge_like_box", False)):
            return False, "Selected candidate is an edge-like detection."

        if confidence < self.verification_threshold:
            return False, "Verifier confidence is below threshold."

        return True, ""

    def attach_crop_captions(
        self,
        candidates,
        rgb_images,
        current_step_num=-1,
        encode_image_fn=None,
        generate_caption_fn=None
    ):
        if encode_image_fn is None or generate_caption_fn is None:
            for candidate in candidates:
                if not self.has_cached_caption(candidate):
                    candidate["crop_caption"] = ""
                    candidate["crop_debug"] = self.default_crop_debug(
                        candidate=candidate,
                        stage="check_function",
                        reason="encode_image_fn or generate_caption_fn is None"
                    )
            return candidates

        crop_bytes_list = []
        crop_indices = []

        for idx, candidate in enumerate(candidates):
            if self.has_cached_caption(candidate):
                crop_debug = candidate.get("crop_debug", {})
                if not isinstance(crop_debug, dict):
                    crop_debug = {}
                crop_debug["ok"] = True
                crop_debug["stage"] = "cached"
                crop_debug["reason"] = "use cached crop caption from candidate keyframe"
                candidate["crop_debug"] = crop_debug
                candidate["caption_source"] = "cached"
                continue

            candidate_step_num = int(candidate.get("step_num", -1))
            if current_step_num >= 0 and candidate_step_num != current_step_num:
                candidate["crop_caption"] = ""
                candidate["crop_debug"] = self.default_crop_debug(
                    candidate=candidate,
                    stage="stale_candidate_without_cached_crop",
                    reason=(
                        "candidate is from an old keyframe and has no cached crop caption; "
                        "skip recropping to avoid using old bbox on current image"
                    )
                )
                candidate["caption_ready"] = False
                candidate["caption_source"] = "missing_cache"
                continue

            crop, crop_debug = self.build_candidate_crop(
                rgb_images=rgb_images,
                candidate=candidate
            )
            candidate["crop_debug"] = crop_debug

            if crop is None:
                candidate["crop_caption"] = ""
                candidate["caption_ready"] = False
                candidate["caption_source"] = "crop_failed"
                continue

            crop_bytes = self.pil_to_png_bytes(crop)
            if crop_bytes is None:
                candidate["crop_caption"] = ""
                candidate["crop_debug"]["ok"] = False
                candidate["crop_debug"]["stage"] = "crop_to_bytes"
                candidate["crop_debug"]["reason"] = "failed to convert crop PIL image to PNG bytes"
                candidate["caption_ready"] = False
                candidate["caption_source"] = "crop_to_bytes_failed"
                continue

            crop_bytes_list.append(crop_bytes)
            crop_indices.append(idx)

        if len(crop_bytes_list) == 0:
            return candidates

        try:
            crop_b64 = encode_image_fn(crop_bytes_list)
            for idx in crop_indices:
                candidates[idx]["crop_debug"]["stage"] = "encode_crop"
                candidates[idx]["crop_debug"]["encoded_type"] = str(type(crop_b64))
                candidates[idx]["crop_debug"]["encoded_count"] = len(crop_b64) if isinstance(crop_b64, list) else 0

            crop_captions = generate_caption_fn(crop_b64)
            for idx in crop_indices:
                candidates[idx]["crop_debug"]["stage"] = "generate_caption"
                candidates[idx]["crop_debug"]["caption_type"] = str(type(crop_captions))
                candidates[idx]["crop_debug"]["caption_raw"] = str(crop_captions)

            if not isinstance(crop_captions, list):
                for idx in crop_indices:
                    candidates[idx]["crop_caption"] = ""
                    candidates[idx]["crop_debug"]["ok"] = False
                    candidates[idx]["crop_debug"]["reason"] = "generate_caption did not return a list"
                    candidates[idx]["caption_ready"] = False
                    candidates[idx]["caption_source"] = "caption_failed"
                return candidates

            for local_idx, candidate_idx in enumerate(crop_indices):
                if local_idx < len(crop_captions):
                    caption = str(crop_captions[local_idx])
                    candidates[candidate_idx]["crop_caption"] = caption
                    candidates[candidate_idx]["crop_debug"]["ok"] = len(caption.strip()) > 0
                    candidates[candidate_idx]["crop_debug"]["caption_len"] = len(caption)
                    candidates[candidate_idx]["crop_debug"]["reason"] = "caption generated"
                    candidates[candidate_idx]["crop_debug"]["stage"] = "done"
                    candidates[candidate_idx]["caption_ready"] = len(caption.strip()) > 0
                    candidates[candidate_idx]["caption_step"] = candidates[candidate_idx].get("step_num", -1)
                    candidates[candidate_idx]["caption_source"] = "current_keyframe"
                else:
                    candidates[candidate_idx]["crop_caption"] = ""
                    candidates[candidate_idx]["crop_debug"]["ok"] = False
                    candidates[candidate_idx]["crop_debug"]["caption_len"] = 0
                    candidates[candidate_idx]["crop_debug"]["reason"] = "caption list shorter than crop list"
                    candidates[candidate_idx]["caption_ready"] = False
                    candidates[candidate_idx]["caption_source"] = "caption_missing"

        except Exception as e:
            for idx in crop_indices:
                candidates[idx]["crop_caption"] = ""
                candidates[idx]["crop_debug"]["ok"] = False
                candidates[idx]["crop_debug"]["stage"] = "exception"
                candidates[idx]["crop_debug"]["reason"] = str(e)
                candidates[idx]["caption_ready"] = False
                candidates[idx]["caption_source"] = "exception"

        return candidates

    def has_cached_caption(self, candidate):
        crop_caption = candidate.get("crop_caption", "")
        if not isinstance(crop_caption, str):
            return False
        if len(crop_caption.strip()) == 0:
            return False
        if bool(candidate.get("caption_ready", False)):
            return True
        return True

    def build_candidate_crop(self, rgb_images, candidate):
        crop_debug = self.default_crop_debug(
            candidate=candidate,
            stage="start",
            reason=""
        )

        try:
            image_index = int(candidate.get("image_index", -1))
            bbox = candidate.get("bbox", None)
            crop_debug["image_index"] = image_index
            crop_debug["bbox"] = bbox

            if rgb_images is None:
                crop_debug["stage"] = "check_rgb_images"
                crop_debug["reason"] = "rgb_images is None"
                return None, crop_debug

            crop_debug["rgb_count"] = len(rgb_images)
            if image_index < 0 or image_index >= len(rgb_images):
                crop_debug["stage"] = "check_image_index"
                crop_debug["reason"] = f"image_index out of range: {image_index}, rgb_count={len(rgb_images)}"
                return None, crop_debug

            if bbox is None or len(bbox) != 4:
                crop_debug["stage"] = "check_bbox"
                crop_debug["reason"] = f"invalid bbox: {bbox}"
                return None, crop_debug

            raw_image = rgb_images[image_index]
            crop_debug["image_type"] = str(type(raw_image))
            image = self.to_pil_image(raw_image)
            if image is None:
                crop_debug["stage"] = "to_pil_image"
                crop_debug["reason"] = "failed to convert image to PIL"
                return None, crop_debug

            width, height = image.size
            crop_debug["image_size"] = [width, height]

            x1, y1, x2, y2 = [float(v) for v in bbox]
            pad_x = max(6.0, (x2 - x1) * 0.20)
            pad_y = max(6.0, (y2 - y1) * 0.20)

            x1 = max(0, int(x1 - pad_x))
            y1 = max(0, int(y1 - pad_y))
            x2 = min(width, int(x2 + pad_x))
            y2 = min(height, int(y2 + pad_y))

            crop_debug["crop_box"] = [x1, y1, x2, y2]
            if x2 <= x1 or y2 <= y1:
                crop_debug["stage"] = "crop_box"
                crop_debug["reason"] = f"invalid crop box: {[x1, y1, x2, y2]}"
                return None, crop_debug

            crop = image.crop((x1, y1, x2, y2))
            crop_debug["crop_size"] = [crop.size[0], crop.size[1]]

            crop_path = self.save_debug_crop(
                crop=crop,
                candidate=candidate,
                image_index=image_index
            )
            crop_debug["crop_path"] = crop_path
            crop_debug["stage"] = "crop"
            crop_debug["reason"] = "crop built"
            candidate["crop_path"] = crop_path

            return crop, crop_debug

        except Exception as e:
            crop_debug["stage"] = "exception"
            crop_debug["reason"] = str(e)
            return None, crop_debug

    def to_pil_image(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, memoryview):
            image = image.tobytes()

        if isinstance(image, (bytes, bytearray)):
            pil_image = self.bytes_to_pil_image(image)
            if pil_image is not None:
                return pil_image
            pil_image = self.raw_bytes_to_pil_image(image)
            if pil_image is not None:
                return pil_image
            return None

        try:
            if isinstance(image, np.ndarray):
                array = image
                if array.dtype != np.uint8:
                    array = np.clip(array, 0, 255).astype(np.uint8)

                if len(array.shape) == 2:
                    return Image.fromarray(array).convert("RGB")

                if len(array.shape) == 3:
                    if array.shape[0] in [1, 3, 4] and array.shape[2] not in [1, 3, 4]:
                        array = np.transpose(array, (1, 2, 0))
                    if array.shape[2] == 1:
                        array = array[:, :, 0]
                    return Image.fromarray(array).convert("RGB")

            return Image.fromarray(image).convert("RGB")
        except Exception:
            return None

    def bytes_to_pil_image(self, image_bytes):
        try:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return None

    def raw_bytes_to_pil_image(self, image_bytes):
        try:
            array = np.frombuffer(image_bytes, dtype=np.uint8)
            for channels in [4, 3, 1]:
                if array.size % channels != 0:
                    continue

                pixels = array.size // channels
                side = int(np.sqrt(pixels))
                if side * side != pixels:
                    continue

                if channels == 1:
                    raw_image = array.reshape((side, side))
                else:
                    raw_image = array.reshape((side, side, channels))

                return Image.fromarray(raw_image).convert("RGB")

            return None
        except Exception:
            return None

    def pil_to_png_bytes(self, image):
        try:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception:
            return None

    def save_debug_crop(self, crop, candidate, image_index):
        if not self.save_debug_crops:
            return ""

        try:
            step_num = int(candidate.get("step_num", -1))
            candidate_id = str(candidate.get("candidate_id", "unknown"))
            track_id = str(candidate.get("track_id", "no_track"))
            score = float(candidate.get("score", 0.0))
            area = float(candidate.get("area_ratio", 0.0))
            region = candidate.get("camera_region", "unknown")

            step_dir = os.path.join(
                self.debug_dir,
                f"step_{step_num:04d}"
            )
            os.makedirs(step_dir, exist_ok=True)

            filename = (
                f"{candidate_id}_"
                f"{track_id}_"
                f"img_{image_index}_"
                f"{region}_"
                f"score_{score:.3f}_"
                f"area_{area:.4f}.png"
            )
            path = os.path.join(step_dir, filename)
            crop.save(path)
            return path
        except Exception as e:
            return f"save crop failed: {e}"

    def default_crop_debug(self, candidate, stage="", reason=""):
        return {
            "ok": False,
            "stage": stage,
            "reason": reason,
            "candidate_id": candidate.get("candidate_id", None) if isinstance(candidate, dict) else None,
            "track_id": candidate.get("track_id", None) if isinstance(candidate, dict) else None,
            "step_num": candidate.get("step_num", None) if isinstance(candidate, dict) else None,
            "image_index": candidate.get("image_index", None) if isinstance(candidate, dict) else None,
            "camera_region": candidate.get("camera_region", None) if isinstance(candidate, dict) else None,
            "bbox": candidate.get("bbox", None) if isinstance(candidate, dict) else None,
            "score": candidate.get("score", 0.0) if isinstance(candidate, dict) else 0.0,
            "area_ratio": candidate.get("area_ratio", 0.0) if isinstance(candidate, dict) else 0.0,
            "rgb_count": 0,
            "image_type": "",
            "image_size": None,
            "crop_box": None,
            "crop_size": None,
            "crop_path": candidate.get("crop_path", "") if isinstance(candidate, dict) else "",
            "encoded_type": "",
            "encoded_count": 0,
            "caption_type": "",
            "caption_len": len(str(candidate.get("crop_caption", ""))) if isinstance(candidate, dict) else 0,
            "caption_raw": "",
        }

    def build_candidate_debug(self, candidates):
        debug_items = []
        if not isinstance(candidates, list):
            return debug_items

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            crop_caption = candidate.get("crop_caption", "")
            crop_debug = candidate.get("crop_debug", {})
            if not isinstance(crop_debug, dict):
                crop_debug = {}

            debug_items.append({
                "candidate_id": candidate.get("candidate_id", None),
                "track_id": candidate.get("track_id", None),
                "track_stable": candidate.get("track_stable", False),
                "track_score": candidate.get("track_score", 0.0),
                "track_hit_count": candidate.get("track_hit_count", 0),
                "stable_step_count": candidate.get("stable_step_count", 0),
                "track_position": candidate.get("track_position", None),
                "step_num": candidate.get("step_num", None),
                "image_index": candidate.get("image_index", None),
                "camera_region": candidate.get("camera_region", None),
                "score": candidate.get("score", 0.0),
                "area_ratio": candidate.get("area_ratio", 0.0),
                "bbox": candidate.get("bbox", None),
                "target_world_position": candidate.get("target_world_position", None),
                "crop_caption": crop_caption,
                "crop_caption_len": len(str(crop_caption)),
                "caption_ready": candidate.get("caption_ready", False),
                "caption_source": candidate.get("caption_source", ""),
                "verification_reject_count": candidate.get("verification_reject_count", 0),
                "verification_priority": candidate.get("verification_priority", 0.0),
                "crop_debug": crop_debug,
            })

        return debug_items

    def get_selected_or_best_crop_caption(self, candidates, selected_candidate):
        if isinstance(selected_candidate, dict):
            caption = selected_candidate.get("crop_caption", "")
            if isinstance(caption, str) and len(caption.strip()) > 0:
                return caption

        if isinstance(candidates, list):
            for candidate in candidates:
                caption = candidate.get("crop_caption", "")
                if isinstance(caption, str) and len(caption.strip()) > 0:
                    return caption

        return ""

    def system_prompt(self):
        return (
            "You are the Task-2 object verification module for an aerial object navigation system. "
            "The detector may hallucinate objects, misclassify background textures, or produce poorly localized boxes. "
            "You must verify whether one stable tracked candidate is exactly the target object described by the instruction. "
            "Only select a candidate when its crop caption, local context, bbox information, and track evidence clearly support the target. "
            "If none clearly matches, return none. "
            "Do not decide navigation actions and do not decide whether the drone should stop."
        )

    def build_prompt(
        self,
        object_name,
        object_size,
        description,
        captions4,
        candidates,
        semantic_result=None,
        tracker_info=None
    ):
        if captions4 is None:
            captions4 = ["", "", "", ""]
        captions4 = list(captions4) + ["", "", "", ""]
        captions4 = captions4[:4]

        if semantic_result is None:
            semantic_result = {}
        if tracker_info is None:
            tracker_info = {}

        candidate_lines = []
        for candidate in candidates:
            crop_debug = candidate.get("crop_debug", {})
            if not isinstance(crop_debug, dict):
                crop_debug = {}

            candidate_lines.append({
                "candidate_id": candidate.get("candidate_id", ""),
                "track_id": candidate.get("track_id", None),
                "track_stable": candidate.get("track_stable", False),
                "stable_step_count": candidate.get("stable_step_count", 0),
                "track_hit_count": candidate.get("track_hit_count", 0),
                "track_score": candidate.get("track_score", 0.0),
                "track_position": candidate.get("track_position", None),
                "phrase": candidate.get("phrase", ""),
                "detector_score": candidate.get("score", 0.0),
                "verification_priority": candidate.get("verification_priority", 0.0),
                "verification_reject_count": candidate.get("verification_reject_count", 0),
                "camera_region": candidate.get("camera_region", "unknown"),
                "image_index": candidate.get("image_index", -1),
                "bbox": candidate.get("bbox", None),
                "area_ratio": candidate.get("area_ratio", 0.0),
                "estimated_depth": candidate.get("estimated_depth", None),
                "target_world_position": candidate.get("target_world_position", None),
                "relative_region": candidate.get("relative_region", "front"),
                "crop_caption": candidate.get("crop_caption", ""),
                "caption_ready": candidate.get("caption_ready", False),
                "caption_source": candidate.get("caption_source", ""),
                "crop_path": candidate.get("crop_path", ""),
                "crop_debug_stage": crop_debug.get("stage", ""),
                "crop_debug_reason": crop_debug.get("reason", ""),
            })

        track_summary = tracker_info.get("track_summary", {})
        best_track = tracker_info.get("best_track", None)
        best_stable_track = tracker_info.get("best_stable_track", None)

        prompt = f"""
Target instruction:
- Name: {object_name}
- Size: {object_size}
- Description: {description}

Four-view scene captions:
- Front: {captions4[0]}
- Left: {captions4[1]}
- Right: {captions4[2]}
- Down: {captions4[3]}

Tracker state:
- track_summary: {json.dumps(track_summary, ensure_ascii=False)}
- best_track: {json.dumps(best_track, ensure_ascii=False)}
- best_stable_track: {json.dumps(best_stable_track, ensure_ascii=False)}

Stable candidate object list:
{json.dumps(candidate_lines, ensure_ascii=False, indent=2)}

Semantic reasoning result:
- target_visible: {semantic_result.get("target_visible", False)}
- target_confidence: {semantic_result.get("target_confidence", 0.0)}
- reason: {semantic_result.get("reason", "")}
- evidence: {semantic_result.get("evidence", [])}

Task:
This is related-object verification, not navigation control. Select exactly one candidate only if it is the target object specified by the instruction.

Selection rules:
1. A valid target must match the detailed object description, not just the detector phrase.
2. Use the crop_caption as the primary visual evidence. Do not select a candidate with an empty or vague crop_caption.
3. Use the four-view captions only as context; they cannot override a negative crop caption.
4. Use track evidence as support. Prefer candidates with stable_step_count >= 2 and track_stable=true.
5. Reject background regions such as trees, vegetation, wall, roof, road, sidewalk, shadow, hut, shed, building facade, bench, sculpture, or vague clutter unless the crop clearly shows the requested target object.
6. Reject candidates whose crop describes a similar but wrong object, wrong color, wrong size, wrong material, wrong shape, or missing key attributes.
7. If the evidence is ambiguous, return decision="none" and selected_candidate_id=null.
8. Verifying an object does not mean the drone should stop.

Output only valid JSON with this exact schema:
{{
  "decision": "select or none",
  "selected_candidate_id": "candidate id string or null",
  "selected_track_id": "track id string or null",
  "confidence": 0.0,
  "reason": "short explanation based on crop caption and context",
  "reject_reason": "short explanation if no candidate is selected, otherwise empty"
}}
"""
        return prompt.strip()

    def parse_json_result(self, text):
        try:
            raw_text = text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                raw_text = raw_text.replace("json", "", 1).strip()

            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                raw_text = raw_text[start_idx:end_idx + 1]

            parsed = json.loads(raw_text)

            if parsed.get("selected_candidate_id", None) in ["", "null", "None", "none"]:
                parsed["selected_candidate_id"] = None
            if parsed.get("selected_track_id", None) in ["", "null", "None", "none"]:
                parsed["selected_track_id"] = None
            if parsed.get("decision", None) is None:
                parsed["decision"] = "select" if parsed.get("selected_candidate_id", None) is not None else "none"

            return parsed
        except Exception:
            return {
                "decision": "none",
                "selected_candidate_id": None,
                "selected_track_id": None,
                "confidence": 0.0,
                "reason": "failed to parse verifier JSON",
                "reject_reason": text
            }

    def find_candidate(self, candidates, candidate_id):
        for candidate in candidates:
            if candidate.get("candidate_id", None) == candidate_id:
                return candidate
        return None

    def parse_size_level(self, object_size):
        if object_size is None:
            return "unknown"

        text = str(object_size).lower()
        if "small" in text or "tiny" in text:
            return "small"
        if "medium" in text or "mid" in text:
            return "medium"
        if "large" in text or "big" in text:
            return "large"
        return "unknown"

    def normalize_score(self, score):
        try:
            score = float(score)
        except Exception:
            score = 0.0
        return max(0.0, min(1.0, score))

    def default_result(
        self,
        checked=False,
        verified=False,
        confidence=0.0,
        reason="",
        reject_reason="",
        candidates=None,
        raw_response="",
        hard_reject=False
    ):
        if candidates is None:
            candidates = []

        return {
            "checked": checked,
            "verified": verified,
            "confidence": self.normalize_score(confidence),
            "same_object": verified,
            "selected_candidate_id": None,
            "selected_track_id": None,
            "selected_candidate": None,
            "reason": reason,
            "reject_reason": reject_reason,
            "hard_reject": hard_reject,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "candidate_debug": self.build_candidate_debug(candidates),
            "crop_caption": self.get_selected_or_best_crop_caption(
                candidates=candidates,
                selected_candidate=None
            ),
            "raw_response": raw_response,
            "parsed_response": {},
            "verification_threshold": self.verification_threshold,
        }