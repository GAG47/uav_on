import asyncio
import json
from PIL import Image


class TargetVerifier:
    def __init__(
        self,
        client=None,
        model="gpt-4.1-mini",
        verification_threshold=0.58,
        max_candidates=6,
        hard_reject_large_area=0.85,
        small_object_large_area=0.18,
    ):
        self.client = client
        self.model = model
        self.verification_threshold = verification_threshold
        self.max_candidates = max_candidates
        self.hard_reject_large_area = hard_reject_large_area
        self.small_object_large_area = small_object_large_area

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

        candidates = tracker_info.get("verification_candidates", [])
        candidates = self.filter_candidates(
            candidates=candidates,
            object_size=object_size
        )

        if len(candidates) == 0:
            return self.default_result(
                checked=True,
                verified=False,
                reason="no valid task-aware object candidates"
            )

        candidates = candidates[:self.max_candidates]

        candidates = self.attach_crop_captions(
            candidates=candidates,
            rgb_images=rgb_images,
            encode_image_fn=encode_image_fn,
            generate_caption_fn=generate_caption_fn
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
            semantic_result=semantic_result
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
            confidence = self.normalize_score(parsed.get("confidence", 0.0))
            reason = parsed.get("reason", "")
            reject_reason = parsed.get("reject_reason", "")

            valid_ids = set([candidate["candidate_id"] for candidate in candidates])

            verified = (
                selected_candidate_id in valid_ids
                and confidence >= self.verification_threshold
            )

            if not verified:
                selected_candidate_id = None

            selected_candidate = None
            if selected_candidate_id is not None:
                selected_candidate = self.find_candidate(
                    candidates=candidates,
                    candidate_id=selected_candidate_id
                )

            return {
                "checked": True,
                "verified": verified,
                "confidence": confidence,
                "same_object": verified,
                "selected_candidate_id": selected_candidate_id,
                "selected_candidate": selected_candidate,
                "reason": reason,
                "reject_reason": reject_reason,
                "hard_reject": False,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "raw_response": text,
                "verification_threshold": self.verification_threshold,
            }

        except Exception as e:
            return self.default_result(
                checked=True,
                verified=False,
                reason=f"target verification failed: {e}",
                candidates=candidates
            )

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

            area_ratio = float(candidate.get("area_ratio", 0.0))

            if area_ratio >= self.hard_reject_large_area:
                continue

            if size_level == "small" and area_ratio >= self.small_object_large_area:
                continue

            valid_candidates.append(candidate)

        valid_candidates.sort(
            key=lambda item: item.get("candidate_quality", 0.0),
            reverse=True
        )

        return valid_candidates

    def attach_crop_captions(
        self,
        candidates,
        rgb_images,
        encode_image_fn=None,
        generate_caption_fn=None
    ):
        if encode_image_fn is None or generate_caption_fn is None:
            return candidates

        crops = []
        crop_indices = []

        for idx, candidate in enumerate(candidates):
            crop = self.build_candidate_crop(
                rgb_images=rgb_images,
                candidate=candidate
            )

            if crop is None:
                continue

            crops.append(crop)
            crop_indices.append(idx)

        if len(crops) == 0:
            return candidates

        try:
            crop_b64 = encode_image_fn(crops)
            crop_captions = generate_caption_fn(crop_b64)

            if not isinstance(crop_captions, list):
                return candidates

            for idx, caption in zip(crop_indices, crop_captions):
                candidates[idx]["crop_caption"] = str(caption)

        except Exception as e:
            for idx in crop_indices:
                candidates[idx]["crop_caption"] = f"crop caption failed: {e}"

        return candidates

    def build_candidate_crop(self, rgb_images, candidate):
        try:
            image_index = int(candidate.get("image_index", -1))
            bbox = candidate.get("bbox", None)

            if image_index < 0 or image_index >= len(rgb_images):
                return None

            if bbox is None or len(bbox) != 4:
                return None

            image = self.to_pil_image(rgb_images[image_index])

            if image is None:
                return None

            width, height = image.size
            x1, y1, x2, y2 = [float(v) for v in bbox]

            pad_x = max(6.0, (x2 - x1) * 0.20)
            pad_y = max(6.0, (y2 - y1) * 0.20)

            x1 = max(0, int(x1 - pad_x))
            y1 = max(0, int(y1 - pad_y))
            x2 = min(width, int(x2 + pad_x))
            y2 = min(height, int(y2 + pad_y))

            if x2 <= x1 or y2 <= y1:
                return None

            crop = image.crop((x1, y1, x2, y2))
            return crop

        except Exception:
            return None

    def to_pil_image(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        try:
            return Image.fromarray(image).convert("RGB")
        except Exception:
            return None

    def system_prompt(self):
        return (
            "You are the object verification module for an aerial object navigation system. "
            "The detector may hallucinate objects or misclassify background textures. "
            "You will receive several detected object candidates with crop captions, "
            "camera regions, bounding boxes, detector scores, and estimated 3D positions. "
            "Your task is to select exactly one candidate only if it truly matches the target instruction. "
            "If none of the candidates clearly match the target object, return null. "
            "Do not decide whether the drone should stop. Only verify the target object."
        )

    def build_prompt(
        self,
        object_name,
        object_size,
        description,
        captions4,
        candidates,
        semantic_result=None
    ):
        if captions4 is None:
            captions4 = ["", "", "", ""]

        captions4 = list(captions4) + ["", "", "", ""]
        captions4 = captions4[:4]

        if semantic_result is None:
            semantic_result = {}

        candidate_lines = []

        for candidate in candidates:
            candidate_lines.append(
                {
                    "candidate_id": candidate.get("candidate_id", ""),
                    "phrase": candidate.get("phrase", ""),
                    "detector_score": candidate.get("score", 0.0),
                    "camera_region": candidate.get("camera_region", "unknown"),
                    "image_index": candidate.get("image_index", -1),
                    "bbox": candidate.get("bbox", None),
                    "area_ratio": candidate.get("area_ratio", 0.0),
                    "estimated_depth": candidate.get("estimated_depth", None),
                    "target_world_position": candidate.get("target_world_position", None),
                    "relative_region": candidate.get("relative_region", "front"),
                    "crop_caption": candidate.get("crop_caption", ""),
                }
            )

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

Candidate object list:
{json.dumps(candidate_lines, ensure_ascii=False, indent=2)}

Semantic reasoning result:
- target_visible: {semantic_result.get("target_visible", False)}
- target_confidence: {semantic_result.get("target_confidence", 0.0)}
- reason: {semantic_result.get("reason", "")}
- evidence: {semantic_result.get("evidence", [])}

Selection rules:
1. Select a candidate only if the crop caption and context support that it is the exact target object.
2. Reject candidates that look like ground, wall, roof, shadow, clutter, or a vague background region.
3. Reject candidates whose bbox is too large for the target size or poorly localized.
4. Side-view candidates may be selected if they clearly show the target object, but this only verifies the object; it does not mean the drone should stop.
5. If no candidate clearly matches the instruction, return selected_candidate_id as null.
6. Be conservative, but do not reject a clear candidate merely because the global scene caption is incomplete.

Output only valid JSON:
{{
  "selected_candidate_id": "candidate id string or null",
  "confidence": a number from 0 to 1,
  "reason": "short explanation",
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

            return parsed

        except Exception:
            return {
                "selected_candidate_id": None,
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
            "selected_candidate": None,
            "reason": reason,
            "reject_reason": reject_reason,
            "hard_reject": hard_reject,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "raw_response": raw_response,
            "verification_threshold": self.verification_threshold,
        }