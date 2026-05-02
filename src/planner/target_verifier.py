import asyncio
import json
from PIL import Image


class TargetVerifier:
    def __init__(
        self,
        client=None,
        model="gpt-4.1-mini",
        verification_threshold=0.60,
        hard_reject_large_area=0.80,
        small_object_large_area=0.45,
    ):
        self.client = client
        self.model = model
        self.verification_threshold = verification_threshold
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
        observation = tracker_info.get("observation", {})
        planner_target = tracker_info.get("planner_target", {})

        default_info = self.default_result(
            checked=False,
            verified=False,
            confidence=0.0,
            reason="verification not triggered"
        )

        if not tracker_info.get("confirmed", False):
            return default_info

        if not isinstance(observation, dict) or not observation.get("valid", False):
            return self.default_result(
                checked=True,
                verified=False,
                confidence=0.0,
                reason="invalid detection observation",
                hard_reject=True
            )

        hard_reject_info = self.hard_reject(
            object_name=object_name,
            object_size=object_size,
            observation=observation
        )
        if hard_reject_info["hard_reject"]:
            return hard_reject_info

        crop_caption = self.generate_candidate_crop_caption(
            rgb_images=rgb_images,
            observation=observation,
            encode_image_fn=encode_image_fn,
            generate_caption_fn=generate_caption_fn
        )

        if self.client is None:
            return self.default_result(
                checked=True,
                verified=False,
                confidence=0.0,
                reason="target verifier client is None",
                crop_caption=crop_caption
            )

        prompt = self.build_prompt(
            object_name=object_name,
            object_size=object_size,
            description=description,
            captions4=captions4,
            observation=observation,
            planner_target=planner_target,
            crop_caption=crop_caption,
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

            verified = bool(parsed.get("verified", False))
            confidence = self.normalize_score(parsed.get("confidence", 0.0))
            same_object = bool(parsed.get("same_object", False))
            reject_reason = parsed.get("reject_reason", "")
            reason = parsed.get("reason", "")

            if confidence < self.verification_threshold:
                verified = False

            if not same_object:
                verified = False

            result = {
                "checked": True,
                "verified": verified,
                "confidence": confidence,
                "same_object": same_object,
                "reason": reason,
                "reject_reason": reject_reason,
                "crop_caption": crop_caption,
                "raw_response": text,
                "hard_reject": False,
                "verification_threshold": self.verification_threshold,
            }

            return result

        except Exception as e:
            return self.default_result(
                checked=True,
                verified=False,
                confidence=0.0,
                reason=f"target verification failed: {e}",
                crop_caption=crop_caption
            )

    def hard_reject(self, object_name, object_size, observation):
        camera_region = observation.get("camera_region", "unknown")
        area_ratio = float(observation.get("area_ratio", 0.0))
        full_frame_like_box = bool(observation.get("full_frame_like_box", False))
        abnormal_large_box = bool(observation.get("abnormal_large_box", False))

        if camera_region == "down":
            return self.default_result(
                checked=True,
                verified=False,
                confidence=0.0,
                reason="downward view detection is not accepted as verified target",
                hard_reject=True
            )

        if full_frame_like_box:
            return self.default_result(
                checked=True,
                verified=False,
                confidence=0.0,
                reason="full-frame-like detection is rejected",
                hard_reject=True
            )

        if area_ratio >= self.hard_reject_large_area:
            return self.default_result(
                checked=True,
                verified=False,
                confidence=0.0,
                reason="bbox area is too large and likely not a localized object",
                hard_reject=True
            )

        size_level = self.parse_size_level(object_size)
        if size_level == "small" and area_ratio >= self.small_object_large_area:
            return self.default_result(
                checked=True,
                verified=False,
                confidence=0.0,
                reason="small target has an unusually large bbox",
                hard_reject=True
            )

        return self.default_result(
            checked=False,
            verified=False,
            confidence=0.0,
            reason="not hard rejected",
            hard_reject=False
        )

    def generate_candidate_crop_caption(
        self,
        rgb_images,
        observation,
        encode_image_fn=None,
        generate_caption_fn=None
    ):
        if encode_image_fn is None or generate_caption_fn is None:
            return ""

        try:
            image_index = int(observation.get("image_index", -1))
            bbox = observation.get("bbox", None)

            if image_index < 0 or image_index >= len(rgb_images):
                return ""

            if bbox is None or len(bbox) != 4:
                return ""

            image = rgb_images[image_index]
            image = self.to_pil_image(image)

            if image is None:
                return ""

            width, height = image.size
            x1, y1, x2, y2 = [float(v) for v in bbox]

            pad_x = max(4.0, (x2 - x1) * 0.10)
            pad_y = max(4.0, (y2 - y1) * 0.10)

            x1 = max(0, int(x1 - pad_x))
            y1 = max(0, int(y1 - pad_y))
            x2 = min(width, int(x2 + pad_x))
            y2 = min(height, int(y2 + pad_y))

            if x2 <= x1 or y2 <= y1:
                return ""

            crop = image.crop((x1, y1, x2, y2))
            crop_b64 = encode_image_fn([crop])
            captions = generate_caption_fn(crop_b64)

            if isinstance(captions, list) and len(captions) > 0:
                return str(captions[0])

            return ""

        except Exception as e:
            return f"crop caption failed: {e}"

    def to_pil_image(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        try:
            return Image.fromarray(image).convert("RGB")
        except Exception:
            return None

    def system_prompt(self):
        return (
            "You are an object verification module for UAV object navigation. "
            "A detector has proposed a candidate object, but the detector may hallucinate, "
            "misclassify background textures, or produce overly large boxes. "
            "Your job is to decide whether the candidate truly matches the target instruction. "
            "Be conservative: if the evidence is weak, ambiguous, or only the detector phrase matches, reject it."
        )

    def build_prompt(
        self,
        object_name,
        object_size,
        description,
        captions4,
        observation,
        planner_target,
        crop_caption,
        semantic_result=None
    ):
        if captions4 is None:
            captions4 = ["", "", "", ""]

        captions4 = list(captions4) + ["", "", "", ""]
        captions4 = captions4[:4]

        if semantic_result is None:
            semantic_result = {}

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

Detector candidate:
- phrase: {observation.get("phrase", "")}
- detector_score: {observation.get("score", 0.0)}
- camera_region: {observation.get("camera_region", "unknown")}
- image_index: {observation.get("image_index", -1)}
- bbox: {observation.get("bbox", None)}
- area_ratio: {observation.get("area_ratio", 0.0)}
- center_offset_x: {observation.get("center_offset_x", 0.0)}
- estimated_depth: {observation.get("estimated_depth", None)}
- relative_region: {observation.get("relative_region", "front")}
- target_world_position: {observation.get("target_world_position", None)}

Candidate crop caption:
- {crop_caption}

Planner target:
- {planner_target}

Current semantic reasoning result:
- target_visible: {semantic_result.get("target_visible", False)}
- target_confidence: {semantic_result.get("target_confidence", 0.0)}
- reason: {semantic_result.get("reason", "")}
- evidence: {semantic_result.get("evidence", [])}

Verification rules:
1. Verify only if the candidate appears to be the exact target described by the instruction.
2. Reject if the detector phrase matches but the scene/crop caption does not visually support the target.
3. Reject if the bbox seems to cover a vague region, ground, wall, shadow, clutter, or background texture.
4. Reject if the target is small but the bbox is very large or poorly localized.
5. For side-view candidates, verification can be true, but it only means "navigate toward it"; it does not mean immediate stop.
6. Be conservative. Ambiguous evidence should be rejected.

Output only valid JSON:
{{
  "same_object": true or false,
  "verified": true or false,
  "confidence": a number from 0 to 1,
  "reason": "short explanation",
  "reject_reason": "short explanation if rejected, otherwise empty"
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

            return json.loads(raw_text)

        except Exception:
            return {
                "same_object": False,
                "verified": False,
                "confidence": 0.0,
                "reason": "failed to parse verifier JSON",
                "reject_reason": text
            }

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
        crop_caption="",
        raw_response="",
        hard_reject=False
    ):
        return {
            "checked": checked,
            "verified": verified,
            "confidence": self.normalize_score(confidence),
            "same_object": verified,
            "reason": reason,
            "reject_reason": reject_reason,
            "crop_caption": crop_caption,
            "raw_response": raw_response,
            "hard_reject": hard_reject,
            "verification_threshold": self.verification_threshold,
        }