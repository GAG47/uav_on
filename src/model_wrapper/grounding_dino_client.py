import base64
import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image


class GroundingDINOClient:
    def __init__(
        self,
        enabled=None,
        server_url=None,
        box_threshold=0.25,
        text_threshold=0.20,
        max_images_per_step=4,
        timeout=20,
    ):
        if enabled is None:
            enabled = os.getenv("USE_GROUNDING_DINO", "0") == "1"

        self.enabled = enabled
        self.server_url = server_url or os.getenv(
            "GDINO_SERVER_URL",
            "http://127.0.0.1:8008/detect"
        )

        self.box_threshold = float(os.getenv("GROUNDINGDINO_BOX_THRESHOLD", box_threshold))
        self.text_threshold = float(os.getenv("GROUNDINGDINO_TEXT_THRESHOLD", text_threshold))
        self.max_images_per_step = int(os.getenv("GROUNDINGDINO_MAX_IMAGES", max_images_per_step))
        self.timeout = int(os.getenv("GROUNDINGDINO_TIMEOUT", timeout))

        self.has_warned = False


    def detect_episode(self, rgb_images, object_name, description="", episode_index=0, step_num=0):
        result = self.default_result(
            object_name=object_name,
            description=description,
            episode_index=episode_index,
            step_num=step_num
        )

        if not self.enabled:
            result["error"] = "GroundingDINO is disabled"
            return result

        if rgb_images is None or len(rgb_images) == 0:
            result["error"] = "no rgb images"
            return result

        prompt = self.build_text_prompt(object_name, description)
        result["prompt"] = prompt

        try:
            encoded_images = self.encode_rgb_images(rgb_images)
            if len(encoded_images) == 0:
                result["error"] = "failed to encode rgb images"
                return result

            payload = {
                "text": prompt,
                "images": encoded_images,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold
            }

            response = self.post_json(
                url=self.server_url,
                payload=payload,
                timeout=self.timeout
            )

            result.update(response)
            result["episode_index"] = episode_index
            result["step_num"] = step_num
            result["object_name"] = object_name
            result["description"] = description
            result["prompt"] = response.get("prompt", prompt)
            result["available"] = bool(response.get("available", False))

            return result

        except Exception as e:
            result["error"] = str(e)

            if not self.has_warned:
                print(f"[GroundingDINO] server request failed: {e}")
                self.has_warned = True

            return result


    def post_json(self, url, payload, timeout):
        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=data,
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)


    def encode_rgb_images(self, rgb_images):
        encoded_images = []

        for image_index, image_data in enumerate(rgb_images[:self.max_images_per_step]):
            try:
                image = self.decode_image(image_data)
                buffer = BytesIO()
                image.save(buffer, format="PNG")

                image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                encoded_images.append({
                    "image_index": image_index,
                    "data": image_base64
                })

            except Exception as e:
                if not self.has_warned:
                    print(f"[GroundingDINO] failed to encode rgb image: {e}")
                    self.has_warned = True

        return encoded_images


    def decode_image(self, image_data):
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")

        if isinstance(image_data, bytes):
            return Image.open(BytesIO(image_data)).convert("RGB")

        if isinstance(image_data, bytearray):
            return Image.open(BytesIO(bytes(image_data))).convert("RGB")

        if isinstance(image_data, str):
            return Image.open(image_data).convert("RGB")

        if isinstance(image_data, np.ndarray):
            array = image_data
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)

            if array.ndim == 2:
                array = np.stack([array, array, array], axis=-1)

            return Image.fromarray(array).convert("RGB")

        if hasattr(image_data, "image_data_uint8"):
            raw_data = image_data.image_data_uint8

            if isinstance(raw_data, bytes):
                try:
                    return Image.open(BytesIO(raw_data)).convert("RGB")
                except Exception:
                    pass

            array = np.frombuffer(raw_data, dtype=np.uint8)

            height = getattr(image_data, "height", 0)
            width = getattr(image_data, "width", 0)

            if height > 0 and width > 0:
                array = array.reshape(height, width, 3)
                return Image.fromarray(array).convert("RGB")

        raise TypeError(f"Unsupported image data type: {type(image_data)}")


    def build_text_prompt(self, object_name, description=""):
        phrases = []

        object_name = self.clean_phrase(object_name)
        description = self.clean_phrase(description)

        if object_name:
            phrases.append(object_name)

            tokens = object_name.split()
            if len(tokens) >= 2:
                phrases.append(tokens[-1])

        use_description = os.getenv("GROUNDINGDINO_USE_DESCRIPTION", "0") == "1"
        if use_description and description:
            if len(description) > 160:
                description = description[:160]
            phrases.append(description)

        clean_phrases = []
        for phrase in phrases:
            if phrase and phrase not in clean_phrases:
                clean_phrases.append(phrase)

        if len(clean_phrases) == 0:
            clean_phrases = ["object"]

        prompt = " . ".join(clean_phrases) + " ."
        return prompt


    def clean_phrase(self, text):
        if text is None:
            return ""

        text = str(text).strip().lower()
        text = text.replace("_", " ")
        text = text.replace("-", " ")
        text = text.replace("/", " ")
        text = " ".join(text.split())

        return text


    def summarize_result(self, result):
        if result is None:
            return "no result"

        if not result.get("available", False):
            return f"not available, error={result.get('error', '')}"

        best = result.get("best_detection", None)
        if best is None:
            return (
                f"prompt='{result.get('prompt', '')}', "
                f"num=0, best_score=0.00"
            )

        return (
            f"prompt='{result.get('prompt', '')}', "
            f"num={result.get('num_detections', 0)}, "
            f"best_score={result.get('best_score', 0.0):.3f}, "
            f"image={best.get('image_index', -1)}, "
            f"phrase='{best.get('phrase', '')}', "
            f"bbox={best.get('bbox', None)}"
        )


    def default_result(self, object_name="", description="", episode_index=0, step_num=0):
        return {
            "available": False,
            "episode_index": episode_index,
            "step_num": step_num,
            "object_name": object_name,
            "description": description,
            "prompt": "",
            "detections": [],
            "best_detection": None,
            "best_score": 0.0,
            "num_detections": 0,
            "error": "",
        }