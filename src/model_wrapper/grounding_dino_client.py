import base64
import json
import os
import re
import urllib.request

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

        self.prompt_source = os.getenv("GROUNDINGDINO_PROMPT_SOURCE", "llm").lower()
        self.prompt_model = os.getenv("GROUNDINGDINO_PROMPT_MODEL", "gpt-4.1-mini")
        self.prompt_timeout = int(os.getenv("GROUNDINGDINO_PROMPT_TIMEOUT", "20"))
        self.max_prompt_phrases = int(os.getenv("GROUNDINGDINO_MAX_PROMPT_PHRASES", "8"))
        self.max_prompt_words = int(os.getenv("GROUNDINGDINO_MAX_PROMPT_WORDS", "8"))
        self.use_prompt_cache = os.getenv("GROUNDINGDINO_PROMPT_CACHE", "1") == "1"

        self.prompt_cache = {}
        self.has_warned = False
        self.has_prompt_warned = False

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

        prompt_info = self.build_text_prompt_info(
            object_name=object_name,
            description=description
        )
        prompt = prompt_info.get("prompt", "")

        result["prompt"] = prompt
        result["prompt_phrases"] = prompt_info.get("phrases", [])
        result["prompt_source"] = prompt_info.get("source", "unknown")
        result["prompt_error"] = prompt_info.get("error", "")

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
            result["prompt_phrases"] = response.get(
                "prompt_phrases",
                prompt_info.get("phrases", [])
            )
            result["prompt_source"] = prompt_info.get("source", "unknown")
            result["prompt_error"] = prompt_info.get("error", "")
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
        prompt_info = self.build_text_prompt_info(
            object_name=object_name,
            description=description
        )
        return prompt_info.get("prompt", "object .")

    def build_text_prompt_info(self, object_name, description=""):
        object_name = self.clean_phrase(object_name)
        description = self.clean_phrase(description)

        cache_key = self.make_prompt_cache_key(
            object_name=object_name,
            description=description
        )

        if self.use_prompt_cache and cache_key in self.prompt_cache:
            return dict(self.prompt_cache[cache_key])

        prompt_info = self.extract_detection_phrases(
            object_name=object_name,
            description=description
        )

        if self.use_prompt_cache:
            self.prompt_cache[cache_key] = dict(prompt_info)

        return prompt_info

    def extract_detection_phrases(self, object_name, description):
        llm_error = ""

        if self.prompt_source in ["llm", "auto"]:
            try:
                phrases = self.extract_phrases_with_llm(
                    object_name=object_name,
                    description=description
                )
                phrases = self.validate_prompt_phrases(phrases)
                if len(phrases) > 0:
                    return {
                        "prompt": self.format_text_prompt(phrases),
                        "phrases": phrases,
                        "source": "llm",
                        "error": ""
                    }
            except Exception as e:
                llm_error = str(e)
                if not self.has_prompt_warned:
                    print(f"[GroundingDINO] task phrase extraction failed: {e}")
                    self.has_prompt_warned = True

        phrases = self.extract_phrases_from_text(
            object_name=object_name,
            description=description
        )
        phrases = self.validate_prompt_phrases(phrases)

        if len(phrases) == 0:
            phrases = self.validate_prompt_phrases([object_name])
        if len(phrases) == 0:
            phrases = ["object"]

        return {
            "prompt": self.format_text_prompt(phrases),
            "phrases": phrases,
            "source": "text_fallback" if llm_error else "text",
            "error": llm_error
        }

    def extract_phrases_with_llm(self, object_name, description):
        api_key, base_url, model = self.get_prompt_llm_config()
        if not api_key:
            raise RuntimeError("missing OPENAI_API_KEY or DASHSCOPE_API_KEY")

        instruction = self.build_phrase_extraction_prompt(
            object_name=object_name,
            description=description
        )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract detector text prompts for open-vocabulary object detection. "
                        "Return compact JSON only. Do not output explanations."
                    )
                },
                {
                    "role": "user",
                    "content": instruction
                }
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }

        url = base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key
            },
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=self.prompt_timeout) as response:
            body = response.read().decode("utf-8")
            response_json = json.loads(body)

        content = response_json["choices"][0]["message"]["content"]
        parsed = self.parse_json_from_text(content)
        phrases = parsed.get("phrases", [])

        if not isinstance(phrases, list):
            return []

        return phrases

    def get_prompt_llm_config(self):
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = self.prompt_model

        if not api_key:
            dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
            if dashscope_key:
                api_key = dashscope_key
                base_url = os.getenv(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                if self.prompt_model == "gpt-4.1-mini":
                    model = os.getenv("DASHSCOPE_PROMPT_MODEL", "qwen-plus")

        return api_key, base_url, model

    def build_phrase_extraction_prompt(self, object_name, description):
        if not object_name:
            object_name = "object"

        if not description:
            description = object_name

        return (
            "Given a UAV object-navigation target, generate text phrases for GroundingDINO.\n"
            "The phrases will be used only to propose detection candidates; another VLM verifier "
            "will decide whether a candidate truly matches the instruction.\n\n"
            "Rules:\n"
            "1. Use the natural-language target description to infer task-relevant detection phrases.\n"
            "2. Prefer visible object-centric phrases, including the target category and discriminative visual attributes.\n"
            "3. Do not rely on a fixed category list. Infer phrases from this task only.\n"
            "4. Avoid pure scene context unless it is inseparable from the object.\n"
            "5. Do not output navigation actions, reasoning, coordinates, or stop decisions.\n"
            "6. Output 3 to 8 short phrases. Each phrase should be 1 to 8 words.\n"
            "7. Return JSON only in this exact format: {\"phrases\": [\"...\"]}\n\n"
            f"object_name: {object_name}\n"
            f"target_description: {description}\n"
        )

    def extract_phrases_from_text(self, object_name, description):
        phrases = []

        object_name = self.clean_phrase(object_name)
        description = self.clean_instruction_text(description)

        self.add_phrase(phrases, object_name)

        spaced_object_name = self.split_compact_object_name(object_name)
        self.add_phrase(phrases, spaced_object_name)

        for phrase in self.extract_description_segments(description):
            self.add_phrase(phrases, phrase)

        for phrase in self.extract_object_related_segments(
            object_name=object_name,
            description=description
        ):
            self.add_phrase(phrases, phrase)

        return phrases

    def extract_description_segments(self, description):
        description = self.clean_instruction_text(description)
        if not description:
            return []

        raw_segments = re.split(
            r"\b(?:with|wearing|holding|carrying|near|next to|beside|on|in|at|and|that|which|who)\b",
            description
        )

        segments = []
        for segment in raw_segments:
            segment = self.clean_phrase(segment)
            segment = self.remove_leading_weak_words(segment)
            segment = self.truncate_phrase(
                phrase=segment,
                max_words=self.max_prompt_words
            )

            if self.is_valid_detector_phrase(segment):
                self.add_phrase(segments, segment)

        return segments

    def extract_object_related_segments(self, object_name, description):
        object_name = self.clean_phrase(object_name)
        description = self.clean_instruction_text(description)

        if not object_name or not description:
            return []

        object_tokens = object_name.split()
        description_tokens = description.split()
        segments = []

        for i, token in enumerate(description_tokens):
            if token not in object_tokens:
                continue

            start = max(0, i - 4)
            end = min(len(description_tokens), i + 5)
            segment = " ".join(description_tokens[start:end])
            segment = self.remove_leading_weak_words(segment)
            segment = self.truncate_phrase(
                phrase=segment,
                max_words=self.max_prompt_words
            )

            if self.is_valid_detector_phrase(segment):
                self.add_phrase(segments, segment)

        return segments

    def validate_prompt_phrases(self, phrases):
        valid_phrases = []

        if not isinstance(phrases, list):
            return valid_phrases

        for phrase in phrases:
            phrase = self.clean_phrase(phrase)
            phrase = self.remove_leading_weak_words(phrase)
            if not phrase:
                continue

            phrase = self.truncate_phrase(
                phrase=phrase,
                max_words=self.max_prompt_words
            )

            if not self.is_valid_detector_phrase(phrase):
                continue

            self.add_phrase(valid_phrases, phrase)

            if len(valid_phrases) >= self.max_prompt_phrases:
                break

        return valid_phrases

    def is_valid_detector_phrase(self, phrase):
        phrase = self.clean_phrase(phrase)
        if not phrase:
            return False

        tokens = phrase.split()
        if len(tokens) == 0:
            return False

        if len(tokens) > self.max_prompt_words:
            return False

        if len(tokens) == 1 and len(tokens[0]) <= 1:
            return False

        banned_tokens = set([
            "find",
            "locate",
            "search",
            "navigate",
            "fly",
            "go",
            "move",
            "stop",
            "target",
            "instruction",
            "candidate",
            "image",
            "picture",
            "view",
            "frame",
            "visible",
            "appears",
            "probably",
            "maybe",
            "none",
        ])

        for token in tokens:
            if token in banned_tokens:
                return False

        return True

    def format_text_prompt(self, phrases):
        clean_phrases = []
        for phrase in phrases:
            phrase = self.clean_phrase(phrase)
            if phrase and phrase not in clean_phrases:
                clean_phrases.append(phrase)

        if len(clean_phrases) == 0:
            clean_phrases = ["object"]

        return " . ".join(clean_phrases) + " ."

    def parse_json_from_text(self, text):
        text = str(text).strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise ValueError("no JSON object found in LLM output")

        return json.loads(match.group(0))

    def split_compact_object_name(self, object_name):
        object_name = self.clean_phrase(object_name)
        if not object_name:
            return ""

        if " " in object_name:
            return object_name

        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", object_name)
        text = text.replace("_", " ").replace("-", " ")
        text = self.clean_phrase(text)

        return text

    def remove_leading_weak_words(self, text):
        text = self.clean_phrase(text)
        if not text:
            return ""

        weak_prefix_pattern = (
            r"^(find|locate|search|identify|fly to|go to|navigate to|"
            r"can you find|please find|the|a|an|this|that)\s+"
        )

        last_text = None
        while last_text != text:
            last_text = text
            text = re.sub(weak_prefix_pattern, "", text).strip()

        return text

    def clean_instruction_text(self, text):
        text = self.clean_phrase(text)
        if not text:
            return ""

        text = re.sub(r"\b(find|locate|search for|search|identify|fly to|go to|navigate to)\b", " ", text)
        text = re.sub(r"\b(the target|target object|target)\b", " ", text)
        text = " ".join(text.split())

        return text

    def make_prompt_cache_key(self, object_name, description):
        return json.dumps(
            {
                "object_name": object_name,
                "description": description,
                "source": self.prompt_source,
                "model": self.prompt_model,
                "max_phrases": self.max_prompt_phrases,
            },
            sort_keys=True
        )

    def add_phrase(self, phrases, phrase):
        phrase = self.clean_phrase(phrase)
        if not phrase:
            return
        if phrase not in phrases:
            phrases.append(phrase)

    def truncate_phrase(self, phrase, max_words=8):
        phrase = self.clean_phrase(phrase)
        tokens = phrase.split()
        if len(tokens) <= max_words:
            return phrase
        return " ".join(tokens[:max_words])

    def clean_phrase(self, text):
        if text is None:
            return ""

        text = str(text).strip().lower()
        text = text.replace("_", " ")
        text = text.replace("-", " ")
        text = text.replace("/", " ")
        text = text.replace(".", " ")
        text = text.replace(",", " ")
        text = text.replace(";", " ")
        text = text.replace(":", " ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.replace("[", " ")
        text = text.replace("]", " ")
        text = text.replace("{", " ")
        text = text.replace("}", " ")
        text = text.replace("\"", " ")
        text = text.replace("'", " ")
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
                f"source={result.get('prompt_source', 'unknown')}, "
                f"num=0, best_score=0.00"
            )

        return (
            f"prompt='{result.get('prompt', '')}', "
            f"source={result.get('prompt_source', 'unknown')}, "
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
            "prompt_phrases": [],
            "prompt_source": "unknown",
            "prompt_error": "",
            "detections": [],
            "best_detection": None,
            "best_score": 0.0,
            "num_detections": 0,
            "error": "",
        }