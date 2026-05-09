import argparse
import base64
import json
import os
import sys
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


MODEL = None
LOAD_IMAGE = None
PREDICT = None
SERVER_ARGS = None


def add_grounding_dino_path():
    candidates = [
        os.getenv("SVNAV_GDINO_HOME", ""),
        os.getenv("GROUNDINGDINO_HOME", ""),
        os.getenv("GDINO_HOME", ""),
    ]

    for candidate in candidates:
        if not candidate:
            continue

        path = str(Path(candidate).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def to_numpy(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass

    return np.array(value)


def strip_data_uri(value):
    if not value:
        return value

    text = str(value)
    if text.lower().startswith("data:") and "," in text:
        return text.split(",", 1)[1]

    return text


def box_cxcywh_to_xyxy(box, width, height):
    cx = float(box[0]) * width
    cy = float(box[1]) * height
    bw = float(box[2]) * width
    bh = float(box[3]) * height

    x1 = max(0.0, cx - bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    x2 = min(float(width - 1), cx + bw / 2.0)
    y2 = min(float(height - 1), cy + bh / 2.0)

    return [
        round(x1, 2),
        round(y1, 2),
        round(x2, 2),
        round(y2, 2),
    ]


def infer_default_config():
    explicit = (
        os.getenv("SVNAV_GDINO_CONFIG", "")
        or os.getenv("GROUNDINGDINO_CONFIG", "")
        or os.getenv("GDINO_CONFIG", "")
    )
    if explicit:
        return explicit

    root = (
        os.getenv("SVNAV_GDINO_HOME", "")
        or os.getenv("GROUNDINGDINO_HOME", "")
        or os.getenv("GDINO_HOME", "")
    )
    if not root:
        return ""

    candidate = Path(root).expanduser() / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    if candidate.exists():
        return str(candidate)

    return ""


def infer_default_weights():
    explicit = (
        os.getenv("SVNAV_GDINO_WEIGHT", "")
        or os.getenv("SVNAV_GDINO_CHECKPOINT", "")
        or os.getenv("GROUNDINGDINO_WEIGHT", "")
        or os.getenv("GROUNDINGDINO_CHECKPOINT", "")
        or os.getenv("GDINO_WEIGHT", "")
    )
    if explicit:
        return explicit

    root = (
        os.getenv("SVNAV_GDINO_HOME", "")
        or os.getenv("GROUNDINGDINO_HOME", "")
        or os.getenv("GDINO_HOME", "")
    )
    if not root:
        return ""

    root_path = Path(root).expanduser()
    candidates = [
        root_path / "weights" / "groundingdino_swint_ogc.pth",
        root_path / "groundingdino_swint_ogc.pth",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return ""


def load_grounding_model(config_path, checkpoint_path, device):
    global MODEL, LOAD_IMAGE, PREDICT

    add_grounding_dino_path()

    from groundingdino.util.inference import load_model, load_image, predict

    try:
        MODEL = load_model(config_path, checkpoint_path, device=device)
    except TypeError:
        MODEL = load_model(config_path, checkpoint_path)

    LOAD_IMAGE = load_image
    PREDICT = predict

    print(
        "[SVNavGDINOServer] model loaded: config={}, weights={}, device={}".format(
            config_path,
            checkpoint_path,
            device,
        ),
        flush=True,
    )


def run_predict(model, image, prompt, box_threshold, text_threshold, device):
    try:
        boxes, logits, phrases = PREDICT(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
    except TypeError:
        boxes, logits, phrases = PREDICT(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    return boxes, logits, phrases


def decode_base64_image(image_base64):
    image_base64 = strip_data_uri(image_base64)
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image


def save_temp_image(image):
    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    image.save(tmp_path)
    return tmp_path


def detect_images(payload):
    started_at = time.time()

    prompt = (
        payload.get("prompt", "")
        or payload.get("text", "")
        or payload.get("caption", "")
    )
    images = payload.get("images", [])

    box_threshold = float(payload.get("box_threshold", SERVER_ARGS.box_threshold))
    text_threshold = float(payload.get("text_threshold", SERVER_ARGS.text_threshold))
    device = payload.get("device", SERVER_ARGS.device)
    request_id = payload.get("request_id", "")

    result = {
        "ok": True,
        "available": True,
        "request_id": request_id,
        "prompt": prompt,
        "detections": [],
        "best_detection": None,
        "best_score": 0.0,
        "num_detections": 0,
        "latency_ms": None,
        "error": "",
    }

    if MODEL is None:
        result["ok"] = False
        result["available"] = False
        result["error"] = "model is not loaded"
        return result

    if not prompt:
        result["ok"] = False
        result["available"] = False
        result["error"] = "empty prompt"
        return result

    if len(images) == 0:
        result["ok"] = False
        result["available"] = False
        result["error"] = "no images"
        return result

    for image_index, image_item in enumerate(images):
        image_path = None

        try:
            image_base64 = (
                image_item.get("data", "")
                or image_item.get("image_base64", "")
                or image_item.get("b64", "")
            )
            if not image_base64:
                continue

            image = decode_base64_image(image_base64)
            image_path = save_temp_image(image)

            image_source, image_tensor = LOAD_IMAGE(image_path)
            height, width = image_source.shape[:2]

            boxes, logits, phrases = run_predict(
                model=MODEL,
                image=image_tensor,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device,
            )

            boxes = to_numpy(boxes)
            logits = to_numpy(logits)

            for idx in range(len(boxes)):
                box = boxes[idx]
                score = float(logits[idx])
                phrase = str(phrases[idx]) if idx < len(phrases) else ""

                detection = {
                    "image_index": int(image_index),
                    "frame_id": image_item.get("frame_id", ""),
                    "view_id": image_item.get("view_id", ""),
                    "step_id": image_item.get("step_id", None),
                    "score": score,
                    "phrase": phrase,
                    "bbox": box_cxcywh_to_xyxy(
                        box=box,
                        width=width,
                        height=height,
                    ),
                    "bbox_norm_cxcywh": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],
                    "image_width": int(width),
                    "image_height": int(height),
                }

                result["detections"].append(detection)

        except Exception as e:
            result["error"] = str(e)
            traceback.print_exc()

        finally:
            if image_path is not None:
                try:
                    os.remove(image_path)
                except Exception:
                    pass

    result["detections"].sort(
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    result["num_detections"] = len(result["detections"])

    if result["detections"]:
        best_detection = result["detections"][0]
        result["best_detection"] = best_detection
        result["best_score"] = float(best_detection.get("score", 0.0))

    result["latency_ms"] = round((time.time() - started_at) * 1000.0, 2)
    return result


class SVNavGroundingDINOHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.write_json(
                {
                    "ok": True,
                    "available": True,
                    "model_loaded": MODEL is not None,
                    "device": SERVER_ARGS.device if SERVER_ARGS is not None else "",
                }
            )
            return

        self.write_json(
            {
                "ok": False,
                "available": False,
                "error": "unknown endpoint",
            },
            status=404,
        )

    def do_POST(self):
        if self.path != "/detect":
            self.write_json(
                {
                    "ok": False,
                    "available": False,
                    "error": "unknown endpoint",
                    "detections": [],
                    "best_detection": None,
                    "best_score": 0.0,
                    "num_detections": 0,
                },
                status=404,
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))

            result = detect_images(payload)
            self.write_json(result)

        except Exception as e:
            traceback.print_exc()
            self.write_json(
                {
                    "ok": False,
                    "available": False,
                    "error": str(e),
                    "detections": [],
                    "best_detection": None,
                    "best_score": 0.0,
                    "num_detections": 0,
                },
                status=500,
            )

    def log_message(self, format, *args):
        return

    def write_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv(
            "SVNAV_GDINO_SERVER_HOST",
            os.getenv("GDINO_SERVER_HOST", "127.0.0.1"),
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(
            os.getenv(
                "SVNAV_GDINO_SERVER_PORT",
                os.getenv("GDINO_SERVER_PORT", "8008"),
            )
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=infer_default_config(),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=infer_default_weights(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv(
            "SVNAV_GDINO_DEVICE",
            os.getenv("GROUNDINGDINO_DEVICE", "cuda"),
        ),
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=float(
            os.getenv(
                "SVNAV_GDINO_BOX_THRESHOLD",
                os.getenv("GROUNDINGDINO_BOX_THRESHOLD", "0.25"),
            )
        ),
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=float(
            os.getenv(
                "SVNAV_GDINO_TEXT_THRESHOLD",
                os.getenv("GROUNDINGDINO_TEXT_THRESHOLD", "0.20"),
            )
        ),
    )

    return parser.parse_args()


def main():
    global SERVER_ARGS

    SERVER_ARGS = parse_args()

    if not SERVER_ARGS.config:
        raise ValueError(
            "GDINO config is empty. Set SVNAV_GDINO_CONFIG or GROUNDINGDINO_CONFIG."
        )

    if not SERVER_ARGS.weights:
        raise ValueError(
            "GDINO weights are empty. Set SVNAV_GDINO_WEIGHT or GROUNDINGDINO_WEIGHT."
        )

    load_grounding_model(
        config_path=SERVER_ARGS.config,
        checkpoint_path=SERVER_ARGS.weights,
        device=SERVER_ARGS.device,
    )

    server = ThreadingHTTPServer(
        (SERVER_ARGS.host, SERVER_ARGS.port),
        SVNavGroundingDINOHandler,
    )

    print(
        "[SVNavGDINOServer] listening on http://{}:{}".format(
            SERVER_ARGS.host,
            SERVER_ARGS.port,
        ),
        flush=True,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
