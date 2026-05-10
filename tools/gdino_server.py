import argparse
import base64
import json
import os
import sys
import tempfile
import traceback
import gc
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image


MODEL = None
LOAD_IMAGE = None
PREDICT = None
SERVER_ARGS = None
REQUEST_COUNT = 0
PREDICT_LOCK = threading.Lock()


def add_grounding_dino_path():
    grounding_dino_home = os.getenv("GROUNDINGDINO_HOME", "")

    if grounding_dino_home:
        grounding_dino_home = str(Path(grounding_dino_home).expanduser().resolve())
        if grounding_dino_home not in sys.path:
            sys.path.insert(0, grounding_dino_home)


def to_numpy(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass

    return np.array(value)


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
        round(y2, 2)
    ]


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
        "[GroundingDINO Server] "
        f"model loaded, config={config_path}, weights={checkpoint_path}, device={device}",
        flush=True
    )



def cleanup_runtime(reason="", force=False):
    """
    Release temporary Python objects and CUDA cached memory after GDINO inference.

    This does not unload the GroundingDINO model. It only clears objects that are
    safe to release between requests, so GROUNDINGDINO_MAX_IMAGES can remain high.
    """
    if not force and os.getenv("GDINO_CLEANUP_EACH_REQUEST", "1") != "1":
        return

    try:
        gc.collect()
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available() and os.getenv("GDINO_CUDA_EMPTY_CACHE", "1") == "1":
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception as e:
        if os.getenv("GDINO_VERBOSE_CLEANUP", "0") == "1":
            print(f"[GroundingDINO Server] cleanup skipped: {e}", flush=True)

    if os.getenv("GDINO_VERBOSE_CLEANUP", "0") == "1":
        print(f"[GroundingDINO Server] cleanup done: {reason}", flush=True)

def run_predict(model, image, prompt, box_threshold, text_threshold, device):
    """
    Run GroundingDINO prediction.

    The lock avoids overlapping CUDA inference inside ThreadingHTTPServer if
    multiple requests arrive at the same time.
    """
    with PREDICT_LOCK:
        try:
            boxes, logits, phrases = PREDICT(
                model=model,
                image=image,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device
            )
        except TypeError:
            boxes, logits, phrases = PREDICT(
                model=model,
                image=image,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )

    return boxes, logits, phrases
def decode_base64_image(image_base64):
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
    global REQUEST_COUNT

    REQUEST_COUNT += 1

    prompt = payload.get("text", "")
    images = payload.get("images", [])
    box_threshold = float(payload.get("box_threshold", SERVER_ARGS.box_threshold))
    text_threshold = float(payload.get("text_threshold", SERVER_ARGS.text_threshold))
    device = payload.get("device", SERVER_ARGS.device)

    result = {
        "available": True,
        "prompt": prompt,
        "detections": [],
        "best_detection": None,
        "best_score": 0.0,
        "num_detections": 0,
        "error": ""
    }

    if MODEL is None:
        result["available"] = False
        result["error"] = "model is not loaded"
        return result

    if len(images) == 0:
        result["available"] = False
        result["error"] = "no images"
        return result

    for image_index, image_item in enumerate(images):
        image_path = None
        image = None
        image_source = None
        image_tensor = None
        boxes = None
        logits = None
        phrases = None

        try:
            image_base64 = image_item.get("data", "")
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
                device=device
            )

            boxes = to_numpy(boxes)
            logits = to_numpy(logits)

            for idx in range(len(boxes)):
                box = boxes[idx]
                score = float(logits[idx])
                phrase = str(phrases[idx]) if idx < len(phrases) else ""

                detection = {
                    "image_index": image_index,
                    "score": score,
                    "phrase": phrase,
                    "bbox": box_cxcywh_to_xyxy(
                        box=box,
                        width=width,
                        height=height
                    ),
                    "bbox_norm_cxcywh": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3])
                    ],
                    "image_width": int(width),
                    "image_height": int(height)
                }

                result["detections"].append(detection)

        except Exception as e:
            result["error"] = str(e)
            traceback.print_exc()

        finally:
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass

            if image_path is not None:
                try:
                    os.remove(image_path)
                except Exception:
                    pass

            image = None
            image_source = None
            image_tensor = None
            boxes = None
            logits = None
            phrases = None

            if os.getenv("GDINO_CLEANUP_EACH_IMAGE", "0") == "1":
                cleanup_runtime(
                    reason=f"request={REQUEST_COUNT}, image={image_index}",
                    force=True
                )

    result["num_detections"] = len(result["detections"])

    if len(result["detections"]) > 0:
        best_detection = max(
            result["detections"],
            key=lambda item: item.get("score", 0.0)
        )
        result["best_detection"] = best_detection
        result["best_score"] = float(best_detection.get("score", 0.0))

    cleanup_runtime(
        reason=f"request={REQUEST_COUNT}, images={len(images)}",
        force=True
    )

    return result
class GroundingDINOHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.write_json({
                "ok": True,
                "model_loaded": MODEL is not None
            })
            return

        self.write_json({
            "ok": False,
            "error": "unknown endpoint"
        }, status=404)


    def do_POST(self):
        if self.path != "/detect":
            self.write_json({
                "available": False,
                "error": "unknown endpoint"
            }, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))

            result = detect_images(payload)
            self.write_json(result)

        except Exception as e:
            traceback.print_exc()
            self.write_json({
                "available": False,
                "error": str(e),
                "detections": [],
                "best_detection": None,
                "best_score": 0.0,
                "num_detections": 0
            }, status=500)

        finally:
            cleanup_runtime(reason="do_POST finally", force=True)


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
        default=os.getenv("GDINO_SERVER_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GDINO_SERVER_PORT", "8008"))
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.getenv("GROUNDINGDINO_CONFIG", "")
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.getenv("GROUNDINGDINO_WEIGHT", "")
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("GROUNDINGDINO_DEVICE", "cuda")
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=float(os.getenv("GROUNDINGDINO_BOX_THRESHOLD", "0.25"))
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=float(os.getenv("GROUNDINGDINO_TEXT_THRESHOLD", "0.20"))
    )

    return parser.parse_args()


def main():
    global SERVER_ARGS

    SERVER_ARGS = parse_args()

    if not SERVER_ARGS.config:
        raise ValueError("GROUNDINGDINO_CONFIG is empty")

    if not SERVER_ARGS.weights:
        raise ValueError("GROUNDINGDINO_WEIGHT is empty")

    load_grounding_model(
        config_path=SERVER_ARGS.config,
        checkpoint_path=SERVER_ARGS.weights,
        device=SERVER_ARGS.device
    )

    server = ThreadingHTTPServer(
        (SERVER_ARGS.host, SERVER_ARGS.port),
        GroundingDINOHandler
    )

    print(
        "[GroundingDINO Server] "
        f"listening on http://{SERVER_ARGS.host}:{SERVER_ARGS.port}",
        flush=True
    )

    server.serve_forever()


if __name__ == "__main__":
    main()