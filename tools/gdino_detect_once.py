import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


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
    from groundingdino.util.inference import load_model

    try:
        model = load_model(config_path, checkpoint_path, device=device)
    except TypeError:
        model = load_model(config_path, checkpoint_path)

    return model


def run_predict(model, image, prompt, box_threshold, text_threshold, device):
    from groundingdino.util.inference import predict

    try:
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )
    except TypeError:
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

    return boxes, logits, phrases


def detect_images(args):
    add_grounding_dino_path()

    from groundingdino.util.inference import load_image

    result = {
        "available": False,
        "prompt": args.text,
        "detections": [],
        "best_detection": None,
        "best_score": 0.0,
        "num_detections": 0,
        "error": ""
    }

    try:
        model = load_grounding_model(
            config_path=args.config,
            checkpoint_path=args.weights,
            device=args.device
        )

        result["available"] = True

        for image_index, image_path in enumerate(args.images):
            image_source, image = load_image(image_path)
            height, width = image_source.shape[:2]

            boxes, logits, phrases = run_predict(
                model=model,
                image=image,
                prompt=args.text,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=args.device
            )

            boxes = to_numpy(boxes)
            logits = to_numpy(logits)

            for idx in range(len(boxes)):
                box = boxes[idx]
                score = float(logits[idx])
                phrase = str(phrases[idx]) if idx < len(phrases) else ""

                detection = {
                    "image_index": image_index,
                    "image_path": image_path,
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

        result["num_detections"] = len(result["detections"])

        if len(result["detections"]) > 0:
            best_detection = max(
                result["detections"],
                key=lambda item: item.get("score", 0.0)
            )
            result["best_detection"] = best_detection
            result["best_score"] = float(best_detection.get("score", 0.0))

        return result

    except Exception as e:
        result["available"] = False
        result["error"] = str(e)
        return result


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Input image paths."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="GroundingDINO text prompt, e.g. 'wooden box . box .'"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output json path."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.getenv("GROUNDINGDINO_CONFIG", ""),
        help="GroundingDINO config path."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.getenv("GROUNDINGDINO_WEIGHT", ""),
        help="GroundingDINO checkpoint path."
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
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("GROUNDINGDINO_DEVICE", "cuda")
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = detect_images(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()