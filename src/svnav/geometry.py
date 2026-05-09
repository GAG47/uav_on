from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .types import BBox


def clamp(value: float, low: float, high: float) -> float:
    try:
        value = float(value)
    except Exception:
        return float(low)

    if value < low:
        return float(low)
    if value > high:
        return float(high)
    return float(value)


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def bbox_basic_stats(bbox: BBox) -> Dict[str, Any]:
    box = bbox.clipped()
    width = float(box.width)
    height = float(box.height)
    area_ratio = float(box.area_ratio)
    aspect_ratio = safe_aspect_ratio(width, height)
    center_x, center_y = box.center

    return {
        "x1": float(box.x1),
        "y1": float(box.y1),
        "x2": float(box.x2),
        "y2": float(box.y2),
        "width": width,
        "height": height,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "center": [float(center_x), float(center_y)],
        "image_width": int(box.image_width),
        "image_height": int(box.image_height),
    }


def safe_aspect_ratio(width: float, height: float) -> float:
    width = max(0.0, float(width))
    height = max(0.0, float(height))
    if height <= 1e-6 or width <= 1e-6:
        return 0.0
    ratio = width / height
    if ratio < 1.0:
        ratio = 1.0 / max(ratio, 1e-6)
    return float(ratio)


def bbox_border_stats(bbox: BBox, margin_px: float = 2.0) -> Dict[str, Any]:
    box = bbox.clipped()
    margin_px = float(margin_px)

    touches_left = box.x1 <= margin_px
    touches_top = box.y1 <= margin_px
    touches_right = box.x2 >= float(box.image_width) - margin_px
    touches_bottom = box.y2 >= float(box.image_height) - margin_px

    count = int(touches_left) + int(touches_top) + int(touches_right) + int(touches_bottom)

    return {
        "touches_border": count > 0,
        "border_touch_count": count,
        "touches_left": touches_left,
        "touches_top": touches_top,
        "touches_right": touches_right,
        "touches_bottom": touches_bottom,
    }


def bbox_crop_slices(
    bbox: BBox,
    image_shape: Tuple[int, int],
    padding_ratio: float = 0.0,
) -> Tuple[slice, slice]:
    """
    Return y_slice, x_slice for a bbox crop.

    image_shape is (height, width). This helper is intentionally independent
    from PIL so it can also be used for depth arrays.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    box = bbox.clipped()

    pad_x = float(box.width) * float(padding_ratio)
    pad_y = float(box.height) * float(padding_ratio)

    x1 = int(max(0, math.floor(box.x1 - pad_x)))
    y1 = int(max(0, math.floor(box.y1 - pad_y)))
    x2 = int(min(width, math.ceil(box.x2 + pad_x)))
    y2 = int(min(height, math.ceil(box.y2 + pad_y)))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return slice(y1, y2), slice(x1, x2)


def depth_array_from_any(depth: Any) -> Optional[np.ndarray]:
    if depth is None:
        return None

    try:
        arr = np.asarray(depth)
    except Exception:
        return None

    if arr.size == 0:
        return None

    arr = np.squeeze(arr)

    if arr.ndim != 2:
        return None

    try:
        arr = arr.astype(np.float32)
    except Exception:
        return None

    return arr


def depth_stats_in_bbox(
    depth: Any,
    bbox: BBox,
    min_depth: float = 0.05,
    max_depth: float = 200.0,
) -> Dict[str, Any]:
    arr = depth_array_from_any(depth)

    if arr is None:
        return {
            "depth_available": False,
            "depth_valid": False,
            "depth_valid_ratio": 0.0,
            "depth_median": None,
            "depth_mean": None,
            "depth_min": None,
            "depth_max": None,
            "depth_sample_count": 0,
            "depth_valid_count": 0,
        }

    y_slice, x_slice = bbox_crop_slices(
        bbox=bbox,
        image_shape=(arr.shape[0], arr.shape[1]),
        padding_ratio=0.0,
    )
    crop = arr[y_slice, x_slice]

    if crop.size == 0:
        return {
            "depth_available": True,
            "depth_valid": False,
            "depth_valid_ratio": 0.0,
            "depth_median": None,
            "depth_mean": None,
            "depth_min": None,
            "depth_max": None,
            "depth_sample_count": 0,
            "depth_valid_count": 0,
        }

    valid_mask = np.isfinite(crop)
    valid_mask &= crop > float(min_depth)
    valid_mask &= crop < float(max_depth)

    valid_values = crop[valid_mask]
    valid_count = int(valid_values.size)
    sample_count = int(crop.size)
    valid_ratio = float(valid_count) / float(max(1, sample_count))

    if valid_count <= 0:
        return {
            "depth_available": True,
            "depth_valid": False,
            "depth_valid_ratio": valid_ratio,
            "depth_median": None,
            "depth_mean": None,
            "depth_min": None,
            "depth_max": None,
            "depth_sample_count": sample_count,
            "depth_valid_count": valid_count,
        }

    return {
        "depth_available": True,
        "depth_valid": True,
        "depth_valid_ratio": valid_ratio,
        "depth_median": float(np.median(valid_values)),
        "depth_mean": float(np.mean(valid_values)),
        "depth_min": float(np.min(valid_values)),
        "depth_max": float(np.max(valid_values)),
        "depth_sample_count": sample_count,
        "depth_valid_count": valid_count,
    }


def normalize_quality_score(score: float) -> float:
    return clamp01(score)
