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


# ----------------------------------------------------------------------
# Candidate 3D geometry helpers
# ----------------------------------------------------------------------

def robust_depth_in_bbox(
    depth: Any,
    bbox: BBox,
    center_patch_ratio: float = 0.15,
    min_depth: float = 0.05,
    max_depth: float = 200.0,
) -> Dict[str, Any]:
    """
    Robustly estimate depth for a detection bbox.

    Prefer a small center patch. Fall back to whole-bbox median. This is used
    for candidate-level rough 3D estimation, not final target localization.
    """
    arr = depth_array_from_any(depth)
    base = depth_stats_in_bbox(
        depth=depth,
        bbox=bbox,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    if arr is None:
        base.update(
            {
                "center_patch_available": False,
                "center_patch_valid": False,
                "center_patch_depth": None,
                "bbox_median_depth": None,
                "used_depth": None,
                "used_depth_source": "no_depth",
                "depth_std": None,
                "depth_consistency": 0.0,
            }
        )
        return base

    box = bbox.clipped()
    cx, cy = box.center

    patch_w = max(3.0, float(box.width) * float(center_patch_ratio))
    patch_h = max(3.0, float(box.height) * float(center_patch_ratio))

    patch_bbox = BBox(
        x1=cx - patch_w / 2.0,
        y1=cy - patch_h / 2.0,
        x2=cx + patch_w / 2.0,
        y2=cy + patch_h / 2.0,
        image_width=box.image_width,
        image_height=box.image_height,
        normalized=False,
    )

    center_stats = depth_stats_in_bbox(
        depth=arr,
        bbox=patch_bbox,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    bbox_median_depth = base.get("depth_median")
    center_patch_depth = center_stats.get("depth_median")

    used_depth = None
    used_depth_source = "no_valid_depth"

    if center_stats.get("depth_valid", False):
        used_depth = center_patch_depth
        used_depth_source = "center_patch_median"
    elif base.get("depth_valid", False):
        used_depth = bbox_median_depth
        used_depth_source = "bbox_median_depth"

    depth_std = None
    depth_consistency = 0.0

    try:
        y_slice, x_slice = bbox_crop_slices(
            bbox=box,
            image_shape=(arr.shape[0], arr.shape[1]),
            padding_ratio=0.0,
        )
        crop = arr[y_slice, x_slice]
        valid = crop[np.isfinite(crop)]
        valid = valid[(valid > float(min_depth)) & (valid < float(max_depth))]
        if valid.size > 1:
            depth_std = float(np.std(valid))
            ref_depth = float(np.median(valid))
            depth_consistency = 1.0 - min(1.0, depth_std / max(1e-6, ref_depth))
        elif valid.size == 1:
            depth_std = 0.0
            depth_consistency = 1.0
    except Exception:
        depth_std = None
        depth_consistency = 0.0

    base.update(
        {
            "center_patch_available": True,
            "center_patch_valid": bool(center_stats.get("depth_valid", False)),
            "center_patch_depth": center_patch_depth,
            "center_patch_valid_ratio": center_stats.get("depth_valid_ratio", 0.0),
            "bbox_median_depth": bbox_median_depth,
            "used_depth": used_depth,
            "used_depth_source": used_depth_source,
            "depth_std": depth_std,
            "depth_consistency": float(max(0.0, min(1.0, depth_consistency))),
        }
    )
    return base


def pixel_to_camera_ray(
    u: float,
    v: float,
    image_width: int,
    image_height: int,
    horizontal_fov_deg: float = 90.0,
    vertical_fov_deg: float = 90.0,
) -> Dict[str, Any]:
    """
    Convert image pixel to a simple pinhole camera ray.

    Camera convention used by SVNav:
        forward: positive camera forward axis
        right:   positive image-right axis
        down:    positive image-down axis

    Returned ray is unit-normalized.
    """
    image_width = max(1, int(image_width))
    image_height = max(1, int(image_height))

    x_norm = (float(u) - float(image_width) / 2.0) / (float(image_width) / 2.0)
    y_norm = (float(v) - float(image_height) / 2.0) / (float(image_height) / 2.0)

    tan_x = math.tan(math.radians(float(horizontal_fov_deg)) / 2.0)
    tan_y = math.tan(math.radians(float(vertical_fov_deg)) / 2.0)

    right = x_norm * tan_x
    down = y_norm * tan_y
    forward = 1.0

    norm = math.sqrt(forward * forward + right * right + down * down)
    if norm <= 1e-6:
        norm = 1.0

    ray = {
        "forward": float(forward / norm),
        "right": float(right / norm),
        "down": float(down / norm),
        "x_norm": float(x_norm),
        "y_norm": float(y_norm),
    }
    return ray


def camera_point_from_depth(ray: Dict[str, Any], depth: float) -> Dict[str, float]:
    """
    Convert unit ray + depth to a camera-frame point.

    Depth is treated as distance along the ray. This is a rough candidate-level
    estimate and should not be used as final StopGate evidence.
    """
    depth = float(depth)
    return {
        "forward": float(ray.get("forward", 1.0) * depth),
        "right": float(ray.get("right", 0.0) * depth),
        "down": float(ray.get("down", 0.0) * depth),
    }


def view_yaw_offset_deg(view_id: Any) -> float:
    text = getattr(view_id, "value", str(view_id)).lower()
    if text == "left":
        return -90.0
    if text == "right":
        return 90.0
    if text == "back" or text == "rear":
        return 180.0
    return 0.0


def rotate_2d(x: float, y: float, yaw_deg: float) -> Tuple[float, float]:
    yaw = math.radians(float(yaw_deg))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        float(x) * cos_yaw - float(y) * sin_yaw,
        float(x) * sin_yaw + float(y) * cos_yaw,
    )


def camera_point_to_world(
    camera_point: Dict[str, float],
    pose: Any,
    view_id: Any,
    down_view_mode: str = "down",
) -> Dict[str, Any]:
    """
    Convert rough camera-frame point to world point using UAV pose and view id.

    This is intentionally lightweight. It uses UAV pose yaw and view-specific
    yaw offset for horizontal cameras. For down view, two modes are supported:
        down: treat camera forward axis as vertical downward.
        rear/back: treat it as a horizontal rear camera.
    """
    view_text = getattr(view_id, "value", str(view_id)).lower()
    down_view_mode = str(down_view_mode or "down").lower()

    pose_x = float(getattr(pose, "x", 0.0))
    pose_y = float(getattr(pose, "y", 0.0))
    pose_z = float(getattr(pose, "z", 0.0))
    pose_yaw = float(getattr(pose, "yaw", 0.0))

    forward = float(camera_point.get("forward", 0.0))
    right = float(camera_point.get("right", 0.0))
    down = float(camera_point.get("down", 0.0))

    if view_text == "down" and down_view_mode == "down":
        # Down camera approximation:
        # - camera forward is vertical downward;
        # - image right maps to UAV right;
        # - image down roughly maps to UAV backward/forward axis.
        right_dx, right_dy = rotate_2d(0.0, right, pose_yaw)
        forward_dx, forward_dy = rotate_2d(-down, 0.0, pose_yaw)

        world_x = pose_x + right_dx + forward_dx
        world_y = pose_y + right_dy + forward_dy
        world_z = pose_z + forward

        method = "down_vertical_projection"
    else:
        yaw = pose_yaw + view_yaw_offset_deg(view_id)
        if view_text == "down" and down_view_mode in ("rear", "back"):
            yaw = pose_yaw + 180.0

        forward_dx, forward_dy = rotate_2d(forward, 0.0, yaw)
        right_dx, right_dy = rotate_2d(0.0, right, yaw)

        world_x = pose_x + forward_dx + right_dx
        world_y = pose_y + forward_dy + right_dy
        world_z = pose_z + down

        method = "horizontal_yaw_projection"

    return {
        "world_position": [float(world_x), float(world_y), float(world_z)],
        "pose": {
            "x": pose_x,
            "y": pose_y,
            "z": pose_z,
            "yaw": pose_yaw,
        },
        "view_id": view_text,
        "down_view_mode": down_view_mode,
        "projection_method": method,
    }

