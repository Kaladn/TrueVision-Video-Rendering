from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now
from truevision_runtime.learning_intake.angular_seismic import (
    build_angular_seismic_profile_from_frames,
    detect_content_bounds_from_frame,
    derive_virtual_grid,
)


CALIBRATION_SCHEMA_VERSION = "driving_calibration_receipt_v1"
SCENE_PROFILE_SCHEMA_VERSION = "driving_scene_profile_v1"
EVENT_RECEIPT_SCHEMA_VERSION = "driving_event_receipt_v1"
MOCK_ROAD_WORLD_SCHEMA_VERSION = "mock_road_world_v1"
BATCH_RESULT_SCHEMA_VERSION = "driving_school_batch_result_v1"


CANDIDATE_OBJECT_TYPES = (
    "candidate_road_plane",
    "candidate_scene_depth_layer",
    "candidate_high_speed_forward_flow",
    "candidate_stop_sign",
    "candidate_traffic_light",
    "candidate_speed_limit_sign",
    "candidate_lane_line",
    "candidate_vehicle",
    "candidate_vehicle_shape",
    "candidate_sign_shape",
    "candidate_light_source",
    "candidate_occlusion_event",
    "candidate_tree_mass",
    "candidate_tree_line",
    "candidate_building_mass",
    "candidate_building_edge",
    "candidate_water_surface",
    "candidate_city_skyline",
    "candidate_cloud_field",
    "candidate_red_cloud",
    "candidate_grass_motion",
    "candidate_reflection_field",
    "candidate_fog_reveal",
    "candidate_object_resolving_through_fog",
    "candidate_reflection_human_motion",
)


UNKNOWN_CANDIDATE_TYPES = (
    "candidate_unknown_sign",
    "candidate_unknown_motion",
    "candidate_unknown_reflection",
    "candidate_unknown_roadside_object",
    "candidate_unknown_object",
)


REJECTION_REASONS = (
    "rejected_low_persistence",
    "rejected_no_edge_support",
    "rejected_fog_blob",
    "rejected_camera_motion",
    "rejected_opening_black_bar",
    "rejected_reflection_only",
    "rejected_insufficient_frames",
)


AWARENESS_CURRICULUM = (
    "road_geometry",
    "high_speed_forward_flow",
    "center_attention",
    "roadside_motion",
    "fog_reveal",
    "object_resolving_through_fog",
    "tree_line",
    "grass_motion",
    "building_edge",
    "city_skyline",
    "water_surface",
    "cloud_field",
    "red_cloud",
    "reflection_field",
    "light_source",
    "occlusion_event",
)


def _safe_id(value: str | None, fallback: str = "driving_school") -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "")).strip("_")
    return safe or fallback


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _resize_gray(frame_bgr: np.ndarray, size: tuple[int, int] = (96, 64)) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def _crop_frame(frame_bgr: np.ndarray, bounds: dict[str, Any] | None) -> np.ndarray:
    if not bounds:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    x = int(np.clip(int(bounds.get("x", 0)), 0, max(0, width - 1)))
    y = int(np.clip(int(bounds.get("y", 0)), 0, max(0, height - 1)))
    w = int(np.clip(int(bounds.get("width", width)), 1, width - x))
    h = int(np.clip(int(bounds.get("height", height)), 1, height - y))
    return frame_bgr[y : y + h, x : x + w]


def _frame_meters(frame_bgr: np.ndarray) -> dict[str, float]:
    frame = np.asarray(frame_bgr)
    if frame.size == 0:
        return {
            "luma_mean": 0.0,
            "edge_density": 0.0,
            "texture_energy": 0.0,
            "saturation_mean": 0.0,
            "softness": 1.0,
            "fog_density": 1.0,
        }
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    blur = cv2.GaussianBlur(gray, (5, 5), 0.0)
    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y) * 2.8, 0.0, 1.0)
    texture = np.clip(np.abs(cv2.Laplacian(blur, cv2.CV_32F)) * 3.4, 0.0, 1.0)
    softness = float(np.clip(1.0 - np.mean(edge), 0.0, 1.0))
    contrast_loss = float(np.clip(1.0 - np.std(gray) * 4.0, 0.0, 1.0))
    fog_density = float(np.clip(softness * 0.62 + contrast_loss * 0.38, 0.0, 1.0))
    return {
        "luma_mean": round(float(np.mean(gray)), 6),
        "edge_density": round(float(np.mean(edge)), 6),
        "texture_energy": round(float(np.mean(texture)), 6),
        "saturation_mean": round(float(np.mean(saturation)), 6),
        "softness": round(softness, 6),
        "fog_density": round(fog_density, 6),
    }


def _motion_baseline(frames: list[np.ndarray]) -> dict[str, float]:
    if len(frames) < 2:
        return {"mean_luma_delta": 0.0, "peak_luma_delta": 0.0}
    deltas = []
    previous = _resize_gray(frames[0])
    for frame in frames[1:]:
        current = _resize_gray(frame)
        deltas.append(float(np.mean(np.abs(current - previous))))
        previous = current
    return {
        "mean_luma_delta": round(float(np.mean(deltas)), 6),
        "peak_luma_delta": round(float(np.max(deltas)), 6),
    }


def _black_bar_bounds(content_bounds: dict[str, Any], frame_shape: tuple[int, ...]) -> dict[str, int]:
    height, width = frame_shape[:2]
    x = int(content_bounds.get("x", 0))
    y = int(content_bounds.get("y", 0))
    w = int(content_bounds.get("width", width))
    h = int(content_bounds.get("height", height))
    return {
        "left": max(0, x),
        "top": max(0, y),
        "right": max(0, width - (x + w)),
        "bottom": max(0, height - (y + h)),
    }


def _mean_region(values: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> float:
    rows, cols = values.shape[:2]
    ya = int(np.clip(round(rows * y0), 0, rows))
    yb = int(np.clip(round(rows * y1), ya + 1, rows))
    xa = int(np.clip(round(cols * x0), 0, cols))
    xb = int(np.clip(round(cols * x1), xa + 1, cols))
    return float(np.mean(values[ya:yb, xa:xb])) if yb > ya and xb > xa else 0.0


def _average_frame(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    common_size = (frames[0].shape[1], frames[0].shape[0])
    accum = np.zeros_like(frames[0], dtype=np.float32)
    for frame in frames:
        if (frame.shape[1], frame.shape[0]) != common_size:
            frame = cv2.resize(frame, common_size, interpolation=cv2.INTER_AREA)
        accum += frame.astype(np.float32)
    return np.clip(accum / max(1, len(frames)), 0, 255).astype(np.uint8)


def _candidate_event(
    candidate: str,
    *,
    status: str,
    confidence: float,
    region: dict[str, int],
    meter_evidence: dict[str, float] | None = None,
    rejection_reasons: list[str] | None = None,
    frame_start: int = 0,
    frame_end: int = 0,
) -> dict[str, Any]:
    event = {
        "candidate": candidate,
        "status": status,
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 6),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "region": region,
        "meter_evidence": meter_evidence or {},
        "rejection_reasons": rejection_reasons or [],
        "truth_promotion": "blocked_v0",
    }
    if not event["meter_evidence"] and not event["rejection_reasons"]:
        event["rejection_reasons"] = ["rejected_no_edge_support"]
        event["status"] = "rejected"
    return event


def _awareness_contract(*, sample_fps: float, source_frame_count: int, sampled_frame_count: int) -> dict[str, Any]:
    return {
        "schema_version": "truevision_high_speed_awareness_contract_v0",
        "mode": "high_speed_awareness_v0",
        "purpose": "learn broad near-real-time perception behavior from local video state",
        "driving_claim": False,
        "self_driving_claim": False,
        "traffic_truth_claim": False,
        "near_real_time_perception_target": True,
        "curriculum": list(AWARENESS_CURRICULUM),
        "sampling": {
            "sample_fps": round(float(sample_fps), 6),
            "source_frame_count": int(source_frame_count),
            "sampled_frame_count": int(sampled_frame_count),
            "compact_profile_only": True,
        },
        "law": "Do not learn the movie. Learn high-speed awareness behavior.",
        "boundary": {
            "local_video_only": True,
            "candidate_first": True,
            "review_promotes": True,
            "raw_video_retained_by_default": False,
            "vehicle_control_allowed": False,
        },
    }


def build_driving_calibration_receipt_from_frames(
    frames_bgr: Iterable[np.ndarray],
    *,
    run_id: str,
    source_label: str,
    fps: float,
) -> dict[str, Any]:
    frames = [np.asarray(frame) for frame in frames_bgr]
    if not frames:
        raise ValueError("no frames available for driving calibration")
    first_index = 0
    middle_index = len(frames) // 2
    last_index = len(frames) - 1
    middle_frame = frames[middle_index]
    content_bounds = detect_content_bounds_from_frame(middle_frame)
    cropped_middle = _crop_frame(middle_frame, content_bounds)
    middle_meters = _frame_meters(cropped_middle)
    frame_subset = [frames[first_index], middle_frame, frames[last_index]]
    receipt = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "run_id": _safe_id(run_id),
        "source_label": source_label,
        "fps": round(float(fps or 0.0), 6),
        "first_valid_frame": first_index,
        "middle_frame": middle_index,
        "last_valid_frame": last_index,
        "black_bar_bounds": _black_bar_bounds(content_bounds, middle_frame.shape),
        "content_bounds": content_bounds,
        "camera_motion_baseline": _motion_baseline(frame_subset),
        "noise_baseline": {
            "edge_density": middle_meters["edge_density"],
            "texture_energy": middle_meters["texture_energy"],
        },
        "fog_visibility_baseline": {
            "fog_density": middle_meters["fog_density"],
            "softness": middle_meters["softness"],
            "edge_visibility": round(float(np.clip(middle_meters["edge_density"], 0.0, 1.0)), 6),
        },
        "retention": {
            "raw_video_copied": False,
            "raw_frames_retained": False,
            "compact_receipt_only": True,
        },
    }
    receipt["receipt_sha256"] = stable_hash({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    return receipt


def _road_geometry(frames: list[np.ndarray]) -> dict[str, Any]:
    average = _average_frame(frames)
    height, width = average.shape[:2]
    gray = cv2.cvtColor(average, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (5, 5), 0.0)
    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y) * 2.8, 0.0, 1.0)
    lower = gray[int(height * 0.45) :, :]
    darkness = np.clip(0.58 - lower, 0.0, 1.0)
    yy, xx = np.mgrid[0 : darkness.shape[0], 0 : darkness.shape[1]]
    weight = darkness * np.linspace(0.45, 1.0, darkness.shape[0], dtype=np.float32).reshape(-1, 1)
    total = float(np.sum(weight))
    center_x = float(np.sum(xx * weight) / total) if total > 1.0e-9 else width / 2.0
    road_confidence = float(np.clip(np.mean(darkness) * 2.2 + np.mean(edge[int(height * 0.45) :, :]) * 0.8, 0.0, 1.0))
    left_edge = int(np.clip(center_x - width * 0.34, 0, width - 1))
    right_edge = int(np.clip(center_x + width * 0.34, 0, width - 1))
    horizon_y = int(np.clip(height * 0.40, 0, height - 1))
    return {
        "road_center": {
            "x_normalized": round(center_x / max(width - 1, 1), 6),
            "confidence": round(road_confidence, 6),
        },
        "road_edge_left": {"x": left_edge, "confidence": round(road_confidence * 0.82, 6)},
        "road_edge_right": {"x": right_edge, "confidence": round(road_confidence * 0.82, 6)},
        "horizon_line": {"y": horizon_y, "confidence": round(float(np.mean(edge[max(0, horizon_y - 2) : horizon_y + 3, :])), 6)},
        "vanishing_point": {
            "x_normalized": round(float(np.clip(center_x / max(width - 1, 1), 0.0, 1.0)), 6),
            "y_normalized": round(float(horizon_y / max(height - 1, 1)), 6),
            "confidence": round(road_confidence, 6),
        },
        "curve_direction": "straight_or_unknown",
        "camera_pitch": "level_or_unknown",
    }


def _frame_series_metrics(frames: list[np.ndarray]) -> dict[str, Any]:
    if not frames:
        return {"frame_count": 0}
    meters = [_frame_meters(frame) for frame in frames]
    gray_frames = [_resize_gray(frame) for frame in frames]
    edge_values = np.array([row["edge_density"] for row in meters], dtype=np.float32)
    fog_values = np.array([row["fog_density"] for row in meters], dtype=np.float32)
    motion_values = []
    for previous, current in zip(gray_frames, gray_frames[1:]):
        motion_values.append(float(np.mean(np.abs(current - previous))))
    first_edge = float(edge_values[0]) if edge_values.size else 0.0
    last_edge = float(edge_values[-1]) if edge_values.size else 0.0
    return {
        "frame_count": len(frames),
        "edge_mean": round(float(np.mean(edge_values)), 6),
        "edge_recovery_delta": round(float(last_edge - first_edge), 6),
        "motion_mean": round(float(np.mean(motion_values)) if motion_values else 0.0, 6),
        "motion_peak": round(float(np.max(motion_values)) if motion_values else 0.0, 6),
        "fog_density_mean": round(float(np.mean(fog_values)), 6),
        "fog_density_peak": round(float(np.max(fog_values)), 6),
    }


def _build_candidates(frames: list[np.ndarray], *, road_geometry: dict[str, Any], weather: dict[str, Any]) -> list[dict[str, Any]]:
    average = _average_frame(frames)
    height, width = average.shape[:2]
    hsv = cv2.cvtColor(average, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(average, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    saturation = hsv[:, :, 1] / 255.0
    hue = hsv[:, :, 0]
    blur = cv2.GaussianBlur(gray, (5, 5), 0.0)
    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y) * 2.8, 0.0, 1.0)
    texture = np.clip(np.abs(cv2.Laplacian(blur, cv2.CV_32F)) * 3.4, 0.0, 1.0)
    red_mask = ((hue < 10) | (hue > 165)) & (saturation > 0.35)
    green_mask = (hue > 35) & (hue < 90) & (saturation > 0.18)
    yellow_mask = (hue > 18) & (hue < 38) & (saturation > 0.25)
    center_band = _mean_region(edge, 0.42, 1.0, 0.42, 0.58)
    lane_score = float(np.clip(center_band * 2.5 + _mean_region(gray, 0.48, 1.0, 0.45, 0.55) * 0.12, 0.0, 1.0))
    tree_score = float(np.clip(_mean_region(green_mask.astype(np.float32), 0.18, 0.85, 0.0, 0.35) * 1.8 + _mean_region(edge, 0.18, 0.85, 0.0, 0.35), 0.0, 1.0))
    building_score = float(np.clip(_mean_region(edge + texture, 0.22, 0.68, 0.62, 1.0) * 0.85, 0.0, 1.0))
    sign_score = float(np.clip(_mean_region(red_mask.astype(np.float32), 0.08, 0.55, 0.45, 1.0) * 6.0, 0.0, 1.0))
    light_score = float(np.clip((_mean_region(red_mask.astype(np.float32), 0.0, 0.45, 0.35, 1.0) + _mean_region(yellow_mask.astype(np.float32), 0.0, 0.45, 0.35, 1.0)) * 3.0, 0.0, 1.0))
    motion_mean = float(weather.get("motion_mean") or 0.0)
    road_confidence = float(road_geometry["road_center"]["confidence"])
    fog_density = float(weather.get("fog_density") or 0.0)
    sky_luma = _mean_region(gray, 0.0, 0.30, 0.0, 1.0)
    sky_saturation = _mean_region(saturation, 0.0, 0.30, 0.0, 1.0)
    red_cloud_score = float(np.clip(_mean_region(red_mask.astype(np.float32), 0.0, 0.35, 0.0, 1.0) * 4.0, 0.0, 1.0))
    cloud_score = float(np.clip((1.0 - sky_saturation) * sky_luma * 0.9, 0.0, 1.0))
    water_score = float(np.clip(_mean_region(gray, 0.45, 1.0, 0.0, 1.0) * (1.0 - _mean_region(edge, 0.45, 1.0, 0.0, 1.0)) * 0.55, 0.0, 1.0))
    grass_score = float(np.clip(_mean_region(green_mask.astype(np.float32), 0.55, 1.0, 0.0, 1.0) * max(motion_mean * 5.0, 0.25), 0.0, 1.0))
    skyline_score = float(np.clip(_mean_region(edge + texture, 0.18, 0.48, 0.0, 1.0) * 0.42, 0.0, 1.0))
    reveal_rate = float(weather.get("object_reveal_rate") or 0.0)
    events = [
        _candidate_event(
            "candidate_road_plane",
            status="candidate" if road_confidence >= 0.10 else "rejected",
            confidence=road_confidence,
            region={"x": int(width * 0.10), "y": int(height * 0.38), "w": int(width * 0.80), "h": int(height * 0.62)},
            meter_evidence={
                "road_center_confidence": round(road_confidence, 6),
                "vanishing_point_confidence": road_geometry["vanishing_point"]["confidence"],
            },
            rejection_reasons=[] if road_confidence >= 0.10 else ["rejected_no_edge_support"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_scene_depth_layer",
            status="candidate" if road_confidence >= 0.10 or fog_density >= 0.52 else "rejected",
            confidence=max(road_confidence, fog_density),
            region={"x": 0, "y": 0, "w": width, "h": height},
            meter_evidence={"road_confidence": round(road_confidence, 6), "fog_density": round(fog_density, 6)},
            rejection_reasons=[] if road_confidence >= 0.10 or fog_density >= 0.52 else ["rejected_no_edge_support"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_high_speed_forward_flow",
            status="candidate" if motion_mean >= 0.02 or road_confidence >= 0.10 else "rejected",
            confidence=max(min(1.0, motion_mean * 5.0), road_confidence),
            region={"x": 0, "y": 0, "w": width, "h": height},
            meter_evidence={"motion_mean": round(motion_mean, 6), "center_attention": road_geometry["road_center"]["x_normalized"]},
            rejection_reasons=[] if motion_mean >= 0.02 or road_confidence >= 0.10 else ["rejected_camera_motion"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_lane_line",
            status="candidate" if lane_score >= 0.08 else "rejected",
            confidence=lane_score,
            region={"x": int(width * 0.42), "y": int(height * 0.42), "w": int(width * 0.16), "h": int(height * 0.58)},
            meter_evidence={"edge_density": round(center_band, 6), "road_center_confidence": road_geometry["road_center"]["confidence"]},
            rejection_reasons=[] if lane_score >= 0.08 else ["rejected_no_edge_support"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_tree_mass",
            status="candidate" if tree_score >= 0.10 else "rejected",
            confidence=tree_score,
            region={"x": 0, "y": int(height * 0.18), "w": int(width * 0.35), "h": int(height * 0.67)},
            meter_evidence={"green_edge_mass": round(tree_score, 6), "edge_density": round(_mean_region(edge, 0.18, 0.85, 0.0, 0.35), 6)},
            rejection_reasons=[] if tree_score >= 0.10 else ["rejected_no_edge_support"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_tree_line",
            status="candidate" if tree_score >= 0.10 else "rejected",
            confidence=tree_score,
            region={"x": 0, "y": int(height * 0.12), "w": int(width * 0.45), "h": int(height * 0.75)},
            meter_evidence={"green_edge_line": round(tree_score, 6)},
            rejection_reasons=[] if tree_score >= 0.10 else ["rejected_no_edge_support"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_building_mass",
            status="candidate" if building_score >= 0.08 else "rejected",
            confidence=building_score,
            region={"x": int(width * 0.62), "y": int(height * 0.22), "w": int(width * 0.38), "h": int(height * 0.46)},
            meter_evidence={"rectilinear_edge_texture": round(building_score, 6)},
            rejection_reasons=[] if building_score >= 0.08 else ["rejected_no_edge_support"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_building_edge",
            status="candidate" if building_score >= 0.08 else "rejected",
            confidence=building_score,
            region={"x": int(width * 0.50), "y": int(height * 0.16), "w": int(width * 0.50), "h": int(height * 0.58)},
            meter_evidence={"rectilinear_edge_texture": round(building_score, 6)},
            rejection_reasons=[] if building_score >= 0.08 else ["rejected_no_edge_support"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_water_surface",
            status="candidate" if water_score >= 0.22 else "rejected",
            confidence=water_score,
            region={"x": 0, "y": int(height * 0.42), "w": width, "h": int(height * 0.58)},
            meter_evidence={"low_edge_horizontal_surface": round(water_score, 6)} if water_score >= 0.22 else None,
            rejection_reasons=[] if water_score >= 0.22 else ["rejected_no_edge_support"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_city_skyline",
            status="candidate" if skyline_score >= 0.12 else "rejected",
            confidence=skyline_score,
            region={"x": 0, "y": int(height * 0.12), "w": width, "h": int(height * 0.40)},
            meter_evidence={"upper_rectilinear_edge_density": round(skyline_score, 6)} if skyline_score >= 0.12 else None,
            rejection_reasons=[] if skyline_score >= 0.12 else ["rejected_no_edge_support"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_cloud_field",
            status="candidate" if cloud_score >= 0.16 else "rejected",
            confidence=cloud_score,
            region={"x": 0, "y": 0, "w": width, "h": int(height * 0.38)},
            meter_evidence={"sky_luma": round(sky_luma, 6), "low_saturation_cloud_hint": round(cloud_score, 6)} if cloud_score >= 0.16 else None,
            rejection_reasons=[] if cloud_score >= 0.16 else ["rejected_low_persistence"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_red_cloud",
            status="candidate" if red_cloud_score >= 0.05 else "rejected",
            confidence=red_cloud_score,
            region={"x": 0, "y": 0, "w": width, "h": int(height * 0.38)},
            meter_evidence={"red_cloud_saturation": round(red_cloud_score, 6)} if red_cloud_score >= 0.05 else None,
            rejection_reasons=[] if red_cloud_score >= 0.05 else ["rejected_low_persistence"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_grass_motion",
            status="candidate" if grass_score >= 0.08 else "rejected",
            confidence=grass_score,
            region={"x": 0, "y": int(height * 0.55), "w": width, "h": int(height * 0.45)},
            meter_evidence={"green_motion_field": round(grass_score, 6)} if grass_score >= 0.08 else None,
            rejection_reasons=[] if grass_score >= 0.08 else ["rejected_low_persistence"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_stop_sign",
            status="candidate" if sign_score >= 0.04 else "rejected",
            confidence=sign_score,
            region={"x": int(width * 0.45), "y": int(height * 0.08), "w": int(width * 0.55), "h": int(height * 0.47)},
            meter_evidence={"red_saturation_shape_hint": round(sign_score, 6)},
            rejection_reasons=[] if sign_score >= 0.04 else ["rejected_low_persistence"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_traffic_light",
            status="candidate" if light_score >= 0.04 else "rejected",
            confidence=light_score,
            region={"x": int(width * 0.35), "y": 0, "w": int(width * 0.65), "h": int(height * 0.45)},
            meter_evidence={"red_yellow_green_state_hint": round(light_score, 6)},
            rejection_reasons=[] if light_score >= 0.04 else ["rejected_low_persistence"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_light_source",
            status="candidate" if light_score >= 0.04 else "rejected",
            confidence=light_score,
            region={"x": int(width * 0.35), "y": 0, "w": int(width * 0.65), "h": int(height * 0.45)},
            meter_evidence={"red_yellow_green_state_hint": round(light_score, 6), "luma_mean": round(float(np.mean(gray)), 6)},
            rejection_reasons=[] if light_score >= 0.04 else ["rejected_low_persistence"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_vehicle",
            status="candidate" if motion_mean >= 0.04 and road_geometry["road_center"]["confidence"] >= 0.18 else "rejected",
            confidence=min(1.0, motion_mean * 6.0),
            region={"x": int(width * 0.34), "y": int(height * 0.35), "w": int(width * 0.32), "h": int(height * 0.28)},
            meter_evidence={"closing_motion_proxy": round(motion_mean, 6), "road_center_confidence": road_geometry["road_center"]["confidence"]},
            rejection_reasons=[] if motion_mean >= 0.04 else ["rejected_low_persistence"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_vehicle_shape",
            status="candidate" if motion_mean >= 0.04 and road_confidence >= 0.18 else "rejected",
            confidence=min(1.0, motion_mean * 6.0),
            region={"x": int(width * 0.34), "y": int(height * 0.35), "w": int(width * 0.32), "h": int(height * 0.28)},
            meter_evidence={"closing_motion_proxy": round(motion_mean, 6), "road_plane_support": round(road_confidence, 6)},
            rejection_reasons=[] if motion_mean >= 0.04 else ["rejected_low_persistence"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_sign_shape",
            status="candidate" if sign_score >= 0.04 else "rejected",
            confidence=sign_score,
            region={"x": int(width * 0.45), "y": int(height * 0.08), "w": int(width * 0.55), "h": int(height * 0.47)},
            meter_evidence={"saturation_shape_hint": round(sign_score, 6), "edge_density": round(_mean_region(edge, 0.08, 0.55, 0.45, 1.0), 6)},
            rejection_reasons=[] if sign_score >= 0.04 else ["rejected_low_persistence"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_occlusion_event",
            status="candidate" if fog_density >= 0.52 else "rejected",
            confidence=fog_density,
            region={"x": 0, "y": 0, "w": width, "h": height},
            meter_evidence={"fog_density": round(fog_density, 6), "object_reveal_rate": float(weather.get("object_reveal_rate") or 0.0)},
            rejection_reasons=[] if fog_density >= 0.52 else ["rejected_fog_blob"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_fog_reveal",
            status="candidate" if fog_density >= 0.52 or reveal_rate > 0.0 else "rejected",
            confidence=max(fog_density, reveal_rate),
            region={"x": 0, "y": 0, "w": width, "h": height},
            meter_evidence={"fog_density": round(fog_density, 6), "object_reveal_rate": round(reveal_rate, 6)},
            rejection_reasons=[] if fog_density >= 0.52 or reveal_rate > 0.0 else ["rejected_fog_blob"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_object_resolving_through_fog",
            status="candidate" if fog_density >= 0.52 and reveal_rate > 0.0 else "rejected",
            confidence=float(np.clip(fog_density * 0.7 + reveal_rate * 3.0, 0.0, 1.0)),
            region={"x": int(width * 0.20), "y": int(height * 0.20), "w": int(width * 0.60), "h": int(height * 0.60)},
            meter_evidence={"fog_density": round(fog_density, 6), "edge_recovery": round(float(weather.get("edge_recovery_distance") or 0.0), 6)},
            rejection_reasons=[] if fog_density >= 0.52 and reveal_rate > 0.0 else ["rejected_fog_blob"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_reflection_human_motion",
            status="candidate" if motion_mean >= 0.035 and float(np.mean(saturation)) < 0.32 else "rejected",
            confidence=min(1.0, motion_mean * 5.0),
            region={"x": 0, "y": 0, "w": width, "h": height},
            meter_evidence={"motion_mean": round(motion_mean, 6), "specular_luma": round(float(np.mean(np.clip(gray - 0.62, 0.0, 1.0))), 6)},
            rejection_reasons=[] if motion_mean >= 0.035 else ["rejected_reflection_only"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_reflection_field",
            status="candidate" if motion_mean >= 0.035 and float(np.mean(saturation)) < 0.32 else "rejected",
            confidence=min(1.0, motion_mean * 5.0),
            region={"x": 0, "y": 0, "w": width, "h": height},
            meter_evidence={"motion_mean": round(motion_mean, 6), "specular_luma": round(float(np.mean(np.clip(gray - 0.62, 0.0, 1.0))), 6)},
            rejection_reasons=[] if motion_mean >= 0.035 else ["rejected_reflection_only"],
            frame_end=max(0, len(frames) - 1),
        ),
        _candidate_event(
            "candidate_speed_limit_sign",
            status="rejected",
            confidence=0.0,
            region={"x": int(width * 0.50), "y": int(height * 0.08), "w": int(width * 0.30), "h": int(height * 0.42)},
            rejection_reasons=["rejected_no_edge_support"],
            frame_end=max(0, len(frames) - 1),
        ),
    ]
    unknown_reasons = ["rejected_fog_blob"] if float(weather.get("fog_density") or 0.0) >= 0.55 else ["rejected_no_edge_support"]
    events.extend(
        [
            _candidate_event(
                "candidate_unknown_sign",
                status="candidate" if sign_score >= 0.02 else "rejected",
                confidence=max(sign_score, 0.02),
                region={"x": int(width * 0.45), "y": int(height * 0.08), "w": int(width * 0.55), "h": int(height * 0.47)},
                meter_evidence={"saturation_hint": round(max(sign_score, light_score), 6)} if sign_score >= 0.02 else None,
                rejection_reasons=[] if sign_score >= 0.02 else ["rejected_no_edge_support"],
                frame_end=max(0, len(frames) - 1),
            ),
            _candidate_event(
                "candidate_unknown_motion",
                status="candidate" if motion_mean >= 0.02 else "rejected",
                confidence=min(1.0, motion_mean * 4.0),
                region={"x": 0, "y": 0, "w": width, "h": height},
                meter_evidence={"motion_mean": round(motion_mean, 6)} if motion_mean >= 0.02 else None,
                rejection_reasons=[] if motion_mean >= 0.02 else ["rejected_insufficient_frames"],
                frame_end=max(0, len(frames) - 1),
            ),
            _candidate_event(
                "candidate_unknown_reflection",
                status="rejected",
                confidence=0.0,
                region={"x": 0, "y": 0, "w": width, "h": height},
                rejection_reasons=["rejected_reflection_only"],
                frame_end=max(0, len(frames) - 1),
            ),
            _candidate_event(
                "candidate_unknown_roadside_object",
                status="rejected",
                confidence=float(weather.get("fog_density") or 0.0),
                region={"x": 0, "y": int(height * 0.15), "w": width, "h": int(height * 0.70)},
                rejection_reasons=unknown_reasons,
                frame_end=max(0, len(frames) - 1),
            ),
            _candidate_event(
                "candidate_unknown_object",
                status="rejected",
                confidence=max(float(weather.get("fog_density") or 0.0), motion_mean),
                region={"x": 0, "y": 0, "w": width, "h": height},
                rejection_reasons=unknown_reasons,
                frame_end=max(0, len(frames) - 1),
            ),
        ]
    )
    return events


def _weather_visibility(frames: list[np.ndarray]) -> dict[str, Any]:
    metrics = _frame_series_metrics(frames)
    edge_recovery = max(0.0, float(metrics.get("edge_recovery_delta") or 0.0))
    fog_density = float(metrics.get("fog_density_mean") or 0.0)
    return {
        "fog_density": round(fog_density, 6),
        "low_visibility": fog_density >= 0.52,
        "edge_recovery_distance": round(edge_recovery, 6),
        "object_reveal_rate": round(max(edge_recovery, float(metrics.get("motion_mean") or 0.0) * 0.4), 6),
        "headlight_bloom": 0.0,
        "road_reflection": 0.0,
        "windshield_occlusion": 0.0,
        "sun_glare": 0.0,
        "motion_mean": metrics.get("motion_mean", 0.0),
        "motion_peak": metrics.get("motion_peak", 0.0),
    }


def _build_mock_road_world(profile_core: dict[str, Any]) -> dict[str, Any]:
    road_geometry = profile_core["road_geometry"]
    weather = profile_core["weather_visibility"]
    mock = {
        "schema_version": MOCK_ROAD_WORLD_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_profile_id": profile_core["run_id"],
        "road_plane": {
            "road_center": road_geometry["road_center"],
            "vanishing_point": road_geometry["vanishing_point"],
            "edge_left": road_geometry["road_edge_left"],
            "edge_right": road_geometry["road_edge_right"],
        },
        "forward_motion": {
            "motion_mean": weather.get("motion_mean", 0.0),
            "center_attention": road_geometry["road_center"],
        },
        "visibility_model": {
            "fog_density": weather["fog_density"],
            "edge_recovery_distance": weather["edge_recovery_distance"],
            "object_reveal_rate": weather["object_reveal_rate"],
        },
        "candidate_population": [
            event["candidate"]
            for event in profile_core["candidate_events"]
            if event["status"] == "candidate" and event["candidate"] in CANDIDATE_OBJECT_TYPES + UNKNOWN_CANDIDATE_TYPES
        ],
        "boundary": {
            "renderable_state_not_source_replay": True,
            "does_not_confirm_traffic_objects": True,
            "source_pixels_not_copied": True,
        },
    }
    mock["mock_road_world_sha256"] = stable_hash(mock)
    return mock


def build_driving_scene_profile_from_frames(
    frames_bgr: Iterable[np.ndarray],
    *,
    run_id: str,
    source_label: str,
    fps: float,
    sample_fps: float = 2.0,
    grid_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    source_frames = [np.asarray(frame) for frame in frames_bgr]
    if not source_frames:
        raise ValueError("no frames available for driving scene profile")
    sample_stride = max(1, int(round(float(fps or sample_fps or 1.0) / max(float(sample_fps or 1.0), 0.1))))
    sampled_frames = source_frames[::sample_stride]
    if not sampled_frames:
        sampled_frames = [source_frames[0]]
    calibration = build_driving_calibration_receipt_from_frames(
        source_frames,
        run_id=run_id,
        source_label=source_label,
        fps=fps,
    )
    bounds = calibration["content_bounds"]
    cropped_frames = [_crop_frame(frame, bounds) for frame in sampled_frames]
    if grid_shape is None:
        grid_shape = derive_virtual_grid(
            int(bounds.get("width") or cropped_frames[0].shape[1]),
            int(bounds.get("height") or cropped_frames[0].shape[0]),
            long_edge_cells=48,
            aspect_mode="source",
        )
    angular = build_angular_seismic_profile_from_frames(
        cropped_frames,
        run_id=run_id,
        source_label=source_label,
        fps=float(fps or 1.0),
        loop_count=1,
        sample_stride=sample_stride,
        grid_shape=grid_shape,
    )
    road_geometry = _road_geometry(cropped_frames)
    weather = _weather_visibility(cropped_frames)
    events = _build_candidates(cropped_frames, road_geometry=road_geometry, weather=weather)
    profile_core = {
        "schema_version": SCENE_PROFILE_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "run_id": _safe_id(run_id),
        "awareness_contract": _awareness_contract(
            sample_fps=sample_fps,
            source_frame_count=len(source_frames),
            sampled_frame_count=len(cropped_frames),
        ),
        "source": {
            "source_label": source_label,
            "source_kind": "local_video_frame_state",
            "sample_fps": round(float(sample_fps), 6),
            "source_frame_count": len(source_frames),
            "sampled_frame_count": len(cropped_frames),
            "sample_stride": sample_stride,
        },
        "calibration": calibration,
        "grid": {
            "rows": int(grid_shape[0]),
            "cols": int(grid_shape[1]),
            "direction_count": 16,
            "content_bounds_source": "middle_frame",
        },
        "angular_seismic": {
            "schema_version": angular["schema_version"],
            "angular_signature": angular["angular_signature"],
            "seismic_trace": angular["seismic_trace"],
            "candidate_profiles": angular["candidate_profiles"],
        },
        "road_geometry": road_geometry,
        "motion_attention": {
            "forward_motion": weather["motion_mean"],
            "center_attention": road_geometry["road_center"],
            "roadside_flow_left": 0.0,
            "roadside_flow_right": 0.0,
        },
        "weather_visibility": weather,
        "candidate_events": events,
        "recognition_boundary": {
            "candidate_only": True,
            "truth_promotion_allowed": False,
            "ocr_claims_allowed": False,
            "vehicle_control_allowed": False,
        },
        "retention": {
            "no_raw_video_copy": True,
            "raw_frames_retained": False,
            "compact_profile_only": True,
        },
    }
    profile_core["mock_road_world_v1"] = _build_mock_road_world(profile_core)
    profile_core["profile_sha256"] = stable_hash({key: value for key, value in profile_core.items() if key != "profile_sha256"})
    return profile_core


def _read_video_frames(
    source_video: Path,
    *,
    sample_fps: float,
    max_frames: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {source_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stride = max(1, int(round(fps / max(sample_fps, 0.1))))
    frames: list[np.ndarray] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        frame_index += 1
    cap.release()
    metadata = {
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": round(frame_count / fps, 6) if fps > 0.0 and frame_count > 0 else 0.0,
        "sample_stride": stride,
        "sampled_frames": len(frames),
    }
    return frames, metadata


def build_driving_scene_profile_from_video(
    source_video: Path,
    *,
    run_id: str,
    sample_fps: float = 2.0,
    max_frames: int = 360,
    long_edge_cells: int = 48,
) -> dict[str, Any]:
    source_video = Path(source_video)
    if not source_video.exists():
        raise FileNotFoundError(str(source_video))
    frames, metadata = _read_video_frames(source_video, sample_fps=sample_fps, max_frames=max_frames)
    if not frames:
        raise ValueError(f"no frames sampled from video: {source_video}")
    middle_bounds = detect_content_bounds_from_frame(frames[len(frames) // 2])
    grid_shape = derive_virtual_grid(
        int(middle_bounds.get("width") or metadata["width"]),
        int(middle_bounds.get("height") or metadata["height"]),
        long_edge_cells=long_edge_cells,
        aspect_mode="source",
    )
    profile = build_driving_scene_profile_from_frames(
        frames,
        run_id=run_id,
        source_label=str(source_video),
        fps=float(metadata["fps"]),
        sample_fps=sample_fps,
        grid_shape=grid_shape,
    )
    profile["source"].update(
        {
            "video_path": str(source_video),
            "video_sha256": _file_sha256(source_video),
            "video_metadata": metadata,
        }
    )
    profile["profile_sha256"] = stable_hash({key: value for key, value in profile.items() if key != "profile_sha256"})
    return profile


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _source_sheet_paths(source_root: Path) -> dict[str, Path]:
    return {
        "driving_profile_manifest": source_root / "driving_profile_manifest.json",
        "awareness_profile_manifest": source_root / "awareness_profile_manifest.json",
        "road_geometry_profile": source_root / "road_geometry_profile.json",
        "visibility_depth_profile": source_root / "visibility_depth_profile.json",
        "motion_pressure_profile": source_root / "motion_pressure_profile.json",
        "candidate_object_sheet": source_root / "candidate_object_sheet.json",
        "rejection_sheet": source_root / "rejection_sheet.json",
        "mock_road_world_v1": source_root / "mock_road_world_v1.json",
        "receipt": source_root / "receipt.json",
    }


def _write_source_artifact_sheets(
    *,
    source_root: Path,
    source_video: Path,
    profile: dict[str, Any],
    manifest: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, str]:
    paths = _source_sheet_paths(source_root)
    rejected = [event for event in profile["candidate_events"] if event["status"] == "rejected"]
    candidate_sheet = {
        "schema_version": "candidate_object_sheet_v1",
        "created_at_utc": utc_now(),
        "source_video": str(source_video),
        "candidate_events": profile["candidate_events"],
        "candidate_count": len(profile["candidate_events"]),
        "boundary": {
            "candidate_only": True,
            "truth_promotion_allowed": False,
            "no_ocr_claims": True,
            "no_vehicle_control": True,
        },
    }
    rejection_sheet = {
        "schema_version": "rejection_sheet_v1",
        "created_at_utc": utc_now(),
        "source_video": str(source_video),
        "rejected_candidates": rejected,
        "rejected_candidate_count": len(rejected),
        "rejection_reasons": sorted(
            {
                reason
                for event in rejected
                for reason in event.get("rejection_reasons", [])
            }
        ),
        "boundary": {
            "unknown_is_allowed": True,
            "weak_candidates_do_not_promote": True,
        },
    }
    outputs = {
        paths["driving_profile_manifest"]: {
            "schema_version": "driving_profile_manifest_v1",
            "created_at_utc": utc_now(),
            "source_video": str(source_video),
            "profile_sha256": profile["profile_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "law": "Do not learn the movie. Learn high-speed awareness behavior.",
            "outputs": {name: str(path) for name, path in paths.items()},
            "boundary": profile["recognition_boundary"],
        },
        paths["awareness_profile_manifest"]: {
            "schema_version": "awareness_profile_manifest_v1",
            "created_at_utc": utc_now(),
            "source_video": str(source_video),
            "mode": "high_speed_awareness_v0",
            "law": "Do not learn the movie. Learn high-speed awareness behavior.",
            "awareness_contract": profile["awareness_contract"],
            "profile_sha256": profile["profile_sha256"],
            "outputs": {name: str(path) for name, path in paths.items()},
            "boundary": {
                "not_driving_identity": True,
                "not_self_driving": True,
                "candidate_first": True,
                "near_real_time_perception_target": True,
            },
        },
        paths["road_geometry_profile"]: {
            "schema_version": "road_geometry_profile_v1",
            "created_at_utc": utc_now(),
            "source_video": str(source_video),
            "road_geometry": profile["road_geometry"],
            "calibration_content_bounds": profile["calibration"]["content_bounds"],
        },
        paths["visibility_depth_profile"]: {
            "schema_version": "visibility_depth_profile_v1",
            "created_at_utc": utc_now(),
            "source_video": str(source_video),
            "weather_visibility": profile["weather_visibility"],
            "fog_visibility_baseline": profile["calibration"]["fog_visibility_baseline"],
        },
        paths["motion_pressure_profile"]: {
            "schema_version": "motion_pressure_profile_v1",
            "created_at_utc": utc_now(),
            "source_video": str(source_video),
            "motion_attention": profile["motion_attention"],
            "angular_seismic": profile["angular_seismic"],
        },
        paths["candidate_object_sheet"]: candidate_sheet,
        paths["rejection_sheet"]: rejection_sheet,
        paths["mock_road_world_v1"]: profile["mock_road_world_v1"],
        paths["receipt"]: receipt,
    }
    for path, payload in outputs.items():
        _write_json(path, payload)
    return {name: str(path) for name, path in paths.items()}


def _source_paths(args: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for item in args.get("sources") or []:
        if str(item).strip():
            paths.append(Path(str(item)))
    folder_value = args.get("source_folders") or args.get("source_folder") or []
    if isinstance(folder_value, (str, Path)):
        folders = [folder_value]
    else:
        folders = list(folder_value)
    for folder in folders:
        root = Path(str(folder))
        if root.exists():
            paths.extend(sorted(root.glob("*.mp4")))
            paths.extend(sorted(root.glob("*.MP4")))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def write_driving_school_run(args: dict[str, Any], *, storage_root: Path) -> dict[str, Any]:
    run_id = _safe_id(str(args.get("run_id") or "driving_school_v0"))
    sample_fps = float(args.get("sample_fps") or 2.0)
    max_frames = int(args.get("max_frames") or 360)
    long_edge_cells = int(args.get("long_edge_cells") or 48)
    artifact_root = storage_root / "artifacts" / "driving_school" / run_id
    manifest_root = storage_root / "manifests" / "driving_school"
    receipt_root = storage_root / "receipts" / "driving_school"
    for path in (artifact_root, manifest_root, receipt_root):
        path.mkdir(parents=True, exist_ok=True)

    source_results: list[dict[str, Any]] = []
    mock_worlds: list[dict[str, Any]] = []
    for index, source in enumerate(_source_paths(args)):
        source_run_id = _safe_id(f"{run_id}_{index:03d}_{source.stem}")
        source_artifact_root = artifact_root / source_run_id
        profile = build_driving_scene_profile_from_video(
            source,
            run_id=source_run_id,
            sample_fps=sample_fps,
            max_frames=max_frames,
            long_edge_cells=long_edge_cells,
        )
        profile_path = artifact_root / f"{source_run_id}_profile.json"
        _write_json(profile_path, profile)
        calibration_path = receipt_root / f"{source_run_id}_calibration_receipt.json"
        _write_json(calibration_path, profile["calibration"])
        planned_sheet_paths = {name: str(path) for name, path in _source_sheet_paths(source_artifact_root).items()}
        manifest = {
            "schema_version": "driving_source_manifest_v1",
            "created_at_utc": utc_now(),
            "run_id": source_run_id,
            "source_video": str(source),
            "profile_json": str(profile_path),
            "profile_sha256": profile["profile_sha256"],
            "calibration_receipt_json": str(calibration_path),
            "calibration": profile["calibration"],
            "candidate_event_count": len(profile["candidate_events"]),
            "mock_road_world_v1": profile["mock_road_world_v1"],
            "artifact_sheets": planned_sheet_paths,
            "retention": profile["retention"],
        }
        manifest["manifest_sha256"] = stable_hash(manifest)
        manifest_path = manifest_root / f"{source_run_id}_manifest.json"
        _write_json(manifest_path, manifest)
        receipt = {
            "schema_version": EVENT_RECEIPT_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "tool": "truevision_driving_school_v0",
            "run_id": source_run_id,
            "source_video": str(source),
            "manifest_json": str(manifest_path),
            "profile_json": str(profile_path),
            "calibration_receipt_json": str(calibration_path),
            "artifact_sheets": planned_sheet_paths,
            "candidate_event_count": len(profile["candidate_events"]),
            "candidates_remain_unconfirmed": True,
            "mock_road_world_hash": profile["mock_road_world_v1"]["mock_road_world_sha256"],
            "boundary": {
                "local_file_only": True,
                "browser_automation": False,
                "live_dashcam": False,
                "vehicle_control": False,
                "raw_video_copied": False,
                "raw_frames_retained": False,
                "truth_promotion_allowed": False,
            },
        }
        receipt["receipt_sha256"] = stable_hash(receipt)
        receipt_path = receipt_root / f"{source_run_id}_receipt.json"
        _write_json(receipt_path, receipt)
        artifact_sheets = _write_source_artifact_sheets(
            source_root=source_artifact_root,
            source_video=source,
            profile=profile,
            manifest=manifest,
            receipt=receipt,
        )
        mock_worlds.append(profile["mock_road_world_v1"])
        source_results.append(
            {
                "source_video": str(source),
                "profile_json": str(profile_path),
                "calibration_receipt_json": str(calibration_path),
                "manifest_json": str(manifest_path),
                "receipt_json": str(receipt_path),
                "artifact_sheets": artifact_sheets,
                "profile_sha256": profile["profile_sha256"],
                "candidate_event_count": len(profile["candidate_events"]),
            }
        )

    aggregate_mock = {
        "schema_version": MOCK_ROAD_WORLD_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "source_profile_count": len(mock_worlds),
        "worlds": mock_worlds,
        "boundary": {
            "renderable_state_not_source_replay": True,
            "source_pixels_not_copied": True,
            "candidate_only": True,
        },
    }
    aggregate_mock["mock_road_world_sha256"] = stable_hash(aggregate_mock)
    mock_path = artifact_root / f"{run_id}_mock_road_world_v1.json"
    _write_json(mock_path, aggregate_mock)
    result = {
        "schema_version": BATCH_RESULT_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "source_count": len(source_results),
        "sources": source_results,
        "mock_road_world_json": str(mock_path),
        "boundary": {
            "local_file_only": True,
            "raw_video_copied": False,
            "raw_frames_retained": False,
            "candidate_only": True,
        },
    }
    result["result_sha256"] = stable_hash(result)
    result_path = manifest_root / f"{run_id}_batch_result.json"
    _write_json(result_path, result)
    result["batch_result_json"] = str(result_path)
    return result
