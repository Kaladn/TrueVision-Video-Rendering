from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now
from truevision_runtime.state_patterns.atmosphere_weather import _read_native_tvcells_frames


METER_NAMES = (
    "luma_mean",
    "luma_peak",
    "luma_delta",
    "saturation",
    "color_temperature",
    "edge_density",
    "edge_orientation",
    "motion_magnitude",
    "motion_direction",
    "texture_energy",
    "flicker_score",
    "bloom_pressure",
    "persistence_frames",
    "softness",
    "occlusion_change",
)


GRAPH_CURVES = {
    "luma_curve.png": "luma_peak",
    "bloom_curve.png": "bloom_pressure",
    "motion_curve.png": "motion_magnitude",
    "edge_density_curve.png": "edge_density",
    "exposure_lift_curve.png": "exposure_lift",
}


def _safe_id(value: str | None, fallback: str = "meter_grid") -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "")).strip("_")
    return safe or fallback


def _feature_index(feature_names: list[str], name: str) -> int | None:
    try:
        return feature_names.index(name)
    except ValueError:
        return None


def _feature_plane(cells: np.ndarray, feature_names: list[str], name: str, default: float = 0.0) -> np.ndarray:
    index = _feature_index(feature_names, name)
    if index is None:
        return np.full(cells.shape[:2], default, dtype=np.float32)
    return np.asarray(cells[:, :, index], dtype=np.float32)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low + 1.0e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _dominant_edge_orientation(luma: np.ndarray) -> dict[str, Any]:
    if luma.size == 0:
        return {"label": "none", "angle_degrees": 0.0, "variance": 0.0}
    gy, gx = np.gradient(np.asarray(luma, dtype=np.float32))
    magnitude = np.sqrt(gx * gx + gy * gy)
    active = magnitude > max(1.0e-6, float(np.percentile(magnitude, 75)))
    if not bool(np.any(active)):
        return {"label": "flat", "angle_degrees": 0.0, "variance": 0.0}
    angles = np.degrees(np.arctan2(gy[active], gx[active]))
    mean_angle = float(np.mean(angles))
    variance = float(np.std(angles) / 180.0)
    absolute = abs(mean_angle)
    if absolute < 22.5 or absolute > 157.5:
        label = "horizontal"
    elif 67.5 < absolute < 112.5:
        label = "vertical"
    else:
        label = "diagonal"
    return {
        "label": label,
        "angle_degrees": round(mean_angle, 6),
        "variance": round(float(np.clip(variance, 0.0, 1.0)), 6),
    }


def _motion_direction(previous_luma: np.ndarray, current_luma: np.ndarray) -> dict[str, Any]:
    if previous_luma.size == 0 or current_luma.size == 0:
        return {"label": "none", "dx": 0.0, "dy": 0.0}
    previous_weight = np.clip(previous_luma, 0.0, None)
    current_weight = np.clip(current_luma, 0.0, None)
    rows, cols = current_weight.shape
    yy, xx = np.mgrid[0:rows, 0:cols]

    def center(weight: np.ndarray) -> tuple[float, float]:
        total = float(np.sum(weight))
        if total <= 1.0e-9:
            return 0.5, 0.5
        return float(np.sum((xx / max(cols - 1, 1)) * weight) / total), float(np.sum((yy / max(rows - 1, 1)) * weight) / total)

    px, py = center(previous_weight)
    cx, cy = center(current_weight)
    dx = cx - px
    dy = cy - py
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        label = "stable"
    elif abs(dx) >= abs(dy):
        label = "right" if dx > 0 else "left"
    else:
        label = "down" if dy > 0 else "up"
    return {"label": label, "dx": round(dx, 6), "dy": round(dy, 6)}


def _cell_bounds(mask: np.ndarray) -> list[int]:
    if mask.size == 0 or not bool(np.any(mask)):
        return [0, 0, 0, 0]
    yy, xx = np.where(mask)
    x0 = int(np.min(xx))
    x1 = int(np.max(xx)) + 1
    y0 = int(np.min(yy))
    y1 = int(np.max(yy)) + 1
    return [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]


def _bloom_radius_cells(luma: np.ndarray, baseline_peak: float, peak_luma: float) -> float:
    if luma.size == 0:
        return 0.0
    delta = max(peak_luma - baseline_peak, 0.0)
    if delta <= 1.0e-9:
        return 0.0
    threshold = baseline_peak + delta * 0.28
    mask = luma >= threshold
    if not bool(np.any(mask)):
        return 0.0
    peak_y, peak_x = np.unravel_index(int(np.argmax(luma)), luma.shape)
    yy, xx = np.where(mask)
    radius = float(np.max(np.sqrt((xx - peak_x) ** 2 + (yy - peak_y) ** 2)))
    return round(radius, 6)


def _derive_frame_summaries(
    frames: list[dict[str, Any]],
    *,
    feature_names: list[str],
    capture_fps: float,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    summaries: list[dict[str, Any]] = []
    luma_planes: list[np.ndarray] = []
    persistence: np.ndarray | None = None
    previous_luma: np.ndarray | None = None
    for sampled_index, item in enumerate(frames):
        cells = item["cells"]
        luma = _feature_plane(cells, feature_names, "luma_mean")
        luma_std = _feature_plane(cells, feature_names, "luma_std")
        luma_delta_feature = _feature_plane(cells, feature_names, "delta_luma_abs")
        if previous_luma is None:
            luma_delta = np.zeros_like(luma, dtype=np.float32)
            previous_for_motion = np.zeros_like(luma)
        else:
            measured_delta = np.abs(luma - previous_luma)
            luma_delta = np.maximum(luma_delta_feature, measured_delta)
            previous_for_motion = previous_luma
        edge = _feature_plane(cells, feature_names, "edge_density")
        texture = _feature_plane(cells, feature_names, "texture_energy")
        motion = _feature_plane(cells, feature_names, "motion_energy")
        saturation = _feature_plane(cells, feature_names, "saturation_mean")
        red = _feature_plane(cells, feature_names, "rgb_mean_r")
        blue = _feature_plane(cells, feature_names, "rgb_mean_b")
        if persistence is None:
            persistence = np.zeros_like(luma, dtype=np.int32)
        hot_threshold = max(0.35, float(np.percentile(luma, 92))) if luma.size else 0.35
        persistence = np.where(luma >= hot_threshold, persistence + 1, 0)
        orientation = _dominant_edge_orientation(luma)
        motion_direction = _motion_direction(previous_for_motion, luma)
        luma_peak = float(np.max(luma)) if luma.size else 0.0
        bloom = np.clip(luma * 0.58 + luma_std * 0.18 + luma_delta * 0.24, 0.0, 1.0)
        summaries.append(
            {
                "frame_index": sampled_index,
                "source_frame_index": int(item["global_frame_index"]),
                "time_sec": round(float(item["global_frame_index"]) / max(capture_fps, 1.0e-9), 6),
                "luma_mean": round(float(np.mean(luma)) if luma.size else 0.0, 6),
                "luma_peak": round(luma_peak, 6),
                "luma_delta": round(float(np.max(luma_delta)) if luma_delta.size else 0.0, 6),
                "saturation": round(float(np.mean(saturation)) if saturation.size else 0.0, 6),
                "color_temperature": round(float(np.mean(blue - red)) if red.size and blue.size else 0.0, 6),
                "edge_density": round(float(np.mean(edge)) if edge.size else 0.0, 6),
                "edge_orientation": orientation,
                "motion_magnitude": round(float(np.mean(np.maximum(motion, luma_delta))) if motion.size else 0.0, 6),
                "motion_direction": motion_direction,
                "texture_energy": round(float(np.mean(texture)) if texture.size else 0.0, 6),
                "flicker_score": round(float(np.clip(np.max(luma_delta) * 1.08 + np.std(luma) * 0.28, 0.0, 1.0)) if luma.size else 0.0, 6),
                "bloom_pressure": round(float(np.max(bloom)) if bloom.size else 0.0, 6),
                "persistence_frames": int(np.max(persistence)) if persistence.size else 0,
                "softness": round(float(np.clip(1.0 - np.mean(edge), 0.0, 1.0)) if edge.size else 0.0, 6),
                "occlusion_change": round(float(np.mean(np.clip(previous_for_motion - luma, 0.0, 1.0))) if luma.size else 0.0, 6),
                "exposure_lift": round(float(np.mean(luma - previous_for_motion)) if luma.size else 0.0, 6),
            }
        )
        luma_planes.append(luma)
        previous_luma = luma
    return summaries, luma_planes


def _curve_values(frame_summaries: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in frame_summaries:
        value = row.get(key, 0.0)
        if isinstance(value, dict):
            value = 0.0
        values.append(float(value))
    return values


def _rise_time(values: np.ndarray, peak_index: int, baseline: float, peak: float) -> int:
    threshold = baseline + max(peak - baseline, 0.0) * 0.20
    start = 0
    for index in range(peak_index, -1, -1):
        if float(values[index]) <= threshold:
            start = index
            break
    return max(0, peak_index - start)


def _falloff_time(values: np.ndarray, peak_index: int, baseline: float, peak: float) -> int:
    threshold = baseline + max(peak - baseline, 0.0) * 0.25
    for index in range(peak_index + 1, len(values)):
        if float(values[index]) <= threshold:
            return max(1, index - peak_index)
    return max(1, len(values) - peak_index - 1)


def _candidate_lightning_profile(
    section_id: str,
    frame_summaries: list[dict[str, Any]],
    luma_planes: list[np.ndarray],
) -> dict[str, Any]:
    if not frame_summaries:
        return {
            "event_type_candidate": "candidate_lightning",
            "section_id": section_id,
            "status": "rejected",
            "rejection_reasons": ["empty_section"],
        }
    luma_peaks = np.array(_curve_values(frame_summaries, "luma_peak"), dtype=np.float32)
    luma_deltas = np.array(_curve_values(frame_summaries, "luma_delta"), dtype=np.float32)
    bloom = np.array(_curve_values(frame_summaries, "bloom_pressure"), dtype=np.float32)
    edge = np.array(_curve_values(frame_summaries, "edge_density"), dtype=np.float32)
    score = luma_deltas * 0.58 + bloom * 0.22 + luma_peaks * 0.20
    peak_index = int(np.argmax(score))
    prior = luma_peaks[:peak_index] if peak_index > 0 else luma_peaks[:1]
    baseline = float(np.median(prior)) if prior.size else float(luma_peaks[0])
    peak = float(luma_peaks[peak_index])
    luma_delta = float(max(np.max(luma_deltas), peak - baseline))
    rise = _rise_time(luma_peaks, peak_index, baseline, peak)
    falloff = _falloff_time(luma_peaks, peak_index, baseline, peak)
    hot_threshold = baseline + max(peak - baseline, 0.0) * 0.28
    peak_plane = luma_planes[peak_index] if luma_planes else np.zeros((0, 0), dtype=np.float32)
    hot_mask = peak_plane >= hot_threshold if peak_plane.size else np.zeros((0, 0), dtype=bool)
    persistence_threshold = baseline + max(peak - baseline, 0.0) * 0.25
    persistence_frames = int(np.sum(luma_peaks >= persistence_threshold))
    radius = _bloom_radius_cells(peak_plane, baseline, peak)
    surrounding_lift = float(np.mean(peak_plane[~hot_mask]) - baseline) if peak_plane.size and bool(np.any(~hot_mask)) else 0.0
    support = {
        "pre_flash_luma_baseline": round(baseline, 6),
        "flash_peak_luma": round(peak, 6),
        "luma_delta": round(luma_delta, 6),
        "rise_time_frames": int(rise),
        "falloff_time_frames": int(falloff),
        "bloom_radius_cells": radius,
        "sky_exposure_lift_radius": round(max(radius - 1.0, 0.0), 6),
        "branch_edge_density": round(float(edge[peak_index]) if edge.size else 0.0, 6),
        "branch_direction_variance": frame_summaries[peak_index]["edge_orientation"]["variance"],
        "afterglow_decay_curve": [round(float(value), 6) for value in luma_peaks[peak_index : min(len(luma_peaks), peak_index + 8)]],
        "surrounding_exposure_lift": round(float(max(surrounding_lift, 0.0)), 6),
        "persistence_frames": persistence_frames,
    }
    rejection_reasons: list[str] = []
    if luma_delta < 0.25:
        rejection_reasons.append("no_temporal_flash")
    if peak >= 0.55 and persistence_frames >= max(8, int(len(luma_peaks) * 0.55)) and luma_delta < 0.35:
        rejection_reasons.append("persistent_bright_region")
    if rise > 3 and luma_delta >= 0.25:
        rejection_reasons.append("slow_rise_not_lightning")
    if falloff > max(18, int(len(luma_peaks) * 0.65)) and luma_delta >= 0.25:
        rejection_reasons.append("too_persistent_for_lightning")
    status = "visually_supported" if not rejection_reasons and luma_delta >= 0.45 and rise <= 3 and radius >= 1.0 else "rejected"
    if status == "rejected" and not rejection_reasons:
        rejection_reasons.append("insufficient_meter_support")
    return {
        "event_type_candidate": "candidate_lightning",
        "section_id": section_id,
        "frame_start": max(0, peak_index - rise),
        "frame_peak": peak_index,
        "frame_end": min(len(frame_summaries) - 1, peak_index + falloff),
        "cell_bounds": _cell_bounds(hot_mask),
        "meter_peaks": {
            "luma_peak": round(float(np.max(luma_peaks)), 6),
            "luma_delta": round(float(np.max(luma_deltas)), 6),
            "bloom_pressure": round(float(np.max(bloom)), 6),
            "edge_density": round(float(np.max(edge)), 6),
        },
        "meter_curves_summary": {
            "luma_curve_min": round(float(np.min(luma_peaks)), 6),
            "luma_curve_max": round(float(np.max(luma_peaks)), 6),
            "bloom_curve_max": round(float(np.max(bloom)), 6),
            "delta_curve_max": round(float(np.max(luma_deltas)), 6),
        },
        "support": support,
        "visual_support_reasons": [
            "fast_luma_spike",
            "measured_bloom_radius",
            "surrounding_exposure_response",
            "short_afterglow_decay",
        ]
        if status == "visually_supported"
        else [],
        "rejection_reasons": rejection_reasons,
        "status": status,
    }


def _event_profile(
    *,
    section_id: str,
    event_type_candidate: str,
    frame_summaries: list[dict[str, Any]],
    luma_planes: list[np.ndarray],
) -> dict[str, Any]:
    if event_type_candidate == "candidate_lightning":
        return _candidate_lightning_profile(section_id, frame_summaries, luma_planes)
    return {
        "event_type_candidate": event_type_candidate,
        "section_id": section_id,
        "status": "unclassified",
        "rejection_reasons": ["unsupported_event_type"],
    }


def _profile_hash(profile: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in profile.items() if key != "profile_sha256"})


def build_meter_grid_profile_from_native_capture(
    manifest_path: str | Path,
    *,
    section_id: str,
    event_type_candidate: str = "candidate_lightning",
    max_frames: int = 180,
    sample_stride: int = 1,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell_state = manifest.get("cell_state") or {}
    feature_names = list(cell_state.get("feature_names") or [])
    frames = _read_native_tvcells_frames(manifest, max_frames=max_frames, sample_stride=sample_stride)
    capture_fps = float((manifest.get("config") or {}).get("capture_fps") or 1.0)
    frame_summaries, luma_planes = _derive_frame_summaries(frames, feature_names=feature_names, capture_fps=capture_fps)
    profile = {
        "schema_version": "truevision_meter_grid_profile_v0",
        "created_at_utc": utc_now(),
        "section_id": section_id,
        "source": {
            "manifest_json": str(manifest_path),
            "run_id": manifest.get("run_id"),
            "capture_fps": capture_fps,
            "grid_size_xy": (manifest.get("config") or {}).get("grid_size_xy"),
            "capture_region": (manifest.get("config") or {}).get("capture_region"),
            "raw_frame_saved": bool((manifest.get("boundary") or {}).get("raw_frame_saved", False)),
        },
        "meter_names": list(METER_NAMES),
        "sampled_frames": len(frame_summaries),
        "frame_meter_summaries": frame_summaries,
        "event_profiles": [
            _event_profile(
                section_id=section_id,
                event_type_candidate=event_type_candidate,
                frame_summaries=frame_summaries,
                luma_planes=luma_planes,
            )
        ],
        "retention": {
            "stores_raw_teacher_video": False,
            "stores_per_cell_raw_dump": False,
            "durable_output": "compact_meter_profile_and_graphs",
        },
        "boundary": {
            "no_meter_no_claim": True,
            "no_graph_no_tuning": True,
            "no_profile_no_renderer_rule": True,
            "state_first_pixels_last": True,
            "generated_media_is_evidence": False,
        },
    }
    profile["profile_sha256"] = _profile_hash(profile)
    return profile


def _lightning_selection_score(event: dict[str, Any]) -> tuple[float, list[str]]:
    support = event.get("support") or {}
    luma_delta = float(support.get("luma_delta") or 0.0)
    rise = float(support.get("rise_time_frames") or 999.0)
    falloff = float(support.get("falloff_time_frames") or 0.0)
    bloom_radius = float(support.get("bloom_radius_cells") or 0.0)
    surrounding_lift = float(support.get("surrounding_exposure_lift") or 0.0)
    branch_edge = float(support.get("branch_edge_density") or 0.0)
    score = 0.0
    reasons: list[str] = []
    if event.get("status") == "visually_supported":
        score += 0.35
        reasons.append("candidate visually supported by meter event")
    score += min(0.25, luma_delta * 0.25)
    if luma_delta >= 0.45:
        reasons.append("strong luma spike")
    if rise <= 3:
        score += 0.12
        reasons.append("fast attack")
    if 1 <= falloff <= 18:
        score += 0.10
        reasons.append("bounded decay")
    score += min(0.10, bloom_radius / 20.0)
    if bloom_radius >= 1.0:
        reasons.append("measured bloom radius")
    score += min(0.05, surrounding_lift * 0.10)
    if surrounding_lift > 0.05:
        reasons.append("surrounding exposure response")
    score += min(0.03, branch_edge * 0.03)
    rejection_penalty = 0.10 * len(event.get("rejection_reasons") or [])
    score = max(0.0, min(1.0, score - rejection_penalty))
    if not reasons:
        reasons.append("weak or rejected meter signature")
    return round(score, 6), reasons


def _generic_selection_score(event: dict[str, Any]) -> tuple[float, list[str]]:
    status = event.get("status")
    if status == "visually_supported":
        return 0.65, ["candidate visually supported"]
    return 0.05, event.get("rejection_reasons") or ["unsupported or weak candidate"]


def build_metered_section_selection_plan(
    meter_profiles: list[dict[str, Any]],
    *,
    target_signature: str,
    controller_id: str = "operator_agent",
) -> dict[str, Any]:
    """Rank probed long-video sections by measured target support.

    This does not control a browser. It produces the action surface that a
    human-owned controller can use: where to go next, why, and when to stop.
    """

    ranked: list[dict[str, Any]] = []
    for profile in meter_profiles:
        event = (profile.get("event_profiles") or [{}])[0]
        if target_signature == "candidate_lightning":
            score, reasons = _lightning_selection_score(event)
        else:
            score, reasons = _generic_selection_score(event)
        ranked.append(
            {
                "section_id": profile.get("section_id"),
                "source_run_id": (profile.get("source") or {}).get("run_id"),
                "target_signature": target_signature,
                "score": score,
                "status": event.get("status", "unknown"),
                "recommended_action": "capture_full_section" if score >= 0.60 else "skip_or_probe_elsewhere",
                "meter_reasons": reasons,
                "rejection_reasons": event.get("rejection_reasons") or [],
                "frame_peak": event.get("frame_peak"),
                "cell_bounds": event.get("cell_bounds"),
            }
        )
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    plan = {
        "schema_version": "truevision_metered_section_selection_plan_v0",
        "created_at_utc": utc_now(),
        "target_signature": target_signature,
        "controller": {
            "controller_id": controller_id,
            "role": "human_owned_navigation_agent",
            "authority": "may choose next probe or capture target from ranked meter plan; may not claim success without receipt",
        },
        "ranked_sections": ranked,
        "stop_conditions": [
            "target score below threshold",
            "source page not verified",
            "capture rectangle not approved",
            "operator stops playback",
        ],
        "boundary": {
            "agent_controls_navigation_only_after_meter_goal": True,
            "no_meter_no_section_choice": True,
            "no_receipt_no_success_claim": True,
            "operator_owns_source_playback": True,
            "browser_control_not_required_by_this_plan": True,
        },
    }
    plan["plan_sha256"] = stable_hash(plan)
    return plan


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_rgb_png(path: Path, pixels: np.ndarray) -> None:
    pixels = np.asarray(pixels, dtype=np.uint8)
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("PNG writer expects RGB pixels")
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, 9))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _write_curve_png(path: Path, values: list[float], *, color: tuple[int, int, int]) -> None:
    width = 360
    height = 140
    pixels = np.full((height, width, 3), 10, dtype=np.uint8)
    pixels[height - 22 : height - 20, 28 : width - 12] = 70
    pixels[18 : height - 20, 28:30] = 70
    if not values:
        _write_rgb_png(path, pixels)
        return
    norm = _normalize(np.array(values, dtype=np.float32))
    if norm.size == 1:
        points = [(28, int(height - 22 - norm[0] * (height - 42)))]
    else:
        points = [
            (
                int(28 + index * (width - 48) / max(1, len(norm) - 1)),
                int(height - 22 - float(value) * (height - 42)),
            )
            for index, value in enumerate(norm)
        ]
    for x, y in points:
        x0 = max(0, x - 1)
        x1 = min(width, x + 2)
        y0 = max(0, y - 1)
        y1 = min(height, y + 2)
        pixels[y0:y1, x0:x1] = color
    _write_rgb_png(path, pixels)


def _verify_profile(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("profile_sha256") == _profile_hash(payload)


def write_meter_grid_from_capture(args: dict[str, Any], *, storage_root: Path) -> dict[str, Any]:
    manifest_path = Path(str(args.get("manifest") or args.get("manifest_json") or ""))
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))
    section_id = str(args.get("section_id") or args.get("element_id") or "meter_grid_section")
    event_type_candidate = str(args.get("event_type_candidate") or "candidate_lightning")
    run_id = _safe_id(str(args.get("run_id") or section_id))
    profile = build_meter_grid_profile_from_native_capture(
        manifest_path,
        section_id=section_id,
        event_type_candidate=event_type_candidate,
        max_frames=int(args.get("max_frames") or 180),
        sample_stride=int(args.get("sample_stride") or 1),
    )
    profile_root = storage_root / "artifacts" / "meter_grid"
    manifest_root = storage_root / "manifests" / "meter_grid"
    receipt_root = storage_root / "receipts" / "meter_grid"
    graph_root = storage_root / "reports" / "meter_grid" / f"{run_id}_graphs"
    for path in (profile_root, manifest_root, receipt_root, graph_root):
        path.mkdir(parents=True, exist_ok=True)
    profile_path = profile_root / f"{run_id}_{_safe_id(section_id)}_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")
    if not _verify_profile(profile_path):
        raise ValueError("meter grid profile verification failed")
    graph_paths: dict[str, str] = {}
    colors = {
        "luma_curve.png": (245, 245, 255),
        "bloom_curve.png": (120, 185, 255),
        "motion_curve.png": (80, 230, 180),
        "edge_density_curve.png": (250, 195, 80),
        "exposure_lift_curve.png": (230, 95, 130),
    }
    for filename, key in GRAPH_CURVES.items():
        path = graph_root / filename
        _write_curve_png(path, _curve_values(profile["frame_meter_summaries"], key), color=colors[filename])
        graph_paths[filename] = str(path)
    tool_manifest = {
        "schema_version": "truevision_meter_grid_manifest_v0",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "section_id": section_id,
        "profile_json": str(profile_path),
        "profile_sha256": profile["profile_sha256"],
        "graphs": graph_paths,
        "event_profiles": profile["event_profiles"],
        "source": profile["source"],
        "boundary": profile["boundary"],
    }
    tool_manifest["manifest_sha256"] = stable_hash(tool_manifest)
    manifest_out = manifest_root / f"{run_id}_manifest.json"
    manifest_out.write_text(json.dumps(tool_manifest, indent=2, allow_nan=False), encoding="utf-8")
    receipt = {
        "schema_version": "truevision_meter_grid_receipt_v0",
        "created_at_utc": utc_now(),
        "tool": "meter_grid_from_capture",
        "run_id": run_id,
        "section_id": section_id,
        "profile_json": str(profile_path),
        "profile_sha256": profile["profile_sha256"],
        "manifest_json": str(manifest_out),
        "graphs": graph_paths,
        "event_status": profile["event_profiles"][0]["status"] if profile["event_profiles"] else "none",
        "boundary": profile["boundary"],
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    receipt_path = receipt_root / f"{run_id}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_id": run_id,
        "section_id": section_id,
        "profile_json": str(profile_path),
        "profile_sha256": profile["profile_sha256"],
        "manifest_json": str(manifest_out),
        "receipt_json": str(receipt_path),
        "graphs": graph_paths,
        "event_status": receipt["event_status"],
    }
