from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now
from truevision_runtime.state_patterns.atmosphere_weather import _read_native_tvcells_frames


REPORT_SCHEMA = "truevision_state_recognition_report_v1"


STATE_CATALOG: tuple[dict[str, Any], ...] = (
    {"state_family": "line", "state_name": "line_appears", "reusable_transform_candidate": True},
    {"state_family": "line", "state_name": "line_disappears", "reusable_transform_candidate": True},
    {"state_family": "line", "state_name": "line_thickens_thins", "reusable_transform_candidate": True},
    {"state_family": "line", "state_name": "line_breathing", "reusable_transform_candidate": True},
    {"state_family": "line", "state_name": "ink_crawl", "reusable_transform_candidate": True},
    {"state_family": "line", "state_name": "edge_shimmer", "reusable_transform_candidate": True},
    {"state_family": "line", "state_name": "edge_recovery", "reusable_transform_candidate": True},
    {"state_family": "light", "state_name": "luminance_rise_fall", "reusable_transform_candidate": True},
    {"state_family": "light", "state_name": "bloom_pressure", "reusable_transform_candidate": True},
    {"state_family": "light", "state_name": "glow_spread", "reusable_transform_candidate": True},
    {"state_family": "light", "state_name": "shadow_deepening", "reusable_transform_candidate": True},
    {"state_family": "light", "state_name": "strobe_rejection", "reusable_transform_candidate": False},
    {"state_family": "surface", "state_name": "texture_shimmer", "reusable_transform_candidate": True},
    {"state_family": "surface", "state_name": "reflection_pulse", "reusable_transform_candidate": True},
    {"state_family": "surface", "state_name": "ripple", "reusable_transform_candidate": True},
    {"state_family": "surface", "state_name": "grain_movement", "reusable_transform_candidate": True},
    {"state_family": "surface", "state_name": "metallic_flicker", "reusable_transform_candidate": True},
    {"state_family": "atmosphere", "state_name": "haze_veil", "reusable_transform_candidate": True},
    {"state_family": "atmosphere", "state_name": "fog_reveal", "reusable_transform_candidate": True},
    {"state_family": "atmosphere", "state_name": "scatter_lift", "reusable_transform_candidate": True},
    {"state_family": "atmosphere", "state_name": "depth_fade", "reusable_transform_candidate": True},
    {"state_family": "atmosphere", "state_name": "edge_loss_under_veil", "reusable_transform_candidate": True},
    {"state_family": "motion", "state_name": "center_energy_pull", "reusable_transform_candidate": True},
    {"state_family": "motion", "state_name": "parallax_like_shift", "reusable_transform_candidate": True},
    {"state_family": "motion", "state_name": "tunnel_compression", "reusable_transform_candidate": True},
    {"state_family": "motion", "state_name": "motion_pressure_pulse", "reusable_transform_candidate": True},
    {"state_family": "motion", "state_name": "frame_wide_state_surge", "reusable_transform_candidate": True},
)


@dataclass(frozen=True)
class StateSpec:
    state_family: str
    state_name: str
    score_key: str
    mask_key: str
    evidence_keys: tuple[str, ...]
    reusable_transform_candidate: bool = True
    threshold: float = 0.45


STATE_SPECS: tuple[StateSpec, ...] = (
    StateSpec("line", "line_appears", "line_appears_score", "line_mask", ("line_coverage_delta", "edge_delta", "line_coverage")),
    StateSpec("line", "line_disappears", "line_disappears_score", "line_mask", ("line_coverage_delta", "edge_delta", "line_coverage")),
    StateSpec("line", "line_thickens_thins", "line_thickness_score", "line_mask", ("line_width_delta", "edge_delta", "line_width_proxy")),
    StateSpec("line", "line_breathing", "line_breathing_score", "line_mask", ("edge_delta", "line_coverage", "line_width_delta")),
    StateSpec("line", "ink_crawl", "ink_crawl_score", "line_mask", ("edge_center_shift", "line_coverage", "motion_mean")),
    StateSpec("line", "edge_shimmer", "edge_shimmer_score", "line_mask", ("edge_delta", "texture_delta", "luma_delta")),
    StateSpec("line", "edge_recovery", "edge_recovery_score", "line_mask", ("edge_delta_positive", "previous_haze_coverage", "edge_mean")),
    StateSpec("light", "luminance_rise_fall", "luminance_score", "bright_mask", ("luma_delta", "luma_mean_delta", "luma_peak")),
    StateSpec("light", "bloom_pressure", "bloom_score", "bright_mask", ("bloom_pressure", "bloom_delta", "bright_area")),
    StateSpec("light", "glow_spread", "glow_spread_score", "bright_mask", ("bright_area_delta", "bright_area", "bloom_pressure")),
    StateSpec("light", "shadow_deepening", "shadow_deepening_score", "shadow_mask", ("shadow_coverage_delta", "luma_mean_delta", "shadow_coverage")),
    StateSpec("light", "strobe_rejection", "strobe_rejection_score", "bright_mask", ("luma_delta", "frame_wide_change", "bright_area"), False, 0.55),
    StateSpec("surface", "texture_shimmer", "texture_shimmer_score", "texture_mask", ("texture_delta", "texture_mean", "luma_delta")),
    StateSpec("surface", "reflection_pulse", "reflection_pulse_score", "surface_mask", ("horizontal_band_score", "texture_delta", "luma_delta")),
    StateSpec("surface", "ripple", "ripple_score", "surface_mask", ("motion_mean", "texture_delta", "horizontal_band_score")),
    StateSpec("surface", "grain_movement", "grain_movement_score", "texture_mask", ("texture_delta", "motion_mean", "edge_delta")),
    StateSpec("surface", "metallic_flicker", "metallic_flicker_score", "bright_mask", ("luma_delta", "texture_mean", "saturation_mean")),
    StateSpec("atmosphere", "haze_veil", "haze_veil_score", "haze_mask", ("haze_coverage", "edge_mean", "saturation_mean")),
    StateSpec("atmosphere", "fog_reveal", "fog_reveal_score", "line_mask", ("haze_coverage_delta", "edge_delta_positive", "edge_mean")),
    StateSpec("atmosphere", "scatter_lift", "scatter_lift_score", "haze_mask", ("scatter_lift_delta", "low_edge_luma", "haze_coverage")),
    StateSpec("atmosphere", "depth_fade", "depth_fade_score", "haze_mask", ("vertical_edge_gradient_delta", "edge_mean", "haze_coverage")),
    StateSpec("atmosphere", "edge_loss_under_veil", "edge_loss_under_veil_score", "haze_mask", ("edge_delta_negative", "haze_coverage_delta", "edge_mean")),
    StateSpec("motion", "center_energy_pull", "center_pull_score", "motion_mask", ("center_shift", "center_x", "center_y")),
    StateSpec("motion", "parallax_like_shift", "parallax_shift_score", "motion_mask", ("center_shift", "motion_mean", "edge_mean")),
    StateSpec("motion", "tunnel_compression", "tunnel_compression_score", "motion_mask", ("active_area_delta", "active_area", "center_shift")),
    StateSpec("motion", "motion_pressure_pulse", "motion_pressure_score", "motion_mask", ("motion_delta", "motion_mean", "motion_peak")),
    StateSpec("motion", "frame_wide_state_surge", "frame_wide_surge_score", "active_mask", ("frame_wide_change", "luma_delta", "motion_mean")),
)


def _safe_id(value: str | None, fallback: str = "state_recognition") -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "")).strip("_")
    return safe or fallback


def _feature_index(feature_names: list[str], *names: str) -> int | None:
    for name in names:
        try:
            return feature_names.index(name)
        except ValueError:
            continue
    return None


def _feature_plane(cells: np.ndarray, feature_names: list[str], *names: str, default: float = 0.0) -> np.ndarray:
    index = _feature_index(feature_names, *names)
    if index is None:
        return np.full(cells.shape[:2], default, dtype=np.float32)
    plane = np.asarray(cells[:, :, index], dtype=np.float32)
    if plane.size and float(np.nanmax(plane)) > 2.0:
        plane = plane / 255.0
    return np.clip(plane, 0.0, 1.0).astype(np.float32)


def _norm_series(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    low = float(np.min(finite))
    high = float(np.percentile(finite, 98))
    if high <= low + 1.0e-9:
        return np.zeros_like(arr)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _weighted_center(weight: np.ndarray) -> tuple[float, float]:
    rows, cols = weight.shape
    total = float(np.sum(weight))
    if total <= 1.0e-9:
        return 0.5, 0.5
    yy, xx = np.mgrid[0:rows, 0:cols]
    return (
        float(np.sum((xx / max(cols - 1, 1)) * weight) / total),
        float(np.sum((yy / max(rows - 1, 1)) * weight) / total),
    )


def _bounds(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not bool(np.any(mask)):
        return {
            "grid_xywh": [0, 0, 0, 0],
            "normalized_xywh": [0.0, 0.0, 0.0, 0.0],
            "coverage": 0.0,
        }
    rows, cols = mask.shape
    yy, xx = np.where(mask)
    x0 = int(np.min(xx))
    x1 = int(np.max(xx)) + 1
    y0 = int(np.min(yy))
    y1 = int(np.max(yy)) + 1
    width = max(0, x1 - x0)
    height = max(0, y1 - y0)
    return {
        "grid_xywh": [x0, y0, width, height],
        "normalized_xywh": [
            round(x0 / max(cols, 1), 6),
            round(y0 / max(rows, 1), 6),
            round(width / max(cols, 1), 6),
            round(height / max(rows, 1), 6),
        ],
        "coverage": round(float(np.mean(mask)), 6),
    }


def _horizontal_band_score(luma: np.ndarray) -> float:
    if luma.size == 0:
        return 0.0
    row_mean = np.mean(luma, axis=1)
    return float(np.clip(np.max(row_mean) - np.mean(row_mean), 0.0, 1.0))


def _vertical_edge_gradient(edge: np.ndarray) -> float:
    if edge.size == 0:
        return 0.0
    rows = edge.shape[0]
    top = float(np.mean(edge[: max(1, rows // 3), :]))
    bottom = float(np.mean(edge[max(0, rows - max(1, rows // 3)) :, :]))
    return bottom - top


def _frame_metric_rows(frames: list[dict[str, Any]], feature_names: list[str], *, capture_fps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_luma: np.ndarray | None = None
    previous_edge: np.ndarray | None = None
    previous_texture: np.ndarray | None = None
    previous_motion = 0.0
    previous_center = (0.5, 0.5)
    previous_line_coverage = 0.0
    previous_line_width = 0.0
    previous_bright_area = 0.0
    previous_bloom = 0.0
    previous_shadow = 0.0
    previous_haze = 0.0
    previous_scatter = 0.0
    previous_active_area = 0.0
    previous_gradient = 0.0

    for sampled_index, item in enumerate(frames):
        cells = np.asarray(item["cells"], dtype=np.float32)
        luma = _feature_plane(cells, feature_names, "luma_mean", "hsv_mean_v")
        edge = _feature_plane(cells, feature_names, "edge_density")
        texture = _feature_plane(cells, feature_names, "texture_energy")
        motion = _feature_plane(cells, feature_names, "motion_energy")
        saturation = _feature_plane(cells, feature_names, "saturation_mean", "hsv_mean_s")
        feature_delta = _feature_plane(cells, feature_names, "delta_luma_abs")
        if previous_luma is None:
            luma_delta_plane = np.zeros_like(luma)
            edge_delta_plane = np.zeros_like(edge)
            texture_delta_plane = np.zeros_like(texture)
        else:
            luma_delta_plane = np.maximum(np.abs(luma - previous_luma), feature_delta)
            edge_delta_plane = np.abs(edge - previous_edge)
            texture_delta_plane = np.abs(texture - previous_texture)

        edge_threshold = max(0.12, float(np.percentile(edge, 82))) if edge.size else 0.12
        bright_threshold = max(0.30, float(np.percentile(luma, 88))) if luma.size else 0.30
        texture_threshold = max(0.12, float(np.percentile(texture, 82))) if texture.size else 0.12
        motion_threshold = max(0.10, float(np.percentile(motion, 80))) if motion.size else 0.10
        line_mask = edge >= edge_threshold
        bright_mask = luma >= bright_threshold
        texture_mask = texture >= texture_threshold
        motion_mask = motion >= motion_threshold
        shadow_mask = luma <= 0.18
        haze_mask = (edge <= 0.14) & (saturation <= 0.32) & (luma >= 0.10) & (luma <= 0.78)
        active_mask = (luma_delta_plane >= 0.055) | (edge_delta_plane >= 0.055) | (motion >= motion_threshold) | (texture_delta_plane >= 0.055)
        surface_mask = bright_mask | texture_mask
        center_weight = np.clip(luma_delta_plane + motion + texture_delta_plane + edge_delta_plane, 0.0, None)
        edge_center = _weighted_center(edge * line_mask.astype(np.float32))
        center = _weighted_center(center_weight)

        line_coverage = float(np.mean(line_mask))
        line_width_proxy = float(np.mean(edge[line_mask])) if bool(np.any(line_mask)) else float(np.mean(edge))
        luma_delta = float(np.max(luma_delta_plane)) if luma_delta_plane.size else 0.0
        edge_delta = float(np.mean(edge_delta_plane)) if edge_delta_plane.size else 0.0
        texture_delta = float(np.mean(texture_delta_plane)) if texture_delta_plane.size else 0.0
        motion_mean = float(np.mean(motion)) if motion.size else 0.0
        motion_peak = float(np.max(motion)) if motion.size else 0.0
        bright_area = float(np.mean(bright_mask))
        bloom_pressure = float(np.clip(np.max(luma * 0.62 + luma_delta_plane * 0.26 + edge * 0.12), 0.0, 1.0)) if luma.size else 0.0
        shadow_coverage = float(np.mean(shadow_mask))
        haze_coverage = float(np.mean(haze_mask))
        low_edge = edge < 0.16
        low_edge_luma = float(np.mean(luma[low_edge])) if bool(np.any(low_edge)) else float(np.mean(luma))
        scatter_lift = low_edge_luma
        active_area = float(np.mean(active_mask))
        center_shift = float(math.hypot(center[0] - previous_center[0], center[1] - previous_center[1]))
        edge_center_shift = float(math.hypot(edge_center[0] - previous_center[0], edge_center[1] - previous_center[1]))
        gradient = _vertical_edge_gradient(edge)

        rows.append(
            {
                "frame_index": sampled_index,
                "source_frame_index": int(item.get("global_frame_index", sampled_index)),
                "time_seconds": round(float(item.get("global_frame_index", sampled_index)) / max(capture_fps, 1.0e-9), 6),
                "luma_mean": float(np.mean(luma)) if luma.size else 0.0,
                "luma_peak": float(np.max(luma)) if luma.size else 0.0,
                "luma_delta": luma_delta,
                "luma_mean_delta": 0.0 if previous_luma is None else float(np.mean(luma) - np.mean(previous_luma)),
                "edge_mean": float(np.mean(edge)) if edge.size else 0.0,
                "edge_delta": edge_delta,
                "edge_delta_positive": max(0.0, 0.0 if previous_edge is None else float(np.mean(edge) - np.mean(previous_edge))),
                "edge_delta_negative": max(0.0, 0.0 if previous_edge is None else float(np.mean(previous_edge) - np.mean(edge))),
                "texture_mean": float(np.mean(texture)) if texture.size else 0.0,
                "texture_delta": texture_delta,
                "motion_mean": motion_mean,
                "motion_peak": motion_peak,
                "motion_delta": max(0.0, motion_mean - previous_motion),
                "saturation_mean": float(np.mean(saturation)) if saturation.size else 0.0,
                "line_coverage": line_coverage,
                "line_coverage_delta": line_coverage - previous_line_coverage,
                "line_width_proxy": line_width_proxy,
                "line_width_delta": line_width_proxy - previous_line_width,
                "bright_area": bright_area,
                "bright_area_delta": bright_area - previous_bright_area,
                "bloom_pressure": bloom_pressure,
                "bloom_delta": bloom_pressure - previous_bloom,
                "shadow_coverage": shadow_coverage,
                "shadow_coverage_delta": shadow_coverage - previous_shadow,
                "haze_coverage": haze_coverage,
                "haze_coverage_delta": haze_coverage - previous_haze,
                "previous_haze_coverage": previous_haze,
                "low_edge_luma": low_edge_luma,
                "scatter_lift_delta": scatter_lift - previous_scatter,
                "vertical_edge_gradient_delta": gradient - previous_gradient,
                "horizontal_band_score": _horizontal_band_score(luma),
                "center_x": center[0],
                "center_y": center[1],
                "center_shift": center_shift,
                "edge_center_shift": edge_center_shift,
                "active_area": active_area,
                "active_area_delta": active_area - previous_active_area,
                "frame_wide_change": active_area,
                "line_appears_score": max(0.0, (line_coverage - previous_line_coverage) * 7.0),
                "line_disappears_score": max(0.0, (previous_line_coverage - line_coverage) * 7.0),
                "line_thickness_score": min(1.0, abs(line_width_proxy - previous_line_width) * 4.0 + edge_delta * 3.0),
                "line_breathing_score": min(1.0, abs(line_width_proxy - previous_line_width) * 3.5 + edge_delta * 2.4),
                "ink_crawl_score": min(1.0, edge_center_shift * 2.4 + motion_mean * 0.8),
                "edge_shimmer_score": min(1.0, edge_delta * 6.0 + texture_delta * 2.4 - luma_delta * 0.15),
                "edge_recovery_score": min(1.0, max(0.0, float(np.mean(edge) - (np.mean(previous_edge) if previous_edge is not None else np.mean(edge)))) * 5.0 + previous_haze * 0.5),
                "luminance_score": min(1.0, luma_delta * 1.3 + abs(0.0 if previous_luma is None else float(np.mean(luma) - np.mean(previous_luma))) * 2.0),
                "bloom_score": min(1.0, bloom_pressure * 0.55 + max(0.0, bloom_pressure - previous_bloom) * 2.2),
                "glow_spread_score": min(1.0, max(0.0, bright_area - previous_bright_area) * 5.0 + bloom_pressure * 0.22),
                "shadow_deepening_score": min(1.0, max(0.0, shadow_coverage - previous_shadow) * 5.5 + max(0.0, -(0.0 if previous_luma is None else float(np.mean(luma) - np.mean(previous_luma)))) * 2.5),
                "strobe_rejection_score": min(1.0, luma_delta * 1.1 + active_area * 0.8) if active_area > 0.45 else 0.0,
                "texture_shimmer_score": min(1.0, texture_delta * 5.0 + float(np.mean(texture)) * 0.35),
                "reflection_pulse_score": min(1.0, _horizontal_band_score(luma) * 2.8 + texture_delta * 3.4 + luma_delta * 0.45),
                "ripple_score": min(1.0, motion_mean * 2.0 + texture_delta * 2.0 + _horizontal_band_score(luma) * 1.2),
                "grain_movement_score": min(1.0, texture_delta * 3.4 + motion_mean * 1.2 + edge_delta * 1.0),
                "metallic_flicker_score": min(1.0, luma_delta * 1.4 + float(np.mean(texture)) * 0.55 + float(np.mean(saturation)) * 0.35),
                "haze_veil_score": min(1.0, haze_coverage * 1.25 + max(0.0, haze_coverage - previous_haze) * 1.8),
                "fog_reveal_score": min(1.0, max(0.0, previous_haze - haze_coverage) * 2.0 + max(0.0, float(np.mean(edge) - (np.mean(previous_edge) if previous_edge is not None else np.mean(edge)))) * 4.0),
                "scatter_lift_score": min(1.0, max(0.0, scatter_lift - previous_scatter) * 3.0 + haze_coverage * 0.25),
                "depth_fade_score": min(1.0, abs(gradient - previous_gradient) * 4.0 + haze_coverage * 0.25),
                "edge_loss_under_veil_score": min(1.0, max(0.0, haze_coverage - previous_haze) * 2.0 + max(0.0, (np.mean(previous_edge) if previous_edge is not None else np.mean(edge)) - float(np.mean(edge))) * 5.0),
                "center_pull_score": min(1.0, center_shift * 3.0),
                "parallax_shift_score": min(1.0, center_shift * 2.2 + motion_mean * 0.75 + float(np.mean(edge)) * 0.25),
                "tunnel_compression_score": min(1.0, max(0.0, previous_active_area - active_area) * 3.0 + center_shift * 1.4),
                "motion_pressure_score": min(1.0, max(0.0, motion_mean - previous_motion) * 3.5 + motion_peak * 0.55),
                "frame_wide_surge_score": min(1.0, active_area * 1.3 + luma_delta * 0.5 + motion_mean * 0.45),
                "masks": {
                    "line_mask": line_mask,
                    "bright_mask": bright_mask,
                    "shadow_mask": shadow_mask,
                    "texture_mask": texture_mask,
                    "surface_mask": surface_mask,
                    "haze_mask": haze_mask,
                    "motion_mask": motion_mask,
                    "active_mask": active_mask,
                },
            }
        )
        previous_luma = luma
        previous_edge = edge
        previous_texture = texture
        previous_motion = motion_mean
        previous_center = center
        previous_line_coverage = line_coverage
        previous_line_width = line_width_proxy
        previous_bright_area = bright_area
        previous_bloom = bloom_pressure
        previous_shadow = shadow_coverage
        previous_haze = haze_coverage
        previous_scatter = scatter_lift
        previous_active_area = active_area
        previous_gradient = gradient
    return rows


def _groups_from_score(rows: list[dict[str, Any]], score_key: str, *, threshold: float) -> list[list[int]]:
    values = [float(row.get(score_key, 0.0)) for row in rows]
    if not values or max(values) <= 1.0e-9:
        return []
    norm = _norm_series(values)
    active = [index for index, value in enumerate(norm) if float(value) >= threshold and values[index] > 0.035]
    groups: list[list[int]] = []
    current: list[int] = []
    previous = -99
    for index in active:
        if current and index > previous + 2:
            groups.append(current)
            current = []
        current.append(index)
        previous = index
    if current:
        groups.append(current)
    return groups


def _event_region(rows: list[dict[str, Any]], group: list[int], mask_key: str) -> dict[str, Any]:
    masks = [rows[index]["masks"].get(mask_key) for index in group if rows[index]["masks"].get(mask_key) is not None]
    if not masks:
        return _bounds(np.zeros((0, 0), dtype=bool))
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= np.asarray(mask, dtype=bool)
    return _bounds(union)


def _evidence(rows: list[dict[str, Any]], group: list[int], keys: tuple[str, ...], score_key: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in keys:
        values = [float(rows[index].get(key, 0.0)) for index in group]
        payload[key] = {
            "mean": round(float(np.mean(values)) if values else 0.0, 6),
            "peak": round(float(np.max(np.abs(values))) if values else 0.0, 6),
        }
    scores = [float(rows[index].get(score_key, 0.0)) for index in group]
    payload["score"] = {
        "mean": round(float(np.mean(scores)) if scores else 0.0, 6),
        "peak": round(float(np.max(scores)) if scores else 0.0, 6),
    }
    return payload


def _detect_events(rows: list[dict[str, Any]], *, run_id: str, source_file: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for spec in STATE_SPECS:
        groups = _groups_from_score(rows, spec.score_key, threshold=spec.threshold)
        for group in groups:
            start = rows[group[0]]
            end = rows[group[-1]]
            score_values = [float(rows[index].get(spec.score_key, 0.0)) for index in group]
            confidence = round(float(np.clip(np.mean(_norm_series(score_values)) * 0.6 + np.max(score_values) * 0.4, 0.0, 1.0)), 6)
            if confidence < 0.18:
                continue
            key = f"{spec.state_family}_{spec.state_name}"
            counters[key] = counters.get(key, 0) + 1
            events.append(
                {
                    "state_id": f"{_safe_id(run_id)}_{spec.state_family}_{spec.state_name}_{counters[key]:04d}",
                    "state_name": spec.state_name,
                    "state_family": spec.state_family,
                    "start_frame": int(start["source_frame_index"]),
                    "end_frame": int(end["source_frame_index"]),
                    "start_time": round(float(start["time_seconds"]), 6),
                    "end_time": round(float(end["time_seconds"]), 6),
                    "affected_region": _event_region(rows, group, spec.mask_key),
                    "evidence_metrics": _evidence(rows, group, spec.evidence_keys, spec.score_key),
                    "confidence": confidence,
                    "source_file": source_file,
                    "raw_frames_saved": False,
                    "reusable_transform_candidate": bool(spec.reusable_transform_candidate),
                }
            )
    events.sort(key=lambda item: (item["start_frame"], item["state_family"], item["state_name"]))
    return events


def _summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, int] = {}
    by_state: dict[str, int] = {}
    for event in events:
        by_family[event["state_family"]] = by_family.get(event["state_family"], 0) + 1
        by_state[event["state_name"]] = by_state.get(event["state_name"], 0) + 1
    return {
        "event_count": len(events),
        "by_family": dict(sorted(by_family.items())),
        "by_state": dict(sorted(by_state.items())),
    }


def _build_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    source_file: str,
    source_kind: str,
    capture_fps: float,
) -> dict[str, Any]:
    events = _detect_events(rows, run_id=run_id, source_file=source_file)
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_at_utc": utc_now(),
        "run_id": _safe_id(run_id),
        "source": {
            "source_file": source_file,
            "source_kind": source_kind,
            "capture_fps": capture_fps,
            "sampled_frames": len(rows),
        },
        "state_catalog": list(STATE_CATALOG),
        "summary": _summarize_events(events),
        "events": events,
        "boundary": {
            "recognition_only": True,
            "raw_frames_saved": False,
            "render_started": False,
            "animation_started": False,
            "camera_motion_started": False,
            "objects_created": False,
            "generated_media_is_evidence": False,
        },
    }
    report["report_sha256"] = stable_hash({key: value for key, value in report.items() if key != "report_sha256"})
    return report


def recognize_states_from_manifest(
    manifest_path: str | Path,
    *,
    run_id: str | None = None,
    max_frames: int = 720,
    sample_stride: int = 1,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell_state = manifest.get("cell_state") or {}
    feature_names = list(cell_state.get("feature_names") or [])
    frames = _read_native_tvcells_frames(manifest, max_frames=max_frames, sample_stride=max(1, sample_stride))
    capture_fps = float((manifest.get("config") or {}).get("capture_fps") or manifest.get("fps") or 1.0)
    rows = _frame_metric_rows(frames, feature_names, capture_fps=capture_fps)
    return _build_report(
        rows,
        run_id=run_id or str(manifest.get("run_id") or manifest_path.stem),
        source_file=str(manifest_path),
        source_kind="truevision_manifest_state",
        capture_fps=capture_fps,
    )


def _cell_view(array: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = grid_shape
    height, width = array.shape[:2]
    cell_h = max(1, height // rows)
    cell_w = max(1, width // cols)
    crop_h = cell_h * rows
    crop_w = cell_w * cols
    cropped = array[:crop_h, :crop_w]
    if array.ndim == 3:
        return cropped.reshape(rows, cell_h, cols, cell_w, array.shape[2]).transpose(0, 2, 1, 3, 4)
    return cropped.reshape(rows, cell_h, cols, cell_w).transpose(0, 2, 1, 3)


def _video_frame_to_cells(frame_bgr: np.ndarray, previous_luma: np.ndarray | None, grid_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_cells = _cell_view(frame, grid_shape)
    rgb_mean = (rgb_cells.mean(axis=(2, 3)).astype(np.float32) / 255.0).clip(0.0, 1.0)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    luma = _cell_view(gray, grid_shape).mean(axis=(2, 3)).astype(np.float32)
    hsv = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = (_cell_view(hsv[:, :, 1], grid_shape).mean(axis=(2, 3)).astype(np.float32) / 255.0).clip(0.0, 1.0)
    edge_img = cv2.Canny((gray * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
    edge = _cell_view(edge_img, grid_shape).mean(axis=(2, 3)).astype(np.float32)
    texture = _cell_view(np.abs(gray - cv2.GaussianBlur(gray, (5, 5), 0.0)), grid_shape).mean(axis=(2, 3)).astype(np.float32) * 3.0
    if previous_luma is None:
        delta = np.zeros_like(luma)
    else:
        delta = np.abs(luma - previous_luma)
    cells = np.dstack(
        [
            rgb_mean[:, :, 0],
            rgb_mean[:, :, 1],
            rgb_mean[:, :, 2],
            luma,
            np.clip(luma * 0.15 + edge * 0.1, 0.0, 1.0),
            sat,
            delta,
            edge,
            np.clip(texture, 0.0, 1.0),
            delta,
        ]
    ).astype(np.float32)
    return cells, luma


def recognize_states_from_video(
    video_path: str | Path,
    *,
    run_id: str | None = None,
    max_frames: int = 720,
    sample_stride: int = 1,
    grid_shape: tuple[int, int] = (27, 48),
) -> dict[str, Any]:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[dict[str, Any]] = []
    previous_luma: np.ndarray | None = None
    index = 0
    sample_stride = max(1, int(sample_stride))
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if index % sample_stride == 0:
            cells, previous_luma = _video_frame_to_cells(frame, previous_luma, grid_shape)
            frames.append({"global_frame_index": index, "cells": cells})
        index += 1
    cap.release()
    rows = _frame_metric_rows(frames, FEATURE_NAMES_FOR_VIDEO, capture_fps=fps)
    return _build_report(
        rows,
        run_id=run_id or video_path.stem,
        source_file=str(video_path),
        source_kind="local_video_sampled_state",
        capture_fps=fps,
    )


FEATURE_NAMES_FOR_VIDEO = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "luma_mean",
    "luma_std",
    "saturation_mean",
    "delta_luma_abs",
    "edge_density",
    "texture_energy",
    "motion_energy",
]


def recognize_states_from_jsonl(
    jsonl_path: str | Path,
    *,
    run_id: str | None = None,
    max_rows: int = 720,
) -> dict[str, Any]:
    jsonl_path = Path(jsonl_path)
    frames: list[dict[str, Any]] = []
    previous_luma = np.zeros((1, 1), dtype=np.float32)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if len(frames) >= max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            resonance = row.get("visual_resonance") if isinstance(row.get("visual_resonance"), dict) else {}
            luma_value = float(
                row.get("luma_mean")
                or row.get("screen_energy")
                or resonance.get("mean")
                or resonance.get("luma_mean")
                or 0.0
            )
            edge_value = float(row.get("edge_density") or resonance.get("edge_density") or resonance.get("edge") or 0.0)
            texture_value = float(row.get("texture_energy") or resonance.get("texture_energy") or 0.0)
            motion_value = float(row.get("motion_energy") or resonance.get("motion_energy") or row.get("screen_energy") or 0.0)
            sat_value = float(row.get("saturation_mean") or resonance.get("saturation_mean") or 0.0)
            scale = max(1.0, luma_value, edge_value, texture_value, motion_value, sat_value)
            luma = np.array([[min(1.0, luma_value / scale)]], dtype=np.float32)
            cells = np.dstack(
                [
                    luma,
                    luma,
                    luma,
                    luma,
                    luma * 0.15,
                    np.array([[min(1.0, sat_value / scale)]], dtype=np.float32),
                    np.abs(luma - previous_luma),
                    np.array([[min(1.0, edge_value / scale)]], dtype=np.float32),
                    np.array([[min(1.0, texture_value / scale)]], dtype=np.float32),
                    np.array([[min(1.0, motion_value / scale)]], dtype=np.float32),
                ]
            ).astype(np.float32)
            frames.append({"global_frame_index": int(row.get("frame_number") or row.get("frame_index") or index), "cells": cells})
            previous_luma = luma
    rows = _frame_metric_rows(frames, FEATURE_NAMES_FOR_VIDEO, capture_fps=1.0)
    return _build_report(
        rows,
        run_id=run_id or jsonl_path.stem,
        source_file=str(jsonl_path),
        source_kind="jsonl_state_rows",
        capture_fps=1.0,
    )


def _event_row(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_id": event["state_id"],
        "state_name": event["state_name"],
        "state_family": event["state_family"],
        "start_frame": event["start_frame"],
        "end_frame": event["end_frame"],
        "start_time": event["start_time"],
        "end_time": event["end_time"],
        "affected_region": json.dumps(event["affected_region"], sort_keys=True),
        "evidence_metrics": json.dumps(event["evidence_metrics"], sort_keys=True),
        "confidence": event["confidence"],
        "source_file": event["source_file"],
        "raw_frames_saved": str(bool(event["raw_frames_saved"])).lower(),
        "reusable_transform_candidate": str(bool(event["reusable_transform_candidate"])).lower(),
    }


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# TrueVision State Recognition",
        "",
        "Recognition only. No rendering, no animation, no camera movement, no object creation.",
        "",
        f"- Run id: `{report['run_id']}`",
        f"- Source: `{report['source']['source_file']}`",
        f"- Source kind: `{report['source']['source_kind']}`",
        f"- Sampled frames: `{report['source']['sampled_frames']}`",
        f"- Event count: `{report['summary']['event_count']}`",
        "",
        "## Families",
        "",
    ]
    for family, count in report["summary"]["by_family"].items():
        lines.append(f"- `{family}`: {count}")
    lines.extend(["", "## Events", "", "| state | family | frames | confidence | reusable |", "|---|---:|---:|---:|---:|"])
    for event in report["events"]:
        lines.append(
            f"| `{event['state_name']}` | `{event['state_family']}` | "
            f"{event['start_frame']}-{event['end_frame']} | {event['confidence']:.3f} | "
            f"{str(bool(event['reusable_transform_candidate'])).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- raw_frames_saved: false",
            "- render_started: false",
            "- animation_started: false",
            "- camera_motion_started: false",
            "- objects_created: false",
        ]
    )
    return "\n".join(lines) + "\n"


def write_state_recognition_outputs(report: dict[str, Any], *, output_root: str | Path, run_id: str | None = None) -> dict[str, Any]:
    output_root = Path(output_root)
    run = _safe_id(run_id or report.get("run_id") or "state_recognition")
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{run}_state_recognition_report.json"
    md_path = output_root / f"{run}_state_recognition_summary.md"
    csv_path = output_root / f"{run}_state_recognition_events.csv"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    md_path.write_text(_markdown_summary(report), encoding="utf-8")
    fieldnames = [
        "state_id",
        "state_name",
        "state_family",
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
        "affected_region",
        "evidence_metrics",
        "confidence",
        "source_file",
        "raw_frames_saved",
        "reusable_transform_candidate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in report["events"]:
            writer.writerow(_event_row(event))
    return {
        "schema_version": "truevision_state_recognition_outputs_v1",
        "run_id": run,
        "json_report": str(json_path),
        "markdown_summary": str(md_path),
        "csv_event_table": str(csv_path),
        "event_count": len(report["events"]),
        "raw_frames_saved": False,
    }
