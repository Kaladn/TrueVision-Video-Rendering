from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ATMOSPHERE_CHANNELS = (
    "density",
    "veil_opacity",
    "scatter_bloom",
    "edge_softness",
    "motion_pressure",
    "curl_pressure",
    "occlusion_pressure",
    "droplet_density",
    "droplet_streak",
    "refraction",
    "surface_wetness",
)


ATMOSPHERE_ELEMENTS: tuple[dict[str, Any], ...] = (
    {
        "element_id": "fog_density_field",
        "name": "Fog Density Field",
        "purpose": "Low-contrast volumetric veil that softens background, lifts bloom, and hides distance.",
        "state_channels": [
            "density",
            "veil_opacity",
            "scatter_bloom",
            "edge_softness",
            "motion_pressure",
            "curl_pressure",
            "occlusion_pressure",
        ],
        "six_one_six_targets": ["density", "scatter_bloom", "motion_pressure", "edge_softness"],
        "render_contract": {
            "composition": "existing_scene_atmosphere",
            "adds_geometry": False,
            "preferred_motion": "slow_lateral_breathing",
            "source_learning": "native_tvcells_density_windows",
        },
        "boundary": {
            "state_first_pixels_last": True,
            "generated_media_is_evidence": False,
            "raw_frames_required": False,
        },
    },
    {
        "element_id": "mist_veil_field",
        "name": "Mist Veil Field",
        "purpose": "Thinner wet air veil with higher transparency, soft light scatter, and subtle foreground lift.",
        "state_channels": ["density", "veil_opacity", "scatter_bloom", "edge_softness", "motion_pressure"],
        "six_one_six_targets": ["density", "veil_opacity", "motion_pressure"],
        "render_contract": {
            "composition": "thin_atmosphere_layer",
            "adds_geometry": False,
            "preferred_motion": "near_static_drift",
            "source_learning": "native_tvcells_luma_edge_motion",
        },
        "boundary": {
            "state_first_pixels_last": True,
            "generated_media_is_evidence": False,
            "raw_frames_required": False,
        },
    },
    {
        "element_id": "cloud_volume_field",
        "name": "Cloud Volume Field",
        "purpose": "Large soft volumes with slow mass travel, internal density pockets, and shadowed edges.",
        "state_channels": [
            "density",
            "scatter_bloom",
            "edge_softness",
            "motion_pressure",
            "curl_pressure",
            "occlusion_pressure",
        ],
        "six_one_six_targets": ["density", "curl_pressure", "occlusion_pressure"],
        "render_contract": {
            "composition": "large_volume_state_layer",
            "adds_geometry": False,
            "preferred_motion": "slow_mass_translation",
            "source_learning": "native_tvcells_density_motion_curl",
        },
        "boundary": {
            "state_first_pixels_last": True,
            "generated_media_is_evidence": False,
            "raw_frames_required": False,
        },
    },
    {
        "element_id": "rain_glass_field",
        "name": "Rain Drops On Glass Field",
        "purpose": "Glass-surface wetness, droplets, streaks, refraction, and beat/pressure-driven runoff.",
        "state_channels": [
            "droplet_density",
            "droplet_streak",
            "refraction",
            "surface_wetness",
            "edge_softness",
            "motion_pressure",
        ],
        "six_one_six_targets": ["droplet_density", "droplet_streak", "refraction", "motion_pressure"],
        "render_contract": {
            "composition": "foreground_glass_surface",
            "adds_geometry": False,
            "preferred_motion": "gravity_streak_and_merge",
            "source_learning": "native_tvcells_edge_texture_motion",
        },
        "boundary": {
            "state_first_pixels_last": True,
            "generated_media_is_evidence": False,
            "raw_frames_required": False,
        },
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _safe_run_id(value: str | None) -> str:
    stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "")).strip("_")
    return stem or f"atmosphere_weather_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _element_by_id(element_id: str) -> dict[str, Any]:
    elements = {item["element_id"]: item for item in ATMOSPHERE_ELEMENTS}
    if element_id not in elements:
        raise ValueError(f"unknown atmosphere element: {element_id}")
    return deepcopy(elements[element_id])


def list_atmosphere_elements() -> list[dict[str, Any]]:
    return [deepcopy(element) for element in ATMOSPHERE_ELEMENTS]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low = float(np.min(finite))
    high = float(np.percentile(finite, 95))
    if high <= low + 1.0e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _feature_index(feature_names: list[str], name: str) -> int | None:
    try:
        return feature_names.index(name)
    except ValueError:
        return None


def _mean_feature(frame: np.ndarray, feature_names: list[str], name: str, default: float = 0.0) -> float:
    index = _feature_index(feature_names, name)
    if index is None:
        return default
    return float(np.mean(frame[:, :, index]))


def _read_native_tvcells_frames(
    manifest: dict[str, Any],
    *,
    max_frames: int,
    sample_stride: int,
) -> list[dict[str, Any]]:
    cell_state = manifest.get("cell_state") or {}
    if cell_state.get("format") != "tvcells_f32le_v1":
        raise ValueError("capture manifest must describe tvcells_f32le_v1 cell state")
    feature_names = list(cell_state.get("feature_names") or [])
    if not feature_names:
        raise ValueError("capture manifest is missing cell feature names")
    frames: list[dict[str, Any]] = []
    global_frame_index = 0
    for chunk in cell_state.get("chunks") or []:
        path = Path(str(chunk.get("path") or ""))
        if not path.exists():
            raise FileNotFoundError(path)
        chunk_frames = int(chunk.get("frames") or 0)
        rows, cols = [int(value) for value in chunk.get("grid_shape") or [0, 0]]
        feature_count = int(chunk.get("feature_count") or len(feature_names))
        if chunk_frames <= 0 or rows <= 0 or cols <= 0 or feature_count <= 0:
            continue
        frame_byte_count = rows * cols * feature_count * 4
        with path.open("rb") as handle:
            for local_frame_index in range(chunk_frames):
                if len(frames) >= max_frames:
                    return frames
                should_sample = global_frame_index % max(1, sample_stride) == 0
                if not should_sample:
                    handle.seek(frame_byte_count, 1)
                    global_frame_index += 1
                    continue
                raw = handle.read(frame_byte_count)
                if len(raw) != frame_byte_count:
                    return frames
                cells = np.frombuffer(raw, dtype="<f4").reshape(rows, cols, feature_count)
                frames.append(
                    {
                        "global_frame_index": global_frame_index,
                        "chunk_id": int(chunk.get("chunk_id") or 0),
                        "chunk_frame_index": local_frame_index,
                        "cells": cells,
                    }
                )
                global_frame_index += 1
    return frames


def _derive_weather_frames(
    frames: list[dict[str, Any]],
    *,
    feature_names: list[str],
    capture_fps: float,
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, float]] = []
    for item in frames:
        cells = item["cells"]
        luma_mean = _mean_feature(cells, feature_names, "luma_mean")
        luma_std = _mean_feature(cells, feature_names, "luma_std")
        edge = _mean_feature(cells, feature_names, "edge_density")
        texture = _mean_feature(cells, feature_names, "texture_energy")
        motion = _mean_feature(cells, feature_names, "motion_energy")
        saturation = _mean_feature(cells, feature_names, "saturation_mean")
        delta = _mean_feature(cells, feature_names, "delta_luma_abs")
        raw_rows.append(
            {
                "source_frame_index": float(item["global_frame_index"]),
                "time_seconds": float(item["global_frame_index"]) / max(capture_fps, 1.0e-9),
                "luma_mean": luma_mean,
                "luma_std": luma_std,
                "edge_density": edge,
                "texture_energy": texture,
                "motion_energy": motion,
                "saturation_mean": saturation,
                "delta_luma_abs": delta,
            }
        )
    if not raw_rows:
        return []
    arrays = {key: _normalize(np.array([row[key] for row in raw_rows], dtype=np.float32)) for key in raw_rows[0] if key not in {"source_frame_index", "time_seconds"}}
    weather_frames: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        luma = float(arrays["luma_mean"][index])
        luma_std = float(arrays["luma_std"][index])
        edge = float(arrays["edge_density"][index])
        texture = float(arrays["texture_energy"][index])
        motion = float(arrays["motion_energy"][index])
        saturation = float(arrays["saturation_mean"][index])
        delta = float(arrays["delta_luma_abs"][index])
        density = float(np.clip(0.38 * luma_std + 0.24 * texture + 0.2 * (1.0 - edge) + 0.12 * saturation + 0.06 * luma, 0.0, 1.0))
        weather_frames.append(
            {
                "frame_index": index,
                "source_frame_index": int(raw["source_frame_index"]),
                "time_seconds": round(float(raw["time_seconds"]), 6),
                "density": round(density, 6),
                "veil_opacity": round(float(np.clip(density * 0.74 + luma * 0.2, 0.0, 1.0)), 6),
                "scatter_bloom": round(float(np.clip(luma * 0.5 + luma_std * 0.3 + (1.0 - edge) * 0.2, 0.0, 1.0)), 6),
                "edge_softness": round(float(np.clip(1.0 - edge * 0.82 + density * 0.18, 0.0, 1.0)), 6),
                "motion_pressure": round(float(np.clip(motion * 0.76 + delta * 0.24, 0.0, 1.0)), 6),
                "curl_pressure": round(float(np.clip(texture * 0.58 + motion * 0.42, 0.0, 1.0)), 6),
                "occlusion_pressure": round(float(np.clip(density * 0.65 + (1.0 - edge) * 0.35, 0.0, 1.0)), 6),
                "droplet_density": round(float(np.clip(edge * 0.42 + texture * 0.28 + luma_std * 0.3, 0.0, 1.0)), 6),
                "droplet_streak": round(float(np.clip(motion * 0.66 + delta * 0.34, 0.0, 1.0)), 6),
                "refraction": round(float(np.clip(edge * 0.4 + luma_std * 0.35 + saturation * 0.25, 0.0, 1.0)), 6),
                "surface_wetness": round(float(np.clip(luma_std * 0.5 + saturation * 0.2 + texture * 0.3, 0.0, 1.0)), 6),
            }
        )
    return weather_frames


def _mean_window(frames: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    if not frames:
        return {f"{key}_mean": 0.0 for key in keys}
    return {f"{key}_mean": round(float(np.mean([float(frame[key]) for frame in frames])), 6) for key in keys}


def _build_six_one_six_windows(
    weather_frames: list[dict[str, Any]],
    *,
    element: dict[str, Any],
    radius: int,
    max_windows: int,
) -> dict[str, Any]:
    keys = [key for key in element["six_one_six_targets"] if weather_frames and key in weather_frames[0]]
    windows: list[dict[str, Any]] = []
    for center_index in range(radius, max(radius, len(weather_frames) - radius)):
        if len(windows) >= max_windows:
            break
        prior = weather_frames[center_index - radius : center_index]
        center = [weather_frames[center_index]]
        future = weather_frames[center_index + 1 : center_index + radius + 1]
        if len(prior) < radius or len(future) < radius:
            continue
        windows.append(
            {
                "center_frame_index": center_index,
                "center_time_seconds": weather_frames[center_index]["time_seconds"],
                "prior": _mean_window(prior, keys),
                "center": _mean_window(center, keys),
                "future": _mean_window(future, keys),
            }
        )
    return {
        "radius": radius,
        "shape": f"{radius}-1-{radius}",
        "targets": keys,
        "window_count": len(windows),
        "windows": windows,
    }


def build_atmosphere_profile_from_native_capture(
    manifest_path: str | Path,
    *,
    element_id: str = "fog_density_field",
    max_frames: int = 180,
    sample_stride: int = 1,
    radius: int = 6,
    max_windows: int = 200,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    element = _element_by_id(element_id)
    cell_state = manifest.get("cell_state") or {}
    feature_names = list(cell_state.get("feature_names") or [])
    frames = _read_native_tvcells_frames(manifest, max_frames=max_frames, sample_stride=sample_stride)
    capture_fps = float((manifest.get("config") or {}).get("capture_fps") or 1.0)
    weather_frames = _derive_weather_frames(frames, feature_names=feature_names, capture_fps=capture_fps)
    summary = {
        "sampled_frames": len(weather_frames),
        "density_mean": round(float(np.mean([frame["density"] for frame in weather_frames])) if weather_frames else 0.0, 6),
        "density_max": round(float(np.max([frame["density"] for frame in weather_frames])) if weather_frames else 0.0, 6),
        "motion_mean": round(float(np.mean([frame["motion_pressure"] for frame in weather_frames])) if weather_frames else 0.0, 6),
        "edge_softness_mean": round(float(np.mean([frame["edge_softness"] for frame in weather_frames])) if weather_frames else 0.0, 6),
    }
    profile = {
        "schema_version": "truevision_atmosphere_profile_v1",
        "created_at_utc": _utc_now(),
        "element_id": element["element_id"],
        "source": {
            "manifest_json": str(manifest_path),
            "run_id": manifest.get("run_id"),
            "records_jsonl": manifest.get("records_jsonl"),
            "capture_fps": capture_fps,
            "grid_size_xy": (manifest.get("config") or {}).get("grid_size_xy"),
            "raw_frame_saved": bool((manifest.get("boundary") or {}).get("raw_frame_saved", False)),
        },
        "sampled_frames": len(weather_frames),
        "weather_frames": weather_frames,
        "six_one_six": _build_six_one_six_windows(weather_frames, element=element, radius=radius, max_windows=max_windows),
        "summary": summary,
        "boundary": {
            "learned_from_state": True,
            "raw_frames_required": False,
            "raw_frames_saved": False,
            "semantic_detection": False,
            "state_first_pixels_last": True,
        },
    }
    profile["profile_sha256"] = _stable_hash({key: value for key, value in profile.items() if key != "profile_sha256"})
    return profile


def _default_element_ids(element_ids: list[str] | tuple[str, ...] | None) -> list[str]:
    return list(element_ids or [element["element_id"] for element in ATMOSPHERE_ELEMENTS])


def build_atmosphere_toolset(
    *,
    storage_root: str | Path,
    run_id: str | None = None,
    capture_manifest: str | Path | None = None,
    element_ids: list[str] | tuple[str, ...] | None = None,
    max_profile_frames: int = 180,
) -> dict[str, Any]:
    storage_root = Path(storage_root)
    run = _safe_run_id(run_id)
    for lane in ("artifacts", "manifests", "receipts", "templates"):
        (storage_root / lane).mkdir(parents=True, exist_ok=True)
    weather_root = storage_root / "artifacts" / "weather"
    weather_root.mkdir(parents=True, exist_ok=True)
    elements = [_element_by_id(element_id) for element_id in _default_element_ids(element_ids)]
    capture_profile_path: Path | None = None
    capture_profile: dict[str, Any] | None = None
    if capture_manifest:
        capture_profile = build_atmosphere_profile_from_native_capture(
            capture_manifest,
            element_id=elements[0]["element_id"],
            max_frames=max_profile_frames,
        )
        capture_profile_path = weather_root / f"{run}_capture_profile.json"
        capture_profile_path.write_text(json.dumps(capture_profile, indent=2, allow_nan=False), encoding="utf-8")
    template = {
        "schema_version": "truevision_atmosphere_toolset_v1",
        "template_id": run,
        "created_at_utc": _utc_now(),
        "purpose": "Reusable TrueVision atmosphere/weather state tools for fog, mist, clouds, and rain on glass.",
        "elements": elements,
        "source_profile": str(capture_profile_path) if capture_profile_path else None,
        "render_law": "state/grid/primitive first; pixels only at final rasterization",
        "timeline_controls": {
            "six_one_six_required_for_honing": True,
            "manual_recalibration_targets": list(ATMOSPHERE_CHANNELS),
        },
        "boundary": {
            "audio_video_only": True,
            "semantic_detection": False,
            "raw_frame_saved": False,
            "generated_media_is_evidence": False,
        },
    }
    template_path = storage_root / "templates" / f"{run}_atmosphere_toolset.json"
    template_path.write_text(json.dumps(template, indent=2, allow_nan=False), encoding="utf-8")
    manifest = {
        "schema_version": "truevision_atmosphere_toolset_manifest_v1",
        "created_at_utc": _utc_now(),
        "run_id": run,
        "toolset": {
            "element_count": len(elements),
            "elements": [element["element_id"] for element in elements],
            "channels": list(ATMOSPHERE_CHANNELS),
            "template_json": str(template_path),
            "capture_profile_json": str(capture_profile_path) if capture_profile_path else None,
        },
        "source_capture": {
            "manifest_json": str(capture_manifest) if capture_manifest else None,
            "profile_sha256": capture_profile.get("profile_sha256") if capture_profile else None,
        },
        "boundary": {
            "raw_frame_saved": False,
            "raw_audio_saved": False,
            "generated_media_is_evidence": False,
            "state_first_pixels_last": True,
        },
    }
    manifest["manifest_sha256"] = _stable_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    manifest_path = storage_root / "manifests" / f"{run}_atmosphere_toolset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_id": run,
        "element_count": len(elements),
        "template_json": str(template_path),
        "manifest_json": str(manifest_path),
        "capture_profile_json": str(capture_profile_path) if capture_profile_path else None,
        "manifest_sha256": manifest["manifest_sha256"],
    }
