from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now
from truevision_runtime.state_patterns.atmosphere_weather import _read_native_tvcells_frames


CREATION_CHANNELS = (
    "shape_behavior",
    "growth_decay",
    "edge_softness",
    "density_opacity",
    "bloom_intensity",
    "occlusion_behavior",
    "rhythm_pulse",
    "transition_behavior",
    "camera_relation",
    "renderer_binding",
)


def _safe_id(value: str | None, fallback: str = "element_creation_profile") -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "")).strip("_")
    return safe or fallback


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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


def _feature_plane(frame: np.ndarray, feature_names: list[str], name: str, default: float = 0.0) -> np.ndarray:
    index = _feature_index(feature_names, name)
    if index is None:
        return np.full(frame.shape[:2], default, dtype=np.float32)
    return np.asarray(frame[:, :, index], dtype=np.float32)


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


def _weighted_center_xy(weight: np.ndarray) -> list[float]:
    rows, cols = weight.shape
    total = float(np.sum(weight))
    if total <= 1.0e-9:
        return [0.5, 0.5]
    yy, xx = np.mgrid[0:rows, 0:cols]
    denom_x = max(1, cols - 1)
    denom_y = max(1, rows - 1)
    center_x = float(np.sum((xx / denom_x) * weight) / total)
    center_y = float(np.sum((yy / denom_y) * weight) / total)
    return [round(center_x, 6), round(center_y, 6)]


def _raw_frame_rows(frames: list[dict[str, Any]], feature_names: list[str], capture_fps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sampled_index, item in enumerate(frames):
        cells = item["cells"]
        luma = _feature_plane(cells, feature_names, "luma_mean")
        motion = _feature_plane(cells, feature_names, "motion_energy")
        texture = _feature_plane(cells, feature_names, "texture_energy")
        edge = _feature_plane(cells, feature_names, "edge_density")
        saturation = _feature_plane(cells, feature_names, "saturation_mean")
        weight = np.clip(luma + motion + texture, 0.0, None)
        hot_threshold = float(np.percentile(luma, 75)) if luma.size else 0.0
        rows.append(
            {
                "frame_index": sampled_index,
                "source_frame_index": int(item["global_frame_index"]),
                "time_seconds": round(float(item["global_frame_index"]) / max(capture_fps, 1.0e-9), 6),
                "rgb_mean": [
                    _mean_feature(cells, feature_names, "rgb_mean_r"),
                    _mean_feature(cells, feature_names, "rgb_mean_g"),
                    _mean_feature(cells, feature_names, "rgb_mean_b"),
                ],
                "luma_mean": _mean_feature(cells, feature_names, "luma_mean"),
                "luma_std": _mean_feature(cells, feature_names, "luma_std"),
                "saturation_mean": _mean_feature(cells, feature_names, "saturation_mean"),
                "delta_luma_abs": _mean_feature(cells, feature_names, "delta_luma_abs"),
                "edge_density": _mean_feature(cells, feature_names, "edge_density"),
                "texture_energy": _mean_feature(cells, feature_names, "texture_energy"),
                "motion_energy": _mean_feature(cells, feature_names, "motion_energy"),
                "center_xy": _weighted_center_xy(weight),
                "hot_area_ratio": float(np.mean(luma >= hot_threshold)) if luma.size else 0.0,
                "local_contrast_mean": float(np.mean(np.abs(luma - np.mean(luma)))) if luma.size else 0.0,
                "spatial_complexity": float(np.clip(np.mean(edge) * 0.46 + np.mean(texture) * 0.38 + np.mean(saturation) * 0.16, 0.0, 1.0)),
            }
        )
    return rows


def _derive_creation_frames(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not raw_rows:
        return []
    scalar_keys = [
        "luma_mean",
        "luma_std",
        "saturation_mean",
        "delta_luma_abs",
        "edge_density",
        "texture_energy",
        "motion_energy",
        "hot_area_ratio",
        "local_contrast_mean",
        "spatial_complexity",
    ]
    normalized = {key: _normalize(np.array([row[key] for row in raw_rows], dtype=np.float32)) for key in scalar_keys}
    creation_frames: list[dict[str, Any]] = []
    previous_density = 0.0
    previous_center = raw_rows[0]["center_xy"]
    for index, row in enumerate(raw_rows):
        luma = float(normalized["luma_mean"][index])
        luma_std = float(normalized["luma_std"][index])
        saturation = float(normalized["saturation_mean"][index])
        delta = float(normalized["delta_luma_abs"][index])
        edge = float(normalized["edge_density"][index])
        texture = float(normalized["texture_energy"][index])
        motion = float(normalized["motion_energy"][index])
        hot_area = float(normalized["hot_area_ratio"][index])
        contrast = float(normalized["local_contrast_mean"][index])
        complexity = float(normalized["spatial_complexity"][index])
        raw_motion = float(row["motion_energy"])
        motion_absolute = float(np.clip(raw_motion if raw_motion <= 1.0 else raw_motion / 255.0, 0.0, 1.0))
        motion_pressure = float(np.clip(max(motion, motion_absolute), 0.0, 1.0))
        density = float(np.clip(0.3 * luma_std + 0.24 * texture + 0.18 * hot_area + 0.16 * contrast + 0.12 * motion, 0.0, 1.0))
        bloom = float(np.clip(0.4 * luma + 0.24 * delta + 0.2 * saturation + 0.16 * luma_std, 0.0, 1.0))
        edge_softness = float(np.clip(1.0 - edge * 0.72 + density * 0.2, 0.0, 1.0))
        occlusion = float(np.clip(density * 0.55 + texture * 0.25 + (1.0 - edge) * 0.2, 0.0, 1.0))
        growth = max(0.0, density - previous_density)
        decay = max(0.0, previous_density - density)
        center = row["center_xy"]
        center_velocity = [round(center[0] - previous_center[0], 6), round(center[1] - previous_center[1], 6)]
        rgb = [round(float(value), 6) for value in row["rgb_mean"]]
        creation_frames.append(
            {
                "frame_index": row["frame_index"],
                "source_frame_index": row["source_frame_index"],
                "time_seconds": row["time_seconds"],
                "center_xy": center,
                "center_velocity_xy": center_velocity,
                "shape_complexity": round(complexity, 6),
                "growth_pressure": round(growth, 6),
                "decay_pressure": round(decay, 6),
                "edge_softness": round(edge_softness, 6),
                "density_opacity": round(density, 6),
                "bloom_intensity": round(bloom, 6),
                "occlusion_pressure": round(occlusion, 6),
                "rhythm_pulse": round(float(np.clip(0.55 * delta + 0.45 * motion, 0.0, 1.0)), 6),
                "motion_pressure": round(motion_pressure, 6),
                "motion_absolute": round(motion_absolute, 6),
                "color_pressure": {
                    "rgb_mean": rgb,
                    "warmth": round(float(np.clip(rgb[0] - rgb[2], -1.0, 1.0)), 6),
                    "saturation": round(saturation, 6),
                },
            }
        )
        previous_density = density
        previous_center = center
    return creation_frames


def _mean_window(frames: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    if not frames:
        return {f"{key}_mean": 0.0 for key in keys}
    return {f"{key}_mean": round(float(np.mean([float(frame[key]) for frame in frames])), 6) for key in keys}


def _build_six_one_six(creation_frames: list[dict[str, Any]], *, radius: int, max_windows: int) -> dict[str, Any]:
    keys = [
        "shape_complexity",
        "growth_pressure",
        "decay_pressure",
        "edge_softness",
        "density_opacity",
        "bloom_intensity",
        "occlusion_pressure",
        "rhythm_pulse",
        "motion_pressure",
    ]
    windows: list[dict[str, Any]] = []
    for center_index in range(radius, max(radius, len(creation_frames) - radius)):
        if len(windows) >= max_windows:
            break
        prior = creation_frames[center_index - radius : center_index]
        future = creation_frames[center_index + 1 : center_index + radius + 1]
        if len(prior) < radius or len(future) < radius:
            continue
        windows.append(
            {
                "center_frame_index": center_index,
                "center_time_seconds": creation_frames[center_index]["time_seconds"],
                "prior": _mean_window(prior, keys),
                "center": _mean_window([creation_frames[center_index]], keys),
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


def _series(frames: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(frame[key]) for frame in frames], dtype=np.float32)


def _confidence(sampled_frames: int, window_count: int) -> float:
    frame_score = min(1.0, sampled_frames / 60.0)
    window_score = min(1.0, window_count / 8.0)
    return round(float(0.45 + 0.35 * frame_score + 0.2 * window_score), 6)


def _classify_transition(frames: list[dict[str, Any]]) -> str:
    if len(frames) < 2:
        return "insufficient_history"
    density = _series(frames, "density_opacity")
    pulse = _series(frames, "rhythm_pulse")
    motion = _series(frames, "motion_pressure")
    if float(np.max(pulse)) >= 0.78 and float(np.max(motion)) >= 0.55:
        return "spike_and_decay"
    if float(np.mean(motion)) >= 0.48 and float(np.std(density)) <= 0.22:
        return "continuous_flow"
    if float(np.std(density)) >= 0.28:
        return "volatile_growth_decay"
    return "soft_state_drift"


def _build_creation_signature(element_id: str, frames: list[dict[str, Any]], six_one_six: dict[str, Any]) -> dict[str, Any]:
    if not frames:
        return {
            "channels": list(CREATION_CHANNELS),
            "confidence": 0.0,
            "status": "empty_profile",
        }
    centers = np.array([frame["center_xy"] for frame in frames], dtype=np.float32)
    drift = centers[-1] - centers[0] if len(centers) > 1 else np.zeros(2, dtype=np.float32)
    center_mean = np.mean(centers, axis=0) if len(centers) else np.array([0.5, 0.5], dtype=np.float32)
    complexity = _series(frames, "shape_complexity")
    density = _series(frames, "density_opacity")
    bloom = _series(frames, "bloom_intensity")
    edge_softness = _series(frames, "edge_softness")
    occlusion = _series(frames, "occlusion_pressure")
    growth = _series(frames, "growth_pressure")
    decay = _series(frames, "decay_pressure")
    pulse = _series(frames, "rhythm_pulse")
    motion = _series(frames, "motion_pressure")
    motion_absolute = _series(frames, "motion_absolute")
    peak_threshold = float(np.percentile(pulse, 85)) if pulse.size else 1.0
    peak_indices = [int(index) for index, value in enumerate(pulse) if float(value) >= peak_threshold and float(value) > 0.0]
    transition_kind = _classify_transition(frames)
    return {
        "channels": list(CREATION_CHANNELS),
        "shape_behavior": {
            "complexity_mean": round(float(np.mean(complexity)), 6),
            "complexity_max": round(float(np.max(complexity)), 6),
            "stability": round(float(np.clip(1.0 - np.std(complexity), 0.0, 1.0)), 6),
            "center_mean_xy": [round(float(center_mean[0]), 6), round(float(center_mean[1]), 6)],
            "center_drift_xy": [round(float(drift[0]), 6), round(float(drift[1]), 6)],
            "scale_pressure_mean": round(float(np.mean(density)), 6),
        },
        "growth_decay": {
            "growth_mean": round(float(np.mean(growth)), 6),
            "growth_max": round(float(np.max(growth)), 6),
            "decay_mean": round(float(np.mean(decay)), 6),
            "decay_max": round(float(np.max(decay)), 6),
            "volatility": round(float(np.clip(np.std(density) + np.std(bloom), 0.0, 1.0)), 6),
        },
        "edge_softness": {
            "mean": round(float(np.mean(edge_softness)), 6),
            "minimum": round(float(np.min(edge_softness)), 6),
            "maximum": round(float(np.max(edge_softness)), 6),
        },
        "density_opacity": {
            "mean": round(float(np.mean(density)), 6),
            "maximum": round(float(np.max(density)), 6),
        },
        "bloom_intensity": {
            "mean": round(float(np.mean(bloom)), 6),
            "maximum": round(float(np.max(bloom)), 6),
        },
        "occlusion_behavior": {
            "mean": round(float(np.mean(occlusion)), 6),
            "maximum": round(float(np.max(occlusion)), 6),
        },
        "rhythm_pulse": {
            "mean": round(float(np.mean(pulse)), 6),
            "maximum": round(float(np.max(pulse)), 6),
            "peak_count": len(peak_indices),
            "peak_frame_indices": peak_indices[:24],
        },
        "transition_behavior": {
            "kind": transition_kind,
            "motion_mean": round(float(np.mean(motion)), 6),
            "motion_max": round(float(np.max(motion)), 6),
            "motion_abs_mean": round(float(np.mean(motion_absolute)), 6),
            "motion_abs_max": round(float(np.max(motion_absolute)), 6),
            "six_one_six_windows": int(six_one_six["window_count"]),
        },
        "camera_relation": {
            "relation": "source_region_locked",
            "center_drift_xy": [round(float(drift[0]), 6), round(float(drift[1]), 6)],
            "camera_motion_claim": "not_estimated_from_semantics",
        },
        "renderer_binding": {
            "element_id": element_id,
            "mode": "state_creation_signature",
            "drive_channels": [
                "density_opacity",
                "edge_softness",
                "bloom_intensity",
                "occlusion_pressure",
                "rhythm_pulse",
                "center_drift_xy",
            ],
            "law": "create from behavior profile; do not replay teacher chunks",
        },
        "confidence": _confidence(len(frames), int(six_one_six["window_count"])),
    }


def _profile_hash(profile: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in profile.items() if key != "profile_sha256"})


def _verify_profile_file(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("profile_sha256") == _profile_hash(payload)


def build_element_creation_profile_from_native_capture(
    manifest_path: str | Path,
    *,
    element_id: str,
    max_frames: int = 180,
    sample_stride: int = 1,
    radius: int = 6,
    max_windows: int = 200,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell_state = manifest.get("cell_state") or {}
    feature_names = list(cell_state.get("feature_names") or [])
    frames = _read_native_tvcells_frames(manifest, max_frames=max_frames, sample_stride=sample_stride)
    capture_fps = float((manifest.get("config") or {}).get("capture_fps") or 1.0)
    raw_rows = _raw_frame_rows(frames, feature_names, capture_fps)
    creation_frames = _derive_creation_frames(raw_rows)
    six_one_six = _build_six_one_six(creation_frames, radius=radius, max_windows=max_windows)
    signature = _build_creation_signature(element_id, creation_frames, six_one_six)
    source_chunks = [
        {
            "path": str(chunk.get("path") or ""),
            "frames": int(chunk.get("frames") or 0),
            "grid_shape": chunk.get("grid_shape"),
            "feature_count": int(chunk.get("feature_count") or 0),
        }
        for chunk in cell_state.get("chunks") or []
    ]
    profile = {
        "schema_version": "truevision_element_creation_profile_v1",
        "created_at_utc": utc_now(),
        "element_id": element_id,
        "source": {
            "manifest_json": str(manifest_path),
            "manifest_sha256": _file_sha256(manifest_path),
            "run_id": manifest.get("run_id"),
            "records_jsonl": manifest.get("records_jsonl"),
            "capture_fps": capture_fps,
            "grid_size_xy": (manifest.get("config") or {}).get("grid_size_xy"),
            "capture_region": (manifest.get("config") or {}).get("capture_region"),
            "raw_frame_saved": bool((manifest.get("boundary") or {}).get("raw_frame_saved", False)),
            "teacher_chunks": source_chunks,
        },
        "sampled_frames": len(creation_frames),
        "creation_frames": creation_frames,
        "six_one_six": six_one_six,
        "creation_signature": signature,
        "retention": {
            "durable_teacher_state": False,
            "durable_output": "compact_creation_profile",
            "teacher_state_deletable_after_profile_verified": True,
            "raw_frames_required": False,
            "raw_frames_saved": False,
        },
        "boundary": {
            "learned_from_state": True,
            "semantic_detection": False,
            "state_creation_not_replay": True,
            "state_first_pixels_last": True,
            "generated_media_is_evidence": False,
        },
    }
    profile["profile_sha256"] = _profile_hash(profile)
    return profile


def _teacher_paths_from_manifest(manifest: dict[str, Any], *, purge_records_jsonl: bool) -> list[Path]:
    paths: list[Path] = []
    for chunk in (manifest.get("cell_state") or {}).get("chunks") or []:
        path = Path(str(chunk.get("path") or ""))
        if path:
            paths.append(path)
    if purge_records_jsonl and manifest.get("records_jsonl"):
        paths.append(Path(str(manifest["records_jsonl"])))
    return paths


def _purge_teacher_state(
    *,
    manifest_path: Path,
    profile_path: Path,
    purge_records_jsonl: bool,
) -> dict[str, Any]:
    profile_verified = _verify_profile_file(profile_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if not profile_verified:
        return {
            "schema_version": "truevision_teacher_state_purge_report_v1",
            "created_at_utc": utc_now(),
            "status": "blocked",
            "profile_verified_before_purge": False,
            "deleted_file_count": 0,
            "deleted_bytes": 0,
            "deleted_files": [],
            "skipped_files": [{"path": str(profile_path), "reason": "profile_hash_verification_failed"}],
        }
    for path in _teacher_paths_from_manifest(manifest, purge_records_jsonl=purge_records_jsonl):
        if not path.exists():
            skipped.append({"path": str(path), "reason": "missing"})
            continue
        if not path.is_file():
            skipped.append({"path": str(path), "reason": "not_file"})
            continue
        size = path.stat().st_size
        path.unlink()
        deleted.append({"path": str(path), "size_bytes": size})
    return {
        "schema_version": "truevision_teacher_state_purge_report_v1",
        "created_at_utc": utc_now(),
        "status": "purged",
        "profile_verified_before_purge": True,
        "deleted_file_count": len(deleted),
        "deleted_bytes": sum(int(item["size_bytes"]) for item in deleted),
        "deleted_files": deleted,
        "skipped_files": skipped,
        "retained_outputs": [str(profile_path)],
    }


def write_element_creation_profile_from_capture(args: dict[str, Any], *, storage_root: Path) -> dict[str, Any]:
    manifest_path = Path(str(args.get("manifest") or args.get("manifest_json") or ""))
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))
    element_id = str(args.get("element_id") or "")
    if not element_id:
        raise ValueError("element_id is required")
    run_id = _safe_id(str(args.get("run_id") or f"{element_id}_creation_profile"))
    profile = build_element_creation_profile_from_native_capture(
        manifest_path,
        element_id=element_id,
        max_frames=int(args.get("max_frames") or 180),
        sample_stride=int(args.get("sample_stride") or 1),
        radius=int(args.get("radius") or 6),
        max_windows=int(args.get("max_windows") or 200),
    )
    profile_root = storage_root / "artifacts" / "element_creation_profiles"
    manifest_root = storage_root / "manifests" / "element_creation_profiles"
    receipt_root = storage_root / "receipts" / "element_creation_profiles"
    report_root = storage_root / "reports" / "element_creation_profiles"
    for path in (profile_root, manifest_root, receipt_root, report_root):
        path.mkdir(parents=True, exist_ok=True)
    profile_path = profile_root / f"{run_id}_{_safe_id(element_id)}_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")
    if not _verify_profile_file(profile_path):
        raise ValueError("profile verification failed before receipt")
    purge_requested = bool(args.get("purge_teacher_state", False))
    if purge_requested:
        purge_report = _purge_teacher_state(
            manifest_path=manifest_path,
            profile_path=profile_path,
            purge_records_jsonl=bool(args.get("purge_records_jsonl", True)),
        )
    else:
        purge_report = {
            "schema_version": "truevision_teacher_state_purge_report_v1",
            "created_at_utc": utc_now(),
            "status": "not_requested",
            "profile_verified_before_purge": True,
            "deleted_file_count": 0,
            "deleted_bytes": 0,
            "deleted_files": [],
            "skipped_files": [],
            "retained_outputs": [str(profile_path)],
        }
    purge_report_path = report_root / f"{run_id}_teacher_state_purge_report.json"
    purge_report_path.write_text(json.dumps(purge_report, indent=2, allow_nan=False), encoding="utf-8")
    tool_manifest = {
        "schema_version": "truevision_element_creation_profile_manifest_v1",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "element_id": element_id,
        "profile_json": str(profile_path),
        "profile_sha256": profile["profile_sha256"],
        "sampled_frames": profile["sampled_frames"],
        "six_one_six_windows": profile["six_one_six"]["window_count"],
        "creation_signature_channels": list(CREATION_CHANNELS),
        "source_manifest_sha256": profile["source"]["manifest_sha256"],
        "retention": {
            "purge_requested": purge_requested,
            "purge_report_json": str(purge_report_path),
            "purge_status": purge_report["status"],
            "deleted_teacher_bytes": purge_report["deleted_bytes"],
        },
        "boundary": profile["boundary"],
    }
    tool_manifest["manifest_sha256"] = stable_hash(tool_manifest)
    tool_manifest_path = manifest_root / f"{run_id}_manifest.json"
    tool_manifest_path.write_text(json.dumps(tool_manifest, indent=2, allow_nan=False), encoding="utf-8")
    receipt = {
        "schema_version": "truevision_element_creation_profile_receipt_v1",
        "created_at_utc": utc_now(),
        "tool": "element_creation_profile_from_capture",
        "run_id": run_id,
        "element_id": element_id,
        "profile_json": str(profile_path),
        "profile_sha256": profile["profile_sha256"],
        "manifest_json": str(tool_manifest_path),
        "purge_report_json": str(purge_report_path),
        "purge": {
            "requested": purge_requested,
            "status": purge_report["status"],
            "deleted_file_count": purge_report["deleted_file_count"],
            "deleted_bytes": purge_report["deleted_bytes"],
        },
        "boundary": profile["boundary"],
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    receipt_path = receipt_root / f"{run_id}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_id": run_id,
        "element_id": element_id,
        "profile_json": str(profile_path),
        "profile_sha256": profile["profile_sha256"],
        "manifest_json": str(tool_manifest_path),
        "receipt_json": str(receipt_path),
        "purge_report_json": str(purge_report_path),
        "sampled_frames": profile["sampled_frames"],
        "six_one_six_windows": profile["six_one_six"]["window_count"],
        "creation_signature": profile["creation_signature"],
        "purge": {
            "requested": purge_requested,
            "status": purge_report["status"],
            "deleted_file_count": purge_report["deleted_file_count"],
            "deleted_bytes": purge_report["deleted_bytes"],
        },
    }
