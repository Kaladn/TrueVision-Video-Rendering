from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now
from truevision_runtime.state_patterns.atmosphere_weather import _read_native_tvcells_frames


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
    high = float(np.percentile(finite, 98))
    if high <= low + 1.0e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _shift_zero_fill(plane: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    source = np.asarray(plane, dtype=np.float32)
    rows, cols = source.shape
    shifted = np.zeros_like(source, dtype=np.float32)
    src_x0 = max(0, -dx)
    src_x1 = min(cols, cols - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(cols, cols + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(rows, rows - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(rows, rows + dy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return shifted
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _focus_score(plane: np.ndarray) -> float:
    plane = np.asarray(plane, dtype=np.float32)
    if plane.size == 0:
        return 0.0
    peak = float(np.max(plane))
    mean = float(np.mean(plane))
    contrast = float(np.std(plane))
    return round(float((peak + contrast) / (mean + 1.0e-6)), 6)


def refocus_lightfield_planes(angular_samples: list[dict[str, Any]], *, focus_depth: float) -> dict[str, Any]:
    """Synthetic aperture integration over pseudo-angular TrueVision samples.

    The implemented shift follows:
    I_z(x,y) = sum L(x + Delta_x(z,theta), y + Delta_y(z,phi), theta, phi)

    with Delta_x approximated as -z*theta in cell units. The samples can come
    from real angular views or from broad temporal/video state slices.
    """

    if not angular_samples:
        return {
            "focus_depth": round(float(focus_depth), 6),
            "sample_count": 0,
            "focus_score": 0.0,
            "peak_intensity": 0.0,
            "mean_intensity": 0.0,
        }
    shifted: list[np.ndarray] = []
    for sample in angular_samples:
        theta = float(sample.get("theta") or 0.0)
        phi = float(sample.get("phi") or 0.0)
        dx = int(round(-float(focus_depth) * theta))
        dy = int(round(-float(focus_depth) * phi))
        shifted.append(_shift_zero_fill(np.asarray(sample["plane"], dtype=np.float32), dx=dx, dy=dy))
    refocused = np.mean(np.stack(shifted, axis=0), axis=0).astype(np.float32)
    return {
        "focus_depth": round(float(focus_depth), 6),
        "sample_count": len(angular_samples),
        "focus_score": _focus_score(refocused),
        "peak_intensity": round(float(np.max(refocused)), 6),
        "mean_intensity": round(float(np.mean(refocused)), 6),
        "energy": round(float(np.sum(refocused)), 6),
    }


def detect_active_bounds(plane: np.ndarray, *, threshold: float | None = None) -> dict[str, Any]:
    """Find the active visual rectangle inside a broad capture.

    This is intentionally content-first and chrome-agnostic: broad state is
    captured first, then the extraction lens chooses the useful region later.
    """

    normalized = _normalize(np.asarray(plane, dtype=np.float32))
    if normalized.size == 0 or float(np.max(normalized)) <= 1.0e-9:
        return {
            "grid_xywh": [0, 0, 0, 0],
            "normalized_xywh": [0.0, 0.0, 0.0, 0.0],
            "coverage": 0.0,
            "orientation": "empty",
        }
    active_threshold = float(threshold) if threshold is not None else max(0.08, float(np.percentile(normalized, 70)))
    mask = normalized >= active_threshold
    if not bool(np.any(mask)):
        mask = normalized > 0.0
    yy, xx = np.where(mask)
    rows, cols = normalized.shape
    x0 = int(np.min(xx))
    x1 = int(np.max(xx)) + 1
    y0 = int(np.min(yy))
    y1 = int(np.max(yy)) + 1
    width = max(0, x1 - x0)
    height = max(0, y1 - y0)
    if height > width * 1.25:
        orientation = "vertical_phone"
    elif width > height * 1.25:
        orientation = "horizontal_wide"
    else:
        orientation = "mixed_or_square"
    return {
        "grid_xywh": [x0, y0, width, height],
        "normalized_xywh": [
            round(x0 / max(cols, 1), 6),
            round(y0 / max(rows, 1), 6),
            round(width / max(cols, 1), 6),
            round(height / max(rows, 1), 6),
        ],
        "coverage": round(float(np.mean(mask)), 6),
        "orientation": orientation,
    }


def _radiance_plane(cells: np.ndarray, feature_names: list[str]) -> np.ndarray:
    luma = _feature_plane(cells, feature_names, "luma_mean")
    edge = _feature_plane(cells, feature_names, "edge_density")
    texture = _feature_plane(cells, feature_names, "texture_energy")
    motion = _feature_plane(cells, feature_names, "motion_energy")
    saturation = _feature_plane(cells, feature_names, "saturation_mean")
    return _normalize(luma + 0.45 * edge + 0.35 * texture + 0.55 * motion + 0.20 * saturation)


def _angular_samples(frames: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    if not frames:
        return []
    center = (len(frames) - 1) / 2.0
    denom = max(center, 1.0)
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(frames):
        theta = (index - center) / denom
        samples.append(
            {
                "theta": float(theta),
                "phi": 0.0,
                "source_frame_index": int(item["global_frame_index"]),
                "plane": _radiance_plane(item["cells"], feature_names),
            }
        )
    return samples


def _hash_profile(profile: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in profile.items() if key != "profile_sha256"})


def _verify_profile(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("profile_sha256") == _hash_profile(payload)


def _safe_id(value: str | None, fallback: str = "state_focus_lens") -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "")).strip("_")
    return safe or fallback


def build_lightfield_focus_profile_from_native_capture(
    manifest_path: str | Path,
    *,
    element_id: str,
    max_frames: int = 180,
    sample_stride: int = 1,
    focus_depths: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0),
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell_state = manifest.get("cell_state") or {}
    feature_names = list(cell_state.get("feature_names") or [])
    frames = _read_native_tvcells_frames(manifest, max_frames=max_frames, sample_stride=sample_stride)
    samples = _angular_samples(frames, feature_names)
    aggregate = (
        np.mean(np.stack([sample["plane"] for sample in samples], axis=0), axis=0)
        if samples
        else np.zeros((0, 0), dtype=np.float32)
    )
    focus_planes = [refocus_lightfield_planes(samples, focus_depth=depth) for depth in focus_depths]
    best_plane = max(focus_planes, key=lambda item: float(item["focus_score"]), default={})
    profile = {
        "schema_version": "truevision_lightfield_focus_profile_v1",
        "created_at_utc": utc_now(),
        "element_id": str(element_id),
        "source": {
            "manifest_json": str(manifest_path),
            "run_id": manifest.get("run_id"),
            "capture_region": (manifest.get("config") or {}).get("capture_region"),
            "grid_size_xy": (manifest.get("config") or {}).get("grid_size_xy"),
            "capture_fps": float((manifest.get("config") or {}).get("capture_fps") or 1.0),
            "sampled_frames": len(frames),
        },
        "lightfield_model": {
            "kind": "truevision_pseudo_lightfield_v1",
            "formula": "L(x,y,theta,phi) -> I_z(x,y)=sum L(x+Delta_x(z,theta),y+Delta_y(z,phi),theta,phi)",
            "ray_state_terms": ["x", "y", "theta", "phi"],
            "angular_source": "temporal_state_slices_from_broad_capture",
            "hardware_claim": "not_a_plenoptic_camera",
        },
        "capture_policy": {
            "record_broad_focus_later": True,
            "overcrop_during_recording": False,
            "extraction_lens_decides_focus": True,
        },
        "active_bounds": detect_active_bounds(aggregate),
        "focus_planes": focus_planes,
        "best_focus": best_plane,
        "renderer_binding": {
            "mode": "lightfield_style_focus_profile",
            "use_for": [
                "video_edge_bounds",
                "vertical_short_focus",
                "post_capture_focus_plane_selection",
                "bloom_depth_weighting",
                "light_beam_focus",
            ],
        },
        "retention": {
            "raw_teacher_state_required_after_profile": False,
            "durable_output": "compact_lightfield_focus_profile",
            "teacher_state_deletable_after_profile_verified": True,
        },
        "boundary": {
            "state_first_pixels_last": True,
            "broad_capture_not_replay": True,
            "generated_media_is_evidence": False,
        },
    }
    profile["profile_sha256"] = _hash_profile(profile)
    return profile


def write_state_focus_lens_from_capture(args: dict[str, Any], *, storage_root: Path) -> dict[str, Any]:
    manifest_path = Path(str(args.get("manifest") or args.get("manifest_json") or ""))
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))
    element_id = str(args.get("element_id") or "")
    if not element_id:
        raise ValueError("element_id is required")
    run_id = _safe_id(str(args.get("run_id") or f"{element_id}_state_focus_lens"))
    raw_depths = args.get("focus_depths")
    if raw_depths:
        focus_depths = tuple(float(value) for value in raw_depths)
    else:
        focus_depths = (-1.0, -0.5, 0.0, 0.5, 1.0)
    profile = build_lightfield_focus_profile_from_native_capture(
        manifest_path,
        element_id=element_id,
        max_frames=int(args.get("max_frames") or 180),
        sample_stride=int(args.get("sample_stride") or 1),
        focus_depths=focus_depths,
    )
    profile_root = storage_root / "artifacts" / "state_focus_lens"
    manifest_root = storage_root / "manifests" / "state_focus_lens"
    receipt_root = storage_root / "receipts" / "state_focus_lens"
    report_root = storage_root / "reports" / "state_focus_lens"
    for path in (profile_root, manifest_root, receipt_root, report_root):
        path.mkdir(parents=True, exist_ok=True)
    safe_element = _safe_id(element_id)
    profile_path = profile_root / f"{run_id}_{safe_element}_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")
    if not _verify_profile(profile_path):
        raise ValueError("state focus lens profile verification failed")
    tool_manifest = {
        "schema_version": "truevision_state_focus_lens_manifest_v1",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "element_id": element_id,
        "profile_json": str(profile_path),
        "profile_sha256": profile["profile_sha256"],
        "active_bounds": profile["active_bounds"],
        "best_focus": profile["best_focus"],
        "source": profile["source"],
        "capture_policy": profile["capture_policy"],
        "boundary": profile["boundary"],
    }
    tool_manifest["manifest_sha256"] = stable_hash(tool_manifest)
    manifest_out = manifest_root / f"{run_id}_manifest.json"
    manifest_out.write_text(json.dumps(tool_manifest, indent=2, allow_nan=False), encoding="utf-8")
    receipt = {
        "schema_version": "truevision_state_focus_lens_receipt_v1",
        "created_at_utc": utc_now(),
        "tool": "state_focus_lens_from_capture",
        "run_id": run_id,
        "element_id": element_id,
        "profile_json": str(profile_path),
        "profile_sha256": profile["profile_sha256"],
        "manifest_json": str(manifest_out),
        "active_bounds": profile["active_bounds"],
        "best_focus": profile["best_focus"],
        "boundary": {
            **profile["boundary"],
            "not_true_optical_lightfield": True,
            "capture_wide_focus_later": True,
        },
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    receipt_path = receipt_root / f"{run_id}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    report = {
        "schema_version": "truevision_state_focus_lens_report_v1",
        "run_id": run_id,
        "operator_summary": "Capture wide. Focus later. Learn only from supported regions.",
        "active_bounds": profile["active_bounds"],
        "best_focus": profile["best_focus"],
        "focus_plane_count": len(profile["focus_planes"]),
    }
    report_path = report_root / f"{run_id}_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_id": run_id,
        "element_id": element_id,
        "profile_json": str(profile_path),
        "profile_sha256": profile["profile_sha256"],
        "manifest_json": str(manifest_out),
        "receipt_json": str(receipt_path),
        "report_json": str(report_path),
        "active_bounds": profile["active_bounds"],
        "best_focus": profile["best_focus"],
        "focus_plane_count": len(profile["focus_planes"]),
    }
