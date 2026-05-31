from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now
from truevision_runtime.state_source_law import STATE_SOURCE_LAW_LINES


LOGGER_ROUNDUP_SCHEMA = "truevision_logger_roundup_manifest_v1"
DEEP_PIXEL_SCHEMA = "truevision_deep_pixel_transform_analysis_v1"


LOGGER_LANE_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "lane_id": "native_rust_cell_state_capture",
        "class": "primary_state_logger",
        "paths": ["native/truevision_capture_rs/src/main.rs"],
        "entrypoints": ["native/truevision_capture_rs/src/main.rs", "native/truevision_capture_rs"],
        "outputs": [".tvcells", "*_frame_state.jsonl", "*_manifest.json"],
        "source_truth_role": "writes native cell state and frame-state logs",
    },
    {
        "lane_id": "meter_grid_from_capture",
        "class": "derived_profile_logger",
        "paths": ["truevision_runtime/learning_intake/meter_grid.py", "scripts/truevision_meter_grid.py"],
        "entrypoints": ["scripts/truevision_meter_grid.py"],
        "outputs": ["meter profile json", "event profile json", "graphs", "receipt"],
        "source_truth_role": "measures state cells; does not replace raw state evidence",
    },
    {
        "lane_id": "angular_seismic_16_direction",
        "class": "derived_profile_logger",
        "paths": ["truevision_runtime/learning_intake/angular_seismic.py", "scripts/truevision_angular_seismic_video.py"],
        "entrypoints": ["scripts/truevision_angular_seismic_video.py"],
        "outputs": ["angular seismic profile", "direction graphs", "receipt"],
        "source_truth_role": "derives motion/direction profile from local video/state samples",
    },
    {
        "lane_id": "state_focus_lens",
        "class": "derived_profile_logger",
        "paths": ["truevision_runtime/learning_intake/lightfield_focus.py", "scripts/truevision_state_focus_lens.py"],
        "entrypoints": ["scripts/truevision_state_focus_lens.py"],
        "outputs": ["focus profile", "active bounds", "receipt"],
        "source_truth_role": "post-capture focus over broad state",
    },
    {
        "lane_id": "truedepth_contracts",
        "class": "contract_logger",
        "paths": ["truevision_runtime/learning_intake/trudepth_contracts.py", "scripts/render_truedepth_fog_reveal_samples.py"],
        "entrypoints": ["scripts/render_truedepth_fog_reveal_samples.py"],
        "outputs": ["TruDepth contracts", "depth/reveal samples"],
        "source_truth_role": "depth/reveal contract and validation lane",
    },
    {
        "lane_id": "atmosphere_weather_profiles",
        "class": "derived_profile_logger",
        "paths": ["truevision_runtime/state_patterns/atmosphere_weather.py", "scripts/truevision_atmosphere_tools.py"],
        "entrypoints": ["scripts/truevision_atmosphere_tools.py"],
        "outputs": ["atmosphere profile", "weather toolset manifest", "receipt"],
        "source_truth_role": "extracts fog/smoke/cloud/weather state behavior",
    },
    {
        "lane_id": "element_creation_profile",
        "class": "derived_profile_logger",
        "paths": ["truevision_runtime/learning_intake/element_creation_profile.py"],
        "entrypoints": [],
        "outputs": ["creation profile", "purge receipt"],
        "source_truth_role": "compresses teacher capture into compact creation signatures",
    },
    {
        "lane_id": "driving_school_awareness",
        "class": "derived_profile_logger",
        "paths": ["truevision_runtime/learning_intake/driving_school.py", "scripts/truevision_driving_school.py"],
        "entrypoints": ["scripts/truevision_driving_school.py"],
        "outputs": ["calibration receipt", "scene profile", "candidate receipts", "mock road world"],
        "source_truth_role": "candidate-first local video awareness profile",
    },
    {
        "lane_id": "high_speed_awareness",
        "class": "derived_profile_logger",
        "paths": ["scripts/truevision_high_speed_awareness.py"],
        "entrypoints": ["scripts/truevision_high_speed_awareness.py"],
        "outputs": ["awareness report"],
        "source_truth_role": "high-speed perception candidate lane",
    },
    {
        "lane_id": "geometry_generation",
        "class": "shape_logger",
        "paths": ["truevision_runtime/geometry_generation.py", "scripts/truevision_geometry_engine.py"],
        "entrypoints": ["scripts/truevision_geometry_engine.py"],
        "outputs": ["shape units", "big-shape library", "geometry manifest", "receipt"],
        "source_truth_role": "binds geometry to source state refs and metrics",
    },
    {
        "lane_id": "trueaudio_file_state_logging",
        "class": "audio_state_logger",
        "paths": [
            "trueaudio_runtime/logging.py",
            "trueaudio_runtime/replayable.py",
            "scripts/trueaudio_log_pre_sound.py",
            "scripts/trueaudio_log_file_replayable.py",
        ],
        "entrypoints": ["scripts/trueaudio_log_pre_sound.py", "scripts/trueaudio_log_file_replayable.py"],
        "outputs": ["*_audio_state.jsonl", "*.trueaudio.npz", "manifest", "receipt"],
        "source_truth_role": "logs derived audio state from local files",
    },
    {
        "lane_id": "trueaudio_machine_state_logging",
        "class": "audio_state_logger",
        "paths": ["scripts/trueaudio_log_machine_pre_sound.py", "scripts/trueaudio_log_machine_replayable.py"],
        "entrypoints": ["scripts/trueaudio_log_machine_pre_sound.py", "scripts/trueaudio_log_machine_replayable.py"],
        "outputs": ["machine audio state", "manifest", "receipt"],
        "source_truth_role": "logs machine mix state before output",
    },
    {
        "lane_id": "timing_audit",
        "class": "receipt_manifest_lane",
        "paths": ["truevision_runtime/timeline_audit.py", "scripts/truevision_timing_audit.py"],
        "entrypoints": ["scripts/truevision_timing_audit.py"],
        "outputs": ["timing audit report"],
        "source_truth_role": "verifies state timeline cadence and manifest timing",
    },
    {
        "lane_id": "state_source_law",
        "class": "contract_logger",
        "paths": ["truevision_runtime/state_source_law.py"],
        "entrypoints": [],
        "outputs": ["artifact authority classification"],
        "source_truth_role": "defines state vs media authority boundary",
    },
    {
        "lane_id": "state_replay",
        "class": "replay_lane",
        "paths": ["scripts/truevision_state_replay.py", "scripts/truevision_state_snap_sequence_renderer.py"],
        "entrypoints": ["scripts/truevision_state_replay.py"],
        "outputs": ["derived replay media", "manifest", "receipt"],
        "source_truth_role": "replays state as derived visualization, not evidence",
    },
    {
        "lane_id": "state_media_qa",
        "class": "receipt_manifest_lane",
        "paths": ["truevision_runtime/state_media_qa.py", "scripts/truevision_state_media_qa_receipt.py"],
        "entrypoints": ["scripts/truevision_state_media_qa_receipt.py"],
        "outputs": ["QA receipt"],
        "source_truth_role": "checks derived media against state contract",
    },
    {
        "lane_id": "av_tool_registry",
        "class": "tool_catalog_lane",
        "paths": ["truevision_runtime/av_tools/av_tool_registry.py", "truevision_runtime/av_tools/av_tool_runner.py"],
        "entrypoints": ["scripts/truevision_studio_server.py"],
        "outputs": ["tool specs", "tool receipts"],
        "source_truth_role": "catalogs callable local tools and receipt behavior",
    },
    {
        "lane_id": "deep_pixel_transform_analysis",
        "class": "pixel_state_analysis_logger",
        "paths": ["truevision_runtime/logger_roundup.py", "scripts/truevision_logger_roundup.py"],
        "entrypoints": ["scripts/truevision_logger_roundup.py"],
        "outputs": ["deep pixel transform analysis json"],
        "source_truth_role": "compares source and transformed pixels without claiming added artifacts as source truth",
    },
)


SCAN_DIRS = ("truevision_runtime", "trueaudio_runtime", "trueframegen", "scripts", "native", "tests")
LOGGER_KEYWORDS = (
    "logger",
    "logging",
    "log_",
    "state",
    "tvcells",
    "meter",
    "angular",
    "seismic",
    "focus",
    "depth",
    "receipt",
    "manifest",
    "replay",
    "geometry",
)


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _exists(repo_root: Path, path: str) -> bool:
    return (repo_root / path).exists()


def _catalog_lane(repo_root: Path, lane: dict[str, Any]) -> dict[str, Any]:
    paths = list(lane["paths"])
    entrypoints = list(lane.get("entrypoints") or [])
    existing_paths = [path for path in paths if _exists(repo_root, path)]
    existing_entrypoints = [path for path in entrypoints if _exists(repo_root, path)]
    payload = dict(lane)
    payload["status"] = "present" if existing_paths else "missing"
    payload["paths_present"] = existing_paths
    payload["paths_missing"] = [path for path in paths if path not in existing_paths]
    payload["entrypoints_present"] = existing_entrypoints
    payload["entrypoints_missing"] = [path for path in entrypoints if path not in existing_entrypoints]
    payload["source_truth_allowed"] = lane["class"] in {
        "primary_state_logger",
        "derived_profile_logger",
        "audio_state_logger",
        "shape_logger",
        "pixel_state_analysis_logger",
        "contract_logger",
        "receipt_manifest_lane",
    }
    payload["render_or_replay_output_is_source_truth"] = False
    return payload


def discover_logger_files(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root)
    discovered: list[dict[str, Any]] = []
    for scan_dir in SCAN_DIRS:
        base = root / scan_dir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or any(part in {"target", "__pycache__", ".git"} for part in path.parts):
                continue
            if path.suffix.lower() not in {".py", ".rs", ".html", ".md", ".json"}:
                continue
            rel = _rel(path, root)
            lowered = rel.lower()
            matched = [keyword for keyword in LOGGER_KEYWORDS if keyword in lowered]
            if not matched:
                continue
            discovered.append(
                {
                    "path": rel,
                    "suffix": path.suffix.lower(),
                    "matched_keywords": matched,
                    "size_bytes": path.stat().st_size,
                }
            )
    return discovered


def build_logger_roundup_manifest(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    lanes = [_catalog_lane(root, lane) for lane in LOGGER_LANE_CATALOG]
    discovered = discover_logger_files(root)
    catalog_paths = {path for lane in lanes for path in lane.get("paths", [])}
    catalog_paths.update(path for lane in lanes for path in lane.get("entrypoints", []))
    uncovered = [item for item in discovered if item["path"] not in catalog_paths]
    entrypoints = sorted({path for lane in lanes for path in lane.get("entrypoints_present", [])})
    manifest = {
        "schema_version": LOGGER_ROUNDUP_SCHEMA,
        "created_at_utc": utc_now(),
        "repo_root": str(root),
        "logger_lanes": lanes,
        "entrypoints": entrypoints,
        "discovered_logger_files": discovered,
        "uncovered_discovered_files": uncovered,
        "deep_pixel_transform": {
            "available": True,
            "function": "truevision_runtime.logger_roundup.analyze_deep_pixel_transform",
            "script": "scripts/truevision_logger_roundup.py",
            "mode": "source_pixel_state_diff",
        },
        "source_truth_types": [".tvcells", "*_frame_state.jsonl", "*_audio_state.jsonl", "*.npz", "manifest", "receipt", "profile_json"],
        "forbidden_source_truth_types": [".mp4", ".mkv", ".mov", ".webm", ".png", ".jpg", ".jpeg", ".wav", ".mp3", "generated_media"],
        "law": list(STATE_SOURCE_LAW_LINES),
        "boundary": {
            "capture_started": False,
            "render_started": False,
            "source_pixel_transform_only": True,
            "generated_media_is_evidence": False,
            "six_one_six_mapping_enabled": False,
            "mapping_policy": "zero_6_1_6_mapping",
        },
    }
    manifest["manifest_sha256"] = stable_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    return manifest


def _load_rgb_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)


def _luma(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    return arr[:, :, 0] * 0.2126 + arr[:, :, 1] * 0.7152 + arr[:, :, 2] * 0.0722


def _saturation(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 1]


def _hue(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 0] / 360.0


def _edge(rgb: np.ndarray) -> np.ndarray:
    gray = _luma(rgb).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (3, 3), 0.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    return np.clip(np.sqrt(gx * gx + gy * gy) * 2.8, 0.0, 1.0)


def _safe_mean(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        values = values[mask]
    if values.size == 0:
        return 0.0
    return round(float(np.mean(values)), 6)


def _safe_peak(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        values = values[mask]
    if values.size == 0:
        return 0.0
    return round(float(np.max(values)), 6)


def _bounds(mask: np.ndarray) -> dict[str, Any]:
    if mask.size == 0 or not bool(np.any(mask)):
        return {"x": 0, "y": 0, "width": 0, "height": 0, "coverage": 0.0}
    yy, xx = np.where(mask)
    x0, x1 = int(np.min(xx)), int(np.max(xx)) + 1
    y0, y1 = int(np.min(yy)), int(np.max(yy)) + 1
    return {
        "x": x0,
        "y": y0,
        "width": max(0, x1 - x0),
        "height": max(0, y1 - y0),
        "coverage": round(float(np.mean(mask)), 6),
    }


def _material_masks(source_rgb: np.ndarray) -> dict[str, np.ndarray]:
    arr = source_rgb.astype(np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    luma = _luma(source_rgb)
    sat = _saturation(source_rgb)
    edge = _edge(source_rgb)
    warm = (r > g * 0.86) & (g > b * 1.10) & (luma > 0.16)
    bright_text = (luma > 0.55) & (sat < 0.42)
    shadow = luma < 0.22
    haze = (sat < 0.35) & (luma >= 0.16) & (luma <= 0.68) & (edge < 0.22)
    linework = edge > max(0.12, float(np.percentile(edge, 82)))
    changed_candidate = luma > 0.0
    return {
        "warm_gold_sun_sky": warm,
        "bright_text_glyphs": bright_text,
        "shadow_regions": shadow,
        "smoke_haze_soft_regions": haze,
        "edge_linework": linework,
        "all_source_pixels": changed_candidate,
    }


def _region_summary(name: str, mask: np.ndarray, luma_delta: np.ndarray, sat_delta: np.ndarray, hue_delta: np.ndarray, edge_delta: np.ndarray) -> dict[str, Any]:
    return {
        "region_id": name,
        "bounds": _bounds(mask),
        "metrics": {
            "mean_luma_delta": _safe_mean(luma_delta, mask),
            "peak_luma_delta": _safe_peak(np.abs(luma_delta), mask),
            "mean_saturation_delta": _safe_mean(sat_delta, mask),
            "mean_hue_delta": _safe_mean(hue_delta, mask),
            "mean_edge_delta": _safe_mean(edge_delta, mask),
        },
        "evidence_boundary": {
            "region_from_source_pixels": True,
            "new_geometry_created": False,
        },
    }


def analyze_deep_pixel_transform(source_image: str | Path, transformed_image: str | Path) -> dict[str, Any]:
    source = _load_rgb_image(source_image)
    transformed = _load_rgb_image(transformed_image)
    if transformed.shape[:2] != source.shape[:2]:
        transformed = cv2.resize(transformed, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)

    source_luma = _luma(source)
    target_luma = _luma(transformed)
    source_sat = _saturation(source)
    target_sat = _saturation(transformed)
    source_hue = _hue(source)
    target_hue = _hue(transformed)
    source_edge = _edge(source)
    target_edge = _edge(transformed)

    luma_delta = target_luma - source_luma
    sat_delta = target_sat - source_sat
    hue_delta = target_hue - source_hue
    edge_delta = target_edge - source_edge
    rgb_delta = np.linalg.norm(transformed.astype(np.float32) - source.astype(np.float32), axis=2) / (255.0 * np.sqrt(3.0))
    changed = rgb_delta > 0.025

    masks = _material_masks(source)
    material_regions = [
        _region_summary(name, mask, luma_delta, sat_delta, hue_delta, edge_delta)
        for name, mask in masks.items()
        if bool(np.any(mask))
    ]
    operators = [
        {
            "operator_id": "luminance_rise_fall",
            "input": "source_luma",
            "output": "target_luma",
            "mean_delta": _safe_mean(luma_delta),
            "peak_abs_delta": _safe_peak(np.abs(luma_delta)),
            "adds_geometry": False,
        },
        {
            "operator_id": "hue_saturation_pressure",
            "input": "source_hue_saturation",
            "output": "target_hue_saturation",
            "mean_hue_delta": _safe_mean(hue_delta),
            "mean_saturation_delta": _safe_mean(sat_delta),
            "adds_geometry": False,
        },
        {
            "operator_id": "edge_contrast_recovery",
            "input": "source_edge_field",
            "output": "target_edge_field",
            "mean_edge_delta": _safe_mean(edge_delta),
            "peak_edge_delta": _safe_peak(np.abs(edge_delta)),
            "adds_geometry": False,
        },
        {
            "operator_id": "shadow_pressure",
            "input": "source_shadow_regions",
            "output": "target_shadow_regions",
            "mean_luma_delta": _safe_mean(luma_delta, masks.get("shadow_regions")),
            "adds_geometry": False,
        },
        {
            "operator_id": "warm_sun_sky_breath",
            "input": "source_warm_gold_sun_sky_pixels",
            "output": "target_warm_gold_sun_sky_pixels",
            "mean_luma_delta": _safe_mean(luma_delta, masks.get("warm_gold_sun_sky")),
            "peak_luma_delta": _safe_peak(np.abs(luma_delta), masks.get("warm_gold_sun_sky")),
            "adds_geometry": False,
        },
    ]
    analysis = {
        "schema_version": DEEP_PIXEL_SCHEMA,
        "created_at_utc": utc_now(),
        "source_image": str(source_image),
        "transformed_image": str(transformed_image),
        "image_shape": {"height": int(source.shape[0]), "width": int(source.shape[1]), "channels": 3},
        "global_delta": {
            "changed_pixel_ratio": round(float(np.mean(changed)), 6),
            "mean_rgb_delta": _safe_mean(rgb_delta),
            "peak_rgb_delta": _safe_peak(rgb_delta),
            "mean_luma_delta": _safe_mean(luma_delta),
            "peak_luma_delta": _safe_peak(np.abs(luma_delta)),
            "mean_saturation_delta": _safe_mean(sat_delta),
            "mean_edge_delta": _safe_mean(edge_delta),
        },
        "material_regions": material_regions,
        "transform_operators": operators,
        "boundary": {
            "source_pixel_transform_only": True,
            "added_artifact_detection_claim": False,
            "new_geometry_created": False,
            "six_one_six_mapping_enabled": False,
            "mapping_policy": "zero_6_1_6_mapping",
            "generated_media_is_evidence": False,
        },
    }
    analysis["analysis_sha256"] = stable_hash({key: value for key, value in analysis.items() if key != "analysis_sha256"})
    return analysis


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return output
