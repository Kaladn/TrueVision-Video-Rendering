from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now
from truevision_runtime.state_source_law import STATE_SOURCE_LAW_LINES


KNOWN_LOGGER_LANES = (
    "native_frame_state",
    "meter_grid",
    "angular_seismic_16dir",
    "state_focus_lens",
    "truedepth",
    "atmosphere_weather",
    "motion_vectors",
    "occlusion",
    "light_shadow_vectors",
    "element_creation_profiles",
    "timing_progress_receipts",
    "trueaudio",
    "driving_awareness",
    "worker_forge",
)

BIG_SHAPE_TYPES = (
    "road_plane",
    "horizon_band",
    "vanishing_corridor",
    "fog_bank",
    "depth_wall",
    "light_cone",
    "occlusion_slab",
    "motion_stream",
    "reflection_vector_field",
    "atmosphere_volume",
    "lightning_branch",
    "water_plane",
)

SHAPE_UNIT_SCHEMA = "truevision_geometry_shape_unit_v1"
SCENE_SCHEMA = "truevision_geometry_scene_v1"
LIBRARY_SCHEMA = "truevision_big_shape_library_v1"
MANIFEST_SCHEMA = "truevision_geometry_generation_manifest_v1"
RECEIPT_SCHEMA = "truevision_geometry_generation_receipt_v1"


def _safe_id(value: str | None, fallback: str = "geometry_generation") -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "")).strip("_")
    return safe or fallback


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _extract_path_refs(payload: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in {"path", "manifest_json", "records_jsonl", "profile_json", "receipt_json", "state_npz", "source_video"}:
                if isinstance(value, str) and value:
                    refs.append(value)
            elif lowered in {"teacher_chunks", "chunks"} and isinstance(value, list):
                for item in value:
                    refs.extend(_extract_path_refs(item))
            else:
                refs.extend(_extract_path_refs(value))
    elif isinstance(payload, list):
        for item in payload:
            refs.extend(_extract_path_refs(item))
    seen: set[str] = set()
    unique: list[str] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            unique.append(ref)
    return unique


def _load_json_if_path(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    path = Path(str(value))
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_big_shape_library() -> dict[str, Any]:
    required_slots = [
        "geometry",
        "source_region",
        "raw_data_refs",
        "true_local_metrics",
        "filtered_metrics",
        "logger_lanes",
        "evidence_boundary",
    ]
    descriptions = {
        "road_plane": "ground or road surface candidate with perspective and edge metrics",
        "horizon_band": "distant horizontal separation line or skyline/sky split",
        "vanishing_corridor": "forward-motion corridor with convergence and center attention",
        "fog_bank": "volumetric haze body with source-bound density and reveal metrics",
        "depth_wall": "stacked depth plane or far-wall surface",
        "light_cone": "light volume with emitter, spread, bloom, and falloff",
        "occlusion_slab": "region that hides or reveals state behind it",
        "motion_stream": "directional field carrying velocity/impulse over cells",
        "reflection_vector_field": "mirror/glass/water reflection direction and persistence field",
        "atmosphere_volume": "fog/smoke/cloud/rain volume with density and softness",
        "lightning_branch": "branching light shape bound to raw local luma/color/edge cells",
        "water_plane": "horizontal water/surface field with shimmer and band motion",
    }
    shapes = [
        {
            "schema_version": "truevision_big_shape_template_v1",
            "shape_type": shape_type,
            "description": descriptions[shape_type],
            "required_data_slots": list(required_slots),
            "truth_boundary": {
                "shape_is_data_container": True,
                "drawing_is_interpretation": True,
                "object_truth_claim": False,
            },
        }
        for shape_type in BIG_SHAPE_TYPES
    ]
    library = {
        "schema_version": LIBRARY_SCHEMA,
        "created_at_utc": utc_now(),
        "shapes": shapes,
        "law": "A shape carries source truth. Filters reveal it, but raw local state owns it.",
        "boundary": {
            "shape_library_only": True,
            "generated_media_is_evidence": False,
            "recognition_backend_truth_authority": False,
        },
    }
    library["library_sha256"] = stable_hash({key: value for key, value in library.items() if key != "library_sha256"})
    return library


def _logger_lane_visibility_plan(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    present_map = {
        "native_frame_state": any(_extract_path_refs(value) for value in bundle.values()),
        "meter_grid": bool(bundle.get("meter_grid_profile")),
        "angular_seismic_16dir": bool(bundle.get("angular_seismic_profile")),
        "state_focus_lens": bool(bundle.get("state_focus_profile")),
        "truedepth": bool(bundle.get("truedepth_profile") or bundle.get("truedepth_signature")),
        "atmosphere_weather": bool(bundle.get("atmosphere_profile") or bundle.get("weather_profile")),
        "motion_vectors": bool(bundle.get("angular_seismic_profile") or bundle.get("meter_grid_profile")),
        "occlusion": bool(bundle.get("meter_grid_profile") or bundle.get("element_creation_profile")),
        "light_shadow_vectors": bool(bundle.get("meter_grid_profile") or bundle.get("element_creation_profile")),
        "element_creation_profiles": bool(bundle.get("element_creation_profile")),
        "timing_progress_receipts": bool(bundle.get("receipt") or bundle.get("manifest") or bundle.get("meter_grid_profile")),
        "trueaudio": bool(bundle.get("trueaudio_manifest") or bundle.get("trueaudio_profile")),
        "driving_awareness": bool(bundle.get("driving_profile") or bundle.get("awareness_profile")),
        "worker_forge": bool(bundle.get("worker_forge_manifest")),
    }
    visible_modes = ("state_panel", "meter_graph", "geometry_marks", "overlay", "receipt_ref")
    plan = []
    for index, lane_id in enumerate(KNOWN_LOGGER_LANES):
        plan.append(
            {
                "lane_id": lane_id,
                "status": "present" if present_map.get(lane_id) else "declared_missing_input",
                "visible_as": visible_modes[index % len(visible_modes)],
                "video_obligation": "must_be_visible_or_marked_missing",
            }
        )
    return plan


def _base_shape_unit(
    *,
    shape_id: str,
    shape_type: str,
    geometry: dict[str, Any],
    source_region: dict[str, Any],
    raw_data_refs: list[str],
    true_local_metrics: dict[str, Any],
    filtered_metrics: dict[str, Any],
    logger_lanes: list[str],
    confidence: float,
    recognizer: str,
) -> dict[str, Any]:
    unit = {
        "schema_version": SHAPE_UNIT_SCHEMA,
        "shape_id": shape_id,
        "shape_type": shape_type,
        "geometry": geometry,
        "source_region": source_region,
        "raw_data_refs": list(raw_data_refs),
        "true_local_metrics": true_local_metrics,
        "filtered_metrics": filtered_metrics,
        "logger_lanes": list(logger_lanes),
        "recognizer": recognizer,
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 6),
        "evidence_boundary": {
            "raw_state_owns_evidence": True,
            "filters_reveal_state": True,
            "filtered_metrics_are_truth": False,
            "drawn_geometry_is_interpretation": True,
            "generated_media_is_evidence": False,
            "object_truth_promoted": False,
        },
    }
    unit["shape_sha256"] = stable_hash({key: value for key, value in unit.items() if key != "shape_sha256"})
    return unit


def _first_candidate_event(meter_profile: dict[str, Any] | None, event_type: str) -> dict[str, Any] | None:
    if not meter_profile:
        return None
    for event in meter_profile.get("event_profiles") or []:
        if event.get("event_type_candidate") == event_type:
            return event
    return None


def _lightning_shape(bundle: dict[str, Any]) -> dict[str, Any] | None:
    meter = bundle.get("meter_grid_profile")
    element = bundle.get("element_creation_profile")
    event = _first_candidate_event(meter, "candidate_lightning")
    if event is None:
        return None
    support = dict(event.get("support") or {})
    cell_bounds = list(event.get("cell_bounds") or [0, 0, 0, 0])
    x, y, w, h = (cell_bounds + [0, 0, 0, 0])[:4]
    branch_points = [
        [x + w * 0.50, y],
        [x + w * 0.46, y + h * 0.32],
        [x + w * 0.62, y + h * 0.52],
        [x + w * 0.38, y + h],
    ]
    raw_refs = _extract_path_refs(meter) + _extract_path_refs(element)
    signature = (element or {}).get("creation_signature") if isinstance((element or {}).get("creation_signature"), dict) else {}
    return _base_shape_unit(
        shape_id="shape_lightning_branch_0001",
        shape_type="lightning_branch",
        geometry={
            "kind": "polyline_branch",
            "cell_bounds": cell_bounds,
            "branch_points": [[round(float(px), 6), round(float(py), 6)] for px, py in branch_points],
            "branching_score": support.get("branching_score", 0.0),
        },
        source_region={
            "section_id": (meter or {}).get("section_id"),
            "frame_start": event.get("frame_start"),
            "frame_peak": event.get("frame_peak"),
            "frame_end": event.get("frame_end"),
            "cell_bounds": cell_bounds,
        },
        raw_data_refs=raw_refs,
        true_local_metrics={
            "luma_delta": support.get("luma_delta", 0.0),
            "flash_peak_luma": support.get("flash_peak_luma", support.get("luma_peak", 0.0)),
            "rise_time_frames": support.get("rise_time_frames", 0),
            "falloff_frames": support.get("falloff_frames", 0),
            "bloom_radius_cells": support.get("bloom_radius_cells", 0.0),
            "surrounding_exposure_lift": support.get("surrounding_exposure_lift", 0.0),
        },
        filtered_metrics={
            "event_type_candidate": event.get("event_type_candidate"),
            "status": event.get("status"),
            "recognizer": "meter_grid_filter",
            "rejection_reasons": list(event.get("rejection_reasons") or []),
            "element_creation_signature": {
                key: signature.get(key)
                for key in ("density_opacity", "bloom_intensity", "edge_softness", "growth_decay", "renderer_binding")
                if key in signature
            },
        },
        logger_lanes=["meter_grid", "native_frame_state", "element_creation_profiles", "light_shadow_vectors"],
        confidence=1.0 if event.get("status") == "visually_supported" else 0.35,
        recognizer="meter_grid_filter",
    )


def _motion_stream_shape(bundle: dict[str, Any]) -> dict[str, Any] | None:
    angular = bundle.get("angular_seismic_profile")
    if not angular:
        return None
    signature = angular.get("angular_signature") or {}
    angle = float(signature.get("dominant_angle_degrees") or 0.0)
    radians = math.radians(angle)
    raw_refs = _extract_path_refs(angular)
    return _base_shape_unit(
        shape_id="shape_motion_stream_0001",
        shape_type="motion_stream",
        geometry={
            "kind": "direction_vector_field",
            "dominant_angle_degrees": round(angle, 6),
            "direction_vector": [round(math.cos(radians), 6), round(math.sin(radians), 6)],
            "rings": [1, 2, 3, 4],
        },
        source_region={"scope": "whole_profile", "source_video": (angular.get("source") or {}).get("source_video")},
        raw_data_refs=raw_refs,
        true_local_metrics={
            "dominant_direction": signature.get("dominant_direction"),
            "field_coherence_mean": signature.get("field_coherence_mean", 0.0),
            "impulse_peak": (angular.get("seismic_trace") or {}).get("impulse_peak", 0.0),
        },
        filtered_metrics={"recognizer": "angular_seismic_16dir", "filter_family": "16_direction_energy"},
        logger_lanes=["angular_seismic_16dir", "motion_vectors"],
        confidence=float(signature.get("field_coherence_mean") or 0.5),
        recognizer="angular_seismic_16dir",
    )


def _focus_depth_shape(bundle: dict[str, Any]) -> dict[str, Any] | None:
    focus = bundle.get("state_focus_profile")
    if not focus:
        return None
    bounds = focus.get("active_bounds") or {}
    best_plane = max(focus.get("focus_planes") or [], key=lambda item: float(item.get("focus_score") or 0.0), default={})
    return _base_shape_unit(
        shape_id="shape_depth_wall_0001",
        shape_type="depth_wall",
        geometry={
            "kind": "active_bounds_depth_plane",
            "grid_xywh": bounds.get("grid_xywh", [0, 0, 0, 0]),
            "normalized_xywh": bounds.get("normalized_xywh", [0.0, 0.0, 0.0, 0.0]),
            "orientation": bounds.get("orientation", "unknown"),
        },
        source_region={"scope": "active_bounds", "source": (focus.get("source") or {}).get("manifest_json")},
        raw_data_refs=_extract_path_refs(focus),
        true_local_metrics={
            "focus_depth": best_plane.get("focus_depth", 0.0),
            "focus_score": best_plane.get("focus_score", 0.0),
            "active_coverage": bounds.get("coverage", 0.0),
        },
        filtered_metrics={"recognizer": "state_focus_lens", "filter_family": "capture_wide_focus_later"},
        logger_lanes=["state_focus_lens", "truedepth"],
        confidence=0.82 if bounds else 0.25,
        recognizer="state_focus_lens",
    )


def _atmosphere_shape(bundle: dict[str, Any]) -> dict[str, Any] | None:
    element = bundle.get("element_creation_profile")
    atmosphere = bundle.get("atmosphere_profile") or bundle.get("weather_profile")
    source = element or atmosphere
    if not source:
        return None
    signature = source.get("creation_signature") if isinstance(source.get("creation_signature"), dict) else {}
    density = signature.get("density_opacity") if isinstance(signature.get("density_opacity"), dict) else {}
    edge = signature.get("edge_softness") if isinstance(signature.get("edge_softness"), dict) else {}
    element_id = str(source.get("element_id") or "atmosphere_volume")
    shape_type = "light_cone" if "light" in element_id or "lightning" in element_id else "atmosphere_volume"
    return _base_shape_unit(
        shape_id=f"shape_{shape_type}_0001",
        shape_type=shape_type,
        geometry={"kind": "volumetric_bounds", "normalized_xywh": [0.08, 0.08, 0.84, 0.72], "element_id": element_id},
        source_region={"scope": "element_creation_profile", "element_id": element_id},
        raw_data_refs=_extract_path_refs(source),
        true_local_metrics={
            "density_mean": density.get("mean", 0.0),
            "density_max": density.get("maximum", 0.0),
            "edge_softness_mean": edge.get("mean", 0.0),
        },
        filtered_metrics={"recognizer": "element_creation_profile", "filter_family": "state_creation_signature"},
        logger_lanes=["element_creation_profiles", "atmosphere_weather", "occlusion"],
        confidence=0.7,
        recognizer="element_creation_profile",
    )


def _default_scene_shapes(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    raw_refs = []
    for value in bundle.values():
        raw_refs.extend(_extract_path_refs(value))
    raw_refs = raw_refs[:8]
    return [
        _base_shape_unit(
            shape_id="shape_road_plane_0001",
            shape_type="road_plane",
            geometry={"kind": "trapezoid", "normalized_points": [[0.18, 1.0], [0.82, 1.0], [0.57, 0.58], [0.43, 0.58]]},
            source_region={"scope": "composed_scene_context"},
            raw_data_refs=raw_refs,
            true_local_metrics={"plane_confidence": 0.5, "source_ref_count": len(raw_refs)},
            filtered_metrics={"recognizer": "geometry_default_filter", "filter_family": "scene_stage"},
            logger_lanes=["driving_awareness", "truedepth"],
            confidence=0.5,
            recognizer="geometry_default_filter",
        ),
        _base_shape_unit(
            shape_id="shape_horizon_band_0001",
            shape_type="horizon_band",
            geometry={"kind": "horizontal_band", "normalized_xywh": [0.0, 0.46, 1.0, 0.045]},
            source_region={"scope": "composed_scene_context"},
            raw_data_refs=raw_refs,
            true_local_metrics={"band_y": 0.46, "source_ref_count": len(raw_refs)},
            filtered_metrics={"recognizer": "geometry_default_filter", "filter_family": "scene_stage"},
            logger_lanes=["native_frame_state", "state_focus_lens"],
            confidence=0.45,
            recognizer="geometry_default_filter",
        ),
    ]


def build_geometry_scene_from_logger_bundle(bundle: dict[str, Any], *, run_id: str = "geometry_generation") -> dict[str, Any]:
    loaded_bundle = {
        key: _load_json_if_path(value)
        for key, value in bundle.items()
        if value is not None and value != ""
    }
    loaded_bundle = {key: value for key, value in loaded_bundle.items() if value is not None}
    shape_units: list[dict[str, Any]] = []
    for builder in (_lightning_shape, _motion_stream_shape, _focus_depth_shape, _atmosphere_shape):
        shape = builder(loaded_bundle)
        if shape is not None:
            shape_units.append(shape)
    shape_units.extend(_default_scene_shapes(loaded_bundle))
    scene = {
        "schema_version": SCENE_SCHEMA,
        "created_at_utc": utc_now(),
        "run_id": _safe_id(run_id),
        "logger_lane_visibility_plan": _logger_lane_visibility_plan(loaded_bundle),
        "shape_units": shape_units,
        "shape_count": len(shape_units),
        "shape_library_ref": "truevision_big_shape_library_v1",
        "boundary": {
            "geometry_state_only": True,
            "shapes_are_data_containers": True,
            "filters_detect_state": True,
            "geometry_names_state": True,
            "raw_local_state_required_for_evidence": True,
            "rendered_overlay_is_interpretation": True,
            "object_truth_promoted": False,
            "generated_media_is_evidence": False,
        },
        "state_source_law": list(STATE_SOURCE_LAW_LINES),
        "law": "Filters detect state. Geometry binds source truth. Renderers demonstrate state.",
    }
    scene["scene_sha256"] = stable_hash({key: value for key, value in scene.items() if key != "scene_sha256"})
    return scene


def _put_text(frame: np.ndarray, text: str, x: int, y: int, *, color: tuple[int, int, int] = (220, 235, 240), scale: float = 0.42) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_shape(frame: np.ndarray, shape: dict[str, Any], width: int, height: int, phase: float) -> None:
    shape_type = shape.get("shape_type")
    geometry = shape.get("geometry") or {}
    if shape_type == "road_plane":
        pts = np.array([[int(x * width), int(y * height)] for x, y in geometry.get("normalized_points", [])], dtype=np.int32)
        if pts.size:
            cv2.polylines(frame, [pts], True, (70, 135, 100), 2, cv2.LINE_AA)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (22, 36, 31))
            frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    elif shape_type == "horizon_band":
        x, y, w, h = geometry.get("normalized_xywh", [0.0, 0.46, 1.0, 0.04])
        cv2.rectangle(frame, (int(x * width), int(y * height)), (int((x + w) * width), int((y + h) * height)), (145, 160, 170), 1)
    elif shape_type == "lightning_branch":
        bounds = geometry.get("cell_bounds", [0, 0, 1, 1])
        bx, by, bw, bh = bounds
        scale_x = width / 16.0
        scale_y = height / 10.0
        points = geometry.get("branch_points") or []
        if points:
            pts = np.array([[int(px * scale_x), int(py * scale_y)] for px, py in points], dtype=np.int32)
            glow = int(80 + 120 * (0.5 + 0.5 * math.sin(phase * math.tau)))
            cv2.polylines(frame, [pts], False, (glow, glow, 255), 5, cv2.LINE_AA)
            cv2.polylines(frame, [pts], False, (240, 245, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (int(bx * scale_x), int(by * scale_y)), (int((bx + bw) * scale_x), int((by + bh) * scale_y)), (120, 170, 255), 1)
    elif shape_type == "motion_stream":
        angle = math.radians(float(geometry.get("dominant_angle_degrees") or 0.0))
        center = np.array([width * 0.5, height * 0.58], dtype=np.float32)
        vector = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        for offset in np.linspace(-0.32, 0.32, 9):
            start = center + np.array([offset * width, 0.0]) - vector * (width * 0.18)
            end = start + vector * (width * (0.22 + 0.05 * math.sin(phase * math.tau)))
            cv2.arrowedLine(frame, tuple(start.astype(int)), tuple(end.astype(int)), (255, 170, 70), 1, cv2.LINE_AA, tipLength=0.22)
    elif shape_type in {"depth_wall", "light_cone", "atmosphere_volume"}:
        x, y, w, h = geometry.get("normalized_xywh", [0.1, 0.1, 0.8, 0.6])
        color = (95, 210, 180) if shape_type == "depth_wall" else (190, 130, 255)
        cv2.rectangle(frame, (int(x * width), int(y * height)), (int((x + w) * width), int((y + h) * height)), color, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (int(x * width), int(y * height)), (int((x + w) * width), int((y + h) * height)), color, -1)
        frame[:] = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)


def render_geometry_overlay_frame(scene: dict[str, Any], *, frame_index: int, total_frames: int, width: int, height: int) -> np.ndarray:
    phase = frame_index / max(1, total_frames - 1)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    vertical = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1)
    frame[:, :, 0] = np.clip(7 + 24 * (1.0 - vertical), 0, 255).astype(np.uint8)
    frame[:, :, 1] = np.clip(10 + 32 * (1.0 - vertical), 0, 255).astype(np.uint8)
    frame[:, :, 2] = np.clip(14 + 42 * (1.0 - vertical), 0, 255).astype(np.uint8)
    for shape in scene.get("shape_units") or []:
        _draw_shape(frame, shape, width, height, phase)
    panel_w = max(230, int(width * 0.34))
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (panel_w, height - 8), (5, 8, 10), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    _put_text(frame, "GEOMETRY LOGGER", 18, 30, color=(245, 245, 245), scale=0.48)
    _put_text(frame, f"run: {scene.get('run_id')}", 18, 50, color=(165, 210, 230), scale=0.34)
    y = 74
    for lane in scene.get("logger_lane_visibility_plan", [])[:14]:
        status_color = (80, 230, 150) if lane.get("status") == "present" else (92, 105, 115)
        cv2.circle(frame, (22, y - 4), 3, status_color, -1)
        _put_text(frame, f"{lane['lane_id']} -> {lane['visible_as']}", 32, y, color=(205, 220, 225), scale=0.30)
        y += 15
    footer_y = height - 28
    _put_text(frame, "raw refs + true metrics bound to each shape", 18, footer_y, color=(255, 210, 120), scale=0.32)
    return frame


def _render_preview_video(scene: dict[str, Any], output_path: Path, *, duration: float, fps: int, width: int, height: int) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(round(float(duration) * int(fps))))
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_index in range(frame_count):
            frame = render_geometry_overlay_frame(scene, frame_index=frame_index, total_frames=frame_count, width=width, height=height)
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    return {
        "path": str(output_path),
        "fps": int(fps),
        "width": int(width),
        "height": int(height),
        "duration_seconds": float(duration),
        "frame_count": frame_count,
    }


def write_geometry_generation_run(call: dict[str, Any], *, storage_root: str | Path = "storage") -> dict[str, Any]:
    storage_root = Path(storage_root)
    run_id = _safe_id(str(call.get("run_id") or "geometry_generation"))
    bundle_keys = [
        "meter_grid_profile",
        "angular_seismic_profile",
        "state_focus_profile",
        "element_creation_profile",
        "truedepth_profile",
        "truedepth_signature",
        "atmosphere_profile",
        "weather_profile",
        "trueaudio_manifest",
        "trueaudio_profile",
        "driving_profile",
        "awareness_profile",
        "worker_forge_manifest",
        "manifest",
        "receipt",
    ]
    bundle = {
        key: call.get(key)
        for key in bundle_keys
        if call.get(key) is not None and call.get(key) != ""
    }
    scene = build_geometry_scene_from_logger_bundle(bundle, run_id=run_id)
    library = build_big_shape_library()
    artifact_root = storage_root / "artifacts" / "geometry_generation"
    manifest_root = storage_root / "manifests" / "geometry_generation"
    receipt_root = storage_root / "receipts" / "geometry_generation"
    scene_path = artifact_root / f"{run_id}_geometry_scene.json"
    library_path = artifact_root / f"{run_id}_big_shape_library.json"
    _write_json(scene_path, scene)
    _write_json(library_path, library)
    preview: dict[str, Any] | None = None
    if bool(call.get("render_preview", False)):
        output_root = Path(str(call.get("output_root") or "outputs/geometry_generation"))
        preview_path = output_root / run_id / f"{run_id}_logger_overlay.mp4"
        preview = _render_preview_video(
            scene,
            preview_path,
            duration=float(call.get("duration") or 10.0),
            fps=int(call.get("fps") or 30),
            width=int(call.get("width") or 1280),
            height=int(call.get("height") or 720),
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "scene_json": str(scene_path),
        "scene_sha256": scene["scene_sha256"],
        "shape_library_json": str(library_path),
        "shape_library_sha256": library["library_sha256"],
        "shape_count": scene["shape_count"],
        "logger_lane_count": len(KNOWN_LOGGER_LANES),
        "logger_lane_visibility_plan": scene["logger_lane_visibility_plan"],
        "preview_video": preview,
        "boundary": scene["boundary"],
    }
    manifest["manifest_sha256"] = stable_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    manifest_path = manifest_root / f"{run_id}_manifest.json"
    _write_json(manifest_path, manifest)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "created_at_utc": utc_now(),
        "tool": "truevision_geometry_generation_engine",
        "run_id": run_id,
        "scene_json": str(scene_path),
        "manifest_json": str(manifest_path),
        "shape_library_json": str(library_path),
        "scene_sha256": scene["scene_sha256"],
        "shape_count": scene["shape_count"],
        "logger_lane_count": len(KNOWN_LOGGER_LANES),
        "preview_video": preview["path"] if preview else None,
        "boundary": {
            "object_truth_promoted": False,
            "raw_video_copied": False,
            "generated_media_is_evidence": False,
            "shapes_are_data_containers": True,
        },
        "state_source_law": list(STATE_SOURCE_LAW_LINES),
    }
    receipt["receipt_sha256"] = stable_hash({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    receipt_path = receipt_root / f"{run_id}_receipt.json"
    _write_json(receipt_path, receipt)
    result = {
        "run_id": run_id,
        "scene_json": str(scene_path),
        "shape_library_json": str(library_path),
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "shape_count": scene["shape_count"],
        "logger_lane_count": len(KNOWN_LOGGER_LANES),
    }
    if preview:
        result["preview_video"] = preview["path"]
    return result
