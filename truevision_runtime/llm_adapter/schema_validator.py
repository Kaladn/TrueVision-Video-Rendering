from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    normalized: dict[str, Any]


ALLOWED_RENDERERS = {"edge_audio_river", "state_scene_generator", "path_tracer"}


def _as_object(value: Any, field: str, errors: list[str]) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    errors.append(f"{field} must be an object")
    return {}


def validate_state_request(payload: Any) -> ValidationResult:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ValidationResult(False, ["state request must be a JSON object"], {})

    normalized = deepcopy(payload)
    normalized["request_kind"] = str(normalized.get("request_kind") or "truevision_state_media_draft")
    if normalized["request_kind"] != "truevision_state_media_draft":
        errors.append("request_kind must be truevision_state_media_draft")

    scene = _as_object(normalized.get("scene"), "scene", errors)
    if not str(scene.get("name") or scene.get("description") or "").strip():
        errors.append("scene.name or scene.description is required")
    normalized["scene"] = scene

    renderer = str(normalized.get("renderer") or "edge_audio_river")
    if renderer not in ALLOWED_RENDERERS:
        errors.append(f"renderer must be one of {sorted(ALLOWED_RENDERERS)}")
    normalized["renderer"] = renderer

    media = _as_object(normalized.get("media", {}), "media", errors)
    media["audio_path"] = str(media.get("audio_path") or "")
    media["sync_to_audio"] = bool(media.get("sync_to_audio", True))
    normalized["media"] = media

    timeline = _as_object(normalized.get("timeline"), "timeline", errors)
    try:
        duration = float(timeline.get("duration_seconds"))
    except (TypeError, ValueError):
        duration = 0.0
    try:
        fps = int(timeline.get("fps"))
    except (TypeError, ValueError):
        fps = 0
    if duration <= 0:
        errors.append("timeline.duration_seconds must be > 0")
    if fps < 1 or fps > 120:
        errors.append("timeline.fps must be between 1 and 120")
    timeline["duration_seconds"] = round(max(duration, 0.0), 6)
    timeline["fps"] = fps
    timeline["frame_count"] = max(0, int(round(max(duration, 0.0) * max(fps, 0))))
    normalized["timeline"] = timeline

    normalized["visual_parameters"] = _as_object(normalized.get("visual_parameters", {}), "visual_parameters", errors)
    boundary = _as_object(normalized.get("safety_boundary", {}), "safety_boundary", errors)
    boundary["generated_state_media"] = bool(boundary.get("generated_state_media", True))
    boundary["evidence"] = bool(boundary.get("evidence", False))
    if boundary["evidence"]:
        errors.append("safety_boundary.evidence must be false")
    normalized["safety_boundary"] = boundary

    return ValidationResult(not errors, errors, normalized)
