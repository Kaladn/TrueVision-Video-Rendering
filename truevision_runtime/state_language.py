from __future__ import annotations

from typing import Any


STATE_LANGUAGE_VERBS: tuple[str, ...] = ("witness", "profile", "plan", "replay", "surface")


STAGE_ALIASES: dict[str, str] = {
    "observe": "witness",
    "observes": "witness",
    "observes_state": "witness",
    "record": "witness",
    "recording": "witness",
    "capture": "witness",
    "capture_state": "witness",
    "witness": "witness",
    "abstract": "profile",
    "abstracts_behavior": "profile",
    "extract": "profile",
    "profile": "profile",
    "recognize": "profile",
    "recognition": "profile",
    "generate": "plan",
    "generation": "plan",
    "generates_state": "plan",
    "plan": "plan",
    "replay": "replay",
    "replays": "replay",
    "render": "surface",
    "rendering": "surface",
    "renders_media": "surface",
    "surface": "surface",
}


DIRECTION_STAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "forward": ("witness", "profile"),
    "forward_observation": ("witness", "profile"),
    "observation": ("witness", "profile"),
    "watch": ("witness", "profile"),
    "witness": ("witness",),
    "record": ("witness",),
    "profile": ("profile",),
    "reverse": ("plan", "replay", "surface"),
    "reverse_generation": ("plan", "replay", "surface"),
    "generate": ("plan", "replay", "surface"),
    "generation": ("plan", "replay", "surface"),
    "plan": ("plan",),
    "replay": ("replay",),
    "render": ("surface",),
    "surface": ("surface",),
}


MEDIA_OUTPUT_TOKENS = ("media", "video", "audio", "wav", "mp4", "webm", "mov", "mkv", "preview")


def _token(value: Any) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value or "")).strip("_")


def normalize_state_stage(value: str) -> str:
    return STAGE_ALIASES.get(_token(value), "")


def required_stages_for_direction(direction: str) -> tuple[str, ...]:
    return DIRECTION_STAGE_ALIASES.get(_token(direction), ())


def _bool_from(tool: dict[str, Any], canonical_key: str, legacy_key: str, *, fallback: bool = False) -> bool:
    if canonical_key in tool:
        return bool(tool.get(canonical_key))
    return bool(tool.get(legacy_key, fallback))


def _infer_behavior_family(tool: dict[str, Any]) -> str:
    explicit = _token(tool.get("behavior_family"))
    if explicit:
        return explicit
    profiles = tool.get("behavior_profiles_supported")
    if isinstance(profiles, list):
        for item in profiles:
            token = _token(item)
            if token and not token.endswith("_contract"):
                return token
    tool_id = _token(tool.get("tool_id"))
    if "fog" in tool_id or "atmosphere" in tool_id or "weather" in tool_id:
        return "fog_reveal"
    if "lightning" in tool_id:
        return "branching_discharge"
    if "audio" in tool_id or "speech" in tool_id:
        return "audio_state"
    if "geometry" in tool_id:
        return "geometry_shape"
    if "meter" in tool_id:
        return "meter_grid"
    if "state_loop" in tool_id:
        return "state_loop_contract"
    if "source_law" in tool_id or "receipt" in tool_id:
        return "state_contract"
    return "unclassified"


def _has_media_output(tool: dict[str, Any]) -> bool:
    outputs = tool.get("media_outputs_optional")
    if isinstance(outputs, list) and outputs:
        return True
    output_types = tool.get("output_types")
    if not isinstance(output_types, list):
        return False
    text = " ".join(str(item).lower() for item in output_types)
    return any(token in text for token in MEDIA_OUTPUT_TOKENS)


def build_state_language(tool: dict[str, Any]) -> dict[str, Any]:
    can_witness = _bool_from(tool, "can_witness", "observes_state")
    can_profile = _bool_from(tool, "can_profile", "abstracts_behavior")
    can_plan = bool(tool.get("can_plan")) if "can_plan" in tool else bool(tool.get("generates_state", False))
    can_replay = bool(tool.get("can_replay")) if "can_replay" in tool else bool(tool.get("generates_state", False))
    can_surface = _bool_from(tool, "can_surface", "renders_media")
    copies_source_media = bool(tool.get("copies_source_media", False))
    raw_media_saved = bool(tool.get("raw_video_saved", False) or tool.get("raw_media_saved", False))
    stages = [
        stage
        for stage, enabled in (
            ("witness", can_witness),
            ("profile", can_profile),
            ("plan", can_plan),
            ("replay", can_replay),
            ("surface", can_surface),
        )
        if enabled
    ]
    media_is_optional_surface = bool(can_surface or _has_media_output(tool))
    return {
        "behavior_family": _infer_behavior_family(tool),
        "can_witness": can_witness,
        "can_profile": can_profile,
        "can_plan": can_plan,
        "can_replay": can_replay,
        "can_surface": can_surface,
        "supported_stages": stages,
        "copies_source_media": copies_source_media,
        "raw_media_saved": raw_media_saved,
        "media_is_optional_surface": media_is_optional_surface,
        "media_is_source_truth": False,
        "source_truth_compliant": bool(tool.get("source_truth_compliant", False)),
    }


def supports_any_stage(tool_or_language: dict[str, Any], stages: tuple[str, ...] | list[str]) -> bool:
    language = tool_or_language if "supported_stages" in tool_or_language else build_state_language(tool_or_language)
    supported = set(language.get("supported_stages") or [])
    return bool(supported.intersection(stages))

