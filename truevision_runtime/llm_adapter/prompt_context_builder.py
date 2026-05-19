from __future__ import annotations

from copy import deepcopy
from typing import Any

from truevision_runtime.state_patterns.audio_video_patterns import AUDIO_SIGNAL_CHANNELS, list_state_patterns
from truevision_runtime.av_tools.av_tool_registry import list_av_tools


def build_prompt_context(prompt: str, project_context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = deepcopy(project_context or {})
    allowed_tools = [tool["name"] for tool in list_av_tools()]
    return {
        "model_endpoint": {
            "role": "draft_generator_only",
            "trusted": False,
        },
        "prompt": str(prompt),
        "project_context": context,
        "allowed_av_tools": allowed_tools,
        "audio_signal_channels": list(AUDIO_SIGNAL_CHANNELS),
        "state_pattern_library": list_state_patterns(),
        "schema": {
            "request_kind": "truevision_state_media_draft",
            "required": ["scene", "renderer", "media", "timeline", "safety_boundary"],
            "preferred_renderer_for_wav": "edge_audio_river",
            "preferred_renderer_for_audio_geometry": "audio_geometry_field",
            "timeline": {
                "duration_seconds": "positive number; probe WAV duration first when unknown",
                "fps": "integer from 1 to 120",
            },
            "media": {
                "audio_path": "local WAV/MP3 path when rendering audio-reactive visuals",
                "sync_to_audio": True,
            },
        },
        "runtime_notes": [
            ".wav files can drive videos through audio_probe_duration, audio_analyze_levels, audio_extract_features, template_create, template_from_audio_signals, and video_render_preview.",
            "The model drafts JSON only. The app trusts only validated state JSON.",
            "Use peaks, valleys, rising energy, and section energy to choose state patterns instead of inventing blind.",
            "Generated media is synthetic state media, not evidence.",
        ],
        "trust_boundary": {
            "model_output_is_trusted": False,
            "validator_decides": True,
            "generator_receives_only_validated_state": True,
            "tool_calls_require_policy": True,
        },
    }


def build_system_prompt() -> str:
    return "\n".join(
        [
            "You are a model-neutral PromptToStateAdapter draft generator.",
            "Return only JSON. No markdown. No prose.",
            "Your output is not trusted until validated.",
            "Use request_kind=truevision_state_media_draft.",
            "Use renderer=edge_audio_river for WAV-driven color river videos unless the operator asks otherwise.",
            "Use renderer=audio_geometry_field when peaks and valleys should drive random geometry patterns.",
            "Never claim generated media is evidence.",
            "Never request non-audio/video tools.",
        ]
    )
