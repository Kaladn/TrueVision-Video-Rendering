from __future__ import annotations

from copy import deepcopy
from typing import Any


AUDIO_SIGNAL_CHANNELS = (
    "level",
    "peak_abs",
    "dbfs",
    "delta",
    "peak_event",
    "valley_event",
    "section_energy",
)


STATE_PATTERNS: tuple[dict[str, Any], ...] = (
    {
        "pattern_id": "pulse_rings",
        "family": "geometry",
        "purpose": "Peaks trigger expanding rings on a black field.",
        "driven_by": ["peak_event", "level"],
        "parameters": {
            "radius": "level",
            "emission_strength": "peak_event",
            "decay_seconds": 0.42,
        },
    },
    {
        "pattern_id": "random_geometry_shards",
        "family": "geometry",
        "purpose": "Peaks spawn deterministic random triangles or line shards.",
        "driven_by": ["peak_event", "delta"],
        "parameters": {
            "spawn_count": "peak_event",
            "rotation_speed": "delta",
            "seed_source": "time_seconds",
        },
    },
    {
        "pattern_id": "quiet_valley_drift",
        "family": "motion",
        "purpose": "Valleys slow motion and let geometry drift.",
        "driven_by": ["valley_event", "section_energy"],
        "parameters": {
            "time_scale": "valley_event",
            "blur_amount": "section_energy_inverse",
        },
    },
    {
        "pattern_id": "rising_energy_expansion",
        "family": "camera",
        "purpose": "Rising level expands geometry and nudges camera push.",
        "driven_by": ["delta", "level"],
        "parameters": {
            "scale": "level",
            "camera_push": "positive_delta",
        },
    },
    {
        "pattern_id": "high_energy_edge_shimmer",
        "family": "surface",
        "purpose": "High activity adds edge shimmer and color pressure.",
        "driven_by": ["level", "peak_abs"],
        "parameters": {
            "edge_activity": "peak_abs",
            "color_pressure": "level",
        },
    },
)


def list_state_patterns() -> list[dict[str, Any]]:
    return [deepcopy(pattern) for pattern in STATE_PATTERNS]


def choose_patterns_for_signal(summary: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    summary = summary or {}
    peak_count = int(summary.get("peak_count") or 0)
    valley_count = int(summary.get("valley_count") or 0)
    average_level = float(summary.get("average_level") or 0.0)
    selected = ["rising_energy_expansion", "high_energy_edge_shimmer"]
    if peak_count > 0:
        selected.extend(["pulse_rings", "random_geometry_shards"])
    if valley_count > 0 or average_level < 0.35:
        selected.append("quiet_valley_drift")
    patterns = {pattern["pattern_id"]: pattern for pattern in STATE_PATTERNS}
    return [deepcopy(patterns[pattern_id]) for pattern_id in selected if pattern_id in patterns]
