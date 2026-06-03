from __future__ import annotations

from typing import Any

from .effect_need_frame import EffectNeedFrame


def plan_timing(frame: EffectNeedFrame, *, emphasis: str = "default") -> dict[str, Any]:
    start = frame.start_seconds
    end = frame.end_seconds
    duration = frame.duration_seconds
    if emphasis in {"impact", "shockwave", "fire"}:
        effect_start = start + duration * 0.18
        effect_end = start + duration * 0.72
        curve = "anticipation_fast_rise_peak_decay"
    elif emphasis in {"fog", "atmosphere", "reveal", "growth"}:
        effect_start = start + duration * 0.35
        effect_end = end
        curve = "slow_reveal_residue"
    else:
        effect_start = start
        effect_end = end
        curve = "scene_span"
    peak = max(effect_start, min(effect_end, frame.peak_seconds))
    return {
        "start": round(effect_start, 3),
        "end": round(effect_end, 3),
        "peak": round(peak, 3),
        "curve": curve,
    }


def plan_strength(score: float, *, emphasis: str = "default") -> dict[str, float]:
    base = max(0.0, min(1.0, float(score)))
    if emphasis in {"impact", "shockwave", "fire"}:
        peak = min(1.0, 0.45 + base * 0.55)
        initial = max(0.08, peak * 0.28)
        final = max(0.02, peak * 0.18)
    elif emphasis in {"fog", "atmosphere", "reveal", "growth"}:
        peak = min(0.82, 0.30 + base * 0.45)
        initial = min(0.72, peak * 0.85)
        final = max(0.08, peak * 0.22)
    else:
        peak = min(0.75, 0.25 + base * 0.45)
        initial = max(0.05, peak * 0.35)
        final = max(0.05, peak * 0.20)
    return {
        "initial": round(initial, 3),
        "peak": round(peak, 3),
        "final": round(final, 3),
    }

