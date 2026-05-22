#!/usr/bin/env python3
"""Render a locked-character flame-walk forecast from ten source states.

The source story is ten deterministic page-like video states over ten seconds.
The output timeline extends to twenty seconds by projecting environmental state
with a 6-1-6 temporal cloud. Characters remain locked; rain, flame, smoke,
light, reflections, and embers move.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from truevision_edge_audio_river import capture_hardware, decode_audio_mono, measure_audio_features, sha256_file
from truevision_runtime.rendering.template_renderer import memory_snapshot


DEFAULT_AUDIO = Path(
    r"C:\Users\mydyi\OneDrive\Documents\Desktop\Album_Builds\Machine_Dread_Album_Sequenced\01_ordered_audio\10 - Burn The Sky.mp3"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/flame_walk_forecast")
DEFAULT_RUN_ID = "child_to_flame_walk_616_forecast"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def slug(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return clean.strip("_")[:96] or DEFAULT_RUN_ID


def build_source_sequence(*, source_seconds: int = 10, source_state_count: int = 10) -> list[dict[str, Any]]:
    if source_state_count < 2:
        raise ValueError("source_state_count must be at least 2")
    states: list[dict[str, Any]] = []
    for index in range(source_state_count):
        progress = index / max(1, source_state_count - 1)
        if progress < 0.38:
            scene_phase = "child_watches_father_walk_away"
            child_presence = 1.0 - progress * 0.55
            father_distance = 0.18 + progress * 0.82
            pair_unity = 0.0
        elif progress < 0.68:
            scene_phase = "memory_bridge_through_fire"
            bridge = (progress - 0.38) / 0.30
            child_presence = 0.46 * (1.0 - bridge)
            father_distance = 0.60 + bridge * 0.18
            pair_unity = bridge * 0.55
        else:
            scene_phase = "walking_toward_flame_together"
            bridge = (progress - 0.68) / 0.32
            child_presence = 0.0
            father_distance = 0.78
            pair_unity = 0.55 + bridge * 0.45

        states.append(
            {
                "source_state_index": index,
                "time_seconds": round(index * (source_seconds / source_state_count), 6),
                "norm": round(progress, 6),
                "scene_phase": scene_phase,
                "child_presence": round(child_presence, 6),
                "father_distance": round(father_distance, 6),
                "pair_unity": round(pair_unity, 6),
                "flame_pressure": round(0.22 + progress * 0.72, 6),
                "smoke_density": round(0.36 + progress * 0.34, 6),
                "rain_pressure": round(0.62 - progress * 0.12, 6),
                "reflection_pressure": round(0.38 + progress * 0.36, 6),
                "ember_pressure": round(0.20 + progress * 0.65, 6),
                "camera_push": round(progress * 0.18, 6),
                "character_motion": "locked_pose_environment_moves",
                "source_authority": "authored_state_sequence",
            }
        )
    return states


def _lerp(a: float, b: float, u: float) -> float:
    return float(a) * (1.0 - u) + float(b) * u


def _audio_at(audio_features: list[dict[str, Any]], frame_index: int) -> dict[str, float]:
    if not audio_features:
        return {"rms": 0.0, "bass": 0.0, "mid": 0.0, "high": 0.0, "beat": 0.0}
    if frame_index < len(audio_features):
        found = audio_features[frame_index]
    else:
        found = audio_features[-1]
    return {key: float(found.get(key, 0.0)) for key in ["rms", "bass", "mid", "high", "beat"]}


def _numeric_fields(states: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for key, value in states[0].items():
        if key in {"source_state_index", "time_seconds", "norm"}:
            continue
        if isinstance(value, (int, float)):
            fields.append(key)
    return fields


def forecast_timeline_616(
    source_states: list[dict[str, Any]],
    audio_features: list[dict[str, Any]],
    *,
    total_seconds: int = 20,
    fps: int = 24,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not source_states:
        raise ValueError("source_states cannot be empty")
    frame_count = int(total_seconds * fps)
    source_window_seconds = 10.0
    max_source_t = float(source_states[-1]["time_seconds"])
    fields = _numeric_fields(source_states)
    timeline: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    for frame_index in range(frame_count):
        time_seconds = frame_index / fps
        audio = _audio_at(audio_features, frame_index)
        if time_seconds <= max_source_t:
            source_pos = (time_seconds / max_source_t) * (len(source_states) - 1) if max_source_t else 0.0
            left = int(math.floor(source_pos))
            right = min(len(source_states) - 1, left + 1)
            u = source_pos - left
            state = dict(source_states[left])
            for field in fields:
                state[field] = round(_lerp(float(source_states[left][field]), float(source_states[right][field]), u), 6)
            state["scene_phase"] = source_states[right]["scene_phase"] if u > 0.55 else source_states[left]["scene_phase"]
            forecast_kind = "source_interpolated"
            center_index = left
        else:
            extension = (time_seconds - max_source_t) / max(1e-6, source_window_seconds)
            history = source_states[-6:]
            trend: dict[str, float] = {}
            for field in fields:
                trend[field] = (float(history[-1][field]) - float(history[0][field])) / max(1, len(history) - 1)
            state = dict(source_states[-1])
            for field in fields:
                state[field] = round(float(source_states[-1][field]) + trend[field] * extension * 2.4, 6)
            state["child_presence"] = 0.0
            state["pair_unity"] = min(1.0, max(float(state["pair_unity"]), 0.94))
            state["flame_pressure"] = min(1.0, max(float(state["flame_pressure"]), 0.82 + extension * 0.12))
            state["ember_pressure"] = min(1.0, max(float(state["ember_pressure"]), 0.84))
            state["scene_phase"] = "walking_toward_flame_together"
            forecast_kind = "six_one_six_projected"
            center_index = len(source_states) - 1

        prior = source_states[max(0, center_index - 6) : center_index]
        future = source_states[center_index + 1 : min(len(source_states), center_index + 7)]
        beat = audio["beat"]
        bass = audio["bass"]
        high = audio["high"]
        state.update(
            {
                "frame_index": frame_index,
                "time_seconds": round(time_seconds, 6),
                "forecast_kind": forecast_kind,
                "forecast_method": "six_one_six_temporal_state_projection",
                "prior_count": len(prior),
                "future_count": len(future) if forecast_kind == "source_interpolated" else 0,
                "rms": round(audio["rms"], 6),
                "bass": round(bass, 6),
                "mid": round(audio["mid"], 6),
                "high": round(high, 6),
                "beat": round(beat, 6),
                "flame_lick_pressure": round(min(1.0, float(state["flame_pressure"]) * 0.58 + beat * 0.62 + bass * 0.34), 6),
                "lightning_pressure": round(min(1.0, max(0.0, beat * 0.72 + high * 0.34 - 0.35)), 6),
                "environment_motion": "beat_flame_rain_smoke_reflection_only",
                "character_motion": "locked_pose_environment_moves",
            }
        )
        timeline.append(state)
        trace.append(
            {
                "frame_index": frame_index,
                "time_seconds": round(time_seconds, 6),
                "forecast_kind": forecast_kind,
                "center_source_index": center_index,
                "prior_count": state["prior_count"],
                "future_count": state["future_count"],
                "flame_lick_pressure": state["flame_lick_pressure"],
                "confidence": round(0.92 if forecast_kind == "source_interpolated" else max(0.64, 0.86 - (time_seconds - max_source_t) * 0.012), 6),
            }
        )
    return timeline, trace


def _safe_line(frame: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def _draw_background(frame: np.ndarray, state: dict[str, Any]) -> None:
    h, w = frame.shape[:2]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    flame = float(state["flame_pressure"])
    lightning = float(state["lightning_pressure"])
    storm = np.clip(1.0 - y * 1.08, 0.0, 1.0)
    fire_core = np.clip(1.0 - np.abs(x - 0.55) * 1.7, 0.0, 1.0) * np.clip(1.0 - y * 1.25, 0.0, 1.0)
    frame[:, :, 0] = np.clip(10 + storm * 44 + lightning * 38, 0, 255).astype(np.uint8)
    frame[:, :, 1] = np.clip(8 + storm * 20 + fire_core * 34 * flame, 0, 255).astype(np.uint8)
    frame[:, :, 2] = np.clip(10 + storm * 18 + fire_core * 130 * flame + lightning * 34, 0, 255).astype(np.uint8)

    horizon = int(h * 0.43)
    rng = np.random.default_rng(6061)
    for index in range(42):
        bx = int(w * index / 41.0 + math.sin(index * 2.3) * w * 0.012)
        bw = int(w * (0.010 + 0.022 * rng.random()))
        bh = int(h * (0.14 + 0.35 * rng.random()))
        top = max(0, horizon - bh)
        cv2.rectangle(frame, (bx, top), (min(w, bx + bw), h), (5, 7, 11), -1)
        if index % 4 == 0:
            cv2.rectangle(frame, (bx + bw // 2, top + bh // 3), (min(w, bx + bw // 2 + 2), min(h, top + bh // 3 + 10)), (40, 70, 145), -1)

    if lightning > 0.48:
        start_x = int(w * (0.42 + 0.16 * math.sin(float(state["time_seconds"]) * 1.7)))
        points = [(start_x, 0)]
        for step in range(1, 10):
            points.append((start_x + int(math.sin(step * 1.9) * w * 0.035), int(h * step / 11)))
        _safe_line(frame, points, (230, 235, 255), max(1, int(2 + 3 * lightning)))
        glow = cv2.GaussianBlur(frame.copy(), (0, 0), sigmaX=2 + 10 * lightning)
        cv2.addWeighted(glow, 0.10 * lightning, frame, 1.0, 0, dst=frame)


def _draw_flames(frame: np.ndarray, state: dict[str, Any]) -> None:
    h, w = frame.shape[:2]
    flame = float(state["flame_lick_pressure"])
    t = float(state["time_seconds"])
    fire = np.zeros_like(frame)
    base_y = int(h * 0.68)
    for index in range(90):
        x0 = int(w * ((index * 37) % 89) / 89.0)
        local = 0.55 + 0.45 * math.sin(index * 1.21 + t * (2.3 + flame * 3.2))
        height = int(h * (0.07 + 0.22 * flame * local))
        width = int(w * (0.006 + 0.020 * local))
        y0 = base_y + int(h * 0.08 * math.sin(index))
        pts = np.asarray(
            [
                (x0 - width, y0),
                (x0 + width, y0),
                (x0 + int(width * 0.45 * math.sin(t + index)), max(0, y0 - height)),
            ],
            dtype=np.int32,
        )
        color = (0, int(70 + 80 * local), int(140 + 105 * flame))
        cv2.fillConvexPoly(fire, pts, color, cv2.LINE_AA)
    fire = cv2.GaussianBlur(fire, (0, 0), sigmaX=4 + 5 * flame, sigmaY=8 + 11 * flame)
    cv2.addWeighted(fire, 0.78, frame, 1.0, 0, dst=frame)


def _draw_rain_smoke_reflections(frame: np.ndarray, state: dict[str, Any]) -> None:
    h, w = frame.shape[:2]
    t = float(state["time_seconds"])
    rain_strength = float(state["rain_pressure"])
    smoke_density = float(state["smoke_density"])
    reflection = float(state["reflection_pressure"])

    pavement_y = int(h * 0.61)
    cv2.rectangle(frame, (0, pavement_y), (w, h), (5, 6, 8), -1)
    yy = np.linspace(0.0, 1.0, h - pavement_y, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    ripple = np.sin((xx * 13.0 + yy * 2.6 + t * 0.35) * math.tau).astype(np.float32)
    hot_reflect = np.clip(1.0 - np.abs(xx - 0.55) * 2.0, 0.0, 1.0) * np.clip(1.0 - yy * 0.8 + ripple * 0.04, 0.0, 1.0)
    layer = np.zeros((h - pavement_y, w, 3), dtype=np.float32)
    layer[:, :, 0] = 22 + 30 * hot_reflect
    layer[:, :, 1] = 24 + 78 * hot_reflect
    layer[:, :, 2] = 28 + 150 * hot_reflect
    alpha = np.clip(hot_reflect * (0.15 + 0.45 * reflection), 0.0, 0.72)
    target = frame[pavement_y:, :, :].astype(np.float32)
    frame[pavement_y:, :, :] = np.clip(target * (1.0 - alpha[:, :, None]) + layer * alpha[:, :, None], 0, 255).astype(np.uint8)

    smoke = np.zeros_like(frame, dtype=np.float32)
    rng = np.random.default_rng(7013)
    density = rng.random((28, 46), dtype=np.float32)
    density = cv2.resize(density, (w, h), interpolation=cv2.INTER_CUBIC)
    density = np.roll(density, int(t * w * 0.018), axis=1)
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=20, sigmaY=11)
    density = np.clip((density - 0.36) * 1.8, 0.0, 1.0) * smoke_density
    smoke[:, :, 0] = 54 + 22 * density
    smoke[:, :, 1] = 52 + 18 * density
    smoke[:, :, 2] = 58 + 26 * density
    alpha_smoke = np.clip(density * 0.34, 0.0, 0.48)
    frame[:] = np.clip(frame.astype(np.float32) * (1 - alpha_smoke[:, :, None]) + smoke * alpha_smoke[:, :, None], 0, 255).astype(np.uint8)

    rain = np.zeros_like(frame)
    rng = np.random.default_rng(1801 + int(t * 24))
    for _ in range(int(160 * rain_strength)):
        x = int(rng.random() * w)
        y = int(rng.random() * h)
        length = int(h * (0.025 + 0.035 * rng.random()))
        cv2.line(rain, (x, y), (x - int(w * 0.016), min(h - 1, y + length)), (95, 98, 105), 1, cv2.LINE_AA)
    rain = cv2.GaussianBlur(rain, (0, 0), sigmaX=0.4, sigmaY=0.8)
    cv2.addWeighted(rain, 0.62, frame, 1.0, 0, dst=frame)


def _draw_embers(frame: np.ndarray, state: dict[str, Any]) -> None:
    h, w = frame.shape[:2]
    t = float(state["time_seconds"])
    ember = float(state["ember_pressure"])
    overlay = np.zeros_like(frame)
    for index in range(int(90 + 120 * ember)):
        drift = (t * (0.025 + ember * 0.025) + index * 0.017) % 1.0
        x = int(w * ((index * 0.061 + 0.15 * math.sin(index + t)) % 1.0))
        y = int(h * (0.88 - drift * 0.92))
        if y < 0:
            continue
        r = max(1, int(1 + 3 * ((index % 5) / 4) * ember))
        cv2.circle(overlay, (x, y), r, (20, 74 + int(70 * ember), 180 + int(70 * ember)), -1, cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=1.2 + 2.0 * ember)
    cv2.addWeighted(overlay, 0.72, frame, 1.0, 0, dst=frame)


def _draw_child_and_father(frame: np.ndarray, state: dict[str, Any]) -> None:
    h, w = frame.shape[:2]
    child = float(state["child_presence"])
    pair = float(state["pair_unity"])
    father_distance = float(state["father_distance"])
    if child <= 0.02:
        return
    alpha = child
    overlay = np.zeros_like(frame)
    child_x = int(w * 0.24)
    child_ground = int(h * 0.78)
    scale = h / 720.0
    cv2.circle(overlay, (child_x, child_ground - int(90 * scale)), int(18 * scale), (4, 5, 8), -1, cv2.LINE_AA)
    cv2.ellipse(overlay, (child_x, child_ground - int(45 * scale)), (int(24 * scale), int(50 * scale)), 0, 0, 360, (3, 4, 7), -1, cv2.LINE_AA)
    cv2.circle(overlay, (child_x + int(26 * scale), child_ground - int(22 * scale)), int(10 * scale), (16, 12, 10), -1, cv2.LINE_AA)

    father_x = int(w * (0.48 + father_distance * 0.08))
    father_ground = int(h * (0.78 - father_distance * 0.06))
    father_scale = scale * (0.78 - father_distance * 0.18)
    cv2.circle(overlay, (father_x, father_ground - int(175 * father_scale)), int(26 * father_scale), (2, 3, 5), -1, cv2.LINE_AA)
    body = np.asarray(
        [
            (father_x - int(55 * father_scale), father_ground - int(140 * father_scale)),
            (father_x + int(55 * father_scale), father_ground - int(140 * father_scale)),
            (father_x + int(28 * father_scale), father_ground),
            (father_x - int(28 * father_scale), father_ground),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(overlay, body, (2, 3, 5), cv2.LINE_AA)
    mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) > 0
    rim = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
    rim &= ~mask
    frame[rim] = np.clip(frame[rim].astype(np.float32) * 0.45 + np.asarray((28, 72, 150), dtype=np.float32) * alpha, 0, 255).astype(np.uint8)
    frame[mask] = np.clip(frame[mask].astype(np.float32) * (0.12 + 0.20 * (1.0 - alpha)), 0, 255).astype(np.uint8)


def _draw_pair_toward_flame(frame: np.ndarray, state: dict[str, Any]) -> None:
    pair = float(state["pair_unity"])
    if pair <= 0.03:
        return
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)
    scale = h / 720.0
    ground = int(h * 0.87)
    for side, height_mul, shoulder_mul, hair in [(-1, 0.92, 0.80, True), (1, 1.08, 1.0, False)]:
        cx = int(w * (0.50 + side * 0.045))
        body_h = int(230 * scale * height_mul)
        shoulder = int(54 * scale * shoulder_mul)
        head_r = int(23 * scale)
        if hair:
            cv2.ellipse(overlay, (cx, ground - body_h), (head_r + int(8 * scale), head_r + int(18 * scale)), 0, 0, 360, (2, 3, 5), -1, cv2.LINE_AA)
        else:
            cv2.circle(overlay, (cx, ground - body_h), head_r, (2, 3, 5), -1, cv2.LINE_AA)
        torso = np.asarray(
            [
                (cx - shoulder, ground - body_h + int(44 * scale)),
                (cx + shoulder, ground - body_h + int(44 * scale)),
                (cx + int(28 * scale), ground - int(32 * scale)),
                (cx - int(28 * scale), ground - int(32 * scale)),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(overlay, torso, (1, 2, 4), cv2.LINE_AA)
        cv2.line(overlay, (cx - int(14 * scale), ground - int(34 * scale)), (cx - int(35 * scale), ground), (1, 2, 4), max(2, int(10 * scale)), cv2.LINE_AA)
        cv2.line(overlay, (cx + int(14 * scale), ground - int(34 * scale)), (cx + int(35 * scale), ground), (1, 2, 4), max(2, int(10 * scale)), cv2.LINE_AA)
    mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) > 0
    aura_mask = cv2.GaussianBlur((mask.astype(np.uint8) * 255), (0, 0), sigmaX=14 + 18 * float(state["flame_lick_pressure"]))
    aura_alpha = np.clip(aura_mask.astype(np.float32) / 255.0 * (0.34 + 0.22 * pair), 0.0, 0.62)
    aura_color = np.zeros_like(frame, dtype=np.float32)
    aura_color[:, :, 0] = 18
    aura_color[:, :, 1] = 82
    aura_color[:, :, 2] = 190
    frame[:] = np.clip(frame.astype(np.float32) * (1.0 - aura_alpha[:, :, None]) + aura_color * aura_alpha[:, :, None], 0, 255).astype(np.uint8)

    rim = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((7, 7), dtype=np.uint8), iterations=1) > 0
    rim &= ~mask
    frame[rim] = np.clip(frame[rim].astype(np.float32) * 0.35 + np.asarray((24, 94, 210), dtype=np.float32) * pair, 0, 255).astype(np.uint8)
    frame[mask] = np.asarray((1, 2, 4), dtype=np.uint8)


def _finish(frame: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    h, w = frame.shape[:2]
    bloom = cv2.GaussianBlur(frame, (0, 0), sigmaX=4 + 8 * float(state["flame_lick_pressure"]))
    cv2.addWeighted(bloom, 0.20 + 0.22 * float(state["beat"]), frame, 1.0, 0, dst=frame)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    vignette = np.clip(1.10 - (x * x + y * y) * 0.52, 0.22, 1.0)
    rng = np.random.default_rng(int(float(state["time_seconds"]) * 18) + 33)
    noise = rng.normal(0, 3.5, size=frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) * vignette[:, :, None] + noise, 0, 255).astype(np.uint8)


def render_flame_walk_frame(width: int, height: int, state: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_background(frame, state)
    _draw_flames(frame, state)
    _draw_rain_smoke_reflections(frame, state)
    _draw_embers(frame, state)
    frame = _finish(frame, state)
    _draw_child_and_father(frame, state)
    _draw_pair_toward_flame(frame, state)
    metadata = {
        "frame_index": int(state.get("frame_index", 0)),
        "time_seconds": round(float(state.get("time_seconds", 0.0)), 6),
        "scene_phase": state.get("scene_phase"),
        "forecast_kind": state.get("forecast_kind"),
        "layers": [
            "burning_city_depth_stack",
            "beat_driven_flame_licks",
            "storm_rain",
            "density_field_smoke",
            "wet_pavement_reflections",
            "locked_character_blocking",
            "ember_ash_particles",
        ],
        "boundary": {
            "generated_state_media": "synthetic_not_evidence",
            "no_external_visual_assets": True,
            "no_lyric_overlay": True,
            "characters_locked_environment_moves": True,
        },
    }
    return frame, metadata


def _machine_cost(start_wall: float, start_cpu: float, memory_start: dict[str, Any]) -> dict[str, Any]:
    elapsed = max(0.000001, time.perf_counter() - start_wall)
    cpu_seconds = max(0.0, time.process_time() - start_cpu)
    logical = max(1, int(__import__("os").cpu_count() or 1))
    return {
        "wall_seconds": round(elapsed, 6),
        "process_cpu_seconds": round(cpu_seconds, 6),
        "avg_cpu_core_equivalent": round(cpu_seconds / elapsed, 6),
        "avg_process_logical_cpu_percent": round((cpu_seconds / (elapsed * logical)) * 100.0, 6),
        "logical_cpu_count": logical,
        "memory_start": memory_start,
        "memory_end": memory_snapshot(),
    }


def render_flame_walk_forecast(
    *,
    audio_path: Path = DEFAULT_AUDIO,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    total_seconds: int = 20,
    source_seconds: int = 10,
    encoder: str = "libx264",
    mux_audio: bool = True,
) -> dict[str, Any]:
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    memory_start = memory_snapshot()
    run_id = slug(run_id)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()

    samples = decode_audio_mono(audio_path, sample_rate=44100, max_seconds=total_seconds)
    audio_features = measure_audio_features(samples, sample_rate=44100, fps=fps)
    source_states = build_source_sequence(source_seconds=source_seconds, source_state_count=10)
    timeline, trace = forecast_timeline_616(source_states, audio_features, total_seconds=total_seconds, fps=fps)

    visual_path = run_dir / f"{run_id}_visual_only.mp4"
    state_path = run_dir / f"{run_id}_frame_state.jsonl"
    trace_path = run_dir / f"{run_id}_616_trace.jsonl"
    final_path = run_dir / f"{run_id}_full_audio.mp4" if mux_audio else visual_path
    thumb_path = run_dir / f"{run_id}_thumbnail.jpg"
    manifest_path = run_dir / f"{run_id}_manifest.json"

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        encoder,
        "-preset",
        "veryfast",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        str(visual_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin was not opened")
    thumbnail: np.ndarray | None = None
    sampled_states: list[dict[str, Any]] = []
    try:
        with state_path.open("w", encoding="utf-8") as state_handle, trace_path.open("w", encoding="utf-8") as trace_handle:
            for index, state in enumerate(timeline):
                frame, metadata = render_flame_walk_frame(width, height, state)
                proc.stdin.write(frame.tobytes())
                state_handle.write(json.dumps({**state, "render_metadata": metadata}, allow_nan=False) + "\n")
                trace_handle.write(json.dumps(trace[index], allow_nan=False) + "\n")
                if index == min(len(timeline) - 1, fps * 15):
                    thumbnail = frame.copy()
                if index % fps == 0:
                    sampled_states.append({**metadata, "flame_lick_pressure": state["flame_lick_pressure"]})
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg visual encoder failed")
    except Exception:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
    if thumbnail is None:
        thumbnail, _ = render_flame_walk_frame(width, height, timeline[-1])
    cv2.imwrite(str(thumb_path), thumbnail)

    audio_muxed = False
    if mux_audio:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(visual_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(final_path),
            ],
            check=True,
        )
        audio_muxed = True

    manifest = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "claim": "locked_character_flame_walk_616_forecast",
        "boundary": {
            "generated_state_media": "synthetic_not_evidence",
            "no_external_visual_assets": True,
            "characters_locked_environment_moves": True,
        },
        "inputs": {
            "audio_path": str(audio_path),
            "audio_sha256": sha256_file(audio_path),
            "source_state_count": len(source_states),
            "source_seconds": source_seconds,
        },
        "forecast": {
            "method": "ten_authored_states_then_six_one_six_temporal_projection",
            "total_seconds": total_seconds,
            "fps": fps,
            "frames": len(timeline),
            "source_interpolated_frames": sum(1 for state in timeline if state["forecast_kind"] == "source_interpolated"),
            "projected_frames": sum(1 for state in timeline if state["forecast_kind"] == "six_one_six_projected"),
            "radius": 6,
            "characters": "locked_pose",
            "moving_channels": ["flame", "smoke", "rain", "reflections", "embers", "lightning"],
        },
        "source_states": source_states,
        "sampled_frame_states": sampled_states,
        "hardware": capture_hardware(),
        "machine_cost": _machine_cost(start_wall, start_cpu, memory_start),
        "outputs": {
            "run_dir": str(run_dir),
            "video_mp4": str(final_path),
            "visual_only_mp4": str(visual_path),
            "audio_muxed": audio_muxed,
            "frame_state_jsonl": str(state_path),
            "trace_jsonl": str(trace_path),
            "thumbnail_jpg": str(thumb_path),
            "manifest_json": str(manifest_path),
        },
    }
    manifest["outputs"]["video_sha256"] = sha256_file(final_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "video_mp4": str(final_path),
        "manifest_json": str(manifest_path),
        "thumbnail_jpg": str(thumb_path),
        "frames": len(timeline),
        "duration_seconds": total_seconds,
        "video_sha256": manifest["outputs"]["video_sha256"],
        "machine_cost": manifest["machine_cost"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render locked-character flame-walk 6-1-6 forecast video.")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--total-seconds", type=int, default=20)
    parser.add_argument("--source-seconds", type=int, default=10)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    result = render_flame_walk_forecast(
        audio_path=Path(args.audio),
        output_root=Path(args.output_root),
        run_id=args.run_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        total_seconds=args.total_seconds,
        source_seconds=args.source_seconds,
        mux_audio=not args.no_audio,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
