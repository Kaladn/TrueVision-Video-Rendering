#!/usr/bin/env python3
"""Render Edge Of The World v3 as edge, smoke, and river-below state media.

This lane keeps the successful audio river idea but adds a cinematic scene
grammar: the edge of the world, smoke rising, a top-down look over the edge,
and a pulsing river of color below. Optional signature profiles add abstract
motion/look pressure learned from TrueVision captures.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from truevision_edge_audio_river import (
    DEFAULT_AUDIO,
    DEFAULT_LYRICS,
    build_edge_theme,
    capture_hardware,
    decode_audio_mono,
    measure_audio_features,
    sha256_file,
)
from truevision_basement_stick_narrative import load_signature_profile


DEFAULT_OUTPUT_ROOT = Path("outputs/edge_of_the_world_v3")
DEFAULT_RUN_ID = "edge_of_the_world_v3_edge_smoke_river"
DEFAULT_SIGNATURE_PROFILE = Path("storage/artifacts/signature_profiles/cod_fullscreen_20m_signature_v2/signature_profile_bundle.json")


@dataclass(frozen=True)
class EdgeScene:
    scene_id: str
    start_norm: float
    end_norm: float
    description: str
    camera: str
    river_depth: str


SCENE_PHASES: tuple[EdgeScene, ...] = (
    EdgeScene("edge_horizon_smoke", 0.00, 0.18, "dark edge horizon with smoke breathing upward", "locked_wide", "far_below"),
    EdgeScene("approach_edge", 0.18, 0.36, "slow push toward the rim of the world", "slow_push", "glimpsed"),
    EdgeScene("looking_down_over_edge", 0.36, 0.50, "top-down look past the earth edge into the river below", "top_down_tilt", "below_visible"),
    EdgeScene("river_below_energy", 0.50, 0.72, "river of colors flows and pulses under black space", "drift_above", "primary"),
    EdgeScene("strands_unite", 0.72, 0.88, "separate currents braid into one living flow", "orbit_slow", "primary_unity"),
    EdgeScene("ascension_column", 0.88, 0.965, "river energy rises from below toward the edge", "vertical_rise", "rising"),
    EdgeScene("return_to_black_edge", 0.965, 1.00, "energy collapses back to a thin glowing edge line", "locked_wide", "vanishing"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def slug(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return clean.strip("_")[:96] or DEFAULT_RUN_ID


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def memory_snapshot() -> dict[str, Any]:
    """Return process memory counters without third-party packages."""
    snapshot: dict[str, Any] = {
        "working_set_bytes": None,
        "peak_working_set_bytes": None,
        "pagefile_usage_bytes": None,
        "private_usage_bytes": None,
    }
    if os.name != "nt":
        return snapshot

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    kernel32 = ctypes.WinDLL("kernel32.dll")
    psapi = ctypes.WinDLL("psapi.dll")
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), ctypes.c_ulong]
    get_process_memory_info.restype = ctypes.c_int
    ok = get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
    if not ok:
        return snapshot
    snapshot.update(
        {
            "working_set_bytes": int(counters.WorkingSetSize),
            "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
            "pagefile_usage_bytes": int(counters.PagefileUsage),
            "private_usage_bytes": int(counters.PrivateUsage),
        }
    )
    return snapshot


def build_edge_world_v3_schedule(duration_seconds: float) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": scene.scene_id,
            "start_seconds": round(duration_seconds * scene.start_norm, 6),
            "end_seconds": round(duration_seconds * scene.end_norm, 6),
            "duration_seconds": round(duration_seconds * (scene.end_norm - scene.start_norm), 6),
            "description": scene.description,
            "camera": scene.camera,
            "river_depth": scene.river_depth,
        }
        for scene in SCENE_PHASES
    ]


def scene_for_time(time_seconds: float, duration_seconds: float) -> EdgeScene:
    norm = 0.0 if duration_seconds <= 0 else min(0.999999, max(0.0, time_seconds / duration_seconds))
    for scene in SCENE_PHASES:
        if scene.start_norm <= norm < scene.end_norm:
            return scene
    return SCENE_PHASES[-1]


def _hsv_to_bgr(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    hsv = np.asarray([[[hue % 180.0, np.clip(saturation, 0.0, 1.0) * 255, np.clip(value, 0.0, 1.0) * 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _signature_sample(signature_profile: dict[str, Any] | None, time_seconds: float, duration_seconds: float) -> dict[str, float]:
    if not signature_profile:
        return {}
    samples = signature_profile.get("timeline_samples") or []
    if not samples:
        return {}
    norm = 0.0 if duration_seconds <= 0 else max(0.0, min(1.0, time_seconds / duration_seconds))
    nearest = min(samples, key=lambda sample: abs(float(sample.get("time_norm", 0.0)) - norm))
    return {str(key): float(value) for key, value in nearest.items() if isinstance(value, (int, float))}


def _draw_stars(frame: np.ndarray, time_seconds: float, high: float) -> None:
    height, width = frame.shape[:2]
    for index in range(130):
        x = int((index * 73 + math.sin(index * 1.7) * 41) % width)
        y = int((index * 29 + math.cos(index * 0.9) * 23) % max(1, int(height * 0.72)))
        twinkle = 0.15 + 0.55 * ((math.sin(time_seconds * (0.6 + high) + index * 2.31) + 1.0) * 0.5)
        value = int(38 * twinkle)
        if value > 5:
            frame[y, x] = np.maximum(frame[y, x], (value, value, value + 10))


def _draw_smoke(frame: np.ndarray, *, time_seconds: float, rms: float, high: float, edge_y: int, strength: float) -> None:
    height, width = frame.shape[:2]
    smoke = np.zeros_like(frame, dtype=np.uint8)
    plume_count = 34
    for index in range(plume_count):
        base_x = width * (0.08 + 0.84 * ((index * 37) % plume_count) / max(1, plume_count - 1))
        drift = math.sin(time_seconds * (0.22 + high * 0.22) + index * 0.83) * width * 0.035
        lift = ((time_seconds * (0.035 + rms * 0.055) + index * 0.061) % 1.0)
        x = int(base_x + drift + math.sin(index) * 12)
        y = int(edge_y - lift * height * (0.36 + 0.18 * strength))
        radius_x = int(width * (0.010 + 0.017 * strength) * (1.0 + 0.4 * math.sin(index)))
        radius_y = int(height * (0.030 + 0.050 * strength) * (1.0 + 0.3 * math.cos(index)))
        color = int(18 + 54 * strength * (1.0 - lift))
        cv2.ellipse(smoke, (x, y), (max(3, radius_x), max(5, radius_y)), index * 11, 0, 360, (color, color + 4, color + 8), -1, cv2.LINE_AA)
    blur = int(17 + 28 * strength)
    if blur % 2 == 0:
        blur += 1
    smoke = cv2.GaussianBlur(smoke, (blur, blur), 0)
    cv2.addWeighted(smoke, 0.58 + 0.25 * strength, frame, 1.0, 0, dst=frame)


def _river_points(
    *,
    width: int,
    height: int,
    time_seconds: float,
    strand: int,
    scene_id: str,
    rms: float,
    bass: float,
    mid: float,
    high: float,
) -> np.ndarray:
    count = 220
    u = np.linspace(0.0, 1.0, count, dtype=np.float32)
    x = u * width
    if scene_id in {"edge_horizon_smoke", "approach_edge"}:
        center = height * (0.72 + 0.035 * strand)
        amp = height * (0.018 + 0.03 * bass)
        y = center + np.sin(u * math.tau * (2.0 + strand * 0.12) + time_seconds * 1.4 + strand) * amp
    elif scene_id == "looking_down_over_edge":
        center = height * (0.60 + 0.030 * strand)
        amp = height * (0.10 + 0.08 * bass)
        x = width * (0.12 + 0.76 * u) + np.sin(u * math.tau * 3 + time_seconds + strand) * width * 0.025
        y = center + np.sin(u * math.tau * (2.7 + mid) + time_seconds * 2.4 + strand * 0.5) * amp
    elif scene_id == "ascension_column":
        y = height * (0.92 - 0.80 * u)
        x = width * (0.50 + (strand - 4) * 0.012) + np.sin(u * math.tau * (2.0 + high) + time_seconds * 2 + strand) * width * (0.04 + 0.04 * bass)
    else:
        center = height * (0.50 + (strand - 4) * 0.024)
        amp = height * (0.15 + 0.10 * bass + 0.03 * rms)
        braid = math.sin(time_seconds * 0.8 + strand) * width * 0.03
        x = width * (-0.06 + 1.12 * u) + braid
        y = center + np.sin(u * math.tau * (1.7 + strand * 0.04) + time_seconds * (1.2 + mid) + strand * 0.7) * amp
        y += np.sin(u * math.tau * (5.0 + high * 1.8) - time_seconds * 1.9) * amp * 0.24
        if scene_id == "strands_unite":
            unity = np.clip((u - 0.15) / 0.70, 0.0, 1.0)
            y = y * (1.0 - unity * 0.72) + (height * 0.50) * (unity * 0.72)
    return np.round(np.stack([x, y], axis=1)).astype(np.int32).reshape((-1, 1, 2))


def _draw_river_below(
    frame: np.ndarray,
    *,
    scene_id: str,
    time_seconds: float,
    rms: float,
    bass: float,
    mid: float,
    high: float,
    beat: float,
    signature: dict[str, float],
) -> None:
    height, width = frame.shape[:2]
    river = np.zeros_like(frame, dtype=np.uint8)
    motion = max(0.0, min(1.0, signature.get("motion", 0.0)))
    saturation_boost = max(0.0, min(1.0, signature.get("saturation", 0.0)))
    strand_count = 10
    for strand in range(strand_count):
        points = _river_points(
            width=width,
            height=height,
            time_seconds=time_seconds,
            strand=strand,
            scene_id=scene_id,
            rms=rms,
            bass=bass,
            mid=mid,
            high=high,
        )
        hue = (time_seconds * 12 + strand * 15 + bass * 58 + motion * 28) % 180
        color = _hsv_to_bgr(hue, 0.74 + 0.22 * saturation_boost, 0.28 + 0.42 * rms + 0.20 * beat)
        line_width = max(1, int(1 + bass * 7 + beat * 3 + motion * 4))
        cv2.polylines(river, [points], False, color, line_width + 6, cv2.LINE_AA)
        cv2.polylines(river, [points], False, color, line_width, cv2.LINE_AA)

    if beat > 0.16:
        for pulse in range(4):
            cx = int(width * (0.18 + pulse * 0.22 + 0.04 * math.sin(time_seconds + pulse)))
            cy = int(height * (0.58 + 0.10 * math.cos(time_seconds * 0.7 + pulse)))
            radius = int(height * (0.045 + beat * 0.14 + pulse * 0.01))
            color = _hsv_to_bgr(time_seconds * 19 + pulse * 34, 0.88, 0.18 + beat * 0.32)
            cv2.circle(river, (cx, cy), radius, color, max(1, int(1 + beat * 3)), cv2.LINE_AA)

    blur = int(13 + bass * 22 + motion * 11)
    if blur % 2 == 0:
        blur += 1
    glow = cv2.GaussianBlur(river, (blur, blur), 0)
    cv2.addWeighted(frame, 1.0, glow, 0.46 + 0.20 * bass, 0, dst=frame)
    cv2.addWeighted(frame, 1.0, river, 0.88, 0, dst=frame)


def _draw_edge_mask(frame: np.ndarray, *, scene_id: str, time_seconds: float, bass: float, beat: float) -> int:
    height, width = frame.shape[:2]
    if scene_id == "looking_down_over_edge":
        edge_y = int(height * 0.26)
        cv2.ellipse(frame, (width // 2, int(height * -0.12)), (int(width * 0.68), int(height * 0.45)), 0, 0, 180, (4, 4, 6), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (width // 2, int(height * -0.12)), (int(width * 0.68), int(height * 0.45)), 0, 0, 180, (28, 32, 38), 3, cv2.LINE_AA)
    else:
        edge_y = int(height * (0.58 - 0.08 * bass + 0.02 * math.sin(time_seconds * 0.4)))
        pts = np.array(
            [
                [0, edge_y],
                [int(width * 0.18), edge_y - int(height * 0.03)],
                [int(width * 0.45), edge_y + int(height * 0.02)],
                [int(width * 0.68), edge_y - int(height * 0.04)],
                [width, edge_y],
                [width, height],
                [0, height],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [pts], (3, 4, 5), cv2.LINE_AA)
        glow_color = (22 + int(beat * 45), 34 + int(beat * 55), 46 + int(beat * 80))
        cv2.polylines(frame, [pts[:5].reshape((-1, 1, 2))], False, glow_color, 2 + int(beat * 4), cv2.LINE_AA)
    return edge_y


def _apply_signature_style(frame: np.ndarray, signature: dict[str, float], profile_id: str | None) -> tuple[np.ndarray, dict[str, Any]]:
    if not signature:
        return frame, {"applied": False}
    motion = max(0.0, min(1.0, signature.get("motion", 0.0)))
    edge = max(0.0, min(1.0, signature.get("edge", 0.0)))
    contrast = max(0.0, min(1.0, signature.get("contrast", 0.0)))
    saturation = max(0.0, min(1.0, signature.get("saturation", 0.0)))
    flash = max(0.0, min(1.0, signature.get("flash", 0.0)))
    shake_x = max(-1.0, min(1.0, signature.get("shake_x", 0.0)))
    shake_y = max(-1.0, min(1.0, signature.get("shake_y", 0.0)))

    styled = frame
    shift_x = int(round(shake_x * (3 + 10 * motion)))
    shift_y = int(round(shake_y * (2 + 7 * motion)))
    if shift_x or shift_y:
        transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        styled = cv2.warpAffine(styled, transform, (styled.shape[1], styled.shape[0]), borderMode=cv2.BORDER_REFLECT)
    styled = cv2.convertScaleAbs(styled, alpha=1.0 + 0.16 * contrast + 0.08 * flash, beta=6 * flash - 2 * motion)

    if saturation > 0.02:
        hsv = cv2.cvtColor(styled, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.20 * saturation), 0, 255)
        styled = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if edge > 0.12:
        gray = cv2.cvtColor(styled, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55, 145)
        edge_layer = np.zeros_like(styled)
        edge_layer[:, :, 0] = np.clip(edges.astype(np.float32) * (0.20 + 0.70 * edge), 0, 255).astype(np.uint8)
        edge_layer[:, :, 2] = np.clip(edges.astype(np.float32) * (0.08 + 0.20 * flash), 0, 255).astype(np.uint8)
        styled = cv2.addWeighted(styled, 1.0, edge_layer, 0.32, 0)
    return styled, {
        "applied": True,
        "profile_id": profile_id,
        "motion": round(motion, 6),
        "edge": round(edge, 6),
        "contrast": round(contrast, 6),
        "saturation": round(saturation, 6),
        "flash": round(flash, 6),
        "shake_x": round(shake_x, 6),
        "shake_y": round(shake_y, 6),
    }


def render_edge_world_v3_frame(
    *,
    width: int,
    height: int,
    fps: int,
    frame_state: dict[str, float],
    duration_seconds: float,
    signature_profile: dict[str, Any] | None = None,
    program_stamp: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    time_seconds = float(frame_state.get("time_seconds", 0.0))
    scene = scene_for_time(time_seconds, duration_seconds)
    rms = float(frame_state.get("rms", 0.0))
    bass = float(frame_state.get("bass", 0.0))
    mid = float(frame_state.get("mid", 0.0))
    high = float(frame_state.get("high", 0.0))
    beat = float(frame_state.get("beat", 0.0))
    signature = _signature_sample(signature_profile, time_seconds, duration_seconds)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    sky_grad = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, np.newaxis]
    frame[:, :, 0] = np.clip(3 + sky_grad * 13 + high * 8, 0, 255).astype(np.uint8)
    frame[:, :, 1] = np.clip(2 + sky_grad * 8 + mid * 5, 0, 255).astype(np.uint8)
    frame[:, :, 2] = np.clip(5 + sky_grad * 18 + beat * 18, 0, 255).astype(np.uint8)
    _draw_stars(frame, time_seconds, high)

    _draw_river_below(
        frame,
        scene_id=scene.scene_id,
        time_seconds=time_seconds,
        rms=rms,
        bass=bass,
        mid=mid,
        high=high,
        beat=beat,
        signature=signature,
    )
    edge_y = _draw_edge_mask(frame, scene_id=scene.scene_id, time_seconds=time_seconds, bass=bass, beat=beat)
    smoke_strength = max(0.18, min(1.0, 0.35 + rms * 0.45 + signature.get("motion", 0.0) * 0.25))
    _draw_smoke(frame, time_seconds=time_seconds, rms=rms, high=high, edge_y=edge_y, strength=smoke_strength)

    if scene.scene_id == "ascension_column":
        center_x = width // 2
        for radius in range(14, int(width * 0.24), 28):
            color = _hsv_to_bgr(time_seconds * 15 + radius * 0.2, 0.85, 0.22 + beat * 0.24)
            cv2.ellipse(frame, (center_x, int(height * 0.52)), (radius, int(radius * 1.45)), 0, 0, 360, color, 1 + int(beat * 2), cv2.LINE_AA)

    frame, signature_style = _apply_signature_style(
        frame,
        signature,
        signature_profile.get("profile_id") if signature_profile else None,
    )
    vignette_x = np.linspace(0.58, 1.0, width, dtype=np.float32)
    vignette_x = np.minimum(vignette_x, vignette_x[::-1])
    vignette_y = np.linspace(0.68, 1.0, height, dtype=np.float32)
    vignette = vignette_y[:, np.newaxis] * vignette_x[np.newaxis, :]
    frame = np.clip(frame.astype(np.float32) * vignette[:, :, np.newaxis], 0, 255).astype(np.uint8)

    if program_stamp:
        cv2.putText(
            frame,
            program_stamp,
            (24, max(24, height - 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.38, height / 1600.0),
            (70, 76, 82),
            1,
            cv2.LINE_AA,
        )

    layers = ["black_field", "edge_silhouette", "smoke_steam", "river_below", "audio_bloom", "signature_motion"]
    return frame, {
        "frame_index": int(frame_state.get("frame_index", 0)),
        "time_seconds": round(time_seconds, 6),
        "scene_id": scene.scene_id,
        "scene_description": scene.description,
        "camera": scene.camera,
        "layers": layers,
        "edge_y": int(edge_y),
        "audio_features": {
            "rms": round(rms, 6),
            "bass": round(bass, 6),
            "mid": round(mid, 6),
            "high": round(high, 6),
            "beat": round(beat, 6),
        },
        "signature_style": signature_style,
        "visual_rules": {
            "no_lyric_overlay": True,
            "no_dialogue_cards": True,
            "program_stamp": program_stamp is not None,
            "generated_state_media": "synthetic_not_evidence",
        },
    }


def _machine_cost(start_wall: float, start_cpu: float, memory_start: dict[str, Any]) -> dict[str, Any]:
    elapsed = max(0.000001, time.perf_counter() - start_wall)
    cpu_seconds = max(0.0, time.process_time() - start_cpu)
    logical = os.cpu_count() or 1
    return {
        "wall_seconds": round(elapsed, 6),
        "process_cpu_seconds": round(cpu_seconds, 6),
        "avg_cpu_core_equivalent": round(cpu_seconds / elapsed, 6),
        "avg_process_logical_cpu_percent": round((cpu_seconds / (elapsed * logical)) * 100.0, 6),
        "logical_cpu_count": logical,
        "memory_start": memory_start,
        "memory_end": memory_snapshot(),
    }


def _format_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    mib = float(value) / (1024.0 * 1024.0)
    return f"{int(value)} bytes ({mib:.2f} MiB)"


def capture_gpu_entries() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Depth 3",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        raw = json.loads(completed.stdout or "[]")
    except Exception:
        return []
    entries = raw if isinstance(raw, list) else [raw]
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized.append(
            {
                "name": entry.get("Name"),
                "adapter_ram_bytes": entry.get("AdapterRAM"),
                "driver_version": entry.get("DriverVersion"),
            }
        )
    return normalized


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    machine = manifest["machine_cost"]
    memory_start = machine.get("memory_start", {})
    memory_end = machine.get("memory_end", {})
    timing = manifest.get("component_timing_seconds", {})
    gpu_entries = manifest.get("hardware", {}).get("gpu", [])
    lines = [
        f"# {manifest['run_id']} Report",
        "",
        "## Claim",
        "",
        "A cinematic Edge Of The World state-media video was generated from audio features, lyric theme, and optional motion/look signature data.",
        "",
        "## Boundary",
        "",
        "Generated state media is synthetic, not evidence. No lyric text or dialogue cards are rendered.",
        "",
        "## Outputs",
        "",
        f"- Video: `{manifest['outputs']['video_mp4']}`",
        f"- Manifest: `{manifest['outputs']['manifest_json']}`",
        f"- Frame state: `{manifest['outputs']['frame_state_jsonl']}`",
        f"- Thumbnail: `{manifest['outputs']['thumbnail_jpg']}`",
        "",
        "## Machine Cost",
        "",
        f"- Started: `{manifest['started_at_utc']}`",
        f"- Completed: `{manifest['completed_at_utc']}`",
        f"- Wall seconds: `{machine['wall_seconds']}`",
        f"- Process CPU seconds: `{machine['process_cpu_seconds']}`",
        f"- Avg CPU core equivalent: `{machine['avg_cpu_core_equivalent']}`",
        f"- Avg logical CPU percent: `{machine['avg_process_logical_cpu_percent']}`",
        f"- Logical CPU count: `{machine['logical_cpu_count']}`",
        f"- Working set start: `{_format_bytes(memory_start.get('working_set_bytes'))}`",
        f"- Working set end: `{_format_bytes(memory_end.get('working_set_bytes'))}`",
        f"- Peak working set: `{_format_bytes(memory_end.get('peak_working_set_bytes'))}`",
        f"- Private/pagefile start: `{_format_bytes(memory_start.get('private_usage_bytes'))}`",
        f"- Private/pagefile end: `{_format_bytes(memory_end.get('private_usage_bytes'))}`",
        "",
        "## Component Timing",
        "",
        *[f"- {key}: `{value}` seconds" for key, value in timing.items()],
        "",
        "## System Components",
        "",
        f"- CPU logical count: `{manifest['hardware'].get('cpu_logical_count')}`",
        f"- Processor: `{manifest['hardware'].get('processor')}`",
        f"- RAM total: `{_format_bytes(manifest['hardware'].get('ram', {}).get('total_physical_bytes'))}`",
        f"- RAM available at capture: `{_format_bytes(manifest['hardware'].get('ram', {}).get('available_physical_bytes'))}`",
        "- GPU acceleration used: `false`",
        "- Render path: `Python + OpenCV/Numpy frame synthesis -> ffmpeg libx264 CPU encode -> ffmpeg AAC mux`",
    ]
    if gpu_entries:
        lines.extend(["", "## Detected GPU Entries", ""])
        for gpu in gpu_entries:
            lines.append(
                f"- `{gpu.get('name')}` | RAM `{_format_bytes(gpu.get('adapter_ram_bytes'))}` | Driver `{gpu.get('driver_version')}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_edge_world_v3(
    *,
    audio_path: Path,
    lyrics_path: Path | None,
    output_root: Path,
    run_id: str = DEFAULT_RUN_ID,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    sample_rate: int = 44100,
    max_seconds: float | None = None,
    mux_audio: bool = True,
    signature_profile_path: Path | None = None,
    program_stamp: str | None = "TrueVision Generation Lab / edge_world_v3",
) -> dict[str, Any]:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    memory_start = memory_snapshot()
    component_timing: dict[str, float] = {}

    def mark_component(name: str, phase_start: float) -> None:
        component_timing[name] = round(time.perf_counter() - phase_start, 6)

    started_at = utc_now()
    phase_start = time.perf_counter()
    audio_path = audio_path.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    lyrics_path = lyrics_path.resolve() if lyrics_path is not None else None
    signature_profile = load_signature_profile(signature_profile_path) if signature_profile_path else None
    run_id = slug(run_id)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    mark_component("setup_resolve_paths_signature_seconds", phase_start)

    visual_path = run_dir / f"{run_id}_visual_only.mp4"
    final_path = run_dir / f"{run_id}_full_audio.mp4" if mux_audio else visual_path
    state_path = run_dir / f"{run_id}_frame_state.jsonl"
    thumb_path = run_dir / f"{run_id}_thumbnail.jpg"
    manifest_path = run_dir / f"{run_id}_manifest.json"
    report_path = run_dir / f"{run_id}_report.md"

    phase_start = time.perf_counter()
    ffmpeg_cmd = [
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
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(visual_path),
    ]
    mark_component("video_encoder_command_build_seconds", phase_start)

    phase_start = time.perf_counter()
    samples = decode_audio_mono(audio_path, sample_rate=sample_rate, max_seconds=max_seconds)
    features = measure_audio_features(samples, sample_rate=sample_rate, fps=fps)
    if max_seconds is not None:
        features = [feature for feature in features if feature["time_seconds"] < max_seconds]
    if not features:
        raise ValueError("Audio produced no renderable features")
    duration_seconds = len(features) / fps
    mark_component("audio_decode_feature_extract_seconds", phase_start)

    phase_start = time.perf_counter()
    theme = build_edge_theme(lyrics_path)
    mark_component("lyric_theme_compile_seconds", phase_start)

    phase_start = time.perf_counter()
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin was not opened")

    sampled_states: list[dict[str, Any]] = []
    thumbnail_frame: np.ndarray | None = None
    try:
        with state_path.open("w", encoding="utf-8") as state_handle:
            for index, feature in enumerate(features):
                frame_state = dict(feature)
                frame_state["frame_index"] = index
                frame, metadata = render_edge_world_v3_frame(
                    width=width,
                    height=height,
                    fps=fps,
                    frame_state=frame_state,
                    duration_seconds=duration_seconds,
                    signature_profile=signature_profile,
                    program_stamp=program_stamp,
                )
                proc.stdin.write(frame.tobytes())
                state_handle.write(json.dumps(metadata, allow_nan=False) + "\n")
                if index % max(1, fps) == 0:
                    sampled_states.append(metadata)
                if index == min(len(features) - 1, max(1, fps * 20)):
                    thumbnail_frame = frame.copy()
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg video encoder failed")
        mark_component("frame_synthesis_and_video_encode_seconds", phase_start)
    except Exception:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise

    phase_start = time.perf_counter()
    if thumbnail_frame is None:
        thumbnail_frame = render_edge_world_v3_frame(
            width=width,
            height=height,
            fps=fps,
            frame_state={**features[0], "frame_index": 0},
            duration_seconds=duration_seconds,
            signature_profile=signature_profile,
            program_stamp=program_stamp,
        )[0]
    cv2.imwrite(str(thumb_path), thumbnail_frame)
    mark_component("thumbnail_write_seconds", phase_start)

    audio_muxed = False
    if mux_audio:
        phase_start = time.perf_counter()
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
        mark_component("audio_mux_seconds", phase_start)

    phase_start = time.perf_counter()
    feature_arrays = {key: np.asarray([feature[key] for feature in features], dtype=np.float32) for key in ["rms", "bass", "mid", "high", "beat"]}
    machine_cost = _machine_cost(start_wall, start_cpu, memory_start)
    hardware = capture_hardware()
    hardware["gpu"] = capture_gpu_entries()
    manifest = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "claim": "edge_world_v3_edge_smoke_river_below",
        "boundary": {
            "generated_state_media": "synthetic_not_evidence",
            "no_lyric_overlay": True,
            "no_dialogue_cards": True,
            "signature_usage": "abstract_motion_look_only_not_source_reconstruction",
            "program_stamp": program_stamp,
        },
        "inputs": {
            "audio_path": str(audio_path),
            "audio_sha256": sha256_file(audio_path),
            "sample_rate": sample_rate,
        },
        "theme": theme,
        "render": {
            "width": width,
            "height": height,
            "fps": fps,
            "frames": len(features),
            "duration_seconds": round(duration_seconds, 6),
            "style": "edge_world_v3_edge_smoke_river_below",
            "scene_schedule": build_edge_world_v3_schedule(duration_seconds),
            "signature_profile": {
                "enabled": bool(signature_profile),
                "profile_id": signature_profile.get("profile_id") if signature_profile else None,
                "path": signature_profile.get("source_path") if signature_profile else None,
            },
        },
        "audio_feature_summary": {
            key: {
                "mean": round(float(np.mean(values)), 6),
                "max": round(float(np.max(values)), 6),
                "std": round(float(np.std(values)), 6),
            }
            for key, values in feature_arrays.items()
        },
        "sampled_frame_states": sampled_states[:360],
        "hardware": hardware,
        "machine_cost": machine_cost,
        "component_timing_seconds": component_timing,
        "outputs": {
            "run_dir": str(run_dir),
            "video_mp4": str(final_path),
            "visual_only_mp4": str(visual_path),
            "audio_muxed": audio_muxed,
            "frame_state_jsonl": str(state_path),
            "thumbnail_jpg": str(thumb_path),
            "manifest_json": str(manifest_path),
            "report_md": str(report_path),
        },
    }
    _write_json(manifest_path, manifest)
    _write_report(report_path, manifest)
    manifest["outputs"]["video_sha256"] = sha256_file(final_path)
    manifest["outputs"]["manifest_sha256"] = sha256_file(manifest_path)
    mark_component("manifest_report_hash_seconds", phase_start)
    manifest["component_timing_seconds"] = component_timing
    _write_json(manifest_path, manifest)
    _write_report(report_path, manifest)
    return {
        "run_id": run_id,
        "video_mp4": str(final_path),
        "visual_only_mp4": str(visual_path),
        "audio_muxed": audio_muxed,
        "manifest_json": str(manifest_path),
        "frame_state_jsonl": str(state_path),
        "thumbnail_jpg": str(thumb_path),
        "report_md": str(report_path),
        "frames": len(features),
        "duration_seconds": round(duration_seconds, 6),
        "machine_cost": machine_cost,
        "video_sha256": manifest["outputs"]["video_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Edge Of The World v3 with smoke, edge horizon, and river below.")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO))
    parser.add_argument("--lyrics", default=str(DEFAULT_LYRICS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--signature-profile", default=str(DEFAULT_SIGNATURE_PROFILE))
    parser.add_argument("--program-stamp", default="TrueVision Generation Lab / edge_world_v3")
    parser.add_argument("--no-program-stamp", action="store_true")
    parser.add_argument("--visual-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_edge_world_v3(
        audio_path=Path(args.audio),
        lyrics_path=Path(args.lyrics) if args.lyrics else None,
        output_root=Path(args.output_root),
        run_id=args.run_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
        mux_audio=not args.visual_only,
        signature_profile_path=Path(args.signature_profile) if args.signature_profile else None,
        program_stamp=None if args.no_program_stamp else args.program_stamp,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
