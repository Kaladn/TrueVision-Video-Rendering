#!/usr/bin/env python3
"""Render a TrueVision still-state snap sequence.

The source authority is the TrueVision exact-photo snap bundle. Pixel maps are
used only as the final display surface needed to write a viewable video.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from truevision_edge_audio_river import decode_audio_mono, measure_audio_features, probe_audio_duration
from truevision_resonance_recorder import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAP_ROOT = PROJECT_ROOT / "outputs" / "photo_state_snaps"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "state_snap_sequence_renders"


@dataclass(frozen=True)
class SnapSource:
    run_id: str
    run_dir: Path
    cell_state_npz: Path
    pixel_state_npz: Path
    manifest_json: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def slug(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return clean.strip("_")[:96] or "truevision_state_snap_sequence"


def smootherstep(value: float) -> float:
    x = max(0.0, min(1.0, float(value)))
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def build_ordered_default_snaps(root: Path = DEFAULT_SNAP_ROOT) -> list[SnapSource]:
    """Return top-left, lower-left, lower-right, top-right snap order."""
    order = [
        "screenshot_20260520_224749_exact_snap",
        "screenshot_20260520_224848_exact_snap",
        "screenshot_20260520_224910_exact_snap",
        "screenshot_20260520_224829_exact_snap",
    ]
    snaps: list[SnapSource] = []
    for run_id in order:
        run_dir = root / run_id
        snaps.append(
            SnapSource(
                run_id=run_id,
                run_dir=run_dir,
                cell_state_npz=run_dir / "cell_state_npz" / f"{run_id}_cells_0000.npz",
                pixel_state_npz=run_dir / "state" / f"{run_id}_pixel_state_rgb_u8.npz",
                manifest_json=run_dir / f"{run_id}_manifest.json",
            )
        )
    return snaps


def load_cell_state_frame(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        cells = np.asarray(data["cell_state"], dtype=np.float32)
    if cells.ndim == 4:
        cells = cells[0]
    if cells.ndim != 3 or cells.shape[2] < 16:
        raise ValueError(f"invalid cell state shape in {path}: {cells.shape}")
    return cells


def load_pixel_render_surface(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        rgb = np.asarray(data["rgb"], dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"invalid pixel render surface shape in {path}: {rgb.shape}")
    return rgb


def reconstruct_frame_from_cell_state(cells: np.ndarray, *, output_size: tuple[int, int]) -> np.ndarray:
    """Build a renderer frame from TrueVision cell-state RGB/luma fields."""
    width, height = output_size
    rgb_mean = np.clip(cells[:, :, 0:3], 0.0, 255.0).astype(np.uint8)
    luma = np.clip(cells[:, :, 9], 0.0, 255.0).astype(np.uint8)
    edge = np.clip(cells[:, :, 13] * 255.0, 0.0, 255.0).astype(np.uint8)

    rgb = cv2.resize(rgb_mean, (width, height), interpolation=cv2.INTER_CUBIC)
    luma_map = cv2.resize(luma, (width, height), interpolation=cv2.INTER_CUBIC)
    edge_map = cv2.resize(edge, (width, height), interpolation=cv2.INTER_CUBIC)

    rgb = cv2.addWeighted(rgb, 0.92, cv2.cvtColor(luma_map, cv2.COLOR_GRAY2RGB), 0.08, 0.0)
    edge_bgr = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(rgb, 1.0, edge_bgr, 0.035, 0.0).astype(np.uint8)


def fit_state_frame_to_canvas(frame: np.ndarray, *, canvas_size: tuple[int, int], mode: str = "cover") -> np.ndarray:
    """Fit a render frame to a canvas without aspect distortion."""
    width, height = canvas_size
    source_h, source_w = frame.shape[:2]
    if source_w < 1 or source_h < 1:
        raise ValueError("source frame is empty")
    if mode not in {"cover", "contain"}:
        raise ValueError("mode must be cover or contain")

    scale = max(width / source_w, height / source_h) if mode == "cover" else min(width / source_w, height / source_h)
    scaled_w = max(1, int(round(source_w * scale)))
    scaled_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if mode == "cover":
        x0 = max(0, (scaled_w - width) // 2)
        y0 = max(0, (scaled_h - height) // 2)
        return resized[y0 : y0 + height, x0 : x0 + width].copy()

    x = (width - scaled_w) // 2
    y = (height - scaled_h) // 2
    canvas[y : y + scaled_h, x : x + scaled_w] = resized
    return canvas


def memory_snapshot() -> dict[str, Any]:
    snapshot = {
        "working_set_bytes": None,
        "peak_working_set_bytes": None,
        "pagefile_usage_bytes": None,
        "private_usage_bytes": None,
        "system_available_physical_bytes": None,
        "system_total_physical_bytes": None,
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

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
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
        if ok:
            snapshot.update(
                {
                    "working_set_bytes": int(counters.WorkingSetSize),
                    "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                    "pagefile_usage_bytes": int(counters.PagefileUsage),
                    "private_usage_bytes": int(counters.PrivateUsage),
                }
            )
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            snapshot.update(
                {
                    "system_available_physical_bytes": int(status.ullAvailPhys),
                    "system_total_physical_bytes": int(status.ullTotalPhys),
                }
            )
    except Exception:
        return snapshot
    return snapshot


def capture_hardware() -> dict[str, Any]:
    gpus: list[dict[str, Any]] = []
    if platform.system().lower() == "windows":
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
            entries = raw if isinstance(raw, list) else [raw]
            gpus = [
                {
                    "name": entry.get("Name"),
                    "adapter_ram_bytes": entry.get("AdapterRAM"),
                    "driver_version": entry.get("DriverVersion"),
                }
                for entry in entries
                if isinstance(entry, dict)
            ]
        except Exception:
            gpus = []
    return {
        "os": platform.platform(),
        "processor": platform.processor(),
        "cpu_logical_count": os.cpu_count(),
        "python": platform.python_version(),
        "gpu": gpus,
    }


def sample_gpu_utilization_percent() -> float | None:
    if platform.system().lower() != "windows":
        return None
    command = (
        "$samples = (Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples; "
        "if ($samples) { ($samples | Measure-Object -Property CookedValue -Sum).Sum }"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=4,
        )
        text = completed.stdout.strip()
        if not text:
            return None
        return round(float(text), 6)
    except Exception:
        return None


class MachineSampler:
    def __init__(self, interval_seconds: float = 2.0) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds + 2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            memory = memory_snapshot()
            self.samples.append(
                {
                    "time_utc": utc_now(),
                    "memory": memory,
                    "gpu_engine_utilization_percent_sum": sample_gpu_utilization_percent(),
                }
            )
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, Any]:
        gpu_values = [
            float(sample["gpu_engine_utilization_percent_sum"])
            for sample in self.samples
            if sample.get("gpu_engine_utilization_percent_sum") is not None
        ]
        working_sets = [
            int(sample["memory"]["working_set_bytes"])
            for sample in self.samples
            if sample.get("memory", {}).get("working_set_bytes") is not None
        ]
        available_ram = [
            int(sample["memory"]["system_available_physical_bytes"])
            for sample in self.samples
            if sample.get("memory", {}).get("system_available_physical_bytes") is not None
        ]
        return {
            "sample_count": len(self.samples),
            "gpu_engine_utilization_percent_sum": _stats(gpu_values),
            "process_working_set_bytes": _stats(working_sets),
            "system_available_physical_bytes": _stats(available_ram),
            "samples": self.samples,
        }


def _stats(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    floats = [float(value) for value in values]
    return {
        "count": len(floats),
        "min": min(floats),
        "max": max(floats),
        "mean": sum(floats) / len(floats),
    }


def build_render_surfaces(
    snaps: list[SnapSource],
    *,
    canvas_size: tuple[int, int],
    fit_mode: str,
    render_surface: str,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    surfaces: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for snap in snaps:
        if not snap.cell_state_npz.exists():
            raise FileNotFoundError(snap.cell_state_npz)
        if not snap.manifest_json.exists():
            raise FileNotFoundError(snap.manifest_json)

        cells = load_cell_state_frame(snap.cell_state_npz)
        cell_render = reconstruct_frame_from_cell_state(cells, output_size=canvas_size)
        surface_kind = "truevision_cell_state_rgb_luma_edge"
        source_for_display = str(snap.cell_state_npz)
        if render_surface == "pixel_map":
            if not snap.pixel_state_npz.exists():
                raise FileNotFoundError(snap.pixel_state_npz)
            pixel_surface = load_pixel_render_surface(snap.pixel_state_npz)
            surface = fit_state_frame_to_canvas(pixel_surface, canvas_size=canvas_size, mode=fit_mode)
            surface_kind = "pixel_map_from_truevision_snap_for_render_surface_only"
            source_for_display = str(snap.pixel_state_npz)
        else:
            surface = fit_state_frame_to_canvas(cell_render, canvas_size=canvas_size, mode=fit_mode)

        surfaces.append(cv2.cvtColor(surface, cv2.COLOR_RGB2BGR))
        metadata.append(
            {
                "run_id": snap.run_id,
                "manifest_json": str(snap.manifest_json),
                "cell_state_npz": str(snap.cell_state_npz),
                "pixel_state_npz": str(snap.pixel_state_npz),
                "render_surface_kind": surface_kind,
                "render_surface_path": source_for_display,
                "cell_state_shape": list(cells.shape),
            }
        )
    return surfaces, metadata


def _make_noise_map(width: int, height: int, *, frame_index: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + frame_index * 17)
    low = rng.integers(0, 256, size=(max(4, height // 18), max(4, width // 18)), dtype=np.uint8)
    noise = cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(noise, (0, 0), sigmaX=12.0, sigmaY=12.0)


def derive_existing_motion_masks(frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Find existing picture regions allowed to move.

    Masks identify already-present warm glow/fire, low-contrast haze, and lower
    reflective highlights. They do not create new visual elements.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    b = frame_bgr[:, :, 0].astype(np.int16)
    g = frame_bgr[:, :, 1].astype(np.int16)
    r = frame_bgr[:, :, 2].astype(np.int16)
    height, width = gray.shape
    y = np.arange(height, dtype=np.float32)[:, None]
    lower_weight = y >= height * 0.45

    warm = (((hue < 30) | (hue > 168)) & (sat > 38) & (val > 58)) | ((r > g + 24) & (r > b + 34) & (r > 72))
    warm = warm.astype(np.uint8) * 255
    warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    warm = cv2.dilate(warm, np.ones((7, 7), np.uint8), iterations=1)
    fire = cv2.GaussianBlur(warm, (0, 0), sigmaX=4.0, sigmaY=4.0)

    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=9.0, sigmaY=9.0)
    local_detail = cv2.absdiff(gray, local_mean)
    haze = ((sat < 86) & (val > 34) & (val < 222) & (local_detail < 18)).astype(np.uint8) * 255
    haze = cv2.GaussianBlur(haze, (0, 0), sigmaX=8.0, sigmaY=8.0)

    reflection = (((val > 54) & (sat > 25) & lower_weight) | ((warm > 0) & lower_weight)).astype(np.uint8) * 255
    reflection = cv2.GaussianBlur(reflection, (0, 0), sigmaX=5.5, sigmaY=2.5)

    return {
        "fire": fire.astype(np.uint8),
        "haze": haze.astype(np.uint8),
        "reflection": reflection.astype(np.uint8),
    }


def build_state_keys(*, key_count: int, duration_seconds: float) -> list[dict[str, float]]:
    """Create deterministic joint-offset state keys across the render."""
    if key_count < 2:
        raise ValueError("key_count must be at least 2")
    keys: list[dict[str, float]] = []
    for index in range(key_count):
        t = index / (key_count - 1)
        time_seconds = t * duration_seconds
        keys.append(
            {
                "key_index": float(index),
                "time_seconds": round(time_seconds, 6),
                "fire_dx": 2.8 * math.sin(math.tau * (t * 11.0 + 0.13)) + 1.1 * math.sin(math.tau * (t * 31.0 + 0.61)),
                "fire_dy": -3.6 - 2.5 * math.sin(math.tau * (t * 17.0 + 0.27)),
                "fire_twist": 3.2 * math.sin(math.tau * (t * 9.0 + 0.43)),
                "fire_pulse": 0.50 + 0.50 * math.sin(math.tau * (t * 23.0 + 0.19)),
                "haze_dx": 1.8 * math.sin(math.tau * (t * 3.0 + 0.33)),
                "haze_dy": -1.4 * math.sin(math.tau * (t * 2.0 + 0.55)),
                "reflection_dx": 3.4 * math.sin(math.tau * (t * 7.0 + 0.08)),
                "reflection_wave": 1.2 + 2.3 * (0.5 + 0.5 * math.sin(math.tau * (t * 13.0 + 0.72))),
            }
        )
    return keys


def interpolate_state_key(keys: list[dict[str, float]], *, frame_index: int, frame_count: int) -> dict[str, float]:
    if not keys:
        raise ValueError("keys must not be empty")
    if len(keys) == 1 or frame_count <= 1:
        return dict(keys[0])
    position = (frame_index / (frame_count - 1)) * (len(keys) - 1)
    left = int(math.floor(position))
    right = min(len(keys) - 1, left + 1)
    alpha = smootherstep(position - left)
    result: dict[str, float] = {}
    for key in keys[left]:
        if key in keys[right] and isinstance(keys[left][key], (int, float)) and isinstance(keys[right][key], (int, float)):
            result[key] = float(keys[left][key]) * (1.0 - alpha) + float(keys[right][key]) * alpha
    result["left_key_index"] = float(left)
    result["right_key_index"] = float(right)
    result["key_alpha"] = float(alpha)
    return result


def _alpha_blend_mask(base: np.ndarray, moved: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * max(0.0, min(1.0, strength))
    return (base.astype(np.float32) * (1.0 - alpha) + moved.astype(np.float32) * alpha).astype(np.uint8)


def apply_existing_picture_state(
    frame: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    time_seconds: float,
    frame_index: int,
    state_key: dict[str, float],
    audio_state: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Animate only content already present in the picture."""
    height, width = frame.shape[:2]
    beat = float(audio_state.get("beat", 0.0)) if audio_state else 0.0
    bass = float(audio_state.get("bass", 0.0)) if audio_state else 0.0
    high = float(audio_state.get("high", 0.0)) if audio_state else 0.0
    rms = float(audio_state.get("rms", 0.0)) if audio_state else 0.0
    audio_pressure = max(0.0, min(1.0, 0.38 * beat + 0.35 * bass + 0.17 * high + 0.10 * rms))

    y_indices = np.arange(height, dtype=np.float32)[:, None]
    x_indices = np.arange(width, dtype=np.float32)[None, :]

    fire_phase = time_seconds * 3.1 + float(state_key["fire_twist"]) * 0.07
    fire_dx = float(state_key["fire_dx"]) + np.sin(y_indices * 0.028 + fire_phase) * (1.1 + 2.2 * audio_pressure)
    fire_dy = float(state_key["fire_dy"]) + np.sin(x_indices * 0.021 + fire_phase) * (0.7 + 1.3 * audio_pressure)
    fire_map_x = np.repeat(x_indices, height, axis=0) + fire_dx
    fire_map_y = np.repeat(y_indices, width, axis=1) + fire_dy
    fire_moved = cv2.remap(frame, fire_map_x.astype(np.float32), fire_map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    work = _alpha_blend_mask(frame, fire_moved, masks["fire"], 0.55 + 0.30 * audio_pressure)

    fire_alpha = (masks["fire"].astype(np.float32) / 255.0)[:, :, None]
    fire_gain = 1.0 + 0.045 * float(state_key["fire_pulse"]) + 0.075 * audio_pressure
    fire_region = np.clip(work.astype(np.float32) * fire_gain, 0, 255)
    work = (work.astype(np.float32) * (1.0 - fire_alpha * 0.55) + fire_region * fire_alpha * 0.55).astype(np.uint8)

    haze_dx = float(state_key["haze_dx"]) + math.sin(time_seconds * 0.9) * 0.8
    haze_dy = float(state_key["haze_dy"])
    haze_map_x = np.repeat(x_indices, height, axis=0) + haze_dx
    haze_map_y = np.repeat(y_indices, width, axis=1) + haze_dy
    haze_blur = cv2.GaussianBlur(work, (0, 0), sigmaX=1.4 + 1.8 * (0.3 + rms), sigmaY=1.2 + 1.5 * (0.3 + rms))
    haze_moved = cv2.remap(haze_blur, haze_map_x.astype(np.float32), haze_map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    work = _alpha_blend_mask(work, haze_moved, masks["haze"], 0.18 + 0.18 * rms)

    reflection_wave = float(state_key["reflection_wave"])
    reflection_dx = float(state_key["reflection_dx"]) + np.sin(y_indices * 0.055 + time_seconds * 2.4) * reflection_wave
    reflection_map_x = np.repeat(x_indices, height, axis=0) + reflection_dx
    reflection_map_y = np.repeat(y_indices, width, axis=1)
    reflection_moved = cv2.remap(work, reflection_map_x.astype(np.float32), reflection_map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    work = _alpha_blend_mask(work, reflection_moved, masks["reflection"], 0.28 + 0.20 * bass)

    metadata = {
        "time_seconds": round(time_seconds, 6),
        "frame_index": frame_index,
        "animation_mode": "existing_picture_state_only",
        "state_key_left": round(float(state_key.get("left_key_index", 0.0)), 6),
        "state_key_right": round(float(state_key.get("right_key_index", 0.0)), 6),
        "state_key_alpha": round(float(state_key.get("key_alpha", 0.0)), 6),
        "fire_dx": round(float(state_key["fire_dx"]), 6),
        "fire_dy": round(float(state_key["fire_dy"]), 6),
        "haze_dx": round(float(state_key["haze_dx"]), 6),
        "reflection_dx": round(float(state_key["reflection_dx"]), 6),
        "audio_pressure": round(audio_pressure, 6),
        "beat": round(beat, 6),
        "bass": round(bass, 6),
        "high": round(high, 6),
        "rms": round(rms, 6),
        "state_layers": ["existing_fire_offset", "existing_haze_drift", "existing_reflection_shimmer"],
    }
    return work, metadata


def apply_environment_state(
    frame: np.ndarray,
    *,
    time_seconds: float,
    frame_index: int,
    phase: float,
    audio_state: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = frame.shape[:2]
    if audio_state:
        beat = float(audio_state.get("beat", 0.0))
        rms = float(audio_state.get("rms", 0.0))
        bass = float(audio_state.get("bass", 0.0))
        mid = float(audio_state.get("mid", 0.0))
        high = float(audio_state.get("high", 0.0))
        flame_pressure = min(1.0, 0.22 + 0.48 * bass + 0.30 * beat + 0.18 * rms)
        ember_pressure = min(1.0, 0.20 + 0.52 * high + 0.33 * beat)
        haze_density = min(0.42, 0.12 + 0.18 * rms + 0.08 * mid)
    else:
        beat = 0.5 + 0.5 * math.sin(time_seconds * math.tau * 0.72)
        quick = 0.5 + 0.5 * math.sin(time_seconds * math.tau * 1.9 + 0.8)
        rms = beat
        bass = beat
        mid = 0.5 + 0.5 * math.sin(time_seconds * math.tau * 0.41)
        high = quick
        flame_pressure = 0.38 + 0.42 * beat + 0.20 * quick
        ember_pressure = 0.30 + 0.55 * beat
        haze_density = 0.16 + 0.11 * math.sin(time_seconds * math.tau * 0.18 + 1.7)

    work = frame.astype(np.float32)

    # Slow luma breathing and warm pressure, expressed as render state.
    warm = np.zeros_like(work)
    warm[:, :, 2] = 24.0 * flame_pressure
    warm[:, :, 1] = 8.0 * flame_pressure
    work = np.clip(work + warm, 0, 255)

    # Haze is a broad field, not circles or triangles.
    noise = _make_noise_map(width, height, frame_index=frame_index, seed=911)
    haze_alpha = (noise.astype(np.float32) / 255.0) * haze_density
    haze_color = np.dstack(
        [
            np.full((height, width), 26, dtype=np.float32),
            np.full((height, width), 36, dtype=np.float32),
            np.full((height, width), 54 + 22 * flame_pressure, dtype=np.float32),
        ]
    )
    work = work * (1.0 - haze_alpha[:, :, None]) + haze_color * haze_alpha[:, :, None]

    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    # Flame licks: vertical ribbon strokes rising from bottom and edges.
    for lick in range(52):
        base_x = int(((lick * 97) + math.sin(lick * 3.1) * 41) % width)
        side_bias = 0
        if lick % 5 == 0:
            side_bias = -int(width * 0.39)
        elif lick % 7 == 0:
            side_bias = int(width * 0.38)
        x = int(np.clip(base_x + side_bias, 0, width - 1))
        base_y = height + int(18 * math.sin(lick))
        length = int((height * (0.10 + 0.11 * flame_pressure)) * (0.55 + 0.45 * ((lick * 13) % 17) / 17.0))
        sway = 18 + 34 * flame_pressure
        points: list[list[int]] = []
        for step in range(7):
            p = step / 6.0
            y = int(base_y - p * length)
            drift = math.sin(time_seconds * (1.1 + lick * 0.013) + p * 5.8 + lick) * sway * p
            points.append([int(np.clip(x + drift, 0, width - 1)), int(np.clip(y, 0, height - 1))])
        color = (
            int(5 + 12 * flame_pressure),
            int(60 + 48 * flame_pressure),
            int(150 + 88 * flame_pressure),
        )
        cv2.polylines(overlay, [np.asarray(points, dtype=np.int32)], False, color, thickness=max(1, int(2 + 3 * flame_pressure)), lineType=cv2.LINE_AA)

    # Embers are small drifting streaks.
    for ember in range(115):
        birth = (ember * 0.137) % 1.0
        drift_t = (time_seconds * (0.045 + 0.012 * (ember % 5)) + birth) % 1.0
        x = int((ember * 149 + math.sin(time_seconds * 0.7 + ember) * 75 + phase * width * 0.08) % width)
        y = int(height * (1.03 - drift_t * 1.15))
        if y < -8 or y >= height:
            continue
        size = 1 + (ember % 3)
        dx = int(2 + 10 * math.sin(ember + time_seconds))
        color = (8, int(72 + 74 * ember_pressure), int(178 + 62 * ember_pressure))
        cv2.line(overlay, (x, y), (int(np.clip(x + dx, 0, width - 1)), int(np.clip(y - size * 3, 0, height - 1))), color, thickness=size, lineType=cv2.LINE_AA)

    work = cv2.addWeighted(np.clip(work, 0, 255).astype(np.uint8), 1.0, overlay, 0.45 + 0.18 * flame_pressure, 0.0).astype(np.float32)

    # Heat shimmer: subpixel horizontal flow near lower field.
    y_indices = np.arange(height, dtype=np.float32)[:, None]
    x_indices = np.arange(width, dtype=np.float32)[None, :]
    lower_weight = np.clip((y_indices - height * 0.42) / max(1.0, height * 0.58), 0.0, 1.0)
    wave = np.sin(y_indices * 0.035 + time_seconds * 2.3) * (2.2 + 3.4 * flame_pressure) * lower_weight
    map_x = (x_indices + wave).astype(np.float32)
    map_y = np.repeat(y_indices, width, axis=1).astype(np.float32)
    shimmer = cv2.remap(np.clip(work, 0, 255).astype(np.uint8), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    metadata = {
        "time_seconds": round(time_seconds, 6),
        "frame_index": frame_index,
        "beat": round(beat, 6),
        "rms": round(rms, 6),
        "bass": round(bass, 6),
        "mid": round(mid, 6),
        "high": round(high, 6),
        "flame_pressure": round(flame_pressure, 6),
        "ember_pressure": round(ember_pressure, 6),
        "haze_density": round(haze_density, 6),
        "state_layers": ["luma_pressure", "haze_field", "flame_lattice", "ember_drift", "heat_shimmer"],
    }
    return shimmer, metadata


def compose_sequence_frame(
    surfaces: list[np.ndarray],
    *,
    time_seconds: float,
    frame_index: int,
    frame_count: int,
    segment_seconds: float,
    fade_seconds: float,
    masks: list[dict[str, np.ndarray]] | None = None,
    state_keys: list[dict[str, float]] | None = None,
    animation_mode: str = "existing_regions",
    audio_features: list[dict[str, float]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    total_segments = len(surfaces)
    raw_index = min(total_segments - 1, int(time_seconds // segment_seconds))
    local = time_seconds - raw_index * segment_seconds
    transition_alpha = 0.0
    next_index = raw_index
    audio_state = None
    if audio_features:
        audio_state = audio_features[min(frame_index, len(audio_features) - 1)]
    state_key = interpolate_state_key(state_keys, frame_index=frame_index, frame_count=frame_count) if state_keys else {}

    if raw_index < total_segments - 1 and local >= segment_seconds - fade_seconds:
        next_index = raw_index + 1
        transition_alpha = smootherstep((local - (segment_seconds - fade_seconds)) / max(0.001, fade_seconds))

    phase = 0.0 if total_segments <= 1 else raw_index / (total_segments - 1)
    if animation_mode == "existing_regions":
        if masks is None or state_keys is None:
            raise ValueError("existing_regions mode requires masks and state_keys")
        current, environment = apply_existing_picture_state(
            surfaces[raw_index],
            masks[raw_index],
            time_seconds=time_seconds,
            frame_index=frame_index,
            state_key=state_key,
            audio_state=audio_state,
        )
        frame = current
        if next_index != raw_index:
            following, next_environment = apply_existing_picture_state(
                surfaces[next_index],
                masks[next_index],
                time_seconds=time_seconds,
                frame_index=frame_index,
                state_key=state_key,
                audio_state=audio_state,
            )
            frame = cv2.addWeighted(current, 1.0 - transition_alpha, following, transition_alpha, 0.0)
            environment["next_state_layers"] = next_environment["state_layers"]
    else:
        base = surfaces[raw_index]
        if next_index != raw_index:
            base = cv2.addWeighted(surfaces[raw_index], 1.0 - transition_alpha, surfaces[next_index], transition_alpha, 0.0)
        frame, environment = apply_environment_state(
            base,
            time_seconds=time_seconds,
            frame_index=frame_index,
            phase=phase,
            audio_state=audio_state,
        )
    metadata = {
        **environment,
        "source_segment_index": raw_index,
        "next_segment_index": next_index,
        "transition_alpha": round(float(transition_alpha), 6),
    }
    return frame, metadata


def encoder_command(path: Path, *, width: int, height: int, fps: int, encoder: str) -> list[str]:
    command = [
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
    ]
    if encoder == "h264_qsv":
        command.extend(["-vf", "format=nv12", "-c:v", "h264_qsv", "-global_quality", "18", "-preset", "fast"])
    else:
        command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"])
    command.append(str(path))
    return command


def mux_audio(*, visual_path: Path, audio_path: Path, final_path: Path, duration_seconds: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(visual_path),
        "-i",
        str(audio_path),
        "-t",
        f"{duration_seconds:.6f}",
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
    ]
    subprocess.run(command, check=True)


def render_state_snap_sequence(args: argparse.Namespace) -> dict[str, Any]:
    total_start_wall = time.perf_counter()
    render_start_cpu = time.process_time()
    memory_start = memory_snapshot()
    sampler = MachineSampler(interval_seconds=2.0)
    sampler.start()
    started_at = utc_now()

    run_id = slug(args.run_id)
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    width, height = parse_size(args.size)
    fps = int(args.fps)
    duration_seconds = float(args.seconds_per_picture) * 4.0
    frame_count = int(round(duration_seconds * fps))
    audio_path = Path(args.audio).expanduser().resolve() if args.audio else None
    audio_features: list[dict[str, float]] | None = None
    audio_probe: dict[str, Any] | None = None
    if audio_path is not None:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        audio_duration = probe_audio_duration(audio_path)
        samples = decode_audio_mono(audio_path, sample_rate=int(args.sample_rate), max_seconds=duration_seconds)
        audio_features = measure_audio_features(samples, sample_rate=int(args.sample_rate), fps=fps)
        if len(audio_features) < frame_count:
            audio_features.extend([audio_features[-1] if audio_features else {}] * (frame_count - len(audio_features)))
        audio_features = audio_features[:frame_count]
        audio_probe = {
            "path": str(audio_path),
            "sha256": sha256_file(audio_path),
            "duration_seconds": audio_duration,
            "sample_rate_used": int(args.sample_rate),
            "feature_count": len(audio_features),
            "ffmpeg_observer": True,
            "muxed_into_output": bool(args.mux_audio),
        }

    snaps = build_ordered_default_snaps(Path(args.snap_root))
    surfaces, snap_metadata = build_render_surfaces(
        snaps,
        canvas_size=(width, height),
        fit_mode=args.fit,
        render_surface=args.render_surface,
    )
    surface_masks = [derive_existing_motion_masks(surface) for surface in surfaces]
    state_keys = build_state_keys(key_count=int(args.state_key_count), duration_seconds=duration_seconds)
    mask_stats = [
        {
            "source_index": index,
            "fire_mask_pixels": int(np.count_nonzero(masks["fire"])),
            "haze_mask_pixels": int(np.count_nonzero(masks["haze"])),
            "reflection_mask_pixels": int(np.count_nonzero(masks["reflection"])),
        }
        for index, masks in enumerate(surface_masks)
    ]

    visual_path = run_dir / f"{run_id}_visual_only.mp4" if audio_path is not None and args.mux_audio else run_dir / f"{run_id}.mp4"
    video_path = run_dir / f"{run_id}.mp4"
    state_path = run_dir / f"{run_id}_frame_state.jsonl"
    command = encoder_command(visual_path, width=width, height=height, fps=fps, encoder=args.encoder)
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin was not opened")
    sampled_states: list[dict[str, Any]] = []
    try:
        with state_path.open("w", encoding="utf-8") as state_handle:
            for frame_index in range(frame_count):
                time_seconds = frame_index / fps
                frame, frame_state = compose_sequence_frame(
                    surfaces,
                    time_seconds=time_seconds,
                    frame_index=frame_index,
                    frame_count=frame_count,
                    segment_seconds=float(args.seconds_per_picture),
                    fade_seconds=float(args.fade_seconds),
                    masks=surface_masks,
                    state_keys=state_keys,
                    animation_mode=args.animation_mode,
                    audio_features=audio_features,
                )
                proc.stdin.write(frame.tobytes())
                state_handle.write(json.dumps(frame_state, allow_nan=False) + "\n")
                if frame_index % fps == 0:
                    sampled_states.append(frame_state)
        proc.stdin.close()
        code = proc.wait()
    except Exception:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
    if code != 0:
        raise RuntimeError(f"ffmpeg encoder failed with exit code {code}")
    if audio_path is not None and args.mux_audio:
        mux_audio(visual_path=visual_path, audio_path=audio_path, final_path=video_path, duration_seconds=duration_seconds)

    sampler.stop()
    ended_at = utc_now()
    total_until_open_seconds: float | None = None
    opened_at_utc: str | None = None
    if args.open:
        os.startfile(str(video_path))
        opened_at_utc = utc_now()
        total_until_open_seconds = time.perf_counter() - total_start_wall

    wall_seconds = time.perf_counter() - total_start_wall
    cpu_seconds = time.process_time() - render_start_cpu
    logical = os.cpu_count() or 1
    manifest = {
        "schema": "truevision_state_snap_sequence_render.v1",
        "run_id": run_id,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "opened_at_utc": opened_at_utc,
        "boundary": {
            "source_authority": "truevision_exact_photo_state_snap",
            "pixel_map_role": "render_surface_only",
            "state_is_not_pixels": True,
            "no_external_visual_creation_tools": True,
            "no_stock_assets": True,
            "state_manipulations_only": True,
            "existing_picture_regions_only": args.animation_mode == "existing_regions",
            "no_new_fire_or_ember_geometry": args.animation_mode == "existing_regions",
        },
        "render": {
            "video_path": str(video_path),
            "video_sha256": sha256_file(video_path),
            "visual_only_path": str(visual_path),
            "visual_only_sha256": sha256_file(visual_path),
            "frame_state_jsonl": str(state_path),
            "frame_state_sha256": sha256_file(state_path),
            "width": width,
            "height": height,
            "fps": fps,
            "duration_seconds": duration_seconds,
            "frame_count": frame_count,
            "seconds_per_picture": float(args.seconds_per_picture),
            "fade_seconds": float(args.fade_seconds),
            "fit": args.fit,
            "encoder": args.encoder,
            "animation_mode": args.animation_mode,
            "state_key_count": int(args.state_key_count),
        },
        "sequence_order": ["top_left", "lower_left", "lower_right", "top_right"],
        "state_keys": {
            "count": len(state_keys),
            "rate_hz": round(len(state_keys) / duration_seconds, 6),
            "render_frames_per_key": round(frame_count / len(state_keys), 6),
            "first": state_keys[0],
            "last": state_keys[-1],
        },
        "mask_stats": mask_stats,
        "audio": audio_probe,
        "source_snaps": snap_metadata,
        "sampled_frame_states": sampled_states,
        "machine": {
            "hardware": capture_hardware(),
            "memory_start": memory_start,
            "memory_end": memory_snapshot(),
            "samples": sampler.summary(),
            "wall_seconds_to_video_complete": round(wall_seconds, 6),
            "wall_seconds_until_open": round(total_until_open_seconds, 6) if total_until_open_seconds is not None else None,
            "process_cpu_seconds": round(cpu_seconds, 6),
            "avg_cpu_core_equivalent": round(cpu_seconds / max(0.000001, wall_seconds), 6),
            "avg_process_logical_cpu_percent": round((cpu_seconds / (max(0.000001, wall_seconds) * logical)) * 100.0, 6),
        },
    }
    manifest_path = run_dir / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "video_path": str(video_path),
        "manifest_path": str(manifest_path),
        "frame_state_jsonl": str(state_path),
        "wall_seconds_until_open": manifest["machine"]["wall_seconds_until_open"],
        "wall_seconds_to_video_complete": manifest["machine"]["wall_seconds_to_video_complete"],
        "process_cpu_seconds": manifest["machine"]["process_cpu_seconds"],
    }


def parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must look like 1920x1080")
    width, height = int(parts[0]), int(parts[1])
    if width < 2 or height < 2:
        raise argparse.ArgumentTypeError("size values must be positive")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a sequence from TrueVision still-state snap bundles.")
    parser.add_argument("--snap-root", default=str(DEFAULT_SNAP_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="four_screenshot_state_sequence_20s")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds-per-picture", type=float, default=5.0)
    parser.add_argument("--fade-seconds", type=float, default=0.85)
    parser.add_argument("--fit", choices=["cover", "contain"], default="cover")
    parser.add_argument("--render-surface", choices=["pixel_map", "cell_state"], default="pixel_map")
    parser.add_argument("--encoder", choices=["libx264", "h264_qsv"], default="libx264")
    parser.add_argument("--animation-mode", choices=["existing_regions", "overlay_environment"], default="existing_regions")
    parser.add_argument("--state-key-count", type=int, default=300)
    parser.add_argument("--audio", default="")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--mux-audio", action="store_true", default=True)
    parser.add_argument("--open", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = render_state_snap_sequence(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
