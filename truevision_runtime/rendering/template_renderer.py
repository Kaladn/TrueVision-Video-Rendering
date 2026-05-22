"""Template-driven TrueVision renderer for reusable audio/video visuals."""

from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from truevision_edge_audio_river import decode_audio_mono, measure_audio_features, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "template_renders"


@dataclass(frozen=True)
class RenderTemplate:
    template_id: str
    title: str
    visual_mode: str
    audio_path: Path
    output_root: Path
    run_id: str
    width: int
    height: int
    fps: int
    sample_rate: int
    encoder: str
    max_seconds: float | None
    style: dict[str, Any]
    scenes: list[dict[str, Any]]
    program_stamp: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def slug(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return clean.strip("_")[:96] or "truevision_template_render"


def resolve_project_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_render_template(path: Path) -> RenderTemplate:
    source = json.loads(path.read_text(encoding="utf-8"))
    output = source.get("output", {})
    audio = source.get("audio", {})
    render = source.get("render", {})
    run_id = slug(str(source.get("run_id") or source.get("template_id") or path.stem))
    return RenderTemplate(
        template_id=str(source.get("template_id") or path.stem),
        title=str(source.get("title") or path.stem),
        visual_mode=str(render.get("visual_mode") or source.get("visual_mode") or "mirror_maze_realism"),
        audio_path=resolve_project_path(str(audio["path"])),
        output_root=resolve_project_path(str(output.get("root", DEFAULT_OUTPUT_ROOT))),
        run_id=run_id,
        width=int(output.get("width", 1920)),
        height=int(output.get("height", 1080)),
        fps=int(output.get("fps", 30)),
        sample_rate=int(audio.get("sample_rate", 44100)),
        encoder=str(output.get("encoder", "h264_qsv")),
        max_seconds=float(output["max_seconds"]) if output.get("max_seconds") is not None else None,
        style=dict(source.get("style", {})),
        scenes=list(source.get("scenes", [])),
        program_stamp=output.get("program_stamp", "TrueVision Generation Lab"),
    )


def memory_snapshot() -> dict[str, Any]:
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
    try:
        kernel32 = ctypes.WinDLL("kernel32.dll")
        psapi = ctypes.WinDLL("psapi.dll")
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), ctypes.c_ulong]
        get_process_memory_info.restype = ctypes.c_int
        ok = get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
    except Exception:
        return snapshot
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


def capture_hardware() -> dict[str, Any]:
    ram = {"total_physical_bytes": None, "available_physical_bytes": None}
    gpus: list[dict[str, Any]] = []
    if platform.system().lower() == "windows":
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

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            ram = {
                "total_physical_bytes": int(status.ullTotalPhys),
                "available_physical_bytes": int(status.ullAvailPhys),
            }
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
        "ram": ram,
        "gpu": gpus,
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


def _hsv_to_bgr(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    hsv = np.asarray([[[hue % 180.0, np.clip(saturation, 0.0, 1.0) * 255, np.clip(value, 0.0, 1.0) * 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def scene_for_time(template: RenderTemplate, time_seconds: float, duration_seconds: float) -> dict[str, Any]:
    scenes = template.scenes or [{"scene_id": "mirror_confrontation", "start_norm": 0.0, "end_norm": 1.0}]
    norm = 0.0 if duration_seconds <= 0 else max(0.0, min(0.999999, time_seconds / duration_seconds))
    for scene in scenes:
        if float(scene.get("start_norm", 0.0)) <= norm < float(scene.get("end_norm", 1.0)):
            return scene
    return scenes[-1]


def _perspective_point(width: int, height: int, x_norm: float, y_norm: float, depth: float, shake_x: float, shake_y: float) -> tuple[int, int]:
    vanishing_x = width * (0.5 + shake_x)
    vanishing_y = height * (0.45 + shake_y)
    scale = 1.0 / (1.0 + depth * 1.35)
    x = vanishing_x + (x_norm - 0.5) * width * scale
    y = vanishing_y + (y_norm - 0.5) * height * scale
    return int(x), int(y)


def _draw_gradient_background(frame: np.ndarray, time_seconds: float, rms: float, bass: float, style: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    center = np.sqrt((x - 0.5) ** 2 + (y - 0.47) ** 2)
    pulse = 0.35 + 0.65 * rms
    red = np.clip((1.0 - center * 1.8) * 36 * pulse + bass * 20, 0, 70)
    blue = np.clip((1.0 - y) * 22 + (1.0 - center * 2.2) * 18, 0, 55)
    green = np.clip((1.0 - center * 2.0) * 10, 0, 26)
    frame[:, :, 0] = blue.astype(np.uint8)
    frame[:, :, 1] = green.astype(np.uint8)
    frame[:, :, 2] = red.astype(np.uint8)
    vignette = np.clip(1.0 - center * float(style.get("vignette_strength", 1.55)), 0.05, 1.0)
    frame[:] = (frame.astype(np.float32) * vignette[:, :, None]).astype(np.uint8)


def _draw_mirror_corridor(frame: np.ndarray, *, time_seconds: float, rms: float, bass: float, mid: float, high: float, scene: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    shake_x = math.sin(time_seconds * 1.7) * 0.018 * bass
    shake_y = math.cos(time_seconds * 1.2) * 0.012 * bass
    panel_count = 9
    for index in range(panel_count):
        depth_near = index * 0.18
        depth_far = depth_near + 0.22 + 0.04 * rms
        z = index / max(1, panel_count - 1)
        left_near_top = _perspective_point(width, height, -0.28, 0.02, depth_near, shake_x, shake_y)
        left_near_bottom = _perspective_point(width, height, -0.24, 1.12, depth_near, shake_x, shake_y)
        left_far_bottom = _perspective_point(width, height, -0.24, 1.06, depth_far, shake_x, shake_y)
        left_far_top = _perspective_point(width, height, -0.28, 0.08, depth_far, shake_x, shake_y)
        right_near_top = _perspective_point(width, height, 1.28, 0.02, depth_near, shake_x, shake_y)
        right_near_bottom = _perspective_point(width, height, 1.24, 1.12, depth_near, shake_x, shake_y)
        right_far_bottom = _perspective_point(width, height, 1.24, 1.06, depth_far, shake_x, shake_y)
        right_far_top = _perspective_point(width, height, 1.28, 0.08, depth_far, shake_x, shake_y)
        hue = 104 + 38 * math.sin(time_seconds * 0.2 + index)
        fill = _hsv_to_bgr(hue, 0.35 + high * 0.3, 0.10 + 0.18 * (1.0 - z) + rms * 0.08)
        edge = _hsv_to_bgr(hue + 10, 0.7, 0.42 + 0.35 * beat_value(mid, high))
        overlay = np.zeros_like(frame)
        for pts in (
            np.asarray([left_near_top, left_near_bottom, left_far_bottom, left_far_top], dtype=np.int32),
            np.asarray([right_near_top, right_near_bottom, right_far_bottom, right_far_top], dtype=np.int32),
        ):
            cv2.fillConvexPoly(overlay, pts, fill, cv2.LINE_AA)
            cv2.polylines(overlay, [pts], True, edge, max(1, int(2 + 5 * (1.0 - z))), cv2.LINE_AA)
        alpha = 0.18 + 0.18 * (1.0 - z)
        cv2.addWeighted(overlay, alpha, frame, 1.0, 0, dst=frame)

    horizon_y = int(height * (0.62 + 0.03 * math.sin(time_seconds * 0.4)))
    floor = np.asarray([[0, height], [width, height], [int(width * 0.56), horizon_y], [int(width * 0.44), horizon_y]], dtype=np.int32)
    cv2.fillConvexPoly(frame, floor, (7, 8, 12), cv2.LINE_AA)
    for lane in np.linspace(-0.42, 0.42, 9):
        p1 = (int(width * (0.5 + lane)), height)
        p2 = (int(width * (0.5 + lane * 0.12)), horizon_y)
        cv2.line(frame, p1, p2, (16, 28 + int(32 * rms), 48 + int(40 * mid)), 1, cv2.LINE_AA)


def _draw_crack_webs(frame: np.ndarray, *, time_seconds: float, rms: float, high: float, beat: float) -> None:
    height, width = frame.shape[:2]
    cluster_count = 8 + int(8 * beat)
    for cluster in range(cluster_count):
        side = -1 if cluster % 2 == 0 else 1
        cx = int(width * (0.5 + side * (0.22 + 0.25 * ((cluster * 11) % 7) / 7.0)))
        cy = int(height * (0.18 + 0.56 * ((cluster * 17) % 11) / 11.0))
        spokes = 5 + (cluster % 5)
        length = int(height * (0.035 + 0.055 * rms))
        for spoke in range(spokes):
            angle = spoke * math.tau / spokes + time_seconds * 0.025 + cluster
            last = (cx, cy)
            segments = 2 + (spoke % 3)
            for segment in range(segments):
                scale = (segment + 1) / segments
                bend = math.sin(cluster * 1.7 + spoke + segment) * 0.35
                px = int(cx + math.cos(angle + bend) * length * scale)
                py = int(cy + math.sin(angle + bend) * length * scale)
                value = int(70 + 130 * high + 45 * beat)
                cv2.line(frame, last, (px, py), (value, value + 10, value + 18), 1, cv2.LINE_AA)
                last = (px, py)


def beat_value(mid: float, high: float) -> float:
    return max(0.0, min(1.0, 0.45 * mid + 0.55 * high))


def _draw_smoke(frame: np.ndarray, *, time_seconds: float, rms: float, bass: float, high: float, scene: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    scene_seed = sum(ord(char) for char in str(scene.get("scene_id", "mirror"))) % 9973
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    density = np.zeros((height, width), dtype=np.float32)

    octave_specs = (
        (34, 19, 0.54, 0.020, 0.009, 18.0),
        (67, 37, 0.31, -0.013, 0.014, 34.0),
        (131, 73, 0.15, 0.007, -0.006, 58.0),
    )
    for index, (cols, rows, weight, drift_x, drift_y, blur_sigma) in enumerate(octave_specs):
        rng = np.random.default_rng(scene_seed + index * 1459)
        base = rng.random((rows, cols), dtype=np.float32)
        layer = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
        layer = np.roll(layer, int(time_seconds * width * (drift_x + 0.012 * bass)), axis=1)
        layer = np.roll(layer, int(time_seconds * height * (drift_y - 0.010 * high)), axis=0)
        layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma * 0.62)
        density += layer * weight

    curl = (
        np.sin((x * 7.4 + y * 2.1 + time_seconds * (0.17 + 0.10 * bass)) * math.tau)
        + np.sin((x * 2.7 - y * 5.8 - time_seconds * (0.11 + 0.08 * high)) * math.tau)
    ).astype(np.float32)
    density += curl * 0.055

    floor_rise = np.clip((y - 0.26) / 0.72, 0.0, 1.0) ** 1.25
    far_veil = np.clip(1.0 - np.abs(y - 0.47) * 1.7, 0.0, 1.0) ** 1.8
    side_spill = np.clip(1.0 - np.abs(x - 0.5) * 1.55, 0.0, 1.0)
    density = (density - float(density.min())) / max(1e-6, float(density.max() - density.min()))
    density = np.clip((density - 0.31) * (1.5 + 0.7 * rms), 0.0, 1.0)
    density *= np.clip(floor_rise * 0.72 + far_veil * 0.38 + side_spill * 0.14, 0.0, 1.0)
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=5.5 + 10.0 * rms, sigmaY=9.0 + 18.0 * rms)

    light_shaft = np.clip(1.0 - (x * 0.65 + y * 0.42), 0.0, 1.0) ** 2.4
    density = np.clip(density + light_shaft * (0.04 + 0.08 * high) * far_veil, 0.0, 1.0)

    alpha = np.clip(density * (0.20 + 0.34 * rms + 0.08 * bass), 0.0, 0.72)
    fog_color = np.zeros_like(frame, dtype=np.float32)
    fog_color[:, :, 0] = 58 + 38 * high + 18 * far_veil
    fog_color[:, :, 1] = 63 + 28 * high + 12 * far_veil
    fog_color[:, :, 2] = 70 + 42 * bass + 24 * far_veil

    frame_float = frame.astype(np.float32)
    frame_float = frame_float * (1.0 - alpha[:, :, None]) + fog_color * alpha[:, :, None]
    shadow = np.clip(1.0 - density * (0.07 + 0.10 * bass) * floor_rise, 0.78, 1.0)
    frame_float *= shadow[:, :, None]
    frame[:, :, :] = np.clip(frame_float, 0, 255).astype(np.uint8)


def _draw_shards(frame: np.ndarray, *, time_seconds: float, rms: float, bass: float, high: float, beat: float, scene: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    shard_count = 42 + int(45 * max(rms, beat))
    overlay = np.zeros_like(frame)
    for index in range(shard_count):
        seed = index * 17.13
        orbit = time_seconds * (0.18 + bass * 0.18) + seed
        depth = 0.16 + ((index * 0.037 + time_seconds * 0.018) % 1.0)
        spread = 0.12 + 0.72 * depth
        cx = int(width * (0.5 + math.sin(orbit) * spread * 0.72))
        cy = int(height * (0.49 + math.cos(orbit * 0.7) * spread * 0.42))
        size = int((height * (0.012 + 0.032 * (1.0 - depth))) * (1.0 + beat * 0.9))
        angle = orbit * 2.7
        points = []
        for side in range(3 + (index % 2)):
            radius = size * (0.65 + 0.55 * ((side + index) % 3) / 2.0)
            a = angle + side * math.tau / (3 + (index % 2))
            points.append((int(cx + math.cos(a) * radius), int(cy + math.sin(a) * radius)))
        pts = np.asarray(points, dtype=np.int32)
        hue = 92 + 44 * math.sin(index * 0.37 + time_seconds * 0.55)
        fill = _hsv_to_bgr(hue, 0.22 + 0.32 * high, 0.18 + 0.45 * (1.0 - depth) + 0.25 * beat)
        edge = _hsv_to_bgr(hue + 9, 0.78, 0.60 + 0.35 * beat)
        cv2.fillConvexPoly(overlay, pts, fill, cv2.LINE_AA)
        cv2.polylines(overlay, [pts], True, edge, 1 + int(2 * high), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.54, frame, 1.0, 0, dst=frame)


def _draw_silhouette(frame: np.ndarray, *, time_seconds: float, rms: float, bass: float, mid: float, beat: float, scene: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    cx = int(width * (0.5 + math.sin(time_seconds * 0.31) * 0.015 * bass))
    floor = int(height * 0.88)
    scale = height / 720.0
    glow = np.zeros_like(frame)
    body_h = int(205 * scale * (1.0 + 0.04 * rms))
    shoulder = int(58 * scale)
    head_r = int(29 * scale)
    torso = np.asarray(
        [
            (cx - shoulder, floor - body_h + int(36 * scale)),
            (cx + shoulder, floor - body_h + int(36 * scale)),
            (cx + int(21 * scale), floor),
            (cx - int(21 * scale), floor),
        ],
        dtype=np.int32,
    )
    cv2.circle(glow, (cx, floor - body_h), head_r + int(8 * scale), (8, 18, 35 + int(58 * beat)), -1, cv2.LINE_AA)
    cv2.circle(glow, (cx, floor - body_h), head_r, (1, 2, 5), -1, cv2.LINE_AA)
    cv2.fillConvexPoly(glow, torso, (1, 2, 5), cv2.LINE_AA)
    arm_y = floor - int(58 * scale)
    for side in (-1, 1):
        hand = (cx + side * int((86 + 44 * mid) * scale), arm_y + int(24 * math.sin(time_seconds * 2 + side) * scale))
        cv2.line(glow, (cx + side * int(30 * scale), arm_y), hand, (4, 8, 14), max(3, int(10 * scale)), cv2.LINE_AA)
        cv2.circle(glow, hand, max(3, int(8 * scale)), (35, 94 + int(90 * mid), 170 + int(80 * beat)), -1, cv2.LINE_AA)
    aura = cv2.GaussianBlur(glow, (0, 0), sigmaX=9 + 18 * beat)
    cv2.addWeighted(aura, 0.78, frame, 1.0, 0, dst=frame)
    cv2.addWeighted(glow, 1.0, frame, 1.0, 0, dst=frame)


def _draw_live_wires(frame: np.ndarray, *, time_seconds: float, rms: float, mid: float, high: float, beat: float) -> None:
    height, width = frame.shape[:2]
    wire_count = 6 + int(10 * beat)
    for index in range(wire_count):
        points = []
        side = -1 if index % 2 == 0 else 1
        base_x = width * (0.5 + side * (0.12 + 0.36 * ((index * 5) % max(1, wire_count)) / max(1, wire_count)))
        amplitude = width * (0.010 + 0.020 * high)
        for step in range(22):
            y = int(height * (0.12 + 0.80 * step / 21))
            x = int(base_x + math.sin(step * 0.72 + time_seconds * (2.2 + mid) + index) * amplitude)
            points.append((x, y))
        color = _hsv_to_bgr(102 + 34 * math.sin(index), 0.82, 0.38 + 0.50 * beat)
        cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, color, 1 + int(2 * high), cv2.LINE_AA)


def _draw_storm_city_base(frame: np.ndarray, *, time_seconds: float, rms: float, bass: float, high: float, scene: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    sky = np.zeros_like(frame, dtype=np.float32)
    storm = np.clip(1.0 - y * 1.28, 0.0, 1.0)
    fire_side = np.clip(1.0 - np.abs(x - 0.74) * 2.5, 0.0, 1.0) * np.clip(1.0 - y * 1.7, 0.0, 1.0)
    blue_side = np.clip(1.0 - np.abs(x - 0.26) * 2.2, 0.0, 1.0) * storm
    sky[:, :, 0] = 16 + 60 * storm + 22 * blue_side + 10 * high
    sky[:, :, 1] = 13 + 25 * storm + 8 * blue_side + 20 * fire_side * bass
    sky[:, :, 2] = 14 + 16 * storm + 78 * fire_side * (0.45 + 0.55 * rms)
    frame[:, :, :] = np.clip(sky, 0, 255).astype(np.uint8)

    horizon = int(height * (0.43 + 0.015 * math.sin(time_seconds * 0.2)))
    rng = np.random.default_rng(1129 + int(sum(ord(char) for char in str(scene.get("scene_id", "")))))
    for index in range(34):
        bx = int(width * (index / 33.0) + math.sin(index * 1.7) * width * 0.02)
        bw = int(width * (0.012 + 0.025 * rng.random()))
        bh = int(height * (0.10 + 0.26 * rng.random()))
        top = max(0, horizon - bh)
        cv2.rectangle(frame, (bx, top), (min(width, bx + bw), height), (7, 10, 16), -1)
        if index % 5 == 0:
            glow = 30 + int(70 * bass)
            cv2.rectangle(frame, (bx + bw // 2, top + bh // 3), (min(width, bx + bw // 2 + 2), min(height, top + bh // 3 + 8)), (glow, glow // 2, glow), -1)

    pavement_y = int(height * 0.58)
    cv2.rectangle(frame, (0, pavement_y), (width, height), (7, 8, 10), -1)
    for line in range(9):
        z = line / 8.0
        y_pos = int(pavement_y + (height - pavement_y) * (z ** 1.8))
        color = int(18 + 35 * (1.0 - z))
        cv2.line(frame, (0, y_pos), (width, y_pos + int(8 * z)), (color, color, color + 4), 1, cv2.LINE_AA)


def _draw_wet_pavement_reflections(frame: np.ndarray, *, time_seconds: float, rms: float, bass: float, high: float) -> None:
    height, width = frame.shape[:2]
    pavement_y = int(height * 0.58)
    yy = np.linspace(0.0, 1.0, height - pavement_y, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    ripple = (
        np.sin((xx * 10.0 + yy * 3.2 + time_seconds * 0.33) * math.tau)
        + 0.6 * np.sin((xx * 22.0 - yy * 7.5 - time_seconds * 0.51) * math.tau)
    ).astype(np.float32)
    mask = np.clip((1.0 - yy) ** 1.5 + ripple * 0.055, 0.0, 1.0)
    reflection = np.zeros((height - pavement_y, width, 3), dtype=np.float32)
    fire_band = np.clip(1.0 - np.abs(xx - 0.72) * 3.0, 0.0, 1.0)
    blue_band = np.clip(1.0 - np.abs(xx - 0.25) * 3.2, 0.0, 1.0)
    reflection[:, :, 0] = 22 + 86 * blue_band * (0.25 + high) + 12 * mask
    reflection[:, :, 1] = 16 + 35 * blue_band + 34 * fire_band * bass
    reflection[:, :, 2] = 18 + 120 * fire_band * (0.24 + rms) + 18 * mask
    reflection = cv2.GaussianBlur(reflection, (0, 0), sigmaX=9.0, sigmaY=2.2)
    alpha = np.clip(mask * (0.24 + 0.22 * bass), 0.0, 0.58)
    target = frame[pavement_y:, :, :].astype(np.float32)
    target = target * (1.0 - alpha[:, :, None]) + reflection * alpha[:, :, None]
    frame[pavement_y:, :, :] = np.clip(target, 0, 255).astype(np.uint8)


def _draw_rain(frame: np.ndarray, *, time_seconds: float, high: float, beat: float) -> None:
    height, width = frame.shape[:2]
    rain = np.zeros_like(frame)
    drop_count = max(80, int((width * height) / 5800 * (0.75 + high)))
    rng = np.random.default_rng(2441 + int(time_seconds * 18))
    slant = int(width * (-0.012 - 0.025 * beat))
    for index in range(drop_count):
        x = int(rng.random() * width)
        y = int(rng.random() * height)
        length = int(height * (0.018 + 0.045 * rng.random()) * (1.0 + high * 0.55))
        brightness = int(48 + 82 * rng.random() + 48 * high)
        cv2.line(rain, (x, y), (x + slant, min(height - 1, y + length)), (brightness, brightness, brightness + 5), 1, cv2.LINE_AA)
    rain = cv2.GaussianBlur(rain, (0, 0), sigmaX=0.45, sigmaY=0.95)
    cv2.addWeighted(rain, 0.62, frame, 1.0, 0, dst=frame)


def _draw_ember_ash_particles(frame: np.ndarray, *, time_seconds: float, rms: float, bass: float, high: float) -> None:
    height, width = frame.shape[:2]
    overlay = np.zeros_like(frame)
    count = max(55, int((width * height) / 9000 * (0.65 + bass + 0.35 * rms)))
    for index in range(count):
        seed = index * 19.713
        drift = (time_seconds * (0.028 + 0.018 * bass) + index * 0.037) % 1.0
        side_bias = 0.72 + 0.22 * math.sin(seed)
        x = int(width * ((side_bias + 0.20 * math.sin(seed + time_seconds * 0.9)) % 1.0))
        y = int(height * (0.88 - drift * 0.90 + 0.04 * math.sin(seed * 0.31 + time_seconds)))
        if y < 0 or y >= height:
            continue
        radius = max(1, int(height * (0.0025 + 0.0045 * ((index % 5) / 4.0)) * (1.0 + bass)))
        warmth = 120 + int(95 * bass)
        if index % 3 == 0:
            color = (35 + int(34 * high), 52 + int(56 * bass), warmth)
        else:
            gray = 62 + int(55 * high)
            color = (gray, gray, gray + 4)
        cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=1.6 + 2.4 * bass)
    cv2.addWeighted(overlay, 0.74, frame, 1.0, 0, dst=frame)


def _draw_lonely_silhouettes(frame: np.ndarray, *, time_seconds: float, rms: float, bass: float, mid: float, beat: float) -> None:
    height, width = frame.shape[:2]
    ground = int(height * 0.86)
    glow = np.zeros_like(frame)

    cx = int(width * (0.55 + 0.018 * math.sin(time_seconds * 0.62)))
    scale = height / 720.0
    body_h = int(238 * scale)
    head_r = max(3, int(27 * scale))
    shoulder = int(58 * scale)
    leg_sway = int(24 * scale * math.sin(time_seconds * 2.2))
    aura_color = (22 + int(38 * beat), 52 + int(90 * mid), 130 + int(90 * beat))
    cv2.ellipse(glow, (cx, ground - int(body_h * 0.35)), (int(86 * scale), int(150 * scale)), 0, 0, 360, aura_color, -1, cv2.LINE_AA)
    glow_blur = cv2.GaussianBlur(glow, (0, 0), sigmaX=22 + 26 * beat)
    cv2.addWeighted(glow_blur, 0.58, frame, 1.0, 0, dst=frame)

    body = np.asarray(
        [
            (cx - shoulder, ground - body_h + int(45 * scale)),
            (cx + shoulder, ground - body_h + int(45 * scale)),
            (cx + int(25 * scale), ground - int(42 * scale)),
            (cx - int(25 * scale), ground - int(42 * scale)),
        ],
        dtype=np.int32,
    )
    cv2.circle(frame, (cx, ground - body_h), head_r, (1, 2, 4), -1, cv2.LINE_AA)
    cv2.fillConvexPoly(frame, body, (1, 2, 4), cv2.LINE_AA)
    cv2.line(frame, (cx - int(14 * scale), ground - int(45 * scale)), (cx - int(33 * scale) - leg_sway, ground), (1, 2, 4), max(2, int(11 * scale)), cv2.LINE_AA)
    cv2.line(frame, (cx + int(14 * scale), ground - int(45 * scale)), (cx + int(34 * scale) + leg_sway, ground), (1, 2, 4), max(2, int(11 * scale)), cv2.LINE_AA)

    ghost_x = int(width * 0.31)
    ghost_ground = int(height * 0.77)
    ghost = np.zeros_like(frame)
    cv2.circle(ghost, (ghost_x, ghost_ground - int(118 * scale)), max(2, int(17 * scale)), (70, 74, 86), -1, cv2.LINE_AA)
    cv2.ellipse(ghost, (ghost_x, ghost_ground - int(58 * scale)), (int(28 * scale), int(65 * scale)), 0, 0, 360, (58, 62, 76), -1, cv2.LINE_AA)
    ghost = cv2.GaussianBlur(ghost, (0, 0), sigmaX=3.0 + 4.0 * rms)
    cv2.addWeighted(ghost, 0.20 + 0.16 * mid, frame, 1.0, 0, dst=frame)


def _apply_finish(frame: np.ndarray, *, time_seconds: float, rms: float, bass: float, high: float, style: dict[str, Any]) -> np.ndarray:
    bloom_strength = float(style.get("bloom_strength", 0.68))
    blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=3.0 + 8.0 * rms)
    cv2.addWeighted(blur, bloom_strength * (0.35 + 0.65 * high), frame, 1.0, 0, dst=frame)
    height, width = frame.shape[:2]
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    vignette = np.clip(1.12 - (x * x + y * y) * 0.45, 0.24, 1.0)
    finished = (frame.astype(np.float32) * vignette[:, :, None]).astype(np.uint8)
    noise_seed = int(time_seconds * 24) % 997
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(0.0, 3.0 + 9.0 * bass, size=finished.shape).astype(np.float32)
    finished = np.clip(finished.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return finished


def render_mirror_maze_frame(
    *,
    template: RenderTemplate,
    frame_state: dict[str, float],
    duration_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    width, height = template.width, template.height
    time_seconds = float(frame_state["time_seconds"])
    rms = float(frame_state["rms"])
    bass = float(frame_state["bass"])
    mid = float(frame_state["mid"])
    high = float(frame_state["high"])
    beat = float(frame_state["beat"])
    scene = scene_for_time(template, time_seconds, duration_seconds)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_gradient_background(frame, time_seconds, rms, bass, template.style)
    _draw_mirror_corridor(frame, time_seconds=time_seconds, rms=rms, bass=bass, mid=mid, high=high, scene=scene)
    _draw_crack_webs(frame, time_seconds=time_seconds, rms=rms, high=high, beat=beat)
    _draw_smoke(frame, time_seconds=time_seconds, rms=rms, bass=bass, high=high, scene=scene)
    _draw_shards(frame, time_seconds=time_seconds, rms=rms, bass=bass, high=high, beat=beat, scene=scene)
    _draw_live_wires(frame, time_seconds=time_seconds, rms=rms, mid=mid, high=high, beat=beat)
    _draw_silhouette(frame, time_seconds=time_seconds, rms=rms, bass=bass, mid=mid, beat=beat, scene=scene)

    if template.program_stamp:
        cv2.putText(
            frame,
            str(template.program_stamp),
            (24, height - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, height / 1080.0 * 0.72),
            (92, 120, 132),
            1,
            cv2.LINE_AA,
        )

    frame = _apply_finish(frame, time_seconds=time_seconds, rms=rms, bass=bass, high=high, style=template.style)
    metadata = {
        "frame_index": int(frame_state.get("frame_index", 0)),
        "time_seconds": round(time_seconds, 6),
        "scene_id": scene.get("scene_id"),
        "visual_mode": template.visual_mode,
        "layers": [
            "mirror_corridor",
            "mirror_cracks",
            "density_field_fog",
            "volumetric_smoke",
            "mirror_shards",
            "live_wires",
            "silhouette",
            "bloom_grain",
        ],
        "audio_features": {
            "rms": round(rms, 6),
            "bass": round(bass, 6),
            "mid": round(mid, 6),
            "high": round(high, 6),
            "beat": round(beat, 6),
        },
        "boundary": {
            "generated_state_media": "synthetic_not_evidence",
            "template_driven": True,
            "no_lyric_overlay": True,
            "fog_uses_density_field": True,
        },
    }
    return frame, metadata


def render_storm_ember_city_frame(
    *,
    template: RenderTemplate,
    frame_state: dict[str, float],
    duration_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    width, height = template.width, template.height
    time_seconds = float(frame_state["time_seconds"])
    rms = float(frame_state["rms"])
    bass = float(frame_state["bass"])
    mid = float(frame_state["mid"])
    high = float(frame_state["high"])
    beat = float(frame_state["beat"])
    scene = scene_for_time(template, time_seconds, duration_seconds)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_storm_city_base(frame, time_seconds=time_seconds, rms=rms, bass=bass, high=high, scene=scene)
    _draw_wet_pavement_reflections(frame, time_seconds=time_seconds, rms=rms, bass=bass, high=high)
    _draw_smoke(frame, time_seconds=time_seconds, rms=0.45 + rms * 0.45, bass=bass, high=high, scene=scene)
    _draw_lonely_silhouettes(frame, time_seconds=time_seconds, rms=rms, bass=bass, mid=mid, beat=beat)
    _draw_ember_ash_particles(frame, time_seconds=time_seconds, rms=rms, bass=bass, high=high)
    _draw_rain(frame, time_seconds=time_seconds, high=high, beat=beat)

    frame = _apply_finish(frame, time_seconds=time_seconds, rms=rms, bass=bass, high=high, style=template.style)
    metadata = {
        "frame_index": int(frame_state.get("frame_index", 0)),
        "time_seconds": round(time_seconds, 6),
        "scene_id": scene.get("scene_id"),
        "visual_mode": template.visual_mode,
        "layers": [
            "storm_city_base",
            "wet_pavement_reflections",
            "density_field_fog",
            "lonely_backlit_silhouette",
            "ember_ash_particles",
            "rain_streaks",
            "bloom_grain",
        ],
        "audio_features": {
            "rms": round(rms, 6),
            "bass": round(bass, 6),
            "mid": round(mid, 6),
            "high": round(high, 6),
            "beat": round(beat, 6),
        },
        "boundary": {
            "generated_state_media": "synthetic_not_evidence",
            "template_driven": True,
            "no_lyric_overlay": True,
            "no_external_visual_assets": True,
            "fog_uses_density_field": True,
        },
    }
    return frame, metadata


def render_template_frame(
    *,
    template: RenderTemplate,
    frame_state: dict[str, float],
    duration_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if template.visual_mode == "mirror_maze_realism":
        return render_mirror_maze_frame(template=template, frame_state=frame_state, duration_seconds=duration_seconds)
    if template.visual_mode == "storm_ember_city":
        return render_storm_ember_city_frame(template=template, frame_state=frame_state, duration_seconds=duration_seconds)
    raise ValueError(f"Unsupported visual_mode: {template.visual_mode}")


def _encoder_command(template: RenderTemplate, visual_path: Path, encoder: str) -> list[str]:
    base = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{template.width}x{template.height}",
        "-r",
        str(template.fps),
        "-i",
        "-",
        "-an",
    ]
    if encoder == "h264_qsv":
        base.extend(["-vf", "format=nv12", "-c:v", "h264_qsv", "-global_quality", "18", "-preset", "fast"])
    elif encoder == "hevc_qsv":
        base.extend(["-vf", "format=nv12", "-c:v", "hevc_qsv", "-global_quality", "20", "-preset", "fast"])
    else:
        base.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"])
    base.append(str(visual_path))
    return base


def _render_visual(template: RenderTemplate, features: list[dict[str, float]], duration_seconds: float, run_dir: Path, encoder: str) -> tuple[Path, Path, list[dict[str, Any]]]:
    visual_path = run_dir / f"{template.run_id}_visual_only.mp4"
    state_path = run_dir / f"{template.run_id}_frame_state.jsonl"
    proc = subprocess.Popen(_encoder_command(template, visual_path, encoder), stdin=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin was not opened")
    sampled: list[dict[str, Any]] = []
    try:
        with state_path.open("w", encoding="utf-8") as state_handle:
            for index, feature in enumerate(features):
                frame_state = dict(feature)
                frame_state["frame_index"] = index
                frame, metadata = render_template_frame(template=template, frame_state=frame_state, duration_seconds=duration_seconds)
                proc.stdin.write(frame.tobytes())
                state_handle.write(json.dumps(metadata, allow_nan=False) + "\n")
                if index % max(1, template.fps) == 0:
                    sampled.append(metadata)
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
        raise RuntimeError(f"ffmpeg encoder failed with exit code {code}: {encoder}")
    return visual_path, state_path, sampled


def render_template(template_path: Path, *, max_seconds: float | None = None, mux_audio: bool = True) -> dict[str, Any]:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    memory_start = memory_snapshot()
    component_timing: dict[str, float] = {}

    def mark(name: str, started: float) -> None:
        component_timing[name] = round(time.perf_counter() - started, 6)

    template = load_render_template(template_path)
    if max_seconds is not None:
        template = RenderTemplate(
            **{**template.__dict__, "max_seconds": max_seconds}
        )
    if template.visual_mode not in {"mirror_maze_realism", "storm_ember_city"}:
        raise ValueError(f"Unsupported visual_mode: {template.visual_mode}")
    if not template.audio_path.exists():
        raise FileNotFoundError(template.audio_path)

    started_at = utc_now()
    run_dir = template.output_root / template.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    phase = time.perf_counter()
    samples = decode_audio_mono(template.audio_path, sample_rate=template.sample_rate, max_seconds=template.max_seconds)
    features = measure_audio_features(samples, sample_rate=template.sample_rate, fps=template.fps)
    if template.max_seconds is not None:
        features = [feature for feature in features if feature["time_seconds"] < template.max_seconds]
    if not features:
        raise ValueError("Audio produced no renderable features")
    duration_seconds = len(features) / template.fps
    mark("audio_decode_feature_extract_seconds", phase)

    phase = time.perf_counter()
    encoder_used = template.encoder
    encoder_fallback = None
    try:
        visual_path, state_path, sampled = _render_visual(template, features, duration_seconds, run_dir, encoder_used)
    except RuntimeError:
        encoder_fallback = "libx264"
        encoder_used = encoder_fallback
        visual_path, state_path, sampled = _render_visual(template, features, duration_seconds, run_dir, encoder_used)
    mark("frame_synthesis_and_encode_seconds", phase)

    phase = time.perf_counter()
    thumb_path = run_dir / f"{template.run_id}_thumbnail.jpg"
    thumb_index = min(len(features) - 1, max(0, int(template.fps * min(8.0, duration_seconds * 0.4))))
    frame_state = dict(features[thumb_index])
    frame_state["frame_index"] = thumb_index
    thumb_frame, _ = render_template_frame(template=template, frame_state=frame_state, duration_seconds=duration_seconds)
    cv2.imwrite(str(thumb_path), thumb_frame)
    mark("thumbnail_write_seconds", phase)

    final_path = run_dir / f"{template.run_id}_full_audio.mp4" if mux_audio else visual_path
    audio_muxed = False
    if mux_audio:
        phase = time.perf_counter()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(visual_path),
                "-i",
                str(template.audio_path),
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
        mark("audio_mux_seconds", phase)

    phase = time.perf_counter()
    feature_arrays = {key: np.asarray([feature[key] for feature in features], dtype=np.float32) for key in ["rms", "bass", "mid", "high", "beat"]}
    manifest_path = run_dir / f"{template.run_id}_manifest.json"
    report_path = run_dir / f"{template.run_id}_report.md"
    manifest = {
        "run_id": template.run_id,
        "template_id": template.template_id,
        "title": template.title,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "claim": f"template_driven_{template.visual_mode}_video",
        "boundary": {
            "generated_state_media": "synthetic_not_evidence",
            "template_driven": True,
            "no_lyric_overlay": True,
        },
        "inputs": {
            "template_path": str(template_path.resolve()),
            "audio_path": str(template.audio_path),
            "audio_sha256": sha256_file(template.audio_path),
        },
        "render": {
            "visual_mode": template.visual_mode,
            "width": template.width,
            "height": template.height,
            "fps": template.fps,
            "frames": len(features),
            "duration_seconds": round(duration_seconds, 6),
            "encoder_requested": template.encoder,
            "encoder_used": encoder_used,
            "encoder_fallback": encoder_fallback,
            "hardware_encode": encoder_used.endswith("_qsv"),
            "scenes": template.scenes,
            "style": template.style,
        },
        "audio_feature_summary": {
            key: {
                "mean": round(float(np.mean(values)), 6),
                "max": round(float(np.max(values)), 6),
                "std": round(float(np.std(values)), 6),
            }
            for key, values in feature_arrays.items()
        },
        "sampled_frame_states": sampled[:360],
        "hardware": capture_hardware(),
        "machine_cost": _machine_cost(start_wall, start_cpu, memory_start),
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
    manifest["outputs"]["video_sha256"] = sha256_file(final_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(build_report(manifest), encoding="utf-8")
    manifest["outputs"]["manifest_sha256"] = sha256_file(manifest_path)
    mark("manifest_report_hash_seconds", phase)
    manifest["component_timing_seconds"] = component_timing
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(build_report(manifest), encoding="utf-8")
    return {
        "run_id": template.run_id,
        "video_mp4": str(final_path),
        "manifest_json": str(manifest_path),
        "report_md": str(report_path),
        "thumbnail_jpg": str(thumb_path),
        "frames": len(features),
        "duration_seconds": round(duration_seconds, 6),
        "encoder_used": encoder_used,
        "machine_cost": manifest["machine_cost"],
        "video_sha256": manifest["outputs"]["video_sha256"],
    }


def _format_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    return f"{int(value)} bytes ({float(value) / (1024.0 * 1024.0):.2f} MiB)"


def build_report(manifest: dict[str, Any]) -> str:
    machine = manifest["machine_cost"]
    memory_start = machine.get("memory_start", {})
    memory_end = machine.get("memory_end", {})
    visual_mode = str(manifest["render"]["visual_mode"])
    visual_label = {
        "mirror_maze_realism": "mirror-maze realism",
        "storm_ember_city": "storm ember city",
    }.get(visual_mode, visual_mode.replace("_", " "))
    lines = [
        f"# {manifest['run_id']} Report",
        "",
        "## Claim",
        "",
        f"Template-driven {visual_label} video generated from audio features and explicit AV state.",
        "",
        "## Boundary",
        "",
        "Generated state media is synthetic, not evidence. No lyric overlay was rendered.",
        "",
        "## Output",
        "",
        f"- Video: `{manifest['outputs']['video_mp4']}`",
        f"- Manifest: `{manifest['outputs']['manifest_json']}`",
        f"- Thumbnail: `{manifest['outputs']['thumbnail_jpg']}`",
        "",
        "## Render",
        "",
        f"- Visual mode: `{manifest['render']['visual_mode']}`",
        f"- Size/FPS: `{manifest['render']['width']}x{manifest['render']['height']} @ {manifest['render']['fps']}fps`",
        f"- Frames: `{manifest['render']['frames']}`",
        f"- Duration seconds: `{manifest['render']['duration_seconds']}`",
        f"- Encoder requested: `{manifest['render']['encoder_requested']}`",
        f"- Encoder used: `{manifest['render']['encoder_used']}`",
        f"- Hardware encode: `{manifest['render']['hardware_encode']}`",
        "",
        "## Machine Cost",
        "",
        f"- Wall seconds: `{machine['wall_seconds']}`",
        f"- Process CPU seconds: `{machine['process_cpu_seconds']}`",
        f"- Avg CPU core equivalent: `{machine['avg_cpu_core_equivalent']}`",
        f"- Avg logical CPU percent: `{machine['avg_process_logical_cpu_percent']}`",
        f"- Working set start: `{_format_bytes(memory_start.get('working_set_bytes'))}`",
        f"- Working set end: `{_format_bytes(memory_end.get('working_set_bytes'))}`",
        f"- Peak working set: `{_format_bytes(memory_end.get('peak_working_set_bytes'))}`",
        "",
        "## Component Timing",
        "",
    ]
    lines.extend(f"- {key}: `{value}` seconds" for key, value in manifest.get("component_timing_seconds", {}).items())
    lines.extend(["", "## System Components", ""])
    lines.append(f"- Render path: `template -> audio features -> {visual_mode} frames -> ffmpeg encoder -> manifest`")
    for gpu in manifest.get("hardware", {}).get("gpu", []):
        lines.append(f"- GPU detected: `{gpu.get('name')}` | RAM `{_format_bytes(gpu.get('adapter_ram_bytes'))}` | Driver `{gpu.get('driver_version')}`")
    return "\n".join(lines) + "\n"
