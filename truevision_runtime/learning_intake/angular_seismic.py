from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now


DIRECTION_COUNT = 16
DEGREES_PER_DIRECTION = 360.0 / DIRECTION_COUNT
DEFAULT_RINGS = (1, 2, 3, 4)


def derive_aspect_matched_grid(width: int, height: int, *, long_edge_cells: int = 48) -> tuple[int, int]:
    """Return rows, cols for a virtual analysis grid that preserves source aspect."""
    width = int(width or 0)
    height = int(height or 0)
    long_edge_cells = max(1, int(long_edge_cells or 48))
    if width <= 0 or height <= 0:
        return (max(1, round(long_edge_cells * 9 / 16)), long_edge_cells)
    if width >= height:
        cols = long_edge_cells
        rows = max(1, int(round(long_edge_cells * height / width)))
    else:
        rows = long_edge_cells
        cols = max(1, int(round(long_edge_cells * width / height)))
    return rows, cols


def derive_virtual_grid(width: int, height: int, *, long_edge_cells: int = 48, aspect_mode: str = "source") -> tuple[int, int]:
    mode = str(aspect_mode or "source").strip().lower().replace("-", "_")
    long_edge_cells = max(1, int(long_edge_cells or 48))
    if mode in {"square", "square_1_1", "1_1"}:
        return long_edge_cells, long_edge_cells
    if mode in {"landscape", "landscape_16_9", "16_9"}:
        return max(1, int(round(long_edge_cells * 9 / 16))), long_edge_cells
    if mode in {"portrait", "portrait_9_16", "9_16"}:
        return long_edge_cells, max(1, int(round(long_edge_cells * 9 / 16)))
    return derive_aspect_matched_grid(width, height, long_edge_cells=long_edge_cells)


def detect_content_bounds_from_frame(frame_bgr: np.ndarray) -> dict[str, Any]:
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[0] <= 1 or frame.shape[1] <= 1:
        return {"x": 0, "y": 0, "width": 0, "height": 0, "aspect_ratio": 1.0, "method": "invalid_full_frame"}
    height, width = frame.shape[:2]
    frame_f = frame.astype(np.float32) / 255.0
    border = max(2, int(round(min(width, height) * 0.035)))
    border_pixels = np.concatenate(
        [
            frame_f[:border, :, :].reshape(-1, 3),
            frame_f[-border:, :, :].reshape(-1, 3),
            frame_f[:, :border, :].reshape(-1, 3),
            frame_f[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    border_color = np.median(border_pixels, axis=0)
    color_distance = np.linalg.norm(frame_f - border_color.reshape(1, 1, 3), axis=2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    edge_threshold = max(0.025, float(np.percentile(edge, 82)) * 0.45)
    color_threshold = max(0.035, float(np.percentile(color_distance, 80)) * 0.35)
    mask = (color_distance > color_threshold) | (edge > edge_threshold)
    # Close small holes so large content regions survive even when fog/sky are low contrast.
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask = mask_u8 > 0
    row_score = np.mean(mask, axis=1)
    col_score = np.mean(mask, axis=0)
    row_threshold = max(0.01, float(np.max(row_score)) * 0.12)
    col_threshold = max(0.01, float(np.max(col_score)) * 0.12)
    active_rows = np.where(row_score >= row_threshold)[0]
    active_cols = np.where(col_score >= col_threshold)[0]
    if active_rows.size == 0 or active_cols.size == 0:
        x, y, w, h = 0, 0, width, height
        method = "fallback_full_frame"
    else:
        pad_x = max(2, int(round(width * 0.012)))
        pad_y = max(2, int(round(height * 0.012)))
        x0 = max(0, int(active_cols[0]) - pad_x)
        x1 = min(width, int(active_cols[-1]) + 1 + pad_x)
        y0 = max(0, int(active_rows[0]) - pad_y)
        y1 = min(height, int(active_rows[-1]) + 1 + pad_y)
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        if (w * h) < (width * height * 0.18):
            x, y, w, h = 0, 0, width, height
            method = "small_mask_fallback_full_frame"
        else:
            x, y = x0, y0
            method = "middle_frame_active_region"
    return {
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
        "aspect_ratio": round(float(w) / max(float(h), 1.0), 6),
        "method": method,
    }


def detect_middle_content_bounds(source_video: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {source_video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return {"x": 0, "y": 0, "width": 0, "height": 0, "aspect_ratio": 1.0, "method": "middle_frame_unavailable"}
    bounds = detect_content_bounds_from_frame(frame)
    bounds["middle_frame_index"] = frame_count // 2 if frame_count > 0 else 0
    return bounds


def _crop_frame(frame_bgr: np.ndarray, bounds: dict[str, Any] | None) -> np.ndarray:
    if not bounds:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    x = int(np.clip(int(bounds.get("x", 0)), 0, max(0, width - 1)))
    y = int(np.clip(int(bounds.get("y", 0)), 0, max(0, height - 1)))
    w = int(np.clip(int(bounds.get("width", width)), 1, width - x))
    h = int(np.clip(int(bounds.get("height", height)), 1, height - y))
    return frame_bgr[y : y + h, x : x + w]


def _safe_id(value: str | None, fallback: str = "angular_seismic") -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "")).strip("_")
    return safe or fallback


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_rgb_png(path: Path, pixels: np.ndarray) -> None:
    pixels = np.asarray(pixels, dtype=np.uint8)
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("PNG writer expects RGB pixels")
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, 9))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low + 1.0e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _write_curve_png(path: Path, values: list[float], *, color: tuple[int, int, int]) -> None:
    width = 440
    height = 160
    pixels = np.full((height, width, 3), 8, dtype=np.uint8)
    pixels[height - 24 : height - 22, 30 : width - 14] = 64
    pixels[18 : height - 22, 30:32] = 64
    if values:
        norm = _normalize(np.array(values, dtype=np.float32))
        points = [
            (
                int(30 + index * (width - 48) / max(1, len(norm) - 1)),
                int(height - 24 - float(value) * (height - 44)),
            )
            for index, value in enumerate(norm)
        ]
        for x, y in points:
            pixels[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)] = color
    _write_rgb_png(path, pixels)


def _write_direction_png(path: Path, radial: list[float], director: list[float]) -> None:
    width = 420
    height = 220
    pixels = np.full((height, width, 3), 8, dtype=np.uint8)
    center = np.array([width // 2, height // 2], dtype=np.float32)
    radius = 76.0
    radial_norm = _normalize(np.array(radial, dtype=np.float32))
    director_norm = _normalize(np.array(director, dtype=np.float32))
    for index in range(DIRECTION_COUNT):
        angle = math.radians(index * DEGREES_PER_DIRECTION)
        vector = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        endpoint = center + vector * (18.0 + radius * (0.25 + 0.75 * float(radial_norm[index])))
        pocket_angle = math.radians((index + 0.5) * DEGREES_PER_DIRECTION)
        pocket = center + np.array([math.cos(pocket_angle), math.sin(pocket_angle)], dtype=np.float32) * (
            18.0 + radius * (0.25 + 0.75 * float(director_norm[index]))
        )
        for scale, point, color in [
            (1.0, endpoint, (76, 210, 255)),
            (0.72, pocket, (255, 190, 82)),
        ]:
            x0 = int(round(point[0]))
            y0 = int(round(point[1]))
            cv2.line(pixels, tuple(center.astype(int)), (x0, y0), color, max(1, int(2 * scale)))
            cv2.circle(pixels, (x0, y0), 3, color, -1)
    cv2.circle(pixels, tuple(center.astype(int)), 5, (245, 245, 245), -1)
    _write_rgb_png(path, pixels)


def _grid_average(field: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return cv2.resize(np.asarray(field, dtype=np.float32), (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32)


def _weighted_center(weight: np.ndarray) -> tuple[float, float]:
    rows, cols = weight.shape
    total = float(np.sum(weight))
    if total <= 1.0e-9:
        return (cols - 1) / 2.0, (rows - 1) / 2.0
    yy, xx = np.mgrid[0:rows, 0:cols]
    return float(np.sum(xx * weight) / total), float(np.sum(yy * weight) / total)


def _sample_grid(field: np.ndarray, x: float, y: float) -> float:
    rows, cols = field.shape
    xi = int(np.clip(round(x), 0, cols - 1))
    yi = int(np.clip(round(y), 0, rows - 1))
    return float(field[yi, xi])


def _direction_label(index: int) -> str:
    labels = [
        "east",
        "east_southeast",
        "southeast",
        "south_southeast",
        "south",
        "south_southwest",
        "southwest",
        "west_southwest",
        "west",
        "west_northwest",
        "northwest",
        "north_northwest",
        "north",
        "north_northeast",
        "northeast",
        "east_northeast",
    ]
    return labels[index % DIRECTION_COUNT]


def _cell_maps(frame_bgr: np.ndarray, previous_luma: np.ndarray | None, *, grid_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    rows, cols = grid_shape
    frame = cv2.resize(frame_bgr, (max(64, cols * 4), max(64, rows * 4)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    saturation = hsv[:, :, 1] / 255.0
    blur = cv2.GaussianBlur(gray, (5, 5), 0.0)
    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y) * 2.6, 0.0, 1.0)
    texture = np.clip(np.abs(cv2.Laplacian(blur, cv2.CV_32F)) * 3.2, 0.0, 1.0)
    if previous_luma is not None and previous_luma.shape != gray.shape:
        previous_luma = cv2.resize(previous_luma, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    if previous_luma is None:
        delta = np.zeros_like(gray, dtype=np.float32)
    else:
        delta = np.abs(gray - previous_luma)
    specular = np.clip((gray - 0.62) * 2.6, 0.0, 1.0) * np.clip(1.0 - saturation * 0.55, 0.0, 1.0)
    maps = {
        "luma": _grid_average(gray, rows, cols),
        "edge": _grid_average(edge, rows, cols),
        "texture": _grid_average(texture, rows, cols),
        "motion": _grid_average(delta, rows, cols),
        "saturation": _grid_average(saturation, rows, cols),
        "specular": _grid_average(specular, rows, cols),
    }
    maps["softness"] = np.clip(1.0 - maps["edge"], 0.0, 1.0)
    maps["reflection_candidate"] = np.clip(maps["specular"] * 0.48 + maps["motion"] * 0.32 + maps["edge"] * 0.20, 0.0, 1.0)
    maps["human_motion_candidate"] = np.clip(maps["motion"] * 0.55 + maps["edge"] * 0.30 + maps["texture"] * 0.15, 0.0, 1.0)
    return maps


def _analyze_directions(maps: dict[str, np.ndarray], *, center_x: float, center_y: float, rings: tuple[int, ...]) -> dict[str, Any]:
    radial_energy = np.zeros(DIRECTION_COUNT, dtype=np.float32)
    director_energy = np.zeros(DIRECTION_COUNT, dtype=np.float32)
    radial_samples: list[dict[str, Any]] = []
    director_samples: list[dict[str, Any]] = []
    energy = np.clip(
        maps["motion"] * 0.34
        + maps["edge"] * 0.22
        + maps["reflection_candidate"] * 0.25
        + maps["texture"] * 0.12
        + maps["specular"] * 0.07,
        0.0,
        1.0,
    )
    for index in range(DIRECTION_COUNT):
        angle = math.radians(index * DEGREES_PER_DIRECTION)
        pocket_angle = math.radians((index + 0.5) * DEGREES_PER_DIRECTION)
        radial_values = []
        director_values = []
        for ring in rings:
            radial_values.append(_sample_grid(energy, center_x + math.cos(angle) * ring, center_y + math.sin(angle) * ring))
            director_values.append(
                _sample_grid(energy, center_x + math.cos(pocket_angle) * ring, center_y + math.sin(pocket_angle) * ring)
            )
        radial_mean = float(np.mean(radial_values)) if radial_values else 0.0
        director_mean = float(np.mean(director_values)) if director_values else 0.0
        radial_energy[index] = radial_mean
        director_energy[index] = director_mean
        radial_samples.append(
            {
                "direction_index": index,
                "label": _direction_label(index),
                "angle_degrees": round(index * DEGREES_PER_DIRECTION, 3),
                "energy": round(radial_mean, 6),
            }
        )
        director_samples.append(
            {
                "direction_index": index,
                "label": f"{_direction_label(index)}_pocket",
                "angle_degrees": round((index + 0.5) * DEGREES_PER_DIRECTION, 3),
                "energy": round(director_mean, 6),
            }
        )
    dominant_index = int(np.argmax(radial_energy + director_energy))
    return {
        "dominant_direction_index": dominant_index,
        "dominant_direction_label": _direction_label(dominant_index),
        "dominant_angle_degrees": round(dominant_index * DEGREES_PER_DIRECTION, 3),
        "radial_energy": [round(float(value), 6) for value in radial_energy],
        "director_energy": [round(float(value), 6) for value in director_energy],
        "radial_samples": radial_samples,
        "director_samples": director_samples,
        "field_coherence": round(float(np.max(radial_energy + director_energy) - np.mean(radial_energy + director_energy)), 6),
    }


def _seismic_trace(frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    impulse = np.array([float(row["impulse"]) for row in frame_rows], dtype=np.float32)
    if impulse.size == 0:
        return {"peak_frame_index": 0, "peak_time_seconds": 0.0, "impulse_peak": 0.0, "rise_time_frames": 0, "decay_time_frames": 0}
    peak_index = int(np.argmax(impulse))
    baseline = float(np.median(impulse[: max(1, peak_index)])) if peak_index else float(impulse[0])
    peak = float(impulse[peak_index])
    threshold = baseline + max(peak - baseline, 0.0) * 0.22
    start = 0
    for index in range(peak_index, -1, -1):
        if float(impulse[index]) <= threshold:
            start = index
            break
    end = len(impulse) - 1
    for index in range(peak_index + 1, len(impulse)):
        if float(impulse[index]) <= threshold:
            end = index
            break
    after = impulse[peak_index : min(len(impulse), peak_index + 12)]
    return {
        "baseline": round(baseline, 6),
        "peak_frame_index": peak_index,
        "peak_time_seconds": frame_rows[peak_index]["time_seconds"],
        "impulse_peak": round(peak, 6),
        "rise_time_frames": int(max(0, peak_index - start)),
        "decay_time_frames": int(max(0, end - peak_index)),
        "aftershock_curve": [round(float(value), 6) for value in after],
        "oscillation": round(float(np.std(np.diff(impulse))) if impulse.size > 2 else 0.0, 6),
    }


def _read_video_frames(
    source_video: Path,
    *,
    loop_count: int,
    sample_stride: int,
    max_frames: int,
    crop_bounds: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {source_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []
    global_index = 0
    for _loop in range(max(1, loop_count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        local_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if local_index % max(1, sample_stride) == 0:
                frames.append(_crop_frame(frame, crop_bounds))
                if len(frames) >= max_frames:
                    break
            local_index += 1
            global_index += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    metadata = {
        "fps": round(fps, 6),
        "source_frame_count": frame_count,
        "width": width,
        "height": height,
        "source_duration_seconds": round(frame_count / fps, 6) if fps > 0 and frame_count > 0 else 0.0,
        "loop_count": loop_count,
        "sample_stride": sample_stride,
        "sampled_frames": len(frames),
        "logical_duration_seconds": round((frame_count / fps) * max(1, loop_count), 6) if fps > 0 and frame_count > 0 else 0.0,
        "content_bounds": crop_bounds or None,
    }
    return frames, metadata


def build_angular_seismic_profile_from_frames(
    frames_bgr: Iterable[np.ndarray],
    *,
    run_id: str,
    source_label: str,
    fps: float,
    loop_count: int,
    sample_stride: int = 1,
    grid_shape: tuple[int, int] = (27, 48),
    rings: tuple[int, ...] = DEFAULT_RINGS,
) -> dict[str, Any]:
    frame_rows: list[dict[str, Any]] = []
    previous_luma_full: np.ndarray | None = None
    radial_accum = np.zeros(DIRECTION_COUNT, dtype=np.float32)
    director_accum = np.zeros(DIRECTION_COUNT, dtype=np.float32)
    center_path: list[list[float]] = []
    previous_center: tuple[float, float] | None = None
    for sampled_index, frame_bgr in enumerate(frames_bgr):
        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        maps = _cell_maps(frame_bgr, previous_luma_full, grid_shape=grid_shape)
        weight = np.clip(
            maps["human_motion_candidate"] * 0.40 + maps["reflection_candidate"] * 0.42 + maps["edge"] * 0.18,
            0.0,
            1.0,
        )
        center_x, center_y = _weighted_center(weight)
        center_path.append([round(center_x / max(grid_shape[1] - 1, 1), 6), round(center_y / max(grid_shape[0] - 1, 1), 6)])
        direction = _analyze_directions(maps, center_x=center_x, center_y=center_y, rings=rings)
        radial_accum += np.array(direction["radial_energy"], dtype=np.float32)
        director_accum += np.array(direction["director_energy"], dtype=np.float32)
        if previous_center is None:
            center_velocity = [0.0, 0.0]
        else:
            center_velocity = [
                round((center_x - previous_center[0]) / max(grid_shape[1] - 1, 1), 6),
                round((center_y - previous_center[1]) / max(grid_shape[0] - 1, 1), 6),
            ]
        impulse = float(
            np.mean(maps["motion"]) * 0.38
            + np.mean(maps["reflection_candidate"]) * 0.28
            + np.mean(maps["edge"]) * 0.18
            + np.mean(maps["texture"]) * 0.16
        )
        frame_rows.append(
            {
                "frame_index": sampled_index,
                "time_seconds": round(sampled_index * max(1, sample_stride) / max(fps, 1.0), 6),
                "center_xy": center_path[-1],
                "center_velocity_xy": center_velocity,
                "dominant_direction": direction["dominant_direction_label"],
                "dominant_angle_degrees": direction["dominant_angle_degrees"],
                "luma_mean": round(float(np.mean(maps["luma"])), 6),
                "edge_density": round(float(np.mean(maps["edge"])), 6),
                "texture_energy": round(float(np.mean(maps["texture"])), 6),
                "motion_magnitude": round(float(np.mean(maps["motion"])), 6),
                "specular_reflection": round(float(np.mean(maps["specular"])), 6),
                "reflection_candidate": round(float(np.mean(maps["reflection_candidate"])), 6),
                "human_motion_candidate": round(float(np.mean(maps["human_motion_candidate"])), 6),
                "softness": round(float(np.mean(maps["softness"])), 6),
                "field_coherence": direction["field_coherence"],
                "impulse": round(impulse, 6),
            }
        )
        previous_luma_full = gray_full
        previous_center = (center_x, center_y)
    if not frame_rows:
        raise ValueError("no frames available for angular seismic profile")
    radial_mean = radial_accum / max(1, len(frame_rows))
    director_mean = director_accum / max(1, len(frame_rows))
    combined = radial_mean + director_mean
    dominant_index = int(np.argmax(combined))
    profile = {
        "schema_version": "truevision_angular_seismic_profile_v0",
        "created_at_utc": utc_now(),
        "run_id": _safe_id(run_id),
        "source": {
            "source_label": source_label,
            "source_kind": "local_video_frame_state",
            "loop_count": loop_count,
        },
        "grid": {
            "rows": grid_shape[0],
            "cols": grid_shape[1],
            "direction_count": DIRECTION_COUNT,
            "degrees_per_direction": DEGREES_PER_DIRECTION,
            "rings": list(rings),
            "radial_cells": DIRECTION_COUNT * len(rings),
            "director_cells": DIRECTION_COUNT * len(rings),
        },
        "frame_count": len(frame_rows),
        "frame_summaries": frame_rows,
        "angular_signature": {
            "dominant_direction_index": dominant_index,
            "dominant_direction_label": _direction_label(dominant_index),
            "dominant_angle_degrees": round(dominant_index * DEGREES_PER_DIRECTION, 3),
            "radial_energy": [round(float(value), 6) for value in radial_mean],
            "director_energy": [round(float(value), 6) for value in director_mean],
            "field_coherence_mean": round(float(np.mean([row["field_coherence"] for row in frame_rows])), 6),
        },
        "seismic_trace": _seismic_trace(frame_rows),
        "candidate_profiles": {
            "human_movement": {
                "mean": round(float(np.mean([row["human_motion_candidate"] for row in frame_rows])), 6),
                "peak": round(float(np.max([row["human_motion_candidate"] for row in frame_rows])), 6),
                "center_path_start": center_path[0],
                "center_path_end": center_path[-1],
            },
            "glass_reflections": {
                "mean": round(float(np.mean([row["reflection_candidate"] for row in frame_rows])), 6),
                "peak": round(float(np.max([row["reflection_candidate"] for row in frame_rows])), 6),
                "specular_mean": round(float(np.mean([row["specular_reflection"] for row in frame_rows])), 6),
            },
            "walking_camera_relation": {
                "center_path_variance": round(float(np.var(np.array(center_path, dtype=np.float32))), 6),
                "motion_mean": round(float(np.mean([row["motion_magnitude"] for row in frame_rows])), 6),
                "softness_mean": round(float(np.mean([row["softness"] for row in frame_rows])), 6),
            },
        },
        "boundary": {
            "source_video_preserved": True,
            "no_frame_dump": True,
            "compact_profile_only": True,
            "not_optical_lightfield": True,
            "state_focus_after_capture": True,
        },
    }
    profile["profile_sha256"] = stable_hash(profile)
    return profile


def build_angular_seismic_profile_from_video(
    source_video: Path,
    *,
    run_id: str,
    loop_count: int = 3,
    sample_stride: int = 6,
    max_frames: int = 360,
    grid_shape: tuple[int, int] | None = None,
    long_edge_cells: int = 48,
    aspect_mode: str = "source",
    rings: tuple[int, ...] = DEFAULT_RINGS,
) -> dict[str, Any]:
    if not source_video.exists():
        raise FileNotFoundError(str(source_video))
    content_bounds = detect_middle_content_bounds(source_video) if str(aspect_mode).lower() in {"middle", "middle_content", "content"} else None
    frames, metadata = _read_video_frames(
        source_video,
        loop_count=loop_count,
        sample_stride=sample_stride,
        max_frames=max_frames,
        crop_bounds=content_bounds,
    )
    if grid_shape is None:
        aspect_width = int((content_bounds or {}).get("width") or metadata.get("width") or 0)
        aspect_height = int((content_bounds or {}).get("height") or metadata.get("height") or 0)
        grid_shape = derive_virtual_grid(
            aspect_width,
            aspect_height,
            long_edge_cells=long_edge_cells,
            aspect_mode=aspect_mode,
        )
    profile = build_angular_seismic_profile_from_frames(
        frames,
        run_id=run_id,
        source_label=str(source_video),
        fps=float(metadata["fps"]),
        loop_count=loop_count,
        sample_stride=sample_stride,
        grid_shape=grid_shape,
        rings=rings,
    )
    profile["source"].update(
        {
            "video_path": str(source_video),
            "video_sha256": _file_sha256(source_video),
            "metadata": metadata,
            "virtual_surface": {
                "aspect_mode": aspect_mode,
                "grid_rows": grid_shape[0],
                "grid_cols": grid_shape[1],
                "source_aspect_ratio": round(float(metadata["width"]) / max(float(metadata["height"]), 1.0), 6),
                "analysis_aspect_ratio": round(float(grid_shape[1]) / max(float(grid_shape[0]), 1.0), 6),
                "content_bounds": content_bounds,
            },
        }
    )
    profile["profile_sha256"] = stable_hash({key: value for key, value in profile.items() if key != "profile_sha256"})
    return profile


def _verify_profile(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("profile_sha256")
    clone = {key: value for key, value in payload.items() if key != "profile_sha256"}
    return expected == stable_hash(clone)


def write_angular_seismic_profile_from_video(args: dict[str, Any], *, storage_root: Path) -> dict[str, Any]:
    source_video = Path(str(args.get("source_video") or args.get("video") or ""))
    run_id = _safe_id(str(args.get("run_id") or source_video.stem or "angular_seismic_video"))
    aspect_mode = str(args.get("aspect_mode") or "source").strip().lower()
    grid_cols = int(args.get("grid_cols") or 48)
    grid_rows = int(args.get("grid_rows") or 27)
    grid_shape = (grid_rows, grid_cols) if aspect_mode == "fixed" else None
    rings = tuple(int(part) for part in args.get("rings", DEFAULT_RINGS)) if not isinstance(args.get("rings"), str) else tuple(
        int(part.strip()) for part in str(args.get("rings")).split(",") if part.strip()
    )
    profile = build_angular_seismic_profile_from_video(
        source_video,
        run_id=run_id,
        loop_count=int(args.get("loop_count") or 3),
        sample_stride=int(args.get("sample_stride") or 6),
        max_frames=int(args.get("max_frames") or 360),
        grid_shape=grid_shape,
        long_edge_cells=int(args.get("long_edge_cells") or max(grid_cols, grid_rows, 48)),
        aspect_mode=aspect_mode,
        rings=rings or DEFAULT_RINGS,
    )
    profile_root = storage_root / "artifacts" / "angular_seismic"
    manifest_root = storage_root / "manifests" / "angular_seismic"
    receipt_root = storage_root / "receipts" / "angular_seismic"
    report_root = storage_root / "reports" / "angular_seismic" / f"{run_id}_graphs"
    for path in (profile_root, manifest_root, receipt_root, report_root):
        path.mkdir(parents=True, exist_ok=True)
    profile_path = profile_root / f"{run_id}_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")
    if not _verify_profile(profile_path):
        raise ValueError("angular seismic profile verification failed")
    frame_rows = profile["frame_summaries"]
    graphs = {
        "impulse_curve.png": report_root / "impulse_curve.png",
        "reflection_curve.png": report_root / "reflection_curve.png",
        "human_motion_curve.png": report_root / "human_motion_curve.png",
        "direction_energy.png": report_root / "direction_energy.png",
    }
    _write_curve_png(graphs["impulse_curve.png"], [row["impulse"] for row in frame_rows], color=(245, 245, 255))
    _write_curve_png(graphs["reflection_curve.png"], [row["reflection_candidate"] for row in frame_rows], color=(120, 190, 255))
    _write_curve_png(graphs["human_motion_curve.png"], [row["human_motion_candidate"] for row in frame_rows], color=(255, 184, 88))
    _write_direction_png(
        graphs["direction_energy.png"],
        profile["angular_signature"]["radial_energy"],
        profile["angular_signature"]["director_energy"],
    )
    graph_paths = {name: str(path) for name, path in graphs.items()}
    manifest = {
        "schema_version": "truevision_angular_seismic_manifest_v0",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "profile_json": str(profile_path),
        "profile_sha256": profile["profile_sha256"],
        "graphs": graph_paths,
        "source": profile["source"],
        "angular_signature": profile["angular_signature"],
        "seismic_trace": profile["seismic_trace"],
        "boundary": profile["boundary"],
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    manifest_path = manifest_root / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    report = {
        "schema_version": "truevision_angular_seismic_report_v0",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "summary": {
            "frames_sampled": profile["frame_count"],
            "loop_count": profile["source"]["loop_count"],
            "dominant_direction": profile["angular_signature"]["dominant_direction_label"],
            "dominant_angle_degrees": profile["angular_signature"]["dominant_angle_degrees"],
            "human_movement_peak": profile["candidate_profiles"]["human_movement"]["peak"],
            "reflection_peak": profile["candidate_profiles"]["glass_reflections"]["peak"],
            "impulse_peak": profile["seismic_trace"]["impulse_peak"],
        },
        "useful_for_renderer": [
            "glass_reflection_directionality",
            "walking_camera_relation",
            "human_movement_center_path",
            "reflection_specular_pulse",
        ],
        "profile_json": str(profile_path),
        "manifest_json": str(manifest_path),
        "graphs": graph_paths,
    }
    report["report_sha256"] = stable_hash(report)
    report_path = storage_root / "reports" / "angular_seismic" / f"{run_id}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    receipt = {
        "schema_version": "truevision_angular_seismic_receipt_v0",
        "created_at_utc": utc_now(),
        "tool": "angular_seismic_from_local_video",
        "run_id": run_id,
        "source_video": str(source_video),
        "source_video_preserved": source_video.exists(),
        "profile_json": str(profile_path),
        "profile_sha256": profile["profile_sha256"],
        "manifest_json": str(manifest_path),
        "report_json": str(report_path),
        "graphs": graph_paths,
        "boundary": profile["boundary"],
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    receipt_path = receipt_root / f"{run_id}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_id": run_id,
        "profile_json": str(profile_path),
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "report_json": str(report_path),
        "graphs": graph_paths,
        "summary": report["summary"],
    }
