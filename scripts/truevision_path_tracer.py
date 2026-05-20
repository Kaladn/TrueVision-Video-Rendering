#!/usr/bin/env python3
"""Deterministic CPU path-tracing lane for TrueVision state media.

This renderer is synthetic only. It converts declared geometry, material,
lighting, and camera state into RGB frames, then stores those frames as the
same replayable TrueVision cell-state vectors used by the capture lane.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from truevision_resonance_recorder import (
    CELL_FEATURE_NAMES,
    build_record,
    clean_value,
    sha256_file,
    write_capture_bundle,
    write_cell_state_chunk,
)
from truevision_state_replay import replay_capture


DEFAULT_OUTPUT_ROOT = Path("storage/artifacts/truevision_generated")
DEFAULT_RUN_ID = "path_traced_grounded_sphere_5s"


@dataclass(frozen=True)
class PathTraceFrame:
    rgb: np.ndarray
    metadata: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _feature_index(name: str) -> int:
    return CELL_FEATURE_NAMES.index(name)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, 1.0e-8)


def _intersect_sphere(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    oc = origin - center
    half_b = np.sum(oc * direction, axis=-1)
    c = np.sum(oc * oc, axis=-1) - radius * radius
    disc = half_b * half_b - c
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    near = -half_b - sqrt_disc
    far = -half_b + sqrt_disc
    t = np.where(near > 1.0e-4, near, far)
    return np.where((disc >= 0.0) & (t > 1.0e-4), t, np.inf)


def _intersect_scene(
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    sphere_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sphere_t = _intersect_sphere(origin, direction, sphere_center, radius=0.72)
    plane_t = np.where(direction[..., 1] < -1.0e-5, (-1.0 - origin[..., 1]) / direction[..., 1], np.inf)
    plane_t = np.where(plane_t > 1.0e-4, plane_t, np.inf)

    hit_sphere = sphere_t < plane_t
    hit_t = np.where(hit_sphere, sphere_t, plane_t)
    hit = np.isfinite(hit_t)
    safe_hit_t = np.where(hit, hit_t, 0.0)
    hit_point = origin + direction * safe_hit_t[..., None]

    sphere_normal = _normalize(hit_point - sphere_center)
    plane_normal = np.zeros_like(hit_point)
    plane_normal[..., 1] = 1.0
    normal = np.where(hit_sphere[..., None], sphere_normal, plane_normal)

    sphere_albedo = np.asarray([0.82, 0.25, 0.18], dtype=np.float32)
    plane_albedo_a = np.asarray([0.23, 0.42, 0.18], dtype=np.float32)
    plane_albedo_b = np.asarray([0.36, 0.52, 0.22], dtype=np.float32)
    checker = (
        (np.floor(hit_point[..., 0] * 1.6).astype(np.int32)
        + np.floor(hit_point[..., 2] * 1.6).astype(np.int32))
        & 1
    ).astype(bool)
    plane_albedo = np.where(checker[..., None], plane_albedo_a, plane_albedo_b)
    albedo = np.where(hit_sphere[..., None], sphere_albedo, plane_albedo)
    roughness = np.where(hit_sphere, 0.22, 0.86)
    return hit, hit_t, hit_point, normal, albedo, roughness


def _sky(direction: np.ndarray) -> np.ndarray:
    t = np.clip(0.5 * (direction[..., 1] + 1.0), 0.0, 1.0)
    low = np.asarray([0.72, 0.78, 0.86], dtype=np.float32)
    high = np.asarray([0.18, 0.42, 0.78], dtype=np.float32)
    return low * (1.0 - t[..., None]) + high * t[..., None]


def _random_unit_vectors(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    vectors = rng.normal(0.0, 1.0, size=(*shape, 3)).astype(np.float32)
    return _normalize(vectors)


def _trace_sample(
    *,
    frame_shape: tuple[int, int],
    frame_index: int,
    total_frames: int,
    sample_index: int,
    max_bounces: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    height, width = frame_shape
    rng = np.random.default_rng(seed + frame_index * 1009 + sample_index * 9176)
    yy, xx = np.indices((height, width), dtype=np.float32)
    jitter = rng.random((height, width, 2), dtype=np.float32) - 0.5
    aspect = width / max(1, height)
    fov = math.radians(52.0)
    scale = math.tan(fov * 0.5)
    px = ((xx + 0.5 + jitter[..., 0]) / width * 2.0 - 1.0) * aspect * scale
    py = (1.0 - (yy + 0.5 + jitter[..., 1]) / height * 2.0) * scale
    direction = _normalize(np.dstack([px, py - 0.08, -np.ones_like(px)]).astype(np.float32))
    origin = np.zeros_like(direction)
    origin[..., 1] = 0.22
    origin[..., 2] = 1.3
    throughput = np.ones_like(direction)
    radiance = np.zeros_like(direction)

    progress = frame_index / max(1, total_frames - 1)
    sphere_center = np.asarray([-0.45 + 0.9 * progress, -0.22 + math.sin(progress * math.tau) * 0.04, -2.55], dtype=np.float32)
    light_dir = _normalize(np.asarray([-0.45, 0.82, 0.22], dtype=np.float32).reshape(1, 1, 3))
    light_color = np.asarray([1.0, 0.92, 0.78], dtype=np.float32)
    shadow_ray_tests = 0
    rays_cast = height * width

    for bounce in range(max_bounces):
        hit, _hit_t, hit_point, normal, albedo, roughness = _intersect_scene(
            origin,
            direction,
            sphere_center=sphere_center,
        )
        radiance += throughput * _sky(direction) * (~hit)[..., None]
        if not np.any(hit):
            break

        light_amount = np.clip(np.sum(normal * light_dir, axis=-1), 0.0, 1.0)
        shadow_origin = hit_point + normal * 1.0e-3
        shadow_t = _intersect_sphere(shadow_origin, np.broadcast_to(light_dir, direction.shape), sphere_center, radius=0.72)
        in_shadow = np.isfinite(shadow_t)
        shadow_ray_tests += int(np.count_nonzero(hit))
        direct = light_amount * np.where(in_shadow, 0.12, 1.0)
        radiance += throughput * albedo * direct[..., None] * light_color * hit[..., None] * (1.0 if bounce == 0 else 0.45)

        diffuse = _normalize(normal + _random_unit_vectors(rng, (height, width)))
        reflect = _normalize(direction - 2.0 * np.sum(direction * normal, axis=-1, keepdims=True) * normal)
        next_direction = _normalize(diffuse * roughness[..., None] + reflect * (1.0 - roughness[..., None]))
        origin = hit_point + normal * 1.0e-3
        direction = next_direction
        throughput *= albedo * (0.72 if bounce == 0 else 0.52) * hit[..., None]
        rays_cast += int(np.count_nonzero(hit))

    return radiance, {"rays_cast": rays_cast, "shadow_ray_tests": shadow_ray_tests}


def trace_path_frame(
    *,
    frame_index: int,
    total_frames: int,
    frame_shape: tuple[int, int] = (180, 320),
    samples_per_pixel: int = 4,
    max_bounces: int = 3,
    seed: int = 616,
) -> PathTraceFrame:
    """Render one deterministic CPU path-traced RGB frame."""
    if samples_per_pixel < 1:
        raise ValueError("samples_per_pixel must be >= 1")
    if max_bounces < 1:
        raise ValueError("max_bounces must be >= 1")

    started = time.perf_counter()
    accum = np.zeros((*frame_shape, 3), dtype=np.float32)
    rays_cast = 0
    shadow_ray_tests = 0
    for sample_index in range(samples_per_pixel):
        sample, stats = _trace_sample(
            frame_shape=frame_shape,
            frame_index=frame_index,
            total_frames=total_frames,
            sample_index=sample_index,
            max_bounces=max_bounces,
            seed=seed,
        )
        accum += sample
        rays_cast += stats["rays_cast"]
        shadow_ray_tests += stats["shadow_ray_tests"]

    color = np.clip(accum / samples_per_pixel, 0.0, None)
    color = color / (color + 1.0)
    color = np.power(np.clip(color, 0.0, 1.0), 1.0 / 2.2)
    rgb = np.clip(np.rint(color * 255.0), 0, 255).astype(np.uint8)
    metadata = {
        "renderer": "cpu_path_tracer",
        "samples_per_pixel": samples_per_pixel,
        "max_bounces": max_bounces,
        "seed": seed,
        "frame_index": frame_index,
        "total_frames": total_frames,
        "frame_shape": list(frame_shape),
        "rays_cast": rays_cast,
        "shadow_ray_tests": shadow_ray_tests,
        "render_seconds": round(time.perf_counter() - started, 6),
        "gpu_acceleration_used": False,
        "objects": ["ground_plane", "animated_diffuse_sphere", "sun_area_proxy", "sky_gradient"],
    }
    return PathTraceFrame(rgb=rgb, metadata=metadata)


def _cell_mean(channel: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = grid_shape
    height, width = channel.shape[:2]
    if height % rows == 0 and width % cols == 0:
        row_scale = height // rows
        col_scale = width // cols
        return channel[: rows * row_scale, : cols * col_scale].reshape(rows, row_scale, cols, col_scale).transpose(0, 2, 1, 3).mean(axis=(2, 3))
    return cv2.resize(channel.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)


def _cell_std(channel: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = grid_shape
    height, width = channel.shape[:2]
    if height % rows == 0 and width % cols == 0:
        row_scale = height // rows
        col_scale = width // cols
        return channel[: rows * row_scale, : cols * col_scale].reshape(rows, row_scale, cols, col_scale).transpose(0, 2, 1, 3).std(axis=(2, 3))
    local_mean = cv2.resize(channel.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
    local_sq = cv2.resize((channel.astype(np.float32) ** 2), (cols, rows), interpolation=cv2.INTER_AREA)
    return np.sqrt(np.maximum(local_sq - local_mean**2, 0.0))


def _cells_from_rgb(
    rgb: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    previous_luma: np.ndarray | None,
) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    rgb_f = rgb.astype(np.float32)
    luma = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
    gray = np.clip(luma, 0, 255).astype(np.uint8)
    edges = cv2.Canny(gray, 60, 140).astype(np.float32) / 255.0

    rows, cols = grid_shape
    cells = np.zeros((rows, cols, len(CELL_FEATURE_NAMES)), dtype=np.float32)
    for channel_index, name in enumerate(("rgb_mean_r", "rgb_mean_g", "rgb_mean_b")):
        cells[:, :, _feature_index(name)] = _cell_mean(rgb_f[:, :, channel_index], grid_shape)
    for channel_index, name in enumerate(("rgb_std_r", "rgb_std_g", "rgb_std_b")):
        cells[:, :, _feature_index(name)] = _cell_std(rgb_f[:, :, channel_index], grid_shape)
    cells[:, :, _feature_index("hsv_mean_h")] = _cell_mean(hsv[:, :, 0], grid_shape)
    cells[:, :, _feature_index("hsv_mean_s")] = _cell_mean(hsv[:, :, 1], grid_shape)
    cells[:, :, _feature_index("hsv_mean_v")] = _cell_mean(hsv[:, :, 2], grid_shape)
    luma_mean = _cell_mean(luma, grid_shape)
    cells[:, :, _feature_index("luma_mean")] = luma_mean
    cells[:, :, _feature_index("luma_std")] = _cell_std(luma, grid_shape)
    cells[:, :, _feature_index("saturation_mean")] = cells[:, :, _feature_index("hsv_mean_s")]
    cells[:, :, _feature_index("edge_density")] = _cell_mean(edges, grid_shape)
    cells[:, :, _feature_index("texture_energy")] = np.clip(cells[:, :, _feature_index("luma_std")] / 32.0, 0.0, 1.0)
    if previous_luma is None or previous_luma.shape != luma_mean.shape:
        delta = np.zeros_like(luma_mean, dtype=np.float32)
    else:
        delta = np.abs(luma_mean - previous_luma)
    cells[:, :, _feature_index("delta_luma_abs")] = delta
    cells[:, :, _feature_index("motion_energy")] = delta
    return cells


def _block_motion(cells: np.ndarray, block_shape: tuple[int, int] = (9, 16)) -> np.ndarray:
    motion = cells[:, :, _feature_index("motion_energy")]
    return cv2.resize(motion.astype(np.float32), (block_shape[1], block_shape[0]), interpolation=cv2.INTER_AREA)


def _visual_resonance(cells: np.ndarray, block_deltas: np.ndarray) -> dict[str, float]:
    luma = cells[:, :, _feature_index("luma_mean")]
    edge = cells[:, :, _feature_index("edge_density")]
    texture = cells[:, :, _feature_index("texture_energy")]
    motion = cells[:, :, _feature_index("motion_energy")]
    total_motion = float(np.sum(block_deltas))
    center = block_deltas[
        block_deltas.shape[0] // 3 : (block_deltas.shape[0] * 2) // 3,
        block_deltas.shape[1] // 3 : (block_deltas.shape[1] * 2) // 3,
    ]
    return {
        "vis_energy_total": total_motion,
        "vis_energy_mean": float(np.mean(block_deltas)),
        "vis_energy_std": float(np.std(block_deltas)),
        "vis_center_energy_ratio": float(np.sum(center) / total_motion) if total_motion else 0.0,
        "vis_flash_intensity": float(np.max(luma) / 255.0),
        "vis_static_ratio": float(np.mean(motion <= 0.001)),
        "vis_highfreq_energy": float(np.sum(edge)),
        "vis_lowfreq_energy": float(np.mean(texture)),
        "vis_stutter_score": float(np.max(block_deltas) - np.mean(block_deltas)),
    }


def build_path_traced_cells(
    *,
    frame_index: int,
    total_frames: int,
    frame_shape: tuple[int, int] = (180, 320),
    grid_shape: tuple[int, int] = (90, 160),
    samples_per_pixel: int = 4,
    max_bounces: int = 3,
    seed: int = 616,
    previous_luma: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    traced = trace_path_frame(
        frame_index=frame_index,
        total_frames=total_frames,
        frame_shape=frame_shape,
        samples_per_pixel=samples_per_pixel,
        max_bounces=max_bounces,
        seed=seed,
    )
    cells = _cells_from_rgb(traced.rgb, grid_shape=grid_shape, previous_luma=previous_luma)
    progress = frame_index / max(1, total_frames - 1)
    state = {
        "scene": "path_traced_grounded_sphere",
        "renderer": "cpu_path_tracer",
        "synthetic": True,
        "evidence": False,
        "frame_index": frame_index,
        "total_frames": total_frames,
        "progress": round(progress, 6),
        "path_tracing": traced.metadata,
        "state_language": {
            "geometry": ["ground_plane", "animated_sphere", "camera_pinhole"],
            "lighting": ["sun_directional_proxy", "sky_gradient"],
            "materials": ["diffuse_ground_checker", "low_roughness_red_sphere"],
            "math_lanes": ["ray_sphere_intersection", "ray_plane_intersection", "cosine_bounce", "shadow_ray"],
        },
    }
    return cells, clean_value(state)


def _system_hardware_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "compute_path": "CPU numpy/OpenCV deterministic path tracing plus CPU OpenCV replay encode",
        "gpu_acceleration_used": False,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        snapshot.update(
            {
                "cpu_logical": psutil.cpu_count(logical=True),
                "cpu_physical": psutil.cpu_count(logical=False),
                "ram_total_bytes": int(vm.total),
                "ram_available_bytes_at_start": int(vm.available),
            }
        )
    except Exception as exc:  # pragma: no cover - optional environment detail
        snapshot["psutil_error"] = str(exc)
    try:
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        if completed.returncode == 0 and completed.stdout.strip():
            gpu_data = json.loads(completed.stdout)
            snapshot["gpu_adapters_detected"] = gpu_data if isinstance(gpu_data, list) else [gpu_data]
    except Exception as exc:  # pragma: no cover - optional environment detail
        snapshot["gpu_inventory_error"] = str(exc)
    return clean_value(snapshot)


def _write_path_trace_report(
    *,
    path: Path,
    result: dict[str, Any],
    hardware: dict[str, Any],
    timing: dict[str, Any],
) -> None:
    path.write_text(
        "\n".join(
            [
                "# TrueVision Path-Tracing Lane Run Report",
                "",
                "## Claim",
                "",
                "A synthetic path-traced scene was generated from explicit geometry, lighting, material, and camera state, then stored as TrueVision cell-state vectors.",
                "",
                "## Boundary",
                "",
                "```text",
                "Forward TrueVision witnesses.",
                "Reverse TrueVision replays or demonstrates.",
                "Path tracing improves synthetic rendering state.",
                "Generated media remains synthetic and is not evidence.",
                "```",
                "",
                "## Renderer",
                "",
                "```json",
                json.dumps(result["renderer"], indent=2, allow_nan=False),
                "```",
                "",
                "## Outputs",
                "",
                f"- Run ID: `{result['run_id']}`",
                f"- Run dir: `{result['run_dir']}`",
                f"- Records: `{result['records_jsonl']}`",
                f"- Manifest: `{result['manifest_json']}`",
                f"- Summary: `{result['summary_json']}`",
                f"- Report: `{result['report_md']}`",
                "",
                "## Hardware",
                "",
                "```json",
                json.dumps(hardware, indent=2, allow_nan=False),
                "```",
                "",
                "## Timing",
                "",
                "```json",
                json.dumps(timing, indent=2, allow_nan=False),
                "```",
                "",
                "## Future Constraint",
                "",
                "One-hour full-power capture, math engines, and path-traced rendering must write through coordinated temporal-causality records. The learning path and rendering path should remain twin lanes: the learner scores and teaches; the renderer consumes explicit state and proves what it used.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def generate_path_traced_scene(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
    duration_seconds: float = 5.0,
    fps: int = 9,
    frame_shape: tuple[int, int] = (180, 320),
    grid_shape: tuple[int, int] = (90, 160),
    samples_per_pixel: int = 4,
    max_bounces: int = 3,
    seed: int = 616,
    chunk_frames: int = 30,
    replay: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_utc = utc_now()
    total_frames = int(round(duration_seconds * fps))
    block_shape = (9, 16)
    run_dir = output_root / run_id
    cell_dir = run_dir / "cell_state_npz"
    records: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    cell_buffer: list[np.ndarray] = []
    frame_numbers: list[int] = []
    previous_luma: np.ndarray | None = None
    hardware = _system_hardware_snapshot()

    def flush() -> None:
        if not cell_buffer:
            return
        chunk_id = len(chunks)
        chunk_path = cell_dir / f"{run_id}_cells_{chunk_id:04d}.npz"
        chunks.append(
            write_cell_state_chunk(
                chunk_path=chunk_path,
                chunk_id=chunk_id,
                cell_frames=cell_buffer,
                frame_numbers=frame_numbers,
                grid_shape=grid_shape,
            )
        )
        cell_buffer.clear()
        frame_numbers.clear()

    first_renderer_state: dict[str, Any] | None = None
    render_seconds = 0.0
    for index in range(total_frames):
        frame_started = time.perf_counter()
        cells, scene_state = build_path_traced_cells(
            frame_index=index,
            total_frames=total_frames,
            frame_shape=frame_shape,
            grid_shape=grid_shape,
            samples_per_pixel=samples_per_pixel,
            max_bounces=max_bounces,
            seed=seed,
            previous_luma=previous_luma,
        )
        render_seconds += time.perf_counter() - frame_started
        previous_luma = cells[:, :, _feature_index("luma_mean")]
        if first_renderer_state is None:
            first_renderer_state = scene_state["path_tracing"]
        frame_number = index + 1
        block_deltas = _block_motion(cells, block_shape=block_shape)
        cell_ref_path = cell_dir / f"{run_id}_cells_{len(chunks):04d}.npz"
        features = {
            "observed_at_utc": utc_now(),
            "wall_time_unix": time.time(),
            "timestamp": index / fps,
            "frame_number": frame_number,
            "fps": fps,
            "block_vector": block_deltas.reshape(-1),
            "blocks": block_deltas,
            "block_deltas": block_deltas,
            "visual_resonance": _visual_resonance(cells, block_deltas),
            "capture_geometry": {
                "source_height": frame_shape[0],
                "source_width": frame_shape[1],
                "frame_height": frame_shape[0],
                "frame_width": frame_shape[1],
                "grid_rows": grid_shape[0],
                "grid_cols": grid_shape[1],
                "block_rows": block_shape[0],
                "block_cols": block_shape[1],
                "capture_region": None,
            },
            "cell_state_ref": {
                "format": "npz_compressed_float32",
                "path": str(cell_ref_path),
                "chunk_id": len(chunks),
                "chunk_frame_index": len(cell_buffer),
                "frame_number": frame_number,
                "grid_shape": list(grid_shape),
                "cell_count": grid_shape[0] * grid_shape[1],
                "feature_names": list(CELL_FEATURE_NAMES),
                "feature_count": len(CELL_FEATURE_NAMES),
                "scene_state": scene_state,
            },
        }
        cell_buffer.append(cells)
        frame_numbers.append(frame_number)
        records.append(build_record(features, run_id=run_id, elapsed_seconds=index / fps, include_blocks=True))
        if len(cell_buffer) >= chunk_frames:
            flush()
    flush()

    config = {
        "duration_seconds": duration_seconds,
        "fps": fps,
        "frame_shape_rows_cols": list(frame_shape),
        "grid_shape_rows_cols": list(grid_shape),
        "grid_size_xy": [grid_shape[1], grid_shape[0]],
        "cell_feature_names": list(CELL_FEATURE_NAMES),
        "cell_chunk_frames": chunk_frames,
        "source": "declared_path_traced_scene_formula",
        "scene": "path_traced_grounded_sphere",
        "audio_saved": False,
        "raw_frame_saved": False,
        "renderer": {
            "renderer": "cpu_path_tracer",
            "samples_per_pixel": samples_per_pixel,
            "max_bounces": max_bounces,
            "seed": seed,
            "gpu_acceleration_used": False,
        },
        "temporal_causality_boundary": {
            "coordinated_write_required_for_long_runs": True,
            "one_hour_full_power_capture_future": True,
            "learner_renderer_twin_future": True,
        },
    }
    bundle = write_capture_bundle(
        output_root=output_root,
        run_id=run_id,
        records=records,
        config=config,
        cell_state_chunks=chunks,
    )

    replay_outputs: dict[str, Any] = {}
    replay_start = time.perf_counter()
    if replay:
        replay_result = replay_capture(bundle["run_dir"], output_dir=bundle["run_dir"] / "replay", fps=fps)
        replay_outputs = {
            "replay_report": replay_result["report"],
            "lossless_ffv1_mkv": replay_result["lossless_ffv1_mkv"]["path"],
            "lossless_ffv1_sha256": replay_result["lossless_ffv1_mkv"]["sha256"],
            "preview_mp4v": replay_result["preview_mp4v"]["path"],
            "preview_mp4v_sha256": replay_result["preview_mp4v"]["sha256"],
        }
    replay_seconds = time.perf_counter() - replay_start
    timing = {
        "started_at_utc": started_utc,
        "completed_at_utc": utc_now(),
        "total_seconds": round(time.perf_counter() - started, 6),
        "render_seconds": round(render_seconds, 6),
        "replay_seconds": round(replay_seconds, 6),
        "frames": total_frames,
    }
    result = {
        "run_id": run_id,
        "frames": total_frames,
        "duration_seconds": duration_seconds,
        "fps": fps,
        "audio_saved": False,
        "run_dir": str(bundle["run_dir"]),
        "records_jsonl": str(bundle["records_jsonl"]),
        "summary_json": str(bundle["summary_json"]),
        "manifest_json": str(bundle["manifest_json"]),
        "renderer": first_renderer_state
        or {
            "renderer": "cpu_path_tracer",
            "samples_per_pixel": samples_per_pixel,
            "max_bounces": max_bounces,
            "seed": seed,
        },
        **replay_outputs,
    }
    report_path = bundle["run_dir"] / f"{run_id}_path_trace_report.md"
    result["report_md"] = str(report_path)
    _write_path_trace_report(path=report_path, result=result, hardware=hardware, timing=timing)
    result["report_sha256"] = sha256_file(report_path)
    result["hardware"] = hardware
    result["timing"] = timing
    return clean_value(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic TrueVision path-traced state-media scene.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=9)
    parser.add_argument("--frame-shape", default="180x320", help="HEIGHTxWIDTH")
    parser.add_argument("--grid", default="90x160", help="ROWSxCOLS")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--bounces", type=int, default=3)
    parser.add_argument("--seed", type=int, default=616)
    parser.add_argument("--no-replay", action="store_true")
    return parser


def _parse_shape(value: str) -> tuple[int, int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"invalid shape {value!r}")
    first, second = int(parts[0]), int(parts[1])
    if first <= 0 or second <= 0:
        raise ValueError(f"invalid shape {value!r}")
    return first, second


def main() -> None:
    args = build_parser().parse_args()
    result = generate_path_traced_scene(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        duration_seconds=args.duration,
        fps=args.fps,
        frame_shape=_parse_shape(args.frame_shape),
        grid_shape=_parse_shape(args.grid),
        samples_per_pixel=args.samples,
        max_bounces=args.bounces,
        seed=args.seed,
        replay=not args.no_replay,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
