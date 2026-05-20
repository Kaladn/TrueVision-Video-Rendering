#!/usr/bin/env python3
"""Generate synthetic TrueVision-style state media from explicit scene rules.

This does not generate from prompts and does not produce evidence. It writes
the same replayable 16:9 cell-state shape used by the TrueVision capture lane,
then optionally renders it through the state replay path.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
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
DEFAULT_RUN_ID = "person_field_walk_5s_state_media"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _feature_index(name: str) -> int:
    return CELL_FEATURE_NAMES.index(name)


def _draw_disc(mask: np.ndarray, center_row: float, center_col: float, radius: float) -> None:
    rows, cols = mask.shape
    rr, cc = np.indices((rows, cols))
    mask |= (rr - center_row) ** 2 + (cc - center_col) ** 2 <= radius**2


def _draw_capsule(mask: np.ndarray, row_a: float, col_a: float, row_b: float, col_b: float, radius: float) -> None:
    rows, cols = mask.shape
    rr, cc = np.indices((rows, cols))
    ab_r = row_b - row_a
    ab_c = col_b - col_a
    denom = (ab_r * ab_r) + (ab_c * ab_c)
    if denom <= 0:
        _draw_disc(mask, row_a, col_a, radius)
        return
    t = ((rr - row_a) * ab_r + (cc - col_a) * ab_c) / denom
    t = np.clip(t, 0.0, 1.0)
    near_r = row_a + t * ab_r
    near_c = col_a + t * ab_c
    mask |= (rr - near_r) ** 2 + (cc - near_c) ** 2 <= radius**2


def _build_scene_rgb(
    *,
    frame_index: int,
    total_frames: int,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    rows, cols = grid_shape
    rr, cc = np.indices((rows, cols))
    progress = frame_index / max(1, total_frames - 1)
    walk_phase = progress * math.tau * 2.0
    horizon = int(rows * 0.47)

    rgb = np.zeros((rows, cols, 3), dtype=np.float32)
    sky_t = np.clip(rr / max(1, horizon), 0.0, 1.0)
    sky_top = np.asarray([92.0, 152.0, 220.0], dtype=np.float32)
    sky_low = np.asarray([178.0, 202.0, 224.0], dtype=np.float32)
    rgb[:] = sky_top * (1.0 - sky_t[:, :, None]) + sky_low * sky_t[:, :, None]

    field_mask = rr >= horizon
    field_t = np.clip((rr - horizon) / max(1, rows - horizon - 1), 0.0, 1.0)
    field_near = np.asarray([34.0, 112.0, 42.0], dtype=np.float32)
    field_far = np.asarray([92.0, 154.0, 72.0], dtype=np.float32)
    rgb[field_mask] = (field_far * (1.0 - field_t[:, :, None]) + field_near * field_t[:, :, None])[field_mask]

    # Thin horizon and field texture stay in state space so replay has structure.
    horizon_band = np.abs(rr - horizon) <= 1
    rgb[horizon_band] = [82.0, 108.0, 78.0]
    grass_wave = ((np.sin(cc * 0.33 + frame_index * 0.18) + np.sin((rr + cc) * 0.09)) * 5.0)
    rgb[field_mask, 1] += grass_wave[field_mask]
    rgb[field_mask, 0] -= grass_wave[field_mask] * 0.25

    ground_row = rows * 0.78
    center_col = cols * (0.18 + 0.64 * progress)
    bob = math.sin(walk_phase) * rows * 0.012
    head_row = ground_row - rows * 0.2 + bob
    torso_top = head_row + rows * 0.045
    hip_row = ground_row - rows * 0.055 + bob
    shoulder_row = torso_top + rows * 0.035
    leg_swing = math.sin(walk_phase) * cols * 0.035
    arm_swing = -math.sin(walk_phase) * cols * 0.03

    person_mask = np.zeros((rows, cols), dtype=bool)
    shadow_mask = np.zeros((rows, cols), dtype=bool)
    _draw_disc(person_mask, head_row, center_col, max(1.1, rows * 0.022))
    _draw_capsule(person_mask, torso_top, center_col, hip_row, center_col, max(1.1, cols * 0.008))
    _draw_capsule(person_mask, shoulder_row, center_col, hip_row - rows * 0.02, center_col + arm_swing, max(0.8, cols * 0.005))
    _draw_capsule(person_mask, shoulder_row, center_col, hip_row - rows * 0.02, center_col - arm_swing, max(0.8, cols * 0.005))
    _draw_capsule(person_mask, hip_row, center_col, ground_row, center_col + leg_swing, max(0.9, cols * 0.006))
    _draw_capsule(person_mask, hip_row, center_col, ground_row, center_col - leg_swing, max(0.9, cols * 0.006))
    _draw_capsule(shadow_mask, ground_row + rows * 0.018, center_col - cols * 0.035, ground_row + rows * 0.018, center_col + cols * 0.055, max(1.0, rows * 0.015))

    rgb[shadow_mask] = rgb[shadow_mask] * 0.45
    rgb[person_mask] = [30.0, 34.0, 38.0]

    state = {
        "scene": "person_walking_in_field",
        "frame_index": frame_index,
        "total_frames": total_frames,
        "progress": round(progress, 6),
        "layers": ["sky_layer", "horizon_line", "field_layer", "shadow_contact", "walking_person_actor"],
        "actor": {
            "kind": "walking_person",
            "center_col": round(float(center_col), 3),
            "ground_row": round(float(ground_row), 3),
            "walk_phase": round(float(walk_phase), 6),
            "leg_swing_cells": round(float(leg_swing), 3),
            "arm_swing_cells": round(float(arm_swing), 3),
        },
    }
    return np.clip(rgb, 0, 255).astype(np.uint8), state, person_mask, shadow_mask


def _luma_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb[:, :, 0].astype(np.float32)
        + 0.587 * rgb[:, :, 1].astype(np.float32)
        + 0.114 * rgb[:, :, 2].astype(np.float32)
    )


def build_person_field_cells(
    *,
    frame_index: int,
    total_frames: int,
    grid_shape: tuple[int, int] = (90, 160),
    previous_luma: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb, state, person_mask, shadow_mask = _build_scene_rgb(
        frame_index=frame_index,
        total_frames=total_frames,
        grid_shape=grid_shape,
    )
    rows, cols = grid_shape
    luma = _luma_from_rgb(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    if previous_luma is None or previous_luma.shape != (rows, cols):
        delta_luma = np.zeros((rows, cols), dtype=np.float32)
    else:
        delta_luma = np.abs(luma - previous_luma).astype(np.float32)

    edge_density = np.zeros((rows, cols), dtype=np.float32)
    edge_density[person_mask] = 1.0
    edge_density[shadow_mask] = np.maximum(edge_density[shadow_mask], 0.35)
    horizon = int(rows * 0.47)
    edge_density[np.abs(np.indices((rows, cols))[0] - horizon) <= 1] = np.maximum(
        edge_density[np.abs(np.indices((rows, cols))[0] - horizon) <= 1],
        0.45,
    )

    field_mask = np.indices((rows, cols))[0] >= horizon
    texture_energy = np.zeros((rows, cols), dtype=np.float32)
    texture_energy[field_mask] = 0.18
    texture_energy[person_mask] = 0.72
    motion_energy = delta_luma.copy()
    motion_energy[person_mask] = np.maximum(motion_energy[person_mask], 35.0)

    cells = np.zeros((rows, cols, len(CELL_FEATURE_NAMES)), dtype=np.float32)
    cells[:, :, _feature_index("rgb_mean_r")] = rgb[:, :, 0]
    cells[:, :, _feature_index("rgb_mean_g")] = rgb[:, :, 1]
    cells[:, :, _feature_index("rgb_mean_b")] = rgb[:, :, 2]
    cells[:, :, _feature_index("hsv_mean_h")] = hsv[:, :, 0]
    cells[:, :, _feature_index("hsv_mean_s")] = hsv[:, :, 1]
    cells[:, :, _feature_index("hsv_mean_v")] = hsv[:, :, 2]
    cells[:, :, _feature_index("luma_mean")] = luma
    cells[:, :, _feature_index("saturation_mean")] = hsv[:, :, 1]
    cells[:, :, _feature_index("delta_luma_abs")] = delta_luma
    cells[:, :, _feature_index("edge_density")] = edge_density
    cells[:, :, _feature_index("texture_energy")] = texture_energy
    cells[:, :, _feature_index("motion_energy")] = motion_energy
    return cells, state


def _block_deltas_from_cells(cells: np.ndarray, block_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = cells.shape[:2]
    block_rows, block_cols = block_shape
    row_scale = rows // block_rows
    col_scale = cols // block_cols
    motion = cells[:, :, _feature_index("motion_energy")]
    cropped = motion[: block_rows * row_scale, : block_cols * col_scale]
    return cropped.reshape(block_rows, row_scale, block_cols, col_scale).transpose(0, 2, 1, 3).mean(axis=(2, 3))


def _visual_resonance_from_cells(cells: np.ndarray, block_deltas: np.ndarray) -> dict[str, float]:
    energy = block_deltas.astype(np.float32)
    total = float(np.sum(energy))
    center = energy[energy.shape[0] // 3 : (energy.shape[0] * 2) // 3, energy.shape[1] // 3 : (energy.shape[1] * 2) // 3]
    return {
        "vis_energy_total": total,
        "vis_energy_mean": float(np.mean(energy)),
        "vis_energy_std": float(np.std(energy)),
        "vis_center_energy_ratio": float(np.sum(center) / total) if total else 0.0,
        "vis_flash_intensity": float(np.max(cells[:, :, _feature_index("luma_mean")]) / 255.0),
        "vis_static_ratio": float(np.mean(cells[:, :, _feature_index("motion_energy")] <= 0.001)),
        "vis_highfreq_energy": float(np.sum(cells[:, :, _feature_index("edge_density")])),
        "vis_lowfreq_energy": float(np.mean(cells[:, :, _feature_index("texture_energy")])),
        "vis_stutter_score": float(np.max(energy) - np.mean(energy)),
    }


def _system_hardware_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "compute_path": "CPU numpy/OpenCV state generation plus CPU OpenCV VideoWriter encode",
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


def _process_memory_snapshot() -> dict[str, Any]:
    try:
        import psutil

        process = psutil.Process()
        memory = process.memory_info()
        return {
            "rss_bytes": int(memory.rss),
            "vms_bytes": int(memory.vms),
        }
    except Exception as exc:  # pragma: no cover - optional environment detail
        return {"process_memory_error": str(exc)}


def _write_formula_report(
    *,
    path: Path,
    run_id: str,
    result: dict[str, Any],
    hardware: dict[str, Any],
    timing: dict[str, Any],
) -> None:
    path.write_text(
        "\n".join(
            [
                "# TrueVision Synthetic State Media Formula Report",
                "",
                "## Claim",
                "",
                "A 5-second no-sound walking-person scene was generated as TrueVision-shaped cell-state vectors, then replayed as video from those vectors.",
                "",
                "## Boundary",
                "",
                "```text",
                "This is synthetic state media.",
                "It is not evidence.",
                "It is not prompt video.",
                "It is a declared scene formula rendered into the same 16:9 cell-state shape used by TrueVision capture.",
                "```",
                "",
                "## Formula",
                "",
                "```text",
                "SceneState(t)",
                "  -> sky_layer + horizon_line + field_layer",
                "  -> walking_person_actor(x(t), walk_phase(t), limb_phase(t))",
                "  -> 90x160 addressed cells",
                "  -> 16-feature cell vectors",
                "  -> replayable TrueVision bundle",
                "```",
                "",
                "## Outputs",
                "",
                f"- Run ID: `{run_id}`",
                f"- Run dir: `{result['run_dir']}`",
                f"- Manifest: `{result['manifest_json']}`",
                f"- Summary: `{result['summary_json']}`",
                f"- Records: `{result['records_jsonl']}`",
                f"- Lossless replay: `{result.get('lossless_ffv1_mkv')}`",
                f"- Preview replay: `{result.get('preview_mp4v')}`",
                "",
                "## Hardware Used",
                "",
                "```json",
                json.dumps(hardware, indent=2),
                "```",
                "",
                "## Timing",
                "",
                "```json",
                json.dumps(timing, indent=2),
                "```",
                "",
                "## No Audio",
                "",
                "`audio_saved=false`; no audio stream is generated or muxed.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def generate_person_field_scene(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
    duration_seconds: float = 5.0,
    fps: int = 9,
    frame_shape: tuple[int, int] = (540, 960),
    grid_shape: tuple[int, int] = (90, 160),
    chunk_frames: int = 30,
    replay: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_utc = utc_now()
    hardware = _system_hardware_snapshot()
    process_start = _process_memory_snapshot()
    total_frames = int(round(duration_seconds * fps))
    block_shape = (9, 16)
    run_dir = output_root / run_id
    cell_dir = run_dir / "cell_state_npz"
    records: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    cell_buffer: list[np.ndarray] = []
    frame_numbers: list[int] = []
    previous_luma: np.ndarray | None = None

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

    for index in range(total_frames):
        elapsed = index / fps
        cells, scene_state = build_person_field_cells(
            frame_index=index,
            total_frames=total_frames,
            grid_shape=grid_shape,
            previous_luma=previous_luma,
        )
        previous_luma = cells[:, :, _feature_index("luma_mean")]
        frame_number = index + 1
        block_deltas = _block_deltas_from_cells(cells, block_shape)
        features = {
            "observed_at_utc": utc_now(),
            "wall_time_unix": time.time(),
            "timestamp": elapsed,
            "frame_number": frame_number,
            "fps": fps,
            "block_vector": block_deltas.reshape(-1),
            "blocks": block_deltas,
            "block_deltas": block_deltas,
            "visual_resonance": _visual_resonance_from_cells(cells, block_deltas),
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
                "path": str(cell_dir / f"{run_id}_cells_{len(chunks):04d}.npz"),
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
        records.append(build_record(features, run_id=run_id, elapsed_seconds=elapsed, include_blocks=True))
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
        "source": "declared_scene_formula",
        "scene": "person_walking_in_field",
        "audio_saved": False,
        "raw_frame_saved": False,
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
    replay_elapsed = time.perf_counter() - replay_start
    total_elapsed = time.perf_counter() - started
    process_end = _process_memory_snapshot()
    timing = {
        "started_at_utc": started_utc,
        "completed_at_utc": utc_now(),
        "total_seconds": round(total_elapsed, 6),
        "state_generation_seconds": round(total_elapsed - replay_elapsed, 6),
        "replay_seconds": round(replay_elapsed, 6),
        "frames": total_frames,
        "process_memory_start": process_start,
        "process_memory_end": process_end,
    }
    report_path = bundle["run_dir"] / f"{run_id}_formula_report.md"
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
        "formula_report": str(report_path),
        **replay_outputs,
    }
    _write_formula_report(path=report_path, run_id=run_id, result=result, hardware=hardware, timing=timing)
    result["formula_report_sha256"] = sha256_file(report_path)
    result["hardware"] = hardware
    result["timing"] = timing
    return clean_value(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a no-sound TrueVision-shaped synthetic state-media scene.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=9)
    parser.add_argument("--no-replay", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_person_field_scene(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        duration_seconds=args.duration,
        fps=args.fps,
        replay=not args.no_replay,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
