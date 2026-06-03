#!/usr/bin/env python3
"""Record full TrueVision frame-state telemetry.

This recorder maps observed screen frames into cell-level visual state:
grid deltas plus the full ScreenResonanceState feature set.

No raw frames are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from modules.screen_grid_mapper import ScreenGridMapper


RECORD_KIND = "compucogvision_full_frame_state"
DEFAULT_OUTPUT_ROOT = Path("storage/artifacts/truevision_captures")
CELL_FEATURE_NAMES = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "hsv_mean_h",
    "hsv_mean_s",
    "hsv_mean_v",
    "luma_mean",
    "luma_std",
    "saturation_mean",
    "delta_luma_abs",
    "edge_density",
    "texture_energy",
    "motion_energy",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def clean_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, dict):
        return {str(key): clean_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_value(child) for child in value]
    return value


def array_to_list(value: Any, *, flatten: bool = False) -> list:
    array = np.asarray(value)
    if flatten:
        array = array.reshape(-1)
    return clean_value(array.tolist())


def _cell_view(array: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = grid_shape
    height, width = array.shape[:2]
    cell_h = height // rows
    cell_w = width // cols
    if cell_h < 1 or cell_w < 1:
        raise ValueError("grid is too fine for frame dimensions")
    crop_h = cell_h * rows
    crop_w = cell_w * cols
    cropped = array[:crop_h, :crop_w]
    if array.ndim == 3:
        return cropped.reshape(rows, cell_h, cols, cell_w, array.shape[2]).transpose(0, 2, 1, 3, 4)
    return cropped.reshape(rows, cell_h, cols, cell_w).transpose(0, 2, 1, 3)


def build_video_cell_state(
    frame: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    previous_luma: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build the 16:9 addressed per-cell source vector layer."""
    rows, cols = grid_shape
    frame_rgb = np.asarray(frame, dtype=np.uint8)
    rgb_cells = _cell_view(frame_rgb, grid_shape)
    rgb_mean = rgb_cells.mean(axis=(2, 3)).astype(np.float32)
    rgb_std = rgb_cells.std(axis=(2, 3)).astype(np.float32)

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    hsv_mean = _cell_view(hsv, grid_shape).mean(axis=(2, 3)).astype(np.float32)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_cells = _cell_view(gray, grid_shape)
    luma_mean = gray_cells.mean(axis=(2, 3)).astype(np.float32)
    luma_std = gray_cells.std(axis=(2, 3)).astype(np.float32)

    edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32) / 255.0
    edge_density = _cell_view(edges, grid_shape).mean(axis=(2, 3)).astype(np.float32)

    if previous_luma is None or previous_luma.shape != (rows, cols):
        delta_luma = np.zeros((rows, cols), dtype=np.float32)
    else:
        delta_luma = np.abs(luma_mean - previous_luma).astype(np.float32)

    saturation_mean = hsv_mean[:, :, 1]
    cells = np.dstack(
        [
            rgb_mean[:, :, 0],
            rgb_mean[:, :, 1],
            rgb_mean[:, :, 2],
            rgb_std[:, :, 0],
            rgb_std[:, :, 1],
            rgb_std[:, :, 2],
            hsv_mean[:, :, 0],
            hsv_mean[:, :, 1],
            hsv_mean[:, :, 2],
            luma_mean,
            luma_std,
            saturation_mean,
            delta_luma,
            edge_density,
            luma_std,
            delta_luma,
        ]
    ).astype(np.float32)
    return {
        "cells": cells,
        "luma": luma_mean,
        "feature_names": list(CELL_FEATURE_NAMES),
    }


def build_record(
    features: dict[str, Any],
    *,
    run_id: str,
    elapsed_seconds: float,
    include_blocks: bool = True,
) -> dict[str, Any]:
    block_vector = np.asarray(features.get("block_vector", []), dtype=np.float64).reshape(-1)
    visual_resonance = clean_value(features.get("visual_resonance", {}))
    record = {
        "schema_version": 1,
        "record_kind": RECORD_KIND,
        "source": "truevision.screen_resonance_state",
        "run_id": run_id,
        "observed_at_utc": features.get("observed_at_utc") or utc_now(),
        "timestamp_unix": clean_value(features.get("wall_time_unix", features.get("timestamp"))),
        "capture_perf_seconds": clean_value(features.get("timestamp")),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "frame_number": int(features.get("frame_number", 0)),
        "fps": clean_value(float(features.get("fps", 0.0) or 0.0)),
        "screen_energy": clean_value(float(np.nansum(block_vector)) if block_vector.size else 0.0),
        "visual_resonance": visual_resonance,
        "geometry": build_geometry(features),
        "block_vector": array_to_list(block_vector, flatten=True),
        "raw_frame_saved": False,
        "raw_grid_saved": False,
        "notes": "Full CompuCogVision state telemetry: aspect-aware block deltas plus 20 visual resonance features.",
    }
    if features.get("cell_state_ref"):
        record["cell_state_ref"] = clean_value(features["cell_state_ref"])
    if include_blocks:
        record["blocks"] = array_to_list(features.get("blocks", []))
        record["block_deltas"] = array_to_list(features.get("block_deltas", []))
    return clean_value(record)


def build_geometry(features: dict[str, Any]) -> dict[str, Any]:
    geometry = dict(features.get("capture_geometry") or {})
    frame_shape = [
        int(geometry.get("frame_height", 0) or 0),
        int(geometry.get("frame_width", 0) or 0),
    ]
    grid_shape = [
        int(geometry.get("grid_rows", 0) or 0),
        int(geometry.get("grid_cols", 0) or 0),
    ]
    block_shape = [
        int(geometry.get("block_rows", 0) or 0),
        int(geometry.get("block_cols", 0) or 0),
    ]
    return {
        "source_shape": [
            int(geometry.get("source_height", 0) or 0),
            int(geometry.get("source_width", 0) or 0),
        ],
        "frame_shape": frame_shape,
        "frame_aspect_ratio": round(frame_shape[1] / frame_shape[0], 6) if frame_shape[0] else None,
        "grid_shape": grid_shape,
        "grid_aspect_ratio": round(grid_shape[1] / grid_shape[0], 6) if grid_shape[0] else None,
        "block_shape": block_shape,
        "block_aspect_ratio": round(block_shape[1] / block_shape[0], 6) if block_shape[0] else None,
        "capture_region": geometry.get("capture_region"),
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": sum(clean) / len(clean),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    visual_keys = sorted(
        {
            key
            for record in records
            for key in record.get("visual_resonance", {}).keys()
        }
    )
    visual_stats = {
        key: _stats([record.get("visual_resonance", {}).get(key) for record in records])
        for key in visual_keys
    }
    return {
        "schema_version": 1,
        "record_kind": "compucogvision_capture_summary",
        "frame_count": len(records),
        "duration_seconds": records[-1]["elapsed_seconds"] if records else 0.0,
        "geometry": records[0].get("geometry") if records else {},
        "screen_energy": _stats([record.get("screen_energy") for record in records]),
        "fps": _stats([record.get("fps") for record in records]),
        "visual_resonance": visual_stats,
        "raw_frame_saved": False,
        "raw_grid_saved": False,
    }


def write_capture_bundle(
    *,
    output_root: Path,
    run_id: str,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    cell_state_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / f"{run_id}_records.jsonl"
    summary_path = run_dir / f"{run_id}_summary.json"
    manifest_path = run_dir / f"{run_id}_manifest.json"

    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(clean_value(record), separators=(",", ":"), allow_nan=False) + "\n")

    summary = summarize_records(records)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "kind": "compucogvision_capture_manifest",
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "config": clean_value(config),
        "records": {
            "frame_count": len(records),
            "jsonl_path": str(records_path),
            "jsonl_sha256": sha256_file(records_path),
        },
        "cell_state": {
            "enabled": bool(cell_state_chunks),
            "format": "npz_compressed_float32",
            "feature_names": list(CELL_FEATURE_NAMES),
            "feature_count": len(CELL_FEATURE_NAMES),
            "chunks": clean_value(cell_state_chunks or []),
        },
        "summary": {
            "json_path": str(summary_path),
            "json_sha256": sha256_file(summary_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "boundary": {
            "raw_frame_saved": False,
            "raw_grid_saved": False,
            "screen_state_saved": "aspect-aware block summaries plus optional 16:9 addressed cell-state vectors",
            "visual_resonance_saved": "full 20-feature ScreenResonanceState output",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_dir": run_dir,
        "records_jsonl": records_path,
        "summary_json": summary_path,
        "manifest_json": manifest_path,
    }


def parse_resolution(value: str | None) -> tuple[int, int] | None:
    if not value or value.lower() in {"native", "none"}:
        return None
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError("resolution must look like 960x540 or native")
    return int(parts[0]), int(parts[1])


def parse_shape_xy(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError("shape must look like 160x90")
    width = int(parts[0])
    height = int(parts[1])
    if width < 1 or height < 1:
        raise ValueError("shape values must be positive")
    return height, width


def parse_region(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = value.replace("x", ",").split(",")
    if len(parts) != 4:
        raise ValueError("region must look like left,top,width,height")
    left, top, width, height = [int(part) for part in parts]
    if width < 1 or height < 1:
        raise ValueError("region width/height must be positive")
    return left, top, width, height


def capture_stop_requested(stop_file: str | Path | None) -> bool:
    """Return True when the operator-created stop file exists."""
    if not stop_file:
        return False
    return Path(stop_file).exists()


def write_cell_state_chunk(
    *,
    chunk_path: Path,
    chunk_id: int,
    cell_frames: list[np.ndarray],
    frame_numbers: list[int],
    grid_shape: tuple[int, int],
) -> dict[str, Any]:
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    cells = np.stack(cell_frames).astype(np.float32)
    np.savez_compressed(
        chunk_path,
        cell_state=cells,
        frame_numbers=np.asarray(frame_numbers, dtype=np.int32),
        feature_names=np.asarray(CELL_FEATURE_NAMES),
        grid_shape=np.asarray(grid_shape, dtype=np.int32),
    )
    return {
        "chunk_id": chunk_id,
        "path": str(chunk_path),
        "sha256": sha256_file(chunk_path),
        "frame_start": int(frame_numbers[0]),
        "frame_end": int(frame_numbers[-1]),
        "frame_count": len(frame_numbers),
        "shape": list(cells.shape),
    }


def run_capture(args: argparse.Namespace) -> dict[str, str | float | int]:
    run_id = args.run_id or f"music_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    capture_resolution = parse_resolution(args.resolution)
    grid_shape = parse_shape_xy(args.grid)
    block_shape = parse_shape_xy(args.blocks)
    capture_region = parse_region(args.region)
    run_dir = Path(args.output_root) / run_id
    cell_dir = run_dir / "cell_state_npz"
    config = {
        "duration_seconds": args.duration,
        "capture_fps": args.fps,
        "grid_shape_rows_cols": list(grid_shape),
        "grid_size_xy": [grid_shape[1], grid_shape[0]],
        "block_shape_rows_cols": list(block_shape),
        "block_size_xy": [block_shape[1], block_shape[0]],
        "monitor": args.monitor,
        "capture_region": list(capture_region) if capture_region else None,
        "capture_resolution": list(capture_resolution) if capture_resolution else "native",
        "include_blocks": args.include_blocks,
        "save_cell_state": args.save_cell_state,
        "cell_feature_names": list(CELL_FEATURE_NAMES),
        "cell_chunk_frames": args.cell_chunk_frames,
        "start_delay_seconds": args.start_delay,
        "operator_stop_file": str(args.stop_file) if args.stop_file else None,
    }

    if args.start_delay > 0:
        print(f"Starting capture in {args.start_delay} seconds. Start the video now.")
        time.sleep(args.start_delay)

    mapper = ScreenGridMapper(
        grid_size=grid_shape,
        block_size=block_shape,
        monitor=args.monitor,
        capture_fps=args.fps,
        capture_resolution=capture_resolution,
        capture_region=capture_region,
    )
    records: list[dict[str, Any]] = []
    cell_state_chunks: list[dict[str, Any]] = []
    cell_frame_buffer: list[np.ndarray] = []
    cell_frame_numbers: list[int] = []
    previous_cell_luma: np.ndarray | None = None
    start_wall = time.time()

    def flush_cell_chunk() -> None:
        if not cell_frame_buffer:
            return
        chunk_id = len(cell_state_chunks)
        chunk_path = cell_dir / f"{run_id}_cells_{chunk_id:04d}.npz"
        cell_state_chunks.append(
            write_cell_state_chunk(
                chunk_path=chunk_path,
                chunk_id=chunk_id,
                cell_frames=cell_frame_buffer,
                frame_numbers=cell_frame_numbers,
                grid_shape=grid_shape,
            )
        )
        cell_frame_buffer.clear()
        cell_frame_numbers.clear()

    try:
        while True:
            elapsed = time.time() - start_wall
            if elapsed >= args.duration:
                break
            if capture_stop_requested(args.stop_file):
                break
            frame_start = time.perf_counter()
            features = mapper.extract_features()
            features["wall_time_unix"] = time.time()
            features["observed_at_utc"] = utc_now()
            if args.save_cell_state:
                cell_state = build_video_cell_state(
                    features["frame"],
                    grid_shape=grid_shape,
                    previous_luma=previous_cell_luma,
                )
                previous_cell_luma = cell_state["luma"]
                chunk_id = len(cell_state_chunks)
                chunk_frame_index = len(cell_frame_buffer)
                frame_number = int(features.get("frame_number", 0))
                chunk_path = cell_dir / f"{run_id}_cells_{chunk_id:04d}.npz"
                features["cell_state_ref"] = {
                    "format": "npz_compressed_float32",
                    "path": str(chunk_path),
                    "chunk_id": chunk_id,
                    "chunk_frame_index": chunk_frame_index,
                    "frame_number": frame_number,
                    "grid_shape": list(grid_shape),
                    "cell_count": grid_shape[0] * grid_shape[1],
                    "feature_names": list(CELL_FEATURE_NAMES),
                    "feature_count": len(CELL_FEATURE_NAMES),
                }
                cell_frame_buffer.append(cell_state["cells"])
                cell_frame_numbers.append(frame_number)
                if len(cell_frame_buffer) >= args.cell_chunk_frames:
                    flush_cell_chunk()
            records.append(
                build_record(
                    features,
                    run_id=run_id,
                    elapsed_seconds=elapsed,
                    include_blocks=args.include_blocks,
                )
            )
            spent = time.perf_counter() - frame_start
            time.sleep(max(0.0, (1.0 / args.fps) - spent))
    finally:
        flush_cell_chunk()
        mapper.close()

    bundle = write_capture_bundle(
        output_root=Path(args.output_root),
        run_id=run_id,
        records=records,
        config=config,
        cell_state_chunks=cell_state_chunks,
    )
    return {
        "run_id": run_id,
        "frames": len(records),
        "duration_seconds": round(time.time() - start_wall, 3),
        "run_dir": str(bundle["run_dir"]),
        "records_jsonl": str(bundle["records_jsonl"]),
        "summary_json": str(bundle["summary_json"]),
        "manifest_json": str(bundle["manifest_json"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record full CompuCogVision visual resonance telemetry.")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--resolution", default="960x540")
    parser.add_argument("--grid", default="160x90", help="Video-shaped grid as width x height, e.g. 160x90")
    parser.add_argument("--blocks", default="16x9", help="Video-shaped block grid as width x height, e.g. 16x9")
    parser.add_argument("--region", default="", help="Optional absolute capture crop: left,top,width,height")
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--start-delay", type=float, default=10.0)
    parser.add_argument("--include-blocks", action="store_true", default=True)
    parser.add_argument("--cell-chunk-frames", type=int, default=30)
    parser.add_argument("--stop-file", default="", help="Stop cleanly when this file appears.")
    parser.add_argument("--no-cell-state", action="store_false", dest="save_cell_state")
    parser.set_defaults(save_cell_state=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_capture(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
