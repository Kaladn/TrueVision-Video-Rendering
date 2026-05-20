#!/usr/bin/env python3
"""Replay CompuCogVision 16:9 cell-state tensors as video.

This reconstructs a deterministic playback from stored cell vectors. It is
forensic to the recorded state layer, not to original raw pixels that were not
saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _feature_index(feature_names: list[str], name: str) -> int:
    try:
        return feature_names.index(name)
    except ValueError as exc:
        raise ValueError(f"missing cell feature {name!r}") from exc


def build_rgb_replay_frame(
    cells: np.ndarray,
    *,
    feature_names: list[str],
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Render a cell tensor into an RGB frame by filling each cell with rgb_mean."""
    out_h, out_w = output_shape
    cell_rows, cell_cols = cells.shape[:2]
    if out_h % cell_rows != 0 or out_w % cell_cols != 0:
        raise ValueError("output_shape must divide evenly by cell grid")
    row_scale = out_h // cell_rows
    col_scale = out_w // cell_cols
    rgb = np.dstack(
        [
            cells[:, :, _feature_index(feature_names, "rgb_mean_r")],
            cells[:, :, _feature_index(feature_names, "rgb_mean_g")],
            cells[:, :, _feature_index(feature_names, "rgb_mean_b")],
        ]
    )
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return np.repeat(np.repeat(rgb, row_scale, axis=0), col_scale, axis=1)


def _frame_to_cell_rgb(frame: np.ndarray, cell_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = cell_shape
    height, width = frame.shape[:2]
    cell_h = height // rows
    cell_w = width // cols
    crop = frame[: rows * cell_h, : cols * cell_w]
    return crop.reshape(rows, cell_h, cols, cell_w, 3).transpose(0, 2, 1, 3, 4).mean(axis=(2, 3))


def cell_rgb_accuracy(
    frame: np.ndarray,
    cells: np.ndarray,
    *,
    feature_names: list[str],
) -> dict[str, float | int]:
    expected = np.dstack(
        [
            cells[:, :, _feature_index(feature_names, "rgb_mean_r")],
            cells[:, :, _feature_index(feature_names, "rgb_mean_g")],
            cells[:, :, _feature_index(feature_names, "rgb_mean_b")],
        ]
    )
    observed = _frame_to_cell_rgb(frame, expected.shape[:2])
    err = np.abs(observed.astype(np.float32) - expected.astype(np.float32))
    return {
        "cell_count": int(expected.shape[0] * expected.shape[1]),
        "mean_abs_error": float(err.mean()),
        "max_abs_error": float(err.max()),
        "rmse": float(math.sqrt(float(np.mean(err ** 2)))),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_native_cell_chunk(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read Rust native TVCELL01 chunks: header, frame numbers, f32 cell tensor."""
    data = path.read_bytes()
    if len(data) < 24:
        raise ValueError(f"native cell chunk is too small: {path}")
    if data[:8] != b"TVCELL01":
        raise ValueError(f"unknown native cell chunk magic: {path}")
    header = np.frombuffer(data, dtype="<u4", count=4, offset=8)
    frame_count, grid_rows, grid_cols, feature_count = [int(value) for value in header]
    numbers_offset = 24
    numbers_bytes = frame_count * 4
    frame_numbers = np.frombuffer(data, dtype="<u4", count=frame_count, offset=numbers_offset).astype(np.int32)
    cells_offset = numbers_offset + numbers_bytes
    expected_values = frame_count * grid_rows * grid_cols * feature_count
    cells = np.frombuffer(data, dtype="<f4", count=expected_values, offset=cells_offset)
    if cells.size != expected_values:
        raise ValueError(f"native cell chunk ended early: {path}")
    cells = cells.reshape(frame_count, grid_rows, grid_cols, feature_count).astype(np.float32, copy=False)
    return cells, frame_numbers


def read_cell_chunk(chunk: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(chunk["path"])
    chunk_format = str(chunk.get("format") or "")
    if chunk_format == "tvcells_f32le_v1" or path.suffix.lower() == ".tvcells":
        return read_native_cell_chunk(path)
    with np.load(path, allow_pickle=False) as data:
        return data["cell_state"], data["frame_numbers"]


def _write_video(path: Path, frames: list[np.ndarray], *, fps: float, fourcc_text: str) -> bool:
    if not frames:
        return False
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_text), fps, (width, height))
    if not writer.isOpened():
        return False
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path.exists() and path.stat().st_size > 0


def replay_capture(run_dir: Path, *, output_dir: Path | None = None, fps: float | None = None) -> dict[str, Any]:
    manifest_path = next(run_dir.glob("*_manifest.json"))
    summary_path = next(run_dir.glob("*_summary.json"))
    records_path = next(run_dir.glob("*_records.jsonl"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = read_jsonl(records_path)
    output_dir = output_dir or (run_dir / "replay")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = list(manifest["cell_state"]["feature_names"])
    frame_shape = tuple(summary["geometry"]["frame_shape"])
    duration = float(summary.get("duration_seconds") or 0.0)
    frame_count = int(summary.get("frame_count") or len(records))
    replay_fps = float(fps or (frame_count / duration if duration > 0 else 9.0))

    frames: list[np.ndarray] = []
    accuracy_samples: list[dict[str, Any]] = []
    sample_targets = {0, max(0, frame_count // 2), max(0, frame_count - 1)}

    # Add peak-energy record index when available.
    if records:
        sample_targets.add(max(range(len(records)), key=lambda i: records[i].get("screen_energy") or 0.0))

    record_by_frame = {int(record["frame_number"]): record for record in records}
    frame_index = 0
    for chunk in manifest["cell_state"]["chunks"]:
        cell_state, frame_numbers = read_cell_chunk(chunk)
        for local_index, frame_number in enumerate(frame_numbers):
            cells = cell_state[local_index]
            frame = build_rgb_replay_frame(cells, feature_names=feature_names, output_shape=frame_shape)
            frames.append(frame)
            if frame_index in sample_targets:
                metrics = cell_rgb_accuracy(frame, cells, feature_names=feature_names)
                record = record_by_frame.get(int(frame_number), {})
                accuracy_samples.append(
                    {
                        "frame_index": frame_index,
                        "frame_number": int(frame_number),
                        "elapsed_seconds": record.get("elapsed_seconds"),
                        "screen_energy": record.get("screen_energy"),
                        **metrics,
                    }
                )
            frame_index += 1

    lossless_path = output_dir / f"{manifest['run_id']}_cell_rgb_replay_lossless_ffv1.mkv"
    preview_path = output_dir / f"{manifest['run_id']}_cell_rgb_replay_preview_mp4v.mp4"
    lossless_ok = _write_video(lossless_path, frames, fps=replay_fps, fourcc_text="FFV1")
    preview_ok = _write_video(preview_path, frames, fps=replay_fps, fourcc_text="mp4v")

    report = {
        "schema_version": 1,
        "kind": "compucogvision_state_replay_report",
        "created_at_utc": utc_now(),
        "run_id": manifest["run_id"],
        "source_manifest": str(manifest_path),
        "source_records": str(records_path),
        "source_cell_chunks": len(manifest["cell_state"]["chunks"]),
        "frame_count": len(frames),
        "frame_shape": list(frame_shape),
        "cell_grid_shape": summary["geometry"]["grid_shape"],
        "feature_names": feature_names,
        "replay_fps": replay_fps,
        "boundary": {
            "replay_source": "stored 16:9 cell-state vectors",
            "not_original_video": True,
            "raw_original_frames_available": False,
            "accuracy_scope": "cell RGB replay fidelity against stored rgb_mean vectors",
        },
        "pre_encode_accuracy_samples": accuracy_samples,
        "outputs": {
            "lossless_ffv1_mkv": {
                "path": str(lossless_path),
                "written": lossless_ok,
                "sha256": sha256_file(lossless_path) if lossless_ok else None,
            },
            "preview_mp4v": {
                "path": str(preview_path),
                "written": preview_ok,
                "sha256": sha256_file(preview_path) if preview_ok else None,
            },
        },
    }
    report_path = output_dir / f"{manifest['run_id']}_replay_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return {"report": str(report_path), **report["outputs"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a CompuCogVision cell-state capture as video.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--fps", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = replay_capture(
        Path(args.run_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        fps=args.fps or None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
