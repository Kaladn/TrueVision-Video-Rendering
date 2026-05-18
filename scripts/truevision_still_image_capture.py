#!/usr/bin/env python3
"""Convert a still image into TrueVision-shaped video-state records.

This treats one photograph as a static video field. It writes the same core
bundle shape as the screen recorder: JSONL records, manifest, summary, and
compressed 16-channel cell-state chunks. It does not mutate the source image
or treat generated replay artifacts as evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

POC_ROOT = Path(__file__).resolve().parents[1]
for path in (POC_ROOT, POC_ROOT / "scripts", POC_ROOT / "modules"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from screen_resonance_state import ScreenResonanceState
from truevision_resonance_recorder import (
    CELL_FEATURE_NAMES,
    build_record,
    build_video_cell_state,
    clean_value,
    parse_shape_xy,
    sha256_file,
    write_capture_bundle,
    write_cell_state_chunk,
)


DEFAULT_OUTPUT_ROOT = POC_ROOT / "connected_artifacts"


def utc_from(base: datetime, elapsed_seconds: float) -> str:
    return (
        base + timedelta(seconds=elapsed_seconds)
    ).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image as RGB, applying EXIF orientation when Pillow is present."""
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except Exception:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fit_image_to_frame(
    image: np.ndarray,
    *,
    frame_shape: tuple[int, int],
    fill_rgb: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Letterbox an image into a target video frame without aspect distortion."""
    target_h, target_w = frame_shape
    src_h, src_w = image.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    frame = np.full((target_h, target_w, 3), fill_rgb, dtype=np.uint8)
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    frame[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
    return frame, {
        "fit_mode": "letterbox_preserve_aspect",
        "source_shape": [int(src_h), int(src_w)],
        "frame_shape": [int(target_h), int(target_w)],
        "scale": float(scale),
        "resized_shape": [int(resized_h), int(resized_w)],
        "offset_xy": [int(offset_x), int(offset_y)],
        "padding": {
            "left": int(offset_x),
            "top": int(offset_y),
            "right": int(target_w - resized_w - offset_x),
            "bottom": int(target_h - resized_h - offset_y),
        },
    }


def frame_to_grid(frame: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = grid_shape
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32)


def compress_to_blocks(
    grid: np.ndarray,
    *,
    block_shape: tuple[int, int],
) -> np.ndarray:
    grid_rows, grid_cols = grid.shape
    block_rows, block_cols = block_shape
    if grid_rows % block_rows != 0 or grid_cols % block_cols != 0:
        raise ValueError("grid_shape must divide evenly into block_shape")
    cells_per_block_row = grid_rows // block_rows
    cells_per_block_col = grid_cols // block_cols
    view = grid.reshape(
        block_rows,
        cells_per_block_row,
        block_cols,
        cells_per_block_col,
    ).transpose(0, 2, 1, 3)
    return view.mean(axis=(2, 3)).astype(np.float32)


def build_still_features(
    *,
    frame: np.ndarray,
    source_shape: tuple[int, int],
    grid_shape: tuple[int, int],
    block_shape: tuple[int, int],
    frame_number: int,
    fps: float,
    elapsed_seconds: float,
    base_time: datetime,
    resonance: ScreenResonanceState,
    previous_grid: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray]:
    grid = frame_to_grid(frame, grid_shape)
    delta = np.zeros_like(grid) if previous_grid is None else np.abs(grid - previous_grid)
    blocks = compress_to_blocks(grid, block_shape=block_shape)
    block_deltas = compress_to_blocks(delta, block_shape=block_shape)
    block_vector = block_deltas.reshape(-1)
    blocks_normalized = (blocks - blocks.min()) / (blocks.max() - blocks.min() + 1e-8)
    visual_resonance = resonance.update(blocks_normalized)

    features = {
        "frame": frame,
        "grid": grid,
        "delta": delta,
        "blocks": blocks,
        "block_deltas": block_deltas,
        "block_vector": block_vector,
        "visual_resonance": visual_resonance,
        "capture_geometry": {
            "source_width": int(source_shape[1]),
            "source_height": int(source_shape[0]),
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
            "grid_rows": int(grid_shape[0]),
            "grid_cols": int(grid_shape[1]),
            "block_rows": int(block_shape[0]),
            "block_cols": int(block_shape[1]),
            "capture_region": None,
        },
        "timestamp": elapsed_seconds,
        "wall_time_unix": (base_time + timedelta(seconds=elapsed_seconds)).timestamp(),
        "observed_at_utc": utc_from(base_time, elapsed_seconds),
        "fps": 0.0 if frame_number == 1 else fps,
        "frame_number": frame_number,
    }
    return features, grid


def write_still_report(
    *,
    run_dir: Path,
    run_id: str,
    image_path: Path,
    source_sha256: str,
    frame_shape: tuple[int, int],
    grid_shape: tuple[int, int],
    block_shape: tuple[int, int],
    frames: int,
    fps: float,
    fit_metadata: dict[str, Any],
    manifest_path: Path,
    records_path: Path,
    summary_path: Path,
) -> Path:
    report_path = run_dir / f"{run_id}_photo_truevision_report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# TrueVision Still Image Capture Report",
                "",
                "## Claim",
                "",
                "A source photograph was converted into TrueVision-shaped video-state data.",
                "The time axis is a static hold over the same frame, so motion channels are expected to be zero or near-zero.",
                "",
                "## Boundary",
                "",
                "```text",
                "Source photo is an input artifact.",
                "Cell-state records are derived telemetry.",
                "Replay media is a derived visualization.",
                "No generated replay is primary evidence.",
                "```",
                "",
                "## Source",
                "",
                f"- Image: `{image_path}`",
                f"- SHA256: `{source_sha256}`",
                f"- Source shape: `{fit_metadata['source_shape'][1]}x{fit_metadata['source_shape'][0]}`",
                "",
                "## Output Shape",
                "",
                f"- Frame shape: `{frame_shape[1]}x{frame_shape[0]}`",
                f"- Grid shape: `{grid_shape[1]}x{grid_shape[0]}`",
                f"- Block shape: `{block_shape[1]}x{block_shape[0]}`",
                f"- Frames: `{frames}`",
                f"- FPS: `{fps}`",
                f"- Cell features: `{len(CELL_FEATURE_NAMES)}`",
                "",
                "## Fit",
                "",
                f"- Mode: `{fit_metadata['fit_mode']}`",
                f"- Scale: `{fit_metadata['scale']}`",
                f"- Resized shape: `{fit_metadata['resized_shape'][1]}x{fit_metadata['resized_shape'][0]}`",
                f"- Offset XY: `{fit_metadata['offset_xy']}`",
                f"- Padding: `{fit_metadata['padding']}`",
                "",
                "## Files",
                "",
                f"- Manifest: `{manifest_path}`",
                f"- Records: `{records_path}`",
                f"- Summary: `{summary_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def capture_still_image(
    *,
    image_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    frame_shape: tuple[int, int] = (540, 960),
    grid_shape: tuple[int, int] = (90, 160),
    block_shape: tuple[int, int] = (9, 16),
    frames: int = 9,
    fps: float = 9.0,
) -> dict[str, Any]:
    if frames < 1:
        raise ValueError("frames must be at least 1")
    if fps <= 0:
        raise ValueError("fps must be positive")

    image_path = image_path.resolve()
    output_root = output_root.resolve()
    run_id = run_id or f"{image_path.stem}_truevision_still_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    cell_dir = run_dir / "cell_state_npz"
    source = load_image_rgb(image_path)
    frame, fit_metadata = fit_image_to_frame(source, frame_shape=frame_shape)
    source_sha256 = sha256_file(image_path)
    base_time = datetime.now(timezone.utc)
    resonance = ScreenResonanceState(grid_size=block_shape, ema_alpha_fast=0.3, ema_alpha_slow=0.1)
    records: list[dict[str, Any]] = []
    cell_frames: list[np.ndarray] = []
    frame_numbers: list[int] = []
    previous_grid: np.ndarray | None = None
    previous_cell_luma: np.ndarray | None = None

    chunk_path = cell_dir / f"{run_id}_cells_0000.npz"
    start_perf = time.perf_counter()
    for frame_index in range(frames):
        frame_number = frame_index + 1
        elapsed_seconds = frame_index / fps
        features, previous_grid = build_still_features(
            frame=frame,
            source_shape=source.shape[:2],
            grid_shape=grid_shape,
            block_shape=block_shape,
            frame_number=frame_number,
            fps=fps,
            elapsed_seconds=elapsed_seconds,
            base_time=base_time,
            resonance=resonance,
            previous_grid=previous_grid,
        )
        cell_state = build_video_cell_state(
            frame,
            grid_shape=grid_shape,
            previous_luma=previous_cell_luma,
        )
        previous_cell_luma = cell_state["luma"]
        features["cell_state_ref"] = {
            "format": "npz_compressed_float32",
            "path": str(chunk_path),
            "chunk_id": 0,
            "chunk_frame_index": frame_index,
            "frame_number": frame_number,
            "grid_shape": list(grid_shape),
            "cell_count": grid_shape[0] * grid_shape[1],
            "feature_names": list(CELL_FEATURE_NAMES),
            "feature_count": len(CELL_FEATURE_NAMES),
        }
        cell_frames.append(cell_state["cells"])
        frame_numbers.append(frame_number)
        records.append(
            build_record(
                features,
                run_id=run_id,
                elapsed_seconds=elapsed_seconds,
                include_blocks=True,
            )
        )

    chunk = write_cell_state_chunk(
        chunk_path=chunk_path,
        chunk_id=0,
        cell_frames=cell_frames,
        frame_numbers=frame_numbers,
        grid_shape=grid_shape,
    )
    config = {
        "source_kind": "still_image_as_video_state",
        "source_image_path": str(image_path),
        "source_image_sha256": source_sha256,
        "source_image_shape": [int(source.shape[0]), int(source.shape[1])],
        "duration_seconds": round((frames - 1) / fps if frames > 1 else 0.0, 6),
        "capture_fps": fps,
        "frame_count": frames,
        "grid_shape_rows_cols": list(grid_shape),
        "grid_size_xy": [grid_shape[1], grid_shape[0]],
        "block_shape_rows_cols": list(block_shape),
        "block_size_xy": [block_shape[1], block_shape[0]],
        "capture_resolution": [frame_shape[1], frame_shape[0]],
        "fit": fit_metadata,
        "include_blocks": True,
        "save_cell_state": True,
        "cell_feature_names": list(CELL_FEATURE_NAMES),
        "cell_chunk_frames": frames,
    }
    bundle = write_capture_bundle(
        output_root=output_root,
        run_id=run_id,
        records=records,
        config=config,
        cell_state_chunks=[chunk],
    )
    report_path = write_still_report(
        run_dir=bundle["run_dir"],
        run_id=run_id,
        image_path=image_path,
        source_sha256=source_sha256,
        frame_shape=frame_shape,
        grid_shape=grid_shape,
        block_shape=block_shape,
        frames=frames,
        fps=fps,
        fit_metadata=fit_metadata,
        manifest_path=bundle["manifest_json"],
        records_path=bundle["records_jsonl"],
        summary_path=bundle["summary_json"],
    )
    return clean_value(
        {
            "run_id": run_id,
            "frames": frames,
            "fps": fps,
            "runtime_seconds": round(time.perf_counter() - start_perf, 6),
            "run_dir": str(bundle["run_dir"]),
            "records_jsonl": str(bundle["records_jsonl"]),
            "summary_json": str(bundle["summary_json"]),
            "manifest_json": str(bundle["manifest_json"]),
            "report_md": str(report_path),
            "cell_state_npz": str(chunk_path),
            "cell_state_shape": [frames, grid_shape[0], grid_shape[1], len(CELL_FEATURE_NAMES)],
            "source_sha256": source_sha256,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a still image to TrueVision-shaped video-state records.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resolution", default="960x540")
    parser.add_argument("--grid", default="160x90")
    parser.add_argument("--blocks", default="16x9")
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--fps", type=float, default=9.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    width, height = [int(part) for part in args.resolution.lower().replace("x", ",").split(",")]
    result = capture_still_image(
        image_path=Path(args.image),
        output_root=Path(args.output_root),
        run_id=args.run_id or None,
        frame_shape=(height, width),
        grid_shape=parse_shape_xy(args.grid),
        block_shape=parse_shape_xy(args.blocks),
        frames=args.frames,
        fps=args.fps,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
