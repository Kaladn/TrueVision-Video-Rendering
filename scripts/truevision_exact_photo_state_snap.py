#!/usr/bin/env python3
"""Lossless decoded-photo state snap plus derived TrueVision cell telemetry."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from truevision_resonance_recorder import CELL_FEATURE_NAMES, build_video_cell_state, sha256_file, write_cell_state_chunk


DEFAULT_OUTPUT_ROOT = Path("outputs/photo_state_snaps")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def slug(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return clean.strip("_")[:96] or "truevision_exact_photo_state_snap"


def load_decoded_rgb(image_path: Path) -> np.ndarray:
    raw = image_path.read_bytes()
    encoded = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _choose_grid_shape(image_shape: tuple[int, int], max_grid_shape: tuple[int, int]) -> tuple[int, int]:
    height, width = image_shape
    max_rows, max_cols = max_grid_shape
    rows = max(1, min(int(max_rows), int(height)))
    cols = max(1, min(int(max_cols), int(width)))
    return rows, cols


def _write_png_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"failed to write PNG: {path}")


def _read_png_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read PNG: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _pixel_error(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {
            "pixel_exact": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "max_abs_error": None,
            "mean_abs_error": None,
        }
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "pixel_exact": bool(np.array_equal(a, b)),
        "shape": list(a.shape),
        "max_abs_error": int(diff.max()) if diff.size else 0,
        "mean_abs_error": round(float(diff.mean()) if diff.size else 0.0, 9),
    }


def write_exact_photo_state_snap(
    *,
    image_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    max_grid_shape: tuple[int, int] = (360, 640),
) -> dict[str, Any]:
    image_path = image_path.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    run_id = slug(run_id or f"{image_path.stem}_exact_photo_state_snap")
    run_dir = output_root / run_id
    source_dir = run_dir / "source"
    state_dir = run_dir / "state"
    replay_dir = run_dir / "replay"
    cell_dir = run_dir / "cell_state_npz"
    for directory in (source_dir, state_dir, replay_dir, cell_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_copy = source_dir / f"source_original{image_path.suffix.lower() or '.image'}"
    shutil.copy2(image_path, source_copy)
    source_sha256 = sha256_file(image_path)
    copied_sha256 = sha256_file(source_copy)
    if source_sha256 != copied_sha256:
        raise RuntimeError("source byte copy hash mismatch")

    rgb = load_decoded_rgb(image_path)
    pixel_state_path = state_dir / f"{run_id}_pixel_state_rgb_u8.npz"
    np.savez_compressed(
        pixel_state_path,
        rgb=rgb,
        source_sha256=np.asarray(source_sha256),
        decode_mode=np.asarray("opencv_imdecode_bgr_to_rgb"),
        color_space=np.asarray("RGB"),
    )

    decoded_png = replay_dir / f"{run_id}_decoded_reference.png"
    reconstructed_png = replay_dir / f"{run_id}_reconstructed_from_pixel_state.png"
    _write_png_rgb(decoded_png, rgb)
    with np.load(pixel_state_path, allow_pickle=False) as data:
        reconstructed_rgb = np.asarray(data["rgb"], dtype=np.uint8)
    _write_png_rgb(reconstructed_png, reconstructed_rgb)
    replay_rgb = _read_png_rgb(reconstructed_png)
    exact_reconstruction = _pixel_error(rgb, replay_rgb)

    grid_shape = _choose_grid_shape(rgb.shape[:2], max_grid_shape)
    cell_state = build_video_cell_state(rgb, grid_shape=grid_shape)
    cell_chunk_path = cell_dir / f"{run_id}_cells_0000.npz"
    chunk = write_cell_state_chunk(
        chunk_path=cell_chunk_path,
        chunk_id=0,
        cell_frames=[cell_state["cells"]],
        frame_numbers=[1],
        grid_shape=grid_shape,
    )

    manifest_path = run_dir / f"{run_id}_manifest.json"
    manifest = {
        "schema": "truevision_exact_photo_state_snap.v1",
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "claim": "source_photo_decoded_to_lossless_pixel_state_and_derived_truevision_cells",
        "boundary": {
            "source_photo_is_input_artifact": True,
            "pixel_state_is_exact_decoded_photo_state": True,
            "cell_state_is_derived_telemetry": True,
            "generated_replay_is_not_primary_evidence": True,
            "no_prompt_generation": True,
            "no_external_visual_assets": True,
        },
        "source": {
            "path": str(image_path),
            "source_copy": str(source_copy),
            "source_sha256": source_sha256,
            "source_copy_sha256": copied_sha256,
            "bytes_exact_copy": True,
        },
        "pixel_state": {
            "format": "rgb_u8_lossless_decoded_pixels",
            "path": str(pixel_state_path),
            "sha256": sha256_file(pixel_state_path),
            "shape": [int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])],
            "dtype": str(rgb.dtype),
            "color_space": "RGB",
            "decode_mode": "opencv_imdecode_bgr_to_rgb",
        },
        "exact_reconstruction": {
            **exact_reconstruction,
            "decoded_reference_png": str(decoded_png),
            "decoded_reference_sha256": sha256_file(decoded_png),
            "reconstructed_png": str(reconstructed_png),
            "reconstructed_sha256": sha256_file(reconstructed_png),
        },
        "cell_state": {
            "role": "derived_truevision_telemetry_not_exact_photo",
            "feature_names": list(CELL_FEATURE_NAMES),
            "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
            "chunk": chunk,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "manifest_json": str(manifest_path),
        "source_copy": str(source_copy),
        "pixel_state_npz": str(pixel_state_path),
        "decoded_reference_png": str(decoded_png),
        "reconstructed_png": str(reconstructed_png),
        "cell_state_npz": str(cell_chunk_path),
        "pixel_exact": exact_reconstruction["pixel_exact"],
        "max_abs_error": exact_reconstruction["max_abs_error"],
        "source_sha256": source_sha256,
    }


def parse_grid(value: str) -> tuple[int, int]:
    text = value.lower().replace("x", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("grid must be ROWSxCOLS")
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an exact decoded-photo TrueVision state snap.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-grid", type=parse_grid, default=(360, 640), help="Derived telemetry max grid as ROWSxCOLS.")
    args = parser.parse_args()
    result = write_exact_photo_state_snap(
        image_path=Path(args.image),
        output_root=Path(args.output_root),
        run_id=args.run_id,
        max_grid_shape=args.max_grid,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
