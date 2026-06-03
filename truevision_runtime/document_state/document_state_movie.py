from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DOCUMENT_STATE_MOVIE_SCHEMA_VERSION = "truevision_document_state_movie@1"

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


def record_document_state_movie(
    *,
    source_id: str,
    page_frames: list[np.ndarray],
    output_root: Path,
    run_id: str,
    frames_per_page: int = 3,
    fps: float = 3.0,
    grid_shape: tuple[int, int] = (90, 160),
) -> dict[str, Any]:
    """Witness ordered document pages as TrueVision cell-state frames."""

    if frames_per_page < 1:
        raise ValueError("frames_per_page must be at least 1")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not page_frames:
        raise ValueError("page_frames must not be empty")

    output_root = Path(output_root)
    run_dir = output_root / str(run_id)
    cell_dir = run_dir / "cell_state_npz"
    cell_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    cell_frames: list[np.ndarray] = []
    frame_numbers: list[int] = []
    frame_pages: list[dict[str, int]] = []
    previous_luma: np.ndarray | None = None
    frame_index = 0
    source_shape = _rgb_frame(page_frames[0]).shape[:2]

    for page_index, page in enumerate(page_frames):
        page_start = frame_index
        for repeat_index in range(frames_per_page):
            cells, previous_luma = build_page_cell_state(
                page,
                grid_shape=grid_shape,
                previous_luma=previous_luma,
            )
            cell_frames.append(cells)
            frame_numbers.append(frame_index)
            elapsed_seconds = frame_index / fps
            records.append(
                {
                    "schema_version": DOCUMENT_STATE_MOVIE_SCHEMA_VERSION,
                    "record_kind": "truevision_document_state_frame",
                    "source_id": str(source_id),
                    "run_id": str(run_id),
                    "observed_at_utc": _utc_now(),
                    "frame_index": frame_index,
                    "frame_number": frame_index,
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "page_repeat_index": repeat_index,
                    "elapsed_seconds": round(elapsed_seconds, 6),
                    "fps": float(fps),
                    "screen_energy": float(cells[:, :, _feature_index("delta_luma_abs")].sum()),
                    "raw_frame_saved": False,
                    "raw_grid_saved": False,
                    "cell_state_ref": {
                        "format": "npz_compressed_float32",
                        "chunk_id": 0,
                        "chunk_frame_index": frame_index,
                        "grid_shape": list(grid_shape),
                        "feature_names": list(CELL_FEATURE_NAMES),
                        "feature_count": len(CELL_FEATURE_NAMES),
                    },
                }
            )
            frame_index += 1
        frame_pages.append(
            {
                "frame_start": page_start,
                "frame_end": frame_index - 1,
                "page_index": page_index,
                "page_number": page_index + 1,
            }
        )

    chunk_path = cell_dir / f"{run_id}_cells_0000.npz"
    np.savez_compressed(
        chunk_path,
        cell_state=np.stack(cell_frames).astype(np.float32),
        frame_numbers=np.asarray(frame_numbers, dtype=np.int32),
        feature_names=np.asarray(CELL_FEATURE_NAMES),
    )

    records_path = run_dir / f"{run_id}_records.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in records) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": DOCUMENT_STATE_MOVIE_SCHEMA_VERSION,
        "record_kind": "truevision_document_state_summary",
        "source_id": str(source_id),
        "run_id": str(run_id),
        "frame_count": len(records),
        "page_count": len(page_frames),
        "duration_seconds": round((len(records) - 1) / fps if records else 0.0, 6),
        "geometry": {
            "frame_shape": [int(source_shape[0]), int(source_shape[1])],
            "grid_shape": list(grid_shape),
        },
        "raw_frame_saved": False,
        "raw_grid_saved": False,
    }
    summary_path = run_dir / f"{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    manifest = {
        "schema_version": DOCUMENT_STATE_MOVIE_SCHEMA_VERSION,
        "record_kind": "truevision_document_state_movie",
        "source_id": str(source_id),
        "run_id": str(run_id),
        "created_at_utc": _utc_now(),
        "records": {"jsonl_path": str(records_path), "frame_count": len(records)},
        "summary": summary,
        "summary_json": str(summary_path),
        "config": {
            "source_kind": "document_pages_as_state_movie",
            "frames_per_page": int(frames_per_page),
            "fps": float(fps),
            "capture_resolution": [int(source_shape[1]), int(source_shape[0])],
            "frame_shape_rows_cols": [int(source_shape[1]), int(source_shape[0])],
            "grid_shape_rows_cols": list(grid_shape),
            "page_count": len(page_frames),
            "cell_feature_names": list(CELL_FEATURE_NAMES),
        },
        "frame_pages": frame_pages,
        "cell_state": {
            "enabled": True,
            "format": "npz_compressed_float32",
            "feature_names": list(CELL_FEATURE_NAMES),
            "chunks": [
                {
                    "chunk_id": 0,
                    "path": str(chunk_path),
                    "format": "npz_compressed_float32",
                    "shape": [len(records), grid_shape[0], grid_shape[1], len(CELL_FEATURE_NAMES)],
                    "frames": len(records),
                    "grid_shape": list(grid_shape),
                    "feature_count": len(CELL_FEATURE_NAMES),
                    "sha256": _sha256_file(chunk_path),
                }
            ],
        },
        "boundary": {
            "raw_frame_saved": False,
            "raw_grid_saved": False,
            "media_is_optional_surface": True,
            "generated_media_is_evidence": False,
            "recognition_must_read_state": True,
            "anchorworks_runtime_dependency": False,
            "notes": "Document pages are witnessed as cell-state frames. Replay surfaces are derived display, not source truth.",
        },
    }
    manifest_path = run_dir / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "records_jsonl": str(records_path),
        "summary_json": str(summary_path),
        "manifest_json": str(manifest_path),
        "cell_state_npz": str(chunk_path),
        "frame_count": len(records),
        "page_count": len(page_frames),
    }


def build_page_cell_state(
    frame: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    previous_luma: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame_rgb = _fit_to_grid(_rgb_frame(frame), grid_shape)
    rgb = frame_rgb.astype(np.float32)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    luma = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float32)
    saturation = (rgb.max(axis=2) - rgb.min(axis=2)).astype(np.float32)
    delta = np.zeros_like(luma) if previous_luma is None or previous_luma.shape != luma.shape else np.abs(luma - previous_luma)
    edge = _edge_density(luma)
    texture = np.clip(np.abs(cv2.Laplacian(luma, cv2.CV_32F)) / 255.0, 0.0, 1.0).astype(np.float32)
    zeros_rgb = np.zeros_like(rgb, dtype=np.float32)
    zeros_luma = np.zeros_like(luma, dtype=np.float32)

    cells = np.dstack(
        [
            rgb[:, :, 0],
            rgb[:, :, 1],
            rgb[:, :, 2],
            zeros_rgb[:, :, 0],
            zeros_rgb[:, :, 1],
            zeros_rgb[:, :, 2],
            hsv[:, :, 0],
            hsv[:, :, 1],
            hsv[:, :, 2],
            luma,
            zeros_luma,
            saturation,
            delta,
            edge,
            texture,
            delta,
        ]
    ).astype(np.float32)
    return cells, luma


def replay_document_state_movie_frame(
    manifest_path: str | Path,
    frame_index: int = 0,
    *,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Replay one document-state movie frame as a derived RGB surface."""

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    feature_names = list((manifest.get("cell_state") or {}).get("feature_names") or CELL_FEATURE_NAMES)
    cells = _load_cells_for_frame(manifest_file, manifest, frame_index)
    rgb = np.dstack(
        [
            cells[:, :, feature_names.index("rgb_mean_r")],
            cells[:, :, feature_names.index("rgb_mean_g")],
            cells[:, :, feature_names.index("rgb_mean_b")],
        ]
    )
    rgb_u8 = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    shape = output_shape or _manifest_output_shape(manifest, rgb_u8.shape[:2])
    if rgb_u8.shape[:2] != shape:
        rgb_u8 = cv2.resize(rgb_u8, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return rgb_u8


def write_document_state_surface(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    frame_index: int = 0,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a PNG surface from state and return a derived-display receipt."""

    manifest_file = Path(manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = replay_document_state_movie_frame(manifest_file, frame_index)
    ok = cv2.imwrite(str(output), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"failed to write document state surface: {output}")

    receipt = {
        "schema_version": "truevision_document_state_surface@1",
        "record_kind": "truevision_document_state_surface",
        "created_at_utc": _utc_now(),
        "source_manifest": str(manifest_file),
        "frame_index": int(frame_index),
        "output_path": str(output),
        "output_sha256": _sha256_file(output),
        "boundary": {
            "source_truth_is_state": True,
            "surface_is_derived_display": True,
            "raw_page_saved": False,
            "generated_media_is_evidence": False,
            "anchorworks_runtime_dependency": False,
        },
    }
    if receipt_path is not None:
        receipt_file = Path(receipt_path)
        receipt_file.parent.mkdir(parents=True, exist_ok=True)
        receipt_file.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
        receipt["receipt_path"] = str(receipt_file)
    return receipt


def extract_black_glyph_patterns_from_state_movie(
    *,
    manifest_path: str | Path,
    frame_index: int = 0,
    luma_threshold: float = 128.0,
) -> list[dict[str, Any]]:
    """Extract connected black-cell glyph patterns from stored state only."""

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    feature_names = list((manifest.get("cell_state") or {}).get("feature_names") or CELL_FEATURE_NAMES)
    cells = _load_cells_for_frame(manifest_file, manifest, frame_index)
    luma = cells[:, :, feature_names.index("luma_mean")]
    mask = luma < float(luma_threshold)
    components = _connected_components(mask)
    rows: list[dict[str, Any]] = []
    for order, component in enumerate(components):
        ys = [cell[0] for cell in component]
        xs = [cell[1] for cell in component]
        top, bottom = min(ys), max(ys)
        left, right = min(xs), max(xs)
        points = set(component)
        pattern = [
            "".join("1" if (y, x) in points else "0" for x in range(left, right + 1))
            for y in range(top, bottom + 1)
        ]
        rows.append(
            {
                "order": order,
                "pattern": pattern,
                "bbox": {"x": left, "y": top, "w": right - left + 1, "h": bottom - top + 1},
                "source": "stored_cell_state_luma",
                "raw_frames_saved": False,
            }
        )
    return rows


def _load_cells_for_frame(manifest_file: Path, manifest: dict[str, Any], frame_index: int) -> np.ndarray:
    chunks = list((manifest.get("cell_state") or {}).get("chunks") or [])
    if not chunks:
        raise ValueError("document state movie manifest has no cell chunks")
    target = max(0, int(frame_index))
    base = 0
    for chunk in chunks:
        frames = int(chunk.get("frames") or 0)
        if target < base + frames:
            chunk_path = _resolve_path(str(chunk["path"]), manifest_file.parent)
            with np.load(chunk_path, allow_pickle=False) as data:
                state = np.asarray(data["cell_state"], dtype=np.float32)
            return state[target - base]
        base += frames
    raise IndexError(frame_index)


def _manifest_output_shape(manifest: dict[str, Any], fallback: tuple[int, int]) -> tuple[int, int]:
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    geometry = summary.get("geometry") if isinstance(summary.get("geometry"), dict) else {}
    frame_shape = geometry.get("frame_shape") if isinstance(geometry, dict) else None
    if isinstance(frame_shape, list) and len(frame_shape) == 2:
        return int(frame_shape[0]), int(frame_shape[1])
    return int(fallback[0]), int(fallback[1])


def _rgb_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.dstack([arr, arr, arr])
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("page frame must be grayscale or RGB-like")
    return np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)


def _fit_to_grid(frame: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("grid_shape must contain positive rows and columns")
    if frame.shape[0] == rows and frame.shape[1] == cols:
        return frame[:, :, :3]
    return cv2.resize(frame[:, :, :3], (cols, rows), interpolation=cv2.INTER_AREA)


def _edge_density(luma: np.ndarray) -> np.ndarray:
    vertical = np.zeros_like(luma)
    horizontal = np.zeros_like(luma)
    vertical[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
    horizontal[1:, :] = np.abs(luma[1:, :] - luma[:-1, :])
    return np.clip((vertical + horizontal) / 255.0, 0.0, 1.0).astype(np.float32)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    rows, cols = mask.shape
    for y in range(rows):
        for x in range(cols):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny in range(cy - 1, cy + 2):
                    for nx in range(cx - 1, cx + 2):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            components.append(component)
    components.sort(key=lambda cells: (min(y for y, _x in cells), min(x for _y, x in cells)))
    return components


def _resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _feature_index(name: str) -> int:
    return CELL_FEATURE_NAMES.index(name)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
