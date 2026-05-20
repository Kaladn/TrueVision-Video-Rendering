"""Extract reusable lightning/flash signatures from TrueVision capture state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .temporal_causality_projector import load_capture_cells, slug


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _feature_index(feature_names: list[str], name: str) -> int | None:
    try:
        return feature_names.index(name)
    except ValueError:
        return None


def _norm(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin <= 1.0e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def score_lightning_frames(cells: np.ndarray, *, feature_names: list[str]) -> np.ndarray:
    """Score frames for fast bright/edged intensity spikes."""
    luma_idx = _feature_index(feature_names, "luma_mean")
    edge_idx = _feature_index(feature_names, "edge_density")
    delta_idx = _feature_index(feature_names, "delta_luma_abs")
    texture_idx = _feature_index(feature_names, "texture_energy")
    if luma_idx is None:
        raise ValueError("luma_mean is required to score lightning frames")

    luma = cells[:, :, :, luma_idx].astype(np.float32)
    frame_luma = luma.mean(axis=(1, 2))
    luma_jump = np.zeros(cells.shape[0], dtype=np.float32)
    luma_jump[1:] = np.abs(np.diff(frame_luma))
    local_luma_std = luma.std(axis=(1, 2))
    score = _norm(luma_jump) * 0.42 + _norm(local_luma_std) * 0.18
    if edge_idx is not None:
        score += _norm(cells[:, :, :, edge_idx].mean(axis=(1, 2))) * 0.20
    if delta_idx is not None:
        score += _norm(cells[:, :, :, delta_idx].mean(axis=(1, 2))) * 0.34
    if texture_idx is not None:
        score += _norm(cells[:, :, :, texture_idx].mean(axis=(1, 2))) * 0.12
    return score.astype(np.float32)


def extract_lightning_signature_from_cells(
    cells: np.ndarray,
    frame_numbers: np.ndarray,
    *,
    feature_names: list[str],
    radius: int = 6,
    max_cells: int = 420,
) -> dict[str, Any]:
    """Extract hot cells from the peak 6-1-6 intensity neighborhood."""
    if cells.shape[0] < max(3, radius * 2 + 1):
        raise ValueError("not enough frames to extract a 6-1-6 lightning signature")
    scores = score_lightning_frames(cells, feature_names=feature_names)
    peak_index = int(np.argmax(scores))
    start = max(0, peak_index - radius)
    end = min(cells.shape[0], peak_index + radius + 1)
    window = cells[start:end].astype(np.float32)

    luma_idx = _feature_index(feature_names, "luma_mean")
    edge_idx = _feature_index(feature_names, "edge_density")
    delta_idx = _feature_index(feature_names, "delta_luma_abs")
    motion_idx = _feature_index(feature_names, "motion_energy")
    if luma_idx is None:
        raise ValueError("luma_mean is required")
    peak = cells[peak_index].astype(np.float32)
    neighborhood_mean = np.mean(window, axis=0)
    luma_hot = _norm(peak[:, :, luma_idx] - neighborhood_mean[:, :, luma_idx])
    energy = luma_hot * 0.48
    if edge_idx is not None:
        energy += _norm(peak[:, :, edge_idx]) * 0.28
    if delta_idx is not None:
        energy += _norm(peak[:, :, delta_idx]) * 0.34
    if motion_idx is not None:
        energy += _norm(peak[:, :, motion_idx]) * 0.10
    energy = np.clip(energy, 0.0, 1.0)

    rows, cols = energy.shape
    flat = energy.ravel()
    count = min(max_cells, flat.size)
    top_indices = np.argpartition(flat, -count)[-count:]
    top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
    hot_cells: list[dict[str, Any]] = []
    for flat_index in top_indices:
        value = float(flat[flat_index])
        if value <= 0.05:
            continue
        row = int(flat_index // cols)
        col = int(flat_index % cols)
        hot_cells.append(
            {
                "x_norm": round((col + 0.5) / cols, 6),
                "y_norm": round((row + 0.5) / rows, 6),
                "intensity": round(value, 6),
            }
        )

    if not hot_cells:
        raise ValueError("no hot cells found for lightning signature")
    xs = [cell["x_norm"] for cell in hot_cells]
    ys = [cell["y_norm"] for cell in hot_cells]
    signature = {
        "schema_version": 1,
        "kind": "truevision_lightning_signature_v1",
        "created_at_utc": utc_now(),
        "method": "6_1_6_peak_intensity_cell_extraction",
        "peak": {
            "frame_index": peak_index,
            "frame_number": int(frame_numbers[peak_index]),
            "score": round(float(scores[peak_index]), 6),
            "window_frame_numbers": [int(value) for value in frame_numbers[start:end].tolist()],
            "radius": radius,
        },
        "grid_shape": [int(rows), int(cols)],
        "hot_cell_count": len(hot_cells),
        "bbox_norm": {
            "x_min": round(float(min(xs)), 6),
            "x_max": round(float(max(xs)), 6),
            "y_min": round(float(min(ys)), 6),
            "y_max": round(float(max(ys)), 6),
        },
        "hot_cells": hot_cells,
        "boundary": {
            "source_is_truevision_state": True,
            "raw_pixels_required": False,
            "generated_use_is_synthetic": True,
            "not_evidence": True,
        },
    }
    return signature


def extract_lightning_signature_from_capture(
    *,
    capture_run_dir: Path,
    output_dir: Path,
    signature_id: str,
    radius: int = 6,
    max_cells: int = 420,
    max_source_frames: int | None = None,
) -> dict[str, Any]:
    cells, frame_numbers, feature_names, manifest, _summary = load_capture_cells(capture_run_dir, max_frames=max_source_frames)
    signature = extract_lightning_signature_from_cells(
        cells,
        frame_numbers,
        feature_names=feature_names,
        radius=radius,
        max_cells=max_cells,
    )
    signature["signature_id"] = slug(signature_id)
    signature["source"] = {
        "capture_run_dir": str(capture_run_dir),
        "source_run_id": manifest.get("run_id"),
        "source_record_kind": manifest.get("record_kind"),
        "source_frame_count_loaded": int(cells.shape[0]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{slug(signature_id)}.json"
    path.write_text(json.dumps(signature, indent=2, allow_nan=False), encoding="utf-8")
    return {"signature_json": str(path), "signature": signature}
