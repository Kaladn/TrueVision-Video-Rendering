"""Causal summaries from 6-1-6 TrueVision cell-state windows."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .temporal_616 import TemporalWindow616


CORE_CHANNELS = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "luma_mean",
    "edge_density",
    "motion_energy",
    "delta_luma_abs",
)


def feature_indices(feature_names: list[str], names: tuple[str, ...] = CORE_CHANNELS) -> dict[str, int]:
    indices: dict[str, int] = {}
    for name in names:
        if name in feature_names:
            indices[name] = feature_names.index(name)
    missing = [name for name in names if name not in indices]
    if missing:
        raise ValueError(f"missing required TrueFrameGen channels: {missing}")
    return indices


def build_causal_cell_map(
    cells_by_frame: Mapping[int, np.ndarray],
    window: TemporalWindow616,
    *,
    feature_names: list[str],
) -> dict[str, Any]:
    """Summarize direction/drift/confidence for the missing target frame."""
    indices = feature_indices(feature_names)
    prior_frames = list(window.prior_frames)
    future_frames = list(window.future_frames)
    all_frames = prior_frames + future_frames
    if not prior_frames or not future_frames:
        confidence = 0.0
    else:
        confidence = min(1.0, window.observed_count / float(window.radius * 2))

    first_frame = prior_frames[-1] if prior_frames else all_frames[0]
    last_frame = future_frames[0] if future_frames else all_frames[-1]
    first = cells_by_frame[first_frame]
    last = cells_by_frame[last_frame]

    rgb_delta = np.dstack(
        [
            last[:, :, indices["rgb_mean_r"]] - first[:, :, indices["rgb_mean_r"]],
            last[:, :, indices["rgb_mean_g"]] - first[:, :, indices["rgb_mean_g"]],
            last[:, :, indices["rgb_mean_b"]] - first[:, :, indices["rgb_mean_b"]],
        ]
    )
    luma_delta = last[:, :, indices["luma_mean"]] - first[:, :, indices["luma_mean"]]
    edge_delta = last[:, :, indices["edge_density"]] - first[:, :, indices["edge_density"]]
    motion_delta = last[:, :, indices["motion_energy"]] - first[:, :, indices["motion_energy"]]
    direction = np.sign(luma_delta).astype(np.float32)
    drift_strength = np.clip(np.abs(luma_delta) / 255.0, 0.0, 1.0).astype(np.float32)

    return {
        "target_frame": window.target_frame,
        "prior_frames": prior_frames,
        "future_frames": future_frames,
        "anchor_frames": [int(first_frame), int(last_frame)],
        "confidence": round(float(confidence), 6),
        "available_observed_frames": int(window.observed_count),
        "core_channels": list(indices.keys()),
        "summary": {
            "mean_abs_rgb_delta": round(float(np.mean(np.abs(rgb_delta))), 6),
            "mean_abs_luma_delta": round(float(np.mean(np.abs(luma_delta))), 6),
            "mean_abs_edge_delta": round(float(np.mean(np.abs(edge_delta))), 6),
            "mean_abs_motion_delta": round(float(np.mean(np.abs(motion_delta))), 6),
            "mean_drift_strength": round(float(np.mean(drift_strength)), 6),
        },
        "arrays": {
            "luma_direction": direction,
            "drift_strength": drift_strength,
        },
    }

