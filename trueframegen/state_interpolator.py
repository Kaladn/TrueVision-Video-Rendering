"""State interpolation for missing TrueVision frames."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .causal_cell_map import CORE_CHANNELS, build_causal_cell_map, feature_indices
from .temporal_616 import TemporalWindow616


def _weighted_linear_fit_predict(frame_numbers: np.ndarray, values: np.ndarray, target_frame: int) -> np.ndarray:
    distances = np.maximum(1.0, np.abs(frame_numbers.astype(np.float32) - float(target_frame)))
    weights = 1.0 / distances
    x = frame_numbers.astype(np.float32)
    x_mean = np.sum(weights * x) / np.sum(weights)
    y_mean = np.sum(values * weights[:, None, None], axis=0) / np.sum(weights)
    centered_x = x - x_mean
    denom = np.sum(weights * centered_x * centered_x)
    if float(denom) <= 1.0e-6:
        return y_mean
    slope = np.sum(weights[:, None, None] * centered_x[:, None, None] * (values - y_mean), axis=0) / denom
    return y_mean + slope * (float(target_frame) - x_mean)


def interpolate_missing_state(
    cells_by_frame: Mapping[int, np.ndarray],
    window: TemporalWindow616,
    *,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill target frame from observed 6-1-6 temporal cause/effect cloud."""
    if not window.has_left_anchor or not window.has_right_anchor:
        raise ValueError("TrueFrameGen requires at least one prior and one future anchor")
    frames = np.asarray(list(window.prior_frames) + list(window.future_frames), dtype=np.int32)
    if frames.size < 2:
        raise ValueError("at least two observed frames are required")

    sorted_order = np.argsort(frames)
    frames = frames[sorted_order]
    stack = np.stack([cells_by_frame[int(frame)] for frame in frames], axis=0).astype(np.float32)
    output = stack[np.argmin(np.abs(frames - window.target_frame))].copy()

    indices = feature_indices(feature_names)
    channel_traces: dict[str, dict[str, Any]] = {}
    for channel_name in CORE_CHANNELS:
        index = indices[channel_name]
        values = stack[:, :, :, index]
        predicted = _weighted_linear_fit_predict(frames, values, window.target_frame)
        if channel_name.startswith("rgb_") or channel_name == "luma_mean":
            output[:, :, index] = np.clip(predicted, 0.0, 255.0)
        else:
            max_observed = max(1.0, float(np.nanmax(values)))
            output[:, :, index] = np.clip(predicted, 0.0, max_observed)
        channel_traces[channel_name] = {
            "source_frames": [int(frame) for frame in frames.tolist()],
            "mean_prediction": round(float(np.mean(output[:, :, index])), 6),
            "mean_observed_min": round(float(np.min(values)), 6),
            "mean_observed_max": round(float(np.max(values)), 6),
        }

    causal = build_causal_cell_map(cells_by_frame, window, feature_names=feature_names)
    trace = {
        "target_frame": int(window.target_frame),
        "temporal_616": {
            "prior_frames": [int(frame) for frame in window.prior_frames],
            "center": int(window.target_frame),
            "future_frames": [int(frame) for frame in window.future_frames],
        },
        "rule": "interpolate_from_observed_temporal_cause_first",
        "hallucination_used": False,
        "confidence": causal["confidence"],
        "causal_summary": causal["summary"],
        "channel_traces": channel_traces,
    }
    return output.astype(np.float32), trace

