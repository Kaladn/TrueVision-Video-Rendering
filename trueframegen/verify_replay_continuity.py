"""Continuity verification for TrueFrameGen filled frames."""

from __future__ import annotations

from typing import Any

import numpy as np


def verify_filled_state_continuity(
    previous_cells: np.ndarray,
    filled_cells: np.ndarray,
    next_cells: np.ndarray,
    *,
    feature_names: list[str],
) -> dict[str, Any]:
    """Measure whether filled state sits between adjacent known states."""
    rgb_indices = [feature_names.index("rgb_mean_r"), feature_names.index("rgb_mean_g"), feature_names.index("rgb_mean_b")]
    luma_index = feature_names.index("luma_mean")
    edge_index = feature_names.index("edge_density")
    motion_index = feature_names.index("motion_energy")

    prev_jump = np.abs(filled_cells[:, :, rgb_indices] - previous_cells[:, :, rgb_indices])
    next_jump = np.abs(next_cells[:, :, rgb_indices] - filled_cells[:, :, rgb_indices])
    direct_jump = np.abs(next_cells[:, :, rgb_indices] - previous_cells[:, :, rgb_indices])
    luma_mid_error = np.abs(
        filled_cells[:, :, luma_index]
        - ((previous_cells[:, :, luma_index] + next_cells[:, :, luma_index]) * 0.5)
    )
    return {
        "mean_rgb_jump_from_previous": round(float(np.mean(prev_jump)), 6),
        "mean_rgb_jump_to_next": round(float(np.mean(next_jump)), 6),
        "mean_direct_rgb_gap": round(float(np.mean(direct_jump)), 6),
        "mean_luma_midpoint_error": round(float(np.mean(luma_mid_error)), 6),
        "mean_edge_density": round(float(np.mean(filled_cells[:, :, edge_index])), 6),
        "mean_motion_energy": round(float(np.mean(filled_cells[:, :, motion_index])), 6),
        "continuity_ok": bool(
            np.mean(prev_jump) <= max(1.0, float(np.mean(direct_jump)))
            and np.mean(next_jump) <= max(1.0, float(np.mean(direct_jump)))
        ),
    }

