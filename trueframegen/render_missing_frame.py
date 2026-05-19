"""Render TrueFrameGen-filled cell states back to frames."""

from __future__ import annotations

import cv2
import numpy as np

from truevision_state_replay import build_rgb_replay_frame


def render_missing_frame(
    cells: np.ndarray,
    *,
    feature_names: list[str],
    output_shape: tuple[int, int],
    smooth: bool = True,
) -> np.ndarray:
    """Render filled cell state as RGB frame."""
    frame = build_rgb_replay_frame(cells, feature_names=feature_names, output_shape=output_shape)
    if not smooth:
        return frame
    return cv2.GaussianBlur(frame, (3, 3), 0)

