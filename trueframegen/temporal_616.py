"""6-1-6 temporal window mapping for TrueFrameGen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TemporalWindow616:
    target_frame: int
    prior_frames: tuple[int, ...]
    future_frames: tuple[int, ...]
    radius: int = 6

    @property
    def observed_count(self) -> int:
        return len(self.prior_frames) + len(self.future_frames)

    @property
    def has_left_anchor(self) -> bool:
        return bool(self.prior_frames)

    @property
    def has_right_anchor(self) -> bool:
        return bool(self.future_frames)


def build_temporal_616_map(
    cells_by_frame: Mapping[int, np.ndarray],
    target_frame: int,
    *,
    radius: int = 6,
) -> TemporalWindow616:
    """Return observed frames in the 6-prior/6-future cloud around target."""
    if radius < 1:
        raise ValueError("radius must be >= 1")
    observed = sorted(int(frame) for frame in cells_by_frame.keys())
    prior = tuple(frame for frame in observed if target_frame - radius <= frame < target_frame)
    future = tuple(frame for frame in observed if target_frame < frame <= target_frame + radius)
    return TemporalWindow616(
        target_frame=int(target_frame),
        prior_frames=prior[-radius:],
        future_frames=future[:radius],
        radius=radius,
    )

