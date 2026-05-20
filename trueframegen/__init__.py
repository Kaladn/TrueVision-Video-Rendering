"""TrueFrameGen missing-frame filling and temporal projection."""

from .frame_gap_filler import fill_missing_frames, fill_truevision_capture
from .frame_upsampler import stream_upsample_truevision_capture, upsample_truevision_capture
from .live_upsampler import live_upsample_truevision_native_capture, load_live_native_sequence
from .temporal_616 import build_temporal_616_map
from .temporal_causality_projector import build_temporal_projection_profile, project_capture_to_audio

__all__ = [
    "build_temporal_616_map",
    "build_temporal_projection_profile",
    "fill_missing_frames",
    "fill_truevision_capture",
    "live_upsample_truevision_native_capture",
    "load_live_native_sequence",
    "project_capture_to_audio",
    "stream_upsample_truevision_capture",
    "upsample_truevision_capture",
]
