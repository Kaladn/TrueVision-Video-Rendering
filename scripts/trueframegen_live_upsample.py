#!/usr/bin/env python3
"""Live TrueFrameGen renderer for native TrueVision chunk captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueframegen.live_upsampler import live_upsample_truevision_native_capture


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must look like WIDTHxHEIGHT")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution values must be positive")
    return height, width


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render 60fps TFG MP4 while native TrueVision capture writes chunks.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-fps", type=float, required=True)
    parser.add_argument("--target-fps", type=float, default=60.0)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--resolution", type=parse_resolution, default=parse_resolution("2560x1440"))
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--trailing-source-frames", type=int, default=2)
    parser.add_argument("--wait-timeout", type=float, default=180.0)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = live_upsample_truevision_native_capture(
        args.run_dir,
        output_dir=args.output_dir,
        frame_shape=args.resolution,
        capture_fps=args.capture_fps,
        duration_seconds=args.duration,
        target_fps=args.target_fps,
        radius=args.radius,
        crf=args.crf,
        trailing_source_frames=args.trailing_source_frames,
        wait_timeout_seconds=args.wait_timeout,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
