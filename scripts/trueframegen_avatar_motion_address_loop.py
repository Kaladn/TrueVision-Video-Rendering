#!/usr/bin/env python3
"""Render the TrueVision avatar motion-address loop proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueframegen.avatar_motion_address import DEFAULT_OUTPUT, DEFAULT_POSES, render_motion_address_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a 10->2->10 motion-addressed avatar loop.")
    parser.add_argument("--pose-dir", default=str(DEFAULT_POSES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=700)
    parser.add_argument("--height", type=int, default=696)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--write-frame-pngs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = render_motion_address_loop(
        pose_dir=Path(args.pose_dir),
        output_dir=Path(args.output_dir),
        duration_seconds=args.seconds,
        fps=args.fps,
        size=(args.width, args.height),
        crf=args.crf,
        write_frame_pngs=args.write_frame_pngs,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
