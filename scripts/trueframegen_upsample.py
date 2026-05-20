#!/usr/bin/env python3
"""Generate in-between TrueVision frames at a higher target FPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueframegen.frame_upsampler import upsample_truevision_capture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upsample a TrueVision capture by generating in-between state frames.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--target-fps", type=float, default=60.0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--max-source-frames", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = upsample_truevision_capture(
        Path(args.run_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        target_fps=args.target_fps,
        max_seconds=args.max_seconds or None,
        radius=args.radius,
        crf=args.crf,
        max_source_frames=args.max_source_frames or None,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

