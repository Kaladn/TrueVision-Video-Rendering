#!/usr/bin/env python3
"""Memory-bounded TrueFrameGen upsample from completed TrueVision chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueframegen import stream_upsample_truevision_capture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate high-FPS TFG output with a bounded chunk cache.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-fps", type=float, default=60.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--max-cached-chunks", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = stream_upsample_truevision_capture(
        args.run_dir,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        max_seconds=args.max_seconds,
        radius=args.radius,
        crf=args.crf,
        max_cached_chunks=args.max_cached_chunks,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
