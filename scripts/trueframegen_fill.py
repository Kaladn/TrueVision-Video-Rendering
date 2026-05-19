#!/usr/bin/env python3
"""Fill missing TrueVision frames with TrueFrameGen 6-1-6 interpolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueframegen.frame_gap_filler import fill_truevision_capture


def _parse_targets(value: str) -> list[int] | None:
    if not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill missing TrueVision cell-state frames using 6-1-6 temporal causality.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--target-frames", default="", help="Optional comma-separated missing frame numbers to fill.")
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--fps", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = fill_truevision_capture(
        Path(args.run_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        target_frames=_parse_targets(args.target_frames),
        radius=args.radius,
        fps=args.fps or None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

