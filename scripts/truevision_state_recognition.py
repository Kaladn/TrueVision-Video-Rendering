#!/usr/bin/env python3
"""Recognize TrueVision visual state changes from existing state/video rows.

Recognition only: no rendering, no animation, no camera movement, no objects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truevision_runtime.state_recognition import (  # noqa: E402
    recognize_states_from_jsonl,
    recognize_states_from_manifest,
    recognize_states_from_video,
    write_state_recognition_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and record visual state changes from existing TrueVision state, video, or JSONL rows."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", default="", help="TrueVision capture manifest with .tvcells or .npz cell state.")
    source.add_argument("--video", default="", help="Local video to sample into temporary in-memory state rows.")
    source.add_argument("--state-jsonl", default="", help="Existing frame/state JSONL rows.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", default="state_recognition")
    parser.add_argument("--max-frames", type=int, default=720)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--grid-rows", type=int, default=27)
    parser.add_argument("--grid-cols", type=int, default=48)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.manifest:
        report = recognize_states_from_manifest(
            args.manifest,
            run_id=args.run_id,
            max_frames=args.max_frames,
            sample_stride=args.sample_stride,
        )
    elif args.video:
        report = recognize_states_from_video(
            args.video,
            run_id=args.run_id,
            max_frames=args.max_frames,
            sample_stride=args.sample_stride,
            grid_shape=(args.grid_rows, args.grid_cols),
        )
    else:
        report = recognize_states_from_jsonl(
            args.state_jsonl,
            run_id=args.run_id,
            max_rows=args.max_frames,
        )

    result = write_state_recognition_outputs(report, output_root=Path(args.output_root), run_id=args.run_id)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
