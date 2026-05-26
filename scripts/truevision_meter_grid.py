#!/usr/bin/env python3
"""Build compact TrueVision meter-grid profiles and event graphs from native capture state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from truevision_runtime.learning_intake.meter_grid import write_meter_grid_from_capture


DEFAULT_STORAGE_ROOT = Path("storage")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build TrueVision Meter Grid v0 profiles from native tvcells capture manifests.")
    parser.add_argument("--manifest", required=True, help="Native TrueVision capture manifest JSON.")
    parser.add_argument("--storage-root", default=str(DEFAULT_STORAGE_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--section-id", required=True)
    parser.add_argument("--event-type-candidate", default="candidate_lightning")
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--sample-stride", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = write_meter_grid_from_capture(
        {
            "manifest": args.manifest,
            "run_id": args.run_id,
            "section_id": args.section_id,
            "event_type_candidate": args.event_type_candidate,
            "max_frames": args.max_frames,
            "sample_stride": args.sample_stride,
        },
        storage_root=Path(args.storage_root),
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
