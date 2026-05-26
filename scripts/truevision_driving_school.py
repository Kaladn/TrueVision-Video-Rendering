#!/usr/bin/env python3
"""Build TrueVision high-speed awareness profiles from local videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "modules") not in sys.path:
    sys.path.insert(0, str(ROOT / "modules"))

from truevision_runtime.learning_intake.driving_school import write_driving_school_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build local-only TrueVision high-speed awareness candidate receipts. "
            "This learns road, fog, reflection, terrain, city, and motion behavior; it is not self-driving."
        )
    )
    parser.add_argument("--sources", nargs="*", default=[], help="Explicit local video paths.")
    parser.add_argument("--source-folder", action="append", default=[], help="Folder containing local MP4 files.")
    parser.add_argument("--storage-root", default="storage")
    parser.add_argument("--run-id", default="driving_school_v0")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--long-edge-cells", type=int, default=48)
    args = parser.parse_args()

    result = write_driving_school_run(
        {
            "sources": args.sources,
            "source_folders": args.source_folder,
            "run_id": args.run_id,
            "sample_fps": args.sample_fps,
            "max_frames": args.max_frames,
            "long_edge_cells": args.long_edge_cells,
        },
        storage_root=Path(args.storage_root),
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
