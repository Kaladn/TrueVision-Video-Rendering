#!/usr/bin/env python3
"""Build a 16-direction Angular-Seismic profile from a local video."""

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

from truevision_runtime.learning_intake.angular_seismic import write_angular_seismic_profile_from_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TrueVision Angular-Seismic 16-side profiles from local video.")
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--storage-root", default="storage")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--loop-count", type=int, default=3)
    parser.add_argument("--sample-stride", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--grid-cols", type=int, default=48)
    parser.add_argument("--grid-rows", type=int, default=27)
    parser.add_argument("--rings", default="1,2,3,4")
    args = parser.parse_args()

    result = write_angular_seismic_profile_from_video(
        {
            "source_video": args.source_video,
            "run_id": args.run_id or Path(args.source_video).stem,
            "loop_count": args.loop_count,
            "sample_stride": args.sample_stride,
            "max_frames": args.max_frames,
            "grid_cols": args.grid_cols,
            "grid_rows": args.grid_rows,
            "rings": args.rings,
        },
        storage_root=Path(args.storage_root),
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
