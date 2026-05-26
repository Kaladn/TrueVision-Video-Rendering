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

from truevision_runtime.learning_intake.lightfield_focus import write_state_focus_lens_from_capture


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a TrueVision State Focus Lens profile from broad native capture state.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--element-id", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--focus-depths", default="-1,-0.5,0,0.5,1")
    args = parser.parse_args()

    depths = [float(part.strip()) for part in args.focus_depths.split(",") if part.strip()]
    result = write_state_focus_lens_from_capture(
        {
            "manifest": args.manifest,
            "element_id": args.element_id,
            "run_id": args.run_id or f"{args.element_id}_state_focus_lens",
            "max_frames": args.max_frames,
            "sample_stride": args.sample_stride,
            "focus_depths": depths,
        },
        storage_root=Path(args.storage_root),
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
