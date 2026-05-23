#!/usr/bin/env python3
"""Log TrueAudio state from the local machine output mix before speakers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueaudio_runtime.logging import log_machine_pre_sound_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Write TrueAudio machine pre-sound state logs.")
    parser.add_argument("--storage-root", default="storage", help="Storage root for artifacts/manifests/receipts")
    parser.add_argument("--run-id", default="", help="Optional deterministic run id")
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    result = log_machine_pre_sound_state(
        storage_root=Path(args.storage_root),
        run_id=args.run_id or None,
        duration_seconds=args.duration_seconds,
        fps=args.fps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
