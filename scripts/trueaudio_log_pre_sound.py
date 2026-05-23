#!/usr/bin/env python3
"""Log TrueAudio state from decoded PCM before playback/output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueaudio_runtime.logging import log_pre_sound_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Write TrueAudio pre-sound state logs.")
    parser.add_argument("--audio", required=True, help="Source audio file")
    parser.add_argument("--storage-root", default="storage", help="Storage root for artifacts/manifests/receipts")
    parser.add_argument("--run-id", default="", help="Optional deterministic run id")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    result = log_pre_sound_state(
        Path(args.audio),
        storage_root=Path(args.storage_root),
        run_id=args.run_id or None,
        fps=args.fps,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
