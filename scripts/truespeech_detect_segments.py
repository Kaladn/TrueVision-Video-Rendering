#!/usr/bin/env python3
"""Detect speech/background segments from replayable TrueAudio state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueaudio_runtime.speech import detect_speech_segments_from_replayable_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect speech-like segments from replayable TrueAudio state.")
    parser.add_argument("--state", required=True, help="Replayable TrueAudio .npz state")
    parser.add_argument("--storage-root", default="storage", help="Storage root for artifacts/manifests/receipts")
    parser.add_argument("--run-id", default="", help="Optional deterministic run id")
    parser.add_argument("--speech-threshold", type=float, default=0.48)
    parser.add_argument("--min-segment-seconds", type=float, default=0.12)
    args = parser.parse_args()

    result = detect_speech_segments_from_replayable_state(
        Path(args.state),
        storage_root=Path(args.storage_root),
        run_id=args.run_id or None,
        speech_threshold=args.speech_threshold,
        min_segment_seconds=args.min_segment_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
