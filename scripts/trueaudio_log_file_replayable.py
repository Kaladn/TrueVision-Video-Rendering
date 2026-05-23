#!/usr/bin/env python3
"""Log replayable TrueAudio state from a source audio file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueaudio_runtime.replayable import log_file_replayable_audio_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Write replayable TrueAudio state from a WAV/MP3 source file.")
    parser.add_argument("--audio", "--path", dest="audio", required=True, help="Source WAV/MP3/etc")
    parser.add_argument("--storage-root", default="storage", help="Storage root for artifacts/manifests/receipts")
    parser.add_argument("--run-id", default="", help="Optional deterministic run id")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--frame-size", type=int, default=2048)
    parser.add_argument("--hop-size", type=int, default=512)
    args = parser.parse_args()

    result = log_file_replayable_audio_state(
        Path(args.audio),
        storage_root=Path(args.storage_root),
        run_id=args.run_id or None,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
