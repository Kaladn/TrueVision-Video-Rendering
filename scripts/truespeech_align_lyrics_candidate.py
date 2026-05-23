#!/usr/bin/env python3
"""Align provided lyrics to speech-state segments as candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueaudio_runtime.lyrics import align_lyrics_to_speech_segments


def main() -> int:
    parser = argparse.ArgumentParser(description="Create candidate lyric timing from TrueSpeech segments.")
    parser.add_argument("--segments", required=True, help="TrueSpeech segments JSON")
    parser.add_argument("--lyrics", required=True, help="Lyrics text file")
    parser.add_argument("--storage-root", default="storage", help="Storage root for artifacts/manifests/receipts")
    parser.add_argument("--run-id", default="", help="Optional deterministic run id")
    args = parser.parse_args()

    result = align_lyrics_to_speech_segments(
        Path(args.segments),
        lyrics_path=Path(args.lyrics),
        storage_root=Path(args.storage_root),
        run_id=args.run_id or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
