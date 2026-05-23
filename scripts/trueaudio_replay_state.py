#!/usr/bin/env python3
"""Replay a TrueAudio state log as deterministic state sonification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueaudio_runtime.replay import replay_trueaudio_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a WAV sonification from TrueAudio state JSONL.")
    parser.add_argument("--state", required=True, help="TrueAudio state JSONL")
    parser.add_argument("--storage-root", default="storage", help="Storage root for artifacts/manifests/receipts")
    parser.add_argument("--run-id", default="", help="Optional deterministic run id")
    parser.add_argument("--sample-rate", type=int, default=48000)
    args = parser.parse_args()

    result = replay_trueaudio_state(
        Path(args.state),
        storage_root=Path(args.storage_root),
        run_id=args.run_id or None,
        sample_rate=args.sample_rate,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
