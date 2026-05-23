#!/usr/bin/env python3
"""Replay close TrueAudio spectral state as WAV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueaudio_runtime.replayable import replay_replayable_audio_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Render WAV from replayable TrueAudio spectral state.")
    parser.add_argument("--state", required=True, help="Replayable TrueAudio .npz state")
    parser.add_argument("--storage-root", default="storage", help="Storage root for artifacts/manifests/receipts")
    parser.add_argument("--run-id", default="", help="Optional deterministic run id")
    args = parser.parse_args()

    result = replay_replayable_audio_state(
        Path(args.state),
        storage_root=Path(args.storage_root),
        run_id=args.run_id or None,
    )
    result = {key: value for key, value in result.items() if key != "samples"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
