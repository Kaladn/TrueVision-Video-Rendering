#!/usr/bin/env python3
"""Project an atmospheric TrueVision capture across Edge Of The World."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueframegen.temporal_causality_projector import project_capture_to_audio


DEFAULT_AUDIO = Path(
    r"C:\Users\mydyi\OneDrive\Documents\Desktop\Album_Builds\Machine_Dread_Album_Sequenced"
    r"\01_ordered_audio\01 - Edge Of The World (I Am Your Nightmare).mp3"
)
DEFAULT_LYRICS = Path(r"C:\Users\mydyi\OneDrive\Documents\Desktop\Full Album Lyrics_sound.txt")
DEFAULT_OUTPUT_ROOT = Path(r"E:\TruEVision Generation\library\renders\full")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use a TrueVision capture as a 6-1-6 temporal teacher and project it across Edge Of The World."
    )
    parser.add_argument("--capture-run-dir", required=True, help="TrueVision capture directory containing manifest, summary, and cell chunks.")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO))
    parser.add_argument("--lyrics", default=str(DEFAULT_LYRICS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="edge_of_the_world_616_projected_atmosphere")
    parser.add_argument("--resolution", default="2560x1440")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Optional preview limit. Omit/0 for full song.")
    parser.add_argument("--visual-only", action="store_true", help="Do not mux the audio into the output.")
    parser.add_argument("--max-source-frames", type=int, default=0, help="Optional cap for profiling/tests. Omit/0 for all source frames.")
    parser.add_argument(
        "--style",
        default="projection",
        choices=["projection", "hell_power_walk"],
        help="Optional post-projection visual style. The default keeps raw projected cell RGB.",
    )
    parser.add_argument("--lightning-signature", default="", help="Optional TrueVision lightning signature JSON.")
    return parser


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError("resolution must look like WIDTHxHEIGHT")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("resolution values must be positive")
    return width, height


def main() -> None:
    args = build_parser().parse_args()
    width, height = parse_resolution(args.resolution)
    result = project_capture_to_audio(
        capture_run_dir=Path(args.capture_run_dir),
        audio_path=Path(args.audio),
        lyrics_path=Path(args.lyrics) if args.lyrics else None,
        output_root=Path(args.output_root),
        run_id=args.run_id,
        width=width,
        height=height,
        fps=args.fps,
        sample_rate=args.sample_rate,
        radius=args.radius,
        max_seconds=args.max_seconds or None,
        mux_audio=not args.visual_only,
        max_source_frames=args.max_source_frames or None,
        visual_style=args.style,
        lightning_signature_path=Path(args.lightning_signature) if args.lightning_signature else None,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
