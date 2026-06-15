#!/usr/bin/env python3
"""Generate a short TrueFrameGen midpoint film from the retained avatar poses."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSES = ROOT / "truevision_runtime" / "rendering" / "avatar_assets" / "truvision_avatar_v1" / "poses"
DEFAULT_OUTPUT = ROOT / "outputs" / "truvision_avatar_midpoint"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_pose_frames(pose_dir: Path, size: tuple[int, int]) -> list[np.ndarray]:
    paths = sorted(pose_dir.glob("*.png"))
    if len(paths) < 2:
        raise ValueError(f"need at least two pose frames in {pose_dir}")
    frames: list[np.ndarray] = []
    for path in paths:
        img = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        frames.append(np.asarray(img, dtype=np.uint8))
    return frames


def midpoint_frame(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return ((left.astype(np.uint16) + right.astype(np.uint16)) // 2).astype(np.uint8)


def midpoint_pass(frames: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for left, right in zip(frames, frames[1:]):
        out.append(left)
        out.append(midpoint_frame(left, right))
    out.append(frames[-1])
    return out


def run_midpoint_framegen(source: list[np.ndarray], passes: int) -> tuple[list[np.ndarray], list[dict[str, int]]]:
    frames = source
    trace: list[dict[str, int]] = []
    for run in range(1, passes + 1):
        before = len(frames)
        frames = midpoint_pass(frames)
        trace.append(
            {
                "run": run,
                "input_frames": before,
                "gaps_filled": before - 1,
                "output_frames": len(frames),
            }
        )
    return frames, trace


def write_frames(frames: list[np.ndarray], frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(frame_dir / f"frame_{index:04d}.png")


def write_video(frames: list[np.ndarray], output_mp4: Path, fps: float, crf: int) -> None:
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc:
        raise SystemExit(rc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a midpoint-generated avatar pose film.")
    parser.add_argument("--pose-dir", default=str(DEFAULT_POSES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--passes", type=int, default=6)
    parser.add_argument("--width", type=int, default=700)
    parser.add_argument("--height", type=int, default=696)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--write-frame-pngs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pose_dir = Path(args.pose_dir)
    output_dir = Path(args.output_dir)
    frame_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_frames = load_pose_frames(pose_dir, (args.width, args.height))
    generated_frames, trace = run_midpoint_framegen(source_frames, args.passes)
    fps = len(generated_frames) / float(args.seconds)
    seconds_label = f"{float(args.seconds):g}".replace(".", "_")
    output_mp4 = output_dir / f"truvision_avatar_midpoint_{seconds_label}s.mp4"
    write_video(generated_frames, output_mp4, fps=fps, crf=args.crf)
    if args.write_frame_pngs:
        write_frames(generated_frames, frame_dir)

    manifest = {
        "schema": "trueframegen_avatar_midpoint_manifest_v1",
        "created_at_utc": utc_now(),
        "pose_dir": str(pose_dir),
        "source_pose_frames": len(source_frames),
        "passes": args.passes,
        "math": "For every adjacent A/B pair, insert midpoint M = floor((A + B) / 2). Repeat end-to-end for each pass.",
        "trace": trace,
        "generated_frames": len(generated_frames),
        "seconds": float(args.seconds),
        "fps": fps,
        "resolution": [args.width, args.height],
        "outputs": {
            "video": str(output_mp4),
            "frame_pngs": str(frame_dir) if args.write_frame_pngs else None,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
