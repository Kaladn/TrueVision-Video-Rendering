#!/usr/bin/env python3
"""Run native TrueVision capture and start TrueFrameGen after a trailing delay."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NATIVE_EXE = ROOT / "native" / "truevision_capture_rs" / "target" / "release" / "truevision_capture_rs.exe"
DEFAULT_VAULT = Path(r"E:\TruEVision Generation")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_pair(value: str, *, name: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"{name} must look like WIDTHxHEIGHT")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} values must be positive")
    return width, height


def build_live_pipeline_plan(
    *,
    vault: Path,
    run_id: str,
    duration: float,
    capture_fps: float,
    target_fps: float,
    resolution: str,
    grid: str,
    region: str,
    tfg_start_after: float,
    chunk_frames: int,
    crf: int,
) -> dict[str, Any]:
    resolution_xy = parse_pair(resolution, name="resolution")
    grid_xy = parse_pair(grid, name="grid")
    if resolution_xy[0] % grid_xy[0] != 0 or resolution_xy[1] % grid_xy[1] != 0:
        raise ValueError("resolution must divide evenly by grid")
    capture_root = vault / "library" / "capture_units" / "20_minute" / "incoming"
    run_dir = capture_root / run_id
    tfg_dir = vault / "library" / "trueframegen" / f"{run_id}_{int(round(target_fps))}fps_live"
    capture_command = [
        str(NATIVE_EXE),
        "--duration",
        str(duration),
        "--fps",
        str(capture_fps),
        "--resolution",
        resolution,
        "--grid",
        grid,
        "--output-root",
        str(capture_root),
        "--run-id",
        run_id,
        "--start-delay",
        "0",
        "--cell-chunk-frames",
        str(chunk_frames),
    ]
    if region:
        capture_command.extend(["--region", region])
    tfg_command = [
        sys.executable,
        str(ROOT / "scripts" / "trueframegen_live_upsample.py"),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(tfg_dir),
        "--capture-fps",
        str(capture_fps),
        "--target-fps",
        str(target_fps),
        "--duration",
        str(duration),
        "--resolution",
        resolution,
        "--radius",
        "6",
        "--crf",
        str(crf),
    ]
    return {
        "schema": "trueframegen_live_pipeline.v1",
        "run_id": run_id,
        "boundary": {
            "capture_and_tfg_overlap": True,
            "tfg_trails_capture_seconds": tfg_start_after,
            "raw_frames_saved": False,
            "tfg_generates_in_between_state": True,
            "not_append_at_end": True,
        },
        "capture": {
            "duration_seconds": duration,
            "fps": capture_fps,
            "resolution": resolution,
            "grid": grid,
            "region": region or "full_screen",
            "chunk_frames": chunk_frames,
            "run_dir": str(run_dir),
            "command": capture_command,
        },
        "trueframegen": {
            "target_fps": target_fps,
            "start_after_seconds": tfg_start_after,
            "output_dir": str(tfg_dir),
            "command": tfg_command,
        },
    }


def _wait_for_capture_ready(run_dir: Path, *, min_elapsed: float, timeout: float) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        manifest = list(run_dir.glob("*_manifest.json"))
        chunks = list((run_dir / "cell_state_native").glob("*.tvcells"))
        if manifest and chunks:
            return
        if time.monotonic() - started >= min_elapsed and chunks:
            return
        time.sleep(0.25)
    raise TimeoutError(f"capture did not produce chunks within {timeout:.1f}s: {run_dir}")


def run_pipeline(plan: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(plan["capture"]["run_dir"])
    tfg_dir = Path(plan["trueframegen"]["output_dir"])
    tfg_dir.mkdir(parents=True, exist_ok=True)
    plan_path = tfg_dir / f"{plan['run_id']}_live_pipeline_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")

    capture_log = tfg_dir / f"{plan['run_id']}_capture.log"
    tfg_log = tfg_dir / f"{plan['run_id']}_trueframegen.log"
    with capture_log.open("w", encoding="utf-8") as cap_handle:
        capture_proc = subprocess.Popen(
            plan["capture"]["command"],
            cwd=str(ROOT),
            stdout=cap_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    time.sleep(float(plan["trueframegen"]["start_after_seconds"]))
    _wait_for_capture_ready(run_dir, min_elapsed=float(plan["trueframegen"]["start_after_seconds"]), timeout=30.0)

    with tfg_log.open("w", encoding="utf-8") as tfg_handle:
        tfg_proc = subprocess.Popen(
            plan["trueframegen"]["command"],
            cwd=str(ROOT),
            stdout=tfg_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    capture_code = capture_proc.wait()
    tfg_code = tfg_proc.wait()
    result = {
        "plan_json": str(plan_path),
        "capture_log": str(capture_log),
        "trueframegen_log": str(tfg_log),
        "capture_returncode": capture_code,
        "trueframegen_returncode": tfg_code,
    }
    result_path = tfg_dir / f"{plan['run_id']}_live_pipeline_result.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if capture_code != 0 or tfg_code != 0:
        raise SystemExit(max(capture_code, tfg_code))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start native capture, then start TrueFrameGen while capture continues.")
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--capture-fps", type=float, default=9.0)
    parser.add_argument("--target-fps", type=float, default=60.0)
    parser.add_argument("--resolution", default="2560x1440")
    parser.add_argument("--grid", default="640x360")
    parser.add_argument("--region", default="")
    parser.add_argument("--tfg-start-after", type=float, default=10.0)
    parser.add_argument("--chunk-frames", type=int, default=9)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_id = args.run_id or f"live_tfg_{timestamp_slug()}"
    plan = build_live_pipeline_plan(
        vault=args.vault,
        run_id=run_id,
        duration=args.duration,
        capture_fps=args.capture_fps,
        target_fps=args.target_fps,
        resolution=args.resolution,
        grid=args.grid,
        region=args.region,
        tfg_start_after=args.tfg_start_after,
        chunk_frames=args.chunk_frames,
        crf=args.crf,
    )
    print(json.dumps(plan, indent=2, allow_nan=False))
    if args.execute:
        print(json.dumps(run_pipeline(plan), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
