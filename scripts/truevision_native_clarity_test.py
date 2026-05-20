from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NATIVE_EXE = ROOT / "native" / "truevision_capture_rs" / "target" / "release" / "truevision_capture_rs.exe"
DEFAULT_VAULT = Path(r"E:\TruEVision Generation")


def parse_pair(value: str, *, name: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"{name} must look like WIDTHxHEIGHT")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} values must be positive")
    return width, height


def build_clarity_plan(
    *,
    vault: Path,
    duration: float,
    fps: float,
    resolution: str,
    grid: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    resolution_xy = parse_pair(resolution, name="resolution")
    grid_xy = parse_pair(grid, name="grid")
    if resolution_xy[0] % grid_xy[0] != 0 or resolution_xy[1] % grid_xy[1] != 0:
        raise ValueError("resolution must divide evenly by grid for honest block replay")
    run_id = run_id or f"clarity_rs_{resolution_xy[0]}x{resolution_xy[1]}_{grid_xy[0]}x{grid_xy[1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    capture_root = vault / "library" / "capture_units" / "20_minute" / "incoming"
    preview_root = vault / "library" / "renders" / "previews" / run_id
    run_dir = capture_root / run_id
    native_command = [
        str(NATIVE_EXE),
        "--duration",
        str(duration),
        "--fps",
        str(fps),
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
        "30",
    ]
    replay_command = [
        sys.executable,
        str(ROOT / "scripts" / "truevision_state_replay.py"),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(preview_root),
        "--fps",
        str(fps),
    ]
    return {
        "schema": "truevision_native_clarity_test.v1",
        "run_id": run_id,
        "boundary": {
            "capture_loop_only": True,
            "no_temporal_616": True,
            "no_signature_analysis": True,
            "no_replay_inside_capture": True,
            "raw_frames_saved": False,
        },
        "capture": {
            "duration_seconds": duration,
            "fps": fps,
            "resolution": resolution,
            "grid": grid,
            "pixels_per_cell": [resolution_xy[0] // grid_xy[0], resolution_xy[1] // grid_xy[1]],
            "run_dir": str(run_dir),
            "command": native_command,
        },
        "replay": {
            "preview_dir": str(preview_root),
            "command": replay_command,
        },
    }


def run_command(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or execute a native TrueVision playback clarity test.")
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=9.0)
    parser.add_argument("--resolution", default="2560x1440")
    parser.add_argument("--grid", default="640x360")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--execute", action="store_true", help="Run capture and replay. Omit to print the plan only.")
    args = parser.parse_args()

    plan = build_clarity_plan(
        vault=args.vault,
        duration=args.duration,
        fps=args.fps,
        resolution=args.resolution,
        grid=args.grid,
        run_id=args.run_id or None,
    )
    print(json.dumps(plan, indent=2))

    if args.execute:
        run_command(plan["capture"]["command"], cwd=ROOT)
        run_command(plan["replay"]["command"], cwd=ROOT)


if __name__ == "__main__":
    main()
