#!/usr/bin/env python3
"""Select, snap, save, and run TrueVision screen regions.

This is a thin region/preset tool. It does not replace the recorder. It turns a
human-selected rectangle into a stable TrueVision-compatible capture preset and
can call the existing `truevision_resonance_recorder.py` with that region.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESET_DIR = PROJECT_ROOT / "presets"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "connected_artifacts"
RECORDER_PATH = PROJECT_ROOT / "scripts" / "truevision_resonance_recorder.py"


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center_x(self) -> float:
        return self.left + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2.0

    def as_list(self) -> list[int]:
        return [int(self.left), int(self.top), int(self.width), int(self.height)]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def parse_region(value: str) -> Region:
    parts = value.replace("x", ",").split(",")
    if len(parts) != 4:
        raise ValueError("region must look like left,top,width,height")
    left, top, width, height = [int(part.strip()) for part in parts]
    if width < 1 or height < 1:
        raise ValueError("region width/height must be positive")
    return Region(left, top, width, height)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _align_dimension(value: int, unit: int, minimum: int) -> int:
    aligned = max(minimum, int(round(value / unit)) * unit)
    return max(unit, aligned)


def snap_region_to_truevision(
    region: Region,
    *,
    bounds: Region | None = None,
    aspect_width: int = 16,
    aspect_height: int = 9,
    align_to_grid: bool = True,
) -> Region:
    """Snap a selected rectangle to a 16:9, grid-aligned capture region."""
    if region.width < 1 or region.height < 1:
        raise ValueError("region dimensions must be positive")
    target = aspect_width / aspect_height
    width = float(region.width)
    height = float(region.height)
    current = width / height

    if current > target:
        height = width / target
    else:
        width = height * target

    if align_to_grid:
        width = _align_dimension(int(round(width)), aspect_width, aspect_width)
        height = int(round(width * aspect_height / aspect_width))
        height = _align_dimension(height, aspect_height, aspect_height)
        width = int(round(height * aspect_width / aspect_height))

    if bounds is not None:
        max_w = max(aspect_width, bounds.width)
        max_h = max(aspect_height, bounds.height)
        if width > max_w:
            width = max_w
            height = width * aspect_height / aspect_width
        if height > max_h:
            height = max_h
            width = height * aspect_width / aspect_height
        if align_to_grid:
            width = min(bounds.width, _align_dimension(int(width), aspect_width, aspect_width))
            height = int(round(width * aspect_height / aspect_width))
            if height > bounds.height:
                height = _align_dimension(bounds.height, aspect_height, aspect_height)
                width = int(round(height * aspect_width / aspect_height))

    width_i = max(aspect_width, int(round(width)))
    height_i = max(aspect_height, int(round(width_i * aspect_height / aspect_width)))
    if align_to_grid:
        width_i = _align_dimension(width_i, aspect_width, aspect_width)
        height_i = int(round(width_i * aspect_height / aspect_width))
        if height_i % aspect_height != 0:
            height_i = _align_dimension(height_i, aspect_height, aspect_height)
            width_i = int(round(height_i * aspect_width / aspect_height))

    left = int(round(region.center_x - width_i / 2.0))
    top = int(round(region.center_y - height_i / 2.0))

    if bounds is not None:
        left = _clamp(left, bounds.left, bounds.right - width_i)
        top = _clamp(top, bounds.top, bounds.bottom - height_i)

    return Region(left, top, width_i, height_i)


def _preset_payload(
    *,
    preset_id: str,
    selected: Region,
    snapped: Region,
    monitor: int,
    capture_resolution: tuple[int, int] = (960, 540),
    grid: tuple[int, int] = (160, 90),
    blocks: tuple[int, int] = (16, 9),
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "truevision_region_preset",
        "preset_id": preset_id,
        "created_at_utc": utc_now(),
        "monitor": int(monitor),
        "selected_region": selected.as_list(),
        "snapped_region": snapped.as_list(),
        "snap_rule": "16:9_grid_aligned",
        "capture_resolution": [int(capture_resolution[0]), int(capture_resolution[1])],
        "grid": [int(grid[0]), int(grid[1])],
        "blocks": [int(blocks[0]), int(blocks[1])],
        "boundary": {
            "raw_capture_enabled": False,
            "recorder": "truevision_resonance_recorder.py",
            "operator_selected": True,
        },
    }
    payload["preset_hash"] = sha256_json({k: v for k, v in payload.items() if k != "preset_hash"})
    return payload


def save_preset(
    *,
    path: Path,
    preset_id: str,
    selected: Region,
    snapped: Region,
    monitor: int = 0,
    capture_resolution: tuple[int, int] = (960, 540),
    grid: tuple[int, int] = (160, 90),
    blocks: tuple[int, int] = (16, 9),
) -> dict[str, Any]:
    payload = _preset_payload(
        preset_id=preset_id,
        selected=selected,
        snapped=snapped,
        monitor=monitor,
        capture_resolution=capture_resolution,
        grid=grid,
        blocks=blocks,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def load_preset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "truevision_region_preset":
        raise ValueError("not a TrueVision region preset")
    return payload


def build_recorder_command(
    preset: dict[str, Any],
    *,
    duration: float,
    fps: int,
    output_root: Path,
    run_id: str,
    python_exe: str | None = None,
) -> list[str]:
    region = ",".join(str(int(v)) for v in preset["snapped_region"])
    resolution = "x".join(str(int(v)) for v in preset.get("capture_resolution", [960, 540]))
    grid = "x".join(str(int(v)) for v in preset.get("grid", [160, 90]))
    blocks = "x".join(str(int(v)) for v in preset.get("blocks", [16, 9]))
    return [
        python_exe or sys.executable,
        str(RECORDER_PATH),
        "--duration",
        str(duration),
        "--fps",
        str(int(fps)),
        "--resolution",
        resolution,
        "--grid",
        grid,
        "--blocks",
        blocks,
        "--region",
        region,
        "--monitor",
        str(int(preset.get("monitor", 0))),
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--start-delay",
        "0",
    ]


def select_region_tkinter() -> Region:
    """Interactive primary-screen selector. Esc cancels, mouse drag accepts."""
    import tkinter as tk

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.28)
    root.configure(bg="black")
    canvas = tk.Canvas(root, cursor="crosshair", bg="black", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    state: dict[str, Any] = {"start": None, "rect": None, "region": None}

    def on_down(event: tk.Event) -> None:
        state["start"] = (event.x_root, event.y_root)
        if state["rect"] is not None:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#00ff88", width=3)

    def on_move(event: tk.Event) -> None:
        if state["start"] is None or state["rect"] is None:
            return
        sx, sy = state["start"]
        canvas.coords(state["rect"], sx, sy, event.x_root, event.y_root)

    def on_up(event: tk.Event) -> None:
        if state["start"] is None:
            return
        sx, sy = state["start"]
        left = min(sx, event.x_root)
        top = min(sy, event.y_root)
        width = abs(event.x_root - sx)
        height = abs(event.y_root - sy)
        if width > 4 and height > 4:
            state["region"] = Region(left, top, width, height)
        root.quit()

    def on_escape(_event: tk.Event) -> None:
        state["region"] = None
        root.quit()

    canvas.bind("<ButtonPress-1>", on_down)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_up)
    root.bind("<Escape>", on_escape)
    root.mainloop()
    root.destroy()
    if state["region"] is None:
        raise RuntimeError("region selection cancelled")
    return state["region"]


def _screen_bounds() -> Region:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        bounds = Region(0, 0, root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return bounds
    except Exception:
        return Region(0, 0, 1920, 1080)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select and run TrueVision-compatible screen capture regions.")
    parser.add_argument("--select", action="store_true", help="Drag-select a region with a transparent overlay.")
    parser.add_argument("--region", default="", help="Manual selected region: left,top,width,height.")
    parser.add_argument("--bounds", default="", help="Optional bounds: left,top,width,height. Defaults to primary screen.")
    parser.add_argument("--preset-id", default="truevision_region")
    parser.add_argument("--preset-path", default="")
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Run the existing TrueVision recorder against the preset.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.select:
        selected = select_region_tkinter()
    elif args.region:
        selected = parse_region(args.region)
    else:
        raise SystemExit("provide --select or --region")

    bounds = parse_region(args.bounds) if args.bounds else _screen_bounds()
    snapped = snap_region_to_truevision(selected, bounds=bounds)
    preset_path = Path(args.preset_path) if args.preset_path else DEFAULT_PRESET_DIR / f"{args.preset_id}.json"
    preset = save_preset(
        path=preset_path,
        preset_id=args.preset_id,
        selected=selected,
        snapped=snapped,
        monitor=args.monitor,
    )
    run_id = args.run_id or f"{args.preset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    command = build_recorder_command(
        preset,
        duration=args.duration,
        fps=args.fps,
        output_root=Path(args.output_root),
        run_id=run_id,
    )
    result = {
        "preset_path": str(preset_path),
        "selected_region": selected.as_list(),
        "snapped_region": snapped.as_list(),
        "recorder_command": command,
    }
    print(json.dumps(result, indent=2, allow_nan=False))

    if args.print_command:
        print(" ".join(f'"{part}"' if " " in part else part for part in command))
    if args.watch:
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
