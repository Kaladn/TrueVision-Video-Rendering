from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "storage" / "config" / "coordinate_maps" / "youtube_intake_map.json"


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _cursor_position() -> list[int]:
    point = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return [int(point.x), int(point.y)]


def _screen_size() -> list[int]:
    return [
        int(ctypes.windll.user32.GetSystemMetrics(0)),
        int(ctypes.windll.user32.GetSystemMetrics(1)),
    ]


def _wait_for_point(label: str) -> list[int]:
    input(f"Move mouse to {label}, then press Enter...")
    return _cursor_position()


def build_interactive_map() -> dict[str, Any]:
    screen = _screen_size()
    print(f"Detected screen: {screen[0]}x{screen[1]}")
    print("Map only the current YouTube/operator layout. Do not move windows after saving this map.")
    address_bar = _wait_for_point("browser address bar")
    video_play = _wait_for_point("video play area/button")
    top_left = _wait_for_point("capture region TOP-LEFT")
    bottom_right = _wait_for_point("capture region BOTTOM-RIGHT")
    x1, y1 = top_left
    x2, y2 = bottom_right
    capture_region = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
    return {
        "schema_version": "truevision_coordinate_surface_map_v1",
        "created_by": "truevision_coordinate_map_screen.py",
        "screen_size": screen,
        "points": {
            "address_bar": address_bar,
            "video_play": video_play,
        },
        "capture_region": capture_region,
        "operator_law": "Coordinates are layout-specific. Re-map after moving windows, changing scaling, or changing monitor layout.",
    }


def build_template_map() -> dict[str, Any]:
    screen = _screen_size()
    return {
        "schema_version": "truevision_coordinate_surface_map_v1",
        "created_by": "truevision_coordinate_map_screen.py",
        "template": True,
        "screen_size": screen,
        "points": {
            "address_bar": [0, 0],
            "video_play": [0, 0],
        },
        "capture_region": [0, 0, screen[0], screen[1]],
        "operator_law": "Fill coordinates before execute. Template coordinates are not valid.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a coordinate map for visible TrueVision YouTube intake.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--template", action="store_true", help="Write a template without interactive cursor capture.")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_template_map() if args.template else build_interactive_map()
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"coordinate map written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
