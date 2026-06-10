from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truevision_runtime.rendering.seven_sector_rave import render_seven_sector_rave


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Seven-Sector Rave Reactor proof clip.")
    parser.add_argument(
        "--audio",
        default=r"C:\Users\mydyi\Downloads\Lower the Room x Mind Scrape (Mashup).wav",
    )
    parser.add_argument(
        "--stems",
        default=r"C:\Users\mydyi\Downloads\Lower the Room x Mind Scrape (Mashup) Stems (86BPM).zip",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/seven_sector_rave_reactor/lower_room_mind_scrape_30s",
    )
    parser.add_argument("--run-id", default="lower_room_mind_scrape_30s")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = render_seven_sector_rave(
        audio_path=args.audio,
        stems_zip=args.stems,
        output_root=args.output_root,
        run_id=args.run_id,
        seconds=args.seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(manifest["output_video"])


if __name__ == "__main__":
    main()
