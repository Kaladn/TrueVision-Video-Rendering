#!/usr/bin/env python3
"""Build geometry data containers from existing TrueVision logger artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "modules") not in sys.path:
    sys.path.insert(0, str(ROOT / "modules"))

from truevision_runtime.geometry_generation import write_geometry_generation_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a TrueVision geometry scene and reusable big-shape library from logger outputs. "
            "Shapes carry source refs, true local metrics, and filtered metrics separately."
        )
    )
    parser.add_argument("--storage-root", default="storage")
    parser.add_argument("--run-id", default="geometry_generation")
    parser.add_argument("--output-root", default="outputs/geometry_generation")
    parser.add_argument("--render-preview", action="store_true")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)

    parser.add_argument("--meter-grid-profile", default="")
    parser.add_argument("--angular-seismic-profile", default="")
    parser.add_argument("--state-focus-profile", default="")
    parser.add_argument("--element-creation-profile", default="")
    parser.add_argument("--truedepth-profile", default="")
    parser.add_argument("--truedepth-signature", default="")
    parser.add_argument("--atmosphere-profile", default="")
    parser.add_argument("--weather-profile", default="")
    parser.add_argument("--trueaudio-manifest", default="")
    parser.add_argument("--trueaudio-profile", default="")
    parser.add_argument("--driving-profile", default="")
    parser.add_argument("--awareness-profile", default="")
    parser.add_argument("--worker-forge-manifest", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--receipt", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = write_geometry_generation_run(
        {
            "run_id": args.run_id,
            "output_root": args.output_root,
            "render_preview": bool(args.render_preview),
            "duration": args.duration,
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "meter_grid_profile": args.meter_grid_profile,
            "angular_seismic_profile": args.angular_seismic_profile,
            "state_focus_profile": args.state_focus_profile,
            "element_creation_profile": args.element_creation_profile,
            "truedepth_profile": args.truedepth_profile,
            "truedepth_signature": args.truedepth_signature,
            "atmosphere_profile": args.atmosphere_profile,
            "weather_profile": args.weather_profile,
            "trueaudio_manifest": args.trueaudio_manifest,
            "trueaudio_profile": args.trueaudio_profile,
            "driving_profile": args.driving_profile,
            "awareness_profile": args.awareness_profile,
            "worker_forge_manifest": args.worker_forge_manifest,
            "manifest": args.manifest,
            "receipt": args.receipt,
        },
        storage_root=Path(args.storage_root),
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
