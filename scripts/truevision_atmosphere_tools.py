#!/usr/bin/env python3
"""Create reusable TrueVision atmosphere/weather state toolsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from truevision_runtime.state_patterns.atmosphere_weather import (
    build_atmosphere_profile_from_native_capture,
    build_atmosphere_toolset,
    list_atmosphere_elements,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fog, mist, cloud, and rain-glass TrueVision state tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available atmosphere/weather elements.")

    profile = sub.add_parser("profile", help="Extract an atmosphere profile from a native TrueVision capture manifest.")
    profile.add_argument("--manifest", required=True, help="Native capture manifest JSON.")
    profile.add_argument("--element-id", default="fog_density_field")
    profile.add_argument("--max-frames", type=int, default=180)
    profile.add_argument("--sample-stride", type=int, default=1)
    profile.add_argument("--output", default="", help="Optional profile JSON output path.")

    create = sub.add_parser("create", help="Write a reusable atmosphere/weather toolset template and manifest.")
    create.add_argument("--storage-root", default="storage")
    create.add_argument("--run-id", default="")
    create.add_argument("--capture-manifest", default="")
    create.add_argument(
        "--elements",
        default="fog_density_field,mist_veil_field,cloud_volume_field,rain_glass_field",
        help="Comma-separated element ids.",
    )
    create.add_argument("--max-profile-frames", type=int, default=180)

    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps({"elements": list_atmosphere_elements()}, indent=2, allow_nan=False))
        return 0
    if args.command == "profile":
        result = build_atmosphere_profile_from_native_capture(
            args.manifest,
            element_id=args.element_id,
            max_frames=args.max_frames,
            sample_stride=args.sample_stride,
        )
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps({"profile_json": str(output), "profile_sha256": result["profile_sha256"]}, indent=2))
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    if args.command == "create":
        elements = [item.strip() for item in args.elements.split(",") if item.strip()]
        result = build_atmosphere_toolset(
            storage_root=Path(args.storage_root),
            run_id=args.run_id or None,
            capture_manifest=Path(args.capture_manifest) if args.capture_manifest else None,
            element_ids=elements,
            max_profile_frames=args.max_profile_frames,
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
