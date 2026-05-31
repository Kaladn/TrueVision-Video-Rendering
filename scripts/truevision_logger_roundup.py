from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truevision_runtime.logger_roundup import (
    analyze_deep_pixel_transform,
    build_logger_roundup_manifest,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Round up TrueVision logging/state lanes and optionally analyze source-pixel transforms."
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-image", default="")
    parser.add_argument("--transformed-image", default="")
    args = parser.parse_args()

    manifest = build_logger_roundup_manifest(Path(args.repo_root))
    if args.source_image or args.transformed_image:
        if not args.source_image or not args.transformed_image:
            parser.error("--source-image and --transformed-image must be provided together")
        manifest["deep_pixel_transform"]["analysis"] = analyze_deep_pixel_transform(
            Path(args.source_image),
            Path(args.transformed_image),
        )

    output = write_json(args.output, manifest)
    print(
        json.dumps(
            {
                "schema_version": "truevision_logger_roundup_script_result_v1",
                "manifest_json": str(output),
                "logger_lane_count": len(manifest["logger_lanes"]),
                "discovered_logger_file_count": len(manifest["discovered_logger_files"]),
                "six_one_six_mapping_enabled": False,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
