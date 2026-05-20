from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truevision_runtime.storage_library import ensure_storage_library, storage_report


DEFAULT_EXTERNAL_ROOT = Path(r"E:\TruEVision Generation")


def default_root() -> Path:
    return Path(os.environ.get("TRUEVISION_STORAGE_ROOT") or DEFAULT_EXTERNAL_ROOT)


def print_report(root: Path) -> None:
    rows = storage_report(root)
    print(f"TrueVision storage report: {root}")
    for row in rows:
        print(f"{row['lane']:14} {row['size_gib']:10.3f} GiB  {row['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize and inspect the TrueVision media library vault.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create tidy media-library directories.")
    init.add_argument("--root", type=Path, default=default_root())
    init.add_argument("--json", action="store_true")

    report = sub.add_parser("report", help="Show disk usage by storage lane.")
    report.add_argument("--root", type=Path, default=default_root())
    report.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "init":
        result = ensure_storage_library(args.root)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Initialized TrueVision storage library: {result['root']}")
            print(f"Index: {result['index']}")
            print(f"Clip unit: {result['clip_unit_minutes']} minutes")
        return

    if args.command == "report":
        rows = storage_report(args.root)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print_report(args.root)


if __name__ == "__main__":
    main()
