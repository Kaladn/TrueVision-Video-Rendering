#!/usr/bin/env python3
"""Extract a reusable lightning/flash signature from TrueVision capture state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueframegen.lightning_signature import extract_lightning_signature_from_capture


DEFAULT_OUTPUT_DIR = Path(r"E:\TruEVision Generation\library\signature_profiles\lighting")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract peak intensity cells from a TrueVision capture as a lightning signature.")
    parser.add_argument("--capture-run-dir", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--signature-id", default="truevision_lightning_signature")
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--max-cells", type=int, default=420)
    parser.add_argument("--max-source-frames", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = extract_lightning_signature_from_capture(
        capture_run_dir=Path(args.capture_run_dir),
        output_dir=Path(args.output_dir),
        signature_id=args.signature_id,
        radius=args.radius,
        max_cells=args.max_cells,
        max_source_frames=args.max_source_frames or None,
    )
    print(json.dumps({"signature_json": result["signature_json"], "summary": result["signature"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
