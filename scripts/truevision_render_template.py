#!/usr/bin/env python3
"""Render a TrueVision audio/video state template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from truevision_runtime.rendering.template_renderer import render_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a reusable TrueVision AV template.")
    parser.add_argument("template", help="Path to the JSON render template.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Override template duration for previews.")
    parser.add_argument("--visual-only", action="store_true", help="Do not mux source audio into the output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = render_template(Path(args.template), max_seconds=args.max_seconds, mux_audio=not args.visual_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

