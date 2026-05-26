#!/usr/bin/env python3
"""Compatibility entrypoint for the TrueVision high-speed awareness lane."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "modules") not in sys.path:
    sys.path.insert(0, str(ROOT / "modules"))

from truevision_driving_school import main


if __name__ == "__main__":
    raise SystemExit(main())
