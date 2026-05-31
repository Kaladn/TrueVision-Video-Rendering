from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truevision_runtime.finalized_tools import copy_finalized_tool, finalized_tool_status  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy a finalized TrueVision tool into a new lab file.")
    parser.add_argument("--tool-id", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.status_only:
        print(json.dumps(finalized_tool_status(ROOT, args.tool_id), indent=2))
        return 0
    result = copy_finalized_tool(ROOT, args.tool_id, args.destination, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
