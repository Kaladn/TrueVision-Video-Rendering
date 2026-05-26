from __future__ import annotations

import argparse
import json
from pathlib import Path

from truevision_runtime.timeline_audit import audit_many, audit_timeline_manifest


def _find_manifests(root: Path) -> list[Path]:
    candidates = sorted(root.rglob("*manifest*.json"))
    return [path for path in candidates if path.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit saved TrueVision timeline logs for frame/FPS timing integrity.")
    parser.add_argument("paths", nargs="+", help="Manifest JSON files or directories containing manifests.")
    parser.add_argument("--write", type=Path, default=None, help="Optional JSON path for the audit report.")
    args = parser.parse_args()

    manifests: list[Path] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.is_dir():
            manifests.extend(_find_manifests(path))
        else:
            manifests.append(path)

    result = audit_many(manifests) if len(manifests) != 1 else audit_timeline_manifest(manifests[0])
    payload = json.dumps(result, indent=2, allow_nan=False)
    print(payload)
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(payload + "\n", encoding="utf-8")
    return 0 if (result.get("status") == "pass" or result.get("fail_count", 0) == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
