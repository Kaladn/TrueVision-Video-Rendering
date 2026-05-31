from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truevision_runtime.transform_learning_frontdoor import (  # noqa: E402
    run_transform_learning_cycle,
    write_transform_learning_cycle,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TrueVision transform-learning cycle from observed state events and generated attempts."
    )
    parser.add_argument("--observed-events-json", required=True, help="JSON list of observed TrueVision event packets.")
    parser.add_argument("--generated-attempts-json", required=True, help="JSON list of generated transform attempt packets.")
    parser.add_argument("--transform-kind", required=True, help="Transform kind, e.g. lightning, fog_reveal, water_shimmer.")
    parser.add_argument("--output-root", default="storage/manifests/transform_learning_frontdoor", help="Manifest output root.")
    parser.add_argument("--run-id", default="transform_learning_frontdoor", help="Run id for manifest/receipt names.")
    parser.add_argument("--tolerance", type=float, default=0.12, help="Relative behavior tolerance.")
    return parser.parse_args()


def _read_json_list(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return payload


def main() -> int:
    args = _parse_args()
    observed_events = _read_json_list(args.observed_events_json)
    generated_attempts = _read_json_list(args.generated_attempts_json)
    cycle = run_transform_learning_cycle(
        observed_events,
        generated_attempts,
        transform_kind=args.transform_kind,
        tolerance=args.tolerance,
        profile_id=f"{args.run_id}_{args.transform_kind}",
    )
    result = write_transform_learning_cycle(cycle=cycle, output_root=args.output_root, run_id=args.run_id)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
