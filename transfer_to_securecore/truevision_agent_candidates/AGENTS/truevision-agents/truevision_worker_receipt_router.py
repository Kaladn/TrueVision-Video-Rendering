#!/usr/bin/env python3
"""Route one TrueVision worker receipt into a SecureCore-style decision packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


AGENT_ID = "truevision_worker_receipt_router"


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("worker receipt must be a JSON object")
    return value


def route_receipt(receipt_path: Path) -> dict[str, Any]:
    receipt = load_json(receipt_path)
    worker_name = str(
        receipt.get("worker_id")
        or receipt.get("worker_name")
        or receipt.get("tool")
        or receipt.get("agent_id")
        or "unknown_worker"
    )
    status = str(receipt.get("status") or receipt.get("result") or "unknown")
    output_refs = []
    for key in ("output_path", "artifact_json", "manifest_json", "receipt_json", "report_json"):
        if receipt.get(key):
            output_refs.append({"field": key, "value": str(receipt[key])})
    decision = {
        "schema_version": "securecore_truevision_worker_route_decision_v1",
        "agent_id": AGENT_ID,
        "decision_type": "worker_receipt_routed",
        "confidence": 1.0 if worker_name != "unknown_worker" else 0.25,
        "source_receipt": str(receipt_path),
        "worker_name": worker_name,
        "worker_status": status,
        "output_refs": output_refs,
        "recommended_action": "review_worker_output",
        "context": {
            "truevision_boundary": "worker remains in TrueVision",
            "securecore_boundary": "agent routes and reasons only",
            "truth_promotion": False,
        },
    }
    decision["decision_hash"] = stable_hash(decision)
    return decision


def write_decision(decision: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{AGENT_ID}_decision.json"
    output_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-receipt", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    decision = route_receipt(Path(args.worker_receipt))
    output_path = write_decision(decision, Path(args.out_dir))
    print(json.dumps({"agent_id": AGENT_ID, "decision_path": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

