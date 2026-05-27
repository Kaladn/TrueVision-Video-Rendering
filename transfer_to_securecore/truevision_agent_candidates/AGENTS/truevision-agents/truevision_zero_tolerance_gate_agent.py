#!/usr/bin/env python3
"""Validate a TrueVision-to-SecureCore agent handoff package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


AGENT_ID = "truevision_zero_tolerance_gate_agent"
REQUIRED_MANIFEST_FIELDS = {
    "agent_id",
    "name",
    "version",
    "runtime_language",
    "entrypoint",
    "entrypoint_hash",
    "allowed_reads",
    "allowed_writes",
    "requires_approval",
    "approval_phrase",
    "mutation_class",
    "dry_run_supported",
    "log_stream",
    "test_command",
    "risk_tier",
    "prompt_only_allowed",
    "required_params",
}
HASH_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(manifest))
    errors.extend(f"missing:{field}" for field in missing)
    if manifest.get("prompt_only_allowed") is not False:
        errors.append("prompt_only_allowed_not_false")
    if str(manifest.get("runtime_language", "")).lower() == "prompt":
        errors.append("prompt_runtime_forbidden")
    if not HASH_RE.match(str(manifest.get("entrypoint_hash", ""))):
        errors.append("entrypoint_hash_invalid")
    entrypoint = root / str(manifest.get("entrypoint", ""))
    if not entrypoint.exists():
        errors.append("entrypoint_missing")
    elif file_hash(entrypoint) != manifest.get("entrypoint_hash"):
        errors.append("entrypoint_hash_mismatch")
    return {
        "manifest": str(manifest_path),
        "agent_id": str(manifest.get("agent_id", "")),
        "passed": not errors,
        "errors": errors,
    }


def validate_catalog(path: Path) -> dict[str, Any]:
    required_header = [
        "operator_id",
        "name",
        "category",
        "agent_tier",
        "definition",
        "input_shape",
        "output_shape",
        "assumptions_in",
        "assumptions_out",
        "side_effects",
        "dependencies",
        "statefulness",
        "sync_mode",
        "destruction_score",
        "risk_type",
        "risk_reason",
        "requires_confirmation",
        "sandbox_required",
        "promotion_ready",
        "contract_status",
        "category_confidence",
        "source_file",
    ]
    if not path.exists():
        return {"catalog": str(path), "passed": False, "errors": ["catalog_missing"], "row_count": 0}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)
    errors = []
    if header != required_header:
        errors.append("catalog_header_mismatch")
    if not rows:
        errors.append("catalog_empty")
    return {"catalog": str(path), "passed": not errors, "errors": errors, "row_count": len(rows)}


def build_gate_report(handoff_root: Path) -> dict[str, Any]:
    agents_root = handoff_root / "AGENTS"
    manifest_results = [
        validate_manifest(handoff_root, path)
        for path in sorted((agents_root / "agents").glob("*.agent.json"))
    ]
    catalog_results = [
        validate_catalog(agents_root / "catalog" / "agent_catalog.csv"),
        validate_catalog(agents_root / "catalog" / "securecore_agents.csv"),
    ]
    passed = (
        bool(manifest_results)
        and all(item["passed"] for item in manifest_results)
        and all(item["passed"] for item in catalog_results)
    )
    report = {
        "schema_version": "securecore_truevision_zero_tolerance_gate_report_v1",
        "agent_id": AGENT_ID,
        "decision_type": "handoff_package_gate",
        "confidence": 1.0 if passed else 0.0,
        "handoff_root": str(handoff_root),
        "passed": passed,
        "manifest_results": manifest_results,
        "catalog_results": catalog_results,
        "recommended_action": "securecore_review" if passed else "reject_until_fixed",
        "context": {
            "only_agents_transfer": True,
            "workers_remain_with_organs": True,
            "prompt_only_agents_forbidden": True,
        },
    }
    report["decision_hash"] = stable_hash(report)
    return report


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{AGENT_ID}_decision.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    report = build_gate_report(Path(args.handoff_root))
    output_path = write_report(report, Path(args.out_dir))
    print(json.dumps({"agent_id": AGENT_ID, "decision_path": str(output_path), "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
