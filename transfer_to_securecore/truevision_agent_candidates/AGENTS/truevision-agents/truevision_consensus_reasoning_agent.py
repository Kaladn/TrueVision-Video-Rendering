#!/usr/bin/env python3
"""Build a candidate consensus packet from worker-produced evidence packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


AGENT_ID = "truevision_consensus_reasoning_agent"


SUPPORT_KEYS = (
    "shape_support",
    "glyph_support",
    "context_support",
    "meter_support",
    "persistence_support",
    "external_label_support",
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_packets(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("candidate file JSON array expected")
        return [item for item in value if isinstance(item, dict)]
    packets: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            packets.append(value)
    return packets


def support_score(packet: dict[str, Any]) -> float:
    support = packet.get("support")
    if not isinstance(support, dict):
        support = packet
    hits = 0
    for key in SUPPORT_KEYS:
        value = support.get(key)
        if value is True:
            hits += 1
        elif isinstance(value, (int, float)) and float(value) > 0.0:
            hits += 1
        elif isinstance(value, str) and value.strip():
            hits += 1
    return hits / float(len(SUPPORT_KEYS))


def build_consensus(candidate_path: Path) -> dict[str, Any]:
    packets = load_packets(candidate_path)
    scored = []
    for index, packet in enumerate(packets):
        score = support_score(packet)
        candidate_id = str(packet.get("candidate_id") or packet.get("region_id") or f"candidate_{index:04d}")
        scored.append({
            "candidate_id": candidate_id,
            "score": round(score, 4),
            "support_status": "candidate_supported" if score >= 0.5 else "candidate_weak",
            "source_index": index,
        })
    supported = [item for item in scored if item["score"] >= 0.5]
    decision = {
        "schema_version": "securecore_truevision_consensus_decision_v1",
        "agent_id": AGENT_ID,
        "decision_type": "candidate_consensus",
        "confidence": round(len(supported) / float(len(scored) or 1), 4),
        "source_candidates": str(candidate_path),
        "candidate_count": len(scored),
        "supported_count": len(supported),
        "candidate_results": scored,
        "recommended_action": "operator_review",
        "context": {
            "truth_promotion": False,
            "search_is_support_not_evidence": True,
            "workers_remain_with_organs": True,
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
    parser.add_argument("--candidate-packets", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    decision = build_consensus(Path(args.candidate_packets))
    output_path = write_decision(decision, Path(args.out_dir))
    print(json.dumps({"agent_id": AGENT_ID, "decision_path": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

