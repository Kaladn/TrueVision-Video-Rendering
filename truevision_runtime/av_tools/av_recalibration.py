from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .av_tool_receipts import utc_now


def append_recalibration_event(*, storage_root: Path, event: dict[str, Any]) -> dict[str, Any]:
    events = storage_root / "events"
    events.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_kind": "truevision_av_recalibration_v1",
        "written_at_utc": utc_now(),
        **event,
    }
    path = events / "av_recalibration.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    return {"path": str(path), "event": payload}


def list_recalibration_events(*, storage_root: Path, template_id: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
    path = storage_root / "events" / "av_recalibration.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if template_id and payload.get("template_id") != template_id:
            continue
        if kind and payload.get("kind") != kind:
            continue
        records.append(payload)
    return records
