from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def safe_slug(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value)).strip("_")
    return clean[:80] or "av_tool"


def write_tool_receipt(
    *,
    storage_root: Path,
    tool: str,
    status: str,
    call: dict[str, Any],
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    receipts = storage_root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    receipt = {
        "receipt_kind": "truevision_av_tool_receipt_v1",
        "written_at_utc": now,
        "tool": tool,
        "status": status,
        "call_hash": stable_hash(call),
        "result_hash": stable_hash(result or {}),
        "call": call,
        "result": result or {},
        "error": error,
        "boundary": {
            "domain": "audio_video",
            "qwen_requested_only": True,
            "server_validated": True,
        },
    }
    path = receipts / f"{now.replace(':', '').replace('.', '_')}_{safe_slug(tool)}.json"
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "name": path.name,
        "path": str(path),
        "sha256": stable_hash(receipt),
        "status": status,
    }
