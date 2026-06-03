from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path


def build_harness_receipt(
    *,
    run_id: str,
    scene_id: str,
    selected_count: int,
    rejected_count: int,
    output_paths: dict[str, str],
    truevideo_allowed: bool,
) -> dict[str, Any]:
    receipt = {
        "schema_version": "truevision_tool_harness_receipt_v1",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "scene_id": scene_id,
        "selected_tool_count": int(selected_count),
        "rejected_tool_count": int(rejected_count),
        "outputs": output_paths,
        "truevideo": {
            "allowed_by_scene": bool(truevideo_allowed),
            "called": False,
        },
        "boundary": {
            "planning_only": True,
            "tools_invoked": False,
            "render_started": False,
            "capture_started": False,
            "external_services_called": False,
            "files_moved": False,
            "implementations_changed": False,
        },
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return receipt

