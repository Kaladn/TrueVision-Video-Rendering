from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path("truevision_runtime/finalized_tools_registry.json")


def _normalized_file_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_finalized_tools_registry(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    return json.loads((root / REGISTRY_PATH).read_text(encoding="utf-8"))


def _tool_record(repo_root: str | Path, tool_id: str) -> dict[str, Any]:
    registry = load_finalized_tools_registry(repo_root)
    tools = registry.get("tools") or {}
    if tool_id not in tools:
        raise KeyError(f"unknown finalized tool: {tool_id}")
    return dict(tools[tool_id])


def finalized_tool_status(repo_root: str | Path, tool_id: str) -> dict[str, Any]:
    root = Path(repo_root)
    tool = _tool_record(root, tool_id)
    path = root / tool["path"]
    actual_hash = _normalized_file_sha256(path) if path.exists() else ""
    expected_hash = str(tool.get("normalized_sha256") or "")
    return {
        "schema_version": "truevision_finalized_tool_status_v1",
        "tool_id": tool_id,
        "path": str(path),
        "exists": path.exists(),
        "lifecycle": tool.get("lifecycle"),
        "edit_policy": tool.get("edit_policy"),
        "copy_target_hint": tool.get("copy_target_hint"),
        "expected_normalized_sha256": expected_hash,
        "actual_normalized_sha256": actual_hash,
        "hash_matches": bool(actual_hash and actual_hash == expected_hash),
        "filesystem_read_only": bool(path.exists() and not path.stat().st_mode & 0o200),
    }


def copy_finalized_tool(
    repo_root: str | Path,
    tool_id: str,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root)
    tool = _tool_record(root, tool_id)
    source = (root / tool["path"]).resolve()
    dest = Path(destination)
    if not dest.is_absolute():
        dest = (root / dest).resolve()
    else:
        dest = dest.resolve()

    if source == dest:
        raise ValueError("finalized tools are copy-only; destination cannot be the source path")
    if dest.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    source_hash = _normalized_file_sha256(source)
    dest_hash = _normalized_file_sha256(dest)
    return {
        "schema_version": "truevision_finalized_tool_copy_receipt_v1",
        "tool_id": tool_id,
        "source": str(source),
        "destination": str(dest),
        "status": "copied",
        "source_normalized_sha256": source_hash,
        "destination_normalized_sha256": dest_hash,
        "hash_matches_source": source_hash == dest_hash,
        "copy_only": True,
        "promotion_policy": "promote_only_as_preset_after_review",
    }
