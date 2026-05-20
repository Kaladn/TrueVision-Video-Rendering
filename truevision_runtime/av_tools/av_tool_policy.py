from __future__ import annotations

from pathlib import Path
from typing import Any

from .av_tool_registry import get_av_tool


class AVToolPolicyError(ValueError):
    """Raised when a requested AV tool call violates local policy."""


SAFE_TEMPLATE_TOOLS = {
    "template_load",
    "template_save",
    "template_patch",
    "template_create_variant",
    "template_from_audio_signals",
    "template_delete",
}
SAFE_MEDIA_LANES = {"artifacts", "manifests", "reports", "templates", "receipts", "events", "library"}


def safe_flat_json_name(name: str) -> str:
    filename = Path(str(name)).name
    if filename != str(name):
        raise AVToolPolicyError("template names must be flat filenames")
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    stem = filename[:-5]
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem).strip("_")
    if not clean or clean != stem:
        raise AVToolPolicyError("template name contains unsafe characters")
    return f"{clean}.json"


def validate_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise AVToolPolicyError("tool call must be an object")
    tool_name = str(call.get("tool") or "")
    tool = get_av_tool(tool_name)
    if tool is None:
        raise AVToolPolicyError(f"unknown AV tool: {tool_name}")
    args = call.get("args", {})
    if not isinstance(args, dict):
        raise AVToolPolicyError("tool args must be an object")
    if tool.approval_required and not bool(call.get("human_confirmed")):
        raise AVToolPolicyError(f"{tool_name} requires human_confirmed=true")
    if tool_name in SAFE_TEMPLATE_TOOLS and args.get("name"):
        args = {**args, "name": safe_flat_json_name(str(args["name"]))}
    if tool_name == "storage_list_artifacts" and args.get("lane") not in {None, *SAFE_MEDIA_LANES}:
        raise AVToolPolicyError("storage listing is limited to AV media lanes")
    return {
        "tool": tool_name,
        "args": args,
        "human_confirmed": bool(call.get("human_confirmed", False)),
        "requested_by": str(call.get("requested_by") or "qwen_or_operator")[:80],
    }
