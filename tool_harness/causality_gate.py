from __future__ import annotations

from typing import Any

from .effect_need_frame import EffectNeedFrame


def evaluate_causality_gate(tool: dict[str, Any], frame: EffectNeedFrame) -> dict[str, Any]:
    tool_id = str(tool.get("tool_id") or "")
    status = str(tool.get("status") or "")
    starts_render = bool(tool.get("starts_render"))

    if status == "parked_experimental":
        return {
            "allowed": False,
            "reason": "tool is parked experimental and not active for harness selection",
        }
    if starts_render and not frame.allow_render_execution:
        return {
            "allowed": False,
            "reason": "render execution is forbidden by scene approval",
        }
    if tool_id.startswith("truevideo") and not frame.allow_truevideo:
        return {
            "allowed": False,
            "reason": "TrueVideo/lifelike generation is forbidden by scene approval",
        }
    return {"allowed": True, "reason": "causality and approval gates passed"}

