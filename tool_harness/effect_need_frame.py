from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def normalize_token(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")


@dataclass(frozen=True)
class EffectNeedFrame:
    scene_id: str
    operation_direction: str
    moment: str
    visual_goal: str
    state_needs: tuple[str, ...]
    forbidden: tuple[str, ...]
    environment_tokens: tuple[str, ...]
    motion_tokens: tuple[str, ...]
    start_seconds: float
    end_seconds: float
    peak_seconds: float
    allow_truevideo: bool
    allow_render_execution: bool

    @property
    def duration_seconds(self) -> float:
        return max(0.001, self.end_seconds - self.start_seconds)


def build_effect_need_frame(scene_contract: dict[str, Any]) -> EffectNeedFrame:
    timing = scene_contract.get("timing") if isinstance(scene_contract.get("timing"), dict) else {}
    approval = scene_contract.get("approval") if isinstance(scene_contract.get("approval"), dict) else {}
    environment = scene_contract.get("environment") if isinstance(scene_contract.get("environment"), dict) else {}
    motion = scene_contract.get("motion_pressure") if isinstance(scene_contract.get("motion_pressure"), dict) else {}
    end = float(timing.get("end_seconds") or scene_contract.get("duration_seconds") or 10.0)
    start = float(timing.get("start_seconds") or 0.0)
    peak = float(timing.get("peak_seconds") or (start + (end - start) * 0.45))
    return EffectNeedFrame(
        scene_id=str(scene_contract.get("scene_id") or "scene"),
        operation_direction=normalize_token(str(scene_contract.get("operation_direction") or "forward_observation")),
        moment=str(scene_contract.get("moment") or ""),
        visual_goal=str(scene_contract.get("visual_goal") or ""),
        state_needs=tuple(normalize_token(item) for item in _as_list(scene_contract.get("state_needs"))),
        forbidden=tuple(normalize_token(item) for item in _as_list(scene_contract.get("forbidden"))),
        environment_tokens=tuple(normalize_token(value) for value in environment.values()),
        motion_tokens=tuple(normalize_token(value) for value in motion.values()),
        start_seconds=start,
        end_seconds=max(start + 0.001, end),
        peak_seconds=max(start, min(max(start + 0.001, end), peak)),
        allow_truevideo=bool(approval.get("allow_truevideo", False)),
        allow_render_execution=bool(approval.get("allow_render_execution", False)),
    )


def text_haystack(frame: EffectNeedFrame) -> str:
    return " ".join(
        [
            normalize_token(frame.visual_goal),
            normalize_token(frame.moment),
            *frame.state_needs,
            *frame.environment_tokens,
            *frame.motion_tokens,
        ]
    )
