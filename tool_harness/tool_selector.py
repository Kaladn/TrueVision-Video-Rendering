from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .causality_gate import evaluate_causality_gate
from .effect_need_frame import EffectNeedFrame, build_effect_need_frame, text_haystack
from .timing_planner import plan_strength, plan_timing
from truevision_runtime.state_language import build_state_language, required_stages_for_direction, supports_any_stage


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EFFECT_POLICY = ROOT / "tool_harness" / "policies" / "effect_selection_policy.json"
DEFAULT_TRUEVIDEO_POLICY = ROOT / "tool_harness" / "policies" / "truevideo_usage_policy.json"


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_tool_catalog(catalog_path: str | Path) -> dict[str, Any]:
    catalog_path = Path(catalog_path)
    catalog = _read_json(catalog_path)
    root = catalog_path.parent
    tools: dict[str, dict[str, Any]] = {}
    for row in catalog.get("tools", []):
        manifest_path = root / str(row["manifest"])
        manifest = _read_json(manifest_path)
        manifest["_manifest_path"] = str(manifest_path)
        tools[str(manifest["tool_id"])] = manifest
    return {
        "catalog": catalog,
        "catalog_path": str(catalog_path),
        "tool_drop_root": str(root),
        "tools": tools,
    }


def _effect_policy(path: str | Path | None = None) -> dict[str, Any]:
    return _read_json(path or DEFAULT_EFFECT_POLICY)


def _truevideo_policy(path: str | Path | None = None) -> dict[str, Any]:
    return _read_json(path or DEFAULT_TRUEVIDEO_POLICY)


def _keyword_hit(haystack: str, keywords: list[str]) -> bool:
    return any(str(keyword).lower() in haystack for keyword in keywords)


def _state_direction(tool: dict[str, Any]) -> dict[str, Any]:
    observes = bool(tool.get("observes_state", False))
    abstracts = bool(tool.get("abstracts_behavior", False))
    generates = bool(tool.get("generates_state", False))
    renders = bool(tool.get("renders_media", tool.get("starts_render", False)))
    copies = bool(tool.get("copies_source_media", False))
    roles: list[str] = []
    if observes and abstracts and not renders:
        roles.append("state_behavior_observer")
    if generates:
        roles.append("state_generator")
    if renders:
        roles.append("state_render_surface")
    if not roles:
        roles.append("contract_or_planning_surface")
    return {
        "observes_state": observes,
        "abstracts_behavior": abstracts,
        "generates_state": generates,
        "renders_media": renders,
        "copies_source_media": copies,
        "role": roles,
    }


def _score_tool(tool_id: str, tool: dict[str, Any], frame: EffectNeedFrame, policy: dict[str, Any]) -> tuple[float, list[str], str]:
    haystack = text_haystack(frame)
    tool_rules = policy.get("tool_rules", {}).get(tool_id, {})
    score = 0.0
    why: list[str] = []
    emphasis = str(tool_rules.get("emphasis") or "default")
    state_language = build_state_language(tool)

    for need, rule in policy.get("need_rules", {}).items():
        keywords = [str(item) for item in rule.get("keywords", [])]
        if not _keyword_hit(haystack, keywords):
            continue
        if tool_id in rule.get("preferred_tools", []):
            score += float(rule.get("weight", 0.2))
            why.append(str(rule.get("reason") or f"matches {need} need"))
            if rule.get("emphasis"):
                emphasis = str(rule["emphasis"])
        if tool_id in rule.get("rejected_tools", []):
            score -= float(rule.get("penalty", 0.35))
            why.append(str(rule.get("reject_reason") or f"rejected for {need} need"))

    for keyword in tool_rules.get("positive_keywords", []):
        if str(keyword).lower() in haystack:
            score += 0.12
            why.append(f"scene text supports {keyword}")

    for forbidden in frame.forbidden:
        for keyword in tool_rules.get("forbidden_keywords", []):
            if str(keyword).lower() in forbidden:
                score -= 0.45
                why.append(f"forbidden instruction rejects {keyword}")

    if str(tool.get("status")) == "active_callable":
        score += 0.08
    if bool(tool.get("source_truth_compliant")):
        score += 0.04
    required_stages = required_stages_for_direction(frame.operation_direction)
    if frame.operation_direction == "reverse_generation" and supports_any_stage(state_language, required_stages):
        score += 0.18
        why.append("reverse generation needs plan/replay/surface capability")
    elif frame.operation_direction == "forward_observation" and supports_any_stage(state_language, required_stages):
        score += 0.08
        why.append("forward observation needs witness/profile capability")

    return max(-1.0, min(1.0, score)), why, emphasis


def _truevideo_candidate(frame: EffectNeedFrame, policy: dict[str, Any]) -> dict[str, Any]:
    haystack = text_haystack(frame)
    allowed_keywords = [str(item).lower() for item in policy.get("allowed_when_keywords", [])]
    not_allowed_keywords = [str(item).lower() for item in policy.get("not_allowed_when_keywords", [])]
    desire = any(keyword in haystack for keyword in allowed_keywords)
    simple_effect = any(keyword in haystack for keyword in not_allowed_keywords)
    if frame.allow_truevideo and desire and not simple_effect:
        score = 0.82
        return {
            "tool_id": "truevideo_lifelike_scene_generator",
            "score": score,
            "why": ["scene approval allows lifelike generation", "scene need matches TrueVideo usage policy"],
            "timing": plan_timing(frame, emphasis="default"),
            "strength": plan_strength(score, emphasis="default"),
            "safe_for_sc_call": "requires_model_call_gate_and_operator_approval",
            "output_types": ["generated_state", "generated_media"],
            "state_direction": {
                "observes_state": False,
                "abstracts_behavior": False,
                "generates_state": True,
                "renders_media": True,
                "copies_source_media": False,
                "role": ["state_generator", "state_render_surface"],
            },
            "state_language": {
                "behavior_family": "lifelike_scene_state",
                "can_witness": False,
                "can_profile": False,
                "can_plan": True,
                "can_replay": True,
                "can_surface": True,
                "supported_stages": ["plan", "replay", "surface"],
                "copies_source_media": False,
                "raw_media_saved": False,
                "media_is_optional_surface": True,
                "media_is_source_truth": False,
                "source_truth_compliant": False,
            },
            "required_state_stages": list(required_stages_for_direction(frame.operation_direction)),
            "behavior_profiles_supported": ["lifelike_scene_state"],
            "forward_inputs": [],
            "reverse_inputs": ["scene_contract", "state_behavior_profile"],
            "state_outputs": ["generated_state"],
            "media_outputs_optional": ["generated_media"],
            "invoke_now": False,
        }
    reasons = []
    if not frame.allow_truevideo:
        reasons.append("TrueVideo/lifelike generation is forbidden by scene approval")
    if not desire:
        reasons.append("scene does not require lifelike actor/world synthesis")
    if simple_effect:
        reasons.append("state/effect tools are preferred for this need")
    return {
        "tool_id": "truevideo_lifelike_scene_generator",
        "score": 0.0,
        "rejected": True,
        "why": reasons or ["TrueVideo policy did not pass"],
    }


def select_tools_for_scene(
    scene_contract: dict[str, Any],
    catalog_bundle: dict[str, Any],
    *,
    effect_policy: dict[str, Any] | None = None,
    truevideo_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = effect_policy or _effect_policy()
    truevideo = truevideo_policy or _truevideo_policy()
    frame = build_effect_need_frame(scene_contract)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    min_score = float(policy.get("minimum_select_score", 0.34))

    for tool_id, tool in sorted(catalog_bundle["tools"].items()):
        score, why, emphasis = _score_tool(tool_id, tool, frame, policy)
        gate = evaluate_causality_gate(tool, frame)
        state_language = build_state_language(tool)
        required_stages = list(required_stages_for_direction(frame.operation_direction))
        if not gate["allowed"]:
            if score > 0.0 or tool_id in policy.get("always_report_rejections", []):
                rejected.append(
                    {
                        "tool_id": tool_id,
                        "score": round(max(0.0, score), 3),
                        "rejected": True,
                        "why": [gate["reason"], *why],
                    }
                )
            continue
        if score >= min_score:
            selected.append(
                {
                    "tool_id": tool_id,
                    "score": round(score, 3),
                    "why": why or ["scene state matches tool policy"],
                    "timing": plan_timing(frame, emphasis=emphasis),
                    "strength": plan_strength(score, emphasis=emphasis),
                    "applies_to": list(frame.state_needs),
                    "safe_for_sc_call": tool.get("safe_for_sc_call"),
                    "output_types": tool.get("output_types", []),
                    "state_direction": _state_direction(tool),
                    "state_language": state_language,
                    "required_state_stages": required_stages,
                    "behavior_profiles_supported": tool.get("behavior_profiles_supported", []),
                    "forward_inputs": tool.get("forward_inputs", []),
                    "reverse_inputs": tool.get("reverse_inputs", []),
                    "state_outputs": tool.get("state_outputs", []),
                    "media_outputs_optional": tool.get("media_outputs_optional", []),
                    "invoke_now": False,
                }
            )
        elif tool_id in policy.get("always_report_rejections", []) or score < 0.0:
            rejected.append(
                {
                    "tool_id": tool_id,
                    "score": round(max(0.0, score), 3),
                    "rejected": True,
                    "why": why or ["tool did not fit current scene state"],
                }
            )

    tv = _truevideo_candidate(frame, truevideo)
    if tv.get("rejected"):
        rejected.append(tv)
    else:
        selected.append(tv)

    selected.sort(key=lambda item: (-float(item["score"]), str(item["tool_id"])))
    rejected.sort(key=lambda item: (str(item["tool_id"])))
    return {
        "scene_id": frame.scene_id,
        "mode": "planning_only",
        "selected_tools": selected,
        "rejected_tools": rejected,
        "boundary": {
            "tools_invoked": False,
            "render_started": False,
            "external_services_called": False,
            "truevideo_called": False,
        },
    }
