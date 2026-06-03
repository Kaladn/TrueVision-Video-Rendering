#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool_harness.receipt_writer import build_harness_receipt, write_json
from tool_harness.tool_selector import load_tool_catalog, select_tools_for_scene


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_id(value: str | None, fallback: str = "truevision_tool_harness") -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "")).strip("_")
    return safe or fallback


def build_invocation_plan(scene_contract: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(selection["selected_tools"], start=1):
        steps.append(
            {
                "step_id": f"step_{index:03d}",
                "tool_id": item["tool_id"],
                "invoke_now": False,
                "mode": "planned_not_executed",
                "why": item.get("why", []),
                "timing": item.get("timing", {}),
                "strength": item.get("strength", {}),
                "applies_to": item.get("applies_to", []),
                "safe_for_sc_call": item.get("safe_for_sc_call"),
                "expected_outputs": item.get("output_types", []),
                "state_direction": item.get("state_direction", {}),
                "state_language": item.get("state_language", {}),
                "required_state_stages": item.get("required_state_stages", []),
                "behavior_profiles_supported": item.get("behavior_profiles_supported", []),
                "forward_inputs": item.get("forward_inputs", []),
                "reverse_inputs": item.get("reverse_inputs", []),
                "state_outputs": item.get("state_outputs", []),
                "media_outputs_optional": item.get("media_outputs_optional", []),
            }
        )
    return {
        "schema_version": "truevision_tool_invocation_plan_v1",
        "scene_id": selection["scene_id"],
        "mode": "planning_only",
        "visual_goal": scene_contract.get("visual_goal", ""),
        "tool_steps": steps,
        "boundary": {
            "tools_invoked": False,
            "render_started": False,
            "external_services_called": False,
        },
    }


def run_harness(*, scene_contract_path: str | Path, catalog_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    scene_path = Path(scene_contract_path)
    out = Path(output_dir)
    scene_contract = _read_json(scene_path)
    catalog = load_tool_catalog(catalog_path)
    selection = select_tools_for_scene(scene_contract, catalog)
    run_id = _safe_id(str(scene_contract.get("run_id") or scene_contract.get("scene_id") or "harness_run"))
    out.mkdir(parents=True, exist_ok=True)

    selected_path = out / "selected_tools.json"
    rejected_path = out / "rejected_tools.json"
    plan_path = out / "tool_invocation_plan.json"
    receipt_path = out / "harness_receipt.json"

    plan = build_invocation_plan(scene_contract, selection)
    write_json(selected_path, {"scene_id": selection["scene_id"], "selected_tools": selection["selected_tools"]})
    write_json(rejected_path, {"scene_id": selection["scene_id"], "rejected_tools": selection["rejected_tools"]})
    write_json(plan_path, plan)
    outputs = {
        "selected_tools_json": str(selected_path),
        "rejected_tools_json": str(rejected_path),
        "tool_invocation_plan_json": str(plan_path),
        "harness_receipt_json": str(receipt_path),
    }
    receipt = build_harness_receipt(
        run_id=run_id,
        scene_id=selection["scene_id"],
        selected_count=len(selection["selected_tools"]),
        rejected_count=len(selection["rejected_tools"]),
        output_paths=outputs,
        truevideo_allowed=bool((scene_contract.get("approval") or {}).get("allow_truevideo", False)),
    )
    write_json(receipt_path, receipt)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan TrueVision tool use from a scene/effect contract.")
    parser.add_argument("--scene-contract", required=True)
    parser.add_argument("--catalog", default=str(ROOT / "tool_drop" / "TRUEVISION_TOOL_DROP_CATALOG.json"))
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_harness(scene_contract_path=args.scene_contract, catalog_path=args.catalog, output_dir=args.out)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
