import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool_harness.tool_selector import load_tool_catalog, select_tools_for_scene  # noqa: E402
from tool_harness.truevision_tool_harness import run_harness  # noqa: E402


def _scene_contract() -> dict:
    return {
        "scene_id": "phoenix_impact_transform_test",
        "moment": "impact_transform",
        "visual_goal": "destructive fire changes into multicolor life-energy and reveals a healed forest",
        "environment": {
            "location": "burned_forest",
            "visibility": "smoky",
            "depth": "deep_trail",
            "time": "twilight",
        },
        "motion_pressure": {
            "speed": "high",
            "impact": "massive",
            "camera": "flyover_to_ground_rest",
        },
        "state_needs": [
            "fire",
            "smoke",
            "shockwave",
            "color_transition",
            "fog_reveal",
            "growth_recovery",
            "depth_haze",
        ],
        "forbidden": [
            "random overlays",
            "unmotivated lasers",
            "flat sticker effects",
        ],
        "timing": {
            "start_seconds": 0.0,
            "end_seconds": 22.0,
            "peak_seconds": 6.0,
        },
        "approval": {
            "allow_truevideo": False,
            "allow_render_execution": False,
        },
    }


class TrueVisionToolHarnessTests(unittest.TestCase):
    def test_load_tool_catalog_reads_manifests(self):
        catalog = load_tool_catalog(ROOT / "tool_drop" / "TRUEVISION_TOOL_DROP_CATALOG.json")

        self.assertIn("truevision_state_video_watcher", catalog["tools"])
        self.assertIn("meter_grid_from_capture", catalog["tools"])
        self.assertFalse(catalog["tools"]["truevision_state_video_watcher"]["implementation_owned_by_tool_drop"])
        self.assertEqual(catalog["catalog"]["implementation_policy"], "catalog_only_no_code_moves")

    def test_selector_chooses_tools_by_scene_need_and_rejects_bad_fit(self):
        catalog = load_tool_catalog(ROOT / "tool_drop" / "TRUEVISION_TOOL_DROP_CATALOG.json")
        result = select_tools_for_scene(_scene_contract(), catalog)

        selected_ids = [item["tool_id"] for item in result["selected_tools"]]
        rejected_ids = [item["tool_id"] for item in result["rejected_tools"]]

        self.assertIn("atmosphere_profile_from_capture", selected_ids)
        self.assertIn("element_creation_profile_from_capture", selected_ids)
        self.assertIn("meter_grid_from_capture", selected_ids)
        self.assertIn("render_trudepth_rave_laser_sample", rejected_ids)
        self.assertIn("truevideo_lifelike_scene_generator", rejected_ids)
        self.assertTrue(all(item["score"] >= 0.0 for item in result["selected_tools"]))
        for item in result["selected_tools"]:
            self.assertIn("why", item)
            self.assertIn("timing", item)
            self.assertIn("strength", item)
            self.assertIn("state_direction", item)
            self.assertIn("state_language", item)
            self.assertIn("behavior_profiles_supported", item)
        profile_item = next(item for item in result["selected_tools"] if item["tool_id"] == "atmosphere_profile_from_capture")
        self.assertEqual(profile_item["state_language"]["behavior_family"], "fog_reveal")
        self.assertTrue(profile_item["state_language"]["can_witness"])
        self.assertTrue(profile_item["state_language"]["can_profile"])
        self.assertEqual(profile_item["required_state_stages"], ["witness", "profile"])

    def test_selector_can_plan_reverse_state_generation_without_copying_media(self):
        contract = _scene_contract()
        contract["operation_direction"] = "reverse_generation"
        contract["approval"]["allow_render_execution"] = True
        catalog = load_tool_catalog(ROOT / "tool_drop" / "TRUEVISION_TOOL_DROP_CATALOG.json")

        result = select_tools_for_scene(contract, catalog)
        selected_ids = [item["tool_id"] for item in result["selected_tools"]]

        self.assertIn("render_truedepth_fog_reveal_samples", selected_ids)
        for item in result["selected_tools"]:
            self.assertFalse(item["state_direction"]["copies_source_media"])
        render_item = next(item for item in result["selected_tools"] if item["tool_id"] == "render_truedepth_fog_reveal_samples")
        self.assertTrue(render_item["state_language"]["can_plan"])
        self.assertTrue(render_item["state_language"]["can_replay"])
        self.assertTrue(render_item["state_language"]["can_surface"])
        self.assertEqual(render_item["required_state_stages"], ["plan", "replay", "surface"])

    def test_selector_language_uses_state_loop_terms_not_media_first_terms(self):
        catalog = load_tool_catalog(ROOT / "tool_drop" / "TRUEVISION_TOOL_DROP_CATALOG.json")
        result = select_tools_for_scene(_scene_contract(), catalog)
        joined_why = " ".join(
            reason
            for item in result["selected_tools"]
            for reason in item.get("why", [])
        ).lower()

        self.assertIn("witness/profile", joined_why)
        self.assertNotIn("record video", joined_why)
        self.assertNotIn("copy media", joined_why)
        self.assertNotIn("render output capability", joined_why)

    def test_selector_chooses_document_state_movie_for_page_visual_state(self):
        catalog = load_tool_catalog(ROOT / "tool_drop" / "TRUEVISION_TOOL_DROP_CATALOG.json")
        result = select_tools_for_scene(
            {
                "scene_id": "document_page_state_proof",
                "visual_goal": "witness a document page as visual_state and replay glyph marks from state",
                "state_needs": ["document", "page", "glyph", "document_state_movie"],
                "operation_direction": "forward_observation",
                "approval": {
                    "allow_truevideo": False,
                    "allow_render_execution": False,
                },
            },
            catalog,
        )

        selected = {item["tool_id"]: item for item in result["selected_tools"]}
        self.assertIn("truevision_document_state_movie", selected)
        tool = selected["truevision_document_state_movie"]
        self.assertEqual(tool["state_language"]["behavior_family"], "document_visual_state")
        self.assertTrue(tool["state_language"]["can_witness"])
        self.assertTrue(tool["state_language"]["can_profile"])
        self.assertTrue(tool["state_language"]["can_replay"])
        self.assertTrue(tool["state_language"]["can_surface"])
        self.assertFalse(tool["state_direction"]["copies_source_media"])
        self.assertEqual(tool["required_state_stages"], ["witness", "profile"])

    def test_harness_writes_plans_and_receipt_without_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scene_path = tmp_path / "scene_contract.json"
            out_dir = tmp_path / "harness_run"
            scene_path.write_text(json.dumps(_scene_contract(), indent=2), encoding="utf-8")

            result = run_harness(
                scene_contract_path=scene_path,
                catalog_path=ROOT / "tool_drop" / "TRUEVISION_TOOL_DROP_CATALOG.json",
                output_dir=out_dir,
            )

            expected = {
                "selected_tools_json",
                "rejected_tools_json",
                "tool_invocation_plan_json",
                "harness_receipt_json",
            }
            self.assertTrue(expected.issubset(result.keys()))
            for path in result.values():
                self.assertTrue(Path(path).exists(), path)

            receipt = json.loads(Path(result["harness_receipt_json"]).read_text(encoding="utf-8"))
            plan = json.loads(Path(result["tool_invocation_plan_json"]).read_text(encoding="utf-8"))

        self.assertEqual(receipt["schema_version"], "truevision_tool_harness_receipt_v1")
        self.assertFalse(receipt["boundary"]["render_started"])
        self.assertFalse(receipt["boundary"]["external_services_called"])
        self.assertFalse(receipt["boundary"]["tools_invoked"])
        self.assertEqual(plan["mode"], "planning_only")
        self.assertGreaterEqual(len(plan["tool_steps"]), 1)
        self.assertTrue(all(step["invoke_now"] is False for step in plan["tool_steps"]))
        self.assertTrue(all("state_direction" in step for step in plan["tool_steps"]))
        self.assertTrue(all("state_language" in step for step in plan["tool_steps"]))
        self.assertTrue(all("required_state_stages" in step for step in plan["tool_steps"]))

    def test_cli_writes_harness_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scene_path = tmp_path / "scene_contract.json"
            out_dir = tmp_path / "harness_run"
            scene_path.write_text(json.dumps(_scene_contract(), indent=2), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tool_harness" / "truevision_tool_harness.py"),
                    "--scene-contract",
                    str(scene_path),
                    "--catalog",
                    str(ROOT / "tool_drop" / "TRUEVISION_TOOL_DROP_CATALOG.json"),
                    "--out",
                    str(out_dir),
                ],
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertTrue(Path(payload["harness_receipt_json"]).exists())


if __name__ == "__main__":
    unittest.main()
