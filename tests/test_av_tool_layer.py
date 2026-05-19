import json
import tempfile
import unittest
from pathlib import Path

from truevision_runtime.av_tools.av_tool_policy import AVToolPolicyError, validate_tool_call
from truevision_runtime.av_tools.av_tool_registry import list_av_tools
from truevision_runtime.av_tools.av_tool_runner import run_av_tool_call


class AVToolLayerTests(unittest.TestCase):
    def test_registry_is_audio_video_only(self):
        tools = list_av_tools()
        names = {tool["name"] for tool in tools}

        self.assertIn("audio_probe_duration", names)
        self.assertIn("video_render_preview", names)
        self.assertIn("template_patch", names)
        self.assertNotIn("filesystem_delete", names)
        self.assertNotIn("browser_open", names)
        self.assertNotIn("security_enforce", names)
        self.assertTrue(all(tool["domain"] == "audio_video" for tool in tools))

    def test_policy_rejects_unknown_and_path_escape(self):
        with self.assertRaises(AVToolPolicyError):
            validate_tool_call({"tool": "filesystem_delete", "args": {"path": "D:/"}})

        with self.assertRaises(AVToolPolicyError):
            validate_tool_call({"tool": "template_load", "args": {"name": "../README.md"}})

    def test_policy_requires_confirmation_for_delete_and_execute(self):
        with self.assertRaises(AVToolPolicyError):
            validate_tool_call({"tool": "template_delete", "args": {"name": "edge.json"}})

        with self.assertRaises(AVToolPolicyError):
            validate_tool_call({"tool": "video_execute_full_render", "args": {"job_id": "edge"}})

        validated = validate_tool_call(
            {
                "tool": "video_execute_full_render",
                "args": {"job_id": "edge"},
                "human_confirmed": True,
            }
        )
        self.assertEqual(validated["tool"], "video_execute_full_render")

    def test_template_tools_and_receipts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            created = run_av_tool_call(
                {
                    "tool": "template_create",
                    "args": {
                        "name": "Edge River",
                        "prompt": "thin river on black",
                        "renderer": "edge_audio_river",
                        "duration_seconds": 278.32,
                        "fps": 30,
                    },
                },
                storage_root=storage,
            )
            self.assertTrue(created["ok"])
            self.assertEqual(created["result"]["template"]["timeline"]["frame_count"], 8350)
            self.assertTrue(Path(created["receipt"]["path"]).exists())

            saved = run_av_tool_call(
                {
                    "tool": "template_save",
                    "args": {
                        "name": "edge_river.json",
                        "template": created["result"]["template"],
                    },
                },
                storage_root=storage,
            )
            self.assertTrue(saved["ok"])
            self.assertEqual(saved["result"]["name"], "edge_river.json")

            patched = run_av_tool_call(
                {
                    "tool": "template_patch",
                    "args": {
                        "name": "edge_river.json",
                        "json_path": "visual_parameters.geometry.river_height_ratio",
                        "value": 0.28,
                        "reason": "river too thick at chorus",
                    },
                },
                storage_root=storage,
            )
            self.assertTrue(patched["ok"])
            self.assertEqual(
                patched["result"]["template"]["visual_parameters"]["geometry"]["river_height_ratio"],
                0.28,
            )

            receipt_files = list((storage / "receipts").glob("*.json"))
            self.assertEqual(len(receipt_files), 3)

    def test_markers_recalibration_and_render_prepare_are_structured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            marker = run_av_tool_call(
                {
                    "tool": "time_marker_add",
                    "args": {
                        "template_id": "edge_river",
                        "source_artifact": "edge.mp4",
                        "time_seconds": 72,
                        "note": "river too thick",
                        "target": "river_height_ratio",
                        "direction": "decrease",
                    },
                },
                storage_root=storage,
            )
            self.assertTrue(marker["ok"])
            self.assertEqual(marker["result"]["marker"]["time_seconds"], 72)

            note = run_av_tool_call(
                {
                    "tool": "recalibration_add_note",
                    "args": {
                        "template_id": "edge_river",
                        "source_artifact": "edge.mp4",
                        "time_seconds": 204,
                        "note": "colors should calm down",
                    },
                },
                storage_root=storage,
            )
            self.assertTrue(note["ok"])

            prepare = run_av_tool_call(
                {
                    "tool": "video_prepare_full_render",
                    "args": {
                        "template": {
                            "name": "Edge River",
                            "renderer": "edge_audio_river",
                            "timeline": {"duration_seconds": 278.32, "fps": 30},
                        }
                    },
                },
                storage_root=storage,
            )
            self.assertTrue(prepare["ok"])
            self.assertEqual(prepare["result"]["job"]["status"], "prepared_requires_human_execute")
            self.assertTrue(Path(prepare["result"]["manifest"]["path"]).exists())

            event_lines = (storage / "events" / "av_recalibration.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(event_lines), 2)
            self.assertEqual(json.loads(event_lines[0])["kind"], "time_marker")


if __name__ == "__main__":
    unittest.main()
