import tempfile
import unittest
from pathlib import Path

from truevision_studio_server import (
    build_recording_command_from_request,
    build_downstream_payload,
    handle_assistant_message,
    list_storage_files,
    normalize_provider,
    resolve_assistant_actions,
    validate_local_endpoint,
    write_json_artifact,
)


class TrueVisionStudioServerTests(unittest.TestCase):
    def test_validate_local_endpoint_allows_loopback_only(self):
        self.assertEqual(
            validate_local_endpoint("http://127.0.0.1:11434/api/chat"),
            "http://127.0.0.1:11434/api/chat",
        )
        self.assertEqual(
            validate_local_endpoint("http://localhost:11434/v1/chat/completions"),
            "http://localhost:11434/v1/chat/completions",
        )
        with self.assertRaises(ValueError):
            validate_local_endpoint("https://example.com/api/chat")

    def test_normalize_provider_defaults_to_ollama(self):
        self.assertEqual(normalize_provider(""), "ollama_native")
        self.assertEqual(normalize_provider("ollama_native"), "ollama_native")
        self.assertEqual(normalize_provider("openai_compatible"), "openai_compatible")

    def test_build_downstream_payload_for_ollama(self):
        payload = build_downstream_payload(
            provider="ollama_native",
            model="qwen3-coder:30b",
            system_prompt="system",
            request={"prompt": "make state"},
        )

        self.assertEqual(payload["model"], "qwen3-coder:30b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn("options", payload)

    def test_build_downstream_payload_for_openai_compatible(self):
        payload = build_downstream_payload(
            provider="openai_compatible",
            model="qwen3-coder:30b",
            system_prompt="system",
            request={"prompt": "make state"},
        )

        self.assertEqual(payload["model"], "qwen3-coder:30b")
        self.assertFalse(payload["stream"])
        self.assertIn("temperature", payload)
        self.assertNotIn("options", payload)

    def test_write_json_artifact_persists_to_storage_lane(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = write_json_artifact(
                storage_root=Path(tmpdir),
                lane="outbox",
                prefix="state_request",
                payload={"request_kind": "truevision_state_media_draft"},
            )

            self.assertEqual(artifact["lane"], "outbox")
            self.assertTrue(Path(artifact["path"]).exists())
            self.assertTrue(artifact["sha256"].startswith("sha256:"))

    def test_list_storage_files_reports_saved_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_json_artifact(
                storage_root=Path(tmpdir),
                lane="manifests",
                prefix="state_plan",
                payload={"ok": True},
            )

            files = list_storage_files(Path(tmpdir))

            self.assertEqual(len(files), 1)
            self.assertEqual(files[0]["lane"], "manifests")
            self.assertEqual(files[0]["kind"], "json")

    def test_build_recording_command_from_request_uses_manual_minutes(self):
        request = {
            "capture_shape": {
                "duration_minutes": 5,
                "fps": 15,
                "grid_width": 160,
                "grid_height": 90,
                "resolution_width": 960,
                "resolution_height": 540,
            },
            "record_start_zone": {
                "selected_region": [10, 20, 640, 400],
                "snapped_region": [10, 40, 640, 360],
                "monitor": 0,
            },
        }

        result = build_recording_command_from_request(request, storage_root=Path("D:/tv_storage"))

        self.assertEqual(result["duration_seconds"], 300)
        self.assertIn("--duration", result["command"])
        self.assertIn("300", result["command"])
        self.assertIn("--region", result["command"])
        self.assertIn("10,40,640,360", result["command"])
        self.assertIn("--grid", result["command"])
        self.assertIn("160x90", result["command"])

    def test_resolve_assistant_actions_turns_chat_into_work(self):
        request = {"local_llm": {"enabled": True}}

        actions = resolve_assistant_actions("compile with qwen then prepare the record command", request)

        self.assertIn("save_request", actions)
        self.assertIn("qwen_compile", actions)
        self.assertIn("prepare_record", actions)

    def test_handle_assistant_message_executes_storage_and_record_actions(self):
        request = {
            "request_kind": "truevision_state_media_draft",
            "local_llm": {"enabled": False},
            "capture_shape": {
                "duration_minutes": 0.25,
                "fps": 9,
                "grid_width": 160,
                "grid_height": 90,
                "resolution_width": 960,
                "resolution_height": 540,
            },
            "record_start_zone": {
                "selected_region": [0, 0, 960, 540],
                "snapped_region": [0, 0, 960, 540],
                "monitor": 0,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = handle_assistant_message(
                {
                    "message": "save this and prepare record",
                    "request": request,
                },
                storage_root=Path(tmpdir),
            )

            self.assertTrue(result["ok"])
            self.assertIn("save_request", result["actions"])
            self.assertIn("prepare_record", result["actions"])
            self.assertIn("recording", result["results"])
            self.assertEqual(result["results"]["recording"]["duration_seconds"], 15)
            self.assertGreaterEqual(len(result["files"]), 2)


if __name__ == "__main__":
    unittest.main()
