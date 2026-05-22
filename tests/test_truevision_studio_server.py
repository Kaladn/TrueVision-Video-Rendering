import tempfile
import unittest
from pathlib import Path

from truevision_studio_server import (
    append_chat_message,
    build_generation_template_from_request,
    build_recording_command_from_request,
    build_downstream_payload,
    delete_template,
    handle_assistant_message,
    list_render_presets,
    list_storage_files,
    list_templates,
    normalize_provider,
    read_chat_log,
    resolve_storage_root,
    ROOT,
    resolve_assistant_actions,
    run_av_tool_call,
    save_template,
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

    def test_resolve_storage_root_accepts_external_absolute_root(self):
        self.assertEqual(
            resolve_storage_root(r"E:\TruEVision Generation"),
            Path(r"E:\TruEVision Generation").resolve(),
        )

    def test_resolve_storage_root_resolves_relative_to_project(self):
        self.assertEqual(resolve_storage_root("storage_alt"), (ROOT / "storage_alt").resolve())

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

    def test_append_chat_message_uses_one_flat_file_per_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = append_chat_message(
                storage_root=Path(tmpdir),
                message={"source": "operator", "text": "talk before build"},
                day="2026-05-19",
            )
            second = append_chat_message(
                storage_root=Path(tmpdir),
                message={"source": "Qwen", "text": "plan first"},
                day="2026-05-19",
            )

            self.assertEqual(first["path"], second["path"])
            self.assertEqual(Path(first["path"]).name, "2026-05-19.jsonl")
            entries = read_chat_log(storage_root=Path(tmpdir), day="2026-05-19")
            self.assertEqual([entry["source"] for entry in entries], ["operator", "Qwen"])
            self.assertEqual(entries[0]["text"], "talk before build")

    def test_save_list_and_delete_templates_are_flat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_template(
                storage_root=Path(tmpdir),
                template={
                    "name": "Edge River",
                    "renderer": "edge_audio_river",
                    "timeline": {"duration_seconds": 278.32},
                },
            )

            self.assertEqual(saved["lane"], "templates")
            self.assertTrue(saved["name"].endswith(".json"))
            templates = list_templates(Path(tmpdir))
            self.assertEqual(len(templates), 1)
            self.assertEqual(templates[0]["template"]["name"], "Edge River")

            deleted = delete_template(storage_root=Path(tmpdir), name=saved["name"])

            self.assertTrue(deleted["deleted"])
            self.assertEqual(list_templates(Path(tmpdir)), [])

    def test_build_generation_template_matches_audio_duration_when_present(self):
        request = {
            "prompt": "river of colors in black",
            "renderer": {"name": "edge_audio_river"},
            "media": {
                "audio_path": "D:/music/edge.mp3",
                "audio_duration_seconds": 278.32,
                "sync_to_audio": True,
            },
            "capture_shape": {
                "fps": 30,
                "duration_seconds": 120,
            },
            "qwen_state_plan": {"scene": {"name": "wake river"}},
        }

        template = build_generation_template_from_request(request)

        self.assertEqual(template["renderer"], "edge_audio_river")
        self.assertEqual(template["timeline"]["duration_seconds"], 278.32)
        self.assertEqual(template["timeline"]["frame_count"], 8350)
        self.assertEqual(template["time_distance"]["source"], "audio_duration")
        self.assertEqual(template["state_plan"]["scene"]["name"], "wake river")

    def test_av_tool_call_endpoint_runner_writes_receipt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_av_tool_call(
                {
                    "tool": "template_create",
                    "args": {
                        "name": "Edge River",
                        "prompt": "thin sound river",
                        "duration_seconds": 10,
                        "fps": 10,
                    },
                },
                storage_root=Path(tmpdir),
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["result"]["template"]["timeline"]["frame_count"], 100)
            self.assertTrue(Path(result["receipt"]["path"]).exists())

    def test_render_presets_are_available_to_studio_server(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            presets = list_render_presets(Path(tmpdir))

            preset_ids = {preset["preset_id"] for preset in presets}
            self.assertIn("glitch_444_alive_poster", preset_ids)
            self.assertIn("house_remix_audio_city", preset_ids)

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

    def test_resolve_assistant_actions_treats_visual_prompt_as_compile_work(self):
        request = {"local_llm": {"enabled": True}}

        actions = resolve_assistant_actions("A person walks through a field at sunset", request)

        self.assertIn("save_request", actions)
        self.assertIn("qwen_compile", actions)

    def test_resolve_assistant_actions_routes_plain_chat_to_qwen_chat(self):
        request = {"local_llm": {"enabled": True}}

        actions = resolve_assistant_actions("what can you help me do in this studio?", request)

        self.assertEqual(actions, ["qwen_chat"])

    def test_handle_assistant_message_queues_chat_without_storage_write(self):
        request = {
            "request_kind": "truevision_state_media_draft",
            "local_llm": {"enabled": True},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = handle_assistant_message(
                {
                    "message": "what can you help me do in this studio?",
                    "request": request,
                },
                storage_root=Path(tmpdir),
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["actions"], ["qwen_chat"])
            self.assertEqual(result["files"], [])

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
