import json
import unittest

from truevision_runtime.llm_adapter.prompt_context_builder import build_prompt_context
from truevision_runtime.llm_adapter.prompt_to_state_adapter import PromptToStateAdapter
from truevision_runtime.llm_adapter.schema_validator import validate_state_request


class PromptToStateAdapterTests(unittest.TestCase):
    def test_context_is_model_neutral_and_wav_aware(self):
        context = build_prompt_context(
            prompt="make a river video from this wav",
            project_context={"audio_path": "D:/music/song.wav"},
        )

        self.assertIn("model_endpoint", context)
        self.assertIn("schema", context)
        self.assertIn("allowed_av_tools", context)
        self.assertIn("state_pattern_library", context)
        self.assertIn("audio_extract_features", context["allowed_av_tools"])
        self.assertIn("audio_analyze_levels", context["allowed_av_tools"])
        self.assertIn("random_geometry_shards", {pattern["pattern_id"] for pattern in context["state_pattern_library"]})
        self.assertIn(".wav", "\n".join(context["runtime_notes"]))
        self.assertFalse(context["trust_boundary"]["model_output_is_trusted"])

    def test_validate_state_request_normalizes_minimal_wav_template(self):
        payload = {
            "request_kind": "truevision_state_media_draft",
            "scene": {"name": "thin color river"},
            "renderer": "edge_audio_river",
            "media": {"audio_path": "D:/music/song.wav", "sync_to_audio": True},
            "timeline": {"duration_seconds": 120, "fps": 12},
            "visual_parameters": {"river_height_ratio": 0.24},
            "safety_boundary": {"generated_state_media": True, "evidence": False},
        }

        result = validate_state_request(payload)

        self.assertTrue(result.ok)
        self.assertEqual(result.normalized["timeline"]["frame_count"], 1440)
        self.assertEqual(result.normalized["media"]["audio_path"], "D:/music/song.wav")

    def test_validator_rejects_evidence_claims_and_bad_timing(self):
        payload = {
            "request_kind": "truevision_state_media_draft",
            "scene": {"name": "bad"},
            "renderer": "edge_audio_river",
            "media": {"audio_path": "song.wav"},
            "timeline": {"duration_seconds": 0, "fps": 0},
            "safety_boundary": {"generated_state_media": True, "evidence": True},
        }

        result = validate_state_request(payload)

        self.assertFalse(result.ok)
        self.assertIn("timeline.duration_seconds must be > 0", result.errors)
        self.assertIn("safety_boundary.evidence must be false", result.errors)

    def test_adapter_repairs_invalid_draft_with_validation_errors_only(self):
        calls = []

        def fake_model(messages):
            calls.append(messages)
            if len(calls) == 1:
                return json.dumps(
                    {
                        "request_kind": "truevision_state_media_draft",
                        "scene": {"name": "bad first pass"},
                        "renderer": "edge_audio_river",
                        "media": {"audio_path": "D:/music/song.wav"},
                        "timeline": {"duration_seconds": 0, "fps": 12},
                        "safety_boundary": {"generated_state_media": True, "evidence": False},
                    }
                )
            return json.dumps(
                {
                    "request_kind": "truevision_state_media_draft",
                    "scene": {"name": "repaired river"},
                    "renderer": "edge_audio_river",
                    "media": {"audio_path": "D:/music/song.wav", "sync_to_audio": True},
                    "timeline": {"duration_seconds": 30, "fps": 12},
                    "visual_parameters": {"river_height_ratio": 0.22},
                    "safety_boundary": {"generated_state_media": True, "evidence": False},
                }
            )

        adapter = PromptToStateAdapter(fake_model, max_repairs=1)
        result = adapter.translate("make a thin river video", {"audio_path": "D:/music/song.wav"})

        self.assertTrue(result.ok)
        self.assertEqual(result.state["scene"]["name"], "repaired river")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(calls[1][-1]["role"], "user")
        self.assertIn("validation_errors", calls[1][-1]["content"])
        self.assertNotIn("bad first pass", calls[1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
