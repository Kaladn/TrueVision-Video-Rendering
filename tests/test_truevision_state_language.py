import unittest

from truevision_runtime.state_language import (
    STATE_LANGUAGE_VERBS,
    build_state_language,
    normalize_state_stage,
    required_stages_for_direction,
)


class TrueVisionStateLanguageTests(unittest.TestCase):
    def test_legacy_manifest_fields_map_to_canonical_verbs(self):
        language = build_state_language(
            {
                "tool_id": "atmosphere_profile_from_capture",
                "behavior_family": "fog_reveal",
                "observes_state": True,
                "abstracts_behavior": True,
                "generates_state": False,
                "renders_media": False,
                "copies_source_media": False,
            }
        )

        self.assertEqual(STATE_LANGUAGE_VERBS, ("witness", "profile", "plan", "replay", "surface"))
        self.assertEqual(language["behavior_family"], "fog_reveal")
        self.assertTrue(language["can_witness"])
        self.assertTrue(language["can_profile"])
        self.assertFalse(language["can_plan"])
        self.assertFalse(language["can_replay"])
        self.assertFalse(language["can_surface"])
        self.assertFalse(language["copies_source_media"])

    def test_missing_language_fails_closed(self):
        language = build_state_language({"tool_id": "unknown_tool"})

        self.assertEqual(language["behavior_family"], "unclassified")
        self.assertFalse(language["can_witness"])
        self.assertFalse(language["can_profile"])
        self.assertFalse(language["can_plan"])
        self.assertFalse(language["can_replay"])
        self.assertFalse(language["can_surface"])
        self.assertFalse(language["copies_source_media"])

    def test_direction_aliases_select_canonical_stages(self):
        self.assertEqual(required_stages_for_direction("forward_observation"), ("witness", "profile"))
        self.assertEqual(required_stages_for_direction("reverse_generation"), ("plan", "replay", "surface"))
        self.assertEqual(required_stages_for_direction("record"), ("witness",))
        self.assertEqual(required_stages_for_direction("render"), ("surface",))
        self.assertEqual(required_stages_for_direction("generate"), ("plan", "replay", "surface"))

    def test_media_terms_are_compatibility_aliases_only(self):
        self.assertEqual(normalize_state_stage("witness"), "witness")
        self.assertEqual(normalize_state_stage("record"), "witness")
        self.assertEqual(normalize_state_stage("render"), "surface")
        self.assertEqual(normalize_state_stage("surface"), "surface")
        self.assertEqual(normalize_state_stage("unknown"), "")

    def test_surface_media_is_optional_not_authority(self):
        language = build_state_language(
            {
                "tool_id": "render_truedepth_fog_reveal_samples",
                "behavior_family": "fog_reveal",
                "generates_state": True,
                "renders_media": True,
                "raw_video_saved": False,
                "copies_source_media": False,
                "output_types": ["generated_proof_media", "manifest", "receipt"],
            }
        )

        self.assertTrue(language["can_plan"])
        self.assertTrue(language["can_replay"])
        self.assertTrue(language["can_surface"])
        self.assertTrue(language["media_is_optional_surface"])
        self.assertFalse(language["media_is_source_truth"])


if __name__ == "__main__":
    unittest.main()
