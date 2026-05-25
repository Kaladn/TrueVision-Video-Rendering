import json
import tempfile
import unittest
from pathlib import Path

from truevision_runtime.learning_intake.terrain_teacher import (
    DEFAULT_TERRAIN_SEARCH_QUERIES,
    build_terrain_extraction_contract,
    build_terrain_human_review_packet,
    cleanup_terrain_teacher_workspace,
    initialize_terrain_teacher_workspace,
    promote_terrain_rule,
    rank_terrain_candidates,
    terrain_disk_guard_report,
)


class TerrainTeacherLearningLoopTests(unittest.TestCase):
    def test_workspace_setup_prioritizes_realism_sources_before_cinema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "terrain_teacher"

            manifest = initialize_terrain_teacher_workspace(root)

            self.assertEqual(manifest["schema_version"], "truevision_terrain_teacher_workspace_v1")
            self.assertTrue((root / "queue" / "search_queries.jsonl").exists())
            self.assertTrue((root / "queue" / "video_candidates.jsonl").exists())
            self.assertTrue((root / "queue" / "approved_sources.jsonl").exists())
            self.assertTrue((root / "active_job" / "sampled_frames").is_dir())
            self.assertTrue((root / "active_job" / "render_tests").is_dir())
            self.assertTrue((root / "cache" / "temp_video_or_audio").is_dir())
            self.assertTrue((root / "learned" / "ocean_cliff_rules.jsonl").exists())
            self.assertTrue((root / "learned" / "canyon_depth_rules.jsonl").exists())
            self.assertTrue((root / "learned" / "volcano_glow_rules.jsonl").exists())
            self.assertTrue((root / "learned" / "fog_atmosphere_rules.jsonl").exists())

            first_query = json.loads((root / "queue" / "search_queries.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first_query["query"], DEFAULT_TERRAIN_SEARCH_QUERIES[0])
            self.assertEqual(first_query["source_class"], "ocean_cliffs")
            self.assertFalse(manifest["boundary"]["general_internet_scraper"])
            self.assertFalse(manifest["boundary"]["auto_promote_rules"])

    def test_candidate_ranking_prefers_long_real_geography_sources(self):
        ranked = rank_terrain_candidates(
            [
                {
                    "video_id": "abstract",
                    "title": "dark abstract cliff visualizer",
                    "description": "animated background",
                    "duration_seconds": 240,
                    "has_transcript": False,
                    "source_class": "ocean_cliffs",
                },
                {
                    "video_id": "real",
                    "title": "ocean cliffs drone footage 1 hour",
                    "description": "coastal erosion cliffs, waves, horizon, sea caves, real rock texture",
                    "duration_seconds": 3600,
                    "has_transcript": True,
                    "source_class": "ocean_cliffs",
                },
                {
                    "video_id": "gear",
                    "title": "best drone camera settings for cliffs",
                    "description": "gear review and lens settings",
                    "duration_seconds": 4200,
                    "has_transcript": True,
                    "source_class": "ocean_cliffs",
                },
            ]
        )

        self.assertEqual(ranked[0]["video_id"], "real")
        self.assertIn("duration_30_to_90_minutes", ranked[0]["score_reasons"])
        self.assertIn("transcript_available", ranked[0]["score_reasons"])
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])

    def test_extraction_contract_names_physical_scene_rules_not_style_vibes(self):
        contract = build_terrain_extraction_contract("ocean_cliffs")

        self.assertEqual(contract["source_class"], "ocean_cliffs")
        self.assertIn("horizon_behavior", contract["extract_fields"])
        self.assertIn("terrain_edge_shapes", contract["extract_fields"])
        self.assertIn("depth_cues", contract["extract_fields"])
        self.assertIn("renderer_parameter_suggestions", contract["extract_fields"])
        self.assertEqual(contract["first_renderer_target"]["shot_type"], "wide_edge_intro")
        self.assertEqual(contract["learning_goal"], "physical_scene_rules_before_cinematography")

    def test_disk_guard_and_cleanup_keep_rules_but_flush_teacher_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "terrain_teacher"
            initialize_terrain_teacher_workspace(root, max_total_cache_bytes=64)
            cache_file = root / "cache" / "temp_video_or_audio" / "teacher.mp4"
            active_file = root / "active_job" / "sampled_frames" / "frame001.png"
            learned_file = root / "learned" / "ocean_cliff_rules.jsonl"
            cache_file.write_bytes(b"x" * 80)
            active_file.write_bytes(b"frame")
            learned_file.write_text('{"rule_id":"keep"}\n', encoding="utf-8")

            guard = terrain_disk_guard_report(root, max_total_cache_bytes=64)

            self.assertFalse(guard["can_start_new_job"])
            self.assertIn("cache_over_cap", guard["refusal_reasons"])

            cleanup = cleanup_terrain_teacher_workspace(root, dry_run=False)

            self.assertGreater(cleanup["deleted_bytes"], 0)
            self.assertFalse(cache_file.exists())
            self.assertFalse(active_file.exists())
            self.assertTrue(learned_file.exists())

    def test_terrain_rule_promotion_requires_human_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "terrain_teacher"
            initialize_terrain_teacher_workspace(root)
            rule = {
                "rule_id": "ocean_cliff_needs_jagged_rim",
                "source_class": "ocean_cliffs",
                "renderer_action": {"foreground_cliff_rim": "jagged_high_contrast"},
            }

            with self.assertRaises(ValueError):
                promote_terrain_rule(root, rule, human_approved=False)

            receipt = promote_terrain_rule(root, rule, human_approved=True, human_rating={"realism": 5})

            lines = (root / "learned" / "ocean_cliff_rules.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[-1])["rule_id"], "ocean_cliff_needs_jagged_rim")
            self.assertEqual(receipt["status"], "promoted")
            self.assertIn("receipt_hash", receipt)

    def test_review_packet_targets_edge_depth_proof_and_blocks_full_song(self):
        packet = build_terrain_human_review_packet(
            source_meta={"video_id": "real", "title": "Ocean cliffs drone footage"},
            physical_rules=["Cliff rim is jagged and high contrast."],
            proposed_renderer_rules=[
                {"rule_id": "jagged_rim", "renderer_action": {"edge_visibility_min": 0.70}}
            ],
        )

        self.assertEqual(packet["renderer_target"]["scene_mode"], "edge_nightmare_world")
        self.assertEqual(packet["renderer_target"]["shot_type"], "wide_edge_intro")
        self.assertEqual(packet["renderer_target"]["duration_seconds"], 12)
        self.assertFalse(packet["boundary"]["full_song_render_allowed"])
        self.assertIn("ground_plane_visibility", packet["qa_metrics"])


if __name__ == "__main__":
    unittest.main()
