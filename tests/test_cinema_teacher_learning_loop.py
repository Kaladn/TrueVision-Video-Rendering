import json
import tempfile
import unittest
from pathlib import Path

from truevision_runtime.learning_intake.cinema_teacher import (
    DEFAULT_SEARCH_QUERIES,
    build_disk_guard_report,
    build_human_review_packet,
    cleanup_cinema_teacher_workspace,
    initialize_cinema_teacher_workspace,
    promote_cinematography_rule,
    rank_video_candidates,
)


class CinemaTeacherLearningLoopTests(unittest.TestCase):
    def test_workspace_setup_creates_bounded_directory_shape_and_seed_searches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cinema_teacher"

            manifest = initialize_cinema_teacher_workspace(root)

            self.assertEqual(manifest["schema_version"], "truevision_cinema_teacher_workspace_v1")
            self.assertTrue((root / "queue" / "search_queries.jsonl").exists())
            self.assertTrue((root / "queue" / "video_candidates.jsonl").exists())
            self.assertTrue((root / "queue" / "approved_videos.jsonl").exists())
            self.assertTrue((root / "active_job").is_dir())
            self.assertTrue((root / "learned" / "cinematography_rules.jsonl").exists())
            self.assertTrue((root / "learned" / "shot_grammar_presets.json").exists())
            self.assertTrue((root / "cache" / "temp_video_or_audio").is_dir())
            first_query = json.loads((root / "queue" / "search_queries.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first_query["query"], DEFAULT_SEARCH_QUERIES[0])
            self.assertFalse(manifest["boundary"]["general_internet_scraper"])
            self.assertFalse(manifest["boundary"]["auto_promote_rules"])

    def test_disk_guard_refuses_when_cache_exceeds_cap_and_reports_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cinema_teacher"
            initialize_cinema_teacher_workspace(root, max_total_cache_bytes=100)
            (root / "cache" / "temp_video_or_audio" / "chunk.bin").write_bytes(b"x" * 128)

            report = build_disk_guard_report(root, max_total_cache_bytes=100)

            self.assertFalse(report["can_start_new_job"])
            self.assertGreaterEqual(report["cache_bytes"], 128)
            self.assertIn("cache_over_cap", report["refusal_reasons"])
            self.assertEqual(report["cleanup_plan"]["delete_roots"], [str(root / "active_job"), str(root / "cache")])

    def test_cleanup_dry_run_preserves_files_and_actual_cleanup_deletes_only_transients(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cinema_teacher"
            initialize_cinema_teacher_workspace(root)
            cache_file = root / "cache" / "temp_video_or_audio" / "teacher.mp4"
            active_file = root / "active_job" / "lesson_notes.json"
            learned_file = root / "learned" / "cinematography_rules.jsonl"
            cache_file.write_bytes(b"video")
            active_file.write_text("{}", encoding="utf-8")
            learned_file.write_text('{"rule_id":"keep"}\n', encoding="utf-8")

            dry = cleanup_cinema_teacher_workspace(root, dry_run=True)

            self.assertTrue(cache_file.exists())
            self.assertTrue(active_file.exists())
            self.assertEqual(dry["deleted_bytes"], 0)

            cleanup = cleanup_cinema_teacher_workspace(root, dry_run=False)

            self.assertFalse(cache_file.exists())
            self.assertFalse(active_file.exists())
            self.assertTrue(learned_file.exists())
            self.assertGreater(cleanup["deleted_bytes"], 0)

    def test_candidate_ranking_prefers_hourlong_captioned_lesson_videos(self):
        ranked = rank_video_candidates(
            [
                {
                    "video_id": "short",
                    "title": "cinematic montage shorts",
                    "duration_seconds": 120,
                    "has_transcript": False,
                    "description": "pretty montage",
                },
                {
                    "video_id": "gear",
                    "title": "camera gear review",
                    "duration_seconds": 3600,
                    "has_transcript": True,
                    "description": "camera body lens review",
                },
                {
                    "video_id": "lesson",
                    "title": "foreground midground background cinematography tutorial",
                    "duration_seconds": 3660,
                    "has_transcript": True,
                    "description": "blocking, scene geography, camera movement, visual storytelling",
                },
            ]
        )

        self.assertEqual(ranked[0]["video_id"], "lesson")
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])
        self.assertIn("duration_40_to_90_minutes", ranked[0]["score_reasons"])
        self.assertIn("transcript_available", ranked[0]["score_reasons"])

    def test_rule_promotion_requires_human_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cinema_teacher"
            initialize_cinema_teacher_workspace(root)
            rule = {
                "rule_id": "wide_edge_needs_ground_plane",
                "applies_to": ["edge_nightmare_world:wide_edge_intro"],
                "renderer_action": {"ground_plane_visibility_min": 0.70},
            }

            with self.assertRaises(ValueError):
                promote_cinematography_rule(root, rule, human_approved=False)

            receipt = promote_cinematography_rule(root, rule, human_approved=True, human_rating={"depth": 5})
            lines = (root / "learned" / "cinematography_rules.jsonl").read_text(encoding="utf-8").splitlines()

            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["rule_id"], "wide_edge_needs_ground_plane")
            self.assertEqual(receipt["status"], "promoted")
            self.assertIn("receipt_hash", receipt)

    def test_review_packet_targets_wide_edge_intro_without_auto_promotion(self):
        packet = build_human_review_packet(
            source_meta={"video_id": "lesson", "title": "Cinematography lesson"},
            lesson_notes=["Establish geography before mood effects."],
            proposed_rules=[
                {
                    "rule_id": "wide_edge_geography_first",
                    "renderer_action": {"chaos_budget_max": 0.15},
                }
            ],
        )

        self.assertEqual(packet["renderer_target"]["scene_mode"], "edge_nightmare_world")
        self.assertEqual(packet["renderer_target"]["shot_type"], "wide_edge_intro")
        self.assertFalse(packet["boundary"]["auto_promote_rules"])
        self.assertIn("subject_readability", packet["qa_metrics"])


if __name__ == "__main__":
    unittest.main()
