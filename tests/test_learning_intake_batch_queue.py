import unittest

from truevision_runtime.learning_intake.batch_queue import (
    build_batch_queue,
    parse_approved_youtube_sources,
)


class LearningIntakeBatchQueueTests(unittest.TestCase):
    def test_parse_approved_file_keeps_category_and_canonical_url_order(self):
        text = """
Fire

https://www.youtube.com/watch?v=gad7k38N5zw&list=PLPJlGbBaoz-nPJhN75Bk3HNzPHi2yVAT8

SMOKE

https://www.youtube.com/watch?v=l3RqYk4hzMw
"""

        entries = parse_approved_youtube_sources(text)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["category"], "Fire")
        self.assertEqual(entries[0]["element_id"], "fire_flame_licks")
        self.assertEqual(entries[0]["video_id"], "gad7k38N5zw")
        self.assertEqual(entries[0]["address_bar_url"], "https://www.youtube.com/watch?v=gad7k38N5zw")
        self.assertEqual(entries[0]["source_order"], 1)
        self.assertEqual(entries[1]["category"], "SMOKE")
        self.assertEqual(entries[1]["element_id"], "smoke_turbulent_columns")

    def test_parse_approved_file_strips_inline_notes_after_url(self):
        text = "Rain\nhttps://www.youtube.com/watch?v=f46hUEBMnt4  (rain, wind, trees)\n"

        entries = parse_approved_youtube_sources(text)

        self.assertEqual(entries[0]["source_url"], "https://www.youtube.com/watch?v=f46hUEBMnt4")
        self.assertEqual(entries[0]["address_bar_url"], "https://www.youtube.com/watch?v=f46hUEBMnt4")

    def test_build_batch_queue_uses_four_samples_for_large_videos(self):
        entries = parse_approved_youtube_sources(
            "Fire\nhttps://www.youtube.com/watch?v=gad7k38N5zw&list=noise\n"
        )

        queue = build_batch_queue(
            entries,
            metadata_by_video_id={
                "gad7k38N5zw": {
                    "video_title": "Fire Particles Overlay 4K",
                    "duration_seconds": 3600,
                }
            },
            player_region=[0, 0, 2560, 1440],
            run_id="learning_all",
            sample_seconds=12,
        )

        self.assertEqual(queue["schema_version"], "truevision_learning_intake_batch_queue_v1")
        self.assertEqual(queue["source_count"], 1)
        self.assertEqual(queue["sample_count"], 4)
        self.assertEqual(queue["sources"][0]["samples"][0]["start_seconds"], 444.0)
        self.assertEqual(queue["sources"][0]["samples"][3]["start_seconds"], 3144.0)
        self.assertEqual(queue["sources"][0]["samples"][0]["sample_navigation_url"], "https://www.youtube.com/watch?v=gad7k38N5zw&t=444s")
        self.assertTrue(queue["retention"]["purge_teacher_chunks_after_profile"])

    def test_build_batch_queue_skips_sources_without_duration(self):
        entries = parse_approved_youtube_sources(
            "Fire\nhttps://www.youtube.com/watch?v=gad7k38N5zw\n"
        )

        queue = build_batch_queue(
            entries,
            metadata_by_video_id={},
            player_region=[0, 0, 2560, 1440],
            run_id="learning_all",
        )

        self.assertEqual(queue["source_count"], 0)
        self.assertEqual(queue["skipped_count"], 1)
        self.assertEqual(queue["skipped_sources"][0]["reason"], "duration_not_detected")


if __name__ == "__main__":
    unittest.main()
