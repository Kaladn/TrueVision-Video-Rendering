import tempfile
import unittest
from pathlib import Path

from truevision_runtime.learning_intake.coordinate_surface import (
    build_coordinate_intake_plan,
    build_coordinate_intake_receipt,
    validate_coordinate_map,
)


class CoordinateSurfaceTests(unittest.TestCase):
    def test_coordinate_map_requires_named_points_and_region(self):
        coordinate_map = {
            "schema_version": "truevision_coordinate_surface_map_v1",
            "screen_size": [2560, 1440],
            "points": {
                "address_bar": [550, 88],
                "video_play": [840, 515],
            },
            "capture_region": [0, 0, 1600, 900],
        }

        validated = validate_coordinate_map(coordinate_map)

        self.assertEqual(validated["points"]["address_bar"], [550, 88])
        self.assertEqual(validated["capture_region"], [0, 0, 1600, 900])

    def test_coordinate_map_rejects_missing_play_point(self):
        with self.assertRaises(ValueError):
            validate_coordinate_map(
                {
                    "schema_version": "truevision_coordinate_surface_map_v1",
                    "screen_size": [2560, 1440],
                    "points": {"address_bar": [550, 88]},
                    "capture_region": [0, 0, 1600, 900],
                }
            )

    def test_coordinate_intake_plan_starts_capture_before_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_coordinate_intake_plan(
                run_id="coord_fire",
                source={
                    "source_order": 1,
                    "category": "FIRE / FLAME LICKS",
                    "element_id": "fire_flame_licks",
                    "source_url": "https://www.youtube.com/watch?v=abc123&t=44s",
                    "video_title": "Fire Teacher",
                    "duration_seconds": 300.0,
                },
                sample={
                    "sample_index": 0,
                    "start_seconds": 44.0,
                    "duration_seconds": 12.0,
                    "run_id": "coord_fire_sample_01",
                    "sample_navigation_url": "https://www.youtube.com/watch?v=abc123&t=44s",
                },
                coordinate_map={
                    "schema_version": "truevision_coordinate_surface_map_v1",
                    "screen_size": [2560, 1440],
                    "points": {
                        "address_bar": [550, 88],
                        "video_play": [840, 515],
                    },
                    "capture_region": [0, 0, 1600, 900],
                },
                output_root=Path(tmp) / "captures",
                native_capture_exe=Path("capture.exe"),
                fps=15,
                resolution=[1280, 720],
                grid=[320, 180],
            )

        self.assertEqual(plan["timeline"][0]["event"], "capture_start")
        self.assertEqual(plan["timeline"][1]["event"], "paste_url")
        self.assertEqual(plan["timeline"][2]["event"], "play_click")
        self.assertLess(plan["timeline"][0]["at_seconds"], plan["timeline"][2]["at_seconds"])
        self.assertFalse(plan["boundary"]["youtube_search_navigation"])
        self.assertEqual(plan["coordinate_map"]["points"]["video_play"], [840, 515])

    def test_coordinate_receipt_verifies_profile_and_purge_without_source_time_claim(self):
        receipt = build_coordinate_intake_receipt(
            run_id="coord_fire_sample_01",
            approved_url="https://www.youtube.com/watch?v=abc123",
            sample_navigation_url="https://www.youtube.com/watch?v=abc123&t=44s",
            coordinate_map_id="youtube_left_panel_v1",
            coordinate_map_sha256="sha256:abc123",
            capture_region=[0, 0, 1600, 900],
            visual_state_records=180,
            profile_created=True,
            teacher_chunks_purged=True,
            visual_motion_score=0.12,
        )

        self.assertEqual(receipt["status"], "verified")
        self.assertFalse(receipt["boundary"]["source_time_proof"])
        self.assertTrue(receipt["boundary"]["coordinate_map_required_before_run"])
        self.assertTrue(receipt["checks"]["coordinate_map_sha256"])
        self.assertTrue(receipt["checks"]["teacher_chunks_purged"])


if __name__ == "__main__":
    unittest.main()
