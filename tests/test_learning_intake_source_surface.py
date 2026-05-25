import json
import tempfile
import unittest
from pathlib import Path

from truevision_runtime.av_tools.av_tool_runner import run_av_tool_call
from truevision_runtime.learning_intake.source_surface import (
    SourceSurfacePolicyError,
    build_source_surface_capture_plan,
    build_source_surface_multi_sample_plan,
    build_source_surface_video_state_receipt,
    canonicalize_approved_source_url,
)


class SourceSurfaceCapturePlanTests(unittest.TestCase):
    def test_timed_plan_starts_capture_before_play_and_stops_after_video_duration(self):
        plan = build_source_surface_capture_plan(
            element_id="fire_flame_licks",
            source_url="https://www.youtube.com/watch?v=abc123",
            source_title="Fire Lick Teacher",
            video_duration_seconds=42.0,
            player_region=[12, 140, 960, 540],
            run_id="fire_trial",
            pre_roll_seconds=0.35,
            post_roll_seconds=0.65,
        )

        self.assertEqual(plan["schema_version"], "truevision_source_surface_capture_plan_v1")
        self.assertEqual(plan["run_id"], "fire_trial")
        self.assertEqual(plan["source"]["address_bar_url"], "https://www.youtube.com/watch?v=abc123")
        self.assertEqual(plan["source"]["navigation_method"], "browser_address_bar")
        self.assertEqual(plan["navigation_flow"][0]["action"], "focus_address_bar")
        self.assertEqual(plan["navigation_flow"][-1]["action"], "verify_video_state")
        self.assertEqual(plan["capture"]["region"], [12, 140, 960, 540])
        self.assertEqual(plan["capture"]["duration_seconds"], 43.0)
        self.assertEqual(plan["timeline"][0]["event"], "capture_start")
        self.assertEqual(plan["timeline"][1]["event"], "play_button")
        self.assertLess(plan["timeline"][0]["at_capture_seconds"], plan["timeline"][1]["at_capture_seconds"])
        self.assertEqual(plan["timeline"][2]["at_capture_seconds"], 42.35)
        self.assertEqual(plan["timeline"][3]["event"], "capture_stop")
        self.assertIn("--region", plan["native_capture_command"])
        self.assertIn("12,140,960,540", plan["native_capture_command"])
        self.assertFalse(plan["boundary"]["browser_autonomy"])
        self.assertFalse(plan["boundary"]["youtube_search_navigation"])

    def test_canonicalizes_youtube_watch_urls_for_address_bar_navigation(self):
        canonical = canonicalize_approved_source_url(
            "https://www.youtube.com/watch?v=gad7k38N5zw&list=PLPJlGbBaoz-nPJhN75Bk3HNzPHi2yVAT8 "
        )

        self.assertEqual(canonical["address_bar_url"], "https://www.youtube.com/watch?v=gad7k38N5zw")
        self.assertEqual(canonical["video_id"], "gad7k38N5zw")
        self.assertEqual(canonical["removed_query_keys"], ["list"])
        self.assertEqual(canonical["navigation_method"], "browser_address_bar")

    def test_video_state_receipt_requires_verified_page_and_purged_teacher_state(self):
        receipt = build_source_surface_video_state_receipt(
            run_id="fire_trial",
            approved_url="https://www.youtube.com/watch?v=abc123&list=nope",
            resolved_url="https://www.youtube.com/watch?v=abc123",
            video_title="Real Fire Teacher",
            duration_detected_seconds=12.0,
            visual_state_records=180,
            not_gray_screen=True,
            not_error_page=True,
            profile_created=True,
            teacher_chunks_purged=True,
            source_time_delta_seconds=11.8,
            expected_sample_seconds=12.0,
            visual_motion_score=0.25,
            minimum_visual_motion_score=0.001,
        )

        self.assertEqual(receipt["status"], "verified")
        self.assertEqual(receipt["resolved_url"], "https://www.youtube.com/watch?v=abc123")
        self.assertTrue(receipt["checks"]["video_id_match"])
        self.assertTrue(receipt["checks"]["visual_state_records"])
        self.assertTrue(receipt["checks"]["source_video_time_advanced"])

    def test_video_state_receipt_rejects_completed_macro_without_real_video_state(self):
        with self.assertRaises(SourceSurfacePolicyError):
            build_source_surface_video_state_receipt(
                run_id="bad_trial",
                approved_url="https://www.youtube.com/watch?v=abc123",
                resolved_url="https://www.youtube.com/results?search_query=abc123",
                video_title="",
                duration_detected_seconds=0,
                visual_state_records=0,
                not_gray_screen=False,
                not_error_page=False,
                profile_created=False,
                teacher_chunks_purged=False,
                source_time_delta_seconds=0,
                expected_sample_seconds=12,
                visual_motion_score=0,
                minimum_visual_motion_score=0.001,
            )

    def test_video_state_receipt_rejects_paused_video_time(self):
        with self.assertRaises(SourceSurfacePolicyError):
            build_source_surface_video_state_receipt(
                run_id="paused_trial",
                approved_url="https://www.youtube.com/watch?v=abc123",
                resolved_url="https://www.youtube.com/watch?v=abc123",
                video_title="Paused Fire",
                duration_detected_seconds=60,
                visual_state_records=180,
                not_gray_screen=True,
                not_error_page=True,
                profile_created=True,
                teacher_chunks_purged=True,
                source_time_delta_seconds=0.1,
                expected_sample_seconds=12.0,
                visual_motion_score=0.25,
                minimum_visual_motion_score=0.001,
            )

    def test_video_state_receipt_rejects_static_visual_capture(self):
        with self.assertRaises(SourceSurfacePolicyError):
            build_source_surface_video_state_receipt(
                run_id="static_trial",
                approved_url="https://www.youtube.com/watch?v=abc123",
                resolved_url="https://www.youtube.com/watch?v=abc123",
                video_title="Static Fire",
                duration_detected_seconds=60,
                visual_state_records=180,
                not_gray_screen=True,
                not_error_page=True,
                profile_created=True,
                teacher_chunks_purged=True,
                source_time_delta_seconds=12.2,
                expected_sample_seconds=12.0,
                visual_motion_score=0.0,
                minimum_visual_motion_score=0.001,
            )

    def test_video_state_receipt_accepts_verified_target_seek(self):
        receipt = build_source_surface_video_state_receipt(
            run_id="river_target",
            approved_url="https://www.youtube.com/watch?v=river123",
            resolved_url="https://www.youtube.com/watch?v=river123",
            video_title="Long River Teacher",
            duration_detected_seconds=36000,
            visual_state_records=180,
            not_gray_screen=True,
            not_error_page=True,
            profile_created=True,
            teacher_chunks_purged=True,
            source_time_delta_seconds=12.1,
            expected_sample_seconds=12.0,
            requested_start_seconds=13494.0,
            source_time_before_seconds=13494.6,
            source_time_after_seconds=13506.7,
            source_duration_seconds=36000.0,
            visual_motion_score=0.25,
            minimum_visual_motion_score=0.001,
        )

        self.assertTrue(receipt["checks"]["source_player_duration_matches_approved"])
        self.assertTrue(receipt["checks"]["source_video_target_time_reached"])
        self.assertEqual(receipt["requested_start_seconds"], 13494.0)

    def test_video_state_receipt_rejects_preroll_when_target_timestamp_was_not_reached(self):
        with self.assertRaises(SourceSurfacePolicyError) as caught:
            build_source_surface_video_state_receipt(
                run_id="ad_preroll",
                approved_url="https://www.youtube.com/watch?v=river123",
                resolved_url="https://www.youtube.com/watch?v=river123",
                video_title="Long River Teacher",
                duration_detected_seconds=36000,
                visual_state_records=180,
                not_gray_screen=True,
                not_error_page=True,
                profile_created=True,
                teacher_chunks_purged=True,
                source_time_delta_seconds=13.5,
                expected_sample_seconds=12.0,
                requested_start_seconds=13494.0,
                source_time_before_seconds=0.45,
                source_time_after_seconds=13.96,
                source_duration_seconds=29.921,
                visual_motion_score=0.25,
                minimum_visual_motion_score=0.001,
            )

        self.assertIn("source_player_duration_matches_approved", str(caught.exception))

    def test_large_one_hour_video_gets_four_section_samples(self):
        plan = build_source_surface_multi_sample_plan(
            element_id="fire_flame_licks",
            source_url="https://www.youtube.com/watch?v=gad7k38N5zw&list=PLPJlGbBaoz-nPJhN75Bk3HNzPHi2yVAT8",
            video_title="Fire Particles Overlay",
            video_duration_seconds=3600,
            player_region=[0, 0, 2560, 1440],
            run_id="fire_one_hour",
            sample_seconds=12,
        )

        self.assertEqual(plan["schema_version"], "truevision_source_surface_multi_sample_plan_v1")
        self.assertEqual(plan["source"]["address_bar_url"], "https://www.youtube.com/watch?v=gad7k38N5zw")
        self.assertEqual(plan["sampling"]["sample_count"], 4)
        self.assertEqual(plan["sampling"]["sample_seconds"], 12.0)
        self.assertEqual([sample["start_seconds"] for sample in plan["samples"]], [444.0, 1344.0, 2244.0, 3144.0])
        self.assertEqual(plan["samples"][0]["section"], "section_1_of_4")
        self.assertEqual(plan["samples"][0]["sample_navigation_url"], "https://www.youtube.com/watch?v=gad7k38N5zw&t=444s")
        self.assertEqual(plan["samples"][3]["end_seconds"], 3156.0)
        self.assertTrue(plan["boundary"]["profile_each_sample_before_next"])
        self.assertTrue(plan["boundary"]["purge_teacher_chunks_each_sample"])

    def test_short_video_gets_single_sample_window(self):
        plan = build_source_surface_multi_sample_plan(
            element_id="smoke_curl_field",
            source_url="https://www.youtube.com/watch?v=smoke123",
            video_title="Smoke",
            video_duration_seconds=42,
            player_region=[0, 0, 1280, 720],
            run_id="smoke_short",
            sample_seconds=12,
        )

        self.assertEqual(plan["sampling"]["sample_count"], 1)
        self.assertEqual(plan["samples"][0]["start_seconds"], 0.0)
        self.assertEqual(plan["samples"][0]["end_seconds"], 12.0)

    def test_plan_rejects_forbidden_youtube_controls(self):
        with self.assertRaises(SourceSurfacePolicyError):
            build_source_surface_capture_plan(
                element_id="fire_flame_licks",
                source_url="https://www.youtube.com/watch?v=abc123",
                source_title="Fire Lick Teacher",
                video_duration_seconds=10.0,
                player_region=[0, 0, 640, 360],
                button_ids=["yt.button.subscribe"],
            )

    def test_av_tool_runner_writes_plan_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_av_tool_call(
                {
                    "tool": "source_surface_capture_plan",
                    "args": {
                        "element_id": "smoke_curl_field",
                        "source_url": "https://www.youtube.com/watch?v=smoke123",
                        "source_title": "Smoke Teacher",
                        "video_duration_seconds": 12.5,
                        "player_region": [20, 90, 1280, 720],
                        "run_id": "smoke_surface_trial",
                    },
                },
                storage_root=Path(tmp),
            )
            plan_path = Path(result["result"]["plan_json"])
            receipt_path = Path(result["result"]["receipt_json"])
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

        self.assertTrue(result["ok"], result)
        self.assertEqual(plan["run_id"], "smoke_surface_trial")
        self.assertEqual(plan["source"]["address_bar_url"], "https://www.youtube.com/watch?v=smoke123")
        self.assertEqual(receipt["tool"], "source_surface_capture_plan")
        self.assertEqual(receipt["result"]["plan_hash"], result["result"]["plan_hash"])

    def test_av_tool_runner_writes_verified_video_state_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_av_tool_call(
                {
                    "tool": "source_surface_video_state_receipt",
                    "args": {
                        "run_id": "verified_fire",
                        "approved_url": "https://www.youtube.com/watch?v=abc123&list=old",
                        "resolved_url": "https://www.youtube.com/watch?v=abc123",
                        "video_title": "Verified Fire",
                        "duration_detected_seconds": 10.0,
                        "visual_state_records": 150,
                        "not_gray_screen": True,
                        "not_error_page": True,
                        "profile_created": True,
                        "teacher_chunks_purged": True,
                    },
                },
                storage_root=Path(tmp),
            )
            receipt_path = Path(result["result"]["receipt_json"])
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

        self.assertTrue(result["ok"], result)
        self.assertEqual(receipt["status"], "verified")
        self.assertEqual(receipt["checks"]["teacher_chunks_purged"], True)

    def test_av_tool_runner_writes_multi_sample_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_av_tool_call(
                {
                    "tool": "source_surface_multi_sample_plan",
                    "args": {
                        "element_id": "fire_flame_licks",
                        "source_url": "https://www.youtube.com/watch?v=gad7k38N5zw&list=noise",
                        "source_title": "Fire Particles Overlay",
                        "video_duration_seconds": 3600,
                        "player_region": [0, 0, 2560, 1440],
                        "run_id": "fire_multisample",
                    },
                },
                storage_root=Path(tmp),
            )
            plan = json.loads(Path(result["result"]["plan_json"]).read_text(encoding="utf-8"))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["result"]["sample_count"], 4)
        self.assertEqual(plan["samples"][1]["start_seconds"], 1344.0)


if __name__ == "__main__":
    unittest.main()
