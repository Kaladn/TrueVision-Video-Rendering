import json
import tempfile
import unittest
from pathlib import Path

from truevision_runtime.state_media_qa import build_state_media_qa_receipt, render_state_media_qa_markdown


class StateMediaQaReceiptTests(unittest.TestCase):
    def test_dead_memory_receipt_compares_planned_stages_to_logged_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "frame_state.jsonl"
            manifest_path = root / "manifest.json"

            rows = [
                _row(0, 0.0, "whisper_dead_memory", rms=0.10, beat=0.20),
                _row(30, 1.0, "whisper_dead_memory", rms=0.12, beat=0.05),
                _row(330, 11.0, "room_wakes_bitterness", rms=0.24, beat=0.12),
                _row(900, 30.0, "vice_reveal", rms=0.45, beat=0.33),
                _row(1260, 42.0, "pressure_drop_truth_cuts", rms=0.66, beat=0.55),
                _row(1800, 60.0, "fall_from_grace", rms=0.44, beat=0.25),
                _row(2190, 73.0, "collision_core", rms=0.78, beat=0.72),
                _row(2730, 91.0, "final_chorus_peak", rms=0.91, beat=0.86),
                _row(2910, 97.0, "outro_release", rms=0.08, beat=0.02),
            ]
            state_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "truevision_weird_occlusion_rs.v1",
                        "run_id": "dead_memory_test",
                        "render": {
                            "duration_seconds": 100.0,
                            "fps": 30,
                            "frame_count": 3000,
                            "state_log_every": 30,
                            "frame_state_jsonl": str(state_path),
                        },
                        "scene_state": {
                            "scene_mode": "dead_memory_vice_chamber",
                            "motifs": ["black iron vice jaws"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            receipt = build_state_media_qa_receipt(manifest_path)

            self.assertEqual(receipt["schema_version"], "truevision_state_media_qa_receipt_v1")
            self.assertEqual(receipt["scene_mode"], "dead_memory_vice_chamber")
            self.assertTrue(receipt["qa_pass"])
            self.assertEqual(receipt["planned_stage_count"], 8)
            self.assertEqual(receipt["observed_stage_count"], 8)
            self.assertEqual(receipt["missing_stages"], [])
            pressure = receipt["stage_results"]["pressure_drop_truth_cuts"]
            self.assertEqual(pressure["sample_count"], 1)
            self.assertEqual(pressure["timing_status"], "pass")
            self.assertGreater(pressure["audio"]["rms"]["mean"], 0.6)
            self.assertIn("planned_beat", pressure)
            self.assertIn("visual_state_outputs", pressure)

    def test_daughter_star_locket_receipt_supports_effect_depth_qa_tiers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "daughter_state.jsonl"
            manifest_path = root / "daughter_manifest.json"
            stages = [
                ("dark_water_waiting", 4.0),
                ("first_memory_light", 14.0),
                ("distance_opens", 25.0),
                ("heart_fracture", 36.0),
                ("what_did_they_take", 47.0),
                ("father_reaches", 59.0),
                ("daughter_star_answers", 70.0),
                ("hope_holds", 77.0),
            ]
            rows = [
                {
                    **_row(index * 30, time_seconds, stage, rms=0.20 + index * 0.08, beat=0.10 + index * 0.04),
                    "scene": "daughter_star_locket_sea",
                    "stage": stage,
                    "water_reflection_mean": 0.18 + index * 0.01,
                    "star_glow_mean": 0.25 + index * 0.03,
                    "heart_crack_mean": 0.05 + index * 0.025,
                    "hope_light_mean": 0.01 + index * 0.02,
                    "state_layers": ["daughter_star_glow", "cracked_father_heart_locket"],
                }
                for index, (stage, time_seconds) in enumerate(stages)
            ]
            state_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "truevision_weird_occlusion_rs.v1",
                        "run_id": "daughter_star_test",
                        "render": {
                            "duration_seconds": 80.0,
                            "fps": 30,
                            "frame_count": 2400,
                            "state_log_every": 30,
                            "frame_state_jsonl": str(state_path),
                        },
                        "scene_state": {
                            "scene_mode": "daughter_star_locket_sea",
                            "motifs": ["daughter star", "cracked father heart"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            receipt = build_state_media_qa_receipt(manifest_path)

            self.assertTrue(receipt["structural_pass"])
            self.assertTrue(receipt["qa_pass"])
            self.assertEqual(receipt["artistic_depth_pass"], "manual_review_required")
            self.assertFalse(receipt["profile_calibrated"])
            self.assertEqual(receipt["missing_stages"], [])
            self.assertIn("daughter_star_answers", receipt["stage_results"])
            report = render_state_media_qa_markdown(receipt)
            self.assertIn("star_glow_mean:", report)
            self.assertIn("heart_crack_mean:", report)

    def test_edge_nightmare_world_receipt_tracks_pov_and_silhouette_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "edge_state.jsonl"
            manifest_path = root / "edge_manifest.json"
            stages = [
                ("black_edge_wake", 5.0),
                ("walk_to_rim", 16.0),
                ("side_parallax_pressure", 30.0),
                ("just_looking_down", 44.0),
                ("falling_camera_spiral", 57.0),
                ("river_below_answers", 72.0),
                ("storm_power_walk", 86.0),
                ("gold_edge_release", 97.0),
            ]
            rows = [
                {
                    **_row(index * 60, time_seconds, stage, rms=0.22 + index * 0.05, beat=0.08 + index * 0.05),
                    "scene": "edge_nightmare_world",
                    "stage": stage,
                    "camera_state": "top_down_abyss_view" if stage == "just_looking_down" else "wide_angle_push_in",
                    "abyss_river_mean": 0.03 + index * 0.012,
                    "silhouette_mean": 0.01 + index * 0.004,
                    "lightning_bloom_mean": 0.02 + index * 0.010,
                    "transform_pressure_mean": 0.04 + index * 0.018,
                    "hope_release_mean": 0.0 if index < 7 else 0.12,
                    "state_layers": ["nightmare_cliff_rim", "human_silhouette_motion", "arc_learning_transform_mix"],
                }
                for index, (stage, time_seconds) in enumerate(stages)
            ]
            state_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "truevision_weird_occlusion_rs.v1",
                        "run_id": "edge_nightmare_test",
                        "render": {
                            "duration_seconds": 100.0,
                            "fps": 60,
                            "frame_count": 6000,
                            "state_log_every": 60,
                            "frame_state_jsonl": str(state_path),
                        },
                        "scene_state": {
                            "scene_mode": "edge_nightmare_world",
                            "motifs": ["nightmare cliff rim", "human silhouettes", "top-down abyss look"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            receipt = build_state_media_qa_receipt(manifest_path)

            self.assertTrue(receipt["qa_pass"])
            self.assertEqual(receipt["planned_stage_count"], 8)
            self.assertEqual(receipt["observed_stage_count"], 8)
            self.assertEqual(receipt["missing_stages"], [])
            self.assertIn("just_looking_down", receipt["stage_results"])
            report = render_state_media_qa_markdown(receipt)
            self.assertIn("abyss_river_mean:", report)
            self.assertIn("silhouette_mean:", report)
            self.assertIn("transform_pressure_mean:", report)


def _row(frame_index: int, time_seconds: float, stage: str, *, rms: float, beat: float) -> dict:
    return {
        "frame_index": frame_index,
        "time_seconds": time_seconds,
        "scene": "dead_memory_vice_chamber",
        "stage": stage,
        "audio": {
            "rms": rms,
            "bass": rms * 0.8,
            "high": rms * 0.35,
            "beat": beat,
            "vocal_presence": rms * 0.45,
        },
        "phase": time_seconds / 100.0,
        "fog_mean": 0.05 + rms * 0.03,
        "vice_pressure_mean": 0.20 + beat * 0.30,
        "memory_core_mean": 0.01 + rms * 0.04,
        "glow_pixels": int(1000 + rms * 1000),
        "state_layers": ["black_iron_vice_jaws", "cracked_glowing_memory_core"],
    }
