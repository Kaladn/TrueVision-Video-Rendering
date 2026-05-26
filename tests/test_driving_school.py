import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_runtime.learning_intake.driving_school import (
    CANDIDATE_OBJECT_TYPES,
    REJECTION_REASONS,
    UNKNOWN_CANDIDATE_TYPES,
    build_driving_calibration_receipt_from_frames,
    build_driving_scene_profile_from_frames,
    write_driving_school_run,
)


def _road_frame(index: int, *, fog: bool = False) -> np.ndarray:
    height = 96
    width = 128
    frame = np.full((height, width, 3), [56, 67, 76], dtype=np.uint8)
    frame[:38, :] = [90, 110, 128]
    frame[38:, :] = [42, 58, 46]

    vanishing_x = width // 2
    vanishing_y = 39
    road_half_near = 48
    road = np.array(
        [
            [vanishing_x - 8, vanishing_y],
            [vanishing_x + 8, vanishing_y],
            [vanishing_x + road_half_near, height - 1],
            [vanishing_x - road_half_near, height - 1],
        ],
        dtype=np.int32,
    )
    try:
        import cv2

        cv2.fillConvexPoly(frame, road, [45, 45, 43])
        cv2.line(frame, (vanishing_x, vanishing_y + 4), (vanishing_x + 4, height - 1), [225, 225, 205], 2)
        cv2.line(frame, (vanishing_x - 21, vanishing_y + 12), (8, height - 1), [170, 170, 160], 1)
        cv2.line(frame, (vanishing_x + 21, vanishing_y + 12), (120, height - 1), [170, 170, 160], 1)
        cv2.rectangle(frame, (90, 35), (112, 56), [88, 82, 76], -1)
        cv2.rectangle(frame, (18, 28), (25, 70), [31, 54, 35], -1)
        cv2.circle(frame, (100, 31), 5, [35, 30, 210], -1)
    except Exception:
        frame[39:, 18:110] = [45, 45, 43]
        frame[40:, 63:67] = [225, 225, 205]
        frame[35:56, 90:112] = [88, 82, 76]
        frame[28:70, 18:25] = [31, 54, 35]
        frame[26:36, 95:105] = [35, 30, 210]

    shift = min(index, 12)
    frame[22 + shift : 30 + shift, 8:16] = [38, 78, 40]
    if fog:
        haze = np.full_like(frame, [168, 178, 184])
        alpha = np.linspace(0.48, 0.16, height, dtype=np.float32).reshape(height, 1, 1)
        frame = np.clip(frame.astype(np.float32) * (1.0 - alpha) + haze.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return frame


class DrivingSchoolTests(unittest.TestCase):
    def test_calibration_uses_middle_frame_bounds_and_records_baselines(self):
        frames = [np.zeros((120, 90, 3), dtype=np.uint8) for _ in range(5)]
        middle = np.zeros((120, 90, 3), dtype=np.uint8)
        middle[24:96, 0:90] = [80, 92, 104]
        middle[42:78, 20:70] = [170, 180, 190]
        frames[2] = middle

        receipt = build_driving_calibration_receipt_from_frames(
            frames,
            run_id="calibration_middle_bounds",
            source_label="synthetic_black_bars",
            fps=30.0,
        )

        self.assertEqual(receipt["schema_version"], "driving_calibration_receipt_v1")
        self.assertEqual(receipt["first_valid_frame"], 0)
        self.assertEqual(receipt["middle_frame"], 2)
        self.assertEqual(receipt["last_valid_frame"], 4)
        self.assertEqual(receipt["content_bounds"]["method"], "middle_frame_active_region")
        self.assertLessEqual(receipt["content_bounds"]["y"], 28)
        self.assertGreaterEqual(receipt["content_bounds"]["height"], 66)
        self.assertIn("camera_motion_baseline", receipt)
        self.assertIn("noise_baseline", receipt)
        self.assertIn("fog_visibility_baseline", receipt)
        self.assertFalse(receipt["retention"]["raw_frames_retained"])

    def test_scene_profile_emits_candidate_only_road_objects_unknowns_and_mock_world(self):
        frames = [_road_frame(index) for index in range(18)]

        profile = build_driving_scene_profile_from_frames(
            frames,
            run_id="synthetic_road_school",
            source_label="synthetic_road",
            fps=18.0,
            sample_fps=2.0,
            grid_shape=(12, 16),
        )

        self.assertEqual(profile["schema_version"], "driving_scene_profile_v1")
        self.assertEqual(profile["awareness_contract"]["mode"], "high_speed_awareness_v0")
        self.assertTrue(profile["awareness_contract"]["near_real_time_perception_target"])
        self.assertFalse(profile["awareness_contract"]["driving_claim"])
        self.assertIn("fog_reveal", profile["awareness_contract"]["curriculum"])
        self.assertIn("reflection_field", profile["awareness_contract"]["curriculum"])
        self.assertIn("city_skyline", profile["awareness_contract"]["curriculum"])
        self.assertEqual(profile["calibration"]["schema_version"], "driving_calibration_receipt_v1")
        self.assertGreater(profile["road_geometry"]["road_center"]["confidence"], 0.0)
        self.assertIn("vanishing_point", profile["road_geometry"])
        self.assertEqual(profile["recognition_boundary"]["truth_promotion_allowed"], False)
        self.assertTrue(profile["retention"]["no_raw_video_copy"])

        event_types = {event["candidate"] for event in profile["candidate_events"]}
        self.assertIn("candidate_lane_line", event_types)
        self.assertIn("candidate_road_plane", event_types)
        self.assertIn("candidate_vehicle_shape", event_types)
        self.assertIn("candidate_sign_shape", event_types)
        self.assertIn("candidate_light_source", event_types)
        self.assertIn("candidate_occlusion_event", event_types)
        self.assertIn("candidate_tree_mass", event_types)
        self.assertIn("candidate_tree_line", event_types)
        self.assertIn("candidate_building_mass", event_types)
        self.assertIn("candidate_building_edge", event_types)
        self.assertIn("candidate_water_surface", event_types)
        self.assertIn("candidate_city_skyline", event_types)
        self.assertIn("candidate_cloud_field", event_types)
        self.assertIn("candidate_red_cloud", event_types)
        self.assertIn("candidate_grass_motion", event_types)
        self.assertIn("candidate_reflection_field", event_types)
        self.assertIn("candidate_fog_reveal", event_types)
        self.assertIn("candidate_object_resolving_through_fog", event_types)
        self.assertIn("candidate_high_speed_forward_flow", event_types)
        self.assertIn("candidate_scene_depth_layer", event_types)
        self.assertIn("candidate_unknown_sign", event_types)
        self.assertIn("candidate_unknown_object", event_types)
        self.assertTrue(set(UNKNOWN_CANDIDATE_TYPES).issubset(event_types))
        self.assertTrue(set(CANDIDATE_OBJECT_TYPES).issuperset(event for event in event_types if not event.startswith("candidate_unknown")))

        for event in profile["candidate_events"]:
            self.assertIn("status", event)
            self.assertNotEqual(event["status"], "confirmed")
            self.assertTrue(event.get("meter_evidence") or event.get("rejection_reasons"))

        mock = profile["mock_road_world_v1"]
        self.assertEqual(mock["schema_version"], "mock_road_world_v1")
        self.assertIn("road_plane", mock)
        self.assertIn("visibility_model", mock)
        self.assertEqual(mock["boundary"]["renderable_state_not_source_replay"], True)

    def test_fog_scene_records_reveal_metrics_and_rejects_unsupported_candidates(self):
        frames = [_road_frame(index, fog=True) for index in range(16)]

        profile = build_driving_scene_profile_from_frames(
            frames,
            run_id="synthetic_fog_school",
            source_label="synthetic_fog_road",
            fps=16.0,
            sample_fps=2.0,
            grid_shape=(12, 16),
        )

        weather = profile["weather_visibility"]
        self.assertGreater(weather["fog_density"], 0.0)
        self.assertIn("edge_recovery_distance", weather)
        self.assertIn("object_reveal_rate", weather)

        rejected = [
            reason
            for event in profile["candidate_events"]
            for reason in event.get("rejection_reasons", [])
        ]
        self.assertTrue(set(REJECTION_REASONS).intersection(rejected))
        self.assertIn("rejected_fog_blob", rejected)

    def test_write_run_writes_manifests_receipts_and_keeps_raw_out(self):
        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "road.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (128, 96))
            self.assertTrue(writer.isOpened())
            for index in range(24):
                writer.write(_road_frame(index))
            writer.release()

            result = write_driving_school_run(
                {
                    "sources": [str(video)],
                    "run_id": "tiny_road_school",
                    "sample_fps": 2.0,
                    "max_frames": 18,
                    "long_edge_cells": 16,
                },
                storage_root=root / "storage",
            )

            self.assertEqual(result["schema_version"], "driving_school_batch_result_v1")
            self.assertEqual(result["source_count"], 1)
            self.assertTrue(Path(result["sources"][0]["manifest_json"]).exists())
            self.assertTrue(Path(result["sources"][0]["receipt_json"]).exists())
            self.assertTrue(Path(result["sources"][0]["calibration_receipt_json"]).exists())
            receipt = json.loads(Path(result["sources"][0]["receipt_json"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema_version"], "driving_event_receipt_v1")
            self.assertEqual(receipt["boundary"]["raw_video_copied"], False)
            self.assertEqual(receipt["boundary"]["raw_frames_retained"], False)
            self.assertTrue(Path(result["mock_road_world_json"]).exists())

            sheet_paths = result["sources"][0]["artifact_sheets"]
            self.assertEqual(
                set(sheet_paths),
                {
                    "driving_profile_manifest",
                    "awareness_profile_manifest",
                    "road_geometry_profile",
                    "visibility_depth_profile",
                    "motion_pressure_profile",
                    "candidate_object_sheet",
                    "rejection_sheet",
                    "mock_road_world_v1",
                    "receipt",
                },
            )
            for name, path in sheet_paths.items():
                self.assertTrue(Path(path).exists(), name)
            rejection_sheet = json.loads(Path(sheet_paths["rejection_sheet"]).read_text(encoding="utf-8"))
            self.assertGreaterEqual(rejection_sheet["rejected_candidate_count"], 1)
            candidate_sheet = json.loads(Path(sheet_paths["candidate_object_sheet"]).read_text(encoding="utf-8"))
            self.assertFalse(candidate_sheet["boundary"]["truth_promotion_allowed"])
            awareness_sheet = json.loads(Path(sheet_paths["awareness_profile_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(awareness_sheet["mode"], "high_speed_awareness_v0")
            self.assertEqual(awareness_sheet["law"], "Do not learn the movie. Learn high-speed awareness behavior.")

    def test_write_run_accepts_single_source_folder_string(self):
        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_folder = root / "phone_clips"
            source_folder.mkdir()
            video = source_folder / "clip.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (128, 96))
            self.assertTrue(writer.isOpened())
            for index in range(14):
                writer.write(_road_frame(index))
            writer.release()

            result = write_driving_school_run(
                {
                    "source_folder": str(source_folder),
                    "run_id": "folder_road_school",
                    "sample_fps": 2.0,
                    "max_frames": 12,
                    "long_edge_cells": 16,
                },
                storage_root=root / "storage",
            )

            self.assertEqual(result["source_count"], 1)
            self.assertTrue(result["sources"][0]["source_video"].endswith("clip.mp4"))


if __name__ == "__main__":
    unittest.main()
