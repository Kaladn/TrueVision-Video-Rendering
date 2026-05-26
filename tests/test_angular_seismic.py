import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_runtime.learning_intake.angular_seismic import (
    build_angular_seismic_profile_from_frames,
    detect_content_bounds_from_frame,
    derive_aspect_matched_grid,
    derive_virtual_grid,
    write_angular_seismic_profile_from_video,
)


class AngularSeismicTests(unittest.TestCase):
    def test_derive_aspect_matched_grid_preserves_landscape_and_portrait_ratio(self):
        self.assertEqual(derive_aspect_matched_grid(1920, 1080, long_edge_cells=48), (27, 48))
        self.assertEqual(derive_aspect_matched_grid(1080, 1920, long_edge_cells=48), (48, 27))
        self.assertEqual(derive_aspect_matched_grid(1080, 1080, long_edge_cells=48), (48, 48))

    def test_derive_virtual_grid_supports_operator_aspect_modes(self):
        self.assertEqual(derive_virtual_grid(1080, 1920, long_edge_cells=48, aspect_mode="square"), (48, 48))
        self.assertEqual(derive_virtual_grid(1080, 1920, long_edge_cells=48, aspect_mode="landscape"), (27, 48))
        self.assertEqual(derive_virtual_grid(1920, 1080, long_edge_cells=48, aspect_mode="portrait"), (48, 27))

    def test_detect_content_bounds_from_middle_frame_ignores_black_bars(self):
        frame = np.zeros((160, 90, 3), dtype=np.uint8)
        frame[35:125, 0:90] = [90, 100, 110]
        frame[50:110, 20:70] = [180, 190, 200]
        bounds = detect_content_bounds_from_frame(frame)
        self.assertLessEqual(bounds["y"], 38)
        self.assertGreaterEqual(bounds["height"], 84)
        self.assertAlmostEqual(bounds["aspect_ratio"], 1.0, delta=0.15)

    def test_synthetic_moving_reflection_produces_16_direction_profile(self):
        frames = []
        height = 96
        width = 128
        for index in range(24):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 20 + index * 3
            y = 42 + int(np.sin(index / 3.0) * 5)
            frame[max(0, y - 6) : min(height, y + 6), max(0, x - 10) : min(width, x + 10)] = [220, 225, 235]
            frame[58:62, :] = [55, 55, 60]
            frames.append(frame)

        profile = build_angular_seismic_profile_from_frames(
            frames,
            run_id="synthetic_reflection_motion",
            source_label="synthetic",
            fps=24.0,
            loop_count=1,
            grid_shape=(12, 16),
            rings=(1, 2, 3),
        )

        self.assertEqual(profile["schema_version"], "truevision_angular_seismic_profile_v0")
        self.assertEqual(profile["grid"]["direction_count"], 16)
        self.assertEqual(len(profile["angular_signature"]["radial_energy"]), 16)
        self.assertEqual(len(profile["angular_signature"]["director_energy"]), 16)
        self.assertGreater(profile["candidate_profiles"]["human_movement"]["peak"], 0.0)
        self.assertGreater(profile["candidate_profiles"]["glass_reflections"]["peak"], 0.0)
        self.assertTrue(profile["boundary"]["compact_profile_only"])

    def test_write_profile_preserves_source_video_and_writes_receipt(self):
        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "tiny.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48))
            self.assertTrue(writer.isOpened())
            for index in range(18):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[20:28, 8 + index : 20 + index] = [230, 230, 235]
                writer.write(frame)
            writer.release()

            result = write_angular_seismic_profile_from_video(
                {
                    "source_video": str(video),
                    "run_id": "tiny_profile",
                    "loop_count": 2,
                    "sample_stride": 2,
                    "max_frames": 18,
                    "grid_rows": 8,
                    "grid_cols": 12,
                },
                storage_root=root / "storage",
            )
            receipt = json.loads(Path(result["receipt_json"]).read_text(encoding="utf-8"))

            self.assertTrue(Path(result["profile_json"]).exists())
            self.assertTrue(Path(result["manifest_json"]).exists())
            self.assertTrue(Path(result["report_json"]).exists())
            self.assertTrue(Path(receipt["source_video"]).exists())
            self.assertTrue(receipt["source_video_preserved"])
            self.assertEqual(receipt["tool"], "angular_seismic_from_local_video")


if __name__ == "__main__":
    unittest.main()
