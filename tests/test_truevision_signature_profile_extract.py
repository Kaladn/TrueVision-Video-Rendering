import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_signature_profile_extract import extract_signature_profiles


FEATURE_NAMES = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "hsv_mean_h",
    "hsv_mean_s",
    "hsv_mean_v",
    "luma_mean",
    "luma_std",
    "saturation_mean",
    "delta_luma_abs",
    "edge_density",
    "texture_energy",
    "motion_energy",
]


class TrueVisionSignatureProfileExtractTests(unittest.TestCase):
    def _write_fake_capture(self, root: Path) -> Path:
        capture_dir = root / "capture"
        cell_dir = capture_dir / "cell_state_npz"
        cell_dir.mkdir(parents=True)
        frames = []
        for index in range(8):
            cells = np.zeros((4, 6, len(FEATURE_NAMES)), dtype=np.float32)
            cells[:, :, 0] = 30 + index * 3
            cells[:, :, 1] = 40 + index * 2
            cells[:, :, 2] = 55 + index
            cells[:, :, 6] = 80 + index
            cells[:, :, 7] = 95 + index * 2
            cells[:, :, 8] = 120 + index * 4
            cells[:, :, 9] = 65 + index * 4
            cells[:, :, 10] = 5 + index
            cells[:, :, 11] = 70 + index * 2
            cells[:, :, 13] = 0.04 + index * 0.01
            cells[:, :, 14] = 3 + index
            cells[:, :, 15] = 0.02 + index * 0.04
            hot_x = index % 6
            hot_y = (index // 2) % 4
            cells[hot_y, hot_x, 12] = 0.5 + 0.1 * index
            cells[hot_y, hot_x, 15] = 0.8 + 0.05 * index
            frames.append(cells)

        np.savez_compressed(
            cell_dir / "fake_cells_0000.npz",
            cell_state=np.stack(frames[:4]),
            frame_numbers=np.asarray([1, 2, 3, 4], dtype=np.int32),
            feature_names=np.asarray(FEATURE_NAMES),
            grid_shape=np.asarray([4, 6], dtype=np.int32),
        )
        np.savez_compressed(
            cell_dir / "fake_cells_0001.npz",
            cell_state=np.stack(frames[4:]),
            frame_numbers=np.asarray([5, 6, 7, 8], dtype=np.int32),
            feature_names=np.asarray(FEATURE_NAMES),
            grid_shape=np.asarray([4, 6], dtype=np.int32),
        )
        records_path = capture_dir / "fake_records.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for index in range(8):
                handle.write(
                    json.dumps(
                        {
                            "elapsed_seconds": index / 4,
                            "frame_number": index + 1,
                            "screen_energy": 100 + index * 50,
                            "visual_resonance": {
                                "vis_flash_intensity": index / 7,
                                "vis_contrast_shift_score": index / 50,
                                "vis_jitter_band_energy": index * 0.25,
                                "vis_smoothness_index": 1.0 - index * 0.03,
                            },
                        }
                    )
                    + "\n"
                )
        (capture_dir / "fake_manifest.json").write_text(
            json.dumps(
                {
                    "kind": "compucogvision_capture_manifest",
                    "run_id": "fake_capture",
                    "records": {"frame_count": 8, "jsonl_path": str(records_path)},
                    "cell_state": {"enabled": True, "feature_names": FEATURE_NAMES},
                    "config": {"capture_fps": 4, "capture_resolution": [96, 54]},
                }
            ),
            encoding="utf-8",
        )
        return capture_dir

    def test_extract_signature_profiles_writes_reusable_profile_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture_dir = self._write_fake_capture(root)
            result = extract_signature_profiles(
                capture_dir=capture_dir,
                output_dir=root / "profiles",
                profile_id="unit_signature",
                max_timeline_samples=20,
            )
            bundle = json.loads(Path(result["bundle_json"]).read_text(encoding="utf-8"))
            motion = json.loads(Path(result["motion_profile_json"]).read_text(encoding="utf-8"))
            camera = json.loads(Path(result["camera_shake_profile_json"]).read_text(encoding="utf-8"))
            color = json.loads(Path(result["contrast_color_profile_json"]).read_text(encoding="utf-8"))
            cuts = json.loads(Path(result["cut_rhythm_profile_json"]).read_text(encoding="utf-8"))

        self.assertEqual(bundle["kind"], "truevision_signature_profile_bundle")
        self.assertEqual(bundle["profile_id"], "unit_signature")
        self.assertGreater(bundle["source"]["frame_count"], 0)
        self.assertGreater(len(bundle["timeline_samples"]), 0)
        self.assertIn("motion_profile", bundle["profiles"])
        self.assertGreater(motion["stats"]["motion_mean"]["max"], motion["stats"]["motion_mean"]["min"])
        self.assertGreater(camera["stats"]["shake_magnitude"]["max"], 0.0)
        self.assertGreater(color["stats"]["luma_mean"]["max"], color["stats"]["luma_mean"]["min"])
        self.assertIn("cut_events", cuts)


if __name__ == "__main__":
    unittest.main()
