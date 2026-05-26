import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_runtime.learning_intake.lightfield_focus import (
    build_lightfield_focus_profile_from_native_capture,
    detect_active_bounds,
    refocus_lightfield_planes,
    write_state_focus_lens_from_capture,
)


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


def write_lightfield_teacher_capture(root: Path, *, run_id: str = "lightfield_teacher") -> Path:
    run_dir = root / run_id
    cell_dir = run_dir / "cell_state_native"
    cell_dir.mkdir(parents=True)
    frames = 9
    rows = 8
    cols = 10
    feature_count = len(FEATURE_NAMES)
    cells = np.zeros((frames, rows, cols, feature_count), dtype="<f4")
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    for frame_index in range(frames):
        angle = frame_index - frames // 2
        x_center = 4.0 + angle * 0.25
        y_center = 4.0
        beam = np.exp(-(((x_grid - x_center) ** 2) / 0.7 + ((y_grid - y_center) ** 2) / 7.5))
        phone_video_mask = np.zeros((rows, cols), dtype=np.float32)
        phone_video_mask[:, 3:7] = 1.0
        luma = (0.03 + 0.9 * beam) * phone_video_mask
        edge = (0.05 + 0.5 * beam) * phone_video_mask
        texture = (0.04 + 0.4 * beam) * phone_video_mask
        motion = (0.02 + 0.6 * beam) * phone_video_mask
        cells[frame_index, :, :, FEATURE_NAMES.index("rgb_mean_r")] = luma * 0.2
        cells[frame_index, :, :, FEATURE_NAMES.index("rgb_mean_g")] = luma * 0.6
        cells[frame_index, :, :, FEATURE_NAMES.index("rgb_mean_b")] = luma
        cells[frame_index, :, :, FEATURE_NAMES.index("luma_mean")] = luma
        cells[frame_index, :, :, FEATURE_NAMES.index("luma_std")] = luma * 0.4
        cells[frame_index, :, :, FEATURE_NAMES.index("saturation_mean")] = phone_video_mask * 0.7
        cells[frame_index, :, :, FEATURE_NAMES.index("delta_luma_abs")] = motion
        cells[frame_index, :, :, FEATURE_NAMES.index("edge_density")] = edge
        cells[frame_index, :, :, FEATURE_NAMES.index("texture_energy")] = texture
        cells[frame_index, :, :, FEATURE_NAMES.index("motion_energy")] = motion
    chunk = cell_dir / f"{run_id}_cells_0000.tvcells"
    chunk.write_bytes(cells.tobytes())
    records = run_dir / f"{run_id}_records.jsonl"
    with records.open("w", encoding="utf-8") as handle:
        for frame_index in range(frames):
            handle.write(json.dumps({"frame_number": frame_index + 1, "elapsed_seconds": frame_index / 15.0}) + "\n")
    manifest = {
        "schema_version": 1,
        "record_kind": "truevision_native_rs_frame_state",
        "run_id": run_id,
        "records_jsonl": str(records),
        "config": {
            "duration_seconds": frames / 15.0,
            "capture_fps": 15,
            "capture_resolution": [cols * 4, rows * 4],
            "grid_size_xy": [cols, rows],
            "capture_region": [0, 0, cols * 4, rows * 4],
            "cell_chunk_frames": frames,
        },
        "summary": {"frame_count": frames, "duration_seconds": frames / 15.0},
        "cell_state": {
            "enabled": True,
            "format": "tvcells_f32le_v1",
            "feature_names": FEATURE_NAMES,
            "chunks": [
                {
                    "chunk_id": 0,
                    "path": str(chunk),
                    "format": "tvcells_f32le_v1",
                    "frames": frames,
                    "grid_shape": [rows, cols],
                    "feature_count": feature_count,
                }
            ],
        },
        "boundary": {"raw_frame_saved": False},
    }
    manifest_path = run_dir / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


class LightfieldFocusTests(unittest.TestCase):
    def test_refocus_lightfield_planes_aligns_angular_samples(self):
        base = np.zeros((5, 7), dtype=np.float32)
        angular_samples = []
        for theta in [-1.0, 0.0, 1.0]:
            plane = np.copy(base)
            plane[:, int(3 + theta)] = 1.0
            angular_samples.append({"theta": theta, "phi": 0.0, "plane": plane})

        refocused = refocus_lightfield_planes(angular_samples, focus_depth=1.0)
        unfocused = refocus_lightfield_planes(angular_samples, focus_depth=0.0)

        self.assertGreater(refocused["focus_score"], unfocused["focus_score"])
        self.assertGreater(refocused["peak_intensity"], 0.95)

    def test_detect_active_bounds_ignores_letterbox_and_pillarbox(self):
        plane = np.zeros((8, 10), dtype=np.float32)
        plane[:, 3:7] = 0.7

        bounds = detect_active_bounds(plane)

        self.assertEqual(bounds["grid_xywh"], [3, 0, 4, 8])
        self.assertEqual(bounds["orientation"], "vertical_phone")

    def test_profile_records_broad_capture_and_later_focus_planes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_lightfield_teacher_capture(Path(tmp))

            profile = build_lightfield_focus_profile_from_native_capture(
                manifest,
                element_id="light_show_effects",
                max_frames=9,
            )

        self.assertEqual(profile["schema_version"], "truevision_lightfield_focus_profile_v1")
        self.assertEqual(profile["element_id"], "light_show_effects")
        self.assertEqual(profile["capture_policy"]["record_broad_focus_later"], True)
        self.assertEqual(profile["active_bounds"]["orientation"], "vertical_phone")
        self.assertGreaterEqual(len(profile["focus_planes"]), 3)
        self.assertIn("L(x,y,theta,phi)", profile["lightfield_model"]["formula"])
        self.assertEqual(profile["retention"]["raw_teacher_state_required_after_profile"], False)

    def test_write_state_focus_lens_profile_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_lightfield_teacher_capture(root)

            result = write_state_focus_lens_from_capture(
                {
                    "manifest": str(manifest),
                    "element_id": "light_show_effects",
                    "run_id": "light_show_focus_test",
                    "max_frames": 9,
                },
                storage_root=root / "storage",
            )

            profile = json.loads(Path(result["profile_json"]).read_text(encoding="utf-8"))
            receipt = json.loads(Path(result["receipt_json"]).read_text(encoding="utf-8"))

        self.assertEqual(profile["schema_version"], "truevision_lightfield_focus_profile_v1")
        self.assertEqual(receipt["schema_version"], "truevision_state_focus_lens_receipt_v1")
        self.assertEqual(receipt["boundary"]["capture_wide_focus_later"], True)
        self.assertEqual(result["active_bounds"]["orientation"], "vertical_phone")


if __name__ == "__main__":
    unittest.main()
