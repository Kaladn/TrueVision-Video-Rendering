import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_runtime.av_tools.av_tool_runner import run_av_tool_call
from truevision_runtime.learning_intake.element_creation_profile import (
    build_element_creation_profile_from_native_capture,
    write_element_creation_profile_from_capture,
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


def write_tiny_teacher_capture(root: Path, *, run_id: str = "tiny_fire_teacher") -> Path:
    run_dir = root / run_id
    cell_dir = run_dir / "cell_state_native"
    cell_dir.mkdir(parents=True)
    frames = 13
    rows = 3
    cols = 4
    feature_count = len(FEATURE_NAMES)
    cells = np.zeros((frames, rows, cols, feature_count), dtype="<f4")
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    for frame_index in range(frames):
        phase = frame_index / (frames - 1)
        moving_hotspot = np.exp(-(((x_grid - (0.8 + phase * 1.8)) ** 2) + ((y_grid - 1.0) ** 2)) / 1.4)
        luma = 0.18 + 0.58 * moving_hotspot + 0.08 * phase
        motion = 0.05 + 0.35 * moving_hotspot + 0.18 * np.sin(phase * np.pi) ** 2
        edge = 0.18 + 0.24 * moving_hotspot
        texture = 0.14 + 0.18 * moving_hotspot + 0.06 * phase
        cells[frame_index, :, :, FEATURE_NAMES.index("rgb_mean_r")] = 0.35 + 0.42 * moving_hotspot
        cells[frame_index, :, :, FEATURE_NAMES.index("rgb_mean_g")] = 0.18 + 0.18 * moving_hotspot
        cells[frame_index, :, :, FEATURE_NAMES.index("rgb_mean_b")] = 0.08 + 0.04 * moving_hotspot
        cells[frame_index, :, :, FEATURE_NAMES.index("luma_mean")] = luma
        cells[frame_index, :, :, FEATURE_NAMES.index("luma_std")] = 0.08 + 0.12 * moving_hotspot
        cells[frame_index, :, :, FEATURE_NAMES.index("saturation_mean")] = 0.34 + 0.42 * moving_hotspot
        cells[frame_index, :, :, FEATURE_NAMES.index("delta_luma_abs")] = 0.03 + 0.22 * moving_hotspot
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


class ElementCreationProfileTests(unittest.TestCase):
    def test_profile_keeps_creation_useful_fields_and_616_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_tiny_teacher_capture(Path(tmp))

            profile = build_element_creation_profile_from_native_capture(
                manifest,
                element_id="fire_flame_licks",
                max_frames=13,
            )

        self.assertEqual(profile["schema_version"], "truevision_element_creation_profile_v1")
        self.assertEqual(profile["element_id"], "fire_flame_licks")
        self.assertEqual(profile["sampled_frames"], 13)
        self.assertEqual(profile["six_one_six"]["shape"], "6-1-6")
        self.assertEqual(profile["six_one_six"]["window_count"], 1)
        signature = profile["creation_signature"]
        for key in [
            "shape_behavior",
            "growth_decay",
            "edge_softness",
            "density_opacity",
            "bloom_intensity",
            "occlusion_behavior",
            "rhythm_pulse",
            "transition_behavior",
            "camera_relation",
            "renderer_binding",
        ]:
            self.assertIn(key, signature)
        self.assertIn("profile_sha256", profile)
        self.assertFalse(profile["retention"]["durable_teacher_state"])

    def test_profile_keeps_absolute_motion_when_motion_is_steady(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_tiny_teacher_capture(Path(tmp))

            profile = build_element_creation_profile_from_native_capture(
                manifest,
                element_id="steady_motion_teacher",
                max_frames=13,
            )

        signature = profile["creation_signature"]
        self.assertIn("motion_abs_mean", signature["transition_behavior"])
        self.assertGreater(signature["transition_behavior"]["motion_abs_mean"], 0.0)
        self.assertGreater(
            max(frame["motion_absolute"] for frame in profile["creation_frames"]),
            0.0,
        )

    def test_write_profile_can_purge_teacher_state_after_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_tiny_teacher_capture(root)
            original = json.loads(manifest.read_text(encoding="utf-8"))
            chunk_path = Path(original["cell_state"]["chunks"][0]["path"])
            records_path = Path(original["records_jsonl"])

            result = write_element_creation_profile_from_capture(
                {
                    "manifest": str(manifest),
                    "element_id": "smoke_curl_field",
                    "run_id": "smoke_tool",
                    "purge_teacher_state": True,
                },
                storage_root=root / "storage",
            )

            profile = json.loads(Path(result["profile_json"]).read_text(encoding="utf-8"))
            receipt = json.loads(Path(result["receipt_json"]).read_text(encoding="utf-8"))
            purge_report = json.loads(Path(result["purge_report_json"]).read_text(encoding="utf-8"))

        self.assertEqual(result["purge"]["status"], "purged")
        self.assertFalse(chunk_path.exists())
        self.assertFalse(records_path.exists())
        self.assertEqual(profile["element_id"], "smoke_curl_field")
        self.assertEqual(receipt["profile_sha256"], profile["profile_sha256"])
        self.assertEqual(purge_report["deleted_file_count"], 2)
        self.assertTrue(purge_report["profile_verified_before_purge"])

    def test_av_tool_runner_routes_profile_and_purge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_tiny_teacher_capture(root)

            result = run_av_tool_call(
                {
                    "tool": "element_creation_profile_from_capture",
                    "args": {
                        "manifest": str(manifest),
                        "element_id": "lightning_branch_flash",
                        "run_id": "lightning_tool",
                        "purge_teacher_state": True,
                    },
                },
                storage_root=root / "storage",
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["result"]["element_id"], "lightning_branch_flash")
        self.assertEqual(result["result"]["purge"]["status"], "purged")
        self.assertGreaterEqual(result["result"]["six_one_six_windows"], 1)

    def test_three_source_process_leaves_profiles_not_teacher_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            element_ids = ["fire_flame_licks", "smoke_curl_field", "rain_glass_field"]
            outputs = []
            teacher_files = []
            for index, element_id in enumerate(element_ids):
                manifest = write_tiny_teacher_capture(root, run_id=f"teacher_{index}")
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                teacher_files.append(Path(payload["cell_state"]["chunks"][0]["path"]))
                teacher_files.append(Path(payload["records_jsonl"]))
                result = run_av_tool_call(
                    {
                        "tool": "element_creation_profile_from_capture",
                        "args": {
                            "manifest": str(manifest),
                            "element_id": element_id,
                            "run_id": f"{element_id}_process_test",
                            "purge_teacher_state": True,
                        },
                    },
                    storage_root=storage,
                )
                self.assertTrue(result["ok"], result)
                outputs.append(result["result"])

            profiles = sorted((storage / "artifacts" / "element_creation_profiles").glob("*.json"))
            receipts = sorted((storage / "receipts" / "element_creation_profiles").glob("*.json"))
            reports = sorted((storage / "reports" / "element_creation_profiles").glob("*.json"))

        self.assertEqual(len(outputs), 3)
        self.assertEqual(len(profiles), 3)
        self.assertEqual(len(receipts), 3)
        self.assertEqual(len(reports), 3)
        self.assertTrue(all(not path.exists() for path in teacher_files))
        self.assertTrue(all(output["purge"]["status"] == "purged" for output in outputs))


if __name__ == "__main__":
    unittest.main()
