import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_runtime.av_tools.av_tool_runner import run_av_tool_call
from truevision_runtime.state_patterns.atmosphere_weather import (
    build_atmosphere_profile_from_native_capture,
    build_atmosphere_toolset,
    list_atmosphere_elements,
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


def write_tiny_native_capture(root: Path) -> Path:
    run_dir = root / "capture"
    cell_dir = run_dir / "cell_state_native"
    cell_dir.mkdir(parents=True)
    frames = 13
    rows = 2
    cols = 3
    feature_count = len(FEATURE_NAMES)
    cells = np.zeros((frames, rows, cols, feature_count), dtype="<f4")
    for frame_index in range(frames):
        ramp = frame_index / (frames - 1)
        cells[frame_index, :, :, FEATURE_NAMES.index("luma_mean")] = 0.25 + 0.55 * ramp
        cells[frame_index, :, :, FEATURE_NAMES.index("luma_std")] = 0.08 + 0.03 * ramp
        cells[frame_index, :, :, FEATURE_NAMES.index("edge_density")] = 0.36 - 0.12 * ramp
        cells[frame_index, :, :, FEATURE_NAMES.index("texture_energy")] = 0.18 + 0.08 * ramp
        cells[frame_index, :, :, FEATURE_NAMES.index("motion_energy")] = 0.05 + 0.22 * ramp
        cells[frame_index, :, :, FEATURE_NAMES.index("hsv_mean_s")] = 0.22 + 0.05 * ramp
    chunk = cell_dir / "tiny_cells_0000.tvcells"
    chunk.write_bytes(cells.tobytes())
    records = run_dir / "tiny_records.jsonl"
    with records.open("w", encoding="utf-8") as handle:
        for frame_index in range(frames):
            handle.write(json.dumps({"frame_number": frame_index + 1, "elapsed_seconds": frame_index / 10.0}) + "\n")
    manifest = {
        "schema_version": 1,
        "record_kind": "truevision_native_rs_frame_state",
        "run_id": "tiny_weather_capture",
        "records_jsonl": str(records),
        "config": {
            "duration_seconds": 1.3,
            "capture_fps": 10,
            "capture_resolution": [6, 4],
            "grid_size_xy": [cols, rows],
            "capture_region": [0, 0, 6, 4],
            "cell_chunk_frames": frames,
        },
        "summary": {"frame_count": frames, "duration_seconds": 1.3},
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
    manifest_path = run_dir / "tiny_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


class AtmosphereWeatherToolsetTests(unittest.TestCase):
    def test_lists_core_weather_elements_with_render_channels(self):
        elements = {item["element_id"]: item for item in list_atmosphere_elements()}

        self.assertIn("fog_density_field", elements)
        self.assertIn("mist_veil_field", elements)
        self.assertIn("cloud_volume_field", elements)
        self.assertIn("rain_glass_field", elements)
        self.assertIn("density", elements["fog_density_field"]["state_channels"])
        self.assertIn("refraction", elements["rain_glass_field"]["state_channels"])
        self.assertTrue(elements["fog_density_field"]["boundary"]["state_first_pixels_last"])
        self.assertFalse(elements["rain_glass_field"]["boundary"]["generated_media_is_evidence"])

    def test_builds_profile_from_native_capture_with_616_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_tiny_native_capture(Path(tmp))

            profile = build_atmosphere_profile_from_native_capture(
                manifest,
                element_id="fog_density_field",
                max_frames=13,
            )

        self.assertEqual(profile["schema_version"], "truevision_atmosphere_profile_v1")
        self.assertEqual(profile["source"]["run_id"], "tiny_weather_capture")
        self.assertEqual(profile["element_id"], "fog_density_field")
        self.assertEqual(profile["sampled_frames"], 13)
        self.assertEqual(profile["six_one_six"]["radius"], 6)
        self.assertEqual(len(profile["six_one_six"]["windows"]), 1)
        center = profile["six_one_six"]["windows"][0]
        self.assertEqual(center["center_frame_index"], 6)
        self.assertGreater(center["future"]["density_mean"], center["prior"]["density_mean"])
        self.assertIn("density_mean", profile["summary"])
        self.assertIn("motion_mean", profile["summary"])

    def test_toolset_write_preserves_profiles_and_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            manifest = write_tiny_native_capture(Path(tmp))

            result = build_atmosphere_toolset(
                storage_root=storage,
                run_id="weather_lane",
                capture_manifest=manifest,
                element_ids=["fog_density_field", "rain_glass_field"],
            )

            template = json.loads(Path(result["template_json"]).read_text(encoding="utf-8"))
            manifest_payload = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))

        self.assertEqual(template["template_id"], "weather_lane")
        self.assertEqual([item["element_id"] for item in template["elements"]], ["fog_density_field", "rain_glass_field"])
        self.assertEqual(manifest_payload["toolset"]["element_count"], 2)
        self.assertTrue(Path(result["capture_profile_json"]).name.endswith("_capture_profile.json"))
        self.assertFalse(manifest_payload["boundary"]["raw_frame_saved"])

    def test_av_tool_runner_routes_atmosphere_toolset(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            result = run_av_tool_call(
                {
                    "tool": "atmosphere_toolset_create",
                    "args": {
                        "run_id": "runner_weather",
                        "elements": ["mist_veil_field", "cloud_volume_field"],
                    },
                },
                storage_root=storage,
            )
            template_exists = Path(result["result"]["template_json"]).exists()

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["result"]["run_id"], "runner_weather")
        self.assertEqual(result["result"]["element_count"], 2)
        self.assertTrue(template_exists)


if __name__ == "__main__":
    unittest.main()
