from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from truevision_runtime.logger_roundup import (
    analyze_deep_pixel_transform,
    build_logger_roundup_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(pixels, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise AssertionError(f"could not write {path}")


class TrueVisionLoggerRoundupTests(unittest.TestCase):
    def test_roundup_catalogs_all_logger_lanes_without_616_mapping(self):
        manifest = build_logger_roundup_manifest(ROOT)

        self.assertEqual(manifest["schema_version"], "truevision_logger_roundup_manifest_v1")
        self.assertFalse(manifest["boundary"]["six_one_six_mapping_enabled"])
        self.assertEqual(manifest["boundary"]["mapping_policy"], "zero_6_1_6_mapping")

        lane_ids = {lane["lane_id"] for lane in manifest["logger_lanes"]}
        expected = {
            "native_rust_cell_state_capture",
            "meter_grid_from_capture",
            "angular_seismic_16_direction",
            "state_focus_lens",
            "truedepth_contracts",
            "atmosphere_weather_profiles",
            "element_creation_profile",
            "driving_school_awareness",
            "high_speed_awareness",
            "geometry_generation",
            "trueaudio_file_state_logging",
            "trueaudio_machine_state_logging",
            "timing_audit",
            "state_source_law",
            "state_replay",
            "state_media_qa",
            "av_tool_registry",
            "deep_pixel_transform_analysis",
        }
        self.assertTrue(expected.issubset(lane_ids))
        self.assertIn("scripts/truevision_meter_grid.py", manifest["entrypoints"])
        self.assertIn("native/truevision_capture_rs/src/main.rs", manifest["entrypoints"])
        self.assertGreaterEqual(len(manifest["discovered_logger_files"]), len(expected))
        self.assertFalse(manifest["boundary"]["capture_started"])
        self.assertFalse(manifest["boundary"]["render_started"])

        encoded = json.dumps(manifest, sort_keys=True)
        self.assertNotIn('"6-1-6"', encoded)
        self.assertNotIn('"six_one_six"', encoded.replace('"six_one_six_mapping_enabled": false', ""))

    def test_deep_pixel_transform_analysis_reports_existing_pixel_changes_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = np.zeros((16, 16, 3), dtype=np.uint8)
            source[:, :] = [12, 12, 16]
            source[9:15, 4:12] = [80, 44, 18]
            source[2:5, 2:14] = [190, 160, 82]

            transformed = source.copy()
            transformed[9:15, 4:12] = [150, 82, 32]
            transformed[2:5, 2:14] = [220, 200, 128]

            source_path = root / "source.png"
            transformed_path = root / "transformed.png"
            _write_png(source_path, source)
            _write_png(transformed_path, transformed)

            analysis = analyze_deep_pixel_transform(source_path, transformed_path)

        self.assertEqual(analysis["schema_version"], "truevision_deep_pixel_transform_analysis_v1")
        self.assertTrue(analysis["boundary"]["source_pixel_transform_only"])
        self.assertFalse(analysis["boundary"]["added_artifact_detection_claim"])
        self.assertFalse(analysis["boundary"]["six_one_six_mapping_enabled"])
        self.assertGreater(analysis["global_delta"]["changed_pixel_ratio"], 0.0)
        operator_ids = {operator["operator_id"] for operator in analysis["transform_operators"]}
        self.assertIn("luminance_rise_fall", operator_ids)
        self.assertIn("hue_saturation_pressure", operator_ids)
        self.assertIn("edge_contrast_recovery", operator_ids)
        self.assertIn("warm_sun_sky_breath", operator_ids)
        self.assertGreaterEqual(len(analysis["material_regions"]), 3)

    def test_cli_writes_single_roundup_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "roundup.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/truevision_logger_roundup.py",
                    "--repo-root",
                    str(ROOT),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            stdout = json.loads(result.stdout)

        self.assertEqual(payload["schema_version"], "truevision_logger_roundup_manifest_v1")
        self.assertEqual(stdout["manifest_json"], str(output))
        self.assertFalse(payload["boundary"]["six_one_six_mapping_enabled"])


if __name__ == "__main__":
    unittest.main()
