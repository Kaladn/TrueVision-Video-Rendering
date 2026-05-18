import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_resonance_recorder import (
    build_record,
    parse_region,
    parse_shape_xy,
    summarize_records,
    write_capture_bundle,
)


class TrueVisionResonanceRecorderTests(unittest.TestCase):
    def test_build_record_persists_visual_resonance_without_raw_frame(self):
        features = {
            "frame": np.zeros((4, 4, 3), dtype=np.uint8),
            "grid": np.ones((4, 4), dtype=np.float32),
            "blocks": np.arange(100, dtype=np.float32).reshape(10, 10),
            "block_deltas": np.ones((10, 10), dtype=np.float32) * 0.25,
            "block_vector": np.ones(100, dtype=np.float32) * 0.25,
            "visual_resonance": {
                "vis_energy_total": 1.5,
                "vis_stutter_score": float("nan"),
            },
            "timestamp": 123.4,
            "fps": 15.0,
            "frame_number": 7,
            "capture_geometry": {
                "source_width": 960,
                "source_height": 540,
                "frame_width": 960,
                "frame_height": 540,
                "grid_rows": 90,
                "grid_cols": 160,
                "block_rows": 9,
                "block_cols": 16,
            },
        }

        record = build_record(
            features,
            run_id="run-test",
            elapsed_seconds=0.5,
            include_blocks=True,
        )

        self.assertEqual(record["record_kind"], "compucogvision_full_frame_state")
        self.assertEqual(record["run_id"], "run-test")
        self.assertEqual(record["visual_resonance"]["vis_energy_total"], 1.5)
        self.assertIsNone(record["visual_resonance"]["vis_stutter_score"])
        self.assertEqual(record["geometry"]["frame_shape"], [540, 960])
        self.assertEqual(record["geometry"]["grid_shape"], [90, 160])
        self.assertEqual(record["geometry"]["block_shape"], [9, 16])
        self.assertEqual(len(record["block_vector"]), 100)
        self.assertEqual(len(record["blocks"]), 10)
        self.assertNotIn("frame", record)
        self.assertNotIn("grid", record)
        json.dumps(record, allow_nan=False)

    def test_parse_video_shapes_use_width_by_height_cli_order(self):
        self.assertEqual(parse_shape_xy("160x90"), (90, 160))
        self.assertEqual(parse_shape_xy("16,9"), (9, 16))
        self.assertEqual(parse_region("10,20,960,540"), (10, 20, 960, 540))

    def test_write_capture_bundle_writes_records_manifest_and_summary(self):
        records = [
            {
                "record_kind": "compucogvision_full_frame_state",
                "run_id": "run-test",
                "timestamp_unix": 1.0,
                "elapsed_seconds": 0.0,
                "frame_number": 1,
                "fps": 0.0,
                "screen_energy": 0.0,
                "visual_resonance": {"vis_energy_total": 0.0},
            },
            {
                "record_kind": "compucogvision_full_frame_state",
                "run_id": "run-test",
                "timestamp_unix": 2.0,
                "elapsed_seconds": 1.0,
                "frame_number": 2,
                "fps": 15.0,
                "screen_energy": 4.0,
                "visual_resonance": {"vis_energy_total": 2.0},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = write_capture_bundle(
                output_root=Path(tmpdir),
                run_id="run-test",
                records=records,
                config={"duration_seconds": 60},
            )

            self.assertTrue(bundle["records_jsonl"].exists())
            self.assertTrue(bundle["manifest_json"].exists())
            self.assertTrue(bundle["summary_json"].exists())
            self.assertEqual(summarize_records(records)["frame_count"], 2)
            manifest = json.loads(bundle["manifest_json"].read_text())
            self.assertEqual(manifest["run_id"], "run-test")
            self.assertEqual(manifest["records"]["frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
