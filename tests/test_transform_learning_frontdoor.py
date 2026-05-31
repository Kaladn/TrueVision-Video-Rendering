from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from truevision_runtime.transform_learning_frontdoor import (
    build_transform_behavior_profile,
    compare_generated_transform_to_profile,
    run_transform_learning_cycle,
)


ROOT = Path(__file__).resolve().parents[1]


class TransformLearningFrontdoorTests(unittest.TestCase):
    def test_profile_learns_behavior_not_source_shape(self):
        observed = [
            {
                "event_id": "source_lightning_001",
                "candidate_type": "candidate_lightning",
                "source_region": {"frame_start": 10, "frame_peak": 11, "frame_end": 21},
                "geometry": {
                    "points": [[0.1, 0.2], [0.4, 0.5], [0.8, 0.3]],
                    "source_shape_id": "actual_bolt_shape_do_not_copy",
                },
                "true_local_metrics": {
                    "luma_delta": 0.82,
                    "rise_time_frames": 1.0,
                    "falloff_frames": 9.0,
                    "bloom_radius_cells": 12.0,
                    "surrounding_exposure_lift": 0.43,
                    "branch_edge_density": 0.76,
                    "branch_direction_variance": 0.58,
                    "afterglow_decay_rate": 0.72,
                },
            }
        ]

        profile = build_transform_behavior_profile(observed, transform_kind="lightning")

        self.assertEqual(profile["schema_version"], "truevision_transform_behavior_profile_v1")
        self.assertEqual(profile["transform_kind"], "lightning")
        self.assertTrue(profile["boundary"]["copy_behavior_not_pixels"])
        self.assertFalse(profile["boundary"]["source_shape_copy_allowed"])
        self.assertFalse(profile["boundary"]["yolo_truth_authority"])
        self.assertIn("luma_delta", profile["behavior_metrics"])
        encoded = json.dumps(profile, sort_keys=True)
        self.assertNotIn("actual_bolt_shape_do_not_copy", encoded)
        self.assertNotIn("[[0.1, 0.2]", encoded)

    def test_compare_generated_attempt_recommends_behavior_adjustments(self):
        profile = build_transform_behavior_profile(
            [
                {
                    "event_id": "source_lightning_001",
                    "true_local_metrics": {
                        "luma_delta": 0.80,
                        "rise_time_frames": 1.0,
                        "falloff_frames": 8.0,
                        "bloom_radius_cells": 10.0,
                        "surrounding_exposure_lift": 0.40,
                        "branch_edge_density": 0.70,
                        "branch_direction_variance": 0.55,
                        "afterglow_decay_rate": 0.75,
                    },
                }
            ],
            transform_kind="lightning",
        )
        generated = {
            "attempt_id": "generated_lightning_attempt_001",
            "generated_geometry_id": "new_bolt_seed_8492",
            "metrics": {
                "luma_delta": 0.62,
                "rise_time_frames": 4.0,
                "falloff_frames": 18.0,
                "bloom_radius_cells": 4.0,
                "surrounding_exposure_lift": 0.18,
                "branch_edge_density": 0.40,
                "branch_direction_variance": 0.20,
                "afterglow_decay_rate": 0.30,
            },
        }

        comparison = compare_generated_transform_to_profile(profile, generated, tolerance=0.12)

        self.assertEqual(comparison["schema_version"], "truevision_generated_transform_comparison_v1")
        self.assertFalse(comparison["accepted"])
        self.assertGreater(comparison["score"]["mean_relative_error"], 0.12)
        adjustment_ids = {item["adjustment_id"] for item in comparison["adjustments"]}
        self.assertIn("increase_bloom_radius_cells", adjustment_ids)
        self.assertIn("decrease_rise_time_frames", adjustment_ids)
        self.assertIn("increase_surrounding_exposure_lift", adjustment_ids)
        self.assertTrue(comparison["boundary"]["generated_shape_may_differ"])
        self.assertTrue(comparison["boundary"]["behavior_match_required"])

    def test_learning_cycle_and_cli_write_manifest_and_receipt(self):
        observed = [
            {
                "event_id": "source_lightning_001",
                "true_local_metrics": {
                    "luma_delta": 0.80,
                    "rise_time_frames": 1.0,
                    "falloff_frames": 8.0,
                    "bloom_radius_cells": 10.0,
                    "surrounding_exposure_lift": 0.40,
                    "branch_edge_density": 0.70,
                    "branch_direction_variance": 0.55,
                    "afterglow_decay_rate": 0.75,
                },
            }
        ]
        generated = [
            {
                "attempt_id": "attempt_bad",
                "metrics": {
                    "luma_delta": 0.55,
                    "rise_time_frames": 4.0,
                    "falloff_frames": 18.0,
                    "bloom_radius_cells": 4.0,
                    "surrounding_exposure_lift": 0.15,
                    "branch_edge_density": 0.30,
                    "branch_direction_variance": 0.20,
                    "afterglow_decay_rate": 0.25,
                },
            },
            {
                "attempt_id": "attempt_close",
                "metrics": {
                    "luma_delta": 0.79,
                    "rise_time_frames": 1.0,
                    "falloff_frames": 8.5,
                    "bloom_radius_cells": 9.4,
                    "surrounding_exposure_lift": 0.39,
                    "branch_edge_density": 0.68,
                    "branch_direction_variance": 0.53,
                    "afterglow_decay_rate": 0.72,
                },
            },
        ]
        cycle = run_transform_learning_cycle(observed, generated, transform_kind="lightning", tolerance=0.12)
        self.assertTrue(cycle["accepted"])
        self.assertEqual(cycle["best_attempt_id"], "attempt_close")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observed_path = root / "observed.json"
            generated_path = root / "generated.json"
            out = root / "out"
            observed_path.write_text(json.dumps(observed), encoding="utf-8")
            generated_path.write_text(json.dumps(generated), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/truevision_transform_learning_frontdoor.py",
                    "--observed-events-json",
                    str(observed_path),
                    "--generated-attempts-json",
                    str(generated_path),
                    "--transform-kind",
                    "lightning",
                    "--output-root",
                    str(out),
                    "--run-id",
                    "frontdoor_test",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            manifest = json.loads(Path(payload["manifest_json"]).read_text(encoding="utf-8"))
            receipt = json.loads(Path(payload["receipt_json"]).read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], "truevision_transform_learning_cycle_v1")
        self.assertTrue(manifest["accepted"])
        self.assertEqual(receipt["schema_version"], "truevision_transform_learning_frontdoor_receipt_v1")
        self.assertEqual(receipt["status"], "completed")


if __name__ == "__main__":
    unittest.main()
