import tempfile
import unittest
from pathlib import Path

from truevision_youtube_learning_intake_batch import _purge_unverified_profile_artifacts, _visual_temporal_change_score


class YouTubeLearningBatchScriptTests(unittest.TestCase):
    def test_temporal_change_score_accepts_slow_center_drift(self):
        score = _visual_temporal_change_score(
            {
                "transition_behavior": {"motion_mean": 0.0, "motion_abs_mean": 0.0},
                "shape_behavior": {"center_drift_xy": [-0.012, -0.004]},
                "growth_decay": {"volatility": 0.0},
            }
        )

        self.assertEqual(score, 0.012)

    def test_temporal_change_score_stays_zero_for_static_profile(self):
        score = _visual_temporal_change_score(
            {
                "transition_behavior": {"motion_mean": 0.0, "motion_abs_mean": 0.0},
                "shape_behavior": {"center_drift_xy": [0.0, 0.0]},
                "growth_decay": {"volatility": 0.0},
            }
        )

        self.assertEqual(score, 0.0)

    def test_failed_sample_purges_unverified_profile_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.json"
            receipt_path = Path(tmp) / "receipt.json"
            profile_path.write_text("{}", encoding="utf-8")
            receipt_path.write_text("{}", encoding="utf-8")

            removed = _purge_unverified_profile_artifacts(
                {
                    "profile_json": str(profile_path),
                    "receipt_json": str(receipt_path),
                }
            )

        self.assertEqual(set(removed), {str(profile_path), str(receipt_path)})
        self.assertFalse(profile_path.exists())
        self.assertFalse(receipt_path.exists())


if __name__ == "__main__":
    unittest.main()
