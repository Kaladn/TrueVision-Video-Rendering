import unittest
from pathlib import Path

from truevision_native_clarity_test import build_clarity_plan


class TrueVisionNativeClarityTestTests(unittest.TestCase):
    def test_plan_keeps_capture_dumb_and_playback_separate(self):
        plan = build_clarity_plan(
            vault=Path("E:/TruEVision Generation"),
            duration=3.0,
            fps=9.0,
            resolution="2560x1440",
            grid="640x360",
            run_id="clarity_test",
        )

        self.assertTrue(plan["boundary"]["capture_loop_only"])
        self.assertTrue(plan["boundary"]["no_temporal_616"])
        self.assertTrue(plan["boundary"]["no_signature_analysis"])
        self.assertTrue(plan["boundary"]["no_replay_inside_capture"])
        self.assertEqual(plan["capture"]["pixels_per_cell"], [4, 4])
        self.assertIn("--duration", plan["capture"]["command"])
        self.assertIn("truevision_state_replay.py", " ".join(plan["replay"]["command"]))

    def test_plan_rejects_grid_that_cannot_replay_cleanly(self):
        with self.assertRaises(ValueError):
            build_clarity_plan(
                vault=Path("E:/TruEVision Generation"),
                duration=3.0,
                fps=9.0,
                resolution="2560x1440",
                grid="560x315",
                run_id="bad_grid",
            )


if __name__ == "__main__":
    unittest.main()
