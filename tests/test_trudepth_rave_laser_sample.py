import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from render_trudepth_rave_laser_sample import build_laser_show_plan, render_frame, render_video


class _FakeProcess:
    def __init__(self):
        self.stdin = BytesIO()

    def wait(self):
        return 0


class TruDepthRaveLaserSampleTests(unittest.TestCase):
    def test_plan_is_synthetic_state_media(self):
        plan = build_laser_show_plan()
        self.assertEqual(plan["schema_version"], "truevision_trudepth_rave_laser_show_plan_v1")
        self.assertIn("volumetric_haze_medium", plan["state_layers"])
        self.assertIn("collimated_laser_beams", plan["state_layers"])
        self.assertFalse(plan["boundary"]["source_video_frames_used"])

    def test_render_frame_returns_rgb_with_light_energy(self):
        frame, stats = render_frame(frame_index=30, total_frames=120, width=320, height=180)
        self.assertEqual(frame.shape, (180, 320, 3))
        self.assertEqual(frame.dtype.name, "uint8")
        self.assertGreater(frame.max(), 30)
        self.assertGreater(stats["beam_energy"], 0.0)

    def test_render_video_writes_frame_state_jsonl_for_tooling(self):
        with TemporaryDirectory() as tmp_dir:
            with patch("render_trudepth_rave_laser_sample.subprocess.Popen", return_value=_FakeProcess()):
                manifest = render_video(
                    output_root=Path(tmp_dir),
                    run_id="laser_state_log",
                    duration=0.2,
                    fps=10,
                    width=64,
                    height=36,
                    encoder="libx264",
                    label=False,
                )

            state_path = Path(manifest["output"]["frame_state_jsonl"])
            self.assertTrue(state_path.exists())
            lines = state_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('"frame_index": 0', lines[0])
            self.assertIn('"time_seconds": 0.0', lines[0])
            self.assertEqual(manifest["output"]["state_log_every"], 1)


if __name__ == "__main__":
    unittest.main()
