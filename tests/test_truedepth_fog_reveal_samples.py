import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from render_truedepth_fog_reveal_samples import _depth_haze_density, build_sample_plan, render_frame, render_video


class _FakeProcess:
    def __init__(self):
        self.stdin = BytesIO()

    def wait(self):
        return 0


class TrueDepthFogRevealSamplesTests(unittest.TestCase):
    def test_sample_plan_has_isolated_tools_and_combined(self):
        plan = build_sample_plan()
        self.assertEqual(
            [item["mode"] for item in plan["samples"]],
            [
                "fog_field",
                "volumetric_fog",
                "forward_motion",
                "truedepth",
                "object_reveal",
                "angular_drift",
                "effect_state_transform",
                "combined",
            ],
        )
        for sample in plan["samples"][:-1]:
            self.assertEqual(len(sample["active_tools"]), 1)
        self.assertEqual(len(plan["samples"][-1]["active_tools"]), 7)

    def test_render_frame_returns_landscape_rgb_frame(self):
        frame = render_frame(
            mode="object_reveal",
            frame_index=30,
            total_frames=120,
            width=320,
            height=180,
            signature={"fog_softness": 0.75, "direction_angle_degrees": 315.0},
        )
        self.assertEqual(frame.shape, (180, 320, 3))
        self.assertEqual(frame.dtype.name, "uint8")

    def test_effect_state_transform_frame_is_supported(self):
        frame = render_frame(
            mode="effect_state_transform",
            frame_index=45,
            total_frames=120,
            width=320,
            height=180,
            signature={"fog_softness": 0.75, "direction_angle_degrees": 315.0, "motion_pressure": 0.04},
        )
        self.assertEqual(frame.shape, (180, 320, 3))
        self.assertGreater(frame.mean(), 0.0)

    def test_depth_haze_thickens_toward_distance(self):
        haze = _depth_haze_density(
            320,
            180,
            0.5,
            {"fog_softness": 0.75, "direction_angle_degrees": 315.0, "motion_pressure": 0.04},
        )
        far_band = haze[55:85, 120:200].mean()
        near_band = haze[145:175, 120:200].mean()
        self.assertGreater(far_band, near_band)

    def test_vehicle_speed_smooths_stationary_splotches(self):
        signature = {"fog_softness": 0.75, "direction_angle_degrees": 315.0, "motion_pressure": 0.04}
        slow = _depth_haze_density(320, 180, 0.5, signature, vehicle_speed_mph=25.0)
        fast = _depth_haze_density(320, 180, 0.5, signature, vehicle_speed_mph=55.0)
        slow_laplacian = abs(slow[:, 1:] - slow[:, :-1]).mean()
        fast_laplacian = abs(fast[:, 1:] - fast[:, :-1]).mean()
        self.assertLess(fast_laplacian, slow_laplacian)

    def test_render_video_writes_frame_state_jsonl_for_tooling(self):
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "fog.mp4"
            with patch("render_truedepth_fog_reveal_samples.subprocess.Popen", return_value=_FakeProcess()):
                result = render_video(
                    mode="combined",
                    output_path=output_path,
                    signature={"fog_softness": 0.75, "direction_angle_degrees": 315.0, "motion_pressure": 0.04},
                    duration=0.2,
                    fps=10,
                    width=64,
                    height=36,
                    encoder="libx264",
                    label=False,
                    vehicle_speed_mph=55.0,
                )

            state_path = Path(result["frame_state_jsonl"])
            self.assertTrue(state_path.exists())
            lines = state_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('"frame_index": 0', lines[0])
            self.assertIn('"time_seconds": 0.0', lines[0])
            self.assertEqual(result["state_log_every"], 1)


if __name__ == "__main__":
    unittest.main()
