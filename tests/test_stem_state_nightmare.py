from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from render_stem_state_nightmare import (
    DEFAULT_GUITAR_LASER_ALPHA,
    build_generation_banner,
    build_lyric_burn_lines,
    build_stem_control_map,
    compute_frame_metrics,
    render_frame,
)


class StemStateNightmareTests(unittest.TestCase):
    def test_each_stem_controls_two_or_three_visual_lanes(self):
        mapping = build_stem_control_map()
        self.assertGreaterEqual(len(mapping), 6)
        for stem_name, lane_names in mapping.items():
            with self.subTest(stem_name=stem_name):
                self.assertGreaterEqual(len(lane_names), 2)
                self.assertLessEqual(len(lane_names), 3)

    def test_compute_frame_metrics_extracts_rms_and_onset(self):
        samples = {
            "Drums": np.concatenate([np.zeros(400, dtype=np.float32), np.ones(400, dtype=np.float32)]),
            "Bass": np.ones(800, dtype=np.float32) * 0.25,
        }
        metrics = compute_frame_metrics(samples, sample_rate=800, fps=4, duration=1.0)
        self.assertEqual(metrics["frame_count"], 4)
        self.assertEqual(len(metrics["frames"]), 4)
        self.assertIn("Drums", metrics["frames"][2]["stems"])
        self.assertGreater(metrics["frames"][2]["stems"]["Drums"]["onset"], 0.0)
        self.assertGreater(metrics["frames"][0]["stems"]["Bass"]["rms"], 0.0)

    def test_render_frame_returns_rgb_and_lane_log(self):
        mapping = build_stem_control_map()
        frame_state = {
            "frame_index": 4,
            "time_seconds": 0.133333333,
            "stems": {
                stem_name: {"rms": 0.5, "onset": 0.25, "bass": 0.2, "mid": 0.4, "high": 0.3}
                for stem_name in mapping
            },
        }
        frame, lane_log = render_frame(frame_state, width=320, height=180, stem_map=mapping)
        self.assertEqual(frame.shape, (180, 320, 3))
        self.assertEqual(frame.dtype.name, "uint8")
        self.assertGreater(frame.max(), 20)
        self.assertIn("Drums", lane_log["stem_controls"])

    def test_guitar_lasers_are_35_percent_translucent_and_banner_is_logged(self):
        mapping = build_stem_control_map()
        frame_state = {
            "frame_index": 12,
            "time_seconds": 0.4,
            "stems": {
                stem_name: {"rms": 0.55, "onset": 0.35, "bass": 0.25, "mid": 0.5, "high": 0.4}
                for stem_name in mapping
            },
        }
        _, lane_log = render_frame(frame_state, width=320, height=180, stem_map=mapping)
        self.assertAlmostEqual(lane_log["render_lanes"]["guitar_laser_alpha"], DEFAULT_GUITAR_LASER_ALPHA)
        self.assertAlmostEqual(DEFAULT_GUITAR_LASER_ALPHA, 0.35)
        self.assertIn("CORTEX EVOLVED", build_generation_banner())
        self.assertEqual(lane_log["banner"]["position"], "lower_scrolling")

    def test_vocal_stem_drives_fog_laser_lyric_burn(self):
        mapping = build_stem_control_map()
        lines = build_lyric_burn_lines("BECOMING THE WOLF\nSTATE MADE VISIBLE")
        frame_state = {
            "frame_index": 24,
            "time_seconds": 0.8,
            "stems": {
                stem_name: {"rms": 0.15, "onset": 0.05, "bass": 0.1, "mid": 0.1, "high": 0.1}
                for stem_name in mapping
            },
        }
        frame_state["stems"]["Vocals"] = {"rms": 0.82, "onset": 0.44, "bass": 0.1, "mid": 0.75, "high": 0.26}
        _, lane_log = render_frame(frame_state, width=420, height=240, stem_map=mapping, lyric_lines=lines)
        self.assertEqual(lane_log["lyric_burn"]["driver_stem"], "Vocals")
        self.assertTrue(lane_log["lyric_burn"]["fog_laser_beam"])
        self.assertGreater(lane_log["lyric_burn"]["burn_opacity"], 0.0)
        self.assertIn(lane_log["lyric_burn"]["active_text"], lines)


if __name__ == "__main__":
    unittest.main()
