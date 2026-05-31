from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from render_stem_state_nightmare import (
    SPECTRUM_BAND_COUNT,
    build_generation_banner,
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

    def test_compute_frame_metrics_extracts_master_and_stem_spectrum_pairs(self):
        samples = {
            "Master": np.sin(np.linspace(0, np.pi * 24, 1600, dtype=np.float32)),
            "Drums": np.concatenate([np.zeros(800, dtype=np.float32), np.ones(800, dtype=np.float32)]),
            "Bass": np.ones(1600, dtype=np.float32) * 0.25,
        }
        metrics = compute_frame_metrics(samples, sample_rate=1600, fps=4, duration=1.0)
        self.assertEqual(metrics["frame_count"], 4)
        self.assertEqual(len(metrics["frames"]), 4)
        self.assertIn("master", metrics["frames"][0])
        self.assertEqual(len(metrics["frames"][0]["master"]["bands"]), SPECTRUM_BAND_COUNT)
        self.assertEqual(len(metrics["frames"][0]["stems"]["Bass"]["bands"]), SPECTRUM_BAND_COUNT)
        self.assertGreater(metrics["frames"][2]["stems"]["Drums"]["onset"], 0.0)

    def test_render_frame_returns_spectrum_analyzer_with_edge_frame_not_lyrics(self):
        mapping = build_stem_control_map()
        frame_state = {
            "frame_index": 4,
            "time_seconds": 0.133333333,
            "master": {"rms": 0.5, "onset": 0.25, "bands": [0.5] * SPECTRUM_BAND_COUNT},
            "stems": {
                stem_name: {
                    "rms": 0.45,
                    "onset": 0.2,
                    "bass": 0.2,
                    "mid": 0.4,
                    "high": 0.3,
                    "bands": [0.35] * SPECTRUM_BAND_COUNT,
                }
                for stem_name in mapping
            },
        }
        frame, lane_log = render_frame(frame_state, width=320, height=180, stem_map=mapping)
        self.assertEqual(frame.shape, (180, 320, 3))
        self.assertEqual(frame.dtype.name, "uint8")
        self.assertGreater(frame.max(), 20)
        self.assertEqual(lane_log["analyzer"]["band_count"], SPECTRUM_BAND_COUNT)
        self.assertEqual(lane_log["analyzer"]["pairing"], "master_wave_vs_stem")
        self.assertEqual(lane_log["edge_frame"]["mode"], "spectrum_reactive_perimeter")
        self.assertGreater(lane_log["edge_frame"]["intensity"], 0.0)
        self.assertNotIn("lyric_burn", lane_log)
        self.assertFalse(lane_log["boundary"]["lyrics_used"])
        self.assertFalse(lane_log["boundary"]["center_lasers_used"])
        self.assertTrue(lane_log["boundary"]["edge_frame_used"])
        self.assertIn("CORTEX EVOLVED", build_generation_banner())


if __name__ == "__main__":
    unittest.main()
