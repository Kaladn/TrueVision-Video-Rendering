import unittest

from truevision_flame_walk_forecast import build_source_sequence, forecast_timeline_616, render_flame_walk_frame


class TrueVisionFlameWalkForecastTests(unittest.TestCase):
    def test_source_sequence_has_ten_static_character_story_states(self):
        sequence = build_source_sequence(source_seconds=10, source_state_count=10)

        self.assertEqual(len(sequence), 10)
        self.assertEqual(sequence[0]["scene_phase"], "child_watches_father_walk_away")
        self.assertEqual(sequence[-1]["scene_phase"], "walking_toward_flame_together")
        self.assertTrue(all(state["character_motion"] == "locked_pose_environment_moves" for state in sequence))
        self.assertGreater(sequence[0]["child_presence"], sequence[-1]["child_presence"])
        self.assertGreater(sequence[-1]["pair_unity"], sequence[0]["pair_unity"])

    def test_forecast_extends_ten_seconds_to_twenty_with_616_trace(self):
        source = build_source_sequence(source_seconds=10, source_state_count=10)
        audio = [
            {
                "frame_index": index,
                "time_seconds": index / 2,
                "rms": 0.25,
                "bass": 0.3 if index % 3 else 0.9,
                "mid": 0.2,
                "high": 0.4,
                "beat": 1.0 if index % 4 == 0 else 0.1,
            }
            for index in range(40)
        ]

        timeline, trace = forecast_timeline_616(source, audio, total_seconds=20, fps=2)

        self.assertEqual(len(timeline), 40)
        self.assertEqual(len(trace), 40)
        self.assertEqual(timeline[0]["forecast_kind"], "source_interpolated")
        self.assertEqual(timeline[-1]["forecast_kind"], "six_one_six_projected")
        self.assertTrue(all(state["character_motion"] == "locked_pose_environment_moves" for state in timeline))
        self.assertGreaterEqual(max(state["flame_lick_pressure"] for state in timeline), 0.9)
        self.assertEqual(trace[-1]["prior_count"], 6)
        self.assertEqual(trace[-1]["future_count"], 0)

    def test_render_frame_returns_nonblank_cinematic_layers(self):
        source = build_source_sequence(source_seconds=10, source_state_count=10)
        timeline, _ = forecast_timeline_616(source, [{"time_seconds": 0, "beat": 1, "bass": 1, "rms": 0.8, "mid": 0.3, "high": 0.7}], total_seconds=1, fps=1)

        frame, metadata = render_flame_walk_frame(160, 90, timeline[0])

        self.assertEqual(frame.shape, (90, 160, 3))
        self.assertGreater(int(frame.sum()), 20000)
        self.assertIn("beat_driven_flame_licks", metadata["layers"])
        self.assertIn("locked_character_blocking", metadata["layers"])
        self.assertTrue(metadata["boundary"]["no_external_visual_assets"])

    def test_final_pair_silhouette_has_visible_rim_against_fire(self):
        source = build_source_sequence(source_seconds=10, source_state_count=10)
        audio = [{"time_seconds": index / 4, "beat": 0.8, "bass": 0.9, "rms": 0.75, "mid": 0.5, "high": 0.4} for index in range(80)]
        timeline, _ = forecast_timeline_616(source, audio, total_seconds=20, fps=4)

        frame, _ = render_flame_walk_frame(320, 180, timeline[-1])
        foreground = frame[int(frame.shape[0] * 0.56) :, int(frame.shape[1] * 0.38) : int(frame.shape[1] * 0.62), :]
        dark_pixels = (foreground.mean(axis=2) < 12).sum()
        warm_rim_pixels = ((foreground[:, :, 2] > 80) & (foreground[:, :, 1] > 30)).sum()

        self.assertGreater(int(dark_pixels), 250)
        self.assertGreater(int(warm_rim_pixels), 80)


if __name__ == "__main__":
    unittest.main()
