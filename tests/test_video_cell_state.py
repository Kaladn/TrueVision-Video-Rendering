import unittest

import numpy as np

from truevision_resonance_recorder import build_video_cell_state


class VideoCellStateTests(unittest.TestCase):
    def test_builds_16x9_cell_vectors_from_frame_pixels(self):
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        frame[:2, :4] = [10, 20, 30]
        frame[:2, 4:] = [40, 50, 60]
        frame[2:, :4] = [70, 80, 90]
        frame[2:, 4:] = [100, 110, 120]
        previous_luma = np.zeros((2, 4), dtype=np.float32)

        state = build_video_cell_state(frame, grid_shape=(2, 4), previous_luma=previous_luma)

        self.assertEqual(state["cells"].shape, (2, 4, len(state["feature_names"])))
        self.assertEqual(state["luma"].shape, (2, 4))
        self.assertEqual(state["feature_names"][0:3], ["rgb_mean_r", "rgb_mean_g", "rgb_mean_b"])
        np.testing.assert_allclose(state["cells"][0, 0, 0:3], [10.0, 20.0, 30.0])
        self.assertGreater(state["cells"][1, 3, state["feature_names"].index("delta_luma_abs")], 0.0)
        self.assertFalse(np.isnan(state["cells"]).any())


if __name__ == "__main__":
    unittest.main()
