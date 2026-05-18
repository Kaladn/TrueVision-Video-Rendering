import unittest

import numpy as np

from screen_resonance_state import ScreenResonanceState


class ScreenResonanceRectangularTests(unittest.TestCase):
    def test_rectangular_grid_preserves_video_aspect_feature_masks(self):
        state = ScreenResonanceState(grid_size=(9, 16))

        first = np.zeros((9, 16), dtype=np.float32)
        second = first.copy()
        second[3:6, 6:10] = 1.0

        state.update(first)
        features = state.update(second)

        self.assertEqual(state.center_mask.shape, (9, 16))
        self.assertEqual(state.edge_mask.shape, (9, 16))
        self.assertEqual(state.upper_mask.shape, (9, 16))
        self.assertEqual(state.lower_mask.shape, (9, 16))
        self.assertGreater(features["vis_energy_total"], 0.0)
        self.assertGreater(features["vis_center_energy_ratio"], 0.0)
        self.assertFalse(any(np.isnan(value) for value in features.values()))


if __name__ == "__main__":
    unittest.main()
