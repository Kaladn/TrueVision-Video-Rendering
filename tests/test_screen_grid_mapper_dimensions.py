import unittest

import cv2
import numpy as np

from modules.screen_grid_mapper import ScreenGridMapper


class ScreenGridMapperDimensionTests(unittest.TestCase):
    def test_frame_to_grid_uses_actual_frame_dimensions(self):
        mapper = ScreenGridMapper.__new__(ScreenGridMapper)
        mapper.grid_rows = 2
        mapper.grid_cols = 2
        mapper.cell_width = 1000.0
        mapper.cell_height = 1000.0

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:2, :2] = [10, 10, 10]
        frame[:2, 2:] = [50, 50, 50]
        frame[2:, :2] = [100, 100, 100]
        frame[2:, 2:] = [200, 200, 200]

        grid = mapper.frame_to_grid(frame)

        self.assertFalse(np.isnan(grid).any())
        np.testing.assert_allclose(
            grid,
            np.array([[10.0, 50.0], [100.0, 200.0]], dtype=np.float32),
        )

    def test_frame_to_grid_uses_area_weighting_for_fractional_cells(self):
        mapper = ScreenGridMapper.__new__(ScreenGridMapper)
        mapper.grid_rows = 2
        mapper.grid_cols = 2

        gray = np.arange(15, dtype=np.uint8).reshape(3, 5) * 10
        frame = np.dstack([gray, gray, gray])

        grid = mapper.frame_to_grid(frame)
        expected = cv2.resize(gray, (2, 2), interpolation=cv2.INTER_AREA).astype(np.float32)

        np.testing.assert_allclose(grid, expected)


if __name__ == "__main__":
    unittest.main()
