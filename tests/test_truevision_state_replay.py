import unittest
import tempfile
from pathlib import Path

import numpy as np

from truevision_state_replay import build_rgb_replay_frame, cell_rgb_accuracy, read_native_cell_chunk


class TrueVisionStateReplayTests(unittest.TestCase):
    def test_build_rgb_replay_frame_preserves_cell_rgb_addresses(self):
        feature_names = ["rgb_mean_r", "rgb_mean_g", "rgb_mean_b", "motion_energy"]
        cells = np.zeros((2, 4, 4), dtype=np.float32)
        cells[0, 0, 0:3] = [10, 20, 30]
        cells[0, 1, 0:3] = [40, 50, 60]
        cells[1, 3, 0:3] = [100, 110, 120]

        frame = build_rgb_replay_frame(
            cells,
            feature_names=feature_names,
            output_shape=(4, 8),
        )

        self.assertEqual(frame.shape, (4, 8, 3))
        np.testing.assert_array_equal(
            frame[0:2, 0:2],
            np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            frame[0:2, 2:4],
            np.full((2, 2, 3), [40, 50, 60], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            frame[2:4, 6:8],
            np.full((2, 2, 3), [100, 110, 120], dtype=np.uint8),
        )

    def test_cell_rgb_accuracy_is_exact_before_video_encoding(self):
        feature_names = ["rgb_mean_r", "rgb_mean_g", "rgb_mean_b"]
        cells = np.zeros((2, 4, 3), dtype=np.float32)
        cells[:, :, 0] = 12
        cells[:, :, 1] = 34
        cells[:, :, 2] = 56
        frame = build_rgb_replay_frame(cells, feature_names=feature_names, output_shape=(4, 8))

        metrics = cell_rgb_accuracy(frame, cells, feature_names=feature_names)

        self.assertEqual(metrics["max_abs_error"], 0.0)
        self.assertEqual(metrics["mean_abs_error"], 0.0)
        self.assertEqual(metrics["cell_count"], 8)

    def test_read_native_cell_chunk_supports_rust_tvcells(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tiny.tvcells"
            frames = np.array([0, 3], dtype="<u4")
            cells = np.arange(2 * 2 * 3 * 4, dtype="<f4").reshape(2, 2, 3, 4)
            with path.open("wb") as handle:
                handle.write(b"TVCELL01")
                handle.write(np.array([2, 2, 3, 4], dtype="<u4").tobytes())
                handle.write(frames.tobytes())
                handle.write(cells.tobytes())

            observed_cells, observed_numbers = read_native_cell_chunk(path)

            np.testing.assert_array_equal(observed_numbers, np.array([0, 3], dtype=np.int32))
            np.testing.assert_array_equal(observed_cells, cells.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
