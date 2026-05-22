import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_state_snap_sequence_renderer import (
    build_ordered_default_snaps,
    build_state_keys,
    derive_existing_motion_masks,
    fit_state_frame_to_canvas,
    interpolate_state_key,
    load_cell_state_frame,
    reconstruct_frame_from_cell_state,
    smootherstep,
)


class TrueVisionStateSnapSequenceRendererTests(unittest.TestCase):
    def test_reconstructs_frame_from_truevision_cell_state(self):
        cells = np.zeros((3, 4, 16), dtype=np.float32)
        cells[:, :, 0] = 10
        cells[:, :, 1] = np.arange(4, dtype=np.float32)[None, :] * 20
        cells[:, :, 2] = np.arange(3, dtype=np.float32)[:, None] * 30

        frame = reconstruct_frame_from_cell_state(cells, output_size=(80, 60))

        self.assertEqual(frame.shape, (60, 80, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(int(frame[:, :, 1].max()), int(frame[:, :, 1].min()))
        self.assertGreater(int(frame[:, :, 2].max()), int(frame[:, :, 2].min()))

    def test_loads_first_cell_state_frame_from_npz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cells.npz"
            expected = np.ones((1, 2, 3, 16), dtype=np.float32) * 42
            np.savez_compressed(path, cell_state=expected)

            loaded = load_cell_state_frame(path)

        self.assertEqual(loaded.shape, (2, 3, 16))
        self.assertAlmostEqual(float(loaded[0, 0, 0]), 42.0)

    def test_fit_to_canvas_uses_cover_without_stretching(self):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[:, :, 0] = 200

        canvas = fit_state_frame_to_canvas(frame, canvas_size=(192, 108), mode="cover")

        self.assertEqual(canvas.shape, (108, 192, 3))
        self.assertEqual(canvas.dtype, np.uint8)
        self.assertGreater(int(canvas[:, :, 0].mean()), 100)

    def test_smootherstep_is_bounded_and_monotonic(self):
        values = [smootherstep(i / 10.0) for i in range(11)]

        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)
        self.assertEqual(values, sorted(values))

    def test_default_order_is_top_left_lower_left_lower_right_top_right(self):
        snaps = build_ordered_default_snaps(Path("root"))
        names = [snap.run_id for snap in snaps]

        self.assertEqual(
            names,
            [
                "screenshot_20260520_224749_exact_snap",
                "screenshot_20260520_224848_exact_snap",
                "screenshot_20260520_224910_exact_snap",
                "screenshot_20260520_224829_exact_snap",
            ],
        )

    def test_derives_existing_region_masks_without_creating_geometry(self):
        frame = np.full((80, 120, 3), 38, dtype=np.uint8)
        frame[48:70, 10:56] = (10, 80, 220)
        frame[12:44, 64:116] = (112, 118, 122)

        masks = derive_existing_motion_masks(frame)

        self.assertEqual(set(masks), {"fire", "haze", "reflection"})
        self.assertGreater(int(masks["fire"].sum()), 0)
        self.assertGreater(int(masks["haze"].sum()), 0)
        self.assertGreater(int(masks["reflection"].sum()), 0)

    def test_300_state_keys_can_drive_1800_render_frames(self):
        keys = build_state_keys(key_count=300, duration_seconds=60.0)

        first = interpolate_state_key(keys, frame_index=0, frame_count=1800)
        middle = interpolate_state_key(keys, frame_index=900, frame_count=1800)
        last = interpolate_state_key(keys, frame_index=1799, frame_count=1800)

        self.assertEqual(len(keys), 300)
        self.assertIn("fire_dx", first)
        self.assertNotEqual(first["fire_dx"], middle["fire_dx"])
        self.assertNotEqual(middle["fire_dx"], last["fire_dx"])


if __name__ == "__main__":
    unittest.main()
