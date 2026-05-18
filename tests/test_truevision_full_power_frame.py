import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_full_power_frame import generate_full_power_frame, render_full_power_frame_from_cells
from truevision_resonance_recorder import CELL_FEATURE_NAMES


class TrueVisionFullPowerFrameTests(unittest.TestCase):
    def test_renderer_uses_non_rgb_channels_for_subcell_detail(self):
        cells = np.zeros((2, 4, len(CELL_FEATURE_NAMES)), dtype=np.float32)
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_mean_r")] = 90
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_mean_g")] = 120
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_mean_b")] = 150
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_std_r")] = 25
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_std_g")] = 18
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_std_b")] = 12
        cells[:, :, CELL_FEATURE_NAMES.index("luma_std")] = 22
        cells[:, :, CELL_FEATURE_NAMES.index("edge_density")] = 0.75
        cells[:, :, CELL_FEATURE_NAMES.index("texture_energy")] = 19
        cells[:, :, CELL_FEATURE_NAMES.index("motion_energy")] = 30
        cells[:, :, CELL_FEATURE_NAMES.index("saturation_mean")] = 150

        frame = render_full_power_frame_from_cells(
            cells,
            feature_names=CELL_FEATURE_NAMES,
            output_shape=(40, 80),
            seed=616,
        )

        self.assertEqual(frame.shape, (40, 80, 3))
        self.assertGreater(float(frame[0:20, 0:20].std()), 2.0)

    def test_generate_full_power_frame_writes_state_and_clean_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_full_power_frame(
                output_root=Path(tmp),
                run_id="unit_full_power_frame",
                frame_shape=(180, 320),
                grid_shape=(18, 32),
            )

            self.assertTrue(Path(result["state_png"]).exists())
            self.assertTrue(Path(result["source_reference_png"]).exists())
            self.assertTrue(Path(result["cell_state_npz"]).exists())
            self.assertTrue(Path(result["report_md"]).exists())
            self.assertEqual(result["cell_shape"], [18, 32, len(CELL_FEATURE_NAMES)])


if __name__ == "__main__":
    unittest.main()
