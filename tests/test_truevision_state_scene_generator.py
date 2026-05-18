import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_resonance_recorder import CELL_FEATURE_NAMES
from truevision_state_scene_generator import build_person_field_cells, generate_person_field_scene


class TrueVisionStateSceneGeneratorTests(unittest.TestCase):
    def test_person_field_cells_use_truevision_vector_shape(self):
        cells, state = build_person_field_cells(
            frame_index=0,
            total_frames=45,
            grid_shape=(90, 160),
            previous_luma=None,
        )

        self.assertEqual(cells.shape, (90, 160, len(CELL_FEATURE_NAMES)))
        self.assertFalse(np.isnan(cells).any())
        self.assertEqual(state["scene"], "person_walking_in_field")
        self.assertEqual(state["actor"]["kind"], "walking_person")
        self.assertGreater(cells[10, 20, CELL_FEATURE_NAMES.index("rgb_mean_b")], 120)
        self.assertGreater(cells[80, 20, CELL_FEATURE_NAMES.index("rgb_mean_g")], 80)

    def test_walking_person_state_moves_right_over_time(self):
        first_cells, first_state = build_person_field_cells(
            frame_index=0,
            total_frames=45,
            grid_shape=(90, 160),
            previous_luma=None,
        )
        last_cells, last_state = build_person_field_cells(
            frame_index=44,
            total_frames=45,
            grid_shape=(90, 160),
            previous_luma=first_cells[:, :, CELL_FEATURE_NAMES.index("luma_mean")],
        )

        self.assertLess(first_state["actor"]["center_col"], last_state["actor"]["center_col"])
        self.assertGreater(last_cells[:, :, CELL_FEATURE_NAMES.index("motion_energy")].sum(), 0.0)

    def test_generate_scene_bundle_writes_compatible_artifacts_without_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_person_field_scene(
                output_root=Path(tmp),
                run_id="unit_person_field",
                duration_seconds=1.0,
                fps=3,
                frame_shape=(90, 160),
                grid_shape=(9, 16),
                chunk_frames=2,
                replay=False,
            )

            self.assertEqual(result["frames"], 3)
            self.assertTrue(Path(result["manifest_json"]).exists())
            self.assertTrue(Path(result["summary_json"]).exists())
            self.assertTrue(Path(result["records_jsonl"]).exists())
            self.assertFalse(result["audio_saved"])


if __name__ == "__main__":
    unittest.main()
