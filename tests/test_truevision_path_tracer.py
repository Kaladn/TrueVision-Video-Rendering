import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_path_tracer import (
    build_path_traced_cells,
    generate_path_traced_scene,
    trace_path_frame,
)
from truevision_resonance_recorder import CELL_FEATURE_NAMES


class TrueVisionPathTracerTests(unittest.TestCase):
    def test_trace_path_frame_is_deterministic_and_uses_lighting(self):
        first = trace_path_frame(
            frame_index=0,
            total_frames=2,
            frame_shape=(36, 64),
            samples_per_pixel=2,
            max_bounces=2,
            seed=616,
        )
        second = trace_path_frame(
            frame_index=0,
            total_frames=2,
            frame_shape=(36, 64),
            samples_per_pixel=2,
            max_bounces=2,
            seed=616,
        )

        self.assertEqual(first.rgb.shape, (36, 64, 3))
        self.assertTrue(np.array_equal(first.rgb, second.rgb))
        self.assertGreater(float(first.rgb.mean()), 10.0)
        self.assertGreater(float(first.rgb.std()), 8.0)
        self.assertEqual(first.metadata["renderer"], "cpu_path_tracer")
        self.assertEqual(first.metadata["samples_per_pixel"], 2)
        self.assertEqual(first.metadata["max_bounces"], 2)
        self.assertGreater(first.metadata["shadow_ray_tests"], 0)

    def test_path_traced_cells_keep_truevision_vector_shape(self):
        cells, state = build_path_traced_cells(
            frame_index=0,
            total_frames=2,
            frame_shape=(36, 64),
            grid_shape=(9, 16),
            samples_per_pixel=1,
            max_bounces=1,
            seed=123,
            previous_luma=None,
        )

        self.assertEqual(cells.shape, (9, 16, len(CELL_FEATURE_NAMES)))
        self.assertFalse(np.isnan(cells).any())
        self.assertEqual(state["scene"], "path_traced_grounded_sphere")
        self.assertEqual(state["renderer"], "cpu_path_tracer")
        self.assertGreater(float(cells[:, :, CELL_FEATURE_NAMES.index("edge_density")].sum()), 0.0)
        self.assertGreater(float(cells[:, :, CELL_FEATURE_NAMES.index("texture_energy")].mean()), 0.0)

    def test_generate_path_traced_scene_writes_manifest_report_and_no_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_path_traced_scene(
                output_root=Path(tmp),
                run_id="unit_path_trace",
                duration_seconds=1.0,
                fps=2,
                frame_shape=(36, 64),
                grid_shape=(9, 16),
                samples_per_pixel=1,
                max_bounces=1,
                replay=False,
            )

            self.assertEqual(result["frames"], 2)
            self.assertFalse(result["audio_saved"])
            self.assertTrue(Path(result["manifest_json"]).exists())
            self.assertTrue(Path(result["records_jsonl"]).exists())
            self.assertTrue(Path(result["report_md"]).exists())
            self.assertEqual(result["renderer"]["renderer"], "cpu_path_tracer")
            self.assertEqual(result["renderer"]["samples_per_pixel"], 1)
            self.assertEqual(result["renderer"]["max_bounces"], 1)


if __name__ == "__main__":
    unittest.main()
