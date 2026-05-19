import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trueframegen import build_temporal_616_map, fill_missing_frames, fill_truevision_capture
from trueframegen.causal_cell_map import build_causal_cell_map
from trueframegen.render_missing_frame import render_missing_frame
from trueframegen.state_interpolator import interpolate_missing_state
from truevision_resonance_recorder import CELL_FEATURE_NAMES, write_capture_bundle


class TrueFrameGenTests(unittest.TestCase):
    def _cells(self, value: float, shape=(4, 6)) -> np.ndarray:
        cells = np.zeros((shape[0], shape[1], len(CELL_FEATURE_NAMES)), dtype=np.float32)
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_mean_r")] = value
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_mean_g")] = value + 10
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_mean_b")] = value + 20
        cells[:, :, CELL_FEATURE_NAMES.index("luma_mean")] = value + 5
        cells[:, :, CELL_FEATURE_NAMES.index("edge_density")] = value / 100.0
        cells[:, :, CELL_FEATURE_NAMES.index("motion_energy")] = value / 80.0
        cells[:, :, CELL_FEATURE_NAMES.index("delta_luma_abs")] = value / 70.0
        return cells

    def test_temporal_616_map_collects_six_prior_and_future(self):
        cells_by_frame = {frame: self._cells(frame) for frame in list(range(1, 7)) + list(range(8, 14))}

        window = build_temporal_616_map(cells_by_frame, 7)

        self.assertEqual(window.prior_frames, (1, 2, 3, 4, 5, 6))
        self.assertEqual(window.future_frames, (8, 9, 10, 11, 12, 13))
        self.assertEqual(window.observed_count, 12)
        self.assertTrue(window.has_left_anchor)
        self.assertTrue(window.has_right_anchor)

    def test_interpolator_fills_core_channels_from_temporal_cloud(self):
        cells_by_frame = {frame: self._cells(frame * 10) for frame in [1, 2, 4, 5, 6, 7, 8, 9]}
        window = build_temporal_616_map(cells_by_frame, 3)

        filled, trace = interpolate_missing_state(cells_by_frame, window, feature_names=CELL_FEATURE_NAMES)

        r_index = CELL_FEATURE_NAMES.index("rgb_mean_r")
        luma_index = CELL_FEATURE_NAMES.index("luma_mean")
        self.assertAlmostEqual(float(filled[0, 0, r_index]), 30.0, delta=0.01)
        self.assertAlmostEqual(float(filled[0, 0, luma_index]), 35.0, delta=0.01)
        self.assertEqual(trace["target_frame"], 3)
        self.assertFalse(trace["hallucination_used"])
        self.assertEqual(trace["temporal_616"]["center"], 3)
        self.assertIn("rgb_mean_r", trace["channel_traces"])

    def test_causal_cell_map_summarizes_direction_and_confidence(self):
        cells_by_frame = {frame: self._cells(frame * 5) for frame in [1, 2, 4, 5]}
        window = build_temporal_616_map(cells_by_frame, 3)

        causal = build_causal_cell_map(cells_by_frame, window, feature_names=CELL_FEATURE_NAMES)

        self.assertEqual(causal["target_frame"], 3)
        self.assertEqual(causal["anchor_frames"], [2, 4])
        self.assertGreater(causal["summary"]["mean_abs_luma_delta"], 0.0)
        self.assertGreater(causal["confidence"], 0.0)
        self.assertIn("luma_direction", causal["arrays"])

    def test_fill_missing_frames_renders_and_verifies(self):
        cells_by_frame = {frame: self._cells(frame * 10) for frame in [1, 2, 4, 5, 6, 7, 8, 9]}

        filled, traces, continuity = fill_missing_frames(
            cells_by_frame,
            feature_names=CELL_FEATURE_NAMES,
            output_shape=(8, 12),
            radius=6,
        )
        frame = render_missing_frame(filled[3], feature_names=CELL_FEATURE_NAMES, output_shape=(8, 12), smooth=False)

        self.assertIn(3, filled)
        self.assertEqual(frame.shape, (8, 12, 3))
        self.assertEqual(traces[0]["target_frame"], 3)
        self.assertTrue(continuity[0]["continuity_ok"])

    def test_fill_truevision_capture_writes_manifest_trace_and_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "gap-test"
            run_dir = root / run_id
            cell_dir = run_dir / "cell_state_npz"
            cell_dir.mkdir(parents=True)
            observed_frames = [1, 2, 4, 5]
            chunk_path = cell_dir / "gap-test_cells_0000.npz"
            np.savez_compressed(
                chunk_path,
                cell_state=np.stack([self._cells(frame * 10) for frame in observed_frames]).astype(np.float32),
                frame_numbers=np.asarray(observed_frames, dtype=np.int64),
                feature_names=np.asarray(CELL_FEATURE_NAMES),
                grid_shape=np.asarray([4, 6], dtype=np.int32),
            )
            records = [
                {
                    "record_kind": "compucogvision_full_frame_state",
                    "run_id": run_id,
                    "timestamp_unix": float(frame),
                    "elapsed_seconds": float(frame) / 10.0,
                    "frame_number": frame,
                    "fps": 10.0,
                    "screen_energy": float(frame),
                    "visual_resonance": {"vis_energy_total": float(frame)},
                    "geometry": {
                        "frame_shape": [8, 12],
                        "grid_shape": [4, 6],
                        "block_shape": [2, 3],
                    },
                }
                for frame in observed_frames
            ]
            bundle = write_capture_bundle(
                output_root=root,
                run_id=run_id,
                records=records,
                config={"duration_seconds": 1.0},
                cell_state_chunks=[
                    {
                        "path": str(chunk_path),
                        "sha256": "sha256:test",
                        "frame_count": len(observed_frames),
                        "frame_numbers": observed_frames,
                        "shape": [len(observed_frames), 4, 6, len(CELL_FEATURE_NAMES)],
                        "grid_shape": [4, 6],
                    }
                ],
            )

            result = fill_truevision_capture(bundle["run_dir"], output_dir=root / "filled", fps=10.0)
            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))

            self.assertTrue(Path(result["filled_video_mp4"]).exists())
            self.assertTrue(Path(result["temporal_616_trace_jsonl"]).exists())
            self.assertTrue(Path(result["missing_frame_report_md"]).exists())
            self.assertEqual(manifest["filled_frames"], 1)
            self.assertEqual(manifest["feature_scope"][0], "rgb_mean_r")
            self.assertIn("TrueFrameGen fills only missing state", manifest["law"])


if __name__ == "__main__":
    unittest.main()

