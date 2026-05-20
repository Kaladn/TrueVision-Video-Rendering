import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trueframegen import (
    build_temporal_616_map,
    fill_missing_frames,
    fill_truevision_capture,
    stream_upsample_truevision_capture,
    upsample_truevision_capture,
)
from trueframegen.causal_cell_map import build_causal_cell_map
from trueframegen.live_upsampler import load_live_native_sequence
from trueframegen.render_missing_frame import render_missing_frame
from trueframegen.state_interpolator import interpolate_missing_state
from trueframegen_live_pipeline import build_live_pipeline_plan
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

    def _write_native_chunk(self, path: Path, frames: list[int]) -> None:
        stack = np.stack([self._cells(frame * 10) for frame in frames]).astype("<f4")
        rows, cols = stack.shape[1:3]
        with path.open("wb") as handle:
            handle.write(b"TVCELL01")
            handle.write(struct.pack("<IIII", len(frames), rows, cols, len(CELL_FEATURE_NAMES)))
            for frame in frames:
                handle.write(struct.pack("<I", frame))
            handle.write(stack.tobytes())

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

    def test_upsample_truevision_capture_generates_in_between_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "upsample-test"
            run_dir = root / run_id
            cell_dir = run_dir / "cell_state_npz"
            cell_dir.mkdir(parents=True)
            observed_frames = [0, 1, 2]
            chunk_path = cell_dir / "upsample-test_cells_0000.npz"
            np.savez_compressed(
                chunk_path,
                cell_state=np.stack([self._cells(frame * 40) for frame in observed_frames]).astype(np.float32),
                frame_numbers=np.asarray(observed_frames, dtype=np.int64),
                feature_names=np.asarray(CELL_FEATURE_NAMES),
                grid_shape=np.asarray([4, 6], dtype=np.int32),
            )
            records = [
                {
                    "record_kind": "compucogvision_full_frame_state",
                    "run_id": run_id,
                    "timestamp_unix": float(frame),
                    "elapsed_seconds": float(frame) / 2.0,
                    "frame_number": frame,
                    "fps": 2.0,
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
                config={"duration_seconds": 1.0, "capture_fps": 2.0},
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

            result = upsample_truevision_capture(bundle["run_dir"], output_dir=root / "upsampled", target_fps=8.0)
            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))

            self.assertTrue(Path(result["video_mp4"]).exists())
            self.assertTrue(Path(result["trace_jsonl"]).exists())
            self.assertEqual(manifest["upsample"]["target_fps"], 8.0)
            self.assertEqual(manifest["upsample"]["output_frames"], 8)
            self.assertEqual(
                manifest["upsample"]["timeline_rule"],
                "generate_in_between_frames_inside_source_duration_not_append_at_end",
            )
            self.assertFalse(manifest["upsample"]["state_dump_written"])

    def test_stream_upsample_truevision_capture_keeps_chunk_cache_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "stream-upsample-test"
            run_dir = root / run_id
            cell_dir = run_dir / "cell_state_native"
            cell_dir.mkdir(parents=True)
            chunk_a = cell_dir / f"{run_id}_cells_0000.tvcells"
            chunk_b = cell_dir / f"{run_id}_cells_0001.tvcells"
            self._write_native_chunk(chunk_a, [0, 1])
            self._write_native_chunk(chunk_b, [2, 3])
            records = [
                {
                    "record_kind": "truevision_native_rs_frame_state",
                    "run_id": run_id,
                    "elapsed_seconds": float(frame) / 2.0,
                    "frame_number": frame,
                    "fps": 2.0,
                    "geometry": {
                        "frame_shape": [8, 12],
                        "grid_shape": [4, 6],
                    },
                }
                for frame in [0, 1, 2, 3]
            ]
            bundle = write_capture_bundle(
                output_root=root,
                run_id=run_id,
                records=records,
                config={"duration_seconds": 2.0, "capture_fps": 2.0},
                cell_state_chunks=[
                    {
                        "path": str(chunk_a),
                        "format": "tvcells_f32le_v1",
                        "frames": 2,
                        "grid_shape": [4, 6],
                    },
                    {
                        "path": str(chunk_b),
                        "format": "tvcells_f32le_v1",
                        "frames": 2,
                        "grid_shape": [4, 6],
                    },
                ],
            )

            result = stream_upsample_truevision_capture(
                bundle["run_dir"],
                output_dir=root / "streamed",
                target_fps=8.0,
                max_seconds=1.0,
                max_cached_chunks=2,
            )
            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))

            self.assertTrue(Path(result["video_mp4"]).exists())
            self.assertEqual(manifest["upsample"]["output_frames"], 8)
            self.assertLessEqual(manifest["streaming"]["peak_cached_chunks"], 2)
            self.assertFalse(manifest["upsample"]["state_dump_written"])

    def test_live_pipeline_plan_starts_tfg_after_trailing_delay(self):
        plan = build_live_pipeline_plan(
            vault=Path("E:/TruEVision Generation"),
            run_id="live-test",
            duration=60.0,
            capture_fps=9.0,
            target_fps=60.0,
            resolution="2560x1440",
            grid="640x360",
            region="",
            tfg_start_after=10.0,
            chunk_frames=9,
            crf=18,
        )

        self.assertTrue(plan["boundary"]["capture_and_tfg_overlap"])
        self.assertTrue(plan["boundary"]["not_append_at_end"])
        self.assertEqual(plan["trueframegen"]["start_after_seconds"], 10.0)
        self.assertIn("--cell-chunk-frames", plan["capture"]["command"])
        self.assertIn("trueframegen_live_upsample.py", " ".join(plan["trueframegen"]["command"]))
        self.assertIn("--capture-fps", plan["trueframegen"]["command"])
        self.assertIn("--duration", plan["trueframegen"]["command"])

    def test_load_live_native_sequence_uses_chunks_before_manifest_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "live-chunks-only"
            cell_dir = run_dir / "cell_state_native"
            cell_dir.mkdir(parents=True)
            self._write_native_chunk(cell_dir / "live-chunks-only_cells_0000.tvcells", [0, 1])
            self._write_native_chunk(cell_dir / "live-chunks-only_cells_0001.tvcells", [2, 3])

            sequence = load_live_native_sequence(
                run_dir,
                frame_shape=(8, 12),
                capture_fps=2.0,
                feature_names=CELL_FEATURE_NAMES,
                min_frames=4,
                timeout_seconds=0.0,
            )

            self.assertEqual(sequence.cells.shape[0], 4)
            self.assertEqual(sequence.frame_numbers.tolist(), [0, 1, 2, 3])
            self.assertEqual(sequence.times_seconds.tolist(), [0.0, 0.5, 1.0, 1.5])
            self.assertEqual(sequence.summary["geometry"]["frame_shape"], [8, 12])
            self.assertEqual(sequence.manifest["run_id"], "live-chunks-only")

    def test_rust_stream_renderer_exposes_segment_field_mode(self):
        source = Path("native/truevision_capture_rs/src/bin/trueframegen_stream_rs.rs").read_text(encoding="utf-8")

        self.assertIn('"segment-field"', source)
        self.assertIn("SegmentField", source)
        self.assertIn("build_segment_field", source)
        self.assertIn("render_segment_field", source)
        self.assertIn("segment_transition_field_inside_source_duration_not_append_at_end", source)


if __name__ == "__main__":
    unittest.main()
