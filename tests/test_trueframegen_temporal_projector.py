import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from trueframegen.temporal_causality_projector import (
    apply_projection_visual_style,
    audio_soundprint,
    build_temporal_projection_profile,
    project_capture_to_audio,
    project_state_sequence,
    read_edge_lyrics_block,
)
from trueframegen.lightning_signature import extract_lightning_signature_from_cells
from truevision_resonance_recorder import CELL_FEATURE_NAMES


class TrueFrameGenTemporalProjectorTests(unittest.TestCase):
    def _cells(self, frame: int, rows: int = 2, cols: int = 4) -> np.ndarray:
        y = np.arange(rows, dtype=np.float32)[:, None]
        x = np.arange(cols, dtype=np.float32)[None, :]
        cells = np.zeros((rows, cols, len(CELL_FEATURE_NAMES)), dtype=np.float32)
        base = 24.0 + frame * 1.6 + x * 4.0 + y * 6.0
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_mean_r")] = base
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_mean_g")] = base + 12.0 + np.sin(frame * 0.2 + x)
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_mean_b")] = base + 26.0 + np.cos(frame * 0.3 + y)
        cells[:, :, CELL_FEATURE_NAMES.index("luma_mean")] = base + 8.0
        cells[:, :, CELL_FEATURE_NAMES.index("edge_density")] = 0.12 + 0.01 * frame + x * 0.003
        cells[:, :, CELL_FEATURE_NAMES.index("motion_energy")] = 0.08 + 0.02 * np.sin(frame * 0.4 + x)
        cells[:, :, CELL_FEATURE_NAMES.index("delta_luma_abs")] = 0.05 + 0.01 * np.cos(frame * 0.25 + y)
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_std_r")] = 2.0 + x
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_std_g")] = 2.5 + y
        cells[:, :, CELL_FEATURE_NAMES.index("rgb_std_b")] = 3.0 + x * 0.5
        cells[:, :, CELL_FEATURE_NAMES.index("saturation_mean")] = 0.2 + frame * 0.002
        cells[:, :, CELL_FEATURE_NAMES.index("texture_energy")] = 0.15 + x * 0.01 + y * 0.02
        return cells

    def _write_tiny_wav(self, path: Path, *, seconds: float = 0.6, sample_rate: int = 8000) -> None:
        t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
        samples = (0.45 * np.sin(2 * np.pi * 120 * t) + 0.25 * np.sin(2 * np.pi * 360 * t)).astype(np.float32)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes())

    def _write_capture(self, root: Path) -> Path:
        run_dir = root / "source_capture"
        cell_dir = run_dir / "cell_state_npz"
        cell_dir.mkdir(parents=True)
        frame_numbers = np.arange(18, dtype=np.int64)
        chunk_path = cell_dir / "source_cells_0000.npz"
        np.savez_compressed(
            chunk_path,
            cell_state=np.stack([self._cells(int(frame)) for frame in frame_numbers]).astype(np.float32),
            frame_numbers=frame_numbers,
            feature_names=np.asarray(CELL_FEATURE_NAMES),
            grid_shape=np.asarray([2, 4], dtype=np.int32),
        )
        manifest = {
            "schema_version": 1,
            "record_kind": "unit_truevision_capture",
            "run_id": "source_capture",
            "cell_state": {
                "enabled": True,
                "feature_names": CELL_FEATURE_NAMES,
                "chunks": [{"path": str(chunk_path), "frames": len(frame_numbers), "grid_shape": [2, 4]}],
            },
            "boundary": {"raw_frame_saved": False},
        }
        summary = {
            "schema_version": 1,
            "kind": "unit_summary",
            "run_id": "source_capture",
            "frame_count": len(frame_numbers),
            "duration_seconds": 2.0,
            "geometry": {"frame_shape": [4, 8], "grid_shape": [2, 4]},
        }
        (run_dir / "source_capture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "source_capture_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return run_dir

    def test_soundprint_and_lyrics_create_edge_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "edge.wav"
            lyrics = root / "lyrics.txt"
            self._write_tiny_wav(audio)
            lyrics.write_text("Edge Of The World\nJust looking down\nThere is a river of life\nYou me together we fight", encoding="utf-8")

            features, summary = audio_soundprint(audio, fps=4, sample_rate=8000)
            theme = read_edge_lyrics_block(lyrics)

            self.assertGreater(len(features), 0)
            self.assertGreater(summary["average_level"], 0)
            self.assertIn("river of life", theme["anchors"])
            self.assertIn("you me together", theme["anchors"])

    def test_projection_uses_temporal_deltas_not_source_frame_loop(self):
        source = np.stack([self._cells(frame) for frame in range(18)]).astype(np.float32)
        frames = np.arange(18, dtype=np.int32)
        profile = build_temporal_projection_profile(source, frames, feature_names=CELL_FEATURE_NAMES, radius=6)
        audio = [
            {"time_seconds": i / 4, "rms": 0.7, "bass": 0.5, "mid": 0.4, "high": 0.3, "beat": 0.6, "rise": 0.2}
            for i in range(8)
        ]

        projected_a = [cells for cells, _ in project_state_sequence(profile, audio, trace_every=2)]
        projected_b = [cells for cells, _ in project_state_sequence(profile, audio, trace_every=2)]

        self.assertEqual(profile.summary["projection_rule"], "mix_6_1_6_delta_fields_under_audio_control_not_source_frame_loop")
        self.assertEqual(len(projected_a), 8)
        np.testing.assert_allclose(projected_a[-1], projected_b[-1])
        self.assertFalse(np.allclose(projected_a[-1], source[-1]))

    def test_hell_power_walk_style_adds_edges_grade_and_silhouette(self):
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(20, 170, 160, dtype=np.uint8)[None, :]
        frame[:, :, 1] = 45
        frame[:, :, 2] = np.linspace(80, 10, 90, dtype=np.uint8)[:, None]

        styled, metadata = apply_projection_visual_style(
            frame,
            visual_style="hell_power_walk",
            audio={"time_seconds": 2.0, "rms": 0.8, "bass": 0.7, "high": 0.6, "beat": 0.9},
            frame_index=24,
            fps=12,
            duration_seconds=5.0,
        )

        self.assertEqual(styled.shape, frame.shape)
        self.assertFalse(np.array_equal(styled, frame))
        self.assertTrue(metadata["style_applied"])
        self.assertIn("canny", metadata["edge_filters"])
        self.assertEqual(metadata["saturation_scale"], 0.75)
        self.assertIn("music_lighting_strike_pressure", metadata)
        self.assertIn("smooth_transition_flash", metadata)
        self.assertTrue(metadata["red_rhythm_transition"])
        self.assertEqual(metadata["silhouette"], "walking_away_power_projection")

    def test_lightning_signature_extracts_peak_cells_and_can_drive_style(self):
        cells = np.stack([self._cells(frame, rows=4, cols=8) for frame in range(18)]).astype(np.float32)
        luma_index = CELL_FEATURE_NAMES.index("luma_mean")
        edge_index = CELL_FEATURE_NAMES.index("edge_density")
        delta_index = CELL_FEATURE_NAMES.index("delta_luma_abs")
        cells[9, 0:3, 3:5, luma_index] += 180.0
        cells[9, 0:3, 3:5, edge_index] += 1.0
        cells[9, 0:3, 3:5, delta_index] += 1.0

        signature = extract_lightning_signature_from_cells(
            cells,
            np.arange(18, dtype=np.int32),
            feature_names=CELL_FEATURE_NAMES,
            radius=6,
            max_cells=12,
        )
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        styled, metadata = apply_projection_visual_style(
            frame,
            visual_style="hell_power_walk",
            audio={"time_seconds": 2.0, "rms": 0.8, "bass": 0.7, "high": 0.9, "beat": 1.0},
            frame_index=24,
            fps=12,
            duration_seconds=5.0,
            lightning_signature=signature,
        )

        self.assertEqual(signature["peak"]["frame_index"], 9)
        self.assertGreater(signature["hot_cell_count"], 0)
        self.assertTrue(metadata["lightning_signature_applied"])
        self.assertGreater(metadata["lightning_signature_points_applied"], 0)
        self.assertGreater(int(styled.sum()), 0)

    def test_project_capture_to_audio_writes_manifest_video_and_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = self._write_capture(root)
            audio = root / "edge.wav"
            lyrics = root / "lyrics.txt"
            self._write_tiny_wav(audio)
            lyrics.write_text("Edge Of The World\nJust looking down\nriver of life\nYou me together", encoding="utf-8")

            result = project_capture_to_audio(
                capture_run_dir=run_dir,
                audio_path=audio,
                lyrics_path=lyrics,
                output_root=root / "out",
                run_id="unit_projection",
                width=8,
                height=4,
                fps=4,
                sample_rate=8000,
                radius=6,
                max_seconds=0.5,
                mux_audio=False,
                visual_style="hell_power_walk",
            )
            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))

            self.assertTrue(Path(result["video_mp4"]).exists())
            self.assertTrue(Path(result["trace_jsonl"]).exists())
            self.assertTrue(Path(result["soundprint_json"]).exists())
            self.assertTrue(manifest["projection"]["not_clone"])
            self.assertTrue(manifest["projection"]["not_source_frame_loop"])
            self.assertEqual(manifest["render"]["visual_style"]["visual_style"], "hell_power_walk")
            self.assertIn("6-1-6", manifest["claim"])


if __name__ == "__main__":
    unittest.main()
