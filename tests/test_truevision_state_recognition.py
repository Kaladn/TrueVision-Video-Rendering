from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_runtime.state_recognition import recognize_states_from_manifest, write_state_recognition_outputs


FEATURE_NAMES = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "luma_mean",
    "luma_std",
    "saturation_mean",
    "delta_luma_abs",
    "edge_density",
    "texture_energy",
    "motion_energy",
]


def _write_state_recognition_capture(root: Path, *, run_id: str = "state_recognition_teacher") -> Path:
    run_dir = root / run_id
    cell_dir = run_dir / "cell_state_native"
    cell_dir.mkdir(parents=True)
    frames = 48
    rows = 12
    cols = 16
    cells = np.zeros((frames, rows, cols, len(FEATURE_NAMES)), dtype="<f4")
    yy, xx = np.mgrid[0:rows, 0:cols]
    previous_luma = np.full((rows, cols), 0.10, dtype=np.float32)
    previous_edge = np.full((rows, cols), 0.05, dtype=np.float32)
    for frame_index in range(frames):
        luma = np.full((rows, cols), 0.10, dtype=np.float32)
        edge = np.full((rows, cols), 0.05, dtype=np.float32)
        texture = np.full((rows, cols), 0.05, dtype=np.float32)
        motion = np.full((rows, cols), 0.02, dtype=np.float32)
        saturation = np.full((rows, cols), 0.18, dtype=np.float32)

        if 4 <= frame_index <= 13:
            width = 1 if frame_index < 8 or frame_index > 10 else 2
            line = np.abs(xx - 4) <= width
            edge += np.where(line, 0.72 + 0.09 * np.sin(frame_index), 0.0)
            luma += np.where(line, 0.16, 0.0)
            texture += np.where(line, 0.18 + 0.06 * np.sin(frame_index * 2.0), 0.0)
        if 11 <= frame_index <= 16:
            crawl = np.abs(xx - (5 + (frame_index % 2))) <= 1
            crawl &= (yy >= 2) & (yy <= 9)
            edge += np.where(crawl, 0.28, 0.0)
            motion += np.where(crawl, 0.16, 0.0)

        if frame_index == 18:
            dist = np.sqrt((xx - 11) ** 2 + (yy - 4) ** 2)
            luma += np.clip(0.86 - dist * 0.10, 0.0, 0.86)
            edge += np.clip(0.35 - dist * 0.04, 0.0, 0.35)
        if 19 <= frame_index <= 23:
            dist = np.sqrt((xx - 11) ** 2 + (yy - 4) ** 2)
            decay = 0.52 * np.exp(-(frame_index - 19) / 2.0)
            luma += np.clip(decay - dist * 0.05, 0.0, decay)
        if 18 <= frame_index <= 24:
            shadow = (xx < 4) & (yy > 6)
            luma -= np.where(shadow, 0.055 + 0.02 * (frame_index - 18), 0.0)

        if 25 <= frame_index <= 32:
            band = (yy >= 7) & (yy <= 8)
            shimmer = 0.14 + 0.10 * np.sin(frame_index * 2.5)
            luma += np.where(band, shimmer, 0.0)
            texture += np.where(band, 0.30 + 0.12 * np.sin(frame_index * 1.7), 0.0)
            motion += np.where(band, 0.20 + 0.08 * np.cos(frame_index * 1.3), 0.0)
            saturation += np.where(band, 0.12, 0.0)

        if 33 <= frame_index <= 39:
            veil = np.ones((rows, cols), dtype=bool)
            luma += np.where(veil, 0.12, 0.0)
            saturation -= np.where(veil, 0.08, 0.0)
            edge *= 0.35
            texture *= 0.45
        if 40 <= frame_index <= 46:
            reveal = (xx >= 5) & (xx <= 13)
            edge += np.where(reveal, 0.10 * (frame_index - 39), 0.0)
            texture += np.where(reveal, 0.06 * (frame_index - 39), 0.0)

        if 8 <= frame_index <= 36:
            center_x = 3 + (frame_index - 8) * (9 / 28)
            blob = np.exp(-(((xx - center_x) ** 2) / 5.0 + ((yy - 6) ** 2) / 7.0))
            motion += blob * 0.24
            luma += blob * 0.06
            if 28 <= frame_index <= 33:
                compression = np.exp(-(((xx - 8) ** 2) / 2.2 + ((yy - 6) ** 2) / 2.2))
                motion += compression * 0.30

        luma = np.clip(luma, 0.0, 1.0)
        edge = np.clip(edge, 0.0, 1.0)
        texture = np.clip(texture, 0.0, 1.0)
        motion = np.clip(motion, 0.0, 1.0)
        saturation = np.clip(saturation, 0.0, 1.0)
        delta = np.maximum(np.abs(luma - previous_luma), np.abs(edge - previous_edge) * 0.35)
        cells[frame_index, :, :, FEATURE_NAMES.index("rgb_mean_r")] = luma
        cells[frame_index, :, :, FEATURE_NAMES.index("rgb_mean_g")] = np.clip(luma * 0.96, 0.0, 1.0)
        cells[frame_index, :, :, FEATURE_NAMES.index("rgb_mean_b")] = np.clip(luma * 1.03, 0.0, 1.0)
        cells[frame_index, :, :, FEATURE_NAMES.index("luma_mean")] = luma
        cells[frame_index, :, :, FEATURE_NAMES.index("luma_std")] = np.clip(luma * 0.18 + edge * 0.10, 0.0, 1.0)
        cells[frame_index, :, :, FEATURE_NAMES.index("saturation_mean")] = saturation
        cells[frame_index, :, :, FEATURE_NAMES.index("delta_luma_abs")] = delta
        cells[frame_index, :, :, FEATURE_NAMES.index("edge_density")] = edge
        cells[frame_index, :, :, FEATURE_NAMES.index("texture_energy")] = texture
        cells[frame_index, :, :, FEATURE_NAMES.index("motion_energy")] = motion
        previous_luma = luma
        previous_edge = edge

    chunk = cell_dir / f"{run_id}_cells_0000.tvcells"
    chunk.write_bytes(cells.tobytes())
    records = run_dir / f"{run_id}_records.jsonl"
    with records.open("w", encoding="utf-8") as handle:
        for frame_index in range(frames):
            handle.write(json.dumps({"frame_number": frame_index + 1, "elapsed_seconds": frame_index / 24.0}) + "\n")
    manifest = {
        "schema_version": 1,
        "record_kind": "truevision_native_rs_frame_state",
        "run_id": run_id,
        "records_jsonl": str(records),
        "config": {
            "duration_seconds": frames / 24.0,
            "capture_fps": 24,
            "capture_resolution": [cols * 8, rows * 8],
            "grid_size_xy": [cols, rows],
            "capture_region": [0, 0, cols * 8, rows * 8],
            "cell_chunk_frames": frames,
        },
        "summary": {"frame_count": frames, "duration_seconds": frames / 24.0},
        "cell_state": {
            "enabled": True,
            "format": "tvcells_f32le_v1",
            "feature_names": FEATURE_NAMES,
            "chunks": [
                {
                    "chunk_id": 0,
                    "path": str(chunk),
                    "format": "tvcells_f32le_v1",
                    "frames": frames,
                    "grid_shape": [rows, cols],
                    "feature_count": len(FEATURE_NAMES),
                }
            ],
        },
        "boundary": {"raw_frame_saved": False},
    }
    manifest_path = run_dir / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


class TrueVisionStateRecognitionTests(unittest.TestCase):
    def test_manifest_recognizer_emits_required_state_families_without_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_state_recognition_capture(Path(tmp))
            report = recognize_states_from_manifest(
                manifest,
                run_id="state_recognition_test",
                max_frames=48,
                sample_stride=1,
            )

        self.assertEqual(report["schema_version"], "truevision_state_recognition_report_v1")
        self.assertFalse(report["boundary"]["raw_frames_saved"])
        self.assertFalse(report["boundary"]["render_started"])
        self.assertFalse(report["boundary"]["animation_started"])
        families = {event["state_family"] for event in report["events"]}
        self.assertTrue({"line", "light", "surface", "atmosphere", "motion"}.issubset(families))
        names = {event["state_name"] for event in report["events"]}
        expected_names = {
            "line_appears",
            "line_thickens_thins",
            "line_breathing",
            "edge_shimmer",
            "luminance_rise_fall",
            "bloom_pressure",
            "shadow_deepening",
            "texture_shimmer",
            "reflection_pulse",
            "haze_veil",
            "fog_reveal",
            "edge_loss_under_veil",
            "center_energy_pull",
            "motion_pressure_pulse",
            "frame_wide_state_surge",
        }
        self.assertTrue(expected_names.issubset(names))
        for event in report["events"]:
            self.assertIn("state_id", event)
            self.assertGreaterEqual(event["end_frame"], event["start_frame"])
            self.assertIn("affected_region", event)
            self.assertIn("evidence_metrics", event)
            self.assertFalse(event["raw_frames_saved"])

    def test_outputs_json_markdown_and_csv_event_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_state_recognition_capture(root)
            report = recognize_states_from_manifest(manifest, run_id="state_recognition_outputs")
            result = write_state_recognition_outputs(report, output_root=root / "out", run_id="state_recognition_outputs")

            json_report = json.loads(Path(result["json_report"]).read_text(encoding="utf-8"))
            md = Path(result["markdown_summary"]).read_text(encoding="utf-8")
            with Path(result["csv_event_table"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(json_report["schema_version"], "truevision_state_recognition_report_v1")
        self.assertIn("Recognition only", md)
        self.assertGreater(len(rows), 5)
        self.assertIn("state_name", rows[0])
        self.assertIn("raw_frames_saved", rows[0])

    def test_cli_writes_recognition_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_state_recognition_capture(root)
            output_root = root / "recognized"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/truevision_state_recognition.py",
                    "--manifest",
                    str(manifest),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "cli_state_recognition",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=True,
            )
            stdout = json.loads(completed.stdout)
            self.assertTrue(Path(stdout["json_report"]).exists())
            self.assertTrue(Path(stdout["markdown_summary"]).exists())
            self.assertTrue(Path(stdout["csv_event_table"]).exists())
            self.assertEqual(stdout["event_count"], len(json.loads(Path(stdout["json_report"]).read_text())["events"]))


if __name__ == "__main__":
    unittest.main()
