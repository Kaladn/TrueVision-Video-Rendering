from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from truevision_runtime.timeline_audit import audit_many, audit_timeline_manifest


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class TrueVisionTimingAuditTests(unittest.TestCase):
    def test_native_sampled_state_log_is_exact_but_not_full_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state_path = root / "run_frame_state.jsonl"
            _write_jsonl(
                state_path,
                [
                    {"frame_index": 0, "time_seconds": 0.0},
                    {"frame_index": 2, "time_seconds": 2 / 30},
                    {"frame_index": 4, "time_seconds": 4 / 30},
                ],
            )
            manifest_path = root / "run_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "render": {
                            "frame_state_jsonl": str(state_path),
                            "fps": 30,
                            "duration_seconds": 10 / 30,
                            "frame_count": 10,
                            "state_log_every": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_timeline_manifest(manifest_path)

            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["timeline_mode"], "sampled_exact")
            self.assertFalse(audit["full_frame_log"])
            self.assertFalse(audit["usable_for_frame_exact_tooling"])
            self.assertEqual(audit["sample_step_frames"], 2)
            self.assertLess(audit["max_time_error_seconds"], 0.000001)

    def test_native_full_frame_state_log_is_exact_and_tool_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state_path = root / "run_frame_state.jsonl"
            _write_jsonl(
                state_path,
                [
                    {"frame_index": 0, "time_seconds": 0.0},
                    {"frame_index": 1, "time_seconds": 1 / 60},
                    {"frame_index": 2, "time_seconds": 2 / 60},
                ],
            )
            manifest_path = root / "run_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "render": {
                            "frame_state_jsonl": str(state_path),
                            "fps": 60,
                            "duration_seconds": 3 / 60,
                            "frame_count": 3,
                            "state_log_every": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_timeline_manifest(manifest_path)

            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["timeline_mode"], "full_frame_exact")
            self.assertTrue(audit["full_frame_log"])
            self.assertTrue(audit["usable_for_frame_exact_tooling"])

    def test_timestamp_mismatch_fails_the_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state_path = root / "bad_frame_state.jsonl"
            _write_jsonl(
                state_path,
                [
                    {"frame_index": 0, "time_seconds": 0.0},
                    {"frame_index": 1, "time_seconds": 0.5},
                ],
            )
            manifest_path = root / "bad_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "render": {
                            "frame_state_jsonl": str(state_path),
                            "fps": 30,
                            "duration_seconds": 2 / 30,
                            "frame_count": 2,
                            "state_log_every": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_timeline_manifest(manifest_path)

            self.assertEqual(audit["status"], "fail")
            self.assertIn("timestamp_mismatch", audit["issues"])

    def test_capture_records_with_frame_number_and_elapsed_seconds_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            records_path = root / "capture_records.jsonl"
            _write_jsonl(
                records_path,
                [
                    {"frame_number": 1, "elapsed_seconds": 0.0},
                    {"frame_number": 2, "elapsed_seconds": 0.1},
                    {"frame_number": 3, "elapsed_seconds": 0.2},
                ],
            )
            manifest_path = root / "capture_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "records_jsonl": str(records_path),
                        "capture": {"capture_fps": 10},
                        "summary": {"frame_count": 3, "duration_seconds": 0.3},
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_timeline_manifest(manifest_path)

            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["timeline_mode"], "full_frame_exact")
            self.assertEqual(audit["logged_records"], 3)

    def test_output_manifest_shape_with_frame_state_jsonl_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state_path = root / "proof_frame_state.jsonl"
            _write_jsonl(
                state_path,
                [
                    {"frame_index": 0, "time_seconds": 0.0},
                    {"frame_index": 1, "time_seconds": 0.1},
                ],
            )
            manifest_path = root / "proof_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "output": {
                            "frame_state_jsonl": str(state_path),
                            "fps": 10,
                            "frames": 2,
                            "duration_seconds": 0.2,
                            "state_log_every": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_timeline_manifest(manifest_path)

            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["timeline_mode"], "full_frame_exact")
            self.assertTrue(audit["usable_for_frame_exact_tooling"])

    def test_root_relative_log_paths_are_not_doubled_against_manifest_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state_path = root / "storage" / "tmp" / "proof_frame_state.jsonl"
            state_path.parent.mkdir(parents=True)
            _write_jsonl(state_path, [{"frame_index": 0, "time_seconds": 0.0}])
            manifest_dir = root / "storage" / "tmp" / "run"
            manifest_dir.mkdir(parents=True)
            manifest_path = manifest_dir / "proof_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "output": {
                            "frame_state_jsonl": "storage/tmp/proof_frame_state.jsonl",
                            "fps": 10,
                            "frames": 1,
                            "duration_seconds": 0.1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                audit = audit_timeline_manifest(manifest_path)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(audit["status"], "pass")
            self.assertEqual(Path(audit["log_path"]), Path("storage/tmp/proof_frame_state.jsonl"))

    def test_profile_manifest_frame_summaries_are_sampled_exact_not_full_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            profile_path = root / "angular_profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "frame_count": 2,
                        "source": {"metadata": {"fps": 30, "sample_stride": 12, "source_frame_count": 24}},
                        "frame_summaries": [
                            {"frame_index": 0, "time_seconds": 0.0},
                            {"frame_index": 1, "time_seconds": 0.4},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = root / "angular_manifest.json"
            manifest_path.write_text(json.dumps({"profile_json": str(profile_path)}), encoding="utf-8")

            audit = audit_timeline_manifest(manifest_path)

            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["timeline_mode"], "sampled_profile_exact")
            self.assertFalse(audit["full_frame_log"])
            self.assertFalse(audit["usable_for_frame_exact_tooling"])
            self.assertTrue(audit["usable_for_sampled_tooling"])
            self.assertEqual(audit["source_sample_stride"], 12)

            batch = audit_many([manifest_path])
            self.assertEqual(batch["sampled_exact_count"], 1)


if __name__ == "__main__":
    unittest.main()
