import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from truevision_state_video_watcher import (  # noqa: E402
    TOOL_ID,
    WatcherConfig,
    build_capture_command,
    build_post_capture_tool_calls,
    build_recognition_command,
    build_watcher_receipt,
    run_watcher,
)


def _config(root: Path, *, recognize: bool = True, run_tool_calls: bool = True) -> WatcherConfig:
    return WatcherConfig(
        run_id="watcher_test",
        duration_seconds=12.0,
        fps=15,
        resolution="960x540",
        grid="160x90",
        blocks="16x9",
        monitor=0,
        region="10,20,960,540",
        output_root=root / "captures",
        report_root=root / "reports",
        receipt_root=root / "receipts",
        storage_root=root / "storage",
        stop_file=root / "runtime" / "watcher_test.stop",
        start_delay_seconds=0.0,
        save_cell_state=True,
        recognize=recognize,
        run_tool_calls=run_tool_calls,
        recognition_max_frames=300,
        recognition_sample_stride=2,
        tool_profile_max_frames=120,
        tool_profile_sample_stride=3,
    )


class TrueVisionStateVideoWatcherTests(unittest.TestCase):
    def test_capture_command_calls_existing_state_recorder(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            command = build_capture_command(config, python_exe="python")

        self.assertEqual(command[0], "python")
        self.assertTrue(command[1].endswith("truevision_resonance_recorder.py"))
        self.assertIn("--region", command)
        self.assertIn("10,20,960,540", command)
        self.assertIn("--stop-file", command)
        self.assertNotIn("--video", command)
        self.assertNotIn("--raw", command)

    def test_recognition_command_calls_state_recognition_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            command = build_recognition_command(config, python_exe="python")

        self.assertEqual(command[0], "python")
        self.assertTrue(command[1].endswith("truevision_state_recognition.py"))
        self.assertIn("--manifest", command)
        self.assertIn("--sample-stride", command)
        self.assertIn("2", command)

    def test_post_capture_tool_calls_include_profile_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            calls = build_post_capture_tool_calls(config)

        self.assertEqual(
            [call["tool"] for call in calls],
            [
                "meter_grid_from_capture",
                "atmosphere_profile_from_capture",
                "element_creation_profile_from_capture",
            ],
        )
        for call in calls:
            self.assertIn("manifest", call["args"])
            self.assertEqual(call["args"]["max_frames"], 120)
            self.assertEqual(call["args"]["sample_stride"], 3)

    def test_prepare_only_writes_tool_call_receipt_without_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            receipt = run_watcher(config, prepare_only=True)

            receipt_path = config.receipt_path
            self.assertTrue(receipt_path.exists())
            saved = json.loads(receipt_path.read_text(encoding="utf-8"))

        self.assertEqual(receipt["tool_id"], TOOL_ID)
        self.assertEqual(saved["status"], "prepared_not_started")
        tool_ids = [call["tool_id"] for call in saved["tool_calls"]]
        self.assertIn("truevision_resonance_recorder", tool_ids)
        self.assertIn("truevision_state_recognition", tool_ids)
        self.assertIn("meter_grid_from_capture", tool_ids)
        self.assertIn("atmosphere_profile_from_capture", tool_ids)
        self.assertIn("element_creation_profile_from_capture", tool_ids)
        self.assertFalse(saved["boundary"]["raw_video_saved"])
        self.assertFalse(saved["boundary"]["render_started"])

    def test_receipt_records_no_tool_calls_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp), recognize=False, run_tool_calls=False)
            capture = build_capture_command(config, python_exe="python")
            receipt = build_watcher_receipt(
                config,
                status="prepared_not_started",
                capture_command=capture,
                recognition_command=None,
                post_capture_tool_calls=[],
            )

        self.assertEqual(len(receipt["tool_calls"]), 2)
        self.assertFalse(receipt["tool_calls"][1]["enabled"])
        self.assertFalse(receipt["config"]["recognize"])
        self.assertFalse(receipt["config"]["run_tool_calls"])


if __name__ == "__main__":
    unittest.main()
