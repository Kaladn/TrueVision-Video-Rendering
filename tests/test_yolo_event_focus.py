from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from truevision_runtime.yolo_event_focus import (
    build_yolo_event_focus_report,
    discover_video_sources,
)


ROOT = Path(__file__).resolve().parents[1]


class YoloEventFocusTests(unittest.TestCase):
    def test_discovers_phone_video_candidates_without_copying_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phone = root / "MP4 from phone"
            phone.mkdir()
            (phone / "20260525_120631.mp4").write_bytes(b"fake")
            (phone / "notes.txt").write_text("skip", encoding="utf-8")

            found = discover_video_sources([phone])

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["path"], str(phone / "20260525_120631.mp4"))
        self.assertEqual(found[0]["source_kind"], "phone_video_candidate")
        self.assertFalse(found[0]["raw_video_copied"])

    def test_yolo_report_turns_detections_into_candidate_focus_events(self):
        observations = [
            {
                "frame_index": 10,
                "time_sec": 1.0,
                "detections": [
                    {
                        "class_name": "car",
                        "confidence": 0.82,
                        "xywhn": [0.45, 0.50, 0.20, 0.18],
                    },
                    {
                        "class_name": "traffic light",
                        "confidence": 0.71,
                        "xywhn": [0.70, 0.22, 0.05, 0.10],
                    },
                ],
            },
            {
                "frame_index": 20,
                "time_sec": 2.0,
                "detections": [
                    {
                        "class_name": "car",
                        "confidence": 0.77,
                        "xywhn": [0.47, 0.52, 0.22, 0.17],
                    }
                ],
            },
        ]
        report = build_yolo_event_focus_report(
            video_path=Path("C:/Videos/phone.mp4"),
            model_path=Path("storage/models/yolo/yolo11n.pt"),
            observations=observations,
            metadata={"fps": 30.0, "width": 1920, "height": 1080, "duration_seconds": 3.0},
            sample_policy={"sample_fps": 1.0, "max_frames": 8},
        )

        self.assertEqual(report["schema_version"], "truevision_yolo_event_focus_report_v1")
        self.assertFalse(report["boundary"]["yolo_truth_authority"])
        self.assertFalse(report["boundary"]["raw_video_copied"])
        self.assertFalse(report["boundary"]["frames_retained"])
        self.assertFalse(report["boundary"]["six_one_six_mapping_enabled"])

        event_types = {event["candidate_type"] for event in report["focus_events"]}
        self.assertIn("candidate_vehicle", event_types)
        self.assertIn("candidate_traffic_light", event_types)
        car_event = next(event for event in report["focus_events"] if event["candidate_label"] == "car")
        self.assertIn("meter_grid_from_capture", car_event["recommended_truevision_logs"])
        self.assertIn("state_focus_lens", car_event["recommended_truevision_logs"])

    def test_cli_dry_run_writes_report_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_root = root / "videos"
            output_root = root / "out"
            video_root.mkdir()
            (video_root / "clip.mp4").write_bytes(b"fake")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/truevision_yolo_event_focus.py",
                    "--video-root",
                    str(video_root),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "dry",
                    "--dry-run",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            report = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], "truevision_yolo_event_focus_batch_v1")
        self.assertEqual(report["status"], "dry_run_discovery")
        self.assertEqual(len(report["video_sources"]), 1)
        self.assertFalse(report["boundary"]["raw_video_copied"])


if __name__ == "__main__":
    unittest.main()
