import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from truevision_coordinate_youtube_intake import _execute_sample, main


class CoordinateYoutubeIntakeScriptTests(unittest.TestCase):
    def test_execute_sample_writes_meter_grid_before_profile_purge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            sample_run_id = "metered_sample"
            capture_dir = run_root / "captures" / sample_run_id
            capture_dir.mkdir(parents=True)
            manifest = capture_dir / f"{sample_run_id}_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            calls = []

            class FakeProcess:
                returncode = 0

                def communicate(self, timeout):
                    calls.append(("capture", timeout))
                    return "", ""

            def fake_meter(args, *, storage_root):
                calls.append(("meter", Path(args["manifest"]).name))
                return {
                    "profile_json": str(root / "meter.json"),
                    "receipt_json": str(root / "meter_receipt.json"),
                    "graphs": {"luma_curve.png": str(root / "luma_curve.png")},
                    "event_status": "visually_supported",
                }

            def fake_profile(*, manifest_path, element_id, run_id, storage_root):
                calls.append(("profile", Path(manifest_path).name))
                profile_path = root / "profile.json"
                profile_path.write_text("{}", encoding="utf-8")
                return {
                    "profile_json": str(profile_path),
                    "profile_sha256": "sha256:abc",
                    "receipt_json": str(root / "profile_receipt.json"),
                    "sampled_frames": 8,
                    "creation_signature": {
                        "transition_behavior": {"motion_mean": 0.1, "motion_abs_mean": 0.1},
                        "shape_behavior": {"center_drift_xy": [0.0, 0.0]},
                        "growth_decay": {"volatility": 0.0},
                    },
                    "six_one_six_windows": 0,
                    "purge": {"status": "purged", "deleted_bytes": 1},
                }

            sample_entry = {
                "source": {
                    "source_order": 1,
                    "category": "Lightning",
                    "element_id": "lightning_arc_bloom",
                    "video_id": "abc123",
                    "source_url": "https://www.youtube.com/watch?v=abc123",
                },
                "sample": {
                    "sample_index": 1,
                    "sample_navigation_url": "https://www.youtube.com/watch?v=abc123&t=0s",
                    "duration_seconds": 1.0,
                },
                "sample_run_id": sample_run_id,
                "coordinate_plan": {"native_capture_command": ["python", "-c", "pass"]},
            }
            coordinate_map = {
                "map_id": "temp",
                "map_sha256": "sha256:abc123",
                "capture_region": [0, 0, 100, 100],
                "points": {"address_bar": [1, 1], "video_play": [2, 2]},
            }
            patches = [
                patch("truevision_coordinate_youtube_intake._paste_url_in_existing_browser"),
                patch("truevision_coordinate_youtube_intake._mouse_click"),
                patch("truevision_coordinate_youtube_intake.subprocess.Popen", return_value=FakeProcess()),
                patch("truevision_coordinate_youtube_intake._find_capture_manifest", return_value=manifest),
                patch("truevision_coordinate_youtube_intake.write_meter_grid_from_capture", fake_meter),
                patch("truevision_coordinate_youtube_intake._profile_capture", fake_profile),
            ]
            for p in patches:
                p.start()
                self.addCleanup(p.stop)

            result = _execute_sample(
                sample_entry,
                coordinate_map=coordinate_map,
                run_root=run_root,
                storage_root=root / "storage",
                load_wait_seconds=0.0,
                pre_play_seconds=0.0,
            )

        self.assertEqual([item[0] for item in calls], ["capture", "meter", "profile"])
        self.assertEqual(result["meter_grid"]["event_status"], "visually_supported")
        self.assertIn("luma_curve.png", result["meter_grid"]["graphs"])

    def test_execute_max_samples_limits_work_after_queue_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            approved = tmp_path / "approved.md"
            approved.write_text(
                "Fire\n\n"
                "https://www.youtube.com/watch?v=long111\n\n"
                "https://www.youtube.com/watch?v=short11\n\n"
                "https://www.youtube.com/watch?v=short22\n",
                encoding="utf-8",
            )
            coord = tmp_path / "map.json"
            coord.write_text(
                json.dumps(
                    {
                        "schema_version": "truevision_coordinate_surface_map_v1",
                        "screen_size": [2000, 1000],
                        "points": {"address_bar": [100, 20], "video_play": [800, 450]},
                        "capture_region": [0, 0, 1600, 900],
                    }
                ),
                encoding="utf-8",
            )
            run_root = tmp_path / "run"
            metadata = {
                "long111": {
                    "video_title": "Long Fire",
                    "duration_seconds": 3600.0,
                    "source_url": "https://www.youtube.com/watch?v=long111",
                    "address_bar_url": "https://www.youtube.com/watch?v=long111",
                    "video_id": "long111",
                },
                "short11": {
                    "video_title": "Short Fire",
                    "duration_seconds": 60.0,
                    "source_url": "https://www.youtube.com/watch?v=short11",
                    "address_bar_url": "https://www.youtube.com/watch?v=short11",
                    "video_id": "short11",
                },
                "short22": {
                    "video_title": "Second Short Fire",
                    "duration_seconds": 45.0,
                    "source_url": "https://www.youtube.com/watch?v=short22",
                    "address_bar_url": "https://www.youtube.com/watch?v=short22",
                    "video_id": "short22",
                },
            }
            executed = []

            def fake_metadata(url, *, timeout_seconds):
                video_id = url.split("v=", 1)[1]
                return metadata[video_id]

            def fake_execute(entry, **kwargs):
                executed.append(entry["sample_run_id"])
                return {
                    "source_order": entry["source"]["source_order"],
                    "category": entry["source"]["category"],
                    "element_id": entry["source"]["element_id"],
                    "video_id": entry["source"]["video_id"],
                    "sample_index": entry["sample"]["sample_index"],
                    "run_id": entry["sample_run_id"],
                    "sample_url": entry["sample"]["sample_navigation_url"],
                    "status": "ok",
                    "sampled_frames": 1,
                    "purge": {"status": "purged"},
                }

            argv = [
                "truevision_coordinate_youtube_intake.py",
                "--approved-file",
                str(approved),
                "--coordinate-map",
                str(coord),
                "--run-id",
                "limited",
                "--run-root",
                str(run_root),
                "--execute",
                "--max-samples",
                "1",
            ]
            with patch("sys.argv", argv), patch("truevision_coordinate_youtube_intake.fetch_youtube_metadata", fake_metadata), patch(
                "truevision_coordinate_youtube_intake._execute_sample", fake_execute
            ):
                self.assertEqual(main(), 0)

            summary = json.loads((run_root / "coordinate_summary.json").read_text(encoding="utf-8"))
            queue = json.loads((run_root / "coordinate_queue.json").read_text(encoding="utf-8"))

        self.assertEqual(len(executed), 1)
        self.assertEqual(summary["completed_sample_count"], 1)
        self.assertEqual(summary["planned_sample_count"], 6)
        self.assertEqual(summary["skipped_by_max_samples"], 5)
        self.assertTrue(queue["boundary"]["coordinate_map_required_before_run"])
        self.assertTrue(summary["boundary"]["coordinate_map_required_before_run"])
        self.assertRegex(queue["coordinate_map_sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(summary["coordinate_map_sha256"], queue["coordinate_map_sha256"])

    def test_source_window_starts_at_requested_source_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            approved = tmp_path / "approved.md"
            approved.write_text(
                "Fire\n\n"
                "https://www.youtube.com/watch?v=fire01\n\n"
                "Smoke\n\n"
                "https://www.youtube.com/watch?v=smoke1\n\n"
                "https://www.youtube.com/watch?v=smoke2\n\n"
                "Lightning\n\n"
                "https://www.youtube.com/watch?v=light1\n",
                encoding="utf-8",
            )
            coord = tmp_path / "map.json"
            coord.write_text(
                json.dumps(
                    {
                        "schema_version": "truevision_coordinate_surface_map_v1",
                        "screen_size": [2000, 1000],
                        "points": {"address_bar": [100, 20], "video_play": [800, 450]},
                        "capture_region": [0, 0, 1600, 900],
                    }
                ),
                encoding="utf-8",
            )
            run_root = tmp_path / "run_window"
            metadata = {
                "fire01": {"video_title": "Fire", "duration_seconds": 60.0},
                "smoke1": {"video_title": "Smoke One", "duration_seconds": 60.0},
                "smoke2": {"video_title": "Smoke Two", "duration_seconds": 60.0},
                "light1": {"video_title": "Lightning", "duration_seconds": 60.0},
            }
            executed = []

            def fake_metadata(url, *, timeout_seconds):
                video_id = url.split("v=", 1)[1]
                return {"source_url": url, "address_bar_url": url, "video_id": video_id, **metadata[video_id]}

            def fake_execute(entry, **kwargs):
                executed.append(entry["source"]["source_order"])
                return {
                    "source_order": entry["source"]["source_order"],
                    "category": entry["source"]["category"],
                    "element_id": entry["source"]["element_id"],
                    "video_id": entry["source"]["video_id"],
                    "sample_index": entry["sample"]["sample_index"],
                    "run_id": entry["sample_run_id"],
                    "sample_url": entry["sample"]["sample_navigation_url"],
                    "status": "ok",
                    "sampled_frames": 1,
                    "purge": {"status": "purged"},
                }

            argv = [
                "truevision_coordinate_youtube_intake.py",
                "--approved-file",
                str(approved),
                "--coordinate-map",
                str(coord),
                "--run-id",
                "windowed",
                "--run-root",
                str(run_root),
                "--execute",
                "--start-source-order",
                "2",
                "--source-count",
                "2",
            ]
            with patch("sys.argv", argv), patch("truevision_coordinate_youtube_intake.fetch_youtube_metadata", fake_metadata), patch(
                "truevision_coordinate_youtube_intake._execute_sample", fake_execute
            ):
                self.assertEqual(main(), 0)

            queue = json.loads((run_root / "coordinate_queue.json").read_text(encoding="utf-8"))
            summary = json.loads((run_root / "coordinate_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(executed, [2, 3])
        self.assertEqual([source["source_order"] for source in queue["selected_sources"]], [2, 3])
        self.assertEqual(summary["completed_sample_count"], 2)

    def test_all_sources_selects_every_unique_source_from_start_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            approved = tmp_path / "approved.md"
            approved.write_text(
                "Fire\n\n"
                "https://www.youtube.com/watch?v=fire01\n\n"
                "Smoke\n\n"
                "https://www.youtube.com/watch?v=smoke1\n\n"
                "https://www.youtube.com/watch?v=smoke1\n\n"
                "https://www.youtube.com/watch?v=smoke2\n",
                encoding="utf-8",
            )
            coord = tmp_path / "map.json"
            coord.write_text(
                json.dumps(
                    {
                        "schema_version": "truevision_coordinate_surface_map_v1",
                        "screen_size": [2000, 1000],
                        "points": {"address_bar": [100, 20], "video_play": [800, 450]},
                        "capture_region": [0, 0, 1600, 900],
                    }
                ),
                encoding="utf-8",
            )
            run_root = tmp_path / "run_all"
            metadata = {
                "fire01": {"video_title": "Fire", "duration_seconds": 60.0},
                "smoke1": {"video_title": "Smoke One", "duration_seconds": 60.0},
                "smoke2": {"video_title": "Smoke Two", "duration_seconds": 60.0},
            }
            executed = []

            def fake_metadata(url, *, timeout_seconds):
                video_id = url.split("v=", 1)[1]
                return {"source_url": url, "address_bar_url": url, "video_id": video_id, **metadata[video_id]}

            def fake_execute(entry, **kwargs):
                executed.append(entry["source"]["video_id"])
                return {
                    "source_order": entry["source"]["source_order"],
                    "category": entry["source"]["category"],
                    "element_id": entry["source"]["element_id"],
                    "video_id": entry["source"]["video_id"],
                    "sample_index": entry["sample"]["sample_index"],
                    "run_id": entry["sample_run_id"],
                    "sample_url": entry["sample"]["sample_navigation_url"],
                    "status": "ok",
                    "sampled_frames": 1,
                    "purge": {"status": "purged"},
                }

            argv = [
                "truevision_coordinate_youtube_intake.py",
                "--approved-file",
                str(approved),
                "--coordinate-map",
                str(coord),
                "--run-id",
                "all",
                "--run-root",
                str(run_root),
                "--execute",
                "--all-sources",
                "--start-source-order",
                "2",
            ]
            with patch("sys.argv", argv), patch("truevision_coordinate_youtube_intake.fetch_youtube_metadata", fake_metadata), patch(
                "truevision_coordinate_youtube_intake._execute_sample", fake_execute
            ):
                self.assertEqual(main(), 0)

            queue = json.loads((run_root / "coordinate_queue.json").read_text(encoding="utf-8"))
            summary = json.loads((run_root / "coordinate_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(executed, ["smoke1", "smoke2"])
        self.assertEqual(queue["selection_mode"], "all_sources")
        self.assertEqual(queue["selected_source_count"], 2)
        self.assertEqual(summary["completed_sample_count"], 2)

    def test_all_sources_skips_zero_duration_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            approved = tmp_path / "approved.md"
            approved.write_text(
                "Lightning\n\n"
                "https://www.youtube.com/watch?v=bad000\n\n"
                "https://www.youtube.com/watch?v=good01\n",
                encoding="utf-8",
            )
            coord = tmp_path / "map.json"
            coord.write_text(
                json.dumps(
                    {
                        "schema_version": "truevision_coordinate_surface_map_v1",
                        "screen_size": [2000, 1000],
                        "points": {"address_bar": [100, 20], "video_play": [800, 450]},
                        "capture_region": [0, 0, 1600, 900],
                    }
                ),
                encoding="utf-8",
            )
            run_root = tmp_path / "run_zero"

            def fake_metadata(url, *, timeout_seconds):
                video_id = url.split("v=", 1)[1]
                duration = 0.0 if video_id == "bad000" else 60.0
                return {"source_url": url, "address_bar_url": url, "video_id": video_id, "video_title": video_id, "duration_seconds": duration}

            def fake_execute(entry, **kwargs):
                return {
                    "source_order": entry["source"]["source_order"],
                    "category": entry["source"]["category"],
                    "element_id": entry["source"]["element_id"],
                    "video_id": entry["source"]["video_id"],
                    "sample_index": entry["sample"]["sample_index"],
                    "run_id": entry["sample_run_id"],
                    "sample_url": entry["sample"]["sample_navigation_url"],
                    "status": "ok",
                    "sampled_frames": 1,
                    "purge": {"status": "purged"},
                }

            argv = [
                "truevision_coordinate_youtube_intake.py",
                "--approved-file",
                str(approved),
                "--coordinate-map",
                str(coord),
                "--run-id",
                "zero",
                "--run-root",
                str(run_root),
                "--execute",
                "--all-sources",
            ]
            with patch("sys.argv", argv), patch("truevision_coordinate_youtube_intake.fetch_youtube_metadata", fake_metadata), patch(
                "truevision_coordinate_youtube_intake._execute_sample", fake_execute
            ):
                self.assertEqual(main(), 0)

            queue = json.loads((run_root / "coordinate_queue.json").read_text(encoding="utf-8"))

        self.assertEqual([source["video_id"] for source in queue["selected_sources"]], ["good01"])
        self.assertIn("bad000", queue["metadata_by_id"])
        self.assertIn("non_positive_duration", queue["metadata_by_id"]["bad000"]["error"])


if __name__ == "__main__":
    unittest.main()
