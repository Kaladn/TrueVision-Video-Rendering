import json
import tempfile
import unittest
from pathlib import Path

from truevision_region_snip import (
    Region,
    build_recorder_command,
    load_preset,
    save_preset,
    snap_region_to_truevision,
)


class TrueVisionRegionSnipTests(unittest.TestCase):
    def test_snap_region_preserves_center_and_forces_16_by_9(self):
        region = Region(left=100, top=120, width=500, height=500)

        snapped = snap_region_to_truevision(region, bounds=Region(0, 0, 1920, 1080))

        self.assertEqual(snapped.width % 16, 0)
        self.assertEqual(snapped.height % 9, 0)
        self.assertAlmostEqual(snapped.width / snapped.height, 16 / 9, places=3)
        self.assertGreaterEqual(snapped.left, 0)
        self.assertGreaterEqual(snapped.top, 0)
        self.assertLessEqual(snapped.right, 1920)
        self.assertLessEqual(snapped.bottom, 1080)

    def test_save_and_load_preset_records_snap_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preset.json"
            preset = save_preset(
                path=path,
                preset_id="video_window_001",
                selected=Region(10, 20, 640, 400),
                snapped=Region(10, 40, 640, 360),
                monitor=0,
            )

            loaded = load_preset(path)

            self.assertEqual(loaded["preset_id"], "video_window_001")
            self.assertEqual(loaded["selected_region"], [10, 20, 640, 400])
            self.assertEqual(loaded["snapped_region"], [10, 40, 640, 360])
            self.assertEqual(loaded["snap_rule"], "16:9_grid_aligned")
            json.dumps(preset)

    def test_build_recorder_command_uses_existing_recorder(self):
        preset = {
            "snapped_region": [640, 360, 1280, 720],
            "capture_resolution": [960, 540],
            "grid": [160, 90],
            "blocks": [16, 9],
            "monitor": 0,
        }

        command = build_recorder_command(
            preset,
            duration=30,
            fps=15,
            output_root=Path("D:/out"),
            run_id="region_test",
        )

        self.assertIn("truevision_resonance_recorder.py", command[1])
        self.assertIn("--region", command)
        self.assertIn("640,360,1280,720", command)
        self.assertIn("--resolution", command)
        self.assertIn("960x540", command)
        self.assertIn("--grid", command)
        self.assertIn("160x90", command)


if __name__ == "__main__":
    unittest.main()
