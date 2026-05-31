from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StemArtTransformLabCliTests(unittest.TestCase):
    def test_lab_tool_exposes_phone_water_layout_controls(self):
        result = subprocess.run(
            [sys.executable, "scripts/render_stem_art_state_transform_lab.py", "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--layout", result.stdout)
        self.assertIn("--width", result.stdout)
        self.assertIn("--height", result.stdout)
        self.assertIn("--waterline", result.stdout)
        self.assertIn("--camera-drift", result.stdout)
        self.assertIn("--run-instruction", result.stdout)
        self.assertIn("phone_water_reflection", result.stdout)


if __name__ == "__main__":
    unittest.main()
