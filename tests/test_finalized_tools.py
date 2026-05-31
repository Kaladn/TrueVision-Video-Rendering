from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from truevision_runtime.finalized_tools import (
    copy_finalized_tool,
    finalized_tool_status,
    load_finalized_tools_registry,
)


ROOT = Path(__file__).resolve().parents[1]


class FinalizedToolsTests(unittest.TestCase):
    def test_registry_locks_final_renderer_as_copy_only(self):
        registry = load_finalized_tools_registry(ROOT)
        tool = registry["tools"]["cleveland_graffiti_state_proof"]

        self.assertEqual(tool["path"], "scripts/render_cleveland_graffiti_state_proof.py")
        self.assertEqual(tool["lifecycle"], "finalized")
        self.assertEqual(tool["edit_policy"], "copy_only")
        self.assertTrue(tool["promote_only_as_preset"])

        status = finalized_tool_status(ROOT, "cleveland_graffiti_state_proof")
        self.assertTrue(status["hash_matches"])
        self.assertEqual(status["copy_target_hint"], "scripts/render_stem_art_state_transform_lab.py")

    def test_copy_helper_refuses_to_overwrite_finalized_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy_path = Path(tmp) / "copied_tool.py"
            result = copy_finalized_tool(ROOT, "cleveland_graffiti_state_proof", copy_path)
            self.assertTrue(copy_path.exists())
            self.assertEqual(result["status"], "copied")
            self.assertTrue(result["hash_matches_source"])

            source_path = ROOT / "scripts" / "render_cleveland_graffiti_state_proof.py"
            with self.assertRaises(ValueError):
                copy_finalized_tool(ROOT, "cleveland_graffiti_state_proof", source_path)

            with self.assertRaises(FileExistsError):
                copy_finalized_tool(ROOT, "cleveland_graffiti_state_proof", copy_path)

    def test_cli_copies_finalized_tool_to_new_lab_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "lab_tool.py"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/truevision_copy_finalized_tool.py",
                    "--tool-id",
                    "cleveland_graffiti_state_proof",
                    "--destination",
                    str(target),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn('"status": "copied"', result.stdout)

    def test_cli_status_only_does_not_require_destination(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/truevision_copy_finalized_tool.py",
                "--tool-id",
                "cleveland_graffiti_state_proof",
                "--status-only",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn('"tool_id": "cleveland_graffiti_state_proof"', result.stdout)
        self.assertIn('"edit_policy": "copy_only"', result.stdout)


if __name__ == "__main__":
    unittest.main()
