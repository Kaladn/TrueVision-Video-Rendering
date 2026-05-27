import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "truevision_worker_forge.py"


class WorkerForgeScriptTests(unittest.TestCase):
    def test_inventory_command_prints_json_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "scripts" / "truevision_meter_grid.py").write_text("# tool\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "inventory",
                    "--repo-root",
                    str(root),
                    "--storage-root",
                    str(root / "storage"),
                    "--json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["tool_count"], 1)
            self.assertEqual(payload["status"], "manifest_written")

    def test_forge_and_choose_commands_are_manifest_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "scripts" / "truevision_meter_grid.py").write_text("# tool\n", encoding="utf-8")
            storage = root / "storage"
            inventory = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "inventory",
                    "--repo-root",
                    str(root),
                    "--storage-root",
                    str(storage),
                    "--json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            inventory_json = json.loads(inventory.stdout)["manifest_json"]

            forged = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "forge",
                    "--storage-root",
                    str(storage),
                    "--requested-by",
                    "operator",
                    "--chat-text",
                    "need a meter grid worker",
                    "--tool-name",
                    "truevision_meter_grid",
                    "--organ",
                    "truevision",
                    "--purpose",
                    "measure capture cells",
                    "--json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            choice = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "choose",
                    "--storage-root",
                    str(storage),
                    "--inventory-manifest",
                    inventory_json,
                    "--request-text",
                    "meter grid evidence",
                    "--json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(forged.returncode, 0, forged.stderr)
            self.assertEqual(choice.returncode, 0, choice.stderr)
            forged_payload = json.loads(forged.stdout)
            choice_payload = json.loads(choice.stdout)
            self.assertEqual(forged_payload["status"], "forged_manifest_only")
            self.assertEqual(choice_payload["status"], "candidate_selected")
            self.assertFalse(choice_payload["execution_allowed"])


if __name__ == "__main__":
    unittest.main()
