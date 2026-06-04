import json
import tempfile
import unittest
from pathlib import Path

from truevision_runtime.worker_forge import (
    build_manifest_inventory,
    forge_tool_request,
    choose_local_worker,
    load_jsonl,
)


class WorkerForgeTests(unittest.TestCase):
    def test_manifest_inventory_gathers_workers_and_agent_candidates_without_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "scripts" / "truevision_meter_grid.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
            (root / "trueaudio_runtime").mkdir()
            (root / "trueaudio_runtime" / "logging.py").write_text("# audio worker\n", encoding="utf-8")
            agent_root = root / "transfer_to_securecore" / "truevision_agent_candidates"
            (agent_root / "AGENTS" / "agents").mkdir(parents=True)
            (agent_root / "AGENTS" / "agents" / "sample.agent.json").write_text(
                json.dumps(
                    {
                        "agent_id": "sample_agent",
                        "name": "Sample Agent",
                        "version": "0.1.0",
                        "runtime_language": "python",
                        "entrypoint": "AGENTS/sample.py",
                        "entrypoint_hash": "sha256:" + ("a" * 64),
                        "allowed_reads": ["runtime"],
                        "allowed_writes": [],
                        "requires_approval": False,
                        "approval_phrase": "",
                        "mutation_class": "read_only",
                        "dry_run_supported": True,
                        "log_stream": "agent_decision",
                        "test_command": ["python", "-m", "unittest"],
                        "risk_tier": 1,
                        "prompt_only_allowed": False,
                        "required_params": ["input_path"],
                    }
                ),
                encoding="utf-8",
            )

            result = build_manifest_inventory(
                repo_root=root,
                storage_root=root / "storage",
                agent_candidates_root=agent_root,
            )

            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            worker_paths = {item["path"] for item in manifest["workers"]}
            tool_paths = {item["path"] for item in manifest["tools"]}
            agent_ids = {item["agent_id"] for item in manifest["agent_candidates"]}

            self.assertTrue(manifest["policy"]["manifest_only"])
            self.assertTrue(manifest["policy"]["no_worker_migration_to_securecore"])
            self.assertIn("scripts/truevision_meter_grid.py", tool_paths)
            self.assertIn("trueaudio_runtime/logging.py", worker_paths)
            self.assertIn("sample_agent", agent_ids)
            self.assertEqual(result["tool_count"], 1)
            self.assertEqual(result["worker_count"], 1)
            self.assertEqual(result["agent_candidate_count"], 1)

    def test_request_forge_appends_hash_chained_events_and_receipts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_root = Path(tmpdir)

            first = forge_tool_request(
                storage_root=storage_root,
                requested_by="operator",
                request_text="make a glyph region worker",
                tool_name="glyph_region_worker",
                organ="truevision",
                purpose="find text-like regions",
                input_refs=["storage/manifests/source.json"],
            )
            second = forge_tool_request(
                storage_root=storage_root,
                requested_by="operator",
                request_text="make a shape worker",
                tool_name="geometry_shape_worker",
                organ="truevision",
                purpose="extract geometry only",
                input_refs=[],
            )

            event_records = load_jsonl(storage_root / "events" / "worker_forge.jsonl")

            self.assertEqual(len(event_records), 2)
            self.assertEqual(event_records[0]["event_type"], "tool_request_manifest_forged")
            self.assertEqual(event_records[0]["previous_hash"], "")
            self.assertEqual(event_records[1]["previous_hash"], event_records[0]["record_hash"])
            self.assertTrue(Path(first["manifest_json"]).exists())
            self.assertTrue(Path(first["receipt_json"]).exists())
            self.assertTrue(Path(second["manifest_json"]).exists())
            self.assertTrue(Path(second["receipt_json"]).exists())
            self.assertEqual(first["status"], "forged_manifest_only")
            self.assertEqual(second["status"], "forged_manifest_only")

    def test_invalid_agent_candidate_is_recorded_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            agent_root = root / "transfer_to_securecore" / "truevision_agent_candidates"
            (agent_root / "AGENTS" / "agents").mkdir(parents=True)
            (agent_root / "AGENTS" / "agents" / "bad.agent.json").write_text(
                json.dumps({"agent_id": "bad_agent", "runtime_language": "prompt"}),
                encoding="utf-8",
            )

            result = build_manifest_inventory(
                repo_root=root,
                storage_root=root / "storage",
                agent_candidates_root=agent_root,
            )

            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["agent_candidates"][0]["agent_id"], "bad_agent")
            self.assertEqual(manifest["agent_candidates"][0]["status"], "invalid_manifest")
            self.assertIn("missing:name", manifest["agent_candidates"][0]["errors"])
            self.assertIn("prompt_only_forbidden", manifest["agent_candidates"][0]["errors"])

    def test_choose_local_worker_selects_from_manifest_without_running_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "scripts" / "truevision_meter_grid.py").write_text("# meter worker\n", encoding="utf-8")
            (root / "scripts" / "truevision_driving_school.py").write_text("# driving worker\n", encoding="utf-8")

            inventory = build_manifest_inventory(repo_root=root, storage_root=root / "storage")
            choice = choose_local_worker(
                storage_root=root / "storage",
                inventory_manifest=Path(inventory["manifest_json"]),
                request_text="need meter grid evidence from a capture",
            )

            self.assertEqual(choice["status"], "candidate_selected")
            self.assertEqual(choice["selected_worker"]["name"], "truevision_meter_grid")
            self.assertEqual(choice["selected_worker"]["unit_type"], "tool")
            self.assertFalse(choice["execution_allowed"])
            self.assertTrue(Path(choice["choice_manifest_json"]).exists())
            self.assertTrue(Path(choice["receipt_json"]).exists())


if __name__ == "__main__":
    unittest.main()
