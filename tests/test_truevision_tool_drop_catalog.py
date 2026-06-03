import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_DROP = ROOT / "tool_drop"
CATALOG_PATH = TOOL_DROP / "TRUEVISION_TOOL_DROP_CATALOG.json"


REQUIRED_CATEGORIES = [
    "00_core_contracts",
    "10_state_recording",
    "20_state_recognition",
    "30_meter_profile_learning",
    "40_trueaudio_state",
    "50_state_replay_player",
    "60_state_render_proofs",
    "70_studio_planning",
    "80_finalized_copy_only",
    "90_native_rust_power",
    "99_parked_experimental",
]


REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "tool_id",
    "display_name",
    "category",
    "status",
    "implementation_owned_by_tool_drop",
    "entrypoint_type",
    "entrypoint",
    "implementation_paths",
    "runtime",
    "input_types",
    "output_types",
    "writes",
    "calls_tools",
    "source_truth_compliant",
    "raw_video_saved",
    "raw_media_saved",
    "starts_render",
    "safe_for_sc_call",
    "observes_state",
    "abstracts_behavior",
    "generates_state",
    "renders_media",
    "copies_source_media",
    "behavior_profiles_supported",
    "forward_inputs",
    "reverse_inputs",
    "state_outputs",
    "media_outputs_optional",
    "behavior_family",
    "can_witness",
    "can_profile",
    "can_plan",
    "can_replay",
    "can_surface",
}


class TrueVisionToolDropCatalogTests(unittest.TestCase):
    def _load_catalog(self) -> dict:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def _load_manifest(self, manifest: str) -> dict:
        return json.loads((TOOL_DROP / manifest).read_text(encoding="utf-8"))

    def test_catalog_root_and_categories_exist(self):
        self.assertTrue((TOOL_DROP / "README.md").exists())
        self.assertTrue(CATALOG_PATH.exists())
        catalog = self._load_catalog()
        self.assertEqual(catalog["catalog_id"], "truevision_tool_drop_catalog")
        self.assertEqual(catalog["implementation_policy"], "catalog_only_no_code_moves")
        self.assertEqual(catalog["categories"], REQUIRED_CATEGORIES)
        for category in REQUIRED_CATEGORIES:
            self.assertTrue((TOOL_DROP / category).is_dir(), category)

    def test_every_catalog_tool_manifest_exists_and_has_required_fields(self):
        catalog = self._load_catalog()
        self.assertGreaterEqual(len(catalog["tools"]), 20)
        for row in catalog["tools"]:
            manifest_path = TOOL_DROP / row["manifest"]
            self.assertTrue(manifest_path.exists(), row)
            payload = self._load_manifest(row["manifest"])
            self.assertTrue(REQUIRED_MANIFEST_FIELDS.issubset(payload.keys()), payload.get("tool_id"))
            self.assertEqual(payload["tool_id"], row["tool_id"])
            self.assertEqual(payload["tool_id"], manifest_path.name.removesuffix(".tool.json"))
            self.assertEqual(payload["category"], manifest_path.parent.name)
            self.assertIn(payload["category"], REQUIRED_CATEGORIES)
            self.assertFalse(payload["implementation_owned_by_tool_drop"])
            self.assertIsInstance(payload["implementation_paths"], list)
            self.assertGreaterEqual(len(payload["implementation_paths"]), 1)
            self.assertIsInstance(payload["observes_state"], bool)
            self.assertIsInstance(payload["abstracts_behavior"], bool)
            self.assertIsInstance(payload["generates_state"], bool)
            self.assertIsInstance(payload["renders_media"], bool)
            self.assertIsInstance(payload["copies_source_media"], bool)
            self.assertIsInstance(payload["behavior_profiles_supported"], list)
            self.assertIsInstance(payload["forward_inputs"], list)
            self.assertIsInstance(payload["reverse_inputs"], list)
            self.assertIsInstance(payload["state_outputs"], list)
            self.assertIsInstance(payload["media_outputs_optional"], list)
            self.assertIsInstance(payload["behavior_family"], str)
            self.assertTrue(payload["behavior_family"])
            self.assertIsInstance(payload["can_witness"], bool)
            self.assertIsInstance(payload["can_profile"], bool)
            self.assertIsInstance(payload["can_plan"], bool)
            self.assertIsInstance(payload["can_replay"], bool)
            self.assertIsInstance(payload["can_surface"], bool)
            self.assertIsInstance(payload["raw_media_saved"], bool)
            for path in payload["implementation_paths"]:
                self.assertTrue((ROOT / path).exists(), f"{payload['tool_id']} -> {path}")

    def test_catalog_declares_active_watcher_and_finalized_copy_only_tool(self):
        catalog = self._load_catalog()
        rows = {row["tool_id"]: row for row in catalog["tools"]}
        self.assertIn("truevision_state_loop_law", rows)
        self.assertIn("truevision_state_video_watcher", rows)
        self.assertIn("truevision_document_state_movie", rows)
        self.assertIn("cleveland_graffiti_state_proof", rows)

        state_loop_law = self._load_manifest(rows["truevision_state_loop_law"]["manifest"])
        self.assertEqual(state_loop_law["status"], "contract_reference")
        self.assertIn("state_loop_contract", state_loop_law["behavior_profiles_supported"])
        self.assertIn("witness", state_loop_law["state_vocabulary"])
        self.assertIn("surface", state_loop_law["state_vocabulary"])

        watcher = self._load_manifest(rows["truevision_state_video_watcher"]["manifest"])
        self.assertEqual(watcher["status"], "active_callable")
        self.assertIn("truevision_resonance_recorder", watcher["calls_tools"])
        self.assertIn("meter_grid_from_capture", watcher["calls_tools"])
        self.assertFalse(watcher["raw_video_saved"])
        self.assertFalse(watcher["starts_render"])
        self.assertTrue(watcher["observes_state"])
        self.assertFalse(watcher["generates_state"])
        self.assertFalse(watcher["renders_media"])

        document_state_movie = self._load_manifest(rows["truevision_document_state_movie"]["manifest"])
        self.assertEqual(document_state_movie["status"], "active_callable")
        self.assertEqual(document_state_movie["behavior_family"], "document_visual_state")
        self.assertTrue(document_state_movie["can_witness"])
        self.assertTrue(document_state_movie["can_profile"])
        self.assertFalse(document_state_movie["can_plan"])
        self.assertTrue(document_state_movie["can_replay"])
        self.assertTrue(document_state_movie["can_surface"])
        self.assertFalse(document_state_movie["raw_media_saved"])
        self.assertFalse(document_state_movie["copies_source_media"])
        self.assertFalse(document_state_movie["boundary"]["anchorworks_runtime_dependency"])
        self.assertTrue(document_state_movie["boundary"]["pages_are_visual_state"])
        self.assertTrue(document_state_movie["boundary"]["surface_is_derived_display"])

        finalized = self._load_manifest(rows["cleveland_graffiti_state_proof"]["manifest"])
        self.assertEqual(finalized["category"], "80_finalized_copy_only")
        self.assertEqual(finalized["status"], "finalized_copy_only")
        self.assertEqual(finalized["edit_policy"], "copy_only")
        self.assertTrue(finalized["starts_render"])
        self.assertTrue(finalized["renders_media"])
        self.assertFalse(finalized["copies_source_media"])

    def test_no_active_manifest_claims_raw_video_as_source_truth(self):
        catalog = self._load_catalog()
        for row in catalog["tools"]:
            payload = self._load_manifest(row["manifest"])
            if payload["status"] in {"active_callable", "finalized_copy_only"}:
                self.assertFalse(payload["raw_video_saved"], payload["tool_id"])
                self.assertFalse(payload["raw_media_saved"], payload["tool_id"])
                self.assertFalse(payload["copies_source_media"], payload["tool_id"])
                self.assertNotEqual(payload["source_truth_compliant"], "raw_video")

    def test_state_loop_capable_active_tools_declare_state_outputs(self):
        catalog = self._load_catalog()
        for row in catalog["tools"]:
            payload = self._load_manifest(row["manifest"])
            if payload["status"] != "active_callable":
                continue
            state_capable = any(
                bool(payload[field])
                for field in ["can_witness", "can_profile", "can_plan", "can_replay"]
            )
            if state_capable:
                self.assertGreater(len(payload["state_outputs"]), 0, payload["tool_id"])

    def test_media_outputs_are_optional_surfaces_only(self):
        catalog = self._load_catalog()
        for row in catalog["tools"]:
            payload = self._load_manifest(row["manifest"])
            if payload["media_outputs_optional"]:
                self.assertTrue(payload["can_surface"], payload["tool_id"])
                self.assertFalse(payload["raw_media_saved"], payload["tool_id"])
                self.assertFalse(payload["copies_source_media"], payload["tool_id"])

    def test_behavior_profile_tools_are_state_first_not_media_copy_tools(self):
        catalog = self._load_catalog()
        rows = {row["tool_id"]: row for row in catalog["tools"]}
        for tool_id in [
            "meter_grid_from_capture",
            "atmosphere_profile_from_capture",
            "element_creation_profile_from_capture",
        ]:
            payload = self._load_manifest(rows[tool_id]["manifest"])
            self.assertTrue(payload["observes_state"], tool_id)
            self.assertTrue(payload["abstracts_behavior"], tool_id)
            self.assertFalse(payload["copies_source_media"], tool_id)
            self.assertGreater(len(payload["behavior_profiles_supported"]), 0, tool_id)
            self.assertGreater(len(payload["state_outputs"]), 0, tool_id)

    def test_fog_reveal_behavior_family_has_complete_two_way_path(self):
        catalog = self._load_catalog()
        family_tools = [
            self._load_manifest(row["manifest"])
            for row in catalog["tools"]
            if self._load_manifest(row["manifest"])["behavior_family"] == "fog_reveal"
        ]

        self.assertGreaterEqual(len(family_tools), 2)
        self.assertTrue(any(tool["can_witness"] for tool in family_tools))
        self.assertTrue(any(tool["can_profile"] for tool in family_tools))
        self.assertTrue(any(tool["can_plan"] for tool in family_tools))
        self.assertTrue(any(tool["can_replay"] for tool in family_tools))
        self.assertTrue(any(tool["can_surface"] for tool in family_tools))
        self.assertTrue(all(not tool["copies_source_media"] for tool in family_tools))

    def test_state_loop_law_doc_locks_truevision_terms(self):
        law_path = ROOT / "docs" / "TRUEVISION_STATE_LOOP_LAW.md"
        self.assertTrue(law_path.exists())
        law = law_path.read_text(encoding="utf-8")

        for required in [
            "Witness",
            "Profile",
            "Plan",
            "Replay",
            "Surface",
            "TrueVision does not record video to make video.",
            "TrueVision surfaces planned state as media.",
            "Every serious TrueVision behavior family should be two-way unless proven otherwise.",
        ]:
            self.assertIn(required, law)


if __name__ == "__main__":
    unittest.main()
