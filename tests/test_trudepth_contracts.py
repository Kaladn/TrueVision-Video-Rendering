import unittest

from truevision_runtime.learning_intake.trudepth_contracts import (
    build_effect_state_transform_contract,
    build_trudepth_contract_bundle,
    build_trudepth_logging_array_contract,
    build_trudepth_validation_contract,
    build_volumetric_state_field_contract,
)


class TruDepthContractsTests(unittest.TestCase):
    def test_volumetric_state_field_names_depth_and_occlusion_channels(self):
        contract = build_volumetric_state_field_contract("fog")
        self.assertEqual(contract["schema_version"], "truevision_volumetric_state_field_contract_v1")
        self.assertIn("density_slice", contract["channels"])
        self.assertIn("depth_layer", contract["channels"])
        self.assertIn("occlusion_pressure", contract["channels"])
        self.assertTrue(contract["renderer_contract"]["pixels_last"])

    def test_effect_transform_copies_behavior_not_pixels(self):
        contract = build_effect_state_transform_contract("fog")
        self.assertEqual(contract["input"], "effect_state_profile")
        self.assertEqual(contract["output"], "transformed_effect_state")
        self.assertIn("source_pixel_copy", contract["forbidden_controls"])
        self.assertIn("reweight_near_mid_far", contract["operators"])

    def test_logging_array_contains_big_depth_meter_fields(self):
        contract = build_trudepth_logging_array_contract("fog")
        self.assertEqual(contract["schema_version"], "truevision_trudepth_logging_array_contract_v1")
        required = {
            "density_slice_near",
            "density_slice_mid",
            "density_slice_far",
            "reveal_rate",
            "edge_recovery",
            "motion_parallax",
            "parallax_direction_16",
            "angular_energy_16",
            "validation_flags",
        }
        self.assertTrue(required.issubset(set(contract["fields"])))
        self.assertFalse(contract["retention"]["keep_raw_cell_array_by_default"])
        self.assertTrue(contract["retention"]["keep_compact_summaries"])

    def test_validation_contract_keeps_fog_belonging_rules(self):
        contract = build_trudepth_validation_contract("fog")
        self.assertIn("soften_edges", contract["belongs_rules"])
        self.assertIn("reveal_nearer_objects_first", contract["belongs_rules"])
        self.assertIn("no_source_frames_used", contract["pass_conditions"])

    def test_bundle_is_hashable_and_carries_law(self):
        bundle = build_trudepth_contract_bundle("fog")
        self.assertEqual(bundle["schema_version"], "truevision_trudepth_contract_bundle_v1")
        self.assertEqual(bundle["law"]["name"], "TruDepth Law")
        self.assertTrue(bundle["law"]["hard_boundary"]["copy_behavior_profile"])
        self.assertFalse(bundle["law"]["hard_boundary"]["copy_source_frames"])
        self.assertTrue(bundle["bundle_hash"].startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
