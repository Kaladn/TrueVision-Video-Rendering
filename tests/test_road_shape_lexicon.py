import json
import tempfile
import unittest
from pathlib import Path

from truevision_runtime.learning_intake.road_shape_lexicon import (
    build_country_road_aw_bridge,
    build_road_shape_lexicon,
    lookup_shape_symbols,
    write_road_shape_lexicon,
)


class RoadShapeLexiconTests(unittest.TestCase):
    def test_lexicon_starts_with_country_road_geometric_shapes(self):
        lexicon = build_road_shape_lexicon()

        self.assertEqual(lexicon["schema_version"], "truevision_recognition_shape_lexicon_v1")
        self.assertEqual(lexicon["authority"], "truevision")
        self.assertEqual(lexicon["route_context"], "northwest_ohio_country_road_to_monroe_michigan")
        self.assertTrue(lexicon["boundary"]["not_aw_word_lexicon"])
        self.assertTrue(lexicon["boundary"]["recognition_items_not_truth_claims"])

        by_name = {entry["name"]: entry for entry in lexicon["symbols"]}
        for name in [
            "circle",
            "octagon",
            "diamond",
            "triangle",
            "vertical_rectangle",
            "horizontal_rectangle",
            "x_cross",
            "lane_line",
            "road_edge_line",
            "utility_pole",
            "mailbox_box",
            "barn_box",
            "reflector_dot",
        ]:
            self.assertIn(name, by_name)

        codes = [entry["binary_code"] for entry in lexicon["symbols"]]
        self.assertEqual(len(codes), len(set(codes)))
        for entry in lexicon["symbols"]:
            self.assertEqual(len(entry["binary_code"]), 16)
            self.assertEqual(set(entry["binary_code"]) <= {"0", "1"}, True)
            self.assertTrue(entry["not_word_anchor"])

    def test_lookup_returns_shape_symbols_without_promoting_meaning(self):
        found = lookup_shape_symbols(["octagon", "vertical_rectangle", "x_cross", "missing_shape"])

        self.assertEqual([item["name"] for item in found["matches"]], ["octagon", "vertical_rectangle", "x_cross"])
        self.assertEqual(found["unknown_names"], ["missing_shape"])
        self.assertTrue(found["boundary"]["symbols_are_recognition_shapes"])
        self.assertFalse(found["boundary"]["traffic_truth_promoted"])

    def test_aw_bridge_maps_language_to_shape_items_without_sharing_ownership(self):
        bridge = build_country_road_aw_bridge()

        self.assertEqual(bridge["schema_version"], "truevision_aw_shape_bridge_v1")
        self.assertEqual(bridge["ownership"]["aw_owns"], "language_anchors")
        self.assertEqual(bridge["ownership"]["truevision_owns"], "recognition_shape_symbols")
        stop = next(row for row in bridge["mappings"] if row["aw_anchor"] == "stop sign")
        self.assertIn("octagon", stop["shape_names"])
        self.assertEqual(stop["promotion_status"], "candidate_mapping")

    def test_write_lexicon_writes_manifest_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = write_road_shape_lexicon(Path(tmp), run_id="shape_lexicon_test")

            self.assertTrue(Path(result["lexicon_json"]).exists())
            self.assertTrue(Path(result["manifest_json"]).exists())
            self.assertTrue(Path(result["receipt_json"]).exists())
            receipt = json.loads(Path(result["receipt_json"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema_version"], "truevision_recognition_shape_lexicon_receipt_v1")
            self.assertFalse(receipt["boundary"]["source_video_retained"])
            self.assertFalse(receipt["boundary"]["aw_lexicon_mutated"])


if __name__ == "__main__":
    unittest.main()
