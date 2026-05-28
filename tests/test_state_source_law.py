import unittest

from truevision_runtime.state_source_law import (
    STATE_SOURCE_LAW_LINES,
    build_visualization_boundary,
    classify_artifact_authority,
    is_source_truth_allowed,
)


class StateSourceLawTests(unittest.TestCase):
    def test_law_lines_are_locked_for_receipts_and_docs(self):
        self.assertIn("If it is raw pixels, it is not the TrueVision source.", STATE_SOURCE_LAW_LINES)
        self.assertIn("If it is state, it can be replayed.", STATE_SOURCE_LAW_LINES)
        self.assertIn("If it is replayed, it is derived.", STATE_SOURCE_LAW_LINES)
        self.assertIn("If it is generated/cartoon, it is visualization.", STATE_SOURCE_LAW_LINES)
        self.assertIn("If it is not state-backed, it does not count.", STATE_SOURCE_LAW_LINES)

    def test_raw_media_is_not_allowed_as_source_truth(self):
        for path in (
            "capture.mp4",
            "witness_desktop.mkv",
            "salvaged_clip.h264",
            "render.png",
            "source.wav",
        ):
            verdict = classify_artifact_authority(path)
            self.assertFalse(verdict["source_truth_allowed"], path)
            self.assertEqual(verdict["authority_class"], "non_authority_media")

    def test_state_artifacts_are_allowed_as_source_truth(self):
        for path in (
            "cell_state_native/run_cells_0001.tvcells",
            "run_records.jsonl",
            "cell_state_npz/run_cells_0001.npz",
            "meter_grid_profile.json",
            "state_focus_manifest.json",
        ):
            self.assertTrue(is_source_truth_allowed(path), path)

    def test_generated_or_replayed_media_boundary_is_explicit(self):
        boundary = build_visualization_boundary(
            output_path="overlay.mp4",
            state_refs=["storage/artifacts/geometry_generation/scene.json"],
            visualization_kind="geometry_overlay",
        )

        self.assertTrue(boundary["derived_from_state"])
        self.assertTrue(boundary["visualization_only"])
        self.assertFalse(boundary["source_truth_allowed"])
        self.assertFalse(boundary["generated_media_is_evidence"])
        self.assertEqual(boundary["state_refs"], ["storage/artifacts/geometry_generation/scene.json"])


if __name__ == "__main__":
    unittest.main()
