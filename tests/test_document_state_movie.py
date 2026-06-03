import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from truevision_runtime.document_state import (
    extract_black_glyph_patterns_from_state_movie,
    record_document_state_movie,
    replay_document_state_movie_frame,
    write_document_state_surface,
)


def _page_with_black_mark() -> np.ndarray:
    page = np.full((8, 10, 3), 255, dtype=np.uint8)
    page[2:6, 4:6] = 0
    return page


class DocumentStateMovieTests(unittest.TestCase):
    def test_records_document_pages_as_state_movie_without_raw_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = record_document_state_movie(
                source_id="doc-proof",
                page_frames=[_page_with_black_mark()],
                output_root=Path(tmp),
                run_id="doc_state_proof",
                frames_per_page=2,
                fps=2.0,
                grid_shape=(8, 10),
            )

            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))

            self.assertEqual(manifest["record_kind"], "truevision_document_state_movie")
            self.assertEqual(manifest["frame_pages"][0]["frame_start"], 0)
            self.assertEqual(manifest["frame_pages"][0]["frame_end"], 1)
            self.assertFalse(manifest["boundary"]["raw_frame_saved"])
            self.assertFalse(manifest["boundary"]["raw_grid_saved"])
            self.assertFalse(manifest["boundary"]["generated_media_is_evidence"])
            self.assertFalse(manifest["boundary"]["anchorworks_runtime_dependency"])
            self.assertEqual(manifest["config"]["capture_resolution"], [10, 8])
            self.assertTrue(Path(result["cell_state_npz"]).exists())
            self.assertTrue(Path(result["records_jsonl"]).exists())

    def test_replays_page_surface_from_state_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = record_document_state_movie(
                source_id="doc-proof",
                page_frames=[_page_with_black_mark()],
                output_root=Path(tmp),
                run_id="doc_state_proof",
                frames_per_page=1,
                fps=1.0,
                grid_shape=(8, 10),
            )

            replayed = replay_document_state_movie_frame(result["manifest_json"], 0)

            self.assertEqual(replayed.shape, (8, 10, 3))
            self.assertEqual(int(replayed[3, 4, 0]), 0)
            self.assertEqual(int(replayed[0, 0, 0]), 255)

    def test_extracts_glyph_patterns_from_stored_state_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = record_document_state_movie(
                source_id="doc-proof",
                page_frames=[_page_with_black_mark()],
                output_root=Path(tmp),
                run_id="doc_state_proof",
                frames_per_page=1,
                fps=1.0,
                grid_shape=(8, 10),
            )

            glyphs = extract_black_glyph_patterns_from_state_movie(
                manifest_path=result["manifest_json"],
                frame_index=0,
                luma_threshold=128.0,
            )

            self.assertEqual(len(glyphs), 1)
            self.assertEqual(glyphs[0]["bbox"], {"x": 4, "y": 2, "w": 2, "h": 4})
            self.assertEqual(glyphs[0]["pattern"], ["11", "11", "11", "11"])
            self.assertFalse(glyphs[0]["raw_frames_saved"])

    def test_writes_derived_surface_receipt_not_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = record_document_state_movie(
                source_id="doc-proof",
                page_frames=[_page_with_black_mark()],
                output_root=root,
                run_id="doc_state_proof",
                frames_per_page=1,
                fps=1.0,
                grid_shape=(8, 10),
            )
            out = root / "surface.png"
            receipt_path = root / "surface_receipt.json"

            receipt = write_document_state_surface(
                manifest_path=result["manifest_json"],
                frame_index=0,
                output_path=out,
                receipt_path=receipt_path,
            )

            self.assertTrue(out.exists())
            self.assertTrue(receipt_path.exists())
            self.assertTrue(receipt["boundary"]["source_truth_is_state"])
            self.assertTrue(receipt["boundary"]["surface_is_derived_display"])
            self.assertFalse(receipt["boundary"]["raw_page_saved"])
            self.assertFalse(receipt["boundary"]["generated_media_is_evidence"])
            image = cv2.imread(str(out), cv2.IMREAD_COLOR)
            self.assertEqual(tuple(image.shape[:2]), (8, 10))


if __name__ == "__main__":
    unittest.main()
