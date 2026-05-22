import unittest

from truevision_runtime.document_state import (
    DocumentStateReader,
    GlyphLexicon,
    LifetimeCounts,
    build_document_video,
)


class DocumentStateReaderTests(unittest.TestCase):
    def test_document_video_preserves_pages_as_frames(self):
        packet = build_document_video(
            source_id="doc-1",
            source_hash="hash-doc",
            pages=[
                {"page_number": 1, "width": 8, "height": 8, "visual_hash": "hash-page-1"},
                {"page_number": 2, "width": 8, "height": 8, "visual_hash": "hash-page-2"},
            ],
            frame_rate=1.0,
        )

        self.assertEqual(packet["record_type"], "document_video")
        self.assertEqual(packet["frame_count"], 2)
        self.assertEqual(packet["frames"][0]["frame_timestamp_ms"], 0)
        self.assertEqual(packet["frames"][1]["frame_timestamp_ms"], 1000)
        self.assertTrue(packet["truth_boundary"]["pages_are_visual_frames"])
        self.assertFalse(packet["writes_allowed"]["lexicon"])

    def test_reader_uses_read_only_lexicon_and_lifetime_counts(self):
        lexicon = GlyphLexicon.from_records(
            [
                {
                    "glyph_id": "glyph-a",
                    "display": "A",
                    "trim_pattern": ["0110", "1001", "1111", "1001"],
                    "promotion_status": "approved",
                },
                {
                    "glyph_id": "glyph-draft",
                    "display": "X",
                    "trim_pattern": ["1"],
                    "promotion_status": "draft",
                },
            ]
        )
        lifetime = LifetimeCounts.from_records(
            [
                {"glyph_id": "glyph-a", "observed_count": 7, "first_seen_frame": "old-1"},
            ]
        )
        reader = DocumentStateReader(lexicon=lexicon, lifetime_counts=lifetime)

        result = reader.read_page_frame(
            source_id="doc-1",
            frame_id="frame-1",
            page_number=1,
            glyph_cells=[
                {
                    "rows": ["00110", "01001", "01111", "01001"],
                    "bbox": {"x": 2, "y": 3, "w": 5, "h": 4},
                },
                {
                    "rows": ["010", "101", "010"],
                    "bbox": {"x": 12, "y": 3, "w": 3, "h": 3},
                },
            ],
        )

        self.assertEqual(result["record_type"], "document_state_read")
        self.assertEqual(result["glyph_record_count"], 2)
        self.assertEqual(result["ordered_symbols"], ["A"])
        self.assertEqual(result["derived_text"], "A")
        self.assertEqual(result["glyph_records"][0]["glyph_id"], "glyph-a")
        self.assertEqual(result["glyph_records"][0]["lifetime_count"], 7)
        self.assertEqual(result["glyph_records"][1]["recognition_status"], "unknown")
        self.assertEqual(result["glyph_records"][1]["glyph_symbol"], "")
        self.assertTrue(result["truth_boundary"]["strings_are_derived_output_only"])
        self.assertFalse(result["writes_allowed"]["lifetime_counts"])

    def test_reader_is_deterministic_for_same_visual_state(self):
        lexicon = GlyphLexicon.from_records(
            [
                {
                    "glyph_id": "glyph-i",
                    "display": "I",
                    "trim_pattern": ["1", "1", "1"],
                    "promotion_status": "approved",
                }
            ]
        )
        reader = DocumentStateReader(lexicon=lexicon, lifetime_counts=LifetimeCounts.empty())
        kwargs = {
            "source_id": "doc-2",
            "frame_id": "frame-9",
            "page_number": 9,
            "glyph_cells": [{"rows": ["00100", "00100", "00100"], "bbox": {"x": 1, "y": 1, "w": 5, "h": 3}}],
        }

        first = reader.read_page_frame(**kwargs)
        second = reader.read_page_frame(**kwargs)

        self.assertEqual(first["read_hash"], second["read_hash"])
        self.assertEqual(first["glyph_records"][0]["state_hash"], second["glyph_records"][0]["state_hash"])


if __name__ == "__main__":
    unittest.main()
