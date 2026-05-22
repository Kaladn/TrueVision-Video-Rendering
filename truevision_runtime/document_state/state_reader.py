from __future__ import annotations

from typing import Any

from .contracts import build_document_state_read, build_glyph_state_record
from .glyph_lexicon import GlyphLexicon
from .lifetime_counts import LifetimeCounts


class DocumentStateReader:
    """Read document frame glyph cells as visual state, not authoritative text."""

    def __init__(self, *, lexicon: GlyphLexicon, lifetime_counts: LifetimeCounts) -> None:
        self._lexicon = lexicon
        self._lifetime_counts = lifetime_counts

    def read_page_frame(
        self,
        *,
        source_id: str,
        frame_id: str,
        page_number: int,
        glyph_cells: list[dict[str, Any]],
    ) -> dict[str, Any]:
        glyph_records: list[dict[str, Any]] = []
        for ordinal, cell in enumerate(glyph_cells):
            rows = [str(row) for row in cell.get("rows") or []]
            match = self._lexicon.match(rows)
            recognized = match.match_type != "unknown"
            lifetime = self._lifetime_counts.lookup(match.glyph_id)
            glyph_records.append(
                build_glyph_state_record(
                    source_id=source_id,
                    frame_id=frame_id,
                    page_number=page_number,
                    glyph_ordinal=ordinal,
                    glyph_id=match.glyph_id,
                    glyph_symbol=match.display if recognized else "",
                    bbox=dict(cell.get("bbox") or {}),
                    confidence=match.confidence,
                    match_type=match.match_type,
                    pattern_hash=match.pattern_hash,
                    lifetime_count=int(lifetime.get("observed_count") or 0),
                    recognition_status="recognized" if recognized else "unknown",
                )
            )

        return build_document_state_read(
            source_id=source_id,
            frame_id=frame_id,
            page_number=page_number,
            glyph_records=glyph_records,
        )
