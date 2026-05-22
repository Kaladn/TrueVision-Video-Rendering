"""Document-as-state-video reader utilities."""

from .contracts import (
    DOCUMENT_STATE_SCHEMA_VERSION,
    build_document_state_read,
    build_glyph_state_record,
)
from .document_video import build_document_video
from .glyph_lexicon import GlyphLexicon, GlyphMatch, normalize_trim_pattern, pattern_hash
from .lifetime_counts import LifetimeCounts
from .state_reader import DocumentStateReader

__all__ = [
    "DOCUMENT_STATE_SCHEMA_VERSION",
    "DocumentStateReader",
    "GlyphLexicon",
    "GlyphMatch",
    "LifetimeCounts",
    "build_document_state_read",
    "build_document_video",
    "build_glyph_state_record",
    "normalize_trim_pattern",
    "pattern_hash",
]
