from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hashing import stable_hash


def normalize_trim_pattern(rows: list[str]) -> tuple[str, ...]:
    """Trim blank margins from a binary visual glyph pattern."""

    cleaned = [str(row) for row in rows if "1" in str(row)]
    if not cleaned:
        return tuple()
    left = min(row.index("1") for row in cleaned)
    right = max(row.rindex("1") for row in cleaned)
    return tuple(row[left : right + 1] for row in cleaned)


def pattern_hash(pattern: tuple[str, ...]) -> str:
    return stable_hash({"pattern": list(pattern)})


@dataclass(frozen=True)
class GlyphMatch:
    glyph_id: str
    display: str
    confidence: float
    match_type: str
    pattern_hash: str


class GlyphLexicon:
    """Read-only exact matcher for approved visual glyph state."""

    def __init__(self, by_hash: dict[str, GlyphMatch]) -> None:
        self._by_hash = dict(by_hash)

    @classmethod
    def empty(cls) -> "GlyphLexicon":
        return cls({})

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "GlyphLexicon":
        by_hash: dict[str, GlyphMatch] = {}
        for record in records:
            if record.get("promotion_status") != "approved":
                continue
            pattern = normalize_trim_pattern(list(record.get("trim_pattern") or []))
            if not pattern:
                continue
            digest = pattern_hash(pattern)
            by_hash[digest] = GlyphMatch(
                glyph_id=str(record["glyph_id"]),
                display=str(record["display"]),
                confidence=1.0,
                match_type="exact",
                pattern_hash=digest,
            )
        return cls(by_hash)

    def match(self, rows: list[str]) -> GlyphMatch:
        pattern = normalize_trim_pattern(rows)
        digest = pattern_hash(pattern)
        found = self._by_hash.get(digest)
        if found:
            return found
        return GlyphMatch(
            glyph_id="visual_unknown_glyph",
            display="",
            confidence=0.0,
            match_type="unknown",
            pattern_hash=digest,
        )
