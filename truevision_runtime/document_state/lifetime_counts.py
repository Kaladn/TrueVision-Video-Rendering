from __future__ import annotations

from typing import Any


class LifetimeCounts:
    """Read-only lifetime context for observed glyph state."""

    def __init__(self, by_glyph_id: dict[str, dict[str, Any]]) -> None:
        self._by_glyph_id = {str(key): dict(value) for key, value in by_glyph_id.items()}

    @classmethod
    def empty(cls) -> "LifetimeCounts":
        return cls({})

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "LifetimeCounts":
        by_glyph_id: dict[str, dict[str, Any]] = {}
        for record in records:
            glyph_id = str(record.get("glyph_id") or "")
            if not glyph_id:
                continue
            by_glyph_id[glyph_id] = {
                "glyph_id": glyph_id,
                "observed_count": int(record.get("observed_count") or 0),
                "first_seen_frame": str(record.get("first_seen_frame") or ""),
                "last_seen_frame": str(record.get("last_seen_frame") or ""),
            }
        return cls(by_glyph_id)

    def lookup(self, glyph_id: str) -> dict[str, Any]:
        found = self._by_glyph_id.get(str(glyph_id))
        if found:
            return dict(found)
        return {
            "glyph_id": str(glyph_id),
            "observed_count": 0,
            "first_seen_frame": "",
            "last_seen_frame": "",
        }
