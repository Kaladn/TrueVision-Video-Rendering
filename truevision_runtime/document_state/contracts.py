from __future__ import annotations

from typing import Any

from .hashing import stable_hash


DOCUMENT_STATE_SCHEMA_VERSION = "truevision_document_state@1"
READ_ONLY_POLICY = {
    "lexicon": False,
    "lifetime_counts": False,
    "source_pages": False,
    "raw_text": False,
}


def build_glyph_state_record(
    *,
    source_id: str,
    frame_id: str,
    page_number: int,
    glyph_ordinal: int,
    glyph_id: str,
    glyph_symbol: str,
    bbox: dict[str, int | float],
    confidence: float,
    match_type: str,
    pattern_hash: str,
    lifetime_count: int = 0,
    recognition_status: str = "recognized",
) -> dict[str, Any]:
    state_basis = {
        "source_id": str(source_id),
        "frame_id": str(frame_id),
        "page_number": int(page_number),
        "glyph_ordinal": int(glyph_ordinal),
        "glyph_id": str(glyph_id),
        "glyph_symbol": str(glyph_symbol),
        "bbox": dict(bbox),
        "confidence": float(confidence),
        "match_type": str(match_type),
        "pattern_hash": str(pattern_hash),
        "lifetime_count": int(lifetime_count),
        "recognition_status": str(recognition_status),
    }
    return {
        "schema_version": DOCUMENT_STATE_SCHEMA_VERSION,
        "record_type": "glyph_state",
        **state_basis,
        "state_hash": stable_hash({"record_type": "glyph_state", **state_basis}),
        "state_recorded_not_copied": True,
        "truth_boundary": {
            "glyphs_are_observed_marks": True,
            "strings_are_derived_output_only": True,
            "lexicon_is_read_only": True,
            "lifetime_counts_are_read_only": True,
        },
        "writes_allowed": dict(READ_ONLY_POLICY),
    }


def build_document_state_read(
    *,
    source_id: str,
    frame_id: str,
    page_number: int,
    glyph_records: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered_symbols = [
        str(record.get("glyph_symbol") or "")
        for record in glyph_records
        if str(record.get("recognition_status") or "") == "recognized" and record.get("glyph_symbol")
    ]
    read_basis = {
        "source_id": str(source_id),
        "frame_id": str(frame_id),
        "page_number": int(page_number),
        "glyph_state_hashes": [str(record.get("state_hash") or "") for record in glyph_records],
        "ordered_symbols": ordered_symbols,
    }
    return {
        "schema_version": DOCUMENT_STATE_SCHEMA_VERSION,
        "record_type": "document_state_read",
        "source_id": str(source_id),
        "frame_id": str(frame_id),
        "page_number": int(page_number),
        "glyph_record_count": len(glyph_records),
        "glyph_record_refs": [str(record.get("state_hash") or "") for record in glyph_records],
        "glyph_records": [dict(record) for record in glyph_records],
        "ordered_symbols": ordered_symbols,
        "derived_text": "".join(ordered_symbols),
        "read_hash": stable_hash({"record_type": "document_state_read", **read_basis}),
        "state_recorded_not_copied": True,
        "render_allowed": bool(ordered_symbols),
        "truth_boundary": {
            "page_is_visual_state": True,
            "glyphs_are_observed_marks": True,
            "strings_are_derived_output_only": True,
            "missing_glyphs_remain_missing": True,
            "ocr_is_not_authority": True,
        },
        "writes_allowed": dict(READ_ONLY_POLICY),
    }
