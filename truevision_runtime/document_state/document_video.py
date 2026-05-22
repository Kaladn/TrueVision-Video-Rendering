from __future__ import annotations

from typing import Any

from .contracts import DOCUMENT_STATE_SCHEMA_VERSION, READ_ONLY_POLICY
from .hashing import stable_hash


def build_document_video(
    *,
    source_id: str,
    source_hash: str,
    pages: list[dict[str, Any]],
    frame_rate: float = 1.0,
) -> dict[str, Any]:
    """Represent ordered document pages as a one-page-per-frame state video."""

    if not pages:
        raise ValueError("document video requires at least one page frame")
    if frame_rate <= 0:
        raise ValueError("document video frame_rate must be positive")

    frames: list[dict[str, Any]] = []
    for index, page in enumerate(pages):
        page_number = int(page.get("page_number") or index + 1)
        visual_hash = str(page.get("visual_hash") or "")
        frame_basis = {
            "source_id": str(source_id),
            "source_hash": str(source_hash),
            "frame_index": index,
            "page_number": page_number,
            "visual_hash": visual_hash,
        }
        frames.append(
            {
                "schema_version": DOCUMENT_STATE_SCHEMA_VERSION,
                "record_type": "document_page_frame",
                "frame_id": "doc_frame_" + stable_hash(frame_basis)[:16],
                "frame_index": index,
                "frame_timestamp_ms": int(round((index * 1000.0) / frame_rate)),
                "page_number": page_number,
                "width": int(page.get("width") or 0),
                "height": int(page.get("height") or 0),
                "visual_hash": visual_hash,
                "state_recorded_not_copied": True,
                "writes_allowed": dict(READ_ONLY_POLICY),
            }
        )

    video_basis = {
        "source_id": str(source_id),
        "source_hash": str(source_hash),
        "frame_rate": float(frame_rate),
        "frame_ids": [frame["frame_id"] for frame in frames],
    }
    return {
        "schema_version": DOCUMENT_STATE_SCHEMA_VERSION,
        "record_type": "document_video",
        "document_video_id": "doc_video_" + stable_hash(video_basis)[:16],
        "source_id": str(source_id),
        "source_hash": str(source_hash),
        "frame_rate": float(frame_rate),
        "frame_count": len(frames),
        "frames": frames,
        "state_recorded_not_copied": True,
        "truth_boundary": {
            "pages_are_visual_frames": True,
            "text_is_not_primary_state": True,
            "strings_are_derived_output_only": True,
            "missing_pages_remain_missing": True,
        },
        "writes_allowed": dict(READ_ONLY_POLICY),
    }
