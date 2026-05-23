from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .logging import safe_slug, sha256_bytes, sha256_file, stable_json_hash, utc_now, _write_json


_SECTION_PREFIXES = ("intro", "verse", "pre-chorus", "pre chorus", "chorus", "bridge", "outro", "hook", "final chorus")


def _normalize_section(line: str) -> str | None:
    clean = line.strip().strip("[]").strip()
    lowered = clean.lower()
    for prefix in _SECTION_PREFIXES:
        if lowered.startswith(prefix):
            return clean.split("–", 1)[0].split("-", 1)[0].strip() or clean
    return None


def _parse_lyric_lines(lyrics_text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    section = "Unsectioned"
    for raw in lyrics_text.splitlines():
        text = raw.strip()
        if not text:
            continue
        parsed_section = _normalize_section(text)
        if parsed_section:
            section = parsed_section
            continue
        if text.startswith("(") and text.endswith(")"):
            continue
        if text.startswith("[") and text.endswith("]"):
            continue
        words = re.findall(r"[A-Za-z0-9']+", text)
        if not words:
            continue
        lines.append(
            {
                "line_index": len(lines),
                "section": section,
                "text": text,
                "word_candidates": words,
                "word_count": len(words),
            }
        )
    return lines


def _load_segments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("segments JSON must contain a segments list")
    cleaned: list[dict[str, Any]] = []
    for segment in segments:
        start = float(segment["start_seconds"])
        end = float(segment["end_seconds"])
        if end <= start:
            continue
        cleaned.append({**segment, "start_seconds": start, "end_seconds": end, "duration_seconds": end - start})
    if not cleaned:
        raise ValueError("segments JSON contains no usable speech segments")
    return cleaned


def _virtual_to_real_time(segments: list[dict[str, Any]], offset: float) -> tuple[float, int]:
    remaining = max(0.0, offset)
    for index, segment in enumerate(segments):
        duration = float(segment["duration_seconds"])
        if remaining <= duration:
            return float(segment["start_seconds"]) + remaining, index
        remaining -= duration
    return float(segments[-1]["end_seconds"]), len(segments) - 1


def align_lyrics_to_speech_segments(
    segments_json: str | Path,
    *,
    lyrics_text: str | None = None,
    lyrics_path: str | Path | None = None,
    storage_root: str | Path = "storage",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Align provided lyrics to detected speech regions as candidates.

    This is not ASR. The words come from supplied lyrics. Speech state supplies
    the candidate timing windows.
    """
    source_segments = Path(segments_json).expanduser().resolve()
    if not source_segments.exists():
        raise FileNotFoundError(str(source_segments))
    if lyrics_path is not None:
        lyric_source = Path(lyrics_path).expanduser().resolve()
        text = lyric_source.read_text(encoding="utf-8")
        lyric_source_hash = sha256_file(lyric_source)
        lyric_source_path = str(lyric_source)
    elif lyrics_text is not None:
        text = lyrics_text
        lyric_source_hash = sha256_bytes(text.encode("utf-8"))
        lyric_source_path = None
    else:
        raise ValueError("lyrics_text or lyrics_path is required")

    segments = _load_segments(source_segments)
    lines = _parse_lyric_lines(text)
    if not lines:
        raise ValueError("lyrics contain no alignable lines")

    total_vocal_seconds = sum(float(segment["duration_seconds"]) for segment in segments)
    weighted_total = sum(max(1, int(line["word_count"])) for line in lines)
    cursor = 0.0
    aligned: list[dict[str, Any]] = []
    for line in lines:
        share = total_vocal_seconds * (max(1, int(line["word_count"])) / max(1, weighted_total))
        start, start_segment = _virtual_to_real_time(segments, cursor)
        end, end_segment = _virtual_to_real_time(segments, cursor + share)
        if end <= start:
            end = start + 0.001
        involved = segments[start_segment : end_segment + 1]
        confidence = sum(float(item.get("mean_confidence", 0.0)) for item in involved) / max(1, len(involved))
        aligned.append(
            {
                **line,
                "start_seconds": round(start, 6),
                "end_seconds": round(end, 6),
                "duration_seconds": round(end - start, 6),
                "source_segment_start": start_segment,
                "source_segment_end": end_segment,
                "timing_confidence": round(confidence, 6),
                "claim": "candidate_alignment_from_provided_lyrics",
            }
        )
        cursor += share

    root = Path(storage_root).expanduser().resolve()
    for lane in ("artifacts", "manifests", "receipts"):
        (root / lane).mkdir(parents=True, exist_ok=True)
    run = safe_slug(run_id or f"{source_segments.stem}_lyrics_align")
    artifact_root = root / "artifacts" / "truespeech"
    artifact_root.mkdir(parents=True, exist_ok=True)
    alignment_path = artifact_root / f"{run}_lyric_alignment.json"
    manifest_path = root / "manifests" / f"{run}_truespeech_lyric_alignment_manifest.json"
    receipt_path = root / "receipts" / f"{run}_truespeech_lyric_alignment_receipt.json"

    alignment = {
        "schema_version": "truespeech_lyric_alignment_v1",
        "run_id": run,
        "lines": aligned,
    }
    _write_json(alignment_path, alignment)
    manifest = {
        "schema_version": "truespeech_lyric_alignment_manifest_v1",
        "run_id": run,
        "created_at_utc": utc_now(),
        "system": "TrueSpeech In",
        "source_segments": {
            "path": str(source_segments),
            "sha256": sha256_file(source_segments),
        },
        "lyrics": {
            "source_path": lyric_source_path,
            "sha256": lyric_source_hash,
            "line_count": len(lines),
            "word_candidate_count": sum(len(line["word_candidates"]) for line in lines),
        },
        "outputs": {
            "alignment_json": str(alignment_path),
        },
        "summary": {
            "speech_segment_count": len(segments),
            "line_count": len(lines),
            "total_vocal_seconds": round(total_vocal_seconds, 6),
        },
        "boundary": {
            "provided_lyrics_used": True,
            "candidate_alignment_only": True,
            "asr_claim": False,
            "transcript_claim": False,
            "word_recognition_claim": False,
            "speaker_identity_claim": False,
            "audio_state_timing_source": True,
        },
    }
    _write_json(manifest_path, manifest)
    receipt = {
        "receipt_kind": "truespeech_lyric_alignment_receipt_v1",
        "written_at_utc": utc_now(),
        "run_id": run,
        "status": "ok",
        "manifest_sha256": stable_json_hash(manifest),
        "alignment_sha256": sha256_file(alignment_path),
        "boundary": manifest["boundary"],
    }
    _write_json(receipt_path, receipt)
    return {
        "schema_version": "truespeech_lyric_alignment_result_v1",
        "run_id": run,
        "alignment_json": str(alignment_path),
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "line_count": len(lines),
        "word_candidate_count": manifest["lyrics"]["word_candidate_count"],
        "boundary": manifest["boundary"],
    }
