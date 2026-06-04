from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STORAGE_LANES = [
    "artifacts",
    "events",
    "inbox",
    "library",
    "manifests",
    "outbox",
    "presets",
    "receipts",
    "reports",
    "state_chunks",
    "templates",
    "tmp",
]

LIBRARY_DIRECTORIES = [
    "library/source_audio/wav",
    "library/source_audio/mp3",
    "library/source_audio/flac",
    "library/source_audio/other",
    "library/source_video/mp4",
    "library/source_video/mov",
    "library/source_video/mkv",
    "library/source_video/webm",
    "library/source_video/other",
    "library/source_stills/jpg",
    "library/source_stills/png",
    "library/source_stills/webp",
    "library/source_stills/other",
    "library/truevision_captures/screen",
    "library/truevision_captures/video",
    "library/truevision_captures/still",
    "library/capture_units/20_minute/incoming",
    "library/capture_units/20_minute/runs",
    "library/capture_units/20_minute/profiles",
    "library/capture_units/20_minute/reports",
    "library/signature_profiles/fog",
    "library/signature_profiles/smoke",
    "library/signature_profiles/water",
    "library/signature_profiles/camera_motion",
    "library/signature_profiles/gameplay",
    "library/signature_profiles/lighting",
    "library/signature_profiles/color",
    "library/signature_profiles/music_video",
    "library/signature_profiles/other",
    "library/templates/video",
    "library/templates/pictorial",
    "library/templates/calibration",
    "library/trueframegen/inputs",
    "library/trueframegen/outputs",
    "library/trueframegen/traces",
    "library/renders/previews",
    "library/renders/full",
    "library/renders/stills",
    "library/indexes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def ensure_storage_library(root: Path, *, write_readme: bool = True) -> dict[str, Any]:
    root = root.expanduser().resolve()
    created: list[str] = []

    for lane in STORAGE_LANES:
        path = root / lane
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")
        created.append(str(path))

    for relative in LIBRARY_DIRECTORIES:
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")
        created.append(str(path))

    index_path = root / "library" / "indexes" / "library_index.json"
    index_payload = {
        "schema": "truevision_storage_library.v1",
        "written_at_utc": utc_now(),
        "root": str(root),
        "clip_unit_policy": {
            "default_signature_clip_minutes": 20,
            "reason": "long enough for motion/fog signatures, small enough to rerun and discard cleanly",
            "recommended_chunks_for_long_sources": "capture repeated 20-minute windows instead of one huge run",
        },
        "storage_lanes": STORAGE_LANES,
        "library_directories": LIBRARY_DIRECTORIES,
    }
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    if write_readme:
        readme_path = root / "library" / "README.md"
        readme_path.write_text(build_library_readme(root), encoding="utf-8")

    return {
        "root": str(root),
        "created_count": len(created),
        "index": str(index_path),
        "readme": str(root / "library" / "README.md"),
        "clip_unit_minutes": 20,
    }


def build_library_readme(root: Path) -> str:
    return "\n".join(
        [
            "# TrueVision Media Library",
            "",
            "This vault stores audio/video state-media inputs, captures, signatures, renders, and receipts.",
            "",
            "Code stays in the project repo. Heavy generated data lives here.",
            "",
            "## Clip Unit",
            "",
            "Use 20-minute clips for signature learning by default.",
            "",
            "```text",
            "20 minutes is long enough to capture fog drift, camera rhythm, lighting drift, and motion texture.",
            "20 minutes is short enough to index, rerun, compare, and delete without wrecking the drive.",
            "```",
            "",
            "## Directory Rules",
            "",
            "```text",
            "source_audio/      original wav/mp3/flac files",
            "source_video/      original reference video files",
            "source_stills/     original still image references",
            "truevision_captures/ structured TrueVision capture runs",
            "capture_units/20_minute/  queued/running/profiled 20-minute samples",
            "signature_profiles/ learned AV signatures only, grouped by visual behavior",
            "templates/         reusable generation templates",
            "trueframegen/      frame-fill inputs, outputs, and temporal traces",
            "renders/           preview/full/still generated outputs",
            "indexes/           manifest indexes and library maps",
            "```",
            "",
            "## Current Vault",
            "",
            f"```text\n{root}\n```",
            "",
        ]
    )


def storage_report(root: Path) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for lane in STORAGE_LANES:
        path = root / lane
        if not path.exists():
            continue
        size = directory_size(path)
        rows.append(
            {
                "lane": lane,
                "path": str(path),
                "size_bytes": size,
                "size_gib": round(size / (1024**3), 3),
            }
        )
    return sorted(rows, key=lambda item: item["size_bytes"], reverse=True)
