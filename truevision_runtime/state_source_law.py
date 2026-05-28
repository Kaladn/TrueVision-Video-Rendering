from __future__ import annotations

from pathlib import Path
from typing import Any


STATE_SOURCE_LAW_LINES: tuple[str, ...] = (
    "If it is raw pixels, it is not the TrueVision source.",
    "If it is state, it can be replayed.",
    "If it is replayed, it is derived.",
    "If it is generated/cartoon, it is visualization.",
    "If it is not state-backed, it does not count.",
)


STATE_EXTENSIONS = {
    ".tvcells",
    ".jsonl",
    ".json",
    ".npz",
}


NON_AUTHORITY_MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".h264",
    ".hevc",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".wav",
    ".mp3",
    ".flac",
    ".aac",
    ".m4a",
}


def classify_artifact_authority(path_or_name: str | Path, *, artifact_kind: str = "") -> dict[str, Any]:
    path = Path(str(path_or_name))
    suffix = path.suffix.lower()
    kind = str(artifact_kind or "").strip().lower()

    if kind in {"state", "state_log", "profile", "manifest", "receipt", "cell_state", "temporal_pulse"}:
        return {
            "path": str(path_or_name),
            "artifact_kind": artifact_kind,
            "authority_class": "state_source",
            "source_truth_allowed": True,
            "reason": "explicit state/profile/manifest/receipt kind",
            "law": list(STATE_SOURCE_LAW_LINES),
        }

    if suffix in NON_AUTHORITY_MEDIA_EXTENSIONS:
        return {
            "path": str(path_or_name),
            "artifact_kind": artifact_kind,
            "authority_class": "non_authority_media",
            "source_truth_allowed": False,
            "reason": "raw/rendered media may be input or visualization, never TrueVision source truth",
            "law": list(STATE_SOURCE_LAW_LINES),
        }

    if suffix in STATE_EXTENSIONS:
        return {
            "path": str(path_or_name),
            "artifact_kind": artifact_kind,
            "authority_class": "state_source",
            "source_truth_allowed": True,
            "reason": "state-shaped artifact extension",
            "law": list(STATE_SOURCE_LAW_LINES),
        }

    return {
        "path": str(path_or_name),
        "artifact_kind": artifact_kind,
        "authority_class": "unknown",
        "source_truth_allowed": False,
        "reason": "unknown artifact class must not be promoted as source truth",
        "law": list(STATE_SOURCE_LAW_LINES),
    }


def is_source_truth_allowed(path_or_name: str | Path, *, artifact_kind: str = "") -> bool:
    return bool(classify_artifact_authority(path_or_name, artifact_kind=artifact_kind)["source_truth_allowed"])


def build_visualization_boundary(
    *,
    output_path: str | Path,
    state_refs: list[str],
    visualization_kind: str,
) -> dict[str, Any]:
    return {
        "output_path": str(output_path),
        "visualization_kind": str(visualization_kind),
        "state_refs": list(state_refs),
        "derived_from_state": bool(state_refs),
        "visualization_only": True,
        "source_truth_allowed": False,
        "generated_media_is_evidence": False,
        "raw_pixels_are_source": False,
        "law": list(STATE_SOURCE_LAW_LINES),
    }
