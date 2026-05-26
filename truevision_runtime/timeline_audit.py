from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TIME_TOLERANCE_SECONDS = 1.0e-4


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            rows.append({"_parse_error": str(exc), "_line_number": line_number})
        else:
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _resolve_path(manifest_path: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        if path.exists():
            return path
        path = manifest_path.parent / path
    return path


def _first_number(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value not in {None, ""}:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def _first_int(*values: Any, default: int = 0) -> int:
    number = _first_number(*values, default=float(default))
    return int(round(number))


def _extract_manifest_timing(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    render = manifest.get("render") if isinstance(manifest.get("render"), dict) else {}
    output = manifest.get("output") if isinstance(manifest.get("output"), dict) else {}
    capture = manifest.get("capture") if isinstance(manifest.get("capture"), dict) else {}
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}

    log_path = _resolve_path(
        manifest_path,
        render.get("frame_state_jsonl")
        or output.get("frame_state_jsonl")
        or manifest.get("frame_state_jsonl")
        or manifest.get("records_jsonl")
        or summary.get("records_jsonl"),
    )
    fps = _first_number(
        render.get("fps"),
        output.get("fps"),
        capture.get("capture_fps"),
        capture.get("fps"),
        manifest.get("fps"),
        summary.get("capture_fps"),
        default=0.0,
    )
    frame_count = _first_int(
        render.get("frame_count"),
        output.get("frame_count"),
        output.get("frames"),
        summary.get("frame_count"),
        manifest.get("frame_count"),
        default=0,
    )
    duration_seconds = _first_number(
        render.get("duration_seconds"),
        output.get("duration_seconds"),
        summary.get("duration_seconds"),
        manifest.get("duration_seconds"),
        default=0.0,
    )
    state_log_every = _first_int(render.get("state_log_every"), output.get("state_log_every"), manifest.get("state_log_every"), default=1)
    return {
        "log_path": log_path,
        "profile_path": _resolve_path(manifest_path, manifest.get("profile_json")),
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "state_log_every": max(1, state_log_every),
        "source_sample_stride": 1,
        "source_kind": "timeline_log",
    }


def _profile_timing(profile_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile = _read_json(profile_path)
    source = profile.get("source") if isinstance(profile.get("source"), dict) else {}
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    sample_stride = max(1, _first_int(metadata.get("sample_stride"), default=1))
    source_fps = _first_number(metadata.get("fps"), metadata.get("capture_fps"), default=0.0)
    effective_fps = source_fps / sample_stride if source_fps > 0.0 else 0.0
    frame_summaries = profile.get("frame_summaries")
    rows = frame_summaries if isinstance(frame_summaries, list) else []
    frame_count = _first_int(profile.get("frame_count"), len(rows), default=len(rows))
    duration_seconds = _first_number(
        metadata.get("logical_duration_seconds"),
        metadata.get("source_duration_seconds"),
        frame_count / effective_fps if effective_fps > 0.0 else 0.0,
        default=0.0,
    )
    timing = {
        "log_path": profile_path,
        "profile_path": profile_path,
        "fps": effective_fps,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "state_log_every": sample_stride,
        "source_sample_stride": sample_stride,
        "source_kind": "profile_frame_summaries",
    }
    return timing, rows


def _extract_record_time(row: dict[str, Any]) -> tuple[int | None, float | None, str]:
    if "_parse_error" in row:
        return None, None, "parse_error"
    if "frame_index" in row and "time_seconds" in row:
        return _first_int(row.get("frame_index"), default=0), _first_number(row.get("time_seconds")), "frame_index_time_seconds"
    if "source_frame_index" in row and "time_sec" in row:
        return _first_int(row.get("source_frame_index"), default=0), _first_number(row.get("time_sec")), "source_frame_index_time_sec"
    if "global_frame_index" in row and "time_sec" in row:
        return _first_int(row.get("global_frame_index"), default=0), _first_number(row.get("time_sec")), "global_frame_index_time_sec"
    if "frame_number" in row and "elapsed_seconds" in row:
        return max(0, _first_int(row.get("frame_number"), default=1) - 1), _first_number(row.get("elapsed_seconds")), "frame_number_elapsed_seconds"
    return None, None, "missing_time_fields"


def _most_common_step(indices: list[int]) -> int:
    if len(indices) < 2:
        return 0
    diffs = [right - left for left, right in zip(indices, indices[1:])]
    if not diffs:
        return 0
    return int(Counter(diffs).most_common(1)[0][0])


def audit_timeline_manifest(
    manifest_path: str | Path,
    *,
    tolerance_seconds: float = DEFAULT_TIME_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    """Verify saved TrueVision timeline logs against frame index, FPS, and manifest cadence."""
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    timing = _extract_manifest_timing(manifest_path, manifest)
    issues: list[str] = []

    log_path = timing["log_path"]
    rows: list[dict[str, Any]] | None = None
    if log_path is None:
        profile_path = timing.get("profile_path")
        if profile_path is not None and Path(profile_path).exists():
            timing, rows = _profile_timing(Path(profile_path))
            log_path = timing["log_path"]
        else:
            return {
                "schema_version": "truevision_timeline_audit_v1",
                "manifest_path": str(manifest_path),
                "status": "fail",
                "issues": ["missing_log_path"],
            }
    if not log_path.exists():
        return {
            "schema_version": "truevision_timeline_audit_v1",
            "manifest_path": str(manifest_path),
            "log_path": str(log_path),
            "status": "fail",
            "issues": ["log_path_missing"],
        }
    fps = float(timing["fps"])
    if fps <= 0.0:
        issues.append("missing_or_invalid_fps")

    if rows is None:
        rows = _read_jsonl(log_path)
    indices: list[int] = []
    times: list[float] = []
    formats: list[str] = []
    max_error = 0.0
    worst_record: dict[str, Any] | None = None
    previous_index: int | None = None
    previous_time: float | None = None

    for row in rows:
        frame_index, time_seconds, record_format = _extract_record_time(row)
        formats.append(record_format)
        if frame_index is None or time_seconds is None:
            if record_format == "parse_error":
                issues.append("jsonl_parse_error")
            else:
                issues.append("missing_time_fields")
            continue
        indices.append(frame_index)
        times.append(time_seconds)
        if previous_index is not None and frame_index <= previous_index:
            issues.append("non_monotonic_frame_index")
        if previous_time is not None and time_seconds < previous_time:
            issues.append("non_monotonic_time")
        previous_index = frame_index
        previous_time = time_seconds
        if fps > 0.0:
            expected = frame_index / fps
            error = abs(time_seconds - expected)
            if error > max_error:
                max_error = error
                worst_record = {
                    "line_number": row.get("_line_number"),
                    "frame_index": frame_index,
                    "time_seconds": round(time_seconds, 9),
                    "expected_time_seconds": round(expected, 9),
                    "error_seconds": round(error, 9),
                }
            if error > tolerance_seconds:
                issues.append("timestamp_mismatch")

    issues = sorted(set(issues))
    frame_count = int(timing["frame_count"])
    duration_seconds = float(timing["duration_seconds"])
    state_log_every = int(timing["state_log_every"])
    source_sample_stride = int(timing.get("source_sample_stride") or 1)
    source_kind = str(timing.get("source_kind") or "timeline_log")
    sample_step_frames = _most_common_step(indices)
    full_frame_log = bool(
        not issues
        and source_sample_stride <= 1
        and frame_count > 0
        and len(indices) == frame_count
        and sample_step_frames in {0, 1}
        and (not indices or (indices[0] == 0 and indices[-1] == frame_count - 1))
    )
    if not issues and full_frame_log:
        timeline_mode = "full_frame_exact"
    elif not issues and indices and source_kind == "profile_frame_summaries":
        timeline_mode = "sampled_profile_exact"
    elif not issues and indices:
        timeline_mode = "sampled_exact"
    else:
        timeline_mode = "invalid"

    if fps > 0.0 and duration_seconds > 0.0 and frame_count > 0:
        expected_frame_count = int(round(duration_seconds * fps))
        if abs(expected_frame_count - frame_count) > 1:
            issues.append("duration_frame_count_mismatch")
            timeline_mode = "invalid"

    status = "pass" if not issues else "fail"
    return {
        "schema_version": "truevision_timeline_audit_v1",
        "manifest_path": str(manifest_path),
        "log_path": str(log_path),
        "status": status,
        "timeline_mode": timeline_mode,
        "issues": issues,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "manifest_frame_count": frame_count,
        "logged_records": len(indices),
        "state_log_every": state_log_every,
        "source_sample_stride": source_sample_stride,
        "source_kind": source_kind,
        "sample_step_frames": sample_step_frames,
        "full_frame_log": full_frame_log,
        "usable_for_frame_exact_tooling": bool(status == "pass" and full_frame_log),
        "usable_for_sampled_tooling": bool(status == "pass" and indices),
        "max_time_error_seconds": round(max_error, 9),
        "worst_record": worst_record,
        "record_formats": sorted(set(formats)),
        "law": "Frame index and FPS are the clock. Wall time is performance, not timeline truth.",
    }


def audit_many(manifest_paths: list[str | Path]) -> dict[str, Any]:
    audits = [audit_timeline_manifest(path) for path in manifest_paths]
    return {
        "schema_version": "truevision_timeline_audit_batch_v1",
        "count": len(audits),
        "pass_count": sum(1 for audit in audits if audit.get("status") == "pass"),
        "fail_count": sum(1 for audit in audits if audit.get("status") != "pass"),
        "frame_exact_count": sum(1 for audit in audits if audit.get("usable_for_frame_exact_tooling")),
        "sampled_exact_count": sum(1 for audit in audits if str(audit.get("timeline_mode", "")).startswith("sampled")),
        "audits": audits,
    }
