from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now


DEFAULT_SEARCH_QUERIES = [
    "cinematography blocking camera movement one hour",
    "cinematography shot composition masterclass 1 hour",
    "camera movement film language full course",
    "music video cinematography breakdown long form",
    "low budget cinematic lighting masterclass",
    "foreground midground background cinematography tutorial",
    "visual storytelling cinematography lecture",
    "film blocking scene geography masterclass",
    "parallax depth cinematography camera movement",
    "music video director treatment cinematography breakdown",
]

DEFAULT_QA_METRICS = [
    "subject_readability",
    "ground_plane_visibility",
    "edge_visibility",
    "foreground_midground_background_separation",
    "parallax_score",
    "effect_occlusion_ratio",
    "chaos_budget_actual",
]

DEFAULT_RULE_TARGET = {
    "scene_mode": "edge_nightmare_world",
    "shot_type": "wide_edge_intro",
    "duration_seconds": 12,
    "render_scope": "proof_shot_only",
}

DEFAULT_MAX_TOTAL_CACHE_BYTES = 10 * 1024 * 1024 * 1024
DEFAULT_MAX_ACTIVE_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_FRAME_SAMPLES_PER_VIDEO = 120
DEFAULT_MAX_KEEP_FRAMES_AFTER_JOB = 12


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _ensure_empty_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def _bytes_under(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _safe_transient_root(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if resolved_candidate == resolved_root or resolved_root not in resolved_candidate.parents:
        raise ValueError(f"refusing cleanup outside cinema teacher root: {candidate}")
    if resolved_candidate.name not in {"active_job", "cache"}:
        raise ValueError(f"refusing cleanup of durable cinema teacher area: {candidate}")
    return resolved_candidate


def _seed_search_rows() -> list[dict[str, Any]]:
    return [
        {
            "query_id": f"cinema_teacher_seed_{index:02d}",
            "query": query,
            "preferred_duration_minutes": [40, 90],
            "must_teach_one_of": [
                "shot_composition",
                "camera_movement",
                "lighting",
                "blocking",
                "transitions",
                "scene_geography",
                "music_video_language",
                "depth_parallax",
            ],
            "reject_if": ["gear_review_only", "short_montage_bait", "account_required", "non_educational"],
        }
        for index, query in enumerate(DEFAULT_SEARCH_QUERIES, start=1)
    ]


def initialize_cinema_teacher_workspace(
    root: str | Path,
    *,
    max_total_cache_bytes: int = DEFAULT_MAX_TOTAL_CACHE_BYTES,
    max_active_video_bytes: int = DEFAULT_MAX_ACTIVE_VIDEO_BYTES,
) -> dict[str, Any]:
    root = Path(root)
    directories = [
        root / "queue",
        root / "active_job",
        root / "active_job" / "sampled_frames",
        root / "active_job" / "render_tests",
        root / "learned",
        root / "cache",
        root / "cache" / "temp_video_or_audio",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    _jsonl_write(root / "queue" / "search_queries.jsonl", _seed_search_rows())
    _ensure_empty_jsonl(root / "queue" / "video_candidates.jsonl")
    _ensure_empty_jsonl(root / "queue" / "approved_videos.jsonl")
    _ensure_empty_jsonl(root / "learned" / "cinematography_rules.jsonl")
    _ensure_empty_jsonl(root / "learned" / "negative_examples.jsonl")
    _ensure_empty_jsonl(root / "learned" / "human_ratings.jsonl")

    shot_presets = {
        "schema_version": "truevision_cinema_teacher_shot_grammar_presets_v1",
        "default_renderer_target": DEFAULT_RULE_TARGET,
        "promotion": {
            "human_approval_required": True,
            "auto_promote_rules": False,
        },
        "first_proof_gate": {
            "scene_mode": "edge_nightmare_world",
            "shot_type": "wide_edge_intro",
            "required_before_full_song": True,
            "qa_metrics": DEFAULT_QA_METRICS,
        },
    }
    (root / "learned" / "shot_grammar_presets.json").write_text(
        json.dumps(shot_presets, indent=2, sort_keys=True), encoding="utf-8"
    )

    config = {
        "schema_version": "truevision_cinema_teacher_config_v1",
        "max_total_cache_bytes": max_total_cache_bytes,
        "max_active_video_bytes": max_active_video_bytes,
        "max_frame_samples_per_video": DEFAULT_MAX_FRAME_SAMPLES_PER_VIDEO,
        "max_keep_frames_after_job": DEFAULT_MAX_KEEP_FRAMES_AFTER_JOB,
        "delete_raw_video_after_analysis": True,
        "delete_audio_after_transcript": True,
        "keep_transcripts": True,
        "keep_rules": True,
        "keep_qa_reports": True,
        "future_logger_lanes": [
            "raytracing_alternative_capture_logger",
            "pathtracing_alternative_transform_logger",
            "arc_learning_transform_logger",
        ],
    }
    (root / "workspace_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "schema_version": "truevision_cinema_teacher_workspace_v1",
        "created_at_utc": utc_now(),
        "root": str(root),
        "directories": [str(directory) for directory in directories],
        "search_query_count": len(DEFAULT_SEARCH_QUERIES),
        "retention": {
            "learn_rules_not_videos": True,
            "flush_raw_media_after_job": True,
            "keep_compact_lessons_rules_reports": True,
        },
        "boundary": {
            "general_internet_scraper": False,
            "random_self_learning_vacuum": False,
            "auto_promote_rules": False,
            "human_director_required": True,
            "no_full_song_until_12s_depth_proof_passes": True,
        },
        "workspace_hash": "",
    }
    manifest["workspace_hash"] = stable_hash({k: v for k, v in manifest.items() if k != "workspace_hash"})
    (root / "workspace_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_disk_guard_report(
    root: str | Path,
    *,
    max_total_cache_bytes: int = DEFAULT_MAX_TOTAL_CACHE_BYTES,
    max_active_video_bytes: int = DEFAULT_MAX_ACTIVE_VIDEO_BYTES,
) -> dict[str, Any]:
    root = Path(root)
    cache_bytes = _bytes_under(root / "cache")
    active_bytes = _bytes_under(root / "active_job")
    refusal_reasons: list[str] = []
    if cache_bytes > max_total_cache_bytes:
        refusal_reasons.append("cache_over_cap")
    if active_bytes > max_active_video_bytes:
        refusal_reasons.append("active_job_over_cap")
    return {
        "schema_version": "truevision_cinema_teacher_disk_guard_v1",
        "root": str(root),
        "cache_bytes": cache_bytes,
        "active_job_bytes": active_bytes,
        "max_total_cache_bytes": max_total_cache_bytes,
        "max_active_video_bytes": max_active_video_bytes,
        "can_start_new_job": not refusal_reasons,
        "refusal_reasons": refusal_reasons,
        "cleanup_plan": {
            "dry_run_first": True,
            "delete_roots": [str(root / "active_job"), str(root / "cache")],
            "preserve_roots": [str(root / "queue"), str(root / "learned")],
        },
    }


def cleanup_cinema_teacher_workspace(root: str | Path, *, dry_run: bool = True) -> dict[str, Any]:
    root = Path(root)
    delete_roots = [_safe_transient_root(root, root / "active_job"), _safe_transient_root(root, root / "cache")]
    planned_bytes = sum(_bytes_under(path) for path in delete_roots)
    planned_files = [
        str(item)
        for delete_root in delete_roots
        if delete_root.exists()
        for item in delete_root.rglob("*")
        if item.is_file()
    ]
    deleted_bytes = 0
    if not dry_run:
        deleted_bytes = planned_bytes
        for delete_root in delete_roots:
            if delete_root.exists():
                shutil.rmtree(delete_root)
        (root / "active_job" / "sampled_frames").mkdir(parents=True, exist_ok=True)
        (root / "active_job" / "render_tests").mkdir(parents=True, exist_ok=True)
        (root / "cache" / "temp_video_or_audio").mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": "truevision_cinema_teacher_cleanup_receipt_v1",
        "root": str(root),
        "dry_run": dry_run,
        "planned_delete_roots": [str(path) for path in delete_roots],
        "planned_file_count": len(planned_files),
        "planned_bytes": planned_bytes,
        "deleted_bytes": deleted_bytes,
        "preserved_roots": [str(root / "queue"), str(root / "learned")],
    }


def rank_video_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    lesson_terms = {
        "blocking",
        "camera movement",
        "composition",
        "foreground",
        "midground",
        "background",
        "scene geography",
        "visual storytelling",
        "parallax",
        "lighting",
        "music video",
        "shot",
    }
    reject_terms = {"gear review", "lens review", "camera body", "unboxing", "shorts", "montage"}
    for candidate in candidates:
        text = f"{candidate.get('title', '')} {candidate.get('description', '')}".lower()
        duration = float(candidate.get("duration_seconds") or 0.0)
        score = 0.0
        reasons: list[str] = []
        if 40 * 60 <= duration <= 90 * 60:
            score += 0.40
            reasons.append("duration_40_to_90_minutes")
        elif 20 * 60 <= duration < 40 * 60 or 90 * 60 < duration <= 120 * 60:
            score += 0.14
            reasons.append("near_preferred_duration")
        if candidate.get("has_transcript"):
            score += 0.25
            reasons.append("transcript_available")
        matched_terms = sorted(term for term in lesson_terms if term in text)
        if matched_terms:
            score += min(0.28, 0.07 * len(matched_terms))
            reasons.append("lesson_terms:" + ",".join(matched_terms[:5]))
        matched_rejects = sorted(term for term in reject_terms if term in text)
        if matched_rejects:
            score -= 0.30
            reasons.append("reject_terms:" + ",".join(matched_rejects[:5]))
        if duration <= 10 * 60:
            score -= 0.20
            reasons.append("too_short_for_primary_teacher")
        entry = dict(candidate)
        entry["score"] = round(score, 6)
        entry["score_reasons"] = reasons
        ranked.append(entry)
    ranked.sort(key=lambda item: (item["score"], float(item.get("duration_seconds") or 0.0)), reverse=True)
    return ranked


def build_human_review_packet(
    *,
    source_meta: dict[str, Any],
    lesson_notes: list[str],
    proposed_rules: list[dict[str, Any]],
    renderer_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packet = {
        "schema_version": "truevision_cinema_teacher_human_review_packet_v1",
        "created_at_utc": utc_now(),
        "source_meta": source_meta,
        "lesson_notes": lesson_notes,
        "proposed_rules": proposed_rules,
        "renderer_target": renderer_target or DEFAULT_RULE_TARGET,
        "qa_metrics": DEFAULT_QA_METRICS,
        "human_rating_fields": [
            "subject_readability",
            "depth",
            "emotion",
            "camera_language",
            "not_soup_score",
            "promote_rule",
        ],
        "boundary": {
            "auto_promote_rules": False,
            "human_director_required": True,
            "proof_shot_before_full_render": True,
        },
    }
    packet["review_packet_hash"] = stable_hash({k: v for k, v in packet.items() if k != "review_packet_hash"})
    return packet


def promote_cinematography_rule(
    root: str | Path,
    rule: dict[str, Any],
    *,
    human_approved: bool,
    human_rating: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not human_approved:
        raise ValueError("cinematography rules require human approval before promotion")
    root = Path(root)
    learned = root / "learned"
    learned.mkdir(parents=True, exist_ok=True)
    promoted = {
        **rule,
        "schema_version": "truevision_cinematography_rule_v1",
        "promoted_at_utc": utc_now(),
        "human_approved": True,
        "human_rating": human_rating or {},
        "source_boundary": {
            "learn_rules_not_videos": True,
            "raw_teacher_media_retained": False,
        },
    }
    promoted["rule_hash"] = stable_hash(promoted)
    with (learned / "cinematography_rules.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(promoted, sort_keys=True) + "\n")
    if human_rating:
        rating_record = {
            "written_at_utc": utc_now(),
            "rule_id": promoted.get("rule_id"),
            "rule_hash": promoted["rule_hash"],
            "human_rating": human_rating,
        }
        with (learned / "human_ratings.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rating_record, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "truevision_cinema_teacher_rule_promotion_receipt_v1",
        "status": "promoted",
        "rule_id": promoted.get("rule_id"),
        "rule_hash": promoted["rule_hash"],
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    return receipt
