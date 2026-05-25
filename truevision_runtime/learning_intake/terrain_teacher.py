from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now


DEFAULT_TERRAIN_SEARCHES: dict[str, list[str]] = {
    "ocean_cliffs": [
        "ocean cliffs drone footage 1 hour",
        "coastal cliffs drone 4k long video",
        "sea cave ocean cliff drone footage",
        "cliff edge ocean waves cinematic footage",
        "coastal erosion cliffs documentary 1 hour",
        "how ocean cliffs form documentary",
    ],
    "canyons": [
        "canyon drone footage 1 hour",
        "grand canyon drone footage long",
        "desert canyon cinematic drone footage",
        "canyon formation documentary 1 hour",
        "geology of cliffs canyons volcanoes documentary",
    ],
    "volcanoes": [
        "volcano eruption documentary 1 hour",
        "lava field drone footage 4k",
        "volcanic ash cloud time lapse",
        "volcano landscape formation documentary",
    ],
    "stormy_ocean": [
        "storm ocean cliffs drone footage",
        "stormy ocean horizon long footage",
        "ocean waves cliff storm 4k long video",
    ],
    "mountain_ridge_fog": [
        "mountain ridge fog drone footage",
        "mountain ridges abyss drops fog layers long footage",
        "fog over cliffs drone footage 1 hour",
    ],
}

DEFAULT_TERRAIN_SEARCH_QUERIES = [
    query for queries in DEFAULT_TERRAIN_SEARCHES.values() for query in queries
]

TERRAIN_RULE_FILES = {
    "ocean_cliffs": "ocean_cliff_rules.jsonl",
    "canyons": "canyon_depth_rules.jsonl",
    "volcanoes": "volcano_glow_rules.jsonl",
    "stormy_ocean": "storm_ocean_rules.jsonl",
    "mountain_ridge_fog": "mountain_fog_depth_rules.jsonl",
    "fog_smoke_ash": "fog_atmosphere_rules.jsonl",
}

TERRAIN_EXTRACT_FIELDS = [
    "horizon_behavior",
    "foreground_midground_background_separation",
    "scale_cues",
    "texture_behavior",
    "atmosphere_behavior",
    "light_source_direction",
    "occlusion_patterns",
    "terrain_edge_shapes",
    "depth_cues",
    "renderer_parameter_suggestions",
]

TERRAIN_QA_METRICS = [
    "subject_readability",
    "ground_plane_visibility",
    "edge_visibility",
    "foreground_midground_background_separation",
    "parallax_score",
    "effect_occlusion_ratio",
    "terrain_realism_score",
    "chaos_budget_actual",
]

DEFAULT_TERRAIN_RENDERER_TARGET = {
    "scene_mode": "edge_nightmare_world",
    "shot_type": "wide_edge_intro",
    "duration_seconds": 12,
    "render_scope": "terrain_realism_proof_only",
    "full_song_render_allowed": False,
}

DEFAULT_MAX_TOTAL_CACHE_BYTES = 10 * 1024 * 1024 * 1024
DEFAULT_MAX_ACTIVE_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_FRAME_SAMPLES_PER_VIDEO = 120
DEFAULT_MAX_KEEP_FRAMES_AFTER_JOB = 12


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _ensure_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def _bytes_under(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _transient_root(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if resolved_candidate == resolved_root or resolved_root not in resolved_candidate.parents:
        raise ValueError(f"refusing cleanup outside terrain teacher root: {candidate}")
    if resolved_candidate.name not in {"active_job", "cache"}:
        raise ValueError(f"refusing cleanup of durable terrain teacher area: {candidate}")
    return resolved_candidate


def _search_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 1
    for source_class, queries in DEFAULT_TERRAIN_SEARCHES.items():
        for query in queries:
            rows.append(
                {
                    "query_id": f"terrain_teacher_seed_{index:02d}",
                    "source_class": source_class,
                    "query": query,
                    "preferred_duration_minutes": [30, 90],
                    "extract_goal": "physical_scene_rules_before_cinematography",
                    "must_extract": TERRAIN_EXTRACT_FIELDS,
                    "reject_if": [
                        "gear_review_only",
                        "abstract_visualizer",
                        "short_montage_bait",
                        "non_geography_source",
                    ],
                }
            )
            index += 1
    return rows


def initialize_terrain_teacher_workspace(
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

    _write_jsonl(root / "queue" / "search_queries.jsonl", _search_rows())
    _ensure_jsonl(root / "queue" / "video_candidates.jsonl")
    _ensure_jsonl(root / "queue" / "approved_sources.jsonl")
    _ensure_jsonl(root / "learned" / "human_ratings.jsonl")
    _ensure_jsonl(root / "learned" / "negative_examples.jsonl")
    for file_name in TERRAIN_RULE_FILES.values():
        _ensure_jsonl(root / "learned" / file_name)

    config = {
        "schema_version": "truevision_terrain_teacher_config_v1",
        "source_classes": list(DEFAULT_TERRAIN_SEARCHES),
        "max_total_cache_bytes": max_total_cache_bytes,
        "max_active_video_bytes": max_active_video_bytes,
        "max_frame_samples_per_video": DEFAULT_MAX_FRAME_SAMPLES_PER_VIDEO,
        "max_keep_frames_after_job": DEFAULT_MAX_KEEP_FRAMES_AFTER_JOB,
        "delete_raw_video_after_analysis": True,
        "delete_audio_after_transcript": True,
        "keep_transcripts": True,
        "keep_rules": True,
        "keep_qa_reports": True,
        "first_renderer_target": DEFAULT_TERRAIN_RENDERER_TARGET,
        "future_logger_lanes": [
            "raytracing_alternative_capture_logger",
            "pathtracing_alternative_learn_logger",
            "pathtracing_alternative_transform_logger",
            "arc_learning_transform_logger",
        ],
    }
    (root / "workspace_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "schema_version": "truevision_terrain_teacher_workspace_v1",
        "created_at_utc": utc_now(),
        "root": str(root),
        "directories": [str(directory) for directory in directories],
        "search_query_count": len(DEFAULT_TERRAIN_SEARCH_QUERIES),
        "learning_order": [
            "oceans_and_cliffs",
            "canyons_and_depth",
            "volcanoes_and_glow",
            "stormy_ocean_horizon",
            "mountain_ridge_fog_layers",
            "cinematography_after_realism",
        ],
        "retention": {
            "learn_physical_rules_not_videos": True,
            "flush_raw_media_after_job": True,
            "keep_compact_rules_reports": True,
        },
        "boundary": {
            "general_internet_scraper": False,
            "random_self_learning_vacuum": False,
            "auto_promote_rules": False,
            "human_director_required": True,
            "no_full_song_until_terrain_depth_proof_passes": True,
        },
    }
    manifest["workspace_hash"] = stable_hash(manifest)
    (root / "workspace_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_terrain_extraction_contract(source_class: str) -> dict[str, Any]:
    normalized = source_class if source_class in TERRAIN_RULE_FILES else "ocean_cliffs"
    return {
        "schema_version": "truevision_terrain_extraction_contract_v1",
        "source_class": normalized,
        "learning_goal": "physical_scene_rules_before_cinematography",
        "extract_fields": TERRAIN_EXTRACT_FIELDS,
        "first_renderer_target": DEFAULT_TERRAIN_RENDERER_TARGET,
        "durable_outputs": [
            TERRAIN_RULE_FILES[normalized],
            "lesson_notes.json",
            "terrain_qa_report.json",
            "human_review_packet.json",
        ],
        "retention": {
            "keep_teacher_video": False,
            "keep_sparse_frames_after_review": False,
            "keep_rules": True,
        },
    }


def rank_terrain_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terrain_terms = {
        "ocean",
        "cliff",
        "cliffs",
        "coastal",
        "sea cave",
        "waves",
        "horizon",
        "canyon",
        "ravine",
        "desert",
        "volcano",
        "lava",
        "ash cloud",
        "mountain ridge",
        "fog",
        "drone footage",
        "documentary",
        "erosion",
        "geology",
        "rock texture",
    }
    reject_terms = {
        "abstract visualizer",
        "gear review",
        "camera settings",
        "lens review",
        "unboxing",
        "wallpaper",
        "ai generated",
    }
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        text = f"{candidate.get('title', '')} {candidate.get('description', '')}".lower()
        duration = float(candidate.get("duration_seconds") or 0.0)
        score = 0.0
        reasons: list[str] = []
        if 30 * 60 <= duration <= 90 * 60:
            score += 0.40
            reasons.append("duration_30_to_90_minutes")
        elif 15 * 60 <= duration < 30 * 60 or 90 * 60 < duration <= 120 * 60:
            score += 0.12
            reasons.append("near_preferred_duration")
        else:
            score -= 0.14
            reasons.append("outside_preferred_duration")
        if candidate.get("has_transcript"):
            score += 0.20
            reasons.append("transcript_available")
        matched_terms = sorted(term for term in terrain_terms if term in text)
        if matched_terms:
            score += min(0.36, 0.06 * len(matched_terms))
            reasons.append("terrain_terms:" + ",".join(matched_terms[:6]))
        matched_rejects = sorted(term for term in reject_terms if term in text)
        if matched_rejects:
            score -= 0.34
            reasons.append("reject_terms:" + ",".join(matched_rejects[:5]))
        ranked_entry = dict(candidate)
        ranked_entry["score"] = round(score, 6)
        ranked_entry["score_reasons"] = reasons
        ranked.append(ranked_entry)
    ranked.sort(key=lambda item: (item["score"], float(item.get("duration_seconds") or 0.0)), reverse=True)
    return ranked


def terrain_disk_guard_report(
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
        "schema_version": "truevision_terrain_teacher_disk_guard_v1",
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


def cleanup_terrain_teacher_workspace(root: str | Path, *, dry_run: bool = True) -> dict[str, Any]:
    root = Path(root)
    delete_roots = [_transient_root(root, root / "active_job"), _transient_root(root, root / "cache")]
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
        "schema_version": "truevision_terrain_teacher_cleanup_receipt_v1",
        "root": str(root),
        "dry_run": dry_run,
        "planned_file_count": len(planned_files),
        "planned_bytes": planned_bytes,
        "deleted_bytes": deleted_bytes,
        "preserved_roots": [str(root / "queue"), str(root / "learned")],
    }


def build_terrain_human_review_packet(
    *,
    source_meta: dict[str, Any],
    physical_rules: list[str],
    proposed_renderer_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    packet = {
        "schema_version": "truevision_terrain_teacher_human_review_packet_v1",
        "created_at_utc": utc_now(),
        "source_meta": source_meta,
        "physical_rules": physical_rules,
        "proposed_renderer_rules": proposed_renderer_rules,
        "renderer_target": DEFAULT_TERRAIN_RENDERER_TARGET,
        "qa_metrics": TERRAIN_QA_METRICS,
        "human_rating_fields": [
            "realism",
            "ground_plane",
            "depth",
            "scale",
            "not_finger_painting",
            "promote_rule",
        ],
        "boundary": {
            "auto_promote_rules": False,
            "human_director_required": True,
            "full_song_render_allowed": False,
            "proof_shot_before_full_render": True,
        },
    }
    packet["review_packet_hash"] = stable_hash({k: v for k, v in packet.items() if k != "review_packet_hash"})
    return packet


def promote_terrain_rule(
    root: str | Path,
    rule: dict[str, Any],
    *,
    human_approved: bool,
    human_rating: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not human_approved:
        raise ValueError("terrain rules require human approval before promotion")
    root = Path(root)
    learned = root / "learned"
    learned.mkdir(parents=True, exist_ok=True)
    source_class = str(rule.get("source_class") or "ocean_cliffs")
    target_file = learned / TERRAIN_RULE_FILES.get(source_class, "ocean_cliff_rules.jsonl")
    promoted = {
        **rule,
        "schema_version": "truevision_terrain_rule_v1",
        "promoted_at_utc": utc_now(),
        "human_approved": True,
        "human_rating": human_rating or {},
        "boundary": {
            "learn_physical_rules_not_videos": True,
            "raw_teacher_media_retained": False,
            "cinematography_after_realism": True,
        },
    }
    promoted["rule_hash"] = stable_hash(promoted)
    with target_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(promoted, sort_keys=True) + "\n")
    if human_rating:
        with (learned / "human_ratings.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "written_at_utc": utc_now(),
                        "rule_id": promoted.get("rule_id"),
                        "rule_hash": promoted["rule_hash"],
                        "human_rating": human_rating,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    receipt = {
        "schema_version": "truevision_terrain_rule_promotion_receipt_v1",
        "status": "promoted",
        "rule_id": promoted.get("rule_id"),
        "rule_hash": promoted["rule_hash"],
        "target_file": str(target_file),
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    return receipt
