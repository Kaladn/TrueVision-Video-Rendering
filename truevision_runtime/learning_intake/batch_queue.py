from __future__ import annotations

from typing import Any

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now
from truevision_runtime.learning_intake.source_surface import (
    SourceSurfacePolicyError,
    build_source_surface_multi_sample_plan,
    canonicalize_approved_source_url,
)


ELEMENT_CATEGORY_MAP = {
    "fire": "fire_flame_licks",
    "smoke": "smoke_turbulent_columns",
    "lightning": "lightning_arc_bloom",
    "rain": "rain_glass_fall",
    "particles": "ember_particle_field",
    "dust": "dust_particle_drift",
    "neons": "neon_glow_pulse",
    "het distortion/shimmer": "heat_distortion_shimmer",
    "heat distortion/shimmer": "heat_distortion_shimmer",
    "louds/storms": "clouds_storms",
    "clouds/storms": "clouds_storms",
    "water/river/flow energy": "water_river_flow_energy",
    "crashing waves": "crashing_wave_motion",
    "silhouettes / human motion": "silhouette_human_motion",
    "abstracts": "abstract_energy_motion",
    "camera movement": "camera_motion_pan_zoom",
    "cityscapes": "cityscape_light_motion",
    "crowd motion": "crowd_motion",
}


def _safe_id(value: str, fallback: str = "element") -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or fallback


def _category_to_element_id(category: str) -> str:
    normalized = " ".join(category.strip().lower().split())
    return ELEMENT_CATEGORY_MAP.get(normalized, _safe_id(normalized, fallback="uncategorized_element"))


def _is_youtube_url(line: str) -> bool:
    lowered = line.lower()
    return lowered.startswith("https://www.youtube.com/") or lowered.startswith("https://youtu.be/")


def _extract_url_token(line: str) -> str:
    return line.split()[0].strip()


def parse_approved_youtube_sources(text: str) -> list[dict[str, Any]]:
    current_category = "Uncategorized"
    entries: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if _is_youtube_url(line):
            url = _extract_url_token(line)
            try:
                canonical = canonicalize_approved_source_url(url)
            except SourceSurfacePolicyError:
                continue
            source_order = len(entries) + 1
            entries.append(
                {
                    "source_order": source_order,
                    "line_number": line_number,
                    "category": current_category,
                    "element_id": _category_to_element_id(current_category),
                    "source_url": url,
                    "address_bar_url": canonical["address_bar_url"],
                    "video_id": canonical["video_id"],
                    "removed_query_keys": canonical["removed_query_keys"],
                }
            )
            continue
        current_category = line.strip()
    return entries


def _metadata_for_entry(entry: dict[str, Any], metadata_by_video_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    metadata = metadata_by_video_id.get(str(entry["video_id"])) or metadata_by_video_id.get(str(entry["address_bar_url"]))
    if not metadata:
        return None
    duration = metadata.get("duration_seconds") or metadata.get("video_duration_seconds")
    try:
        parsed_duration = float(duration)
    except (TypeError, ValueError):
        return None
    if parsed_duration <= 0:
        return None
    return {
        "video_title": str(metadata.get("video_title") or metadata.get("title") or entry["video_id"]),
        "duration_seconds": parsed_duration,
    }


def build_batch_queue(
    entries: list[dict[str, Any]],
    *,
    metadata_by_video_id: dict[str, dict[str, Any]],
    player_region: list[int],
    run_id: str,
    sample_seconds: float = 12.0,
    sample_count: int = 4,
    large_video_threshold_seconds: float = 600.0,
    fps: float = 15.0,
    resolution: list[int] | tuple[int, int] = (1280, 720),
    grid: list[int] | tuple[int, int] = (320, 180),
    output_root: str = "E:\\TruEVision Generation\\library\\youtube_learning_intake_pilot\\captures",
    native_capture_exe: str = "D:\\TrueVision_Generation_Lab\\native\\truevision_capture_rs\\target\\release\\truevision_capture_rs.exe",
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry in entries:
        metadata = _metadata_for_entry(entry, metadata_by_video_id)
        if metadata is None:
            skipped.append(
                {
                    "source_order": entry["source_order"],
                    "category": entry["category"],
                    "source_url": entry["source_url"],
                    "video_id": entry["video_id"],
                    "reason": "duration_not_detected",
                }
            )
            continue
        source_run_id = f"{run_id}_{entry['source_order']:03d}_{entry['element_id']}_{entry['video_id']}"
        plan = build_source_surface_multi_sample_plan(
            element_id=entry["element_id"],
            source_url=entry["source_url"],
            video_title=metadata["video_title"],
            video_duration_seconds=metadata["duration_seconds"],
            player_region=player_region,
            run_id=source_run_id,
            sample_seconds=sample_seconds,
            sample_count=sample_count,
            large_video_threshold_seconds=large_video_threshold_seconds,
            fps=fps,
            resolution=resolution,
            grid=grid,
            output_root=output_root,
            native_capture_exe=native_capture_exe,
        )
        sources.append(
            {
                "source_order": entry["source_order"],
                "category": entry["category"],
                "element_id": entry["element_id"],
                "video_id": entry["video_id"],
                "source_url": entry["source_url"],
                "address_bar_url": plan["source"]["address_bar_url"],
                "video_title": metadata["video_title"],
                "duration_seconds": metadata["duration_seconds"],
                "run_id": source_run_id,
                "sample_count": plan["sampling"]["sample_count"],
                "samples": plan["samples"],
                "plan_hash": plan["plan_hash"],
            }
        )
    queue = {
        "schema_version": "truevision_learning_intake_batch_queue_v1",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "source_count": len(sources),
        "skipped_count": len(skipped),
        "sample_count": sum(int(source["sample_count"]) for source in sources),
        "sources": sources,
        "skipped_sources": skipped,
        "retention": {
            "keep_teacher_chunks": False,
            "purge_teacher_chunks_after_profile": True,
            "durable_outputs": [
                "element_creation_profile",
                "source_surface_video_state_receipt",
                "profile_manifest",
                "purge_report",
                "batch_summary",
            ],
        },
        "boundary": {
            "raw_download": False,
            "youtube_search_navigation": False,
            "address_bar_navigation_required": True,
            "profile_first_verify_second_purge_third": True,
        },
    }
    queue["queue_hash"] = stable_hash(queue)
    return queue
