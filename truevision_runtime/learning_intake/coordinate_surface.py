from __future__ import annotations

from pathlib import Path
from typing import Any

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now
from truevision_runtime.learning_intake.source_surface import canonicalize_approved_source_url


REQUIRED_POINTS = ("address_bar", "video_play")


def _point(value: Any, name: str, *, screen_size: list[int]) -> list[int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{name} must be [x, y]")
    try:
        x = int(value[0])
        y = int(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer coordinates") from exc
    if x < 0 or y < 0 or x >= screen_size[0] or y >= screen_size[1]:
        raise ValueError(f"{name} is outside screen bounds")
    return [x, y]


def _region(value: Any, *, screen_size: list[int]) -> list[int]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("capture_region must be [x, y, width, height]")
    try:
        x, y, width, height = [int(part) for part in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("capture_region must contain integer values") from exc
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("capture_region must be positive and on-screen")
    if x + width > screen_size[0] or y + height > screen_size[1]:
        raise ValueError("capture_region exceeds screen bounds")
    return [x, y, width, height]


def validate_coordinate_map(coordinate_map: dict[str, Any]) -> dict[str, Any]:
    if coordinate_map.get("schema_version") != "truevision_coordinate_surface_map_v1":
        raise ValueError("coordinate map schema_version must be truevision_coordinate_surface_map_v1")
    raw_screen = coordinate_map.get("screen_size")
    if not isinstance(raw_screen, list | tuple) or len(raw_screen) != 2:
        raise ValueError("screen_size must be [width, height]")
    screen_size = [int(raw_screen[0]), int(raw_screen[1])]
    if screen_size[0] <= 0 or screen_size[1] <= 0:
        raise ValueError("screen_size must be positive")
    raw_points = coordinate_map.get("points")
    if not isinstance(raw_points, dict):
        raise ValueError("points must be a coordinate dictionary")
    points: dict[str, list[int]] = {}
    for point_name in REQUIRED_POINTS:
        if point_name not in raw_points:
            raise ValueError(f"missing required coordinate point: {point_name}")
        points[point_name] = _point(raw_points[point_name], point_name, screen_size=screen_size)
    for point_name, point_value in raw_points.items():
        if point_name in points:
            continue
        points[str(point_name)] = _point(point_value, str(point_name), screen_size=screen_size)
    return {
        "schema_version": "truevision_coordinate_surface_map_v1",
        "screen_size": screen_size,
        "points": points,
        "capture_region": _region(coordinate_map.get("capture_region"), screen_size=screen_size),
    }


def build_coordinate_intake_plan(
    *,
    run_id: str,
    source: dict[str, Any],
    sample: dict[str, Any],
    coordinate_map: dict[str, Any],
    output_root: Path,
    native_capture_exe: Path,
    fps: float,
    resolution: list[int],
    grid: list[int],
) -> dict[str, Any]:
    if not run_id:
        raise ValueError("run_id is required")
    validated_map = validate_coordinate_map(coordinate_map)
    canonical = canonicalize_approved_source_url(str(source["source_url"]))
    sample_url = str(sample.get("sample_navigation_url") or canonical["address_bar_url"])
    sample_run_id = str(sample["run_id"])
    duration = float(sample["duration_seconds"])
    capture_duration = duration + 1.0
    command = [
        str(native_capture_exe),
        "--duration",
        f"{capture_duration:.3f}",
        "--fps",
        f"{float(fps):g}",
        "--resolution",
        f"{int(resolution[0])}x{int(resolution[1])}",
        "--grid",
        f"{int(grid[0])}x{int(grid[1])}",
        "--region",
        ",".join(str(part) for part in validated_map["capture_region"]),
        "--output-root",
        str(output_root),
        "--run-id",
        sample_run_id,
        "--start-delay",
        "0",
        "--cell-chunk-frames",
        "300",
    ]
    return {
        "schema_version": "truevision_coordinate_intake_plan_v1",
        "run_id": run_id,
        "sample_run_id": sample_run_id,
        "source": {
            "source_order": source.get("source_order"),
            "category": source.get("category"),
            "element_id": source.get("element_id"),
            "approved_url": canonical["address_bar_url"],
            "sample_navigation_url": sample_url,
            "video_title": source.get("video_title"),
            "duration_seconds": float(source["duration_seconds"]),
        },
        "coordinate_map": validated_map,
        "timeline": [
            {"event": "capture_start", "at_seconds": 0.0},
            {"event": "paste_url", "at_seconds": 0.15, "point": "address_bar"},
            {"event": "play_click", "at_seconds": 0.50, "point": "video_play"},
            {"event": "capture_stop", "at_seconds": capture_duration},
            {"event": "profile_verify_purge", "at_seconds": capture_duration},
        ],
        "native_capture_command": command,
        "boundary": {
            "coordinate_driven": True,
            "youtube_search_navigation": False,
            "uses_current_operator_browser": True,
            "raw_download": False,
            "profile_first_verify_second_purge_third": True,
            "no_active_control_without_saved_coordinate": True,
        },
    }


def build_coordinate_intake_receipt(
    *,
    run_id: str,
    approved_url: str,
    sample_navigation_url: str,
    coordinate_map_id: str,
    coordinate_map_sha256: str,
    capture_region: list[int],
    visual_state_records: int,
    profile_created: bool,
    teacher_chunks_purged: bool,
    visual_motion_score: float,
    minimum_visual_motion_score: float = 0.001,
) -> dict[str, Any]:
    if not run_id:
        raise ValueError("run_id is required")
    approved = canonicalize_approved_source_url(approved_url)
    sample = canonicalize_approved_source_url(sample_navigation_url)
    if approved["video_id"] != sample["video_id"]:
        raise ValueError("sample_navigation_url video_id does not match approved_url")
    records = int(visual_state_records)
    motion = float(visual_motion_score)
    minimum = float(minimum_visual_motion_score)
    checks = {
        "coordinate_map_id": bool(str(coordinate_map_id).strip()),
        "coordinate_map_sha256": str(coordinate_map_sha256).strip().startswith("sha256:"),
        "video_id_match": approved["video_id"] == sample["video_id"],
        "visual_state_records": records > 0,
        "profile_created": bool(profile_created),
        "teacher_chunks_purged": bool(teacher_chunks_purged),
        "visual_temporal_motion": motion >= minimum,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"coordinate intake receipt rejected; failed checks: {', '.join(failed)}")
    receipt = {
        "schema_version": "truevision_coordinate_intake_receipt_v1",
        "tool": "coordinate_surface_intake",
        "created_at_utc": utc_now(),
        "status": "verified",
        "run_id": run_id,
        "approved_url": approved["address_bar_url"],
        "sample_navigation_url": sample_navigation_url,
        "video_id": approved["video_id"],
        "coordinate_map_id": str(coordinate_map_id).strip(),
        "coordinate_map_sha256": str(coordinate_map_sha256).strip(),
        "capture_region": [int(part) for part in capture_region],
        "visual_state_records": records,
        "visual_motion_score": round(motion, 6),
        "minimum_visual_motion_score": round(minimum, 6),
        "checks": checks,
        "boundary": {
            "uses_current_operator_browser": True,
            "new_browser_instance": False,
            "youtube_search_navigation": False,
            "raw_download": False,
            "source_time_proof": False,
            "profile_first_verify_second_purge_third": True,
            "coordinate_map_required_before_run": True,
        },
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    return receipt
