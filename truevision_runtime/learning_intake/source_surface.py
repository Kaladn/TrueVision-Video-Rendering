from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now


class SourceSurfacePolicyError(ValueError):
    """Raised when a source-surface capture plan violates the intake boundary."""


ALLOWED_DISPLAY_IDS = {
    "yt.display.page_url",
    "yt.display.search_query",
    "yt.display.player_region",
    "yt.display.title",
    "yt.display.channel",
    "yt.display.elapsed_time",
    "yt.display.duration",
    "yt.display.fullscreen_state",
}

ALLOWED_BUTTON_IDS = {
    "yt.button.play_pause",
    "yt.button.seek_to_start",
    "yt.button.fullscreen",
    "yt.button.settings_speed",
}

FORBIDDEN_BUTTON_IDS = {
    "yt.button.like",
    "yt.button.dislike",
    "yt.button.subscribe",
    "yt.button.comment",
    "yt.button.share",
    "yt.button.download",
    "yt.button.upload",
    "yt.button.notifications",
    "yt.button.account_menu",
    "yt.link.recommendation",
    "yt.link.external_ad",
    "yt.input.comment",
}


def _bool_value(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise SourceSurfacePolicyError(f"{name} must be true or false")


def _youtube_video_id_from_url(source_url: str) -> str:
    parsed = urlparse(source_url.strip())
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")
    if host.endswith("youtu.be") and path:
        return path.split("/")[0]
    if "youtube.com" in host:
        if path == "watch":
            values = parse_qs(parsed.query).get("v") or []
            if values and values[0].strip():
                return values[0].strip()
        if path.startswith("shorts/"):
            parts = path.split("/")
            if len(parts) >= 2 and parts[1].strip():
                return parts[1].strip()
    raise SourceSurfacePolicyError("approved source URL must be a YouTube watch, shorts, or youtu.be video URL")


def canonicalize_approved_source_url(source_url: str) -> dict[str, Any]:
    parsed = urlparse(source_url.strip())
    video_id = _youtube_video_id_from_url(source_url)
    query_keys = sorted(parse_qs(parsed.query).keys())
    removed = [key for key in query_keys if key != "v"]
    address_bar_url = f"https://www.youtube.com/watch?v={video_id}"
    return {
        "source_url": source_url.strip(),
        "address_bar_url": address_bar_url,
        "video_id": video_id,
        "removed_query_keys": removed,
        "navigation_method": "browser_address_bar",
    }


def _receipt_video_id_match(approved_url: str, resolved_url: str) -> bool:
    try:
        return _youtube_video_id_from_url(approved_url) == _youtube_video_id_from_url(resolved_url)
    except SourceSurfacePolicyError:
        return False


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SourceSurfacePolicyError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise SourceSurfacePolicyError(f"{name} must be positive")
    return parsed


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SourceSurfacePolicyError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise SourceSurfacePolicyError(f"{name} must be positive")
    return parsed


def _region(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise SourceSurfacePolicyError("player_region must be [left, top, width, height]")
    left = int(value[0])
    top = int(value[1])
    width = _positive_int(value[2], "player_region width")
    height = _positive_int(value[3], "player_region height")
    if left < 0 or top < 0:
        raise SourceSurfacePolicyError("player_region left/top must be non-negative")
    return [left, top, width, height]


def _pair(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SourceSurfacePolicyError(f"{name} must be [width, height]")
    return [_positive_int(value[0], f"{name} width"), _positive_int(value[1], f"{name} height")]


def _validate_surface_ids(display_ids: list[str], button_ids: list[str]) -> None:
    unknown_display = sorted(set(display_ids) - ALLOWED_DISPLAY_IDS)
    if unknown_display:
        raise SourceSurfacePolicyError(f"unknown display IDs: {', '.join(unknown_display)}")
    forbidden = sorted(set(button_ids) & FORBIDDEN_BUTTON_IDS)
    if forbidden:
        raise SourceSurfacePolicyError(f"forbidden button IDs: {', '.join(forbidden)}")
    unknown_button = sorted(set(button_ids) - ALLOWED_BUTTON_IDS)
    if unknown_button:
        raise SourceSurfacePolicyError(f"unknown button IDs: {', '.join(unknown_button)}")


def _native_capture_command(
    *,
    capture_exe: str,
    duration_seconds: float,
    fps: float,
    resolution: list[int],
    grid: list[int],
    region: list[int],
    output_root: str,
    run_id: str,
    chunk_frames: int,
) -> list[str]:
    return [
        capture_exe,
        "--duration",
        f"{duration_seconds:.3f}",
        "--fps",
        f"{fps:g}",
        "--resolution",
        f"{resolution[0]}x{resolution[1]}",
        "--grid",
        f"{grid[0]}x{grid[1]}",
        "--region",
        ",".join(str(item) for item in region),
        "--output-root",
        output_root,
        "--run-id",
        run_id,
        "--start-delay",
        "0",
        "--cell-chunk-frames",
        str(chunk_frames),
    ]


def _sample_navigation_url(address_bar_url: str, start_seconds: float) -> str:
    return f"{address_bar_url}&t={int(round(start_seconds))}s"


def _sample_windows(
    *,
    video_duration_seconds: float,
    sample_seconds: float,
    sample_count: int,
    large_video_threshold_seconds: float,
) -> list[dict[str, Any]]:
    duration = _positive_float(video_duration_seconds, "video_duration_seconds")
    sample = min(_positive_float(sample_seconds, "sample_seconds"), duration)
    count = _positive_int(sample_count, "sample_count") if duration >= large_video_threshold_seconds else 1
    count = max(1, min(count, int(max(1, duration // sample)) if sample > 0 else 1))
    if count == 1:
        return [
            {
                "sample_index": 0,
                "section": "section_1_of_1",
                "start_seconds": 0.0,
                "end_seconds": round(sample, 3),
                "duration_seconds": round(sample, 3),
            }
        ]
    section_seconds = duration / count
    windows: list[dict[str, Any]] = []
    for index in range(count):
        section_start = section_seconds * index
        section_end = section_seconds * (index + 1)
        center = (section_start + section_end) / 2.0
        start = max(0.0, min(duration - sample, center - sample / 2.0))
        end = min(duration, start + sample)
        windows.append(
            {
                "sample_index": index,
                "section": f"section_{index + 1}_of_{count}",
                "section_start_seconds": round(section_start, 3),
                "section_end_seconds": round(section_end, 3),
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
            }
        )
    return windows


def build_source_surface_multi_sample_plan(
    *,
    element_id: str,
    source_url: str,
    source_title: str = "",
    video_title: str = "",
    video_duration_seconds: float,
    player_region: list[int] | tuple[int, int, int, int],
    run_id: str | None = None,
    sample_seconds: float = 12.0,
    sample_count: int = 4,
    large_video_threshold_seconds: float = 600.0,
    fps: float = 15.0,
    resolution: list[int] | tuple[int, int] = (1280, 720),
    grid: list[int] | tuple[int, int] = (320, 180),
    pre_roll_seconds: float = 0.25,
    post_roll_seconds: float = 0.75,
    output_root: str = "E:\\TruEVision Generation\\library\\youtube_learning_intake_pilot\\captures",
    native_capture_exe: str = "D:\\TrueVision_Generation_Lab\\native\\truevision_capture_rs\\target\\release\\truevision_capture_rs.exe",
    chunk_frames: int = 300,
) -> dict[str, Any]:
    if not element_id:
        raise SourceSurfacePolicyError("element_id is required")
    if not source_url:
        raise SourceSurfacePolicyError("source_url is required")
    canonical = canonicalize_approved_source_url(source_url)
    duration = _positive_float(video_duration_seconds, "video_duration_seconds")
    region = _region(player_region)
    capture_resolution = _pair(resolution, "resolution")
    capture_grid = _pair(grid, "grid")
    if capture_resolution[0] % capture_grid[0] != 0 or capture_resolution[1] % capture_grid[1] != 0:
        raise SourceSurfacePolicyError("resolution must divide evenly by grid")
    windows = _sample_windows(
        video_duration_seconds=duration,
        sample_seconds=sample_seconds,
        sample_count=sample_count,
        large_video_threshold_seconds=large_video_threshold_seconds,
    )
    fps_value = _positive_float(fps, "fps")
    safe_run_id = run_id or f"{element_id}_multi_sample"
    samples: list[dict[str, Any]] = []
    for window in windows:
        sample_run_id = f"{safe_run_id}_sample_{int(window['sample_index']) + 1:02d}"
        capture_duration = round(pre_roll_seconds + float(window["duration_seconds"]) + post_roll_seconds, 3)
        samples.append(
            {
                **window,
                "run_id": sample_run_id,
                "sample_navigation_url": _sample_navigation_url(canonical["address_bar_url"], float(window["start_seconds"])),
                "capture": {
                    "duration_seconds": capture_duration,
                    "fps": fps_value,
                    "resolution": capture_resolution,
                    "grid": capture_grid,
                    "region": region,
                    "raw_frames_saved": False,
                    "audio": False,
                },
                "native_capture_command": _native_capture_command(
                    capture_exe=native_capture_exe,
                    duration_seconds=capture_duration,
                    fps=fps_value,
                    resolution=capture_resolution,
                    grid=capture_grid,
                    region=region,
                    output_root=output_root,
                    run_id=sample_run_id,
                    chunk_frames=chunk_frames,
                ),
                "required_closeout": [
                    "source_surface_video_state_receipt",
                    "element_creation_profile_from_capture",
                    "profile_hash_verified",
                    "teacher_chunks_purged",
                ],
            }
        )
    plan = {
        "schema_version": "truevision_source_surface_multi_sample_plan_v1",
        "surface_id": "youtube_source_surface_v1",
        "run_id": safe_run_id,
        "element_id": element_id,
        "source": {
            "kind": "youtube_watch_surface",
            "url": source_url,
            "address_bar_url": canonical["address_bar_url"],
            "video_id": canonical["video_id"],
            "removed_query_keys": canonical["removed_query_keys"],
            "title": source_title or video_title,
            "video_duration_seconds": round(duration, 3),
            "navigation_method": "browser_address_bar",
        },
        "sampling": {
            "mode": "four_section_sampling" if len(samples) == 4 else "single_window_sampling",
            "sample_count": len(samples),
            "requested_sample_count": int(sample_count),
            "sample_seconds": round(float(sample_seconds), 3),
            "large_video_threshold_seconds": round(float(large_video_threshold_seconds), 3),
            "section_rule": "centered window inside each equal-duration section",
        },
        "navigation_flow": [
            "focus_browser_address_bar",
            "paste_sample_navigation_url",
            "press_enter",
            "wait_for_video_page_load",
            "verify_resolved_url_title_duration",
            "capture_visual_state",
            "create_profile",
            "purge_teacher_chunks",
            "write_verified_video_state_receipt",
        ],
        "samples": samples,
        "boundary": {
            "youtube_search_navigation": False,
            "address_bar_navigation_required": True,
            "profile_each_sample_before_next": True,
            "purge_teacher_chunks_each_sample": True,
            "completed_macro_is_not_completed_capture": True,
            "raw_download": False,
        },
    }
    plan["plan_hash"] = stable_hash(plan)
    return plan


def build_source_surface_capture_plan(
    *,
    element_id: str,
    source_url: str,
    source_title: str = "",
    video_duration_seconds: float,
    player_region: list[int] | tuple[int, int, int, int],
    run_id: str | None = None,
    fps: float = 15.0,
    resolution: list[int] | tuple[int, int] = (1280, 720),
    grid: list[int] | tuple[int, int] = (320, 180),
    pre_roll_seconds: float = 0.25,
    post_roll_seconds: float = 0.75,
    display_ids: list[str] | None = None,
    button_ids: list[str] | None = None,
    output_root: str = "E:\\TruEVision Generation\\library\\youtube_learning_intake_pilot\\captures",
    native_capture_exe: str = "D:\\TrueVision_Generation_Lab\\native\\truevision_capture_rs\\target\\release\\truevision_capture_rs.exe",
    chunk_frames: int = 300,
) -> dict[str, Any]:
    if not element_id:
        raise SourceSurfacePolicyError("element_id is required")
    if not source_url:
        raise SourceSurfacePolicyError("source_url is required")
    canonical_source = canonicalize_approved_source_url(source_url)
    source_duration = _positive_float(video_duration_seconds, "video_duration_seconds")
    pre_roll = float(pre_roll_seconds)
    post_roll = float(post_roll_seconds)
    if pre_roll < 0 or post_roll < 0:
        raise SourceSurfacePolicyError("pre/post roll must be non-negative")
    region = _region(player_region)
    capture_resolution = _pair(resolution, "resolution")
    capture_grid = _pair(grid, "grid")
    if capture_resolution[0] % capture_grid[0] != 0 or capture_resolution[1] % capture_grid[1] != 0:
        raise SourceSurfacePolicyError("resolution must divide evenly by grid")
    requested_display_ids = display_ids or [
        "yt.display.page_url",
        "yt.display.player_region",
        "yt.display.title",
        "yt.display.elapsed_time",
        "yt.display.duration",
    ]
    requested_button_ids = button_ids or ["yt.button.play_pause"]
    _validate_surface_ids(requested_display_ids, requested_button_ids)
    safe_run_id = run_id or f"{element_id}_source_surface_trial"
    capture_duration = round(pre_roll + source_duration + post_roll, 3)
    expected_end = round(pre_roll + source_duration, 3)
    fps_value = _positive_float(fps, "fps")
    chunk_count = _positive_int(chunk_frames, "chunk_frames")
    command = _native_capture_command(
        capture_exe=native_capture_exe,
        duration_seconds=capture_duration,
        fps=fps_value,
        resolution=capture_resolution,
        grid=capture_grid,
        region=region,
        output_root=output_root,
        run_id=safe_run_id,
        chunk_frames=chunk_count,
    )
    plan: dict[str, Any] = {
        "schema_version": "truevision_source_surface_capture_plan_v1",
        "surface_id": "youtube_source_surface_v1",
        "run_id": safe_run_id,
        "element_id": element_id,
        "source": {
            "kind": "youtube_watch_surface",
            "url": source_url,
            "address_bar_url": canonical_source["address_bar_url"],
            "video_id": canonical_source["video_id"],
            "navigation_method": canonical_source["navigation_method"],
            "removed_query_keys": canonical_source["removed_query_keys"],
            "title": source_title,
            "video_duration_seconds": round(source_duration, 3),
            "source_authority": "operator_approved_visible_page",
        },
        "navigation_flow": [
            {
                "action": "focus_address_bar",
                "target": "browser.address_bar",
                "purpose": "avoid YouTube search and recommendation surfaces",
            },
            {
                "action": "paste_approved_url",
                "target": "browser.address_bar",
                "value": canonical_source["address_bar_url"],
            },
            {
                "action": "press_enter",
                "target": "browser.address_bar",
            },
            {
                "action": "wait_for_video_page_load",
                "required_display_ids": ["yt.display.page_url", "yt.display.title", "yt.display.duration"],
            },
            {
                "action": "verify_video_state",
                "required_receipt_fields": [
                    "resolved_url",
                    "video_title",
                    "duration_detected_seconds",
                    "visual_state_records",
                    "not_gray_screen",
                    "not_error_page",
                    "profile_created",
                    "teacher_chunks_purged",
                ],
            },
        ],
        "display_ids_observed": requested_display_ids,
        "button_ids_requested": requested_button_ids,
        "button_ids_approved": requested_button_ids,
        "timeline": [
            {
                "event": "capture_start",
                "at_capture_seconds": 0.0,
                "clock": "native_capture_process_start",
            },
            {
                "event": "play_button",
                "at_capture_seconds": round(pre_roll, 3),
                "button_id": "yt.button.play_pause",
                "clock": "source_surface_action",
            },
            {
                "event": "expected_video_end",
                "at_capture_seconds": expected_end,
                "source_elapsed_seconds": round(source_duration, 3),
                "clock": "source_video_time",
            },
            {
                "event": "capture_stop",
                "at_capture_seconds": capture_duration,
                "clock": "native_capture_duration",
            },
        ],
        "capture": {
            "mode": "selected_player_region",
            "region": region,
            "resolution": capture_resolution,
            "grid": capture_grid,
            "fps": fps_value,
            "duration_seconds": capture_duration,
            "pre_roll_seconds": round(pre_roll, 3),
            "post_roll_seconds": round(post_roll, 3),
            "raw_frames_saved": False,
            "audio": False,
            "output_root": output_root,
        },
        "native_capture_command": command,
        "boundary": {
            "display_ids_observe_only": True,
            "button_ids_require_operator_or_harness_approval": True,
            "browser_autonomy": False,
            "youtube_search_navigation": False,
            "address_bar_navigation_required": True,
            "raw_download": False,
            "comments_or_account_actions": False,
            "generated_media_is_evidence": False,
            "completed_macro_is_not_completed_capture": True,
        },
    }
    plan["plan_hash"] = stable_hash(plan)
    return plan


def write_source_surface_capture_plan(args: dict[str, Any], *, storage_root: Path) -> dict[str, Any]:
    plan = build_source_surface_capture_plan(
        element_id=str(args.get("element_id") or ""),
        source_url=str(args.get("source_url") or ""),
        source_title=str(args.get("source_title") or ""),
        video_duration_seconds=args.get("video_duration_seconds"),
        player_region=args.get("player_region"),
        run_id=args.get("run_id"),
        fps=float(args.get("fps", 15.0)),
        resolution=args.get("resolution", [1280, 720]),
        grid=args.get("grid", [320, 180]),
        pre_roll_seconds=float(args.get("pre_roll_seconds", 0.25)),
        post_roll_seconds=float(args.get("post_roll_seconds", 0.75)),
        display_ids=args.get("display_ids"),
        button_ids=args.get("button_ids"),
        output_root=str(args.get("output_root") or storage_root / "artifacts" / "source_surface_captures"),
        native_capture_exe=str(
            args.get("native_capture_exe")
            or "D:\\TrueVision_Generation_Lab\\native\\truevision_capture_rs\\target\\release\\truevision_capture_rs.exe"
        ),
        chunk_frames=int(args.get("chunk_frames", 300)),
    )
    plan_dir = storage_root / "manifests" / "source_surface_capture_plans"
    receipt_dir = storage_root / "receipts" / "source_surface_capture_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / f"{plan['run_id']}_plan.json"
    receipt_path = receipt_dir / f"{plan['run_id']}_receipt.json"
    plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
    receipt = {
        "schema_version": "truevision_source_surface_capture_plan_receipt_v1",
        "tool": "source_surface_capture_plan",
        "created_at_utc": utc_now(),
        "surface_id": plan["surface_id"],
        "run_id": plan["run_id"],
        "result": {
            "plan_json": str(plan_path),
            "plan_hash": plan["plan_hash"],
            "capture_duration_seconds": plan["capture"]["duration_seconds"],
            "capture_region": plan["capture"]["region"],
            "native_capture_command_ready": True,
        },
        "boundary": plan["boundary"],
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_id": plan["run_id"],
        "plan_json": str(plan_path),
        "receipt_json": str(receipt_path),
        "plan_hash": plan["plan_hash"],
        "capture_duration_seconds": plan["capture"]["duration_seconds"],
        "address_bar_url": plan["source"]["address_bar_url"],
        "native_capture_command": plan["native_capture_command"],
    }


def write_source_surface_multi_sample_plan(args: dict[str, Any], *, storage_root: Path) -> dict[str, Any]:
    plan = build_source_surface_multi_sample_plan(
        element_id=str(args.get("element_id") or ""),
        source_url=str(args.get("source_url") or ""),
        source_title=str(args.get("source_title") or ""),
        video_duration_seconds=args.get("video_duration_seconds"),
        player_region=args.get("player_region"),
        run_id=args.get("run_id"),
        sample_seconds=float(args.get("sample_seconds", 12.0)),
        sample_count=int(args.get("sample_count", 4)),
        large_video_threshold_seconds=float(args.get("large_video_threshold_seconds", 600.0)),
        fps=float(args.get("fps", 15.0)),
        resolution=args.get("resolution", [1280, 720]),
        grid=args.get("grid", [320, 180]),
        pre_roll_seconds=float(args.get("pre_roll_seconds", 0.25)),
        post_roll_seconds=float(args.get("post_roll_seconds", 0.75)),
        output_root=str(args.get("output_root") or storage_root / "artifacts" / "source_surface_captures"),
        native_capture_exe=str(
            args.get("native_capture_exe")
            or "D:\\TrueVision_Generation_Lab\\native\\truevision_capture_rs\\target\\release\\truevision_capture_rs.exe"
        ),
        chunk_frames=int(args.get("chunk_frames", 300)),
    )
    plan_dir = storage_root / "manifests" / "source_surface_multi_sample_plans"
    receipt_dir = storage_root / "receipts" / "source_surface_multi_sample_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / f"{plan['run_id']}_multi_sample_plan.json"
    receipt_path = receipt_dir / f"{plan['run_id']}_multi_sample_receipt.json"
    plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
    receipt = {
        "schema_version": "truevision_source_surface_multi_sample_plan_receipt_v1",
        "tool": "source_surface_multi_sample_plan",
        "created_at_utc": utc_now(),
        "run_id": plan["run_id"],
        "plan_json": str(plan_path),
        "plan_hash": plan["plan_hash"],
        "sample_count": plan["sampling"]["sample_count"],
        "sample_starts": [sample["start_seconds"] for sample in plan["samples"]],
        "boundary": plan["boundary"],
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_id": plan["run_id"],
        "plan_json": str(plan_path),
        "receipt_json": str(receipt_path),
        "plan_hash": plan["plan_hash"],
        "sample_count": plan["sampling"]["sample_count"],
        "sample_starts": [sample["start_seconds"] for sample in plan["samples"]],
        "sample_navigation_urls": [sample["sample_navigation_url"] for sample in plan["samples"]],
    }


def build_source_surface_video_state_receipt(
    *,
    run_id: str,
    approved_url: str,
    resolved_url: str,
    video_title: str,
    duration_detected_seconds: float,
    visual_state_records: int,
    not_gray_screen: bool,
    not_error_page: bool,
    profile_created: bool,
    teacher_chunks_purged: bool,
    source_time_delta_seconds: float | None = None,
    expected_sample_seconds: float | None = None,
    requested_start_seconds: float | None = None,
    source_time_before_seconds: float | None = None,
    source_time_after_seconds: float | None = None,
    source_duration_seconds: float | None = None,
    visual_motion_score: float | None = None,
    minimum_visual_motion_score: float | None = None,
) -> dict[str, Any]:
    if not run_id:
        raise SourceSurfacePolicyError("run_id is required")
    approved = canonicalize_approved_source_url(approved_url)
    resolved = canonicalize_approved_source_url(resolved_url)
    duration = _positive_float(duration_detected_seconds, "duration_detected_seconds")
    records = _positive_int(visual_state_records, "visual_state_records")
    title_ok = bool(str(video_title or "").strip())
    source_time_delta: float | None = None
    expected_sample: float | None = None
    video_time_required = source_time_delta_seconds is not None or expected_sample_seconds is not None
    if source_time_delta_seconds is not None:
        try:
            source_time_delta = float(source_time_delta_seconds)
        except (TypeError, ValueError) as exc:
            raise SourceSurfacePolicyError("source_time_delta_seconds must be a number") from exc
        if source_time_delta < 0:
            raise SourceSurfacePolicyError("source_time_delta_seconds must be non-negative")
    if expected_sample_seconds is not None:
        expected_sample = _positive_float(expected_sample_seconds, "expected_sample_seconds")
    if video_time_required and (source_time_delta is None or expected_sample is None):
        raise SourceSurfacePolicyError("source_time_delta_seconds and expected_sample_seconds must be supplied together")
    time_advance_ok = True
    if video_time_required:
        time_advance_ok = bool(source_time_delta is not None and expected_sample is not None and source_time_delta >= expected_sample * 0.75)
    requested_start: float | None = None
    source_time_before: float | None = None
    source_time_after: float | None = None
    source_duration: float | None = None
    target_seek_required = (
        requested_start_seconds is not None
        or source_time_before_seconds is not None
        or source_time_after_seconds is not None
        or source_duration_seconds is not None
    )
    if requested_start_seconds is not None:
        try:
            requested_start = float(requested_start_seconds)
        except (TypeError, ValueError) as exc:
            raise SourceSurfacePolicyError("requested_start_seconds must be a number") from exc
        if requested_start < 0:
            raise SourceSurfacePolicyError("requested_start_seconds must be non-negative")
    if source_time_before_seconds is not None:
        try:
            source_time_before = float(source_time_before_seconds)
        except (TypeError, ValueError) as exc:
            raise SourceSurfacePolicyError("source_time_before_seconds must be a number") from exc
        if source_time_before < 0:
            raise SourceSurfacePolicyError("source_time_before_seconds must be non-negative")
    if source_time_after_seconds is not None:
        try:
            source_time_after = float(source_time_after_seconds)
        except (TypeError, ValueError) as exc:
            raise SourceSurfacePolicyError("source_time_after_seconds must be a number") from exc
        if source_time_after < 0:
            raise SourceSurfacePolicyError("source_time_after_seconds must be non-negative")
    if source_duration_seconds is not None:
        source_duration = _positive_float(source_duration_seconds, "source_duration_seconds")
    if target_seek_required and (
        requested_start is None
        or source_time_before is None
        or source_time_after is None
        or source_duration is None
        or expected_sample is None
    ):
        raise SourceSurfacePolicyError(
            "requested_start_seconds, source_time_before_seconds, source_time_after_seconds, "
            "source_duration_seconds, and expected_sample_seconds must be supplied together"
        )
    player_duration_ok = True
    target_seek_ok = True
    if target_seek_required:
        assert requested_start is not None
        assert source_time_before is not None
        assert source_time_after is not None
        assert source_duration is not None
        assert expected_sample is not None
        seek_tolerance = max(2.0, expected_sample * 0.25)
        player_duration_ok = source_duration >= min(duration, requested_start + expected_sample) - seek_tolerance
        if duration >= 120.0:
            player_duration_ok = player_duration_ok and source_duration >= duration * 0.90
        target_seek_ok = (
            source_time_before >= requested_start - seek_tolerance
            and source_time_after >= requested_start + expected_sample * 0.75
            and source_time_after <= source_duration + seek_tolerance
        )
    visual_motion: float | None = None
    minimum_motion: float | None = None
    visual_motion_required = visual_motion_score is not None or minimum_visual_motion_score is not None
    if visual_motion_score is not None:
        try:
            visual_motion = float(visual_motion_score)
        except (TypeError, ValueError) as exc:
            raise SourceSurfacePolicyError("visual_motion_score must be a number") from exc
        if visual_motion < 0:
            raise SourceSurfacePolicyError("visual_motion_score must be non-negative")
    if minimum_visual_motion_score is not None:
        try:
            minimum_motion = float(minimum_visual_motion_score)
        except (TypeError, ValueError) as exc:
            raise SourceSurfacePolicyError("minimum_visual_motion_score must be a number") from exc
        if minimum_motion < 0:
            raise SourceSurfacePolicyError("minimum_visual_motion_score must be non-negative")
    if visual_motion_required and (visual_motion is None or minimum_motion is None):
        raise SourceSurfacePolicyError("visual_motion_score and minimum_visual_motion_score must be supplied together")
    visual_motion_ok = True
    if visual_motion_required:
        visual_motion_ok = bool(visual_motion is not None and minimum_motion is not None and visual_motion >= minimum_motion)
    checks = {
        "address_bar_navigation": True,
        "video_id_match": approved["video_id"] == resolved["video_id"],
        "resolved_url": resolved["address_bar_url"].startswith("https://www.youtube.com/watch?v="),
        "video_title": title_ok,
        "duration_detected": duration > 0.0,
        "source_player_duration_matches_approved": player_duration_ok,
        "source_video_target_time_reached": target_seek_ok,
        "source_video_time_advanced": time_advance_ok,
        "visual_temporal_motion": visual_motion_ok,
        "visual_state_records": records > 0,
        "not_gray_screen": bool(not_gray_screen),
        "not_error_page": bool(not_error_page),
        "profile_created": bool(profile_created),
        "teacher_chunks_purged": bool(teacher_chunks_purged),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SourceSurfacePolicyError(f"video-state receipt rejected; failed checks: {', '.join(failed)}")
    receipt = {
        "schema_version": "truevision_source_surface_video_state_receipt_v1",
        "tool": "source_surface_video_state_receipt",
        "created_at_utc": utc_now(),
        "status": "verified",
        "run_id": run_id,
        "approved_url": approved["address_bar_url"],
        "resolved_url": resolved["address_bar_url"],
        "video_id": approved["video_id"],
        "video_title": str(video_title).strip(),
        "duration_detected_seconds": round(duration, 6),
        "source_time_delta_seconds": round(source_time_delta, 6) if source_time_delta is not None else None,
        "expected_sample_seconds": round(expected_sample, 6) if expected_sample is not None else None,
        "requested_start_seconds": round(requested_start, 6) if requested_start is not None else None,
        "source_time_before_seconds": round(source_time_before, 6) if source_time_before is not None else None,
        "source_time_after_seconds": round(source_time_after, 6) if source_time_after is not None else None,
        "source_duration_seconds": round(source_duration, 6) if source_duration is not None else None,
        "visual_motion_score": round(visual_motion, 6) if visual_motion is not None else None,
        "minimum_visual_motion_score": round(minimum_motion, 6) if minimum_motion is not None else None,
        "visual_state_records": records,
        "checks": checks,
        "law": "A completed macro is not a completed capture. A verified video-state receipt is a completed capture.",
        "boundary": {
            "youtube_search_navigation": False,
            "address_bar_navigation_required": True,
            "profile_first_verify_second_purge_third": True,
            "raw_download": False,
            "generated_media_is_evidence": False,
        },
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    return receipt


def write_source_surface_video_state_receipt(args: dict[str, Any], *, storage_root: Path) -> dict[str, Any]:
    receipt = build_source_surface_video_state_receipt(
        run_id=str(args.get("run_id") or ""),
        approved_url=str(args.get("approved_url") or args.get("source_url") or ""),
        resolved_url=str(args.get("resolved_url") or ""),
        video_title=str(args.get("video_title") or args.get("title") or ""),
        duration_detected_seconds=args.get("duration_detected_seconds"),
        visual_state_records=args.get("visual_state_records"),
        not_gray_screen=_bool_value(args.get("not_gray_screen"), "not_gray_screen"),
        not_error_page=_bool_value(args.get("not_error_page"), "not_error_page"),
        profile_created=_bool_value(args.get("profile_created"), "profile_created"),
        teacher_chunks_purged=_bool_value(args.get("teacher_chunks_purged"), "teacher_chunks_purged"),
        source_time_delta_seconds=args.get("source_time_delta_seconds"),
        expected_sample_seconds=args.get("expected_sample_seconds"),
        requested_start_seconds=args.get("requested_start_seconds"),
        source_time_before_seconds=args.get("source_time_before_seconds"),
        source_time_after_seconds=args.get("source_time_after_seconds"),
        source_duration_seconds=args.get("source_duration_seconds"),
        visual_motion_score=args.get("visual_motion_score"),
        minimum_visual_motion_score=args.get("minimum_visual_motion_score"),
    )
    receipt_dir = storage_root / "receipts" / "source_surface_video_state"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{receipt['run_id']}_video_state_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_id": receipt["run_id"],
        "receipt_json": str(receipt_path),
        "receipt_hash": receipt["receipt_hash"],
        "status": receipt["status"],
        "resolved_url": receipt["resolved_url"],
        "video_title": receipt["video_title"],
        "duration_detected_seconds": receipt["duration_detected_seconds"],
        "visual_state_records": receipt["visual_state_records"],
    }
