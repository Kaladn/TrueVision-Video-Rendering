from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from truevision_runtime.learning_intake.batch_queue import parse_approved_youtube_sources
from truevision_runtime.learning_intake.coordinate_surface import (
    build_coordinate_intake_plan,
    build_coordinate_intake_receipt,
    validate_coordinate_map,
)
from truevision_runtime.learning_intake.youtube_metadata import fetch_youtube_metadata
from truevision_runtime.learning_intake.source_surface import build_source_surface_multi_sample_plan
from truevision_youtube_learning_intake_batch import (
    NATIVE_CAPTURE_EXE,
    _find_capture_manifest,
    _profile_capture,
    _purge_unverified_profile_artifacts,
    _visual_temporal_change_score,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "storage" / "config" / "coordinate_maps" / "youtube_intake_map.json"
DEFAULT_RUN_ROOT = Path("E:/TruEVision Generation/library/youtube_learning_intake_pilot")


def _now_slug() -> str:
    import datetime as _dt

    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, allow_nan=False) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _set_clipboard_text(text: str) -> None:
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value ([Console]::In.ReadToEnd())"],
        input=text,
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _mouse_click(point: list[int]) -> None:
    ctypes.windll.user32.SetCursorPos(int(point[0]), int(point[1]))
    time.sleep(0.05)
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.03)
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)


def _key_down(vk: int) -> None:
    ctypes.windll.user32.keybd_event(vk, 0, 0, 0)


def _key_up(vk: int) -> None:
    ctypes.windll.user32.keybd_event(vk, 0, 0x0002, 0)


def _press_key(vk: int) -> None:
    _key_down(vk)
    time.sleep(0.02)
    _key_up(vk)


def _hotkey(*keys: int) -> None:
    for key in keys:
        _key_down(key)
    time.sleep(0.04)
    for key in reversed(keys):
        _key_up(key)


def _paste_url_in_existing_browser(url: str, address_bar: list[int]) -> None:
    _set_clipboard_text(url)
    _mouse_click(address_bar)
    time.sleep(0.08)
    _hotkey(0x11, 0x4C)  # Ctrl+L
    time.sleep(0.08)
    _hotkey(0x11, 0x56)  # Ctrl+V
    time.sleep(0.08)
    _press_key(0x0D)  # Enter


def _select_trial_sources(
    entries: list[dict[str, Any]],
    *,
    metadata_timeout: float,
    long_threshold_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    metadata_by_id: dict[str, dict[str, Any]] = {}
    long_selected = False
    short_selected = 0
    seen_ids: set[str] = set()
    for entry in entries:
        video_id = str(entry["video_id"])
        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        try:
            metadata = fetch_youtube_metadata(entry["source_url"], timeout_seconds=metadata_timeout)
        except Exception as exc:  # noqa: BLE001 - keep scanning approved list.
            metadata_by_id[video_id] = {"error": str(exc)}
            continue
        metadata_by_id[video_id] = metadata
        duration = float(metadata["duration_seconds"])
        enriched = {
            **entry,
            "video_title": metadata["video_title"],
            "duration_seconds": duration,
        }
        if not long_selected and duration >= long_threshold_seconds:
            selected.append(enriched)
            long_selected = True
            continue
        if duration < long_threshold_seconds and short_selected < 2:
            selected.append(enriched)
            short_selected += 1
        if long_selected and short_selected >= 2:
            break
    return selected, metadata_by_id


def _select_source_window(
    entries: list[dict[str, Any]],
    *,
    start_source_order: int,
    source_count: int,
    metadata_timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if source_count <= 0:
        raise ValueError("source_count must be positive for source-window selection")
    selected: list[dict[str, Any]] = []
    metadata_by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for entry in entries:
        if int(entry["source_order"]) < int(start_source_order):
            continue
        video_id = str(entry["video_id"])
        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        try:
            metadata = fetch_youtube_metadata(entry["source_url"], timeout_seconds=metadata_timeout)
        except Exception as exc:  # noqa: BLE001 - keep scanning approved list.
            metadata_by_id[video_id] = {"error": str(exc)}
            continue
        metadata_by_id[video_id] = metadata
        selected.append(
            {
                **entry,
                "video_title": metadata["video_title"],
                "duration_seconds": float(metadata["duration_seconds"]),
            }
        )
        if len(selected) >= source_count:
            break
    return selected, metadata_by_id


def _select_all_sources(
    entries: list[dict[str, Any]],
    *,
    start_source_order: int,
    metadata_timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    metadata_by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for entry in entries:
        if int(entry["source_order"]) < int(start_source_order):
            continue
        video_id = str(entry["video_id"])
        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        try:
            metadata = fetch_youtube_metadata(entry["source_url"], timeout_seconds=metadata_timeout)
        except Exception as exc:  # noqa: BLE001 - keep metadata failures in queue report.
            metadata_by_id[video_id] = {"error": str(exc)}
            continue
        metadata_by_id[video_id] = metadata
        duration = float(metadata.get("duration_seconds", 0.0))
        if duration <= 0.0:
            metadata_by_id[video_id] = {**metadata, "error": "non_positive_duration"}
            continue
        selected.append(
            {
                **entry,
                "video_title": metadata["video_title"],
                "duration_seconds": duration,
            }
        )
    return selected, metadata_by_id


def _sample_entries_for_source(
    source: dict[str, Any],
    *,
    run_id: str,
    run_root: Path,
    coordinate_map: dict[str, Any],
    sample_seconds: float,
    long_threshold_seconds: float,
    fps: float,
    resolution: list[int],
    grid: list[int],
) -> list[dict[str, Any]]:
    sample_count = 4 if float(source["duration_seconds"]) >= long_threshold_seconds else 1
    source_run_id = f"{run_id}_{int(source['source_order']):03d}_{source['element_id']}_{source['video_id']}"
    plan = build_source_surface_multi_sample_plan(
        element_id=source["element_id"],
        source_url=source["source_url"],
        video_title=source["video_title"],
        video_duration_seconds=float(source["duration_seconds"]),
        player_region=coordinate_map["capture_region"],
        run_id=source_run_id,
        sample_seconds=sample_seconds,
        sample_count=sample_count,
        large_video_threshold_seconds=long_threshold_seconds,
        fps=fps,
        resolution=resolution,
        grid=grid,
        output_root=str(run_root / "captures"),
        native_capture_exe=str(NATIVE_CAPTURE_EXE),
    )
    samples: list[dict[str, Any]] = []
    for sample in plan["samples"]:
        sample_run_id = str(sample["run_id"])
        coord_plan = build_coordinate_intake_plan(
            run_id=run_id,
            source=source,
            sample=sample,
            coordinate_map=coordinate_map,
            output_root=run_root / "captures",
            native_capture_exe=NATIVE_CAPTURE_EXE,
            fps=fps,
            resolution=resolution,
            grid=grid,
        )
        samples.append(
            {
                "source": source,
                "sample": sample,
                "sample_run_id": sample_run_id,
                "coordinate_plan": coord_plan,
            }
        )
    return samples


def _execute_sample(
    sample_entry: dict[str, Any],
    *,
    coordinate_map: dict[str, Any],
    run_root: Path,
    storage_root: Path,
    load_wait_seconds: float,
    pre_play_seconds: float,
) -> dict[str, Any]:
    source = sample_entry["source"]
    sample = sample_entry["sample"]
    plan = sample_entry["coordinate_plan"]
    sample_run_id = sample_entry["sample_run_id"]
    _paste_url_in_existing_browser(str(sample["sample_navigation_url"]), coordinate_map["points"]["address_bar"])
    time.sleep(load_wait_seconds)

    capture_command = list(plan["native_capture_command"])
    capture_proc = subprocess.Popen(
        capture_command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(pre_play_seconds)
    _mouse_click(coordinate_map["points"]["video_play"])
    stdout, stderr = capture_proc.communicate(timeout=float(sample["duration_seconds"]) + 45.0)
    if capture_proc.returncode != 0:
        raise RuntimeError(stderr.strip() or stdout.strip() or f"capture failed with code {capture_proc.returncode}")

    manifest_path = _find_capture_manifest(run_root / "captures", sample_run_id)
    profile: dict[str, Any] | None = None
    try:
        profile = _profile_capture(
            manifest_path=manifest_path,
            element_id=str(source["element_id"]),
            run_id=sample_run_id,
            storage_root=storage_root,
        )
        teacher_purged = profile["purge"]["status"] == "purged"
        visual_motion_score = _visual_temporal_change_score(profile["creation_signature"])
        receipt = build_coordinate_intake_receipt(
            run_id=sample_run_id,
            approved_url=str(source["source_url"]),
            sample_navigation_url=str(sample["sample_navigation_url"]),
            coordinate_map_id=str(coordinate_map.get("map_id") or coordinate_map.get("created_by") or "operator_coordinate_map"),
            coordinate_map_sha256=str(coordinate_map["map_sha256"]),
            capture_region=coordinate_map["capture_region"],
            visual_state_records=int(profile["sampled_frames"]),
            profile_created=Path(profile["profile_json"]).exists(),
            teacher_chunks_purged=teacher_purged,
            visual_motion_score=visual_motion_score,
        )
    except Exception:
        _purge_unverified_profile_artifacts(profile)
        raise
    receipt_dir = storage_root / "receipts" / "coordinate_surface_intake"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{sample_run_id}_coordinate_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    remaining_chunks = sorted((run_root / "captures" / sample_run_id).glob("**/*.tvcells"))
    return {
        "source_order": source["source_order"],
        "category": source["category"],
        "element_id": source["element_id"],
        "video_id": source["video_id"],
        "sample_index": sample["sample_index"],
        "run_id": sample_run_id,
        "sample_url": sample["sample_navigation_url"],
        "status": "ok",
        "profile_json": profile["profile_json"],
        "profile_sha256": profile["profile_sha256"],
        "coordinate_receipt_json": str(receipt_path),
        "sampled_frames": profile["sampled_frames"],
        "visual_motion_score": round(visual_motion_score, 6),
        "six_one_six_windows": profile["six_one_six_windows"],
        "purge": profile["purge"],
        "remaining_tvcells": len(remaining_chunks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate-driven YouTube intake using the current visible browser only.")
    parser.add_argument("--approved-file", default=str(ROOT / "approved youtube videos.md"))
    parser.add_argument("--coordinate-map", default=str(DEFAULT_MAP))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sample-seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--grid", default="320x180")
    parser.add_argument("--long-threshold-seconds", type=float, default=600.0)
    parser.add_argument("--metadata-timeout", type=float, default=15.0)
    parser.add_argument("--load-wait-seconds", type=float, default=8.0)
    parser.add_argument("--pre-play-seconds", type=float, default=0.35)
    parser.add_argument("--max-samples", type=int, default=0, help="When executing, stop after this many planned samples.")
    parser.add_argument("--start-source-order", type=int, default=1, help="First approved source_order eligible for selection.")
    parser.add_argument("--source-count", type=int, default=0, help="Select this many contiguous approved sources instead of trial selection.")
    parser.add_argument("--all-sources", action="store_true", help="Select every approved unique source from start-source-order onward.")
    args = parser.parse_args()

    map_path = Path(args.coordinate_map)
    if not map_path.exists():
        raise SystemExit(f"coordinate map missing: {map_path}. Run scripts/truevision_coordinate_map_screen.py first.")
    raw_coordinate_map = json.loads(map_path.read_text(encoding="utf-8"))
    if args.execute and raw_coordinate_map.get("template"):
        raise SystemExit(f"coordinate map is only a template, not executable: {map_path}")
    coordinate_map_sha256 = _file_sha256(map_path)
    coordinate_map = validate_coordinate_map(raw_coordinate_map)
    coordinate_map["map_id"] = map_path.stem
    coordinate_map["map_sha256"] = coordinate_map_sha256

    run_id = args.run_id or f"coordinate_youtube_intake_{_now_slug()}"
    run_root = Path(args.run_root) if args.run_root else DEFAULT_RUN_ROOT / run_id
    storage_root = run_root / "storage"
    run_root.mkdir(parents=True, exist_ok=True)
    width, height = [int(part) for part in args.resolution.lower().split("x", 1)]
    grid_w, grid_h = [int(part) for part in args.grid.lower().split("x", 1)]

    approved_text = Path(args.approved_file).read_text(encoding="utf-8")
    approved_entries = parse_approved_youtube_sources(approved_text)
    if args.all_sources:
        selected_sources, metadata_by_id = _select_all_sources(
            approved_entries,
            start_source_order=int(args.start_source_order),
            metadata_timeout=args.metadata_timeout,
        )
        if not selected_sources:
            raise SystemExit(f"no approved sources selected from source_order {int(args.start_source_order)}")
        selection_mode = "all_sources"
    elif int(args.source_count) > 0:
        selected_sources, metadata_by_id = _select_source_window(
            approved_entries,
            start_source_order=int(args.start_source_order),
            source_count=int(args.source_count),
            metadata_timeout=args.metadata_timeout,
        )
        if len(selected_sources) < int(args.source_count):
            raise SystemExit(
                f"expected {int(args.source_count)} source-window sources from source_order {int(args.start_source_order)}, got {len(selected_sources)}"
            )
        selection_mode = "source_window"
    else:
        eligible_entries = [entry for entry in approved_entries if int(entry["source_order"]) >= int(args.start_source_order)]
        selected_sources, metadata_by_id = _select_trial_sources(
            eligible_entries,
            metadata_timeout=args.metadata_timeout,
            long_threshold_seconds=args.long_threshold_seconds,
        )
        if len(selected_sources) < 3:
            raise SystemExit(f"expected 1 long + 2 short sources, got {len(selected_sources)}")
        selection_mode = "trial_long_plus_two_short"
    sample_entries: list[dict[str, Any]] = []
    for source in selected_sources:
        sample_entries.extend(
            _sample_entries_for_source(
                source,
                run_id=run_id,
                run_root=run_root,
                coordinate_map=coordinate_map,
                sample_seconds=args.sample_seconds,
                long_threshold_seconds=args.long_threshold_seconds,
                fps=args.fps,
                resolution=[width, height],
                grid=[grid_w, grid_h],
            )
        )
    planned_sample_count = len(sample_entries)
    skipped_by_max_samples = 0
    if args.execute and args.max_samples > 0 and len(sample_entries) > args.max_samples:
        skipped_by_max_samples = len(sample_entries) - args.max_samples
        sample_entries = sample_entries[: args.max_samples]
    queue = {
        "schema_version": "truevision_coordinate_youtube_intake_queue_v1",
        "run_id": run_id,
        "coordinate_map_path": str(map_path),
        "coordinate_map_sha256": coordinate_map_sha256,
        "coordinate_map": coordinate_map,
        "selection_mode": selection_mode,
        "all_sources": bool(args.all_sources),
        "start_source_order": int(args.start_source_order),
        "source_count": int(args.source_count),
        "selected_source_count": len(selected_sources),
        "planned_sample_count": planned_sample_count,
        "sample_count": len(sample_entries),
        "max_samples": int(args.max_samples),
        "skipped_by_max_samples": skipped_by_max_samples,
        "selected_sources": selected_sources,
        "metadata_by_id": metadata_by_id,
        "samples": [entry["coordinate_plan"] for entry in sample_entries],
        "boundary": {
            "uses_current_operator_browser": True,
            "new_browser_instance": False,
            "youtube_search_navigation": False,
            "raw_download": False,
            "source_time_proof": False,
            "coordinate_map_required_before_run": True,
        },
    }
    _write_json(run_root / "coordinate_queue.json", queue)
    print(f"queue ready: sources={len(selected_sources)} samples={len(sample_entries)} run_root={run_root}")
    if not args.execute:
        print("dry run only. Re-run with --execute when the visible browser layout matches the coordinate map.")
        return 0

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    events_path = run_root / "events" / "coordinate_events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    for entry in sample_entries:
        event = {
            "source_order": entry["source"]["source_order"],
            "category": entry["source"]["category"],
            "element_id": entry["source"]["element_id"],
            "video_id": entry["source"]["video_id"],
            "sample_index": entry["sample"]["sample_index"],
            "run_id": entry["sample_run_id"],
            "sample_url": entry["sample"]["sample_navigation_url"],
        }
        try:
            print(f"sample start: {entry['sample_run_id']} -> {entry['sample']['sample_navigation_url']}")
            result = _execute_sample(
                entry,
                coordinate_map=coordinate_map,
                run_root=run_root,
                storage_root=storage_root,
                load_wait_seconds=args.load_wait_seconds,
                pre_play_seconds=args.pre_play_seconds,
            )
            results.append(result)
            _append_jsonl(events_path, {"event": "sample_complete", **result})
            print(f"sample ok: {entry['sample_run_id']} frames={result['sampled_frames']} purge={result['purge']['status']}")
        except Exception as exc:  # noqa: BLE001 - keep bounded trial moving.
            failure = {**event, "status": "failed", "error": str(exc)}
            failures.append(failure)
            _append_jsonl(events_path, {"event": "sample_failed", **failure})
            print(f"sample failed: {entry['sample_run_id']} | {exc}")
    capture_root = run_root / "captures"
    summary = {
        "schema_version": "truevision_coordinate_youtube_intake_summary_v1",
        "run_id": run_id,
        "coordinate_map_path": str(map_path),
        "coordinate_map_sha256": coordinate_map_sha256,
        "selection_mode": selection_mode,
        "all_sources": bool(args.all_sources),
        "start_source_order": int(args.start_source_order),
        "source_count": int(args.source_count),
        "planned_sample_count": planned_sample_count,
        "max_samples": int(args.max_samples),
        "skipped_by_max_samples": skipped_by_max_samples,
        "completed_sample_count": len(results),
        "failed_sample_count": len(failures),
        "results": results,
        "failures": failures,
        "retention_check": {
            "remaining_tvcells": len(sorted(capture_root.glob("**/*.tvcells"))) if capture_root.exists() else 0,
            "remaining_records_jsonl": len(sorted(capture_root.glob("**/*_records.jsonl"))) if capture_root.exists() else 0,
        },
        "boundary": {
            "uses_current_operator_browser": True,
            "new_browser_instance": False,
            "youtube_search_navigation": False,
            "raw_download": False,
            "source_time_proof": False,
            "coordinate_map_required_before_run": True,
        },
    }
    _write_json(run_root / "coordinate_summary.json", summary)
    print(f"complete: completed={len(results)} failed={len(failures)} summary={run_root / 'coordinate_summary.json'}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
