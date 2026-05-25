from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "modules") not in sys.path:
    sys.path.insert(0, str(ROOT / "modules"))

from truevision_runtime.av_tools.av_tool_runner import run_av_tool_call
from truevision_runtime.learning_intake.batch_queue import (
    build_batch_queue,
    parse_approved_youtube_sources,
)
from truevision_runtime.learning_intake.youtube_cdp import (
    build_video_play_expression,
    build_video_state_expression,
    evaluate_on_first_page,
    navigate_first_page,
)
from truevision_runtime.learning_intake.youtube_metadata import fetch_youtube_metadata


EDGE_EXE = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
NATIVE_CAPTURE_EXE = ROOT / "native" / "truevision_capture_rs" / "target" / "release" / "truevision_capture_rs.exe"


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


def _console(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(str(text).encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _run_powershell(command: str, *, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell", "-NoProfile", "-STA", "-Command", command],
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _send_keys(keys: str, *, delay_ms: int = 120) -> None:
    escaped = keys.replace("'", "''")
    command = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        f"[System.Windows.Forms.SendKeys]::SendWait('{escaped}'); "
        f"Start-Sleep -Milliseconds {int(delay_ms)}"
    )
    completed = _run_powershell(command)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "SendKeys failed")


def _activate_process_window(process_id: int, *, delay_ms: int = 250) -> None:
    command = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName Microsoft.VisualBasic
[Microsoft.VisualBasic.Interaction]::AppActivate({int(process_id)}) | Out-Null
Start-Sleep -Milliseconds {int(delay_ms)}
"""
    completed = _run_powershell(command, timeout=5.0)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "window activation failed")


def _navigate_address_bar(url: str) -> None:
    command = f"""
$ErrorActionPreference = 'Stop'
Set-Clipboard -Value @'
{url}
'@
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.SendKeys]::SendWait('^l')
Start-Sleep -Milliseconds 150
[System.Windows.Forms.SendKeys]::SendWait('^v')
Start-Sleep -Milliseconds 150
[System.Windows.Forms.SendKeys]::SendWait('{{ENTER}}')
"""
    completed = _run_powershell(command, timeout=15.0)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "address bar navigation failed")


def _start_edge(profile_dir: Path, url: str, *, devtools_port: int) -> subprocess.Popen[Any]:
    if not EDGE_EXE.exists():
        raise FileNotFoundError(str(EDGE_EXE))
    profile_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            str(EDGE_EXE),
            f"--user-data-dir={profile_dir}",
            f"--remote-debugging-port={int(devtools_port)}",
            "--new-window",
            "--start-fullscreen",
            "--disable-features=Translate",
            "--disable-gpu",
            "--disable-direct-composition",
            "--disable-accelerated-video-decode",
            "--autoplay-policy=no-user-gesture-required",
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _find_capture_manifest(capture_root: Path, run_id: str) -> Path:
    path = capture_root / run_id / f"{run_id}_manifest.json"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def _profile_capture(
    *,
    manifest_path: Path,
    element_id: str,
    run_id: str,
    storage_root: Path,
) -> dict[str, Any]:
    result = run_av_tool_call(
        {
            "tool": "element_creation_profile_from_capture",
            "args": {
                "manifest": str(manifest_path),
                "element_id": element_id,
                "run_id": run_id,
                "purge_teacher_state": True,
                "purge_records_jsonl": True,
                "max_frames": 180,
                "sample_stride": 1,
                "radius": 6,
            },
        },
        storage_root=storage_root,
    )
    if not result.get("ok"):
        raise RuntimeError(json.dumps(result, indent=2))
    return result["result"]


def _write_video_receipt(
    *,
    storage_root: Path,
    run_id: str,
    approved_url: str,
    resolved_url: str,
    video_title: str,
    duration_seconds: float,
    visual_state_records: int,
    profile_created: bool,
    teacher_chunks_purged: bool,
    source_time_delta_seconds: float,
    expected_sample_seconds: float,
    requested_start_seconds: float,
    source_time_before_seconds: float,
    source_time_after_seconds: float,
    source_duration_seconds: float,
    visual_motion_score: float,
) -> dict[str, Any]:
    result = run_av_tool_call(
        {
            "tool": "source_surface_video_state_receipt",
            "args": {
                "run_id": run_id,
                "approved_url": approved_url,
                "resolved_url": resolved_url,
                "video_title": video_title,
                "duration_detected_seconds": duration_seconds,
                "visual_state_records": visual_state_records,
                "not_gray_screen": visual_state_records > 0,
                "not_error_page": True,
                "profile_created": profile_created,
                "teacher_chunks_purged": teacher_chunks_purged,
                "source_time_delta_seconds": source_time_delta_seconds,
                "expected_sample_seconds": expected_sample_seconds,
                "requested_start_seconds": requested_start_seconds,
                "source_time_before_seconds": source_time_before_seconds,
                "source_time_after_seconds": source_time_after_seconds,
                "source_duration_seconds": source_duration_seconds,
                "visual_motion_score": visual_motion_score,
                "minimum_visual_motion_score": 0.001,
            },
        },
        storage_root=storage_root,
    )
    if not result.get("ok"):
        raise RuntimeError(json.dumps(result, indent=2))
    return result["result"]


def _visual_temporal_change_score(creation_signature: dict[str, Any]) -> float:
    transition = creation_signature.get("transition_behavior") or {}
    shape = creation_signature.get("shape_behavior") or {}
    growth_decay = creation_signature.get("growth_decay") or {}
    center_drift = shape.get("center_drift_xy") or [0.0, 0.0]
    try:
        drift_score = max(abs(float(center_drift[0])), abs(float(center_drift[1])))
    except (TypeError, ValueError, IndexError):
        drift_score = 0.0
    return max(
        float(transition.get("motion_mean") or 0.0),
        float(transition.get("motion_abs_mean") or 0.0),
        float(growth_decay.get("volatility") or 0.0),
        drift_score,
    )


def _purge_unverified_profile_artifacts(profile: dict[str, Any] | None) -> list[str]:
    if not profile:
        return []
    removed: list[str] = []
    for key in ("profile_json", "receipt_json"):
        value = profile.get(key)
        if not value:
            continue
        path = Path(str(value))
        try:
            if path.exists():
                path.unlink()
                removed.append(str(path))
        except OSError:
            continue
    return removed


def _collect_metadata(entries: list[dict[str, Any]], metadata_jsonl: Path, *, timeout_seconds: float) -> dict[str, dict[str, Any]]:
    metadata_by_id: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for entry in entries:
        video_id = str(entry["video_id"])
        if video_id in seen:
            continue
        seen.add(video_id)
        try:
            metadata = fetch_youtube_metadata(entry["source_url"], timeout_seconds=timeout_seconds)
            metadata_by_id[video_id] = metadata
            _append_jsonl(metadata_jsonl, {"status": "ok", **metadata})
            _console(f"metadata ok: {video_id} | {metadata['duration_seconds']}s | {metadata['video_title']}")
        except Exception as exc:  # noqa: BLE001 - batch should skip bad sources and continue.
            _append_jsonl(
                metadata_jsonl,
                {
                    "status": "failed",
                    "video_id": video_id,
                    "source_url": entry["source_url"],
                    "error": str(exc),
                },
            )
            _console(f"metadata failed: {video_id} | {exc}")
    return metadata_by_id


def execute_batch(
    queue: dict[str, Any],
    *,
    run_root: Path,
    load_wait_seconds: float,
    close_browser: bool,
    devtools_port: int,
) -> dict[str, Any]:
    storage_root = run_root / "storage"
    capture_root = run_root / "captures"
    browser_profile = run_root / "browser_profile"
    events_jsonl = run_root / "events" / "batch_events.jsonl"
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    edge_process: subprocess.Popen[Any] | None = None
    first_navigation = True
    for source in queue["sources"]:
        for sample in source["samples"]:
            sample_run_id = sample["run_id"]
            url = sample["sample_navigation_url"]
            profile: dict[str, Any] | None = None
            event_base = {
                "source_order": source["source_order"],
                "category": source["category"],
                "element_id": source["element_id"],
                "video_id": source["video_id"],
                "sample_index": sample["sample_index"],
                "run_id": sample_run_id,
                "sample_url": url,
            }
            try:
                _console(f"\nsource {source['source_order']} sample {sample['sample_index'] + 1}/{source['sample_count']}: {source['category']} -> {url}")
                _append_jsonl(events_jsonl, {"event": "sample_start", **event_base})
                if first_navigation:
                    edge_process = _start_edge(browser_profile, url, devtools_port=devtools_port)
                    first_navigation = False
                else:
                    navigate_first_page(devtools_port, url, timeout_seconds=10.0)
                time.sleep(load_wait_seconds)
                if edge_process is not None:
                    _activate_process_window(edge_process.pid)
                before_state = evaluate_on_first_page(
                    devtools_port,
                    build_video_play_expression(float(sample["start_seconds"])),
                    await_promise=True,
                    timeout_seconds=60.0,
                )
                if not isinstance(before_state, dict) or not before_state.get("ok"):
                    raise RuntimeError(f"source video failed to enter play state: {before_state}")

                capture_command = list(sample["native_capture_command"])
                output_index = capture_command.index("--output-root") + 1
                capture_command[output_index] = str(capture_root)
                _console(f"capture start: {sample_run_id}")
                capture_proc = subprocess.Popen(
                    capture_command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = capture_proc.communicate(timeout=float(sample["capture"]["duration_seconds"]) + 30.0)
                if capture_proc.returncode != 0:
                    raise RuntimeError(stderr.strip() or stdout.strip() or f"capture failed with code {capture_proc.returncode}")
                after_state = evaluate_on_first_page(
                    devtools_port,
                    build_video_state_expression(),
                    timeout_seconds=8.0,
                )
                if not isinstance(after_state, dict) or not after_state.get("ok"):
                    raise RuntimeError(f"source video state could not be read after capture: {after_state}")
                source_time_delta = float(after_state.get("currentTime") or 0.0) - float(before_state.get("currentTime") or 0.0)
                manifest_path = _find_capture_manifest(capture_root, sample_run_id)
                profile = _profile_capture(
                    manifest_path=manifest_path,
                    element_id=source["element_id"],
                    run_id=sample_run_id,
                    storage_root=storage_root,
                )
                teacher_purged = profile["purge"]["status"] == "purged"
                visual_motion_score = _visual_temporal_change_score(profile["creation_signature"])
                receipt = _write_video_receipt(
                    storage_root=storage_root,
                    run_id=sample_run_id,
                    approved_url=source["source_url"],
                    resolved_url=url,
                    video_title=source["video_title"],
                    duration_seconds=float(source["duration_seconds"]),
                    visual_state_records=int(profile["sampled_frames"]),
                    profile_created=Path(profile["profile_json"]).exists(),
                    teacher_chunks_purged=teacher_purged,
                    source_time_delta_seconds=source_time_delta,
                    expected_sample_seconds=float(sample["duration_seconds"]),
                    requested_start_seconds=float(sample["start_seconds"]),
                    source_time_before_seconds=float(before_state.get("currentTime") or 0.0),
                    source_time_after_seconds=float(after_state.get("currentTime") or 0.0),
                    source_duration_seconds=float(after_state.get("duration") or before_state.get("duration") or 0.0),
                    visual_motion_score=visual_motion_score,
                )
                remaining_chunks = sorted((capture_root / sample_run_id).glob("**/*.tvcells"))
                result = {
                    **event_base,
                    "status": "ok",
                    "profile_json": profile["profile_json"],
                    "profile_sha256": profile["profile_sha256"],
                    "profile_receipt_json": profile["receipt_json"],
                    "video_state_receipt_json": receipt["receipt_json"],
                    "sampled_frames": profile["sampled_frames"],
                    "source_time_delta_seconds": round(source_time_delta, 6),
                    "visual_motion_score": round(visual_motion_score, 6),
                    "source_video_before": before_state,
                    "source_video_after": after_state,
                    "six_one_six_windows": profile["six_one_six_windows"],
                    "purge": profile["purge"],
                    "remaining_tvcells": len(remaining_chunks),
                }
                results.append(result)
                _append_jsonl(events_jsonl, {"event": "sample_complete", **result})
                _console(f"sample ok: {sample_run_id} | frames={profile['sampled_frames']} | video_dt={source_time_delta:.3f}s | purge={profile['purge']['status']} | tvcells_left={len(remaining_chunks)}")
            except Exception as exc:  # noqa: BLE001 - keep batch moving.
                removed_unverified_profiles = _purge_unverified_profile_artifacts(profile)
                failure = {
                    **event_base,
                    "status": "failed",
                    "error": str(exc),
                    "unverified_profile_artifacts_purged": removed_unverified_profiles,
                }
                failures.append(failure)
                _append_jsonl(events_jsonl, {"event": "sample_failed", **failure})
                _console(f"sample failed: {sample_run_id} | {exc}")
    if close_browser:
        try:
            _send_keys("%{F4}", delay_ms=250)
        except Exception:
            pass
        if edge_process is not None:
            try:
                edge_process.terminate()
            except Exception:
                pass
    summary = {
        "schema_version": "truevision_learning_intake_batch_summary_v1",
        "run_id": queue["run_id"],
        "source_count": queue["source_count"],
        "planned_sample_count": queue["sample_count"],
        "completed_sample_count": len(results),
        "failed_sample_count": len(failures),
        "results": results,
        "failures": failures,
        "retention_check": {
            "remaining_tvcells": len(sorted(capture_root.glob("**/*.tvcells"))) if capture_root.exists() else 0,
            "remaining_records_jsonl": len(sorted(capture_root.glob("**/*_records.jsonl"))) if capture_root.exists() else 0,
        },
    }
    _write_json(run_root / "batch_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run approved YouTube source-surface learning intake.")
    parser.add_argument("--approved-file", default=str(ROOT / "approved youtube videos.md"))
    parser.add_argument("--run-root", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--sample-seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--grid", default="320x180")
    parser.add_argument("--region", default="0,0,2560,1440")
    parser.add_argument("--metadata-timeout", type=float, default=15.0)
    parser.add_argument("--load-wait-seconds", type=float, default=8.0)
    parser.add_argument("--devtools-port", type=int, default=9223)
    parser.add_argument("--no-close-browser", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"youtube_learning_intake_{_now_slug()}"
    run_root = Path(args.run_root) if args.run_root else Path("E:/TruEVision Generation/library/youtube_learning_intake_pilot") / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    width, height = [int(part) for part in args.resolution.lower().split("x", 1)]
    grid_w, grid_h = [int(part) for part in args.grid.lower().split("x", 1)]
    region = [int(part) for part in args.region.split(",", 3)]

    approved_text = Path(args.approved_file).read_text(encoding="utf-8")
    entries = parse_approved_youtube_sources(approved_text)
    unique_entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in entries:
        if entry["video_id"] in seen_ids:
            continue
        seen_ids.add(entry["video_id"])
        unique_entries.append(entry)
    if args.max_videos > 0:
        unique_entries = unique_entries[: args.max_videos]

    _write_json(
        run_root / "approved_sources_parsed.json",
        {
            "run_id": run_id,
            "source_count": len(unique_entries),
            "sources": unique_entries,
        },
    )
    metadata_by_id = _collect_metadata(
        unique_entries,
        run_root / "metadata" / "youtube_metadata.jsonl",
        timeout_seconds=args.metadata_timeout,
    )
    queue = build_batch_queue(
        unique_entries,
        metadata_by_video_id=metadata_by_id,
        player_region=region,
        run_id=run_id,
        sample_seconds=args.sample_seconds,
        fps=args.fps,
        resolution=[width, height],
        grid=[grid_w, grid_h],
        output_root=str(run_root / "captures"),
        native_capture_exe=str(NATIVE_CAPTURE_EXE),
    )
    _write_json(run_root / "batch_queue.json", queue)
    _console(f"\nqueue ready: sources={queue['source_count']} skipped={queue['skipped_count']} samples={queue['sample_count']}")
    _console(f"run_root={run_root}")
    if not args.execute:
        return 0
    summary = execute_batch(
        queue,
        run_root=run_root,
        load_wait_seconds=args.load_wait_seconds,
        close_browser=not args.no_close_browser,
        devtools_port=args.devtools_port,
    )
    _console(
        "\ncomplete: "
        f"completed={summary['completed_sample_count']} "
        f"failed={summary['failed_sample_count']} "
        f"remaining_tvcells={summary['retention_check']['remaining_tvcells']} "
        f"remaining_records={summary['retention_check']['remaining_records_jsonl']}"
    )
    _console(f"summary={run_root / 'batch_summary.json'}")
    return 0 if summary["failed_sample_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
