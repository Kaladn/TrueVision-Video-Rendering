#!/usr/bin/env python3
"""Watch a playing video and record TrueVision state.

This tool is a coordinator around the existing state recorder. It does not
record raw video and it does not render. It captures cell/state telemetry, then
can optionally run state recognition so the same run has transform candidates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from truevision_runtime.av_tools.av_tool_runner import run_av_tool_call


RECORDER_PATH = PROJECT_ROOT / "scripts" / "truevision_resonance_recorder.py"
RECOGNITION_PATH = PROJECT_ROOT / "scripts" / "truevision_state_recognition.py"
DEFAULT_CAPTURE_ROOT = PROJECT_ROOT / "storage" / "artifacts" / "truevision_captures"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "storage" / "reports" / "state_recognition"
DEFAULT_RECEIPT_ROOT = PROJECT_ROOT / "storage" / "receipts" / "state_video_watcher"
DEFAULT_STOP_ROOT = PROJECT_ROOT / "storage" / "runtime" / "state_video_watcher"
DEFAULT_STORAGE_ROOT = PROJECT_ROOT / "storage"


TOOL_ID = "truevision_state_video_watcher"
WATCHER_SCHEMA = "truevision_state_video_watcher_receipt_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def safe_id(value: str | None, fallback: str = TOOL_ID) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or ""))
    clean = clean.strip("_")
    return clean or fallback


def quote_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


@dataclass(frozen=True)
class WatcherConfig:
    run_id: str
    duration_seconds: float
    fps: int
    resolution: str
    grid: str
    blocks: str
    monitor: int
    region: str
    output_root: Path
    report_root: Path
    receipt_root: Path
    storage_root: Path
    stop_file: Path
    start_delay_seconds: float
    save_cell_state: bool
    recognize: bool
    run_tool_calls: bool
    recognition_max_frames: int
    recognition_sample_stride: int
    tool_profile_max_frames: int
    tool_profile_sample_stride: int

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    @property
    def capture_manifest(self) -> Path:
        return self.run_dir / f"{self.run_id}_manifest.json"

    @property
    def receipt_path(self) -> Path:
        return self.receipt_root / f"{self.run_id}_{TOOL_ID}_receipt.json"


def build_capture_command(config: WatcherConfig, *, python_exe: str | None = None) -> list[str]:
    command = [
        python_exe or sys.executable,
        str(RECORDER_PATH),
        "--duration",
        f"{float(config.duration_seconds):.6f}",
        "--fps",
        str(int(config.fps)),
        "--resolution",
        config.resolution,
        "--grid",
        config.grid,
        "--blocks",
        config.blocks,
        "--monitor",
        str(int(config.monitor)),
        "--run-id",
        config.run_id,
        "--output-root",
        str(config.output_root),
        "--start-delay",
        f"{float(config.start_delay_seconds):.6f}",
        "--stop-file",
        str(config.stop_file),
    ]
    if config.region:
        command.extend(["--region", config.region])
    if not config.save_cell_state:
        command.append("--no-cell-state")
    return command


def build_recognition_command(config: WatcherConfig, *, python_exe: str | None = None) -> list[str]:
    return [
        python_exe or sys.executable,
        str(RECOGNITION_PATH),
        "--manifest",
        str(config.capture_manifest),
        "--output-root",
        str(config.report_root),
        "--run-id",
        f"{config.run_id}_state_recognition",
        "--max-frames",
        str(int(config.recognition_max_frames)),
        "--sample-stride",
        str(int(config.recognition_sample_stride)),
    ]


def build_post_capture_tool_calls(config: WatcherConfig) -> list[dict[str, Any]]:
    """Build AV tool calls that consume the capture manifest after recording."""
    manifest = str(config.capture_manifest)
    common = {
        "manifest": manifest,
        "manifest_json": manifest,
        "run_id": config.run_id,
        "max_frames": int(config.tool_profile_max_frames),
        "sample_stride": int(config.tool_profile_sample_stride),
    }
    return [
        {
            "tool": "meter_grid_from_capture",
            "args": {
                **common,
                "run_id": f"{config.run_id}_meter_grid",
                "section_id": f"{config.run_id}_watched_state",
                "event_type_candidate": "candidate_temporal_transform",
            },
        },
        {
            "tool": "atmosphere_profile_from_capture",
            "args": {
                **common,
                "run_id": f"{config.run_id}_atmosphere_profile",
                "element_id": "watched_atmosphere_state",
                "radius": 6,
                "max_windows": 200,
            },
        },
        {
            "tool": "element_creation_profile_from_capture",
            "args": {
                **common,
                "run_id": f"{config.run_id}_element_creation_profile",
                "element_id": "watched_temporal_transform",
                "radius": 6,
                "max_windows": 200,
                "purge_teacher_state": False,
            },
        },
    ]


def run_post_capture_tool_calls(config: WatcherConfig) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for call in build_post_capture_tool_calls(config):
        result = run_av_tool_call(call, storage_root=config.storage_root)
        results.append(
            {
                "tool_call": call,
                "ok": bool(result.get("ok")),
                "tool": result.get("tool"),
                "result": result.get("result") or {},
                "receipt": result.get("receipt"),
                "error": result.get("error", ""),
            }
        )
    return results


def build_watcher_receipt(
    config: WatcherConfig,
    *,
    status: str,
    capture_command: list[str],
    recognition_command: list[str] | None = None,
    post_capture_tool_calls: list[dict[str, Any]] | None = None,
    capture_result: dict[str, Any] | None = None,
    recognition_result: dict[str, Any] | None = None,
    post_capture_tool_results: list[dict[str, Any]] | None = None,
    return_code: int = 0,
    error: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": WATCHER_SCHEMA,
        "tool_id": TOOL_ID,
        "created_at_utc": utc_now(),
        "run_id": config.run_id,
        "status": status,
        "return_code": int(return_code),
        "paths": {
            "run_dir": str(config.run_dir),
            "capture_manifest": str(config.capture_manifest),
            "receipt": str(config.receipt_path),
            "state_recognition_report_root": str(config.report_root),
            "av_tool_storage_root": str(config.storage_root),
            "operator_stop_file": str(config.stop_file),
        },
        "tool_calls": [
            {
                "tool_id": "truevision_resonance_recorder",
                "tool_call_kind": "subprocess",
                "purpose": "state_capture",
                "command": capture_command,
                "printable": quote_command(capture_command),
                "writes": ["*_records.jsonl", "cell_state_npz/*.npz", "*_summary.json", "*_manifest.json"],
            },
            {
                "tool_id": "truevision_state_recognition",
                "tool_call_kind": "subprocess",
                "purpose": "state_event_recognition",
                "command": recognition_command or [],
                "printable": quote_command(recognition_command or []) if recognition_command else "",
                "enabled": bool(recognition_command),
                "writes": ["*_state_recognition_report.json", "*_state_recognition_summary.md", "*_state_recognition_events.csv"],
            },
            *[
                {
                    "tool_id": str(call.get("tool")),
                    "tool_call_kind": "av_tool_runner",
                    "purpose": "post_capture_state_profile",
                    "call": call,
                }
                for call in (post_capture_tool_calls or [])
            ],
        ],
        "commands": {
            "capture": capture_command,
            "capture_printable": quote_command(capture_command),
            "recognition": recognition_command or [],
            "recognition_printable": quote_command(recognition_command or []) if recognition_command else "",
        },
        "config": {
            "duration_seconds": float(config.duration_seconds),
            "fps": int(config.fps),
            "resolution": config.resolution,
            "grid": config.grid,
            "blocks": config.blocks,
            "monitor": int(config.monitor),
            "region": config.region or None,
            "start_delay_seconds": float(config.start_delay_seconds),
            "include_blocks": True,
            "save_cell_state": bool(config.save_cell_state),
            "recognize": bool(config.recognize),
            "run_tool_calls": bool(config.run_tool_calls),
            "recognition_max_frames": int(config.recognition_max_frames),
            "recognition_sample_stride": int(config.recognition_sample_stride),
            "tool_profile_max_frames": int(config.tool_profile_max_frames),
            "tool_profile_sample_stride": int(config.tool_profile_sample_stride),
        },
        "capture_result": capture_result or {},
        "recognition_result": recognition_result or {},
        "post_capture_tool_results": post_capture_tool_results or [],
        "boundary": {
            "raw_video_saved": False,
            "raw_frames_saved": False,
            "state_capture_required": True,
            "state_recognition_optional": True,
            "render_started": False,
            "animation_started": False,
            "camera_control_started": False,
            "generated_media_is_evidence": False,
            "source_truth": [".tvcells", "cell_state_npz", "*_records.jsonl", "*_manifest.json", "*_summary.json"],
        },
        "stop_instruction": f"Create this file to stop cleanly: {config.stop_file}",
        "error": error,
    }


def write_receipt(config: WatcherConfig, receipt: dict[str, Any]) -> Path:
    config.receipt_path.parent.mkdir(parents=True, exist_ok=True)
    config.receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return config.receipt_path


def _parse_json_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        return {"stdout": text}


def run_watcher(config: WatcherConfig, *, prepare_only: bool = False) -> dict[str, Any]:
    config.stop_file.parent.mkdir(parents=True, exist_ok=True)
    config.receipt_root.mkdir(parents=True, exist_ok=True)
    config.output_root.mkdir(parents=True, exist_ok=True)
    if config.stop_file.exists():
        config.stop_file.unlink()

    capture_command = build_capture_command(config)
    recognition_command = build_recognition_command(config) if config.recognize else None
    post_capture_tool_calls = build_post_capture_tool_calls(config) if config.run_tool_calls else []

    if prepare_only:
        receipt = build_watcher_receipt(
            config,
            status="prepared_not_started",
            capture_command=capture_command,
            recognition_command=recognition_command,
            post_capture_tool_calls=post_capture_tool_calls,
        )
        write_receipt(config, receipt)
        return receipt

    completed = subprocess.run(capture_command, cwd=str(PROJECT_ROOT), text=True, capture_output=True, check=False)
    capture_result = _parse_json_stdout(completed.stdout)
    if completed.returncode != 0:
        receipt = build_watcher_receipt(
            config,
            status="capture_failed",
            capture_command=capture_command,
            recognition_command=recognition_command,
            post_capture_tool_calls=post_capture_tool_calls,
            capture_result=capture_result,
            return_code=completed.returncode,
            error=completed.stderr.strip(),
        )
        write_receipt(config, receipt)
        return receipt

    post_capture_tool_results: list[dict[str, Any]] = []
    if post_capture_tool_calls:
        post_capture_tool_results = run_post_capture_tool_calls(config)

    recognition_result: dict[str, Any] = {}
    recognition_code = 0
    recognition_error = ""
    if recognition_command:
        recognition = subprocess.run(
            recognition_command,
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        recognition_code = recognition.returncode
        recognition_error = recognition.stderr.strip()
        recognition_result = _parse_json_stdout(recognition.stdout)

    status = "completed"
    return_code = 0
    error = ""
    if recognition_command and recognition_code != 0:
        status = "recognition_failed_after_capture"
        return_code = recognition_code
        error = recognition_error
    elif post_capture_tool_results and not all(bool(item.get("ok")) for item in post_capture_tool_results):
        status = "post_capture_tool_call_failed"
        return_code = 2
        error = "one or more post-capture AV tool calls failed"

    receipt = build_watcher_receipt(
        config,
        status=status,
        capture_command=capture_command,
        recognition_command=recognition_command,
        post_capture_tool_calls=post_capture_tool_calls,
        capture_result=capture_result,
        recognition_result=recognition_result,
        post_capture_tool_results=post_capture_tool_results,
        return_code=return_code,
        error=error,
    )
    write_receipt(config, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TrueVision State Video Watcher: capture state from a playing video and optionally recognize transform events."
    )
    parser.add_argument("--run-id", default="", help="Run id. Defaults to truevision_state_video_watcher_<timestamp>.")
    parser.add_argument("--duration", type=float, default=300.0, help="Maximum capture duration in seconds.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--resolution", default="960x540")
    parser.add_argument("--grid", default="160x90")
    parser.add_argument("--blocks", default="16x9")
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--region", default="", help="Optional capture region: left,top,width,height.")
    parser.add_argument("--output-root", default=str(DEFAULT_CAPTURE_ROOT))
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--receipt-root", default=str(DEFAULT_RECEIPT_ROOT))
    parser.add_argument("--storage-root", default=str(DEFAULT_STORAGE_ROOT), help="Storage root used by AV tool calls.")
    parser.add_argument("--stop-file", default="", help="Optional stop flag path. Defaults under storage/runtime.")
    parser.add_argument("--start-delay", type=float, default=2.0)
    parser.add_argument("--no-cell-state", action="store_false", dest="save_cell_state")
    parser.add_argument("--recognize", action="store_true", help="Run TrueVision state recognition after capture.")
    parser.add_argument("--no-tool-calls", action="store_false", dest="run_tool_calls", help="Skip post-capture AV tool calls.")
    parser.add_argument("--recognition-max-frames", type=int, default=1800)
    parser.add_argument("--recognition-sample-stride", type=int, default=1)
    parser.add_argument("--tool-profile-max-frames", type=int, default=240)
    parser.add_argument("--tool-profile-sample-stride", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true", help="Write receipt and commands, but do not start capture.")
    parser.set_defaults(save_cell_state=True, run_tool_calls=True)
    return parser


def config_from_args(args: argparse.Namespace) -> WatcherConfig:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = safe_id(args.run_id, fallback=f"{TOOL_ID}_{timestamp}")
    stop_file = Path(args.stop_file) if args.stop_file else DEFAULT_STOP_ROOT / f"{run_id}.stop"
    return WatcherConfig(
        run_id=run_id,
        duration_seconds=float(args.duration),
        fps=int(args.fps),
        resolution=str(args.resolution),
        grid=str(args.grid),
        blocks=str(args.blocks),
        monitor=int(args.monitor),
        region=str(args.region or ""),
        output_root=Path(args.output_root),
        report_root=Path(args.report_root),
        receipt_root=Path(args.receipt_root),
        storage_root=Path(args.storage_root),
        stop_file=stop_file,
        start_delay_seconds=float(args.start_delay),
        save_cell_state=bool(args.save_cell_state),
        recognize=bool(args.recognize),
        run_tool_calls=bool(args.run_tool_calls),
        recognition_max_frames=int(args.recognition_max_frames),
        recognition_sample_stride=int(args.recognition_sample_stride),
        tool_profile_max_frames=int(args.tool_profile_max_frames),
        tool_profile_sample_stride=int(args.tool_profile_sample_stride),
    )


def main() -> int:
    args = build_parser().parse_args()
    config = config_from_args(args)
    result = run_watcher(config, prepare_only=bool(args.prepare_only))
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(result.get("return_code") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
