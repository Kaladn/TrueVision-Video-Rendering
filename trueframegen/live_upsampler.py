"""Live TrueFrameGen upsampling from native TrueVision chunk files."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from truevision_state_replay import build_rgb_replay_frame, read_native_cell_chunk, sha256_file

from .frame_upsampler import SourceStateSequence, _interpolate_state_at_time, _start_ffmpeg_writer, utc_now


NATIVE_TRUEVISION_FEATURE_NAMES = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "hsv_mean_h",
    "hsv_mean_s",
    "hsv_mean_v",
    "luma_mean",
    "luma_std",
    "saturation_mean",
    "delta_luma_abs",
    "edge_density",
    "texture_energy",
    "motion_energy",
]


def _native_chunk_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "cell_state_native").glob("*.tvcells"))


def _load_native_chunks(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    chunks: list[np.ndarray] = []
    numbers: list[np.ndarray] = []
    paths = _native_chunk_paths(run_dir)
    for path in paths:
        cell_state, frame_numbers = read_native_cell_chunk(path)
        chunks.append(cell_state.astype(np.float32, copy=False))
        numbers.append(frame_numbers.astype(np.int64, copy=False))
    if not chunks:
        raise ValueError(f"no native TrueVision chunks found in {run_dir / 'cell_state_native'}")
    cells = np.concatenate(chunks, axis=0)
    frame_numbers = np.concatenate(numbers, axis=0)
    order = np.argsort(frame_numbers)
    return cells[order], frame_numbers[order], paths


def load_live_native_sequence(
    run_dir: Path,
    *,
    frame_shape: tuple[int, int],
    capture_fps: float,
    feature_names: list[str] | tuple[str, ...] = NATIVE_TRUEVISION_FEATURE_NAMES,
    min_frames: int = 2,
    timeout_seconds: float = 0.0,
    poll_seconds: float = 0.25,
) -> SourceStateSequence:
    """Load native chunk state before the capture manifest exists.

    The native recorder writes `.tvcells` chunks during capture and writes the
    final manifest only when capture closes. This reader treats chunk files as
    the live contract, so TrueFrameGen can trail capture instead of waiting.
    """
    if capture_fps <= 0:
        raise ValueError("capture_fps must be positive")
    if min_frames <= 0:
        raise ValueError("min_frames must be positive")
    started = time.monotonic()
    last_error: Exception | None = None
    while True:
        try:
            cells, frame_numbers, paths = _load_native_chunks(run_dir)
            if cells.shape[0] >= min_frames:
                times = frame_numbers.astype(np.float64) / float(capture_fps)
                times = np.maximum.accumulate(times)
                run_id = run_dir.name
                feature_list = list(feature_names)
                summary = {
                    "kind": "truevision_live_native_summary",
                    "run_id": run_id,
                    "frame_count": int(cells.shape[0]),
                    "duration_seconds": round(float(times[-1]) if times.size else 0.0, 6),
                    "geometry": {
                        "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],
                        "grid_shape": [int(cells.shape[1]), int(cells.shape[2])],
                    },
                }
                manifest = {
                    "schema_version": 1,
                    "kind": "truevision_live_native_chunk_manifest",
                    "run_id": run_id,
                    "config": {
                        "capture_fps": float(capture_fps),
                        "capture_resolution": [int(frame_shape[1]), int(frame_shape[0])],
                    },
                    "cell_state": {
                        "format": "tvcells_f32le_v1",
                        "feature_names": feature_list,
                        "chunks": [
                            {
                                "path": str(path),
                                "format": "tvcells_f32le_v1",
                            }
                            for path in paths
                        ],
                    },
                    "boundary": {
                        "raw_frame_saved": False,
                        "final_manifest_required_for_live_read": False,
                    },
                }
                return SourceStateSequence(
                    cells=cells,
                    frame_numbers=frame_numbers,
                    times_seconds=times,
                    feature_names=feature_list,
                    manifest=manifest,
                    summary=summary,
                )
        except Exception as exc:  # keep waiting while the recorder creates the first full chunk
            last_error = exc
        if timeout_seconds <= 0 or time.monotonic() - started >= timeout_seconds:
            if last_error is not None:
                raise last_error
            raise TimeoutError(f"native chunks did not reach {min_frames} frames in {timeout_seconds:.1f}s")
        time.sleep(poll_seconds)


def _wait_for_time_coverage(
    run_dir: Path,
    *,
    frame_shape: tuple[int, int],
    capture_fps: float,
    feature_names: list[str],
    required_time: float,
    timeout_seconds: float,
    poll_seconds: float,
) -> SourceStateSequence:
    started = time.monotonic()
    min_frames = max(2, int(required_time * capture_fps) + 1)
    while True:
        sequence = load_live_native_sequence(
            run_dir,
            frame_shape=frame_shape,
            capture_fps=capture_fps,
            feature_names=feature_names,
            min_frames=min_frames,
            timeout_seconds=min(timeout_seconds, max(0.0, timeout_seconds - (time.monotonic() - started))),
            poll_seconds=poll_seconds,
        )
        if float(sequence.times_seconds[-1]) >= required_time:
            return sequence
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                f"native chunks reached {sequence.times_seconds[-1]:.3f}s, "
                f"needed {required_time:.3f}s in {run_dir}"
            )
        time.sleep(poll_seconds)


def live_upsample_truevision_native_capture(
    run_dir: Path,
    *,
    output_dir: Path,
    frame_shape: tuple[int, int],
    capture_fps: float,
    duration_seconds: float,
    target_fps: float = 60.0,
    radius: int = 6,
    crf: int = 18,
    feature_names: list[str] | tuple[str, ...] = NATIVE_TRUEVISION_FEATURE_NAMES,
    trailing_source_frames: int = 2,
    wait_timeout_seconds: float = 180.0,
    poll_seconds: float = 0.25,
) -> dict[str, Any]:
    """Render a final 60fps TFG MP4 while native capture is still writing chunks."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if capture_fps <= 0:
        raise ValueError("capture_fps must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_dir.name
    target_label = int(round(target_fps))
    video_path = output_dir / f"{run_id}_trueframegen_live_{target_label}fps.mp4"
    trace_path = output_dir / f"{run_id}_trueframegen_live_{target_label}fps_trace.jsonl"
    report_path = output_dir / f"{run_id}_trueframegen_live_{target_label}fps_report.md"
    manifest_path = output_dir / f"{run_id}_trueframegen_live_{target_label}fps_manifest.json"
    output_frame_count = max(1, int(round(duration_seconds * target_fps)))
    trailing_seconds = max(0.0, float(trailing_source_frames) / float(capture_fps))
    feature_list = list(feature_names)
    sequence = _wait_for_time_coverage(
        run_dir,
        frame_shape=frame_shape,
        capture_fps=capture_fps,
        feature_names=feature_list,
        required_time=0.0,
        timeout_seconds=wait_timeout_seconds,
        poll_seconds=poll_seconds,
    )
    channel_min = np.nanmin(sequence.cells, axis=(0, 1, 2)).astype(np.float32)
    channel_max = np.nanmax(sequence.cells, axis=(0, 1, 2)).astype(np.float32)

    proc = _start_ffmpeg_writer(video_path, width=int(frame_shape[1]), height=int(frame_shape[0]), fps=target_fps, crf=crf)
    trace_every = max(1, int(round(target_fps)))
    trace_samples: list[dict[str, Any]] = []
    loaded_chunks = 0
    started = time.monotonic()
    try:
        assert proc.stdin is not None
        with trace_path.open("w", encoding="utf-8") as trace_handle:
            for output_index in range(output_frame_count):
                target_time = float(output_index) / float(target_fps)
                required_time = min(
                    max(0.0, duration_seconds - (1.0 / capture_fps)),
                    target_time + trailing_seconds,
                )
                if float(sequence.times_seconds[-1]) < required_time:
                    sequence = _wait_for_time_coverage(
                        run_dir,
                        frame_shape=frame_shape,
                        capture_fps=capture_fps,
                        feature_names=feature_list,
                        required_time=required_time,
                        timeout_seconds=wait_timeout_seconds,
                        poll_seconds=poll_seconds,
                    )
                    channel_min = np.nanmin(sequence.cells, axis=(0, 1, 2)).astype(np.float32)
                    channel_max = np.nanmax(sequence.cells, axis=(0, 1, 2)).astype(np.float32)
                loaded_chunks = max(loaded_chunks, len(sequence.manifest["cell_state"]["chunks"]))
                state, trace = _interpolate_state_at_time(
                    sequence,
                    target_time=target_time,
                    channel_min=channel_min,
                    channel_max=channel_max,
                )
                frame = build_rgb_replay_frame(state, feature_names=sequence.feature_names, output_shape=frame_shape)
                proc.stdin.write(frame.tobytes())
                if output_index % trace_every == 0 or output_index == output_frame_count - 1:
                    row = {
                        "output_frame": int(output_index),
                        "target_time_seconds": round(target_time, 6),
                        "required_source_time_seconds": round(required_time, 6),
                        "loaded_source_frames": int(sequence.cells.shape[0]),
                        "loaded_chunks": int(loaded_chunks),
                        "target_fps": float(target_fps),
                        "radius": int(radius),
                        "live_chunk_reader": True,
                        "not_appended": True,
                        **trace,
                    }
                    trace_handle.write(json.dumps(row, allow_nan=False) + "\n")
                    trace_samples.append(row)
    finally:
        if proc.stdin:
            proc.stdin.close()
    stderr = proc.stderr.read() if proc.stderr else b""
    if proc.stdout:
        proc.stdout.read()
        proc.stdout.close()
    if proc.stderr:
        proc.stderr.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))

    elapsed = time.monotonic() - started
    video_sha = sha256_file(video_path) if video_path.exists() else None
    manifest = {
        "schema_version": 1,
        "kind": "trueframegen_live_upsample_manifest",
        "created_at_utc": utc_now(),
        "source_run_id": run_id,
        "law": "TrueVision records chunks. TrueFrameGen trails live chunks and writes in-between state inside the original timeline.",
        "source": {
            "run_dir": str(run_dir),
            "capture_fps": float(capture_fps),
            "duration_seconds": round(float(duration_seconds), 6),
            "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],
            "feature_names": feature_list,
        },
        "live_upsample": {
            "target_fps": float(target_fps),
            "output_frames": int(output_frame_count),
            "trailing_source_frames": int(trailing_source_frames),
            "loaded_chunks": int(loaded_chunks),
            "elapsed_generation_seconds": round(float(elapsed), 6),
            "timeline_rule": "generate_in_between_frames_inside_source_duration_not_append_at_end",
            "state_dump_written": False,
            "final_capture_manifest_required": False,
        },
        "outputs": {
            "video_mp4": str(video_path),
            "video_sha256": video_sha,
            "trace_jsonl": str(trace_path),
            "report_md": str(report_path),
        },
        "trace_samples": trace_samples[:8],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                f"# {run_id} Live TrueFrameGen {target_label}fps Report",
                "",
                "## Claim",
                "",
                "TrueFrameGen generated a final MP4 while reading native TrueVision chunks before the final capture manifest.",
                "",
                "## Boundary",
                "",
                "This is temporal state interpolation from captured TrueVision state, not recovered raw video.",
                "",
                "## Timing",
                "",
                f"- Capture FPS: `{capture_fps}`",
                f"- Target FPS: `{target_fps}`",
                f"- Duration: `{duration_seconds:.6f}s`",
                f"- Output frames: `{output_frame_count}`",
                f"- Generation wall time: `{elapsed:.3f}s`",
                "",
                "## Rule",
                "",
                "`generate_in_between_frames_inside_source_duration_not_append_at_end`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest_json": str(manifest_path),
        "video_mp4": str(video_path),
        "video_sha256": video_sha,
        "trace_jsonl": str(trace_path),
        "report_md": str(report_path),
    }
