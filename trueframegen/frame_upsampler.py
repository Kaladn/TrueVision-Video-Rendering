"""Temporal in-between frame generation for TrueVision captures."""

from __future__ import annotations

import json
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from truevision_state_replay import build_rgb_replay_frame, read_cell_chunk, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class SourceStateSequence:
    cells: np.ndarray
    frame_numbers: np.ndarray
    times_seconds: np.ndarray
    feature_names: list[str]
    manifest: dict[str, Any]
    summary: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_source_sequence(run_dir: Path, *, max_source_frames: int | None = None) -> SourceStateSequence:
    manifest_path = next(run_dir.glob("*_manifest.json"))
    summary_path = next(run_dir.glob("*_summary.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records_paths = list(run_dir.glob("*_records.jsonl"))
    records = _read_jsonl(records_paths[0]) if records_paths else []
    record_times = {
        int(record.get("frame_number")): float(record.get("elapsed_seconds"))
        for record in records
        if record.get("frame_number") is not None and record.get("elapsed_seconds") is not None
    }

    chunks: list[np.ndarray] = []
    numbers: list[np.ndarray] = []
    remaining = max_source_frames
    for chunk in manifest["cell_state"]["chunks"]:
        cell_state, frame_numbers = read_cell_chunk(chunk)
        if remaining is not None:
            if remaining <= 0:
                break
            cell_state = cell_state[:remaining]
            frame_numbers = frame_numbers[:remaining]
            remaining -= int(cell_state.shape[0])
        chunks.append(cell_state.astype(np.float32, copy=False))
        numbers.append(frame_numbers.astype(np.int64, copy=False))
    if not chunks:
        raise ValueError(f"no cell chunks found in {run_dir}")

    cells = np.concatenate(chunks, axis=0)
    frame_numbers = np.concatenate(numbers, axis=0)
    order = np.argsort(frame_numbers)
    cells = cells[order]
    frame_numbers = frame_numbers[order]

    duration = float(summary.get("duration_seconds") or 0.0)
    fallback_fps = float(manifest.get("config", {}).get("capture_fps") or 9.0)
    if record_times:
        times = np.asarray(
            [record_times.get(int(frame), float(index) / fallback_fps) for index, frame in enumerate(frame_numbers)],
            dtype=np.float32,
        )
    else:
        times = np.arange(frame_numbers.size, dtype=np.float32) / fallback_fps
    if duration > 0.0 and times.size > 1 and float(times[-1]) <= 0.0:
        times = np.linspace(0.0, duration, num=times.size, endpoint=False, dtype=np.float32)
    times = np.maximum.accumulate(times)

    return SourceStateSequence(
        cells=cells,
        frame_numbers=frame_numbers,
        times_seconds=times,
        feature_names=list(manifest["cell_state"]["feature_names"]),
        manifest=manifest,
        summary=summary,
    )


def _load_run_metadata(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = next(run_dir.glob("*_manifest.json"))
    summary_path = next(run_dir.glob("*_summary.json"))
    return json.loads(manifest_path.read_text(encoding="utf-8")), json.loads(summary_path.read_text(encoding="utf-8"))


def _catmull_rom(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: float) -> np.ndarray:
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


def _interpolate_state_at_time(
    sequence: SourceStateSequence,
    *,
    target_time: float,
    channel_min: np.ndarray,
    channel_max: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    times = sequence.times_seconds
    cells = sequence.cells
    if target_time <= float(times[0]):
        return cells[0].copy(), {"mode": "left_edge", "anchors": [int(sequence.frame_numbers[0])], "alpha": 0.0}
    if target_time >= float(times[-1]):
        return cells[-1].copy(), {"mode": "right_edge", "anchors": [int(sequence.frame_numbers[-1])], "alpha": 1.0}

    right = int(np.searchsorted(times, target_time, side="right"))
    left = max(0, right - 1)
    right = min(right, cells.shape[0] - 1)
    t0 = float(times[left])
    t1 = float(times[right])
    alpha = 0.0 if t1 <= t0 else (float(target_time) - t0) / (t1 - t0)

    p0 = cells[max(0, left - 1)]
    p1 = cells[left]
    p2 = cells[right]
    p3 = cells[min(cells.shape[0] - 1, right + 1)]
    if left == right:
        out = p1.copy()
    else:
        out = _catmull_rom(p0, p1, p2, p3, alpha)
    out = np.clip(out, channel_min[None, None, :], channel_max[None, None, :]).astype(np.float32)
    return out, {
        "mode": "in_between_temporal_state",
        "anchors": [int(sequence.frame_numbers[left]), int(sequence.frame_numbers[right])],
        "anchor_times": [round(t0, 6), round(t1, 6)],
        "alpha": round(float(alpha), 6),
    }


def _start_ffmpeg_writer(path: Path, *, width: int, height: int, fps: float, crf: int) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class ChunkFrameCache:
    """Small LRU cache for source chunks during streaming TFG renders."""

    def __init__(self, chunks: list[dict[str, Any]], *, max_cached_chunks: int = 3):
        if not chunks:
            raise ValueError("at least one chunk is required")
        if max_cached_chunks <= 0:
            raise ValueError("max_cached_chunks must be positive")
        self.chunks = chunks
        self.max_cached_chunks = max_cached_chunks
        self.ranges: list[tuple[int, int]] = []
        start = 0
        for chunk in chunks:
            count = int(chunk.get("frames") or chunk.get("frame_count") or 0)
            if count <= 0:
                # Fall back to opening an unusual chunk once to discover count.
                cell_state, _ = read_cell_chunk(chunk)
                count = int(cell_state.shape[0])
            self.ranges.append((start, start + count))
            start += count
        self.source_frame_count = start
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.loaded_chunk_count = 0
        self.peak_cached_chunks = 0

    def _chunk_index_for_source_index(self, source_index: int) -> int:
        if source_index < 0 or source_index >= self.source_frame_count:
            raise IndexError(f"source index out of range: {source_index}")
        for index, (start, end) in enumerate(self.ranges):
            if start <= source_index < end:
                return index
        raise IndexError(f"source index out of chunk ranges: {source_index}")

    def _load_chunk(self, chunk_index: int) -> tuple[np.ndarray, np.ndarray]:
        if chunk_index in self.cache:
            value = self.cache.pop(chunk_index)
            self.cache[chunk_index] = value
            return value
        cell_state, frame_numbers = read_cell_chunk(self.chunks[chunk_index])
        value = (cell_state.astype(np.float32, copy=False), frame_numbers.astype(np.int64, copy=False))
        self.cache[chunk_index] = value
        self.loaded_chunk_count += 1
        while len(self.cache) > self.max_cached_chunks:
            self.cache.popitem(last=False)
        self.peak_cached_chunks = max(self.peak_cached_chunks, len(self.cache))
        return value

    def source_frame(self, source_index: int) -> tuple[np.ndarray, int]:
        chunk_index = self._chunk_index_for_source_index(source_index)
        start, _ = self.ranges[chunk_index]
        local_index = source_index - start
        cell_state, frame_numbers = self._load_chunk(chunk_index)
        return cell_state[local_index], int(frame_numbers[local_index])


def _interpolate_stream_state(
    cache: ChunkFrameCache,
    *,
    target_time: float,
    capture_fps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    last_index = cache.source_frame_count - 1
    source_position = max(0.0, min(float(target_time) * float(capture_fps), float(last_index)))
    left = int(np.floor(source_position))
    right = min(last_index, left + 1)
    alpha = float(source_position - left)
    p0, n0 = cache.source_frame(max(0, left - 1))
    p1, n1 = cache.source_frame(left)
    p2, n2 = cache.source_frame(right)
    p3, n3 = cache.source_frame(min(last_index, right + 1))
    if left == right:
        out = p1.copy()
    else:
        out = _catmull_rom(p0, p1, p2, p3, alpha)
    local_min = np.minimum(np.minimum(p0, p1), np.minimum(p2, p3))
    local_max = np.maximum(np.maximum(p0, p1), np.maximum(p2, p3))
    out = np.clip(out, local_min, local_max).astype(np.float32)
    return out, {
        "mode": "streamed_in_between_temporal_state",
        "anchors": [n1, n2],
        "support_frames": [n0, n1, n2, n3],
        "source_position": round(float(source_position), 6),
        "alpha": round(alpha, 6),
    }


def stream_upsample_truevision_capture(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    target_fps: float = 60.0,
    max_seconds: float | None = None,
    radius: int = 6,
    crf: int = 18,
    max_cached_chunks: int = 3,
) -> dict[str, Any]:
    """Memory-bounded TFG upsample that reads only a small chunk window."""
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    manifest, summary = _load_run_metadata(run_dir)
    chunks = list(manifest["cell_state"]["chunks"])
    feature_names = list(manifest["cell_state"]["feature_names"])
    cache = ChunkFrameCache(chunks, max_cached_chunks=max_cached_chunks)
    output_dir = output_dir or (run_dir / "trueframegen_streamed")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_shape = tuple(int(value) for value in summary["geometry"]["frame_shape"])
    height, width = frame_shape
    capture_fps = float(manifest.get("config", {}).get("capture_fps") or 9.0)
    source_duration = float(summary.get("duration_seconds") or (cache.source_frame_count / capture_fps))
    duration = min(source_duration, float(max_seconds)) if max_seconds is not None else source_duration
    output_frame_count = max(1, int(round(duration * target_fps)))

    run_id = str(manifest["run_id"])
    target_label = int(round(target_fps))
    video_path = output_dir / f"{run_id}_trueframegen_stream_{target_label}fps.mp4"
    trace_path = output_dir / f"{run_id}_trueframegen_stream_{target_label}fps_trace.jsonl"
    report_path = output_dir / f"{run_id}_trueframegen_stream_{target_label}fps_report.md"
    manifest_path = output_dir / f"{run_id}_trueframegen_stream_{target_label}fps_manifest.json"

    proc = _start_ffmpeg_writer(video_path, width=width, height=height, fps=target_fps, crf=crf)
    trace_every = max(1, int(round(target_fps)))
    trace_samples: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    try:
        assert proc.stdin is not None
        with trace_path.open("w", encoding="utf-8") as trace_handle:
            for output_index in range(output_frame_count):
                target_time = float(output_index) / target_fps
                state, trace = _interpolate_stream_state(cache, target_time=target_time, capture_fps=capture_fps)
                frame = build_rgb_replay_frame(state, feature_names=feature_names, output_shape=frame_shape)
                proc.stdin.write(frame.tobytes())
                if output_index % trace_every == 0 or output_index == output_frame_count - 1:
                    row = {
                        "output_frame": int(output_index),
                        "target_time_seconds": round(target_time, 6),
                        "target_fps": float(target_fps),
                        "radius": int(radius),
                        "streaming_chunk_reader": True,
                        "cached_chunks": int(len(cache.cache)),
                        "peak_cached_chunks": int(cache.peak_cached_chunks),
                        "not_appended": True,
                        **trace,
                    }
                    trace_handle.write(json.dumps(row, allow_nan=False) + "\n")
                    trace_handle.flush()
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

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    video_sha = sha256_file(video_path) if video_path.exists() else None
    out_manifest = {
        "schema_version": 1,
        "kind": "trueframegen_stream_upsample_manifest",
        "created_at_utc": utc_now(),
        "source_run_id": run_id,
        "law": "TrueVision records. TrueFrameGen streams in-between state without loading the full capture.",
        "source": {
            "run_dir": str(run_dir),
            "source_frames": int(cache.source_frame_count),
            "source_duration_seconds": round(source_duration, 6),
            "source_grid_shape": [int(summary["geometry"]["grid_shape"][0]), int(summary["geometry"]["grid_shape"][1])],
            "feature_names": feature_names,
        },
        "upsample": {
            "target_fps": float(target_fps),
            "duration_seconds": round(duration, 6),
            "output_frames": int(output_frame_count),
            "timeline_rule": "generate_in_between_frames_inside_source_duration_not_append_at_end",
            "interpolation": "streamed_catmull_rom_temporal_state_clipped_to_local_support_range",
            "radius": int(radius),
            "state_dump_written": False,
        },
        "streaming": {
            "max_cached_chunks": int(max_cached_chunks),
            "peak_cached_chunks": int(cache.peak_cached_chunks),
            "chunks_loaded": int(cache.loaded_chunk_count),
            "elapsed_generation_seconds": round(elapsed, 6),
        },
        "outputs": {
            "video_mp4": str(video_path),
            "video_sha256": video_sha,
            "trace_jsonl": str(trace_path),
            "report_md": str(report_path),
        },
        "trace_samples": trace_samples[:8],
    }
    manifest_path.write_text(json.dumps(out_manifest, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                f"# {run_id} Streaming TrueFrameGen {target_label}fps Report",
                "",
                "## Claim",
                "",
                "TrueFrameGen generated in-between state with a bounded chunk cache.",
                "",
                "## Boundary",
                "",
                "This is temporal state interpolation from captured TrueVision state, not recovered raw video.",
                "",
                "## Timing",
                "",
                f"- Source frames: `{cache.source_frame_count}`",
                f"- Source duration: `{source_duration:.6f}s`",
                f"- Output duration: `{duration:.6f}s`",
                f"- Target FPS: `{target_fps}`",
                f"- Output frames: `{output_frame_count}`",
                f"- Generation wall time: `{elapsed:.3f}s`",
                "",
                "## Memory",
                "",
                f"- Max cached chunks: `{max_cached_chunks}`",
                f"- Peak cached chunks: `{cache.peak_cached_chunks}`",
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


def upsample_truevision_capture(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    target_fps: float = 60.0,
    max_seconds: float | None = None,
    radius: int = 6,
    crf: int = 18,
    max_source_frames: int | None = None,
) -> dict[str, Any]:
    """Generate in-between TrueVision frames at target_fps without appending time."""
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    sequence = _load_source_sequence(run_dir, max_source_frames=max_source_frames)
    output_dir = output_dir or (run_dir / "trueframegen_upsampled")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_shape = tuple(int(value) for value in sequence.summary["geometry"]["frame_shape"])
    height, width = frame_shape
    source_duration = float(sequence.summary.get("duration_seconds") or float(sequence.times_seconds[-1]))
    duration = min(source_duration, float(max_seconds)) if max_seconds is not None else source_duration
    output_frame_count = max(1, int(round(duration * target_fps)))
    channel_min = np.nanmin(sequence.cells, axis=(0, 1, 2)).astype(np.float32)
    channel_max = np.nanmax(sequence.cells, axis=(0, 1, 2)).astype(np.float32)

    run_id = str(sequence.manifest["run_id"])
    video_path = output_dir / f"{run_id}_trueframegen_{int(round(target_fps))}fps.mp4"
    trace_path = output_dir / f"{run_id}_trueframegen_{int(round(target_fps))}fps_trace.jsonl"
    report_path = output_dir / f"{run_id}_trueframegen_{int(round(target_fps))}fps_report.md"
    manifest_path = output_dir / f"{run_id}_trueframegen_{int(round(target_fps))}fps_manifest.json"

    proc = _start_ffmpeg_writer(video_path, width=width, height=height, fps=target_fps, crf=crf)
    trace_every = max(1, int(round(target_fps)))
    trace_samples: list[dict[str, Any]] = []
    try:
        assert proc.stdin is not None
        with trace_path.open("w", encoding="utf-8") as trace_handle:
            for output_index in range(output_frame_count):
                target_time = float(output_index) / target_fps
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
                        "target_fps": float(target_fps),
                        "radius": int(radius),
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

    video_sha = sha256_file(video_path) if video_path.exists() else None
    manifest = {
        "schema_version": 1,
        "kind": "trueframegen_upsample_manifest",
        "created_at_utc": utc_now(),
        "source_run_id": run_id,
        "law": "TrueVision records. TrueFrameGen generates in-between state within the original timeline.",
        "source": {
            "run_dir": str(run_dir),
            "source_frames": int(sequence.cells.shape[0]),
            "source_duration_seconds": round(source_duration, 6),
            "source_grid_shape": [int(sequence.cells.shape[1]), int(sequence.cells.shape[2])],
            "feature_names": sequence.feature_names,
        },
        "upsample": {
            "target_fps": float(target_fps),
            "duration_seconds": round(duration, 6),
            "output_frames": int(output_frame_count),
            "timeline_rule": "generate_in_between_frames_inside_source_duration_not_append_at_end",
            "interpolation": "catmull_rom_temporal_state_clipped_to_observed_channel_range",
            "radius": int(radius),
            "state_dump_written": False,
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
                f"# {run_id} TrueFrameGen 60fps Upsample Report",
                "",
                "## Claim",
                "",
                "TrueFrameGen generated in-between state inside the original capture timeline.",
                "",
                "## Boundary",
                "",
                "This is temporal state interpolation from captured TrueVision state, not recovered raw video.",
                "",
                "## Timing",
                "",
                f"- Source frames: `{sequence.cells.shape[0]}`",
                f"- Source duration: `{source_duration:.6f}s`",
                f"- Output duration: `{duration:.6f}s`",
                f"- Target FPS: `{target_fps}`",
                f"- Output frames: `{output_frame_count}`",
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
