"""Fill missing frames from existing TrueVision cell-state captures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from truevision_state_replay import sha256_file

from .render_missing_frame import render_missing_frame
from .state_interpolator import interpolate_missing_state
from .temporal_616 import build_temporal_616_map
from .verify_replay_continuity import verify_filled_state_continuity


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def find_missing_frame_numbers(observed_frames: list[int]) -> list[int]:
    if not observed_frames:
        return []
    ordered = sorted(set(int(frame) for frame in observed_frames))
    missing: list[int] = []
    for left, right in zip(ordered, ordered[1:]):
        if right - left > 1:
            missing.extend(range(left + 1, right))
    return missing


def read_truevision_cells(run_dir: Path) -> tuple[dict[int, np.ndarray], list[str], dict[str, Any], dict[str, Any]]:
    manifest_path = next(run_dir.glob("*_manifest.json"))
    summary_path = next(run_dir.glob("*_summary.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    feature_names = list(manifest["cell_state"]["feature_names"])
    cells_by_frame: dict[int, np.ndarray] = {}
    for chunk in manifest["cell_state"]["chunks"]:
        with np.load(chunk["path"], allow_pickle=False) as data:
            cell_state = data["cell_state"].astype(np.float32)
            frame_numbers = data["frame_numbers"].astype(np.int64)
            for index, frame_number in enumerate(frame_numbers):
                cells_by_frame[int(frame_number)] = cell_state[index]
    return cells_by_frame, feature_names, manifest, summary


def fill_missing_frames(
    cells_by_frame: Mapping[int, np.ndarray],
    *,
    feature_names: list[str],
    output_shape: tuple[int, int],
    target_frames: list[int] | None = None,
    radius: int = 6,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fill requested or detected frame gaps using 6-1-6 temporal causality."""
    observed = sorted(int(frame) for frame in cells_by_frame.keys())
    targets = target_frames if target_frames is not None else find_missing_frame_numbers(observed)
    filled: dict[int, np.ndarray] = {}
    traces: list[dict[str, Any]] = []
    continuity: list[dict[str, Any]] = []

    for target in sorted(set(int(frame) for frame in targets)):
        window = build_temporal_616_map(cells_by_frame, target, radius=radius)
        cells, trace = interpolate_missing_state(cells_by_frame, window, feature_names=feature_names)
        filled[target] = cells
        prior = window.prior_frames[-1]
        future = window.future_frames[0]
        verification = verify_filled_state_continuity(
            cells_by_frame[prior],
            cells,
            cells_by_frame[future],
            feature_names=feature_names,
        )
        trace["verification"] = verification
        traces.append(trace)
        continuity.append({"target_frame": int(target), **verification})
    return filled, traces, continuity


def _write_video(path: Path, frames: list[np.ndarray], *, fps: float) -> bool:
    if not frames:
        return False
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        return False
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path.exists() and path.stat().st_size > 0


def fill_truevision_capture(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    target_frames: list[int] | None = None,
    radius: int = 6,
    fps: float | None = None,
) -> dict[str, Any]:
    """Fill gaps in a TrueVision capture directory and render a filled video."""
    cells_by_frame, feature_names, manifest, summary = read_truevision_cells(run_dir)
    output_dir = output_dir or (run_dir / "trueframegen")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_shape = tuple(int(value) for value in summary["geometry"]["frame_shape"])
    replay_fps = float(fps or summary.get("fps", {}).get("mean") or 9.0)

    filled, traces, continuity = fill_missing_frames(
        cells_by_frame,
        feature_names=feature_names,
        output_shape=frame_shape,
        target_frames=target_frames,
        radius=radius,
    )
    combined = dict(cells_by_frame)
    combined.update(filled)

    npz_path = output_dir / f"{manifest['run_id']}_trueframegen_filled_cells.npz"
    ordered_frames = sorted(combined.keys())
    np.savez_compressed(
        npz_path,
        cell_state=np.stack([combined[frame] for frame in ordered_frames]).astype(np.float32),
        frame_numbers=np.asarray(ordered_frames, dtype=np.int64),
        feature_names=np.asarray(feature_names),
        grid_shape=np.asarray(combined[ordered_frames[0]].shape[:2], dtype=np.int32),
    )

    trace_path = output_dir / f"{manifest['run_id']}_temporal_616_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, allow_nan=False) + "\n")

    frames = [
        render_missing_frame(combined[frame], feature_names=feature_names, output_shape=frame_shape, smooth=False)
        for frame in ordered_frames
    ]
    video_path = output_dir / f"{manifest['run_id']}_trueframegen_filled_video.mp4"
    video_written = _write_video(video_path, frames, fps=replay_fps)

    report_path = output_dir / f"{manifest['run_id']}_missing_frame_report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# {manifest['run_id']} TrueFrameGen Missing Frame Report",
                "",
                "## Law",
                "",
                "TrueVision records. 6-1-6 explains temporal causality. TrueFrameGen fills only missing state between known states.",
                "",
                "## Summary",
                "",
                f"- Observed frames: `{len(cells_by_frame)}`",
                f"- Filled frames: `{len(filled)}`",
                f"- Radius: `{radius}`",
                f"- Output frames: `{len(ordered_frames)}`",
                f"- Video written: `{video_written}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_manifest = {
        "schema_version": 1,
        "kind": "trueframegen_manifest",
        "created_at_utc": utc_now(),
        "source_run_id": manifest["run_id"],
        "source_manifest": str(next(run_dir.glob("*_manifest.json"))),
        "law": "TrueVision records. 6-1-6 explains temporal causality. TrueFrameGen fills only missing state between known states.",
        "radius": radius,
        "observed_frames": len(cells_by_frame),
        "filled_frames": len(filled),
        "output_frames": len(ordered_frames),
        "feature_scope": ["rgb_mean_r", "rgb_mean_g", "rgb_mean_b", "luma_mean", "edge_density", "motion_energy", "delta_luma_abs"],
        "continuity": continuity,
        "outputs": {
            "filled_video_mp4": str(video_path),
            "filled_video_written": video_written,
            "filled_video_sha256": sha256_file(video_path) if video_written else None,
            "filled_cells_npz": str(npz_path),
            "temporal_616_trace_jsonl": str(trace_path),
            "missing_frame_report_md": str(report_path),
        },
    }
    manifest_path = output_dir / f"{manifest['run_id']}_trueframegen_manifest.json"
    manifest_path.write_text(json.dumps(out_manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {"manifest_json": str(manifest_path), **out_manifest["outputs"]}

