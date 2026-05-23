from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from trueaudio_runtime.logging import sha256_file, stable_json_hash, utc_now, _write_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low = float(np.percentile(finite, 5))
    high = float(np.percentile(finite, 95))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _visual_times_and_pulses(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    pulses: list[float] = []
    previous_energy: float | None = None
    for record in records:
        if "elapsed_seconds" not in record:
            continue
        energy = float(record.get("screen_energy") or 0.0)
        motion = float((record.get("visual_resonance") or {}).get("motion_energy") or 0.0)
        delta = 0.0 if previous_energy is None else max(0.0, energy - previous_energy)
        previous_energy = energy
        times.append(float(record["elapsed_seconds"]))
        pulses.append(energy * 0.52 + motion * 0.28 + delta * 0.20)
    return np.asarray(times, dtype=np.float64), _normalize(np.asarray(pulses, dtype=np.float64))


def _audio_times_and_pulses(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    pulses: list[float] = []
    for row in rows:
        level = row.get("level") or {}
        dynamics = row.get("dynamics") or {}
        bands = row.get("bands") or {}
        rms = float(level.get("rms_norm") or 0.0)
        attack = float(dynamics.get("attack") or 0.0)
        transient = 1.0 if dynamics.get("transient") else 0.0
        band_pressure = (
            float(bands.get("bass") or 0.0) * 0.42
            + float(bands.get("mid") or 0.0) * 0.38
            + float(bands.get("high") or 0.0) * 0.20
        )
        times.append(float(row.get("time_seconds") or 0.0))
        pulses.append(rms * 0.44 + attack * 0.24 + transient * 0.12 + band_pressure * 0.20)
    return np.asarray(times, dtype=np.float64), _normalize(np.asarray(pulses, dtype=np.float64))


def _median_step(times: np.ndarray, fallback: float) -> float:
    if times.size < 2:
        return fallback
    diffs = np.diff(np.sort(times))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return fallback
    return float(np.median(diffs))


def _score_offset(
    visual_times: np.ndarray,
    visual_pulse: np.ndarray,
    audio_times: np.ndarray,
    audio_pulse: np.ndarray,
    offset: float,
) -> float:
    start = max(float(visual_times[0]), float(audio_times[0] - offset))
    end = min(float(visual_times[-1]), float(audio_times[-1] - offset))
    if end <= start:
        return -1.0
    step = max(0.001, min(_median_step(visual_times, 1.0 / 30.0), _median_step(audio_times, 1.0 / 30.0)))
    grid = np.arange(start, end, step, dtype=np.float64)
    if grid.size < 3:
        return -1.0
    v = np.interp(grid, visual_times, visual_pulse)
    a = np.interp(grid + offset, audio_times, audio_pulse)
    v = v - float(np.mean(v))
    a = a - float(np.mean(a))
    denom = float(np.sqrt(np.sum(v * v)) * np.sqrt(np.sum(a * a)))
    if denom <= 1.0e-12:
        return -1.0
    return float(np.sum(v * a) / denom)


def estimate_audio_visual_offset(
    visual_times: np.ndarray,
    visual_pulse: np.ndarray,
    audio_times: np.ndarray,
    audio_pulse: np.ndarray,
    *,
    max_search_seconds: float = 0.5,
) -> dict[str, float]:
    if visual_times.size < 2 or audio_times.size < 2:
        return {"audio_minus_visual_offset_seconds": 0.0, "correlation": 0.0}
    step = max(0.001, min(_median_step(visual_times, 1.0 / 30.0), _median_step(audio_times, 1.0 / 30.0)))
    search_steps = max(1, int(round(max_search_seconds / step)))
    best_offset = 0.0
    best_score = -1.0
    for index in range(-search_steps, search_steps + 1):
        offset = index * step
        score = _score_offset(visual_times, visual_pulse, audio_times, audio_pulse, offset)
        if score > best_score:
            best_score = score
            best_offset = offset
    return {
        "audio_minus_visual_offset_seconds": round(float(best_offset), 6),
        "correlation": round(float(max(best_score, 0.0)), 6),
    }


def _nearest_index(sorted_times: np.ndarray, target: float) -> int:
    index = int(np.searchsorted(sorted_times, target, side="left"))
    if index <= 0:
        return 0
    if index >= sorted_times.size:
        return sorted_times.size - 1
    before = index - 1
    return before if abs(sorted_times[before] - target) <= abs(sorted_times[index] - target) else index


def write_temporal_pulse_bridge(
    *,
    visual_records_jsonl: str | Path,
    audio_state_jsonl: str | Path,
    storage_root: str | Path = "storage",
    run_id: str = "trueav_joint",
    max_allowed_offset_ms: float = 20.0,
    max_allowed_pair_delta_ms: float | None = None,
) -> dict[str, Any]:
    visual_path = Path(visual_records_jsonl).expanduser().resolve()
    audio_path = Path(audio_state_jsonl).expanduser().resolve()
    if not visual_path.exists():
        raise FileNotFoundError(str(visual_path))
    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))

    visual_records = _read_jsonl(visual_path)
    audio_rows = _read_jsonl(audio_path)
    visual_times, visual_pulse = _visual_times_and_pulses(visual_records)
    audio_times, audio_pulse = _audio_times_and_pulses(audio_rows)
    if visual_times.size < 2 or audio_times.size < 2:
        raise ValueError("temporal pulse bridge requires at least two visual and audio frames")

    estimate = estimate_audio_visual_offset(visual_times, visual_pulse, audio_times, audio_pulse)
    offset = float(estimate["audio_minus_visual_offset_seconds"])
    visual_step = _median_step(visual_times, 1.0 / 30.0)
    audio_step = _median_step(audio_times, 1.0 / 30.0)
    allowed_pair_delta = (
        float(max_allowed_pair_delta_ms)
        if max_allowed_pair_delta_ms is not None
        else min(visual_step, audio_step) * 1000.0 * 0.75
    )

    bridge_rows: list[dict[str, Any]] = []
    pair_deltas: list[float] = []
    for audio_index, audio_time in enumerate(audio_times):
        aligned_visual_time = float(audio_time - offset)
        visual_index = _nearest_index(visual_times, aligned_visual_time)
        pair_delta_ms = float((visual_times[visual_index] - aligned_visual_time) * 1000.0)
        pair_deltas.append(abs(pair_delta_ms))
        bridge_rows.append(
            {
                "schema_version": "trueav_temporal_pulse_frame_v1",
                "audio_frame_index": int(audio_index),
                "audio_time_seconds": round(float(audio_time), 6),
                "visual_frame_index": int(visual_records[visual_index].get("frame_number", visual_index)),
                "visual_time_seconds": round(float(visual_times[visual_index]), 6),
                "aligned_visual_time_seconds": round(aligned_visual_time, 6),
                "pair_delta_ms": round(pair_delta_ms, 6),
                "audio_pulse": round(float(audio_pulse[audio_index]), 6),
                "visual_pulse": round(float(visual_pulse[visual_index]), 6),
            }
        )

    max_pair_delta = max(pair_deltas) if pair_deltas else math.inf
    mean_pair_delta = sum(pair_deltas) / max(1, len(pair_deltas))
    offset_ms = abs(offset) * 1000.0
    status = "pass" if offset_ms <= max_allowed_offset_ms and max_pair_delta <= allowed_pair_delta else "fail"

    root = Path(storage_root).expanduser().resolve()
    bridge_root = root / "artifacts" / "trueav_sync"
    bridge_root.mkdir(parents=True, exist_ok=True)
    for lane in ("manifests", "receipts"):
        (root / lane).mkdir(parents=True, exist_ok=True)
    run = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in run_id).strip("_") or "trueav_joint"
    bridge_path = bridge_root / f"{run}_temporal_pulse_bridge.jsonl"
    manifest_path = root / "manifests" / f"{run}_trueav_temporal_pulse_manifest.json"
    receipt_path = root / "receipts" / f"{run}_trueav_temporal_pulse_receipt.json"

    bridge_text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in bridge_rows)
    bridge_path.write_text(bridge_text, encoding="utf-8")
    summary = {
        "status": status,
        "audio_minus_visual_offset_ms": round(offset * 1000.0, 6),
        "correlation": estimate["correlation"],
        "max_abs_pair_delta_ms": round(float(max_pair_delta), 6),
        "mean_abs_pair_delta_ms": round(float(mean_pair_delta), 6),
        "max_allowed_offset_ms": round(float(max_allowed_offset_ms), 6),
        "max_allowed_pair_delta_ms": round(float(allowed_pair_delta), 6),
        "visual_frame_count": int(visual_times.size),
        "audio_frame_count": int(audio_times.size),
    }
    manifest = {
        "schema_version": "trueav_temporal_pulse_manifest_v1",
        "run_id": run,
        "created_at_utc": utc_now(),
        "system": "TrueVision+TrueAudio",
        "visual_source": {"path": str(visual_path), "sha256": sha256_file(visual_path)},
        "audio_source": {"path": str(audio_path), "sha256": sha256_file(audio_path)},
        "outputs": {"temporal_pulse_bridge_jsonl": str(bridge_path)},
        "summary": summary,
        "boundary": {
            "sync_receipt_only": True,
            "raw_video_saved": False,
            "raw_audio_saved": False,
            "temporal_causality_claim": "measured_offset_and_pairing_only",
            "zero_tolerance_enforced_by_manifest_status": True,
        },
    }
    _write_json(manifest_path, manifest)
    receipt = {
        "receipt_kind": "trueav_temporal_pulse_receipt_v1",
        "written_at_utc": utc_now(),
        "run_id": run,
        "status": status,
        "manifest_sha256": stable_json_hash(manifest),
        "bridge_sha256": sha256_file(bridge_path),
        "summary": summary,
    }
    _write_json(receipt_path, receipt)
    return {
        "schema_version": "trueav_temporal_pulse_result_v1",
        "run_id": run,
        "status": status,
        "bridge_jsonl": str(bridge_path),
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "summary": summary,
    }
