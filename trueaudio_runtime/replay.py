from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .logging import safe_slug, sha256_file, stable_json_hash, utc_now, _write_json


def _read_state_rows(state_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in state_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != "trueaudio_state_frame_v1":
            raise ValueError("state log contains non-TrueAudio frame rows")
        rows.append(row)
    if not rows:
        raise ValueError("state log is empty")
    rows.sort(key=lambda item: int(item.get("frame_index", 0)))
    return rows


def _infer_fps(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 30.0
    deltas = [
        float(rows[index + 1].get("time_seconds", 0.0)) - float(rows[index].get("time_seconds", 0.0))
        for index in range(len(rows) - 1)
    ]
    usable = [delta for delta in deltas if delta > 0]
    if not usable:
        return 30.0
    return float(1.0 / np.median(np.asarray(usable, dtype=np.float32)))


def _row_value(row: dict[str, Any], group: str, name: str, default: float = 0.0) -> float:
    return float((row.get(group) or {}).get(name, default) or default)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _synthesize_rows(rows: list[dict[str, Any]], *, sample_rate: int) -> np.ndarray:
    fps = _infer_fps(rows)
    frame_samples = max(1, int(round(sample_rate / fps)))
    carriers = {
        "bass": 82.41,
        "mid": 329.63,
        "high": 1975.53,
    }
    phases = {key: 0.0 for key in carriers}
    chunks: list[np.ndarray] = []
    for row in rows:
        level = np.clip(_row_value(row, "level", "rms_norm"), 0.0, 1.0)
        bass = np.clip(_row_value(row, "bands", "bass"), 0.0, 1.0)
        mid = np.clip(_row_value(row, "bands", "mid"), 0.0, 1.0)
        high = np.clip(_row_value(row, "bands", "high"), 0.0, 1.0)
        balance = np.clip(_row_value(row, "channels", "stereo_balance"), -1.0, 1.0)
        width = np.clip(_row_value(row, "channels", "stereo_width"), 0.0, 2.0)
        attack = np.clip(_row_value(row, "dynamics", "attack"), 0.0, 1.0)

        t = np.arange(frame_samples, dtype=np.float32) / float(sample_rate)
        mono = np.zeros(frame_samples, dtype=np.float32)
        weights = {
            "bass": 0.55 + bass * 0.75,
            "mid": 0.18 + mid * 0.55,
            "high": 0.05 + high * 0.30,
        }
        for key, freq in carriers.items():
            phase = phases[key]
            mono += weights[key] * np.sin((math.tau * freq * t) + phase).astype(np.float32)
            phases[key] = float((phase + math.tau * freq * frame_samples / sample_rate) % math.tau)

        if attack > 0.05:
            click = np.exp(-np.linspace(0.0, 9.0, frame_samples, dtype=np.float32))
            mono += click * attack * 0.25

        mono *= float(0.18 + level * 0.62)
        side = np.sin(math.tau * 713.0 * t + phases["high"]).astype(np.float32) * width * level * 0.08
        left_gain = np.clip(1.0 - balance * 0.35, 0.45, 1.55)
        right_gain = np.clip(1.0 + balance * 0.35, 0.45, 1.55)
        left = (mono * left_gain) + side
        right = (mono * right_gain) - side
        chunks.append(np.column_stack([left, right]).astype(np.float32))

    if not chunks:
        return np.zeros((0, 2), dtype=np.float32)
    samples = np.concatenate(chunks, axis=0)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0.96:
        samples = samples * (0.96 / peak)
    return samples.astype(np.float32)


def replay_trueaudio_state(
    state_path: str | Path,
    *,
    storage_root: str | Path = "storage",
    run_id: str | None = None,
    sample_rate: int = 48000,
) -> dict[str, Any]:
    """Render a deterministic sonification WAV from TrueAudio state rows.

    This is not source-audio recovery. It lets a human hear the captured level,
    band, transient, and stereo-shape state.
    """
    source = Path(state_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))
    if sample_rate < 8000 or sample_rate > 192000:
        raise ValueError("sample_rate must be between 8000 and 192000")

    rows = _read_state_rows(source)
    samples = _synthesize_rows(rows, sample_rate=sample_rate)

    root = Path(storage_root).expanduser().resolve()
    for lane in ("artifacts", "manifests", "receipts"):
        (root / lane).mkdir(parents=True, exist_ok=True)

    run = safe_slug(run_id or f"{source.stem}_state_replay")
    wav_path = root / "artifacts" / "trueaudio" / "replay" / f"{run}.wav"
    manifest_path = root / "manifests" / f"{run}_trueaudio_state_replay_manifest.json"
    receipt_path = root / "receipts" / f"{run}_trueaudio_state_replay_receipt.json"

    _write_wav(wav_path, samples, sample_rate)
    wav_sha = sha256_file(wav_path)
    state_sha = sha256_file(source)
    duration = samples.shape[0] / float(sample_rate) if sample_rate else 0.0
    manifest = {
        "schema_version": "trueaudio_state_replay_manifest_v1",
        "run_id": run,
        "created_at_utc": utc_now(),
        "system": "TrueAudio",
        "source_state": {
            "path": str(source),
            "sha256": state_sha,
            "schema": "trueaudio_state_frame_v1",
            "frame_count": len(rows),
            "inferred_fps": round(_infer_fps(rows), 6),
        },
        "output_audio": {
            "path": str(wav_path),
            "sha256": wav_sha,
            "format": "wav_pcm_s16le_stereo",
            "sample_rate": sample_rate,
            "duration_seconds": round(duration, 6),
        },
        "algorithm": {
            "name": "trueaudio_state_sonification_v1",
            "inputs": ["rms_norm", "bass", "mid", "high", "attack", "stereo_balance", "stereo_width"],
            "carriers": ["bass_sine", "mid_sine", "high_sine", "attack_click", "stereo_side_signal"],
        },
        "boundary": {
            "replay_kind": "state_sonification",
            "source_audio_recovered": False,
            "claims_original_audio": False,
            "raw_audio_required": False,
            "pcm_required": False,
            "derived_state_only": True,
        },
    }
    _write_json(manifest_path, manifest)
    manifest_sha = stable_json_hash(manifest)
    receipt = {
        "receipt_kind": "trueaudio_state_replay_receipt_v1",
        "written_at_utc": utc_now(),
        "run_id": run,
        "status": "ok",
        "source_state_sha256": state_sha,
        "wav_sha256": wav_sha,
        "manifest_sha256": manifest_sha,
        "boundary": manifest["boundary"],
    }
    _write_json(receipt_path, receipt)
    return {
        "schema_version": "trueaudio_state_replay_result_v1",
        "run_id": run,
        "wav_path": str(wav_path),
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "wav_sha256": wav_sha,
        "manifest_sha256": manifest_sha,
        "frame_count": len(rows),
        "duration_seconds": round(duration, 6),
    }
