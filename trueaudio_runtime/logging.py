from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .ffmpeg import decode_pcm_f32_stereo, find_media_executable, probe_audio
from .machine import capture_windows_wasapi_loopback


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def safe_slug(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value)).strip("_")
    return clean[:96] or "trueaudio"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(encoded)


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    scale = float(np.percentile(finite, 95)) if finite.size else 1.0
    return np.clip(values / max(scale, 1.0e-8), 0.0, 1.0)


def _band_masks(sample_rate: int, window_size: int) -> dict[str, np.ndarray]:
    freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
    return {
        "bass": (freqs >= 20.0) & (freqs < 180.0),
        "mid": (freqs >= 180.0) & (freqs < 2200.0),
        "high": (freqs >= 2200.0) & (freqs < min(12000.0, sample_rate / 2.0)),
    }


def _zero_crossing_rate(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    signs = np.signbit(values)
    return float(np.count_nonzero(signs[1:] != signs[:-1]) / max(1, values.size - 1))


def _measure_rows(samples: np.ndarray, *, fps: int, sample_rate: int, silence_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    duration = samples.shape[0] / float(sample_rate) if sample_rate else 0.0
    frame_count = max(1, int(math.ceil(duration * fps))) if duration > 0 else 0
    if frame_count == 0:
        return [], {
            "duration_seconds": 0.0,
            "frame_count": 0,
            "max_rms": 0.0,
            "mean_rms": 0.0,
            "peak_abs": 0.0,
            "transient_count": 0,
            "silence_frame_count": 0,
            "stereo_balance_mean": 0.0,
        }

    raw_rms = np.zeros(frame_count, dtype=np.float32)
    raw_peak = np.zeros(frame_count, dtype=np.float32)
    raw_left = np.zeros(frame_count, dtype=np.float32)
    raw_right = np.zeros(frame_count, dtype=np.float32)
    balance = np.zeros(frame_count, dtype=np.float32)
    width = np.zeros(frame_count, dtype=np.float32)
    zero_crossing = np.zeros(frame_count, dtype=np.float32)
    raw_bands = {band: np.zeros(frame_count, dtype=np.float32) for band in ("bass", "mid", "high")}

    window_size = max(256, int(round(sample_rate / fps)))
    if window_size % 2:
        window_size += 1
    window = np.hanning(window_size).astype(np.float32)
    masks = _band_masks(sample_rate, window_size)

    segments: list[tuple[int, int]] = []
    for frame_index in range(frame_count):
        start = int(round((frame_index / fps) * sample_rate))
        end = int(round(((frame_index + 1) / fps) * sample_rate))
        end = min(max(end, start + 1), samples.shape[0])
        segment = samples[start:end]
        segments.append((start, end))
        if segment.size == 0:
            continue
        left = segment[:, 0]
        right = segment[:, 1]
        mono = (left + right) * 0.5
        raw_left[frame_index] = float(np.sqrt(np.mean(left * left)))
        raw_right[frame_index] = float(np.sqrt(np.mean(right * right)))
        raw_rms[frame_index] = float(np.sqrt(np.mean(mono * mono)))
        raw_peak[frame_index] = float(np.max(np.abs(segment)))
        denom = max(float(raw_left[frame_index] + raw_right[frame_index]), 1.0e-8)
        balance[frame_index] = float((raw_right[frame_index] - raw_left[frame_index]) / denom)
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        width[frame_index] = float(np.sqrt(np.mean(side * side)) / max(np.sqrt(np.mean(mid * mid)), 1.0e-8))
        zero_crossing[frame_index] = _zero_crossing_rate(mono)

        padded = np.zeros(window_size, dtype=np.float32)
        take = min(window_size, mono.size)
        padded[:take] = mono[:take]
        spectrum = np.abs(np.fft.rfft(padded * window))
        for band, mask in masks.items():
            raw_bands[band][frame_index] = float(np.mean(spectrum[mask])) if np.any(mask) else 0.0

    level = _normalize(raw_rms)
    bands = {band: _normalize(values) for band, values in raw_bands.items()}
    attack = np.maximum(0.0, level - np.roll(level, 1))
    decay = np.maximum(0.0, np.roll(level, 1) - level)
    attack[0] = float(level[0])
    decay[0] = 0.0
    transient = attack >= 0.22
    silence = raw_rms <= silence_threshold

    rows: list[dict[str, Any]] = []
    for frame_index, (start, end) in enumerate(segments):
        rows.append(
            {
                "schema_version": "trueaudio_state_frame_v1",
                "frame_index": frame_index,
                "time_seconds": round(frame_index / fps, 6),
                "sample_window": {"start": int(start), "end": int(end)},
                "channels": {
                    "rms_left": round(float(raw_left[frame_index]), 8),
                    "rms_right": round(float(raw_right[frame_index]), 8),
                    "peak_abs": round(float(raw_peak[frame_index]), 8),
                    "stereo_balance": round(float(balance[frame_index]), 8),
                    "stereo_width": round(float(min(width[frame_index], 4.0)), 8),
                },
                "level": {
                    "rms": round(float(raw_rms[frame_index]), 8),
                    "rms_norm": round(float(level[frame_index]), 8),
                    "dbfs": round(float(20.0 * math.log10(max(float(raw_rms[frame_index]), 1.0e-8))), 4),
                    "zero_crossing_rate": round(float(zero_crossing[frame_index]), 8),
                },
                "bands": {
                    "bass": round(float(bands["bass"][frame_index]), 8),
                    "mid": round(float(bands["mid"][frame_index]), 8),
                    "high": round(float(bands["high"][frame_index]), 8),
                },
                "dynamics": {
                    "attack": round(float(attack[frame_index]), 8),
                    "decay": round(float(decay[frame_index]), 8),
                    "transient": bool(transient[frame_index]),
                    "silence": bool(silence[frame_index]),
                },
            }
        )

    summary = {
        "duration_seconds": round(duration, 6),
        "frame_count": frame_count,
        "max_rms": round(float(np.max(raw_rms)), 8),
        "mean_rms": round(float(np.mean(raw_rms)), 8),
        "peak_abs": round(float(np.max(raw_peak)), 8),
        "transient_count": int(np.count_nonzero(transient)),
        "silence_frame_count": int(np.count_nonzero(silence)),
        "stereo_balance_mean": round(float(np.mean(balance)), 8),
    }
    return rows, summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _coerce_stereo_samples(samples: np.ndarray) -> np.ndarray:
    if samples.ndim != 2:
        raise ValueError("audio samples must be a 2D array")
    if samples.shape[1] == 1:
        return np.repeat(samples.astype(np.float32), 2, axis=1)
    if samples.shape[1] >= 2:
        return samples[:, :2].astype(np.float32)
    raise ValueError("audio samples must contain at least one channel")


def log_pre_sound_state(
    audio_path: str | Path,
    *,
    storage_root: str | Path = "storage",
    run_id: str | None = None,
    fps: int = 30,
    sample_rate: int = 48000,
    max_seconds: float | None = None,
    silence_threshold: float = 0.005,
) -> dict[str, Any]:
    """Log derived audio state from decoded PCM before playback/output.

    This captures state at the pre-output PCM stage. It does not save raw PCM
    and it does not claim ASR, speaker identity, or semantic understanding.
    """
    source = Path(audio_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))
    if fps < 1 or fps > 240:
        raise ValueError("fps must be between 1 and 240")
    if sample_rate < 8000 or sample_rate > 192000:
        raise ValueError("sample_rate must be between 8000 and 192000")

    root = Path(storage_root).expanduser().resolve()
    for lane in ("artifacts", "manifests", "receipts"):
        (root / lane).mkdir(parents=True, exist_ok=True)

    run = safe_slug(run_id or f"{source.stem}_trueaudio")
    state_path = root / "artifacts" / "trueaudio" / f"{run}_state.jsonl"
    manifest_path = root / "manifests" / f"{run}_trueaudio_manifest.json"
    receipt_path = root / "receipts" / f"{run}_trueaudio_receipt.json"

    probe = probe_audio(source)
    samples = decode_pcm_f32_stereo(source, sample_rate=sample_rate, max_seconds=max_seconds)
    rows, summary = _measure_rows(samples, fps=fps, sample_rate=sample_rate, silence_threshold=silence_threshold)

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    state_path.write_text(state_text, encoding="utf-8")
    state_sha = sha256_bytes(state_text.encode("utf-8"))

    created = utc_now()
    manifest = {
        "schema_version": "trueaudio_pre_sound_manifest_v1",
        "run_id": run,
        "created_at_utc": created,
        "system": "TrueAudio",
        "decode_stage": "decoded_pcm_pre_output",
        "decoder": {
            "name": "ffmpeg",
            "mode": "pcm_f32le_stereo",
            "ffmpeg_path": find_media_executable("ffmpeg"),
            "ffprobe_path": find_media_executable("ffprobe"),
        },
        "source_audio": {
            "path": str(source),
            "sha256": sha256_file(source),
            "probe": probe,
        },
        "state": {
            "path": str(state_path),
            "sha256": state_sha,
            "schema": "trueaudio_state_frame_v1",
            "fps": fps,
            "sample_rate": sample_rate,
            "channels": 2,
            "frame_count": len(rows),
            "summary": summary,
        },
        "boundary": {
            "system_role": "TrueAudio sibling sensor/state system",
            "not_part_of_truevision": True,
            "raw_audio_saved": False,
            "pcm_saved": False,
            "derived_state_only": True,
            "asr_claim": False,
            "speaker_identity_claim": False,
            "replayable_audio": False,
        },
    }
    _write_json(manifest_path, manifest)
    manifest_sha = stable_json_hash(manifest)

    receipt = {
        "receipt_kind": "trueaudio_pre_sound_logging_receipt_v1",
        "written_at_utc": utc_now(),
        "run_id": run,
        "status": "ok",
        "source_audio_sha256": manifest["source_audio"]["sha256"],
        "state_sha256": state_sha,
        "manifest_sha256": manifest_sha,
        "state_frame_count": len(rows),
        "boundary": manifest["boundary"],
    }
    _write_json(receipt_path, receipt)

    return {
        "schema_version": "trueaudio_pre_sound_log_result_v1",
        "run_id": run,
        "state_jsonl": str(state_path),
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "state_sha256": state_sha,
        "manifest_sha256": manifest_sha,
        "frame_count": len(rows),
        "duration_seconds": summary["duration_seconds"],
        "summary": summary,
    }


def log_machine_pre_sound_state(
    *,
    storage_root: str | Path = "storage",
    run_id: str | None = None,
    duration_seconds: float = 10.0,
    fps: int = 30,
    silence_threshold: float = 0.005,
    capture_provider: Any | None = None,
) -> dict[str, Any]:
    """Log derived state from the local machine output mix before speakers.

    The default capture provider uses Windows WASAPI loopback. Tests and other
    runtimes may inject a provider with the same signature so the logger stays
    deterministic and does not depend on active speaker output.
    """
    if duration_seconds <= 0 or duration_seconds > 3600:
        raise ValueError("duration_seconds must be greater than 0 and no more than 3600")
    if fps < 1 or fps > 240:
        raise ValueError("fps must be between 1 and 240")

    root = Path(storage_root).expanduser().resolve()
    for lane in ("artifacts", "manifests", "receipts"):
        (root / lane).mkdir(parents=True, exist_ok=True)

    provider = capture_provider or capture_windows_wasapi_loopback
    samples, capture_metadata = provider(duration_seconds=duration_seconds)
    samples = _coerce_stereo_samples(np.asarray(samples, dtype=np.float32))
    sample_rate = int(capture_metadata.get("sample_rate") or 0)
    if sample_rate < 8000 or sample_rate > 384000:
        raise ValueError("capture provider returned an invalid sample rate")

    rows, summary = _measure_rows(samples, fps=fps, sample_rate=sample_rate, silence_threshold=silence_threshold)

    run = safe_slug(run_id or "machine_trueaudio")
    state_path = root / "artifacts" / "trueaudio" / f"{run}_machine_state.jsonl"
    manifest_path = root / "manifests" / f"{run}_trueaudio_machine_manifest.json"
    receipt_path = root / "receipts" / f"{run}_trueaudio_machine_receipt.json"

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    state_path.write_text(state_text, encoding="utf-8")
    state_sha = sha256_bytes(state_text.encode("utf-8"))

    created = utc_now()
    manifest = {
        "schema_version": "trueaudio_machine_pre_sound_manifest_v1",
        "run_id": run,
        "created_at_utc": created,
        "system": "TrueAudio",
        "decode_stage": "machine_loopback_pre_output",
        "machine_capture": {
            "backend": str(capture_metadata.get("backend") or "unknown"),
            "sample_rate": sample_rate,
            "channels": int(capture_metadata.get("channels") or 2),
            "duration_requested_seconds": round(float(duration_seconds), 6),
            "duration_captured_seconds": round(samples.shape[0] / float(sample_rate), 6),
            "metadata": capture_metadata,
        },
        "state": {
            "path": str(state_path),
            "sha256": state_sha,
            "schema": "trueaudio_state_frame_v1",
            "fps": fps,
            "sample_rate": sample_rate,
            "channels": 2,
            "frame_count": len(rows),
            "summary": summary,
        },
        "boundary": {
            "system_role": "TrueAudio sibling sensor/state system",
            "capture_scope": "local_machine_output_mix",
            "capture_stage": "pre_speaker_loopback",
            "not_part_of_truevision": True,
            "raw_audio_saved": False,
            "pcm_saved": False,
            "derived_state_only": True,
            "asr_claim": False,
            "speaker_identity_claim": False,
            "replayable_audio": False,
        },
    }
    _write_json(manifest_path, manifest)
    manifest_sha = stable_json_hash(manifest)

    receipt = {
        "receipt_kind": "trueaudio_machine_pre_sound_logging_receipt_v1",
        "written_at_utc": utc_now(),
        "run_id": run,
        "status": "ok",
        "machine_capture_backend": manifest["machine_capture"]["backend"],
        "state_sha256": state_sha,
        "manifest_sha256": manifest_sha,
        "state_frame_count": len(rows),
        "boundary": manifest["boundary"],
    }
    _write_json(receipt_path, receipt)

    return {
        "schema_version": "trueaudio_machine_pre_sound_log_result_v1",
        "run_id": run,
        "state_jsonl": str(state_path),
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "state_sha256": state_sha,
        "manifest_sha256": manifest_sha,
        "frame_count": len(rows),
        "duration_seconds": summary["duration_seconds"],
        "summary": summary,
        "machine_capture": manifest["machine_capture"],
    }
