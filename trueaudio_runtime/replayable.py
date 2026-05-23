from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .ffmpeg import decode_pcm_f32_stereo, probe_audio
from .logging import safe_slug, sha256_file, stable_json_hash, utc_now, _write_json
from .machine import capture_windows_wasapi_loopback


def _coerce_stereo(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("samples must be a 2D array")
    if array.shape[1] == 1:
        array = np.repeat(array, 2, axis=1)
    elif array.shape[1] > 2:
        array = array[:, :2]
    if array.shape[1] != 2:
        raise ValueError("samples must contain at least one channel")
    return np.clip(array.astype(np.float32), -1.0, 1.0)


def _validate_stft(frame_size: int, hop_size: int) -> None:
    if frame_size < 64 or frame_size > 65536:
        raise ValueError("frame_size must be between 64 and 65536")
    if hop_size < 1 or hop_size > frame_size:
        raise ValueError("hop_size must be between 1 and frame_size")


def _stft(samples: np.ndarray, *, frame_size: int, hop_size: int) -> tuple[np.ndarray, int]:
    _validate_stft(frame_size, hop_size)
    stereo = _coerce_stereo(samples)
    original_length = int(stereo.shape[0])
    if original_length == 0:
        return np.zeros((0, frame_size // 2 + 1, 2), dtype=np.complex64), 0
    frame_count = int(math.ceil(max(1, original_length - frame_size) / hop_size)) + 1
    padded_length = (frame_count - 1) * hop_size + frame_size
    padded = np.zeros((padded_length, 2), dtype=np.float32)
    padded[:original_length] = stereo
    window = np.sqrt(np.hanning(frame_size).astype(np.float32))
    spectra = np.zeros((frame_count, frame_size // 2 + 1, 2), dtype=np.complex64)
    for frame_index in range(frame_count):
        start = frame_index * hop_size
        frame = padded[start : start + frame_size] * window[:, None]
        spectra[frame_index, :, 0] = np.fft.rfft(frame[:, 0]).astype(np.complex64)
        spectra[frame_index, :, 1] = np.fft.rfft(frame[:, 1]).astype(np.complex64)
    return spectra, original_length


def _istft(spectra: np.ndarray, *, original_length: int, frame_size: int, hop_size: int) -> np.ndarray:
    _validate_stft(frame_size, hop_size)
    if spectra.ndim != 3 or spectra.shape[2] != 2:
        raise ValueError("spectra must have shape [frames, bins, 2]")
    if spectra.shape[0] == 0 or original_length == 0:
        return np.zeros((0, 2), dtype=np.float32)
    padded_length = (spectra.shape[0] - 1) * hop_size + frame_size
    output = np.zeros((padded_length, 2), dtype=np.float32)
    weight = np.zeros(padded_length, dtype=np.float32)
    window = np.sqrt(np.hanning(frame_size).astype(np.float32))
    for frame_index in range(spectra.shape[0]):
        start = frame_index * hop_size
        left = np.fft.irfft(spectra[frame_index, :, 0], n=frame_size).astype(np.float32)
        right = np.fft.irfft(spectra[frame_index, :, 1], n=frame_size).astype(np.float32)
        frame = np.column_stack([left, right]) * window[:, None]
        output[start : start + frame_size] += frame
        weight[start : start + frame_size] += window * window
    valid = weight > 1.0e-8
    output[valid] /= weight[valid, None]
    return np.clip(output[:original_length], -1.0, 1.0).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _summary(samples: np.ndarray) -> dict[str, Any]:
    if samples.size == 0:
        return {"max_abs": 0.0, "rms": 0.0, "sample_count": 0}
    return {
        "max_abs": round(float(np.max(np.abs(samples))), 8),
        "rms": round(float(np.sqrt(np.mean(samples * samples))), 8),
        "sample_count": int(samples.shape[0]),
    }


def write_replayable_audio_state(
    samples: np.ndarray,
    *,
    sample_rate: int,
    storage_root: str | Path = "storage",
    run_id: str | None = None,
    frame_size: int = 2048,
    hop_size: int = 512,
    capture_metadata: dict[str, Any] | None = None,
    source_kind: str = "provided_samples",
    result_schema: str = "trueaudio_replayable_state_log_result_v1",
) -> dict[str, Any]:
    if sample_rate < 8000 or sample_rate > 384000:
        raise ValueError("sample_rate must be between 8000 and 384000")
    spectra, original_length = _stft(samples, frame_size=frame_size, hop_size=hop_size)
    root = Path(storage_root).expanduser().resolve()
    for lane in ("artifacts", "manifests", "receipts"):
        (root / lane).mkdir(parents=True, exist_ok=True)

    run = safe_slug(run_id or "replayable_trueaudio")
    state_path = root / "artifacts" / "trueaudio" / "replayable" / f"{run}.trueaudio.npz"
    manifest_path = root / "manifests" / f"{run}_trueaudio_replayable_manifest.json"
    receipt_path = root / "receipts" / f"{run}_trueaudio_replayable_receipt.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "trueaudio_replayable_spectral_state_v1",
        "run_id": run,
        "created_at_utc": utc_now(),
        "source_kind": source_kind,
        "sample_rate": sample_rate,
        "channels": 2,
        "original_length": original_length,
        "frame_size": frame_size,
        "hop_size": hop_size,
        "frame_count": int(spectra.shape[0]),
        "bin_count": int(spectra.shape[1]) if spectra.ndim == 3 else 0,
        "capture": capture_metadata or {},
    }
    np.savez_compressed(
        state_path,
        spectrum=spectra,
        metadata=json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    )
    state_sha = sha256_file(state_path)
    duration = original_length / float(sample_rate) if sample_rate else 0.0
    manifest = {
        "schema_version": "trueaudio_replayable_state_manifest_v1",
        "run_id": run,
        "created_at_utc": utc_now(),
        "system": "TrueAudio",
        "state": {
            "path": str(state_path),
            "sha256": state_sha,
            "schema": metadata["schema_version"],
            "sample_rate": sample_rate,
            "channels": 2,
            "duration_seconds": round(duration, 6),
            "frame_size": frame_size,
            "hop_size": hop_size,
            "frame_count": metadata["frame_count"],
            "bin_count": metadata["bin_count"],
            "summary": _summary(_coerce_stereo(samples)),
        },
        "capture": capture_metadata or {"source_kind": source_kind},
        "boundary": {
            "system_role": "TrueAudio replayable derived audio-state system",
            "capture_scope": str((capture_metadata or {}).get("capture_scope") or source_kind),
            "capture_stage": str((capture_metadata or {}).get("capture_stage") or "derived_state"),
            "raw_audio_saved": False,
            "pcm_saved": False,
            "replayable_audio_state": True,
            "raw_waveform_state": False,
            "asr_claim": False,
            "speaker_identity_claim": False,
            "close_replay_requires_replayable_state": True,
        },
    }
    _write_json(manifest_path, manifest)
    manifest_sha = stable_json_hash(manifest)
    receipt = {
        "receipt_kind": "trueaudio_replayable_state_receipt_v1",
        "written_at_utc": utc_now(),
        "run_id": run,
        "status": "ok",
        "state_sha256": state_sha,
        "manifest_sha256": manifest_sha,
        "boundary": manifest["boundary"],
    }
    _write_json(receipt_path, receipt)
    return {
        "schema_version": result_schema,
        "run_id": run,
        "state_npz": str(state_path),
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "state_sha256": state_sha,
        "manifest_sha256": manifest_sha,
        "frame_count": metadata["frame_count"],
        "duration_seconds": round(duration, 6),
        "boundary": manifest["boundary"],
    }


def replay_replayable_audio_state(
    state_npz: str | Path,
    *,
    storage_root: str | Path = "storage",
    run_id: str | None = None,
) -> dict[str, Any]:
    source = Path(state_npz).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))
    with np.load(source, allow_pickle=False) as payload:
        spectra = payload["spectrum"].astype(np.complex64)
        metadata = json.loads(str(payload["metadata"]))
    sample_rate = int(metadata["sample_rate"])
    frame_size = int(metadata["frame_size"])
    hop_size = int(metadata["hop_size"])
    original_length = int(metadata["original_length"])
    samples = _istft(spectra, original_length=original_length, frame_size=frame_size, hop_size=hop_size)

    root = Path(storage_root).expanduser().resolve()
    for lane in ("artifacts", "manifests", "receipts"):
        (root / lane).mkdir(parents=True, exist_ok=True)
    run = safe_slug(run_id or f"{source.stem}_replay")
    wav_path = root / "artifacts" / "trueaudio" / "replay" / f"{run}.wav"
    manifest_path = root / "manifests" / f"{run}_trueaudio_replayable_replay_manifest.json"
    receipt_path = root / "receipts" / f"{run}_trueaudio_replayable_replay_receipt.json"
    _write_wav(wav_path, samples, sample_rate)
    wav_sha = sha256_file(wav_path)
    source_sha = sha256_file(source)
    duration = samples.shape[0] / float(sample_rate) if sample_rate else 0.0
    manifest = {
        "schema_version": "trueaudio_replayable_state_replay_manifest_v1",
        "run_id": run,
        "created_at_utc": utc_now(),
        "system": "TrueAudio",
        "source_state": {
            "path": str(source),
            "sha256": source_sha,
            "schema": metadata["schema_version"],
            "frame_count": int(metadata["frame_count"]),
        },
        "output_audio": {
            "path": str(wav_path),
            "sha256": wav_sha,
            "format": "wav_pcm_s16le_stereo",
            "sample_rate": sample_rate,
            "duration_seconds": round(duration, 6),
        },
        "boundary": {
            "replay_kind": "replayable_spectral_state_reconstruction",
            "raw_audio_recovered": False,
            "raw_audio_required": False,
            "pcm_required": False,
            "claims_original_source_file": False,
            "claims_close_replay_from_replayable_state": True,
        },
    }
    _write_json(manifest_path, manifest)
    manifest_sha = stable_json_hash(manifest)
    receipt = {
        "receipt_kind": "trueaudio_replayable_state_replay_receipt_v1",
        "written_at_utc": utc_now(),
        "run_id": run,
        "status": "ok",
        "source_state_sha256": source_sha,
        "wav_sha256": wav_sha,
        "manifest_sha256": manifest_sha,
        "boundary": manifest["boundary"],
    }
    _write_json(receipt_path, receipt)
    return {
        "schema_version": "trueaudio_replayable_state_replay_result_v1",
        "run_id": run,
        "wav_path": str(wav_path),
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "wav_sha256": wav_sha,
        "manifest_sha256": manifest_sha,
        "frame_count": int(metadata["frame_count"]),
        "duration_seconds": round(duration, 6),
        "samples": samples,
    }


def log_file_replayable_audio_state(
    audio_path: str | Path,
    *,
    storage_root: str | Path = "storage",
    run_id: str | None = None,
    sample_rate: int = 48000,
    max_seconds: float | None = None,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> dict[str, Any]:
    source = Path(audio_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be greater than 0 when supplied")
    source_sha = sha256_file(source)
    probe = probe_audio(source)
    samples = decode_pcm_f32_stereo(source, sample_rate=sample_rate, max_seconds=max_seconds)
    metadata = {
        "backend": "ffmpeg",
        "capture_scope": "source_audio_file",
        "capture_stage": "ffmpeg_decoded_pre_output",
        "source_audio_path": str(source),
        "source_audio_sha256": source_sha,
        "source_audio_probe": probe,
        "max_seconds": max_seconds,
    }
    result = write_replayable_audio_state(
        samples,
        sample_rate=sample_rate,
        storage_root=storage_root,
        run_id=run_id or f"{source.stem}_file_replayable",
        frame_size=frame_size,
        hop_size=hop_size,
        capture_metadata=metadata,
        source_kind="source_audio_file",
        result_schema="trueaudio_file_replayable_state_log_result_v1",
    )
    return {
        **result,
        "source_audio_sha256": source_sha,
        "source_audio_path": str(source),
        "source_duration_seconds": probe.get("duration_seconds"),
    }


def log_machine_replayable_audio_state(
    *,
    storage_root: str | Path = "storage",
    run_id: str | None = None,
    duration_seconds: float = 10.0,
    frame_size: int = 2048,
    hop_size: int = 512,
    capture_provider: Any | None = None,
) -> dict[str, Any]:
    if duration_seconds <= 0 or duration_seconds > 3600:
        raise ValueError("duration_seconds must be greater than 0 and no more than 3600")
    provider = capture_provider or capture_windows_wasapi_loopback
    samples, capture_metadata = provider(duration_seconds=duration_seconds)
    sample_rate = int(capture_metadata.get("sample_rate") or 0)
    if sample_rate < 8000 or sample_rate > 384000:
        raise ValueError("capture provider returned an invalid sample rate")
    metadata = {
        **capture_metadata,
        "capture_scope": "local_machine_output_mix",
        "capture_stage": "pre_speaker_loopback",
        "duration_requested_seconds": round(float(duration_seconds), 6),
    }
    return write_replayable_audio_state(
        samples,
        sample_rate=sample_rate,
        storage_root=storage_root,
        run_id=run_id,
        frame_size=frame_size,
        hop_size=hop_size,
        capture_metadata=metadata,
        source_kind="machine_output_mix",
        result_schema="trueaudio_machine_replayable_state_log_result_v1",
    )
