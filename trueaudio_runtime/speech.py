from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .logging import safe_slug, sha256_file, stable_json_hash, utc_now, _write_json


def _load_replayable_state(state_npz: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(state_npz, allow_pickle=False) as payload:
        spectra = payload["spectrum"].astype(np.complex64)
        metadata = json.loads(str(payload["metadata"]))
    if spectra.ndim != 3 or spectra.shape[2] != 2:
        raise ValueError("replayable TrueAudio state must have shape [frames, bins, 2]")
    if metadata.get("schema_version") != "trueaudio_replayable_spectral_state_v1":
        raise ValueError("source is not replayable TrueAudio spectral state")
    return spectra, metadata


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, 1.0e-12)


def _spectral_flatness(magnitude: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = magnitude[:, mask]
    if values.size == 0:
        return np.ones(magnitude.shape[0], dtype=np.float32)
    geometric = np.exp(np.mean(np.log(values + 1.0e-12), axis=1))
    arithmetic = np.mean(values + 1.0e-12, axis=1)
    return np.clip(geometric / arithmetic, 0.0, 1.0).astype(np.float32)


def _build_segments(frames: list[dict[str, Any]], *, threshold: float, min_segment_seconds: float) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    for frame in frames:
        if frame["speech_present"]:
            active.append(frame)
        elif active:
            start = float(active[0]["start_seconds"])
            end = float(active[-1]["end_seconds"])
            if end - start >= min_segment_seconds:
                segments.append(
                    {
                        "start_seconds": round(start, 6),
                        "end_seconds": round(end, 6),
                        "duration_seconds": round(end - start, 6),
                        "mean_confidence": round(float(np.mean([item["speech_confidence"] for item in active])), 6),
                        "max_confidence": round(float(np.max([item["speech_confidence"] for item in active])), 6),
                        "frame_count": len(active),
                    }
                )
            active = []
    if active:
        start = float(active[0]["start_seconds"])
        end = float(active[-1]["end_seconds"])
        if end - start >= min_segment_seconds:
            segments.append(
                {
                    "start_seconds": round(start, 6),
                    "end_seconds": round(end, 6),
                    "duration_seconds": round(end - start, 6),
                    "mean_confidence": round(float(np.mean([item["speech_confidence"] for item in active])), 6),
                    "max_confidence": round(float(np.max([item["speech_confidence"] for item in active])), 6),
                    "frame_count": len(active),
                }
            )
    return [segment for segment in segments if segment["mean_confidence"] >= threshold * 0.75]


def detect_speech_segments_from_replayable_state(
    state_npz: str | Path,
    *,
    storage_root: str | Path = "storage",
    run_id: str | None = None,
    speech_threshold: float = 0.48,
    min_segment_seconds: float = 0.12,
) -> dict[str, Any]:
    """Detect speech-like regions from replayable TrueAudio state.

    This is voice activity detection, not ASR. It produces no transcript and no
    speaker identity. It is intended as the first TrueSpeech In state boundary:
    audio-state -> speech/background timing.
    """
    source = Path(state_npz).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))
    if speech_threshold <= 0.0 or speech_threshold >= 1.0:
        raise ValueError("speech_threshold must be between 0 and 1")

    spectra, metadata = _load_replayable_state(source)
    sample_rate = int(metadata["sample_rate"])
    frame_size = int(metadata["frame_size"])
    hop_size = int(metadata["hop_size"])
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

    magnitude = np.mean(np.abs(spectra), axis=2)
    power = magnitude * magnitude
    total_power = np.sum(power, axis=1)
    speech_mask = (freqs >= 85.0) & (freqs <= 3800.0)
    low_mask = freqs < 85.0
    high_mask = freqs > 4500.0
    speech_power = np.sum(power[:, speech_mask], axis=1)
    low_power = np.sum(power[:, low_mask], axis=1) if np.any(low_mask) else np.zeros_like(total_power)
    high_power = np.sum(power[:, high_mask], axis=1) if np.any(high_mask) else np.zeros_like(total_power)
    speech_ratio = _safe_ratio(speech_power, total_power)
    low_ratio = _safe_ratio(low_power, total_power)
    high_ratio = _safe_ratio(high_power, total_power)
    centroid = np.sum(power * freqs[None, :], axis=1) / np.maximum(total_power, 1.0e-12)
    flatness = _spectral_flatness(magnitude, speech_mask)

    noise_floor = float(np.percentile(total_power, 20)) if total_power.size else 0.0
    energy_score = np.clip(np.log10(_safe_ratio(total_power, np.asarray(noise_floor + 1.0e-12)) + 1.0) / 1.15, 0.0, 1.0)
    band_score = np.clip((speech_ratio - 0.32) / 0.38, 0.0, 1.0)
    flatness_score = np.clip((0.68 - flatness) / 0.58, 0.0, 1.0)
    centroid_score = np.exp(-((centroid - 1200.0) / 1900.0) ** 2)
    noise_penalty = np.clip((high_ratio - 0.26) * 0.75 + np.maximum(0.0, low_ratio - 0.42) * 0.35, 0.0, 0.45)
    confidence = np.clip(
        energy_score * 0.36 + band_score * 0.34 + flatness_score * 0.22 + centroid_score * 0.08 - noise_penalty,
        0.0,
        1.0,
    )

    # One-frame holes should not split speech regions.
    present = confidence >= speech_threshold
    smoothed = present.copy()
    for index in range(1, len(present) - 1):
        if not present[index] and present[index - 1] and present[index + 1]:
            smoothed[index] = True

    frames: list[dict[str, Any]] = []
    for index in range(spectra.shape[0]):
        start = index * hop_size / sample_rate
        end = start + frame_size / sample_rate
        frames.append(
            {
                "schema_version": "truespeech_detection_frame_v1",
                "frame_index": index,
                "start_seconds": round(float(start), 6),
                "end_seconds": round(float(end), 6),
                "speech_confidence": round(float(confidence[index]), 6),
                "speech_present": bool(smoothed[index]),
                "background_confidence": round(float(1.0 - confidence[index]), 6),
                "features": {
                    "energy_score": round(float(energy_score[index]), 6),
                    "speech_band_ratio": round(float(speech_ratio[index]), 6),
                    "low_band_ratio": round(float(low_ratio[index]), 6),
                    "high_band_ratio": round(float(high_ratio[index]), 6),
                    "spectral_flatness": round(float(flatness[index]), 6),
                    "spectral_centroid_hz": round(float(centroid[index]), 3),
                },
            }
        )

    segments = _build_segments(frames, threshold=speech_threshold, min_segment_seconds=min_segment_seconds)
    root = Path(storage_root).expanduser().resolve()
    for lane in ("artifacts", "manifests", "receipts"):
        (root / lane).mkdir(parents=True, exist_ok=True)
    run = safe_slug(run_id or f"{source.stem}_truespeech_detect")
    artifact_root = root / "artifacts" / "truespeech"
    frames_path = artifact_root / f"{run}_frames.jsonl"
    segments_path = artifact_root / f"{run}_segments.json"
    manifest_path = root / "manifests" / f"{run}_truespeech_detection_manifest.json"
    receipt_path = root / "receipts" / f"{run}_truespeech_detection_receipt.json"
    artifact_root.mkdir(parents=True, exist_ok=True)
    frames_text = "".join(json.dumps(frame, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for frame in frames)
    frames_path.write_text(frames_text, encoding="utf-8")
    _write_json(segments_path, {"schema_version": "truespeech_segments_v1", "segments": segments})

    summary = {
        "frame_count": len(frames),
        "speech_frame_count": int(sum(1 for frame in frames if frame["speech_present"])),
        "background_frame_count": int(sum(1 for frame in frames if not frame["speech_present"])),
        "speech_segment_count": len(segments),
        "mean_speech_confidence": round(float(np.mean(confidence)) if confidence.size else 0.0, 6),
        "max_speech_confidence": round(float(np.max(confidence)) if confidence.size else 0.0, 6),
    }
    manifest = {
        "schema_version": "truespeech_detection_manifest_v1",
        "run_id": run,
        "created_at_utc": utc_now(),
        "system": "TrueSpeech In",
        "source_state": {
            "path": str(source),
            "sha256": sha256_file(source),
            "schema": metadata["schema_version"],
            "frame_count": int(metadata["frame_count"]),
            "sample_rate": sample_rate,
        },
        "outputs": {
            "frames_jsonl": str(frames_path),
            "segments_json": str(segments_path),
        },
        "detector": {
            "name": "truespeech_spectral_vad_v1",
            "speech_threshold": speech_threshold,
            "min_segment_seconds": min_segment_seconds,
            "uses_replayable_audio_state": True,
        },
        "summary": summary,
        "boundary": {
            "speech_detection_only": True,
            "asr_claim": False,
            "transcript_claim": False,
            "speaker_identity_claim": False,
            "text_output": False,
            "observed_audio_state_source": True,
        },
    }
    _write_json(manifest_path, manifest)
    manifest_sha = stable_json_hash(manifest)
    receipt = {
        "receipt_kind": "truespeech_detection_receipt_v1",
        "written_at_utc": utc_now(),
        "run_id": run,
        "status": "ok",
        "source_state_sha256": manifest["source_state"]["sha256"],
        "manifest_sha256": manifest_sha,
        "speech_segment_count": len(segments),
        "boundary": manifest["boundary"],
    }
    _write_json(receipt_path, receipt)

    return {
        "schema_version": "truespeech_detection_result_v1",
        "run_id": run,
        "frames_jsonl": str(frames_path),
        "segments_json": str(segments_path),
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "segments": segments,
        "summary": summary,
    }
