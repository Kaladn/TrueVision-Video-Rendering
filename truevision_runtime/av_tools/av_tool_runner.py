from __future__ import annotations

import json
import subprocess
import wave
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from trueaudio_runtime.logging import log_machine_pre_sound_state, log_pre_sound_state
from trueaudio_runtime.lyrics import align_lyrics_to_speech_segments
from trueaudio_runtime.replay import replay_trueaudio_state
from trueaudio_runtime.replayable import (
    log_file_replayable_audio_state,
    log_machine_replayable_audio_state,
    replay_replayable_audio_state,
)
from trueaudio_runtime.speech import detect_speech_segments_from_replayable_state
from truevision_runtime.state_patterns.audio_video_patterns import choose_patterns_for_signal
from truevision_runtime.studio.studio_tooling import (
    build_studio_tool_plan,
    get_render_preset,
    list_render_presets,
    list_studio_tools,
    preset_to_template,
    save_render_preset,
)

from .av_recalibration import append_recalibration_event, list_recalibration_events
from .av_tool_policy import AVToolPolicyError, safe_flat_json_name, validate_tool_call
from .av_tool_receipts import stable_hash, utc_now, write_tool_receipt


def _ensure_storage(storage_root: Path) -> None:
    for lane in ["artifacts", "events", "library", "manifests", "receipts", "reports", "templates"]:
        path = storage_root / lane
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")


def _template_path(storage_root: Path, name: str) -> Path:
    return storage_root / "templates" / safe_flat_json_name(name)


def _read_template(storage_root: Path, name: str) -> dict[str, Any]:
    path = _template_path(storage_root, name)
    if not path.exists():
        raise FileNotFoundError(path.name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("template file must contain an object")
    return payload


def _write_template(storage_root: Path, name: str, template: dict[str, Any]) -> dict[str, Any]:
    path = _template_path(storage_root, name)
    path.write_text(json.dumps(template, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "name": path.name,
        "path": str(path),
        "sha256": stable_hash(template),
        "template": template,
    }


def _set_json_path(payload: dict[str, Any], json_path: str, value: Any) -> dict[str, Any]:
    if not json_path or any(part in {"", ".."} for part in json_path.split(".")):
        raise ValueError("json_path must be a dot-separated object path")
    target = payload
    parts = json_path.split(".")
    for part in parts[:-1]:
        existing = target.get(part)
        if not isinstance(existing, dict):
            existing = {}
            target[part] = existing
        target = existing
    target[parts[-1]] = value
    return payload


def _probe_duration(path: str) -> float | None:
    if not path:
        return None
    media_path = Path(path)
    if not media_path.exists():
        return None
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return round(float(completed.stdout.strip()), 6)
    except (subprocess.SubprocessError, ValueError):
        try:
            with wave.open(str(media_path), "rb") as handle:
                return round(handle.getnframes() / float(handle.getframerate()), 6)
        except (wave.Error, OSError, ZeroDivisionError):
            return None


def _read_wav_mono(path: Path, *, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        if max_seconds is not None:
            frame_count = min(frame_count, max(1, int(sample_rate * max_seconds)))
        raw = handle.readframes(frame_count)
    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported WAV sample width")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.clip(samples.astype(np.float32), -1.0, 1.0), sample_rate


def _decode_audio_mono_ffmpeg(path: Path, *, sample_rate: int, max_seconds: float | None = None) -> np.ndarray:
    command = ["ffmpeg", "-v", "error", "-i", str(path)]
    if max_seconds is not None:
        command.extend(["-t", f"{max_seconds:.6f}"])
    command.extend(["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate), "-"])
    completed = subprocess.run(command, check=True, capture_output=True, timeout=120)
    if not completed.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.clip(np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32), -1.0, 1.0)


def _normalize_series(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    scale = float(np.percentile(values[np.isfinite(values)], 95)) if np.any(np.isfinite(values)) else 1.0
    return np.clip(values / max(scale, 1.0e-6), 0.0, 1.0)


def _extract_wav_features(path: Path, *, fps: int, max_seconds: float | None, max_feature_frames: int) -> dict[str, Any]:
    samples, sample_rate = _read_wav_mono(path, max_seconds=max_seconds)
    duration = samples.size / float(sample_rate) if sample_rate else 0.0
    frame_count = max(1, int(np.ceil(duration * fps))) if duration > 0 else 0
    window_size = min(samples.size, max(256, int(sample_rate * 0.08)))
    if window_size % 2:
        window_size += 1
    half = window_size // 2
    window = np.hanning(window_size).astype(np.float32)
    freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
    masks = {
        "bass": (freqs >= 20.0) & (freqs < 180.0),
        "mid": (freqs >= 180.0) & (freqs < 2200.0),
        "high": (freqs >= 2200.0) & (freqs < min(12000.0, sample_rate / 2.0)),
    }
    raw = {key: np.zeros(frame_count, dtype=np.float32) for key in ["rms", "bass", "mid", "high"]}
    for frame_index in range(frame_count):
        center = int(round((frame_index / fps) * sample_rate))
        start = center - half
        end = start + window_size
        segment = np.zeros(window_size, dtype=np.float32)
        src_start = max(0, start)
        src_end = min(samples.size, end)
        if src_end > src_start:
            dst_start = src_start - start
            segment[dst_start : dst_start + (src_end - src_start)] = samples[src_start:src_end]
        segment *= window
        raw["rms"][frame_index] = float(np.sqrt(np.mean(segment * segment)))
        spectrum = np.abs(np.fft.rfft(segment))
        for band in ["bass", "mid", "high"]:
            mask = masks[band]
            raw[band][frame_index] = float(np.mean(spectrum[mask])) if np.any(mask) else 0.0
    norm = {key: _normalize_series(value) for key, value in raw.items()}
    onset = np.maximum(0.0, norm["rms"] - np.roll(norm["rms"], 1)) if frame_count else np.zeros(0, dtype=np.float32)
    if frame_count:
        onset[0] = norm["rms"][0]
    beat = _normalize_series(onset)
    features = []
    for frame_index in range(min(frame_count, max_feature_frames)):
        features.append(
            {
                "frame_index": frame_index,
                "time_seconds": round(frame_index / fps, 6),
                "rms": round(float(norm["rms"][frame_index]), 6),
                "bass": round(float(norm["bass"][frame_index]), 6),
                "mid": round(float(norm["mid"][frame_index]), 6),
                "high": round(float(norm["high"][frame_index]), 6),
                "beat": round(float(beat[frame_index]), 6),
            }
        )
    return {
        "audio_path": str(path),
        "duration_seconds": round(duration, 6),
        "sample_rate": sample_rate,
        "fps": fps,
        "feature_count": frame_count,
        "features_returned": len(features),
        "features": features,
        "summary": {
            "max_rms": round(float(np.max(norm["rms"])) if frame_count else 0.0, 6),
            "max_bass": round(float(np.max(norm["bass"])) if frame_count else 0.0, 6),
            "max_mid": round(float(np.max(norm["mid"])) if frame_count else 0.0, 6),
            "max_high": round(float(np.max(norm["high"])) if frame_count else 0.0, 6),
        },
    }


def _write_audio_features(storage_root: Path, args: dict[str, Any]) -> dict[str, Any]:
    audio_path = Path(str(args.get("path") or args.get("audio_path") or ""))
    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))
    if audio_path.suffix.lower() != ".wav":
        raise ValueError("audio_extract_features currently accepts WAV files")
    fps = int(args.get("fps") or 12)
    if fps < 1 or fps > 120:
        raise ValueError("fps must be between 1 and 120")
    payload = _extract_wav_features(
        audio_path,
        fps=fps,
        max_seconds=float(args["max_seconds"]) if args.get("max_seconds") not in {None, ""} else None,
        max_feature_frames=int(args.get("max_feature_frames") or 600),
    )
    artifacts = storage_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in audio_path.stem).strip("_") or "audio"
    path = artifacts / f"{safe_stem}_features_{utc_now().replace(':', '').replace('.', '_')}.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "feature_path": str(path),
        "feature_sha256": stable_hash(payload),
        "duration_seconds": payload["duration_seconds"],
        "sample_rate": payload["sample_rate"],
        "fps": payload["fps"],
        "feature_count": payload["feature_count"],
        "features_returned": payload["features_returned"],
        "summary": payload["summary"],
    }


def _classify_section(average_level: float, peak_count: int, valley_count: int) -> str:
    if average_level >= 0.72 or peak_count >= 3:
        return "high_energy"
    if average_level <= 0.22 or valley_count >= 3:
        return "quiet_valley"
    if average_level >= 0.48:
        return "building"
    return "steady"


def _extract_audio_level_signals(
    audio_path: Path,
    *,
    fps: int,
    sample_rate: int,
    max_seconds: float | None,
    section_seconds: float,
    peak_threshold: float,
    valley_threshold: float,
    max_signal_frames: int,
) -> dict[str, Any]:
    samples = _decode_audio_mono_ffmpeg(audio_path, sample_rate=sample_rate, max_seconds=max_seconds)
    duration = samples.size / float(sample_rate) if sample_rate else 0.0
    frame_count = max(1, int(np.ceil(duration * fps))) if duration > 0 else 0
    rms = np.zeros(frame_count, dtype=np.float32)
    peak_abs = np.zeros(frame_count, dtype=np.float32)
    for frame_index in range(frame_count):
        start = int(round((frame_index / fps) * sample_rate))
        end = int(round(((frame_index + 1) / fps) * sample_rate))
        segment = samples[start:min(end, samples.size)]
        if segment.size:
            rms[frame_index] = float(np.sqrt(np.mean(segment * segment)))
            peak_abs[frame_index] = float(np.max(np.abs(segment)))
    level = _normalize_series(rms)
    delta = level - np.roll(level, 1) if frame_count else np.zeros(0, dtype=np.float32)
    if frame_count:
        delta[0] = 0.0
    dbfs = 20.0 * np.log10(np.maximum(rms, 1.0e-6))

    events: list[dict[str, Any]] = []
    signal_frames: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        previous_level = float(level[frame_index - 1]) if frame_index > 0 else float(level[frame_index])
        next_level = float(level[frame_index + 1]) if frame_index + 1 < frame_count else float(level[frame_index])
        current_level = float(level[frame_index])
        is_peak = current_level >= peak_threshold and current_level >= previous_level and current_level >= next_level
        is_valley = current_level <= valley_threshold and current_level <= previous_level and current_level <= next_level
        event_kind = "peak" if is_peak else "valley" if is_valley else "rising" if float(delta[frame_index]) > 0.12 else "falling" if float(delta[frame_index]) < -0.12 else "steady"
        if event_kind in {"peak", "valley", "rising", "falling"}:
            events.append(
                {
                    "event_kind": event_kind,
                    "frame_index": frame_index,
                    "time_seconds": round(frame_index / fps, 6),
                    "level": round(current_level, 6),
                    "delta": round(float(delta[frame_index]), 6),
                    "state_trigger": {
                        "peak": "pulse_or_flash",
                        "valley": "slow_drift_or_hold",
                        "rising": "expand_geometry",
                        "falling": "contract_geometry",
                    }[event_kind],
                }
            )
        if len(signal_frames) < max_signal_frames:
            signal_frames.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": round(frame_index / fps, 6),
                    "level": round(current_level, 6),
                    "peak_abs": round(float(peak_abs[frame_index]), 6),
                    "dbfs": round(float(dbfs[frame_index]), 3),
                    "delta": round(float(delta[frame_index]), 6),
                    "event_kind": event_kind,
                }
            )

    frames_per_section = max(1, int(round(section_seconds * fps)))
    sections: list[dict[str, Any]] = []
    for start in range(0, frame_count, frames_per_section):
        end = min(frame_count, start + frames_per_section)
        section_levels = level[start:end]
        section_events = [event for event in events if start <= int(event["frame_index"]) < end]
        peak_count = sum(1 for event in section_events if event["event_kind"] == "peak")
        valley_count = sum(1 for event in section_events if event["event_kind"] == "valley")
        average_level = float(np.mean(section_levels)) if section_levels.size else 0.0
        sections.append(
            {
                "section_index": len(sections),
                "start_seconds": round(start / fps, 6),
                "end_seconds": round(end / fps, 6),
                "average_level": round(average_level, 6),
                "peak_count": peak_count,
                "valley_count": valley_count,
                "section_energy": _classify_section(average_level, peak_count, valley_count),
            }
        )

    summary = {
        "duration_seconds": round(duration, 6),
        "fps": fps,
        "frame_count": frame_count,
        "average_level": round(float(np.mean(level)) if frame_count else 0.0, 6),
        "max_level": round(float(np.max(level)) if frame_count else 0.0, 6),
        "peak_count": sum(1 for event in events if event["event_kind"] == "peak"),
        "valley_count": sum(1 for event in events if event["event_kind"] == "valley"),
        "rising_count": sum(1 for event in events if event["event_kind"] == "rising"),
        "falling_count": sum(1 for event in events if event["event_kind"] == "falling"),
    }
    return {
        "signal_kind": "ffmpeg_audio_level_signals_v1",
        "audio_path": str(audio_path),
        "ffmpeg_observer": True,
        "sample_rate": sample_rate,
        "summary": summary,
        "signals_returned": len(signal_frames),
        "signal_frames": signal_frames,
        "events": events[: max_signal_frames * 2],
        "sections": sections,
        "state_mappings": {
            "peak": ["pulse_rings", "random_geometry_shards", "flash_pressure"],
            "valley": ["quiet_valley_drift", "slow_motion"],
            "rising": ["rising_energy_expansion", "camera_push"],
            "falling": ["geometry_contraction", "color_cooling"],
        },
        "recommended_patterns": choose_patterns_for_signal(summary),
    }


def _write_audio_level_signals(storage_root: Path, args: dict[str, Any]) -> dict[str, Any]:
    audio_path = Path(str(args.get("path") or args.get("audio_path") or ""))
    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))
    fps = int(args.get("fps") or 12)
    if fps < 1 or fps > 120:
        raise ValueError("fps must be between 1 and 120")
    payload = _extract_audio_level_signals(
        audio_path,
        fps=fps,
        sample_rate=int(args.get("sample_rate") or 44100),
        max_seconds=float(args["max_seconds"]) if args.get("max_seconds") not in {None, ""} else None,
        section_seconds=float(args.get("section_seconds") or 8.0),
        peak_threshold=float(args.get("peak_threshold") or 0.72),
        valley_threshold=float(args.get("valley_threshold") or 0.18),
        max_signal_frames=int(args.get("max_signal_frames") or 900),
    )
    artifacts = storage_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in audio_path.stem).strip("_") or "audio"
    path = artifacts / f"{safe_stem}_level_signals_{utc_now().replace(':', '').replace('.', '_')}.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "signal_path": str(path),
        "signal_sha256": stable_hash(payload),
        "summary": payload["summary"],
        "signals_returned": payload["signals_returned"],
        "event_count": len(payload["events"]),
        "section_count": len(payload["sections"]),
        "recommended_patterns": payload["recommended_patterns"],
    }


def _template_from_audio_signals(storage_root: Path, args: dict[str, Any]) -> dict[str, Any]:
    signal_path = Path(str(args.get("signal_path") or ""))
    if not signal_path.exists():
        raise FileNotFoundError(str(signal_path))
    signal_payload = json.loads(signal_path.read_text(encoding="utf-8"))
    summary = signal_payload.get("summary") or {}
    duration = float(args.get("duration_seconds") or summary.get("duration_seconds") or 60.0)
    fps = int(args.get("fps") or summary.get("fps") or 12)
    patterns = choose_patterns_for_signal(summary)
    template = _create_template(
        {
            "name": str(args.get("name") or "audio geometry template"),
            "renderer": str(args.get("renderer") or "audio_geometry_field"),
            "prompt": str(args.get("prompt") or "audio levels drive geometry"),
            "audio_path": signal_payload.get("audio_path", ""),
            "audio_duration_seconds": duration,
            "duration_seconds": duration,
            "duration_source": "ffmpeg_audio_signal",
            "fps": fps,
            "visual_parameters": {
                "pattern_library": patterns,
                "state_mappings": signal_payload.get("state_mappings", {}),
                "sections": signal_payload.get("sections", []),
                "event_preview": signal_payload.get("events", [])[:120],
                "geometry_mode": str(args.get("geometry_mode") or "deterministic_random_geometry"),
            },
        }
    )
    template["signal_source"] = {
        "signal_path": str(signal_path),
        "signal_sha256": stable_hash(signal_payload),
        "signal_kind": signal_payload.get("signal_kind"),
    }
    return _write_template(storage_root, str(args.get("name") or f"{template['name']}.json"), template)


def _create_template(args: dict[str, Any]) -> dict[str, Any]:
    fps = int(args.get("fps") or 30)
    duration = float(args.get("duration_seconds") or args.get("audio_duration_seconds") or 60)
    frame_count = max(1, int(round(duration * fps)))
    return {
        "schema_version": 1,
        "name": str(args.get("name") or "TrueVision AV template")[:120],
        "renderer": str(args.get("renderer") or "edge_audio_river"),
        "prompt": str(args.get("prompt") or ""),
        "media": {
            "audio_path": str(args.get("audio_path") or ""),
            "audio_duration_seconds": args.get("audio_duration_seconds"),
            "sync_to_audio": bool(args.get("sync_to_audio", True)),
        },
        "timeline": {
            "duration_seconds": round(duration, 6),
            "fps": fps,
            "frame_count": frame_count,
            "start_seconds": 0,
            "end_seconds": round(duration, 6),
        },
        "time_distance": {
            "source": str(args.get("duration_source") or "manual_duration"),
            "seconds_per_frame": round(1 / fps, 9),
            "frames_per_second": fps,
            "total_frames": frame_count,
        },
        "visual_parameters": deepcopy(args.get("visual_parameters") or {}),
        "state_plan": deepcopy(args.get("state_plan") or {}),
        "boundary": {
            "synthetic_state_media": True,
            "evidence": False,
            "audio_video_only": True,
            "renderer_executes_validated_state": True,
        },
    }


def _write_manifest(storage_root: Path, prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    manifests = storage_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    safe_prefix = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in prefix).strip("_") or "manifest"
    manifest = {
        "manifest_kind": "truevision_av_manifest_v1",
        "written_at_utc": now,
        **payload,
    }
    path = manifests / f"{now.replace(':', '').replace('.', '_')}_{safe_prefix}.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {"name": path.name, "path": str(path), "sha256": stable_hash(manifest), "manifest": manifest}


def _prepare_render_job(storage_root: Path, args: dict[str, Any], *, preview: bool) -> dict[str, Any]:
    template = deepcopy(args.get("template") or {})
    if not template and args.get("template_name"):
        template = _read_template(storage_root, str(args["template_name"]))
    if not template:
        template = _create_template(args)
    timeline = template.get("timeline", {})
    fps = int(args.get("fps") or timeline.get("fps") or 30)
    duration = float(args.get("seconds") or args.get("duration_seconds") or timeline.get("duration_seconds") or 5)
    if preview:
        duration = min(duration, float(args.get("max_preview_seconds") or 15))
    frame_count = max(1, int(round(duration * fps)))
    job = {
        "job_kind": "preview_render" if preview else "full_render",
        "job_id": str(args.get("job_id") or f"{'preview' if preview else 'full'}_{utc_now().replace(':', '').replace('.', '_')}"),
        "status": "prepared_preview" if preview else "prepared_requires_human_execute",
        "renderer": template.get("renderer", "edge_audio_river"),
        "timeline": {
            "duration_seconds": round(duration, 6),
            "fps": fps,
            "frame_count": frame_count,
        },
        "template_hash": stable_hash(template),
        "template": template,
        "boundary": {
            "synthetic_state_media": True,
            "evidence": False,
            "execute_requires_human_confirmation": not preview,
        },
    }
    manifest = _write_manifest(storage_root, f"{job['job_kind']}_{job['job_id']}", {"job": job})
    return {"job": job, "manifest": manifest}


def _list_media_artifacts(storage_root: Path, lane: str | None = None) -> list[dict[str, Any]]:
    lanes = [lane] if lane else ["artifacts", "manifests", "reports", "templates", "receipts", "events"]
    files: list[dict[str, Any]] = []
    for item_lane in lanes:
        root = storage_root / item_lane
        if not root.exists():
            continue
        for path in root.glob("*"):
            if path.is_file() and path.name != ".gitkeep":
                files.append({"name": path.name, "lane": item_lane, "path": str(path), "size_bytes": path.stat().st_size})
    return sorted(files, key=lambda item: item["name"])


def _browse_manifests(storage_root: Path, *, limit: int = 50) -> dict[str, Any]:
    candidates: list[Path] = []
    manifest_root = storage_root / "manifests"
    if manifest_root.exists():
        candidates.extend(path for path in manifest_root.glob("*.json") if path.is_file())
    output_root = Path("outputs")
    if output_root.exists():
        candidates.extend(path for path in output_root.glob("*/*_manifest.json") if path.is_file())
    items: list[dict[str, Any]] = []
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        summary: dict[str, Any] = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                summary = {
                    "schema": payload.get("schema") or payload.get("manifest_kind"),
                    "run_id": payload.get("run_id") or payload.get("payload", {}).get("run_id"),
                    "duration_seconds": (payload.get("render") or {}).get("duration_seconds"),
                    "frame_count": (payload.get("render") or {}).get("frame_count"),
                    "wall_seconds": (payload.get("machine") or {}).get("wall_seconds"),
                }
        except (OSError, json.JSONDecodeError):
            summary = {"read_error": True}
        items.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "modified_at_utc": utc_now(),
                "summary": summary,
            }
        )
    return {"manifests": items, "count": len(items)}


def _render_preset_library(storage_root: Path, args: dict[str, Any]) -> dict[str, Any]:
    action = str(args.get("action") or "list")
    if action == "list":
        return {"presets": list_render_presets(storage_root), "tools": list_studio_tools()}
    if action == "load":
        return {"preset": get_render_preset(str(args.get("preset_id") or ""), storage_root)}
    if action == "save":
        preset = args.get("preset")
        if not isinstance(preset, dict):
            raise ValueError("preset must be an object")
        return save_render_preset(storage_root, preset)
    if action == "promote_to_template":
        preset = get_render_preset(str(args.get("preset_id") or ""), storage_root)
        template = preset_to_template(
            preset,
            name=args.get("name"),
            prompt=str(args.get("prompt") or preset.get("purpose") or ""),
            audio_path=str(args.get("audio_path") or ""),
            duration_seconds=float(args["duration_seconds"]) if args.get("duration_seconds") not in {None, ""} else None,
            fps=int(args["fps"]) if args.get("fps") not in {None, ""} else None,
        )
        return _write_template(storage_root, str(args.get("template_name") or f"{preset['preset_id']}.json"), template)
    raise ValueError(f"unsupported preset action: {action}")


def _frame_diff_replay_accuracy(args: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(args.get("source") or args.get("source_path") or ""))
    regen = Path(str(args.get("regen") or args.get("regen_path") or ""))
    report = {
        "source_path": str(source) if str(source) else "",
        "regen_path": str(regen) if str(regen) else "",
        "source_exists": source.exists() if str(source) else False,
        "regen_exists": regen.exists() if str(regen) else False,
        "metrics": {},
        "boundary": {"state_or_manifest_diff_only": True, "semantic_detection": False},
    }
    if source.exists() and regen.exists():
        source_size = source.stat().st_size
        regen_size = regen.stat().st_size
        report["metrics"] = {
            "source_size_bytes": source_size,
            "regen_size_bytes": regen_size,
            "size_delta_bytes": regen_size - source_size,
            "size_ratio": round(regen_size / max(1, source_size), 6),
        }
    else:
        report["metrics"] = {
            "status": "paths_missing_or_not_supplied",
            "planned_metrics": ["frame_count_match", "duration_match", "state_channel_delta", "manifest_field_diff"],
        }
    return report


def _execute_validated_tool(validated: dict[str, Any], storage_root: Path) -> dict[str, Any]:
    tool = validated["tool"]
    args = validated["args"]
    if tool == "audio_probe_duration":
        duration = _probe_duration(str(args.get("path") or args.get("audio_path") or ""))
        return {"duration_seconds": duration, "path": str(args.get("path") or args.get("audio_path") or "")}
    if tool == "audio_analyze_levels":
        return _write_audio_level_signals(storage_root, args)
    if tool == "audio_extract_features":
        return _write_audio_features(storage_root, args)
    if tool == "trueaudio_log_pre_sound":
        return log_pre_sound_state(
            Path(str(args.get("path") or args.get("audio_path") or "")),
            storage_root=storage_root,
            run_id=str(args.get("run_id") or "") or None,
            fps=int(args.get("fps") or 30),
            sample_rate=int(args.get("sample_rate") or 48000),
            max_seconds=float(args["max_seconds"]) if args.get("max_seconds") not in {None, ""} else None,
        )
    if tool == "trueaudio_log_machine_pre_sound":
        return log_machine_pre_sound_state(
            storage_root=storage_root,
            run_id=str(args.get("run_id") or "") or None,
            duration_seconds=float(args.get("duration_seconds") or args.get("seconds") or 10.0),
            fps=int(args.get("fps") or 30),
        )
    if tool == "trueaudio_replay_state":
        return replay_trueaudio_state(
            Path(str(args.get("state") or args.get("state_path") or "")),
            storage_root=storage_root,
            run_id=str(args.get("run_id") or "") or None,
            sample_rate=int(args.get("sample_rate") or 48000),
        )
    if tool == "trueaudio_log_machine_replayable":
        return log_machine_replayable_audio_state(
            storage_root=storage_root,
            run_id=str(args.get("run_id") or "") or None,
            duration_seconds=float(args.get("duration_seconds") or args.get("seconds") or 10.0),
            frame_size=int(args.get("frame_size") or 2048),
            hop_size=int(args.get("hop_size") or 512),
        )
    if tool == "trueaudio_log_file_replayable":
        return log_file_replayable_audio_state(
            Path(str(args.get("path") or args.get("audio_path") or args.get("audio") or "")),
            storage_root=storage_root,
            run_id=str(args.get("run_id") or "") or None,
            sample_rate=int(args.get("sample_rate") or 48000),
            max_seconds=float(args["max_seconds"]) if args.get("max_seconds") not in {None, ""} else None,
            frame_size=int(args.get("frame_size") or 2048),
            hop_size=int(args.get("hop_size") or 512),
        )
    if tool == "trueaudio_replay_replayable":
        result = replay_replayable_audio_state(
            Path(str(args.get("state") or args.get("state_npz") or "")),
            storage_root=storage_root,
            run_id=str(args.get("run_id") or "") or None,
        )
        return {key: value for key, value in result.items() if key != "samples"}
    if tool == "truespeech_detect_segments":
        return detect_speech_segments_from_replayable_state(
            Path(str(args.get("state") or args.get("state_npz") or "")),
            storage_root=storage_root,
            run_id=str(args.get("run_id") or "") or None,
            speech_threshold=float(args.get("speech_threshold") or 0.48),
            min_segment_seconds=float(args.get("min_segment_seconds") or 0.12),
        )
    if tool == "truespeech_align_lyrics_candidate":
        return align_lyrics_to_speech_segments(
            Path(str(args.get("segments") or args.get("segments_json") or "")),
            lyrics_text=str(args["lyrics_text"]) if args.get("lyrics_text") not in {None, ""} else None,
            lyrics_path=Path(str(args["lyrics_path"])) if args.get("lyrics_path") not in {None, ""} else None,
            storage_root=storage_root,
            run_id=str(args.get("run_id") or "") or None,
        )
    if tool == "template_from_audio_signals":
        return _template_from_audio_signals(storage_root, args)
    if tool == "template_create":
        return {"template": _create_template(args)}
    if tool == "template_save":
        template = args.get("template")
        if not isinstance(template, dict):
            template = _create_template(args)
        name = str(args.get("name") or f"{template.get('name', 'template')}.json")
        return _write_template(storage_root, name, template)
    if tool == "template_load":
        name = str(args.get("name") or "")
        return {"name": safe_flat_json_name(name), "template": _read_template(storage_root, name)}
    if tool == "template_patch":
        name = str(args.get("name") or "")
        template = _read_template(storage_root, name)
        patched = _set_json_path(deepcopy(template), str(args.get("json_path") or ""), args.get("value"))
        patched.setdefault("patch_history", []).append(
            {
                "patched_at_utc": utc_now(),
                "json_path": str(args.get("json_path") or ""),
                "reason": str(args.get("reason") or ""),
            }
        )
        return _write_template(storage_root, name, patched)
    if tool == "template_create_variant":
        source = _read_template(storage_root, str(args.get("source_name") or args.get("name") or ""))
        variant = deepcopy(source)
        variant["name"] = str(args.get("variant_name") or f"{source.get('name', 'template')} variant")[:120]
        for path, value in (args.get("changes") or {}).items():
            _set_json_path(variant, str(path), value)
        variant.setdefault("variant_of", source.get("template_id") or source.get("name"))
        return _write_template(storage_root, str(args.get("variant_file") or f"{variant['name']}.json"), variant)
    if tool == "template_delete":
        path = _template_path(storage_root, str(args.get("name") or ""))
        existed = path.exists()
        if existed:
            path.unlink()
        return {"name": path.name, "deleted": existed}
    if tool == "storage_list_templates":
        return {"templates": _list_media_artifacts(storage_root, "templates")}
    if tool == "storage_list_artifacts":
        return {"files": _list_media_artifacts(storage_root, args.get("lane"))}
    if tool in {
        "source_snap_tool",
        "existing_state_animator",
        "electric_glow_intensity_animator",
        "spectrum_audio_reactive_city",
        "local_qwen_controller",
    }:
        plan = build_studio_tool_plan(tool, args)
        if tool == "spectrum_audio_reactive_city" and args.get("promote_to_template"):
            preset = get_render_preset("house_remix_audio_city", storage_root)
            template = preset_to_template(
                preset,
                name=str(args.get("name") or "House Remix Audio City"),
                prompt=str(args.get("prompt") or preset.get("purpose") or ""),
                audio_path=str(args.get("audio_path") or ""),
                duration_seconds=float(args["duration_seconds"]) if args.get("duration_seconds") not in {None, ""} else None,
                fps=int(args["fps"]) if args.get("fps") not in {None, ""} else None,
            )
            return _write_template(storage_root, str(args.get("template_name") or "house_remix_audio_city.json"), template)
        return {"plan": plan}
    if tool == "frame_diff_replay_accuracy":
        return _frame_diff_replay_accuracy(args)
    if tool == "manifest_browser":
        return _browse_manifests(storage_root, limit=int(args.get("limit") or 50))
    if tool == "render_preset_library":
        return _render_preset_library(storage_root, args)
    if tool == "time_marker_add":
        event = {
            "kind": "time_marker",
            "template_id": str(args.get("template_id") or ""),
            "source_artifact": str(args.get("source_artifact") or ""),
            "time_seconds": float(args.get("time_seconds") or 0),
            "note": str(args.get("note") or ""),
            "target": str(args.get("target") or ""),
            "direction": str(args.get("direction") or ""),
            "confidence": float(args.get("confidence") or 1.0),
        }
        return {"marker": append_recalibration_event(storage_root=storage_root, event=event)["event"]}
    if tool == "time_marker_list":
        return {"markers": list_recalibration_events(storage_root=storage_root, template_id=args.get("template_id"), kind="time_marker")}
    if tool == "recalibration_add_note":
        event = {
            "kind": "recalibration_note",
            "template_id": str(args.get("template_id") or ""),
            "source_artifact": str(args.get("source_artifact") or ""),
            "time_seconds": float(args.get("time_seconds") or 0),
            "note": str(args.get("note") or ""),
            "target": str(args.get("target") or ""),
            "direction": str(args.get("direction") or ""),
            "confidence": float(args.get("confidence") or 1.0),
        }
        return {"note": append_recalibration_event(storage_root=storage_root, event=event)["event"]}
    if tool == "recalibration_apply":
        notes = list_recalibration_events(storage_root=storage_root, template_id=args.get("template_id"), kind="recalibration_note")
        return {"patch_proposal": {"template_id": args.get("template_id"), "notes_used": len(notes), "notes": notes}}
    if tool == "video_render_preview":
        return _prepare_render_job(storage_root, args, preview=True)
    if tool == "video_prepare_full_render":
        return _prepare_render_job(storage_root, args, preview=False)
    if tool == "video_execute_full_render":
        return {"status": "execution_gated", "message": "Full render execution is not wired here; prepare job was approved only."}
    if tool == "manifest_generate":
        return _write_manifest(storage_root, str(args.get("name") or "av_manifest"), {"payload": args.get("payload") or args})
    if tool == "learning_record_save":
        event = {"kind": "learning_record", **args}
        return {"learning_record": append_recalibration_event(storage_root=storage_root, event=event)["event"]}
    if tool == "receipt_create":
        return {"status": "receipt_create_is_internal"}
    raise AVToolPolicyError(f"tool has no runner implementation: {tool}")


def run_av_tool_call(call: dict[str, Any], *, storage_root: Path) -> dict[str, Any]:
    _ensure_storage(storage_root)
    tool = str(call.get("tool") or "unknown") if isinstance(call, dict) else "invalid"
    try:
        validated = validate_tool_call(call)
        result = _execute_validated_tool(validated, storage_root)
        receipt = write_tool_receipt(storage_root=storage_root, tool=validated["tool"], status="ok", call=validated, result=result)
        return {"ok": True, "tool": validated["tool"], "result": result, "receipt": receipt}
    except Exception as exc:
        receipt = write_tool_receipt(storage_root=storage_root, tool=tool, status="rejected", call=call if isinstance(call, dict) else {}, error=str(exc))
        return {"ok": False, "tool": tool, "error": str(exc), "receipt": receipt}
