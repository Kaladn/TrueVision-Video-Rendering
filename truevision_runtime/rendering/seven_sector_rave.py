from __future__ import annotations

import json
import math
import subprocess
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


SECTOR_ROLES = ["vocal", "drums", "bass", "synth", "guitar", "keys", "other"]
ROLE_COLORS = {
    "vocal": (240, 245, 255),
    "drums": (255, 255, 255),
    "bass": (45, 45, 210),
    "synth": (255, 190, 40),
    "guitar": (55, 120, 255),
    "keys": (120, 220, 255),
    "other": (180, 90, 210),
}
MAX_STEM_MEMBER_BYTES = 512 * 1024 * 1024
MAX_STEM_ZIP_BYTES = 2 * 1024 * 1024 * 1024
STEM_COPY_CHUNK_BYTES = 1024 * 1024


def map_stem_name_to_role(name: str) -> str:
    lowered = name.lower()
    if "vocal" in lowered or "voice" in lowered or "lead" in lowered:
        return "vocal"
    if "drum" in lowered or "kick" in lowered or "snare" in lowered:
        return "drums"
    if "bass" in lowered or "808" in lowered or "sub" in lowered:
        return "bass"
    if "synth" in lowered:
        return "synth"
    if "guitar" in lowered:
        return "guitar"
    if "key" in lowered or "piano" in lowered:
        return "keys"
    return "other"


def assign_stem_paths(paths: list[Path]) -> tuple[dict[str, Path], list[str]]:
    mapping: dict[str, Path] = {}
    for path in paths:
        role = map_stem_name_to_role(path.name)
        mapping.setdefault(role, path)
    fallbacks = [role for role in SECTOR_ROLES if role not in mapping]
    if "other" not in mapping and paths:
        used = set(mapping.values())
        extra = next((path for path in paths if path not in used), paths[-1])
        mapping["other"] = extra
        if "other" in fallbacks:
            fallbacks.remove("other")
    return mapping, fallbacks


def _safe_stem_filename(filename: str) -> str:
    basename = Path(filename).name
    safe = "".join(char if char.isalnum() or char in ".-_" else "_" for char in basename).strip("._")
    return safe or "stem"


def extract_stems(stems_zip: Path, work_dir: Path, seconds: float) -> list[Path]:
    raw_dir = work_dir / "raw_stems"
    wav_dir = work_dir / "wav_stems"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    decoded: list[Path] = []
    total_uncompressed = 0
    with zipfile.ZipFile(stems_zip, "r") as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                continue
            if entry.file_size > MAX_STEM_MEMBER_BYTES:
                raise ValueError(f"Stem ZIP member exceeds size limit: {entry.filename}")
            total_uncompressed += entry.file_size
            if total_uncompressed > MAX_STEM_ZIP_BYTES:
                raise ValueError("Stem ZIP exceeds total uncompressed size limit")

            safe_name = _safe_stem_filename(entry.filename)
            indexed_name = f"{len(decoded):03d}_{safe_name}"
            raw_path = raw_dir / indexed_name
            wav_path = wav_dir / f"{raw_path.stem}.wav"
            with archive.open(entry, "r") as source, raw_path.open("wb") as target:
                while True:
                    chunk = source.read(STEM_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    target.write(chunk)
            subprocess.run(
                ["ffmpeg", "-y", "-t", f"{seconds:.3f}", "-i", str(raw_path), str(wav_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            decoded.append(wav_path)
    return decoded


def normalize_audio(samples: Iterable[float]) -> list[float]:
    values = [float(sample) for sample in samples]
    peak = max((abs(value) for value in values), default=0.0)
    if peak <= 1e-9:
        return [0.0 for _ in values]
    return [value / peak for value in values]


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    if sample_width == 1:
        raw = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        raw = (raw - 128.0) / 128.0
    elif sample_width == 2:
        raw = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        raw = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1)
    return raw.astype(np.float32), sample_rate


def build_envelope(samples: np.ndarray, sample_rate: int, fps: int, duration_seconds: float) -> list[float]:
    frame_count = max(1, int(round(duration_seconds * fps)))
    values: list[float] = []
    for frame_index in range(frame_count):
        start = int(frame_index / fps * sample_rate)
        end = int((frame_index + 1) / fps * sample_rate)
        chunk = samples[start:end]
        if chunk.size == 0:
            values.append(0.0)
        else:
            values.append(float(np.sqrt(np.mean(np.square(chunk)))))
    peak = max(values, default=0.0)
    if peak <= 1e-9:
        return [0.0 for _ in values]
    return [min(1.0, value / peak) for value in values]


def build_sector_states(envelopes: dict[str, list[float]], fps: int, duration_seconds: float) -> list[dict[str, Any]]:
    frame_count = max(1, int(round(duration_seconds * fps)))
    states: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        t = frame_index / fps
        sectors: dict[str, dict[str, float]] = {}
        for role in SECTOR_ROLES:
            envelope = envelopes.get(role)
            if envelope:
                energy = float(envelope[min(frame_index, len(envelope) - 1)])
                previous = float(envelope[max(0, min(frame_index - 1, len(envelope) - 1))])
            else:
                energy = 0.0
                previous = 0.0
            transient = max(0.0, energy - previous)
            sectors[role] = {
                "energy": round(energy, 5),
                "transient": round(transient, 5),
                "phase": round((t * (0.35 + energy)) % 1.0, 5),
            }
        states.append({"frame": frame_index, "time": round(t, 5), "sectors": sectors})
    return states


def build_manifest(
    *,
    run_id: str,
    audio_path: str,
    stems_zip: str,
    output_video: str,
    fps: int,
    seconds: float,
    width: int,
    height: int,
    stem_mapping: dict[str, str],
    fallbacks: list[str],
    sector_drivers: dict[str, dict[str, str]] | None = None,
    frame_count: int | None = None,
    manifest_path: str | None = None,
    state_trace_path: str | None = None,
    state_sample: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "kind": "truevision_seven_sector_rave_reactor_manifest",
        "run_id": run_id,
        "audio_path": audio_path,
        "stems_zip": stems_zip,
        "output_video": output_video,
        "fps": fps,
        "seconds": seconds,
        "width": width,
        "height": height,
        "sector_roles": list(SECTOR_ROLES),
        "sector_law": {
            "center": "vocal exact waveform",
            "outer": {
                "drums": "impact strobes",
                "bass": "pressure tunnel rings",
                "synth": "rotating blade beam",
                "guitar": "jagged shard cuts",
                "keys": "harmonic glass arcs",
                "other": "fog sparks atmosphere",
            },
        },
        "stem_mapping": dict(stem_mapping),
        "fallbacks": list(fallbacks),
        "sector_drivers": dict(sector_drivers or {}),
    }
    if frame_count is not None:
        manifest["frame_count"] = frame_count
    if manifest_path is not None:
        manifest["manifest_path"] = manifest_path
    if state_trace_path is not None:
        manifest["state_trace_path"] = state_trace_path
    if state_sample is not None:
        manifest["state_sample"] = state_sample
    return manifest


def _state_sample(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not states:
        return []
    indexes = sorted({0, len(states) // 2, len(states) - 1})
    return [states[index] for index in indexes]


def _build_sector_drivers(stem_mapping: dict[str, Path], audio_path: Path) -> dict[str, dict[str, str]]:
    drivers: dict[str, dict[str, str]] = {}
    for role in SECTOR_ROLES:
        stem_path = stem_mapping.get(role)
        if stem_path is None:
            drivers[role] = {"source_type": "master_mix_fallback", "source_path": str(audio_path)}
        else:
            drivers[role] = {"source_type": "stem", "source_path": str(stem_path)}
    return drivers


def _write_state_trace(states: list[dict[str, Any]], state_trace_path: Path) -> None:
    with state_trace_path.open("w", encoding="utf-8") as trace_file:
        for state in states:
            trace_file.write(json.dumps(state, separators=(",", ":")))
            trace_file.write("\n")


def render_seven_sector_rave(
    *,
    audio_path: str | Path,
    stems_zip: str | Path,
    output_root: str | Path,
    run_id: str,
    seconds: float = 30.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> dict[str, Any]:
    audio_path = Path(audio_path)
    stems_zip = Path(stems_zip)
    output_root = Path(output_root)
    work_dir = output_root / "work"
    frame_dir = work_dir / "frames"
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    decoded_stems = extract_stems(stems_zip, work_dir, seconds)
    stem_mapping, fallbacks = assign_stem_paths(decoded_stems)
    sector_drivers = _build_sector_drivers(stem_mapping, audio_path)

    master_samples, master_sample_rate = read_wav_mono(audio_path)
    envelopes: dict[str, list[float]] = {}
    for role in SECTOR_ROLES:
        driver = sector_drivers[role]
        if driver["source_type"] == "master_mix_fallback":
            samples = master_samples
            sample_rate = master_sample_rate
        else:
            samples, sample_rate = read_wav_mono(Path(driver["source_path"]))
        envelopes[role] = build_envelope(samples, sample_rate=sample_rate, fps=fps, duration_seconds=seconds)

    states = build_sector_states(envelopes, fps=fps, duration_seconds=seconds)
    frame_count = len(states)

    vocal_path = stem_mapping.get("vocal")
    if vocal_path is None:
        vocal_samples = master_samples
        vocal_sample_rate = master_sample_rate
    else:
        vocal_samples, vocal_sample_rate = read_wav_mono(vocal_path)
    vocal_limit = min(len(vocal_samples), int(seconds * vocal_sample_rate))
    vocal_waveform = normalize_audio(vocal_samples[:vocal_limit].tolist())

    silent_video = work_dir / f"{run_id}_silent.mp4"
    output_video = output_root / f"{run_id}.mp4"
    manifest_path = output_root / f"{run_id}_manifest.json"
    state_trace_path = output_root / f"{run_id}_state_trace.jsonl"

    writer = cv2.VideoWriter(
        str(silent_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {silent_video}")
    try:
        for frame_index, state in enumerate(states):
            vocal_end = min(len(vocal_waveform), int((frame_index + 1) / fps * vocal_sample_rate))
            waveform_slice = vocal_waveform[:vocal_end]
            writer.write(render_frame(state, width=width, height=height, waveform=waveform_slice))
    finally:
        writer.release()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(silent_video),
            "-t",
            f"{seconds:.3f}",
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    _write_state_trace(states, state_trace_path)
    manifest = build_manifest(
        run_id=run_id,
        audio_path=str(audio_path),
        stems_zip=str(stems_zip),
        output_video=str(output_video),
        fps=fps,
        seconds=seconds,
        width=width,
        height=height,
        stem_mapping={role: str(path) for role, path in stem_mapping.items()},
        fallbacks=fallbacks,
        sector_drivers=sector_drivers,
        frame_count=frame_count,
        manifest_path=str(manifest_path),
        state_trace_path=str(state_trace_path),
        state_sample=_state_sample(states),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _bgr(color: tuple[int, int, int], scale: float = 1.0) -> tuple[int, int, int]:
    red, green, blue = color
    return (
        max(0, min(255, int(blue * scale))),
        max(0, min(255, int(green * scale))),
        max(0, min(255, int(red * scale))),
    )


def _sector_values(state: dict[str, Any], role: str) -> tuple[float, float, float]:
    sector = state.get("sectors", {}).get(role, {})
    return (
        float(sector.get("energy", 0.0)),
        float(sector.get("transient", 0.0)),
        float(sector.get("phase", 0.0)),
    )


def render_frame(state: dict[str, Any], width: int, height: int, waveform: list[float]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (8, 6, 12)

    center = (width // 2, height // 2)
    radius = int(min(width, height) * 0.16)
    outer_radius = int(min(width, height) * 0.39)
    cv2.circle(frame, center, outer_radius + 26, (20, 18, 28), 2, cv2.LINE_AA)
    cv2.circle(frame, center, radius, (42, 52, 64), 2, cv2.LINE_AA)

    roles = ["synth", "guitar", "keys", "other", "bass", "drums"]
    angles = [-90, -30, 30, 90, 150, 210]
    for role, angle in zip(roles, angles):
        energy, transient, phase = _sector_values(state, role)
        theta = math.radians(angle)
        sx = int(center[0] + math.cos(theta) * outer_radius)
        sy = int(center[1] + math.sin(theta) * outer_radius)
        color = ROLE_COLORS[role]

        cv2.circle(frame, (sx, sy), int(radius * 0.78), _bgr(color, 0.18 + 0.75 * energy), 2 + int(4 * transient), cv2.LINE_AA)
        cv2.line(frame, center, (sx, sy), _bgr(color, 0.18 + 0.55 * energy), 1 + int(3 * energy), cv2.LINE_AA)

        if role == "drums":
            for ring in range(3):
                rr = int(radius * (0.35 + ring * 0.22 + transient * 0.5))
                cv2.circle(frame, (sx, sy), rr, _bgr(color, 0.25 + transient), 1, cv2.LINE_AA)
        elif role == "bass":
            for ring in range(5):
                rr = int(radius * (0.22 + ring * 0.16 + energy * 0.20))
                cv2.ellipse(frame, (sx, sy), (rr, max(3, rr // 3)), 0, 0, 360, _bgr(color, 0.22 + energy * 0.7), 1, cv2.LINE_AA)
        elif role == "synth":
            blade_angle = math.radians(angle + phase * 360.0)
            ex = int(sx + math.cos(blade_angle) * radius * 0.86)
            ey = int(sy + math.sin(blade_angle) * radius * 0.86)
            cv2.line(frame, (sx, sy), (ex, ey), _bgr(color, 0.55 + energy), 3, cv2.LINE_AA)
        elif role == "guitar":
            for shard in range(5):
                dx = int(math.cos(theta + shard) * radius * energy)
                dy = int(math.sin(theta - shard * 0.7) * radius * energy)
                cv2.line(frame, (sx - dx, sy - dy), (sx + dy, sy + dx), _bgr(color, 0.35 + transient), 1, cv2.LINE_AA)
        elif role == "keys":
            for arc in range(4):
                rr = int(radius * (0.25 + arc * 0.16))
                cv2.ellipse(frame, (sx, sy), (rr, rr), angle, 20, 200, _bgr(color, 0.25 + energy * 0.6), 1, cv2.LINE_AA)
        elif role == "other":
            for dot in range(18):
                px = int(sx + math.sin(dot * 1.7 + phase * 6.28) * radius * (0.2 + energy))
                py = int(sy + math.cos(dot * 1.1 + phase * 6.28) * radius * (0.2 + energy))
                cv2.circle(frame, (px, py), 1 + int(2 * energy), _bgr(color, 0.20 + energy * 0.5), -1, cv2.LINE_AA)

    vocal_energy, vocal_transient, _ = _sector_values(state, "vocal")
    glow = 0.35 + vocal_energy * 0.85
    cv2.circle(frame, center, int(radius * (1.02 + vocal_energy * 0.12)), _bgr(ROLE_COLORS["vocal"], glow), 2 + int(3 * vocal_transient), cv2.LINE_AA)
    if waveform:
        points = []
        samples = waveform[-160:]
        for index, value in enumerate(samples):
            x = int(center[0] - radius * 0.82 + index / max(1, len(samples) - 1) * radius * 1.64)
            y = int(center[1] + float(value) * radius * 0.52)
            points.append((x, y))
        for a, b in zip(points, points[1:]):
            cv2.line(frame, a, b, (255, 250, 245), 1 + int(vocal_energy * 2), cv2.LINE_AA)

    return cv2.GaussianBlur(frame, (0, 0), 0.35)
