from __future__ import annotations

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
            envelope = envelopes.get(role, [0.0] * frame_count)
            energy = float(envelope[min(frame_index, len(envelope) - 1)]) if envelope else 0.0
            previous = float(envelope[max(0, min(frame_index - 1, len(envelope) - 1))]) if envelope else 0.0
            transient = max(0.0, energy - previous)
            sectors[role] = {
                "energy": round(energy, 5),
                "transient": round(transient, 5),
                "phase": round((t * (0.35 + energy)) % 1.0, 5),
            }
        states.append({"frame": frame_index, "time": round(t, 5), "sectors": sectors})
    return states
