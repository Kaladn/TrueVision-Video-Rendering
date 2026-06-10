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
