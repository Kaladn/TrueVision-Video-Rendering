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
