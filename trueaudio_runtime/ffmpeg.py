from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def find_media_executable(name: str) -> str:
    """Find ffmpeg/ffprobe without relying on one shell's PATH quirks."""
    env_names = [f"TRUEAUDIO_{name.upper()}", f"{name.upper()}_PATH"]
    for env_name in env_names:
        configured = os.environ.get(env_name)
        if configured and Path(configured).exists():
            return configured

    discovered = shutil.which(name)
    if discovered:
        return discovered

    candidates = [
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / f"{name}.exe",
    ]
    winget = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget.exists():
        candidates.extend(winget.glob(f"**/{name}.exe"))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(f"{name} executable was not found")


def probe_audio(path: Path) -> dict[str, Any]:
    ffprobe = find_media_executable("ffprobe")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = payload.get("format") or {}
    duration = stream.get("duration") or fmt.get("duration") or 0.0
    return {
        "duration_seconds": round(float(duration or 0.0), 6),
        "codec_name": str(stream.get("codec_name") or ""),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "channel_layout": str(stream.get("channel_layout") or ""),
        "bit_rate": int(float(stream.get("bit_rate") or fmt.get("bit_rate") or 0)),
    }


def decode_pcm_f32_stereo(path: Path, *, sample_rate: int, max_seconds: float | None = None) -> np.ndarray:
    ffmpeg = find_media_executable("ffmpeg")
    command = [ffmpeg, "-v", "error", "-i", str(path)]
    if max_seconds is not None:
        command.extend(["-t", f"{max_seconds:.6f}"])
    command.extend(["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sample_rate), "-"])
    completed = subprocess.run(command, check=True, capture_output=True, timeout=180)
    if not completed.stdout:
        return np.zeros((0, 2), dtype=np.float32)
    pcm = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32)
    if pcm.size % 2:
        pcm = pcm[:-1]
    return np.clip(pcm.reshape((-1, 2)), -1.0, 1.0)
