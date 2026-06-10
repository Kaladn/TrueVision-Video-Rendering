# Seven-Sector Rave Reactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 30-second TrueVision-rendered rave light show where a vocal waveform center and six surrounding sectors are driven by stems from "Lower the Room x Mind Scrape (Mashup)".

**Architecture:** Add a focused runtime module that decodes audio/stems, derives per-frame envelopes, maps stems to seven visual sectors, renders deterministic OpenCV frames, muxes the full mix with ffmpeg, and writes a manifest. Keep the renderer separate from CLI wrappers so tests can exercise analysis/mapping/math without rendering a full video.

**Tech Stack:** Python 3.11, standard library `wave`/`zipfile`/`subprocess`, NumPy, OpenCV, ffmpeg, pytest.

---

## File Structure

- Create `truevision_runtime/rendering/seven_sector_rave.py`
  - Owns stem discovery, audio decoding, envelope extraction, sector mapping, frame rendering, mp4 muxing, manifest creation.
- Create `scripts/render_seven_sector_rave_reactor.py`
  - CLI entrypoint for the specific proof and reusable future renders.
- Create `tests/test_seven_sector_rave.py`
  - Unit tests for stem mapping, envelope normalization, waveform sampling, and manifest shape.
- Output runtime artifacts under `outputs/seven_sector_rave_reactor/<run_id>/`.

Do not modify the existing avatar renderer, markdown frame renderer, or old rave sample except as reference material.

---

### Task 1: Stem Mapping And Audio Contract

**Files:**
- Create: `truevision_runtime/rendering/seven_sector_rave.py`
- Test: `tests/test_seven_sector_rave.py`

- [ ] **Step 1: Write failing tests for stem role mapping**

Add this to `tests/test_seven_sector_rave.py`:

```python
from truevision_runtime.rendering.seven_sector_rave import map_stem_name_to_role, normalize_audio


def test_map_stem_name_to_role_prefers_explicit_names():
    assert map_stem_name_to_role("Lead Vocals.wav") == "vocal"
    assert map_stem_name_to_role("Drums.wav") == "drums"
    assert map_stem_name_to_role("Bass.wav") == "bass"
    assert map_stem_name_to_role("Synth.wav") == "synth"
    assert map_stem_name_to_role("Guitar.wav") == "guitar"
    assert map_stem_name_to_role("Keyboard.wav") == "keys"
    assert map_stem_name_to_role("Other.wav") == "other"


def test_normalize_audio_handles_silence_and_peak():
    silent = normalize_audio([0.0, 0.0, 0.0])
    assert silent == [0.0, 0.0, 0.0]
    normalized = normalize_audio([0.0, 2.0, -1.0])
    assert normalized == [0.0, 1.0, -0.5]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests\test_seven_sector_rave.py -q
```

Expected: FAIL because `truevision_runtime.rendering.seven_sector_rave` does not exist.

- [ ] **Step 3: Implement minimal mapping and normalization**

Create `truevision_runtime/rendering/seven_sector_rave.py` with:

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
python -m pytest tests\test_seven_sector_rave.py -q
```

Expected: PASS.

---

### Task 2: Audio Decode And Envelope Extraction

**Files:**
- Modify: `truevision_runtime/rendering/seven_sector_rave.py`
- Modify: `tests/test_seven_sector_rave.py`

- [ ] **Step 1: Add tests for envelope extraction**

Append:

```python
import numpy as np

from truevision_runtime.rendering.seven_sector_rave import build_envelope


def test_build_envelope_returns_one_value_per_frame():
    samples = np.ones(48000, dtype=np.float32) * 0.25
    envelope = build_envelope(samples, sample_rate=48000, fps=30, duration_seconds=1.0)
    assert len(envelope) == 30
    assert all(0.99 <= value <= 1.0 for value in envelope)


def test_build_envelope_tracks_quiet_and_loud_regions():
    samples = np.concatenate(
        [
            np.zeros(24000, dtype=np.float32),
            np.ones(24000, dtype=np.float32),
        ]
    )
    envelope = build_envelope(samples, sample_rate=48000, fps=10, duration_seconds=1.0)
    assert max(envelope[:5]) == 0.0
    assert min(envelope[5:]) > 0.9
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests\test_seven_sector_rave.py -q
```

Expected: FAIL because `build_envelope` is missing.

- [ ] **Step 3: Implement WAV decode and envelope extraction**

Add to `seven_sector_rave.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests\test_seven_sector_rave.py -q
```

Expected: PASS.

---

### Task 3: Stem Zip Decoding

**Files:**
- Modify: `truevision_runtime/rendering/seven_sector_rave.py`
- Modify: `tests/test_seven_sector_rave.py`

- [ ] **Step 1: Add test for decoded stem mapping from paths**

Append:

```python
from pathlib import Path

from truevision_runtime.rendering.seven_sector_rave import assign_stem_paths


def test_assign_stem_paths_records_fallbacks():
    paths = [
        Path("Lead Vocals.wav"),
        Path("Drums.wav"),
        Path("Bass.wav"),
        Path("Mystery.wav"),
    ]
    mapping, fallbacks = assign_stem_paths(paths)
    assert mapping["vocal"].name == "Lead Vocals.wav"
    assert mapping["drums"].name == "Drums.wav"
    assert mapping["bass"].name == "Bass.wav"
    assert mapping["other"].name == "Mystery.wav"
    assert "synth" in fallbacks
    assert "guitar" in fallbacks
    assert "keys" in fallbacks
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m pytest tests\test_seven_sector_rave.py::test_assign_stem_paths_records_fallbacks -q
```

Expected: FAIL because `assign_stem_paths` is missing.

- [ ] **Step 3: Implement stem assignment and zip extraction**

Add:

```python
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


def extract_stems(stems_zip: Path, work_dir: Path, seconds: float) -> list[Path]:
    raw_dir = work_dir / "raw_stems"
    wav_dir = work_dir / "wav_stems"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    decoded: list[Path] = []
    with zipfile.ZipFile(stems_zip, "r") as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                continue
            raw_path = raw_dir / Path(entry.filename).name
            raw_path.write_bytes(archive.read(entry))
            wav_path = wav_dir / f"{raw_path.stem}.wav"
            if raw_path.suffix.lower() == ".wav":
                wav_path.write_bytes(raw_path.read_bytes())
            else:
                subprocess.run(
                    ["ffmpeg", "-y", "-t", f"{seconds:.3f}", "-i", str(raw_path), str(wav_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            decoded.append(wav_path)
    return decoded
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests\test_seven_sector_rave.py -q
```

Expected: PASS.

---

### Task 4: Seven-Sector State Build

**Files:**
- Modify: `truevision_runtime/rendering/seven_sector_rave.py`
- Modify: `tests/test_seven_sector_rave.py`

- [ ] **Step 1: Add test for sector state manifest shape**

Append:

```python
from truevision_runtime.rendering.seven_sector_rave import build_sector_states


def test_build_sector_states_contains_all_roles():
    envelopes = {role: [0.0, 0.5, 1.0] for role in ["vocal", "drums", "bass", "synth", "guitar", "keys", "other"]}
    states = build_sector_states(envelopes, fps=3, duration_seconds=1.0)
    assert len(states) == 3
    assert set(states[0]["sectors"]) == {"vocal", "drums", "bass", "synth", "guitar", "keys", "other"}
    assert states[2]["sectors"]["vocal"]["energy"] == 1.0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m pytest tests\test_seven_sector_rave.py::test_build_sector_states_contains_all_roles -q
```

Expected: FAIL because `build_sector_states` is missing.

- [ ] **Step 3: Implement sector state construction**

Add:

```python
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
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests\test_seven_sector_rave.py -q
```

Expected: PASS.

---

### Task 5: Frame Renderer

**Files:**
- Modify: `truevision_runtime/rendering/seven_sector_rave.py`
- Modify: `tests/test_seven_sector_rave.py`

- [ ] **Step 1: Add nonblank frame test**

Append:

```python
from truevision_runtime.rendering.seven_sector_rave import render_frame


def test_render_frame_is_nonblank():
    state = {
        "frame": 0,
        "time": 0.0,
        "sectors": {role: {"energy": 0.8, "transient": 0.3, "phase": 0.2} for role in ["vocal", "drums", "bass", "synth", "guitar", "keys", "other"]},
    }
    frame = render_frame(state, width=640, height=360, waveform=[0.0, 0.5, -0.5, 0.0])
    assert frame.shape == (360, 640, 3)
    assert int(frame.max()) > 20
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m pytest tests\test_seven_sector_rave.py::test_render_frame_is_nonblank -q
```

Expected: FAIL because `render_frame` is missing.

- [ ] **Step 3: Implement deterministic OpenCV renderer**

Add:

```python
ROLE_COLORS = {
    "vocal": (240, 245, 255),
    "drums": (255, 255, 255),
    "bass": (45, 45, 210),
    "synth": (255, 190, 40),
    "guitar": (55, 120, 255),
    "keys": (120, 220, 255),
    "other": (180, 90, 210),
}


def _bgr(color: tuple[int, int, int], scale: float = 1.0) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * scale))) for channel in color)


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
        sector = state["sectors"][role]
        energy = sector["energy"]
        transient = sector["transient"]
        phase = sector["phase"]
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

    vocal = state["sectors"]["vocal"]
    glow = 0.35 + vocal["energy"] * 0.85
    cv2.circle(frame, center, int(radius * (1.02 + vocal["energy"] * 0.12)), _bgr(ROLE_COLORS["vocal"], glow), 2 + int(3 * vocal["transient"]), cv2.LINE_AA)
    if waveform:
        points = []
        samples = waveform[-160:]
        for index, value in enumerate(samples):
            x = int(center[0] - radius * 0.82 + index / max(1, len(samples) - 1) * radius * 1.64)
            y = int(center[1] + value * radius * 0.52)
            points.append((x, y))
        for a, b in zip(points, points[1:]):
            cv2.line(frame, a, b, (245, 250, 255), 1 + int(vocal["energy"] * 2), cv2.LINE_AA)
    return cv2.GaussianBlur(frame, (0, 0), 0.35)
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests\test_seven_sector_rave.py -q
```

Expected: PASS.

---

### Task 6: Full Render Pipeline And Manifest

**Files:**
- Modify: `truevision_runtime/rendering/seven_sector_rave.py`
- Create: `scripts/render_seven_sector_rave_reactor.py`
- Modify: `tests/test_seven_sector_rave.py`

- [ ] **Step 1: Add manifest test**

Append:

```python
from truevision_runtime.rendering.seven_sector_rave import build_manifest


def test_build_manifest_records_contract():
    manifest = build_manifest(
        run_id="demo",
        audio_path="song.wav",
        stems_zip="stems.zip",
        output_video="out.mp4",
        fps=30,
        seconds=30.0,
        width=1280,
        height=720,
        stem_mapping={"vocal": "Lead Vocals.wav"},
        fallbacks=["synth"],
    )
    assert manifest["kind"] == "truevision_seven_sector_rave_reactor_manifest"
    assert manifest["run_id"] == "demo"
    assert manifest["sector_law"]["center"] == "vocal exact waveform"
    assert manifest["fallbacks"] == ["synth"]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m pytest tests\test_seven_sector_rave.py::test_build_manifest_records_contract -q
```

Expected: FAIL because `build_manifest` is missing.

- [ ] **Step 3: Implement manifest and render function**

Add:

```python
def build_manifest(
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
) -> dict[str, Any]:
    return {
        "kind": "truevision_seven_sector_rave_reactor_manifest",
        "run_id": run_id,
        "audio_path": audio_path,
        "stems_zip": stems_zip,
        "output_video": output_video,
        "fps": fps,
        "seconds": seconds,
        "resolution": {"width": width, "height": height},
        "stem_mapping": stem_mapping,
        "fallbacks": fallbacks,
        "sector_law": {
            "center": "vocal exact waveform",
            "outer": {
                "drums": "strobes and impact gates",
                "bass": "tunnel and pressure",
                "synth": "saber spin and blade orbit",
                "guitar": "shards and scrape cuts",
                "keys": "harmonic glass arcs",
                "other": "fog particles and atmosphere",
            },
        },
        "boundary": {
            "truevision_rendered": True,
            "synthetic_visual_media": True,
            "not_foundation_video_model_training": True,
        },
    }
```

Then add `render_seven_sector_rave(...)` in the same file:

```python
def render_seven_sector_rave(
    audio_path: Path,
    stems_zip: Path,
    output_root: Path,
    run_id: str,
    seconds: float = 30.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = output_root / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    stems = extract_stems(stems_zip, work_dir, seconds)
    mapping, fallbacks = assign_stem_paths(stems)
    master, master_sr = read_wav_mono(audio_path)
    frame_count = max(1, int(round(seconds * fps)))
    envelopes: dict[str, list[float]] = {}
    for role in SECTOR_ROLES:
        source = mapping.get(role)
        if source and source.exists():
            samples, sample_rate = read_wav_mono(source)
        else:
            samples, sample_rate = master, master_sr
        envelopes[role] = build_envelope(samples, sample_rate, fps, seconds)
    states = build_sector_states(envelopes, fps, seconds)
    vocal_samples, _ = read_wav_mono(mapping.get("vocal", audio_path))
    vocal_wave = normalize_audio(vocal_samples[: int(seconds * master_sr)].tolist())

    raw_video = output_root / f"{run_id}_silent.mp4"
    final_video = output_root / f"{run_id}.mp4"
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame_index, state in enumerate(states):
        end = int((frame_index + 1) / frame_count * len(vocal_wave))
        start = max(0, end - 220)
        frame = render_frame(state, width, height, vocal_wave[start:end])
        writer.write(frame)
    writer.release()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_video),
            "-t",
            f"{seconds:.3f}",
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(final_video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = build_manifest(
        run_id=run_id,
        audio_path=str(audio_path),
        stems_zip=str(stems_zip),
        output_video=str(final_video),
        fps=fps,
        seconds=seconds,
        width=width,
        height=height,
        stem_mapping={role: str(path) for role, path in mapping.items()},
        fallbacks=fallbacks,
    )
    manifest["frame_count"] = frame_count
    manifest["state_sample"] = states[:5]
    manifest_path = output_root / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
```

- [ ] **Step 4: Add CLI wrapper**

Create `scripts/render_seven_sector_rave_reactor.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from truevision_runtime.rendering.seven_sector_rave import render_seven_sector_rave


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=r"C:\Users\mydyi\Downloads\Lower the Room x Mind Scrape (Mashup).wav")
    parser.add_argument("--stems", default=r"C:\Users\mydyi\Downloads\Lower the Room x Mind Scrape (Mashup) Stems (86BPM).zip")
    parser.add_argument("--output-root", default="outputs/seven_sector_rave_reactor/lower_room_mind_scrape_30s")
    parser.add_argument("--run-id", default="lower_room_mind_scrape_30s")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    manifest = render_seven_sector_rave(
        audio_path=Path(args.audio),
        stems_zip=Path(args.stems),
        output_root=Path(args.output_root),
        run_id=args.run_id,
        seconds=args.seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(manifest["output_video"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests\test_seven_sector_rave.py -q
```

Expected: PASS.

---

### Task 7: Render Proof And Verify Media

**Files:**
- Runtime output only under `outputs/seven_sector_rave_reactor/lower_room_mind_scrape_30s/`

- [ ] **Step 1: Render the 30-second proof**

Run:

```powershell
python scripts\render_seven_sector_rave_reactor.py --seconds 30 --fps 30 --width 1280 --height 720
```

Expected: exits 0 and prints:

```text
outputs\seven_sector_rave_reactor\lower_room_mind_scrape_30s\lower_room_mind_scrape_30s.mp4
```

- [ ] **Step 2: Verify output files exist**

Run:

```powershell
Test-Path outputs\seven_sector_rave_reactor\lower_room_mind_scrape_30s\lower_room_mind_scrape_30s.mp4
Test-Path outputs\seven_sector_rave_reactor\lower_room_mind_scrape_30s\lower_room_mind_scrape_30s_manifest.json
```

Expected:

```text
True
True
```

- [ ] **Step 3: Probe duration and stream presence**

Run:

```powershell
ffprobe -v error -show_entries format=duration -show_streams -of json outputs\seven_sector_rave_reactor\lower_room_mind_scrape_30s\lower_room_mind_scrape_30s.mp4
```

Expected: JSON contains one video stream, one audio stream, and duration near `30.0`.

- [ ] **Step 4: Commit implementation**

Run:

```powershell
git add truevision_runtime/rendering/seven_sector_rave.py scripts/render_seven_sector_rave_reactor.py tests/test_seven_sector_rave.py docs/superpowers/plans/2026-06-10-seven-sector-rave-reactor.md
git commit -m "Add seven-sector rave reactor renderer"
```

Expected: commit succeeds. Do not add unrelated existing dirty files.

---

## Self-Review

- Spec coverage: the plan covers stem analysis, center vocal waveform, six sector roles, video rendering, muxed audio, manifest, and tests.
- Placeholder scan: no unresolved TBD/TODO placeholders are present.
- Type consistency: function names and paths are consistent across tests, implementation, and CLI.
- Scope check: this is a first deterministic TrueVision render proof, not a full trained video model or full-song final master.
