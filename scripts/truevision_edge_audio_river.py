#!/usr/bin/env python3
"""Audio-reactive Edge Of The World color-river renderer.

This is a small first-pass visualizer, not a MilkDrop engine integration. It
turns audio bands into deterministic river state and renders a black-field,
snake-like color current with no lettering or glyph layers.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_AUDIO = Path(
    r"C:\Users\mydyi\OneDrive\Documents\Desktop\Album_Builds\Machine_Dread_Album_Sequenced"
    r"\01_ordered_audio\01 - Edge Of The World (I Am Your Nightmare).mp3"
)
DEFAULT_LYRICS = Path(r"C:\Users\mydyi\OneDrive\Documents\Desktop\Full Album Lyrics_sound.txt")
DEFAULT_OUTPUT_ROOT = Path("outputs/edge_of_the_world_audio_river")
DEFAULT_RUN_ID = "edge_of_the_world_audio_river"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def slug(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return clean.strip("_")[:96] or "edge_audio_river"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def probe_audio_duration(audio_path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def decode_audio_mono(audio_path: Path, *, sample_rate: int = 44100, max_seconds: float | None = None) -> np.ndarray:
    command = ["ffmpeg", "-v", "error", "-i", str(audio_path)]
    if max_seconds is not None:
        command.extend(["-t", f"{max_seconds:.6f}"])
    command.extend(["-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate), "-"])
    completed = subprocess.run(command, check=True, capture_output=True)
    if not completed.stdout:
        return np.zeros(0, dtype=np.float32)
    pcm = np.frombuffer(completed.stdout, dtype="<i2").astype(np.float32)
    return np.clip(pcm / 32768.0, -1.0, 1.0)


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    value = float(np.percentile(finite, percentile))
    return max(value, 1.0e-6)


def _normalize(values: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    scale = _safe_percentile(values, percentile)
    return np.clip(values / scale, 0.0, 1.0)


def measure_audio_features(samples: np.ndarray, *, sample_rate: int, fps: int) -> list[dict[str, float]]:
    if samples.size == 0:
        return []
    duration = samples.size / sample_rate
    frame_count = max(1, int(math.ceil(duration * fps)))
    window_size = min(samples.size, max(256, int(sample_rate * 0.08)))
    if window_size % 2:
        window_size += 1
    half = window_size // 2
    window = np.hanning(window_size).astype(np.float32)
    freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
    bass_mask = (freqs >= 20.0) & (freqs < 180.0)
    mid_mask = (freqs >= 180.0) & (freqs < 2200.0)
    high_mask = (freqs >= 2200.0) & (freqs < min(12000.0, sample_rate / 2.0))

    raw = {
        "rms": np.zeros(frame_count, dtype=np.float32),
        "bass": np.zeros(frame_count, dtype=np.float32),
        "mid": np.zeros(frame_count, dtype=np.float32),
        "high": np.zeros(frame_count, dtype=np.float32),
    }

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
        raw["bass"][frame_index] = float(np.mean(spectrum[bass_mask])) if np.any(bass_mask) else 0.0
        raw["mid"][frame_index] = float(np.mean(spectrum[mid_mask])) if np.any(mid_mask) else 0.0
        raw["high"][frame_index] = float(np.mean(spectrum[high_mask])) if np.any(high_mask) else 0.0

    norm = {key: _normalize(value) for key, value in raw.items()}
    smoothed_rms = np.zeros(frame_count, dtype=np.float32)
    last = 0.0
    for index, value in enumerate(norm["rms"]):
        last = 0.72 * last + 0.28 * float(value)
        smoothed_rms[index] = last
    onset = np.maximum(0.0, norm["rms"] - np.roll(smoothed_rms, 1))
    onset[0] = norm["rms"][0]
    beat = _normalize(onset, percentile=90.0)

    features: list[dict[str, float]] = []
    for frame_index in range(frame_count):
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
    return features


def build_edge_theme(lyrics_path: Path | None) -> dict[str, Any]:
    visual_rules = {
        "no_lettering": True,
        "no_glyphs": True,
        "no_lyric_overlay": True,
        "black_field": True,
        "sound_reactive_only": True,
    }
    if lyrics_path is None or not lyrics_path.exists():
        return {
            "track_title": "Edge Of The World",
            "source": None,
            "source_sha256": None,
            "theme": "waking up and joining people as one",
            "theme_phrases": ["river of life", "you me together", "still standing"],
            "visual_rules": visual_rules,
        }
    text = lyrics_path.read_text(encoding="utf-8", errors="replace")
    block = text.split("\n---", 1)[0]
    title = block.splitlines()[0].strip() or "Edge Of The World"
    lowered = block.lower()
    phrases = []
    if "river of life" in lowered:
        phrases.append("river of life")
    if "you! me! together" in lowered or "you! me!" in lowered:
        phrases.append("you me together")
    if "wake up" in lowered or "open your eyes" in lowered:
        phrases.append("wake up")
    if "still standing" in lowered:
        phrases.append("still standing")
    return {
        "track_title": title,
        "source": str(lyrics_path),
        "source_sha256": sha256_file(lyrics_path),
        "theme": "waking up and joining people as one",
        "theme_phrases": phrases or ["river of life", "you me together"],
        "visual_rules": visual_rules,
    }


def _hsv_to_bgr(hue_degrees: float, saturation: float, value: float) -> tuple[int, int, int]:
    hsv = np.asarray([[[hue_degrees % 180.0, np.clip(saturation, 0.0, 1.0) * 255.0, np.clip(value, 0.0, 1.0) * 255.0]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _polyline_points(
    *,
    width: int,
    height: int,
    time_seconds: float,
    strand: int,
    rms: float,
    bass: float,
    mid: float,
    high: float,
    river_height_ratio: float,
) -> np.ndarray:
    point_count = 190
    x = np.linspace(-0.05 * width, 1.05 * width, point_count, dtype=np.float32)
    u = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    band_height = height * np.clip(river_height_ratio, 0.18, 0.9)
    center_y = height * 0.5 + band_height * ((strand - 3.5) * 0.014)
    amp = band_height * (0.16 + 0.11 * bass + 0.035 * rms)
    phase = time_seconds * (0.7 + 0.7 * mid) + strand * 0.48
    y = (
        center_y
        + np.sin(u * math.tau * (1.4 + strand * 0.045) + phase * math.tau) * amp
        + np.sin(u * math.tau * (4.2 + high * 1.7) - phase * 2.1) * amp * 0.28
    )
    x_shift = math.sin(time_seconds * 0.23 + strand) * width * 0.04
    y += np.sin(time_seconds * 0.9 + strand * 1.7) * height * 0.025
    points = np.stack([x + x_shift, y], axis=1)
    return np.round(points).astype(np.int32).reshape((-1, 1, 2))


def render_river_frame(
    *,
    width: int,
    height: int,
    fps: int,
    frame_state: dict[str, float],
    trail: np.ndarray | None,
    river_height_ratio: float = 0.52,
    program_stamp: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if trail is None:
        trail = np.zeros((height, width, 3), dtype=np.float32)
    rms = float(frame_state.get("rms", 0.0))
    bass = float(frame_state.get("bass", 0.0))
    mid = float(frame_state.get("mid", 0.0))
    high = float(frame_state.get("high", 0.0))
    beat = float(frame_state.get("beat", 0.0))
    time_seconds = float(frame_state.get("time_seconds", 0.0))
    frame_index = int(frame_state.get("frame_index", 0))

    decay = 0.885 - 0.08 * high
    trail *= np.clip(decay, 0.78, 0.92)
    overlay = np.zeros_like(trail, dtype=np.uint8)

    strand_count = 8
    for strand in range(strand_count):
        points = _polyline_points(
            width=width,
            height=height,
            time_seconds=time_seconds,
            strand=strand,
            rms=rms,
            bass=bass,
            mid=mid,
            high=high,
            river_height_ratio=river_height_ratio,
        )
        hue = (time_seconds * 18.0 + strand * 14.0 + bass * 45.0 + high * 22.0) % 180.0
        color = _hsv_to_bgr(hue, 0.88 + 0.1 * mid, 0.34 + 0.42 * rms)
        width_px = max(2, int(round(2 + bass * 8 + beat * 3 + strand % 2)))
        cv2.polylines(overlay, [points], False, color, width_px + 4, lineType=cv2.LINE_AA)
        cv2.polylines(overlay, [points], False, color, width_px, lineType=cv2.LINE_AA)

    if beat > 0.18:
        pulse_count = 3
        for pulse in range(pulse_count):
            cx = int(width * (0.22 + 0.28 * pulse + 0.05 * math.sin(time_seconds + pulse)))
            cy = int(height * (0.5 + 0.18 * math.sin(time_seconds * 1.7 + pulse)))
            radius = int((height * 0.08) + beat * height * (0.08 + pulse * 0.025))
            color = _hsv_to_bgr((time_seconds * 24 + pulse * 42 + 90) % 180, 0.92, 0.24 + beat * 0.34)
            cv2.circle(overlay, (cx, cy), radius, color, max(1, int(1 + beat * 3)), lineType=cv2.LINE_AA)

    blur_size = int(9 + bass * 14)
    if blur_size % 2 == 0:
        blur_size += 1
    glow = cv2.GaussianBlur(overlay, (blur_size, blur_size), 0)
    glow_layer = glow.astype(np.float32) * (0.24 + 0.18 * bass)
    overlay_layer = overlay.astype(np.float32) * (0.72 + 0.18 * rms)
    trail[:] = np.maximum(trail, glow_layer)
    trail[:] = np.maximum(trail, overlay_layer)

    # Beat shock subtly pushes the whole field without adding symbols.
    if beat > 0.7 and frame_index % max(1, fps // 10) == 0:
        trail[:] = np.roll(trail, shift=int(beat * 4), axis=1)

    frame = np.clip(trail, 0, 210).astype(np.uint8)
    if program_stamp:
        cv2.putText(
            frame,
            program_stamp,
            (24, max(24, height - 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.38, height / 1600.0),
            (72, 78, 84),
            1,
            cv2.LINE_AA,
        )
    metadata = {
        "frame_index": frame_index,
        "time_seconds": round(time_seconds, 6),
        "visual_state": "black_field_color_river",
        "river": {
            "strand_count": strand_count,
            "decay": round(float(decay), 6),
            "line_width_base": round(float(2 + bass * 11 + beat * 5), 6),
            "bloom": round(float(0.22 + 0.35 * bass), 6),
            "river_height_ratio": round(float(river_height_ratio), 6),
        },
        "audio_features": {
            "rms": round(rms, 6),
            "bass": round(bass, 6),
            "mid": round(mid, 6),
            "high": round(high, 6),
            "beat": round(beat, 6),
        },
        "visual_rules": {
            "no_lettering": program_stamp is None,
            "no_glyphs": True,
            "no_lyric_overlay": True,
            "program_stamp": program_stamp is not None,
        },
    }
    return frame, metadata


def capture_hardware() -> dict[str, Any]:
    ram = {"total_physical_bytes": None, "available_physical_bytes": None}
    if platform.system().lower() == "windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            ram = {
                "total_physical_bytes": int(status.ullTotalPhys),
                "available_physical_bytes": int(status.ullAvailPhys),
            }
    return {
        "os": platform.platform(),
        "processor": platform.processor(),
        "cpu_logical_count": os.cpu_count(),
        "python": platform.python_version(),
        "ram": ram,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {manifest['run_id']} Report",
        "",
        "## Claim",
        "",
        "A full-song audio-reactive color-river video was generated from Track 1 audio features and lyric-derived theme rules.",
        "",
        "## Boundary",
        "",
        "Generated state media is synthetic, not evidence. No lettering, glyphs, or lyric overlays are rendered.",
        "",
        "## Inputs",
        "",
        f"- Audio: `{manifest['inputs']['audio_path']}`",
        f"- Audio SHA256: `{manifest['inputs']['audio_sha256']}`",
        f"- Lyrics: `{manifest['theme'].get('source')}`",
        "",
        "## Outputs",
        "",
        f"- Video: `{manifest['outputs']['video_mp4']}`",
        f"- Frame state JSONL: `{manifest['outputs']['frame_state_jsonl']}`",
        f"- Manifest: `{manifest['outputs']['manifest_json']}`",
        f"- Thumbnail: `{manifest['outputs']['thumbnail_jpg']}`",
        "",
        "## Render",
        "",
        f"- Resolution: `{manifest['render']['width']}x{manifest['render']['height']}`",
        f"- FPS: `{manifest['render']['fps']}`",
        f"- Frames: `{manifest['render']['frames']}`",
        f"- Duration seconds: `{manifest['render']['duration_seconds']}`",
        f"- River height ratio: `{manifest['render']['river_height_ratio']}`",
        f"- Audio muxed: `{manifest['outputs']['audio_muxed']}`",
        f"- Program stamp: `{manifest['boundary']['program_stamp']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_edge_audio_river(
    *,
    audio_path: Path,
    lyrics_path: Path | None,
    output_root: Path,
    run_id: str = DEFAULT_RUN_ID,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    sample_rate: int = 44100,
    max_seconds: float | None = None,
    mux_audio: bool = True,
    river_height_ratio: float = 0.38,
    program_stamp: str | None = "TrueVision Generation Lab / edge_audio_river",
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = utc_now()
    audio_path = audio_path.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    lyrics_path = lyrics_path.resolve() if lyrics_path is not None else None
    run_id = slug(run_id)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = decode_audio_mono(audio_path, sample_rate=sample_rate, max_seconds=max_seconds)
    features = measure_audio_features(samples, sample_rate=sample_rate, fps=fps)
    if max_seconds is not None:
        features = [feature for feature in features if feature["time_seconds"] < max_seconds]
    if not features:
        raise ValueError("Audio produced no renderable features")
    duration_seconds = len(features) / fps
    theme = build_edge_theme(lyrics_path)

    visual_path = run_dir / f"{run_id}_visual_only.mp4"
    final_path = run_dir / f"{run_id}_full_audio.mp4" if mux_audio else visual_path
    state_path = run_dir / f"{run_id}_frame_state.jsonl"
    thumb_path = run_dir / f"{run_id}_thumbnail.jpg"
    manifest_path = run_dir / f"{run_id}_manifest.json"
    report_path = run_dir / f"{run_id}_report.md"

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(visual_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin was not opened")

    trail = np.zeros((height, width, 3), dtype=np.float32)
    sampled_states: list[dict[str, Any]] = []
    thumbnail_frame: np.ndarray | None = None
    with state_path.open("w", encoding="utf-8") as state_handle:
        for index, feature in enumerate(features):
            frame_state = dict(feature)
            frame_state["frame_index"] = index
            frame, metadata = render_river_frame(
                width=width,
                height=height,
                fps=fps,
                frame_state=frame_state,
                trail=trail,
                river_height_ratio=river_height_ratio,
                program_stamp=program_stamp,
            )
            proc.stdin.write(frame.tobytes())
            state_handle.write(json.dumps(metadata, allow_nan=False) + "\n")
            if index % max(1, fps) == 0:
                sampled_states.append(metadata)
            if index == min(len(features) - 1, max(1, fps * 20)):
                thumbnail_frame = frame.copy()
    proc.stdin.close()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg video encoder failed with exit code {return_code}")

    if thumbnail_frame is None:
        thumbnail_frame = np.clip(trail, 0, 255).astype(np.uint8)
    cv2.imwrite(str(thumb_path), thumbnail_frame)

    audio_muxed = False
    if mux_audio:
        mux_cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(visual_path),
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
            "-b:a",
            "192k",
            "-shortest",
            str(final_path),
        ]
        subprocess.run(mux_cmd, check=True)
        audio_muxed = True

    feature_arrays = {key: np.asarray([feature[key] for feature in features], dtype=np.float32) for key in ["rms", "bass", "mid", "high", "beat"]}
    manifest = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "claim": "audio_reactive_color_river_for_edge_of_the_world",
        "boundary": {
            "generated_state_media": "synthetic_not_evidence",
            "lettering_policy": "program_stamp_only" if program_stamp else "none",
            "program_stamp": program_stamp,
            "no_glyphs": True,
            "no_lyric_overlay": True,
        },
        "inputs": {
            "audio_path": str(audio_path),
            "audio_sha256": sha256_file(audio_path),
            "sample_rate": sample_rate,
        },
        "theme": theme,
        "render": {
            "width": width,
            "height": height,
            "fps": fps,
            "frames": len(features),
            "duration_seconds": round(duration_seconds, 6),
            "style": "black_field_swirl_river_windows_snake_screensaver_influence",
            "river_height_ratio": river_height_ratio,
        },
        "audio_feature_summary": {
            key: {
                "mean": round(float(np.mean(values)), 6),
                "max": round(float(np.max(values)), 6),
                "std": round(float(np.std(values)), 6),
            }
            for key, values in feature_arrays.items()
        },
        "sampled_frame_states": sampled_states[:360],
        "hardware": capture_hardware(),
        "outputs": {
            "run_dir": str(run_dir),
            "video_mp4": str(final_path),
            "visual_only_mp4": str(visual_path),
            "audio_muxed": audio_muxed,
            "frame_state_jsonl": str(state_path),
            "thumbnail_jpg": str(thumb_path),
            "manifest_json": str(manifest_path),
            "report_md": str(report_path),
        },
    }
    _write_json(manifest_path, manifest)
    _write_report(report_path, manifest)
    manifest["outputs"]["video_sha256"] = sha256_file(final_path)
    manifest["outputs"]["manifest_sha256"] = sha256_file(manifest_path)
    _write_json(manifest_path, manifest)
    return {
        "run_id": run_id,
        "video_mp4": str(final_path),
        "visual_only_mp4": str(visual_path),
        "audio_muxed": audio_muxed,
        "manifest_json": str(manifest_path),
        "frame_state_jsonl": str(state_path),
        "thumbnail_jpg": str(thumb_path),
        "report_md": str(report_path),
        "frames": len(features),
        "duration_seconds": round(duration_seconds, 6),
        "video_sha256": manifest["outputs"]["video_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Edge Of The World as an audio-reactive TrueVision color river.")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO))
    parser.add_argument("--lyrics", default=str(DEFAULT_LYRICS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--river-height-ratio", type=float, default=0.38)
    parser.add_argument("--program-stamp", default="TrueVision Generation Lab / edge_audio_river")
    parser.add_argument("--no-program-stamp", action="store_true")
    parser.add_argument("--visual-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_edge_audio_river(
        audio_path=Path(args.audio),
        lyrics_path=Path(args.lyrics) if args.lyrics else None,
        output_root=Path(args.output_root),
        run_id=args.run_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
        mux_audio=not args.visual_only,
        river_height_ratio=args.river_height_ratio,
        program_stamp=None if args.no_program_stamp else args.program_stamp,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
