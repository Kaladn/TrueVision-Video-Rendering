"""Project TrueVision capture dynamics across a full audio timeline.

This is not a capture replay loop. It learns a deterministic 6-1-6 temporal
delta grammar from observed TrueVision cell state, then projects that grammar
forward under audio control.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from truevision_state_replay import build_rgb_replay_frame, read_cell_chunk, sha256_file

from .causal_cell_map import CORE_CHANNELS, feature_indices


PROJECTABLE_CHANNELS = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "luma_mean",
    "edge_density",
    "motion_energy",
    "delta_luma_abs",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "saturation_mean",
    "texture_energy",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def slug(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return clean.strip("_")[:96] or "edge_temporal_projection"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_edge_lyrics_block(lyrics_path: Path | None) -> dict[str, Any]:
    """Read the first song block and summarize the visual theme anchors."""
    if not lyrics_path or not lyrics_path.exists():
        return {
            "source": str(lyrics_path) if lyrics_path else None,
            "track_title": "Edge Of The World",
            "anchors": ["edge of the world", "looking down", "river of life", "you me together"],
            "notes": "lyrics file unavailable; default Edge anchors used",
        }
    text = lyrics_path.read_text(encoding="utf-8", errors="ignore")
    next_markers = ["I Am The Machine", "Mirror Made", "The Basement"]
    end = len(text)
    for marker in next_markers:
        index = text.find(marker, 1)
        if index > 0:
            end = min(end, index)
    block = text[:end].strip()
    lowered = block.lower()
    anchors: list[str] = []
    for phrase in [
        "edge of the world",
        "just looking down",
        "nobody hears me",
        "nobody sees me",
        "becoming what i need to be",
        "river of life",
        "colors swirling",
        "you me together",
        "we are the storm inside",
    ]:
        if phrase in lowered:
            anchors.append(phrase)
    title = block.splitlines()[0].strip() if block else "Edge Of The World"
    return {
        "source": str(lyrics_path),
        "track_title": title,
        "line_count": len([line for line in block.splitlines() if line.strip()]),
        "anchors": anchors,
        "visual_arc": [
            "cold edge / isolation",
            "looking down over the rim",
            "truth pressure and system break",
            "river of life below",
            "separate currents unite into one storm",
        ],
    }


def load_capture_cells(run_dir: Path, *, max_frames: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any], dict[str, Any]]:
    """Load TrueVision cell chunks from NPZ or native Rust TVCELLS format."""
    manifest_path = next(run_dir.glob("*_manifest.json"))
    summary_path = next(run_dir.glob("*_summary.json"))
    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    feature_names = list(manifest["cell_state"]["feature_names"])

    cell_batches: list[np.ndarray] = []
    frame_batches: list[np.ndarray] = []
    remaining = max_frames
    for chunk in manifest["cell_state"]["chunks"]:
        cells, frame_numbers = read_cell_chunk(chunk)
        cells = cells.astype(np.float32, copy=False)
        frame_numbers = frame_numbers.astype(np.int32, copy=False)
        if remaining is not None:
            if remaining <= 0:
                break
            cells = cells[:remaining]
            frame_numbers = frame_numbers[:remaining]
            remaining -= int(cells.shape[0])
        if cells.size:
            cell_batches.append(cells)
            frame_batches.append(frame_numbers)
    if not cell_batches:
        raise ValueError(f"no TrueVision cell state frames found in {run_dir}")
    return np.concatenate(cell_batches, axis=0), np.concatenate(frame_batches, axis=0), feature_names, manifest, summary


def _decode_audio_mono(audio_path: Path, *, sample_rate: int, max_seconds: float | None) -> np.ndarray:
    command = ["ffmpeg", "-v", "error", "-i", str(audio_path)]
    if max_seconds is not None:
        command.extend(["-t", f"{max_seconds:.6f}"])
    command.extend(["-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate), "-"])
    completed = subprocess.run(command, check=True, capture_output=True)
    if not completed.stdout:
        return np.zeros(0, dtype=np.float32)
    pcm = np.frombuffer(completed.stdout, dtype="<i2").astype(np.float32)
    return np.clip(pcm / 32768.0, -1.0, 1.0)


def _normalize(values: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    scale = max(float(np.percentile(finite, percentile)), 1.0e-6)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


def audio_soundprint(audio_path: Path, *, fps: int, sample_rate: int, max_seconds: float | None = None) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Create a compact audio feature timeline for projection control."""
    samples = _decode_audio_mono(audio_path, sample_rate=sample_rate, max_seconds=max_seconds)
    if samples.size == 0:
        return [], {"duration_seconds": 0.0, "frame_count": 0}
    duration = samples.size / float(sample_rate)
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
        spectrum = np.abs(np.fft.rfft(segment))
        raw["rms"][frame_index] = float(np.sqrt(np.mean(segment * segment)))
        raw["bass"][frame_index] = float(np.mean(spectrum[bass_mask])) if np.any(bass_mask) else 0.0
        raw["mid"][frame_index] = float(np.mean(spectrum[mid_mask])) if np.any(mid_mask) else 0.0
        raw["high"][frame_index] = float(np.mean(spectrum[high_mask])) if np.any(high_mask) else 0.0

    norm = {key: _normalize(value) for key, value in raw.items()}
    smooth = np.zeros(frame_count, dtype=np.float32)
    last = 0.0
    for index, value in enumerate(norm["rms"]):
        last = 0.72 * last + 0.28 * float(value)
        smooth[index] = last
    onset = np.maximum(0.0, norm["rms"] - np.roll(smooth, 1))
    onset[0] = norm["rms"][0]
    beat = _normalize(onset, 90.0)
    delta = np.maximum(0.0, norm["rms"] - np.roll(norm["rms"], 1))
    delta[0] = norm["rms"][0]

    features: list[dict[str, float]] = []
    for frame_index in range(frame_count):
        features.append(
            {
                "frame_index": float(frame_index),
                "time_seconds": round(frame_index / fps, 6),
                "rms": round(float(norm["rms"][frame_index]), 6),
                "bass": round(float(norm["bass"][frame_index]), 6),
                "mid": round(float(norm["mid"][frame_index]), 6),
                "high": round(float(norm["high"][frame_index]), 6),
                "beat": round(float(beat[frame_index]), 6),
                "rise": round(float(delta[frame_index]), 6),
            }
        )

    sections = max(1, int(math.ceil(duration / 8.0)))
    section_energy: list[dict[str, float]] = []
    for section in range(sections):
        start = int(section * 8.0 * fps)
        end = min(frame_count, int((section + 1) * 8.0 * fps))
        if end <= start:
            continue
        section_energy.append(
            {
                "start_seconds": round(start / fps, 3),
                "end_seconds": round(end / fps, 3),
                "mean_rms": round(float(np.mean(norm["rms"][start:end])), 6),
                "max_beat": round(float(np.max(beat[start:end])), 6),
            }
        )
    summary = {
        "duration_seconds": round(duration, 6),
        "fps": fps,
        "frame_count": frame_count,
        "average_level": round(float(np.mean(norm["rms"])), 6),
        "max_level": round(float(np.max(norm["rms"])), 6),
        "peak_count": int(np.count_nonzero(beat > 0.62)),
        "valley_count": int(np.count_nonzero(norm["rms"] < 0.18)),
        "section_count": len(section_energy),
        "section_energy": section_energy[:12],
    }
    return features, summary


@dataclass
class TemporalProjectionProfile:
    feature_names: list[str]
    channel_indices: dict[str, int]
    base_state: np.ndarray
    observed_mean: np.ndarray
    observed_min: np.ndarray
    observed_max: np.ndarray
    delta_library: np.ndarray
    source_frame_numbers: np.ndarray
    radius: int
    summary: dict[str, Any]


def build_temporal_projection_profile(
    source_cells: np.ndarray,
    frame_numbers: np.ndarray,
    *,
    feature_names: list[str],
    radius: int = 6,
) -> TemporalProjectionProfile:
    """Build 6-1-6 projection deltas from observed cell-state history."""
    if source_cells.ndim != 4:
        raise ValueError("source_cells must have shape frame,row,col,feature")
    if source_cells.shape[0] < (radius * 2 + 2):
        raise ValueError("projection requires at least 14 observed frames for radius=6")
    feature_indices(feature_names)
    channel_names = [name for name in PROJECTABLE_CHANNELS if name in feature_names]
    channel_indices = {name: feature_names.index(name) for name in channel_names}

    observed_mean = np.mean(source_cells, axis=0).astype(np.float32)
    observed_min = np.min(source_cells, axis=0).astype(np.float32)
    observed_max = np.max(source_cells, axis=0).astype(np.float32)
    deltas: list[np.ndarray] = []
    trace_rows: list[dict[str, Any]] = []
    for index in range(radius, source_cells.shape[0] - radius - 1):
        past = source_cells[index - radius : index]
        future = source_cells[index + 1 : index + radius + 1]
        direct = source_cells[index + 1] - source_cells[index]
        cause_effect = (np.mean(future, axis=0) - np.mean(past, axis=0)) / float(radius)
        delta = (0.65 * direct + 0.35 * cause_effect).astype(np.float32)
        deltas.append(delta)
        if len(trace_rows) < 24:
            trace_rows.append(
                {
                    "center_source_frame": int(frame_numbers[index]),
                    "prior_frames": [int(value) for value in frame_numbers[index - radius : index].tolist()],
                    "future_frames": [int(value) for value in frame_numbers[index + 1 : index + radius + 1].tolist()],
                    "mean_abs_delta": round(float(np.mean(np.abs(delta))), 6),
                }
            )
    if not deltas:
        raise ValueError("no 6-1-6 deltas could be built")
    delta_library = np.stack(deltas, axis=0).astype(np.float32)
    rgb_indices = [feature_names.index(name) for name in ("rgb_mean_r", "rgb_mean_g", "rgb_mean_b")]
    summary = {
        "radius": radius,
        "source_frames": int(source_cells.shape[0]),
        "delta_frames": int(delta_library.shape[0]),
        "cell_grid_shape": [int(source_cells.shape[1]), int(source_cells.shape[2])],
        "feature_count": int(source_cells.shape[3]),
        "projectable_channels": channel_names,
        "mean_abs_delta": round(float(np.mean(np.abs(delta_library[:, :, :, rgb_indices]))), 6),
        "trace_sample": trace_rows,
        "projection_rule": "mix_6_1_6_delta_fields_under_audio_control_not_source_frame_loop",
    }
    return TemporalProjectionProfile(
        feature_names=feature_names,
        channel_indices=channel_indices,
        base_state=source_cells[0].astype(np.float32).copy(),
        observed_mean=observed_mean,
        observed_min=observed_min,
        observed_max=observed_max,
        delta_library=delta_library,
        source_frame_numbers=frame_numbers.astype(np.int32, copy=True),
        radius=radius,
        summary=summary,
    )


def _audio_value(feature: dict[str, float], key: str) -> float:
    return float(feature.get(key, 0.0))


def _clamp_projected_state(state: np.ndarray, profile: TemporalProjectionProfile) -> np.ndarray:
    for name, index in profile.channel_indices.items():
        low = profile.observed_min[:, :, index]
        high = profile.observed_max[:, :, index]
        if name.startswith("rgb_") or name in {"luma_mean", "hsv_mean_v"}:
            low = np.maximum(0.0, low - 18.0)
            high = np.minimum(255.0, high + 18.0)
        else:
            high = np.maximum(high * 1.55 + 1.0e-4, 1.0)
            low = np.maximum(0.0, low * 0.35)
        state[:, :, index] = np.clip(state[:, :, index], low, high)
    return state


def project_state_sequence(
    profile: TemporalProjectionProfile,
    audio_features: Iterable[dict[str, float]],
    *,
    trace_every: int = 120,
) -> Iterable[tuple[np.ndarray, dict[str, Any]]]:
    """Yield projected states; each frame is caused by prior projected state plus learned deltas."""
    current = profile.base_state.copy()
    delta_count = int(profile.delta_library.shape[0])
    phase = 0.0
    rgb_indices = [profile.feature_names.index(name) for name in ("rgb_mean_r", "rgb_mean_g", "rgb_mean_b")]
    luma_index = profile.feature_names.index("luma_mean")
    edge_index = profile.feature_names.index("edge_density") if "edge_density" in profile.feature_names else None
    motion_index = profile.feature_names.index("motion_energy") if "motion_energy" in profile.feature_names else None
    delta_luma_index = profile.feature_names.index("delta_luma_abs") if "delta_luma_abs" in profile.feature_names else None

    for frame_index, audio in enumerate(audio_features):
        rms = _audio_value(audio, "rms")
        bass = _audio_value(audio, "bass")
        mid = _audio_value(audio, "mid")
        high = _audio_value(audio, "high")
        beat = _audio_value(audio, "beat")
        rise = _audio_value(audio, "rise")
        phase += 0.72 + 0.62 * rms + 0.24 * beat + 0.16 * bass
        a = int(phase) % delta_count
        b = (a + profile.radius + 1 + int(beat * 9.0) + int(frame_index * 0.037)) % delta_count
        mix = 0.52 + 0.26 * math.sin(frame_index * 0.013 + mid * math.tau)
        delta = profile.delta_library[a] * mix + profile.delta_library[b] * (1.0 - mix)
        shift_x = int(round(math.sin(frame_index * 0.017 + bass * 2.0) * (1.0 + 2.0 * rms)))
        shift_y = int(round(math.cos(frame_index * 0.011 + high * 2.0) * (1.0 + 1.5 * beat)))
        if shift_x or shift_y:
            delta = np.roll(delta, shift=(shift_y, shift_x), axis=(0, 1))

        drift_scale = 0.62 + 0.52 * rms + 0.22 * rise
        current += delta * drift_scale
        current += (profile.observed_mean - current) * (0.0025 + 0.0035 * (1.0 - rms))
        current[:, :, rgb_indices] += (beat * 2.8 + high * 1.2 - 0.55) * np.array([0.62, 0.74, 0.92], dtype=np.float32)
        current[:, :, luma_index] += beat * 2.1 + bass * 0.65 - 0.42
        if edge_index is not None:
            current[:, :, edge_index] += (high * 0.024 + beat * 0.018) * (1.0 + rms)
        if motion_index is not None:
            current[:, :, motion_index] = current[:, :, motion_index] * 0.965 + (rms + beat) * 0.018
        if delta_luma_index is not None:
            current[:, :, delta_luma_index] = current[:, :, delta_luma_index] * 0.94 + rise * 0.035
        _clamp_projected_state(current, profile)

        trace: dict[str, Any] = {}
        if trace_every > 0 and frame_index % trace_every == 0:
            trace = {
                "output_frame": frame_index,
                "time_seconds": float(audio.get("time_seconds", frame_index)),
                "delta_a": a,
                "delta_b": b,
                "delta_mix": round(float(mix), 6),
                "audio": {key: round(float(audio.get(key, 0.0)), 6) for key in ["rms", "bass", "mid", "high", "beat", "rise"]},
                "mean_rgb": [round(float(np.mean(current[:, :, idx])), 6) for idx in rgb_indices],
                "mean_luma": round(float(np.mean(current[:, :, luma_index])), 6),
                "rule": "project_prior_state_plus_mixed_6_1_6_deltas",
            }
        yield current.copy(), trace


def _ffmpeg_writer(path: Path, *, width: int, height: int, fps: int, audio_path: Path | None, duration: float, mux_audio: bool) -> subprocess.Popen:
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    if mux_audio and audio_path is not None:
        command.extend(["-i", str(audio_path), "-t", f"{duration:.6f}", "-map", "0:v:0", "-map", "1:a:0"])
    command.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"])
    if mux_audio and audio_path is not None:
        command.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
    command.append(str(path))
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _flame_polyline(
    *,
    width: int,
    height: int,
    seed: int,
    base_x: float,
    base_y: float,
    length: float,
    sway: float,
) -> np.ndarray:
    points: list[list[int]] = []
    for step in range(9):
        t = step / 8.0
        x = base_x + math.sin(seed * 1.37 + t * 8.0 + sway) * width * (0.006 + 0.014 * t)
        y = base_y - length * t
        points.append([int(round(x)), int(round(y))])
    return np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))


def _scale_rgb_saturation(frame: np.ndarray, scale: float) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(scale), 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _smooth_pulse(time_seconds: float, center_seconds: float, width_seconds: float) -> float:
    distance = abs(time_seconds - center_seconds)
    if distance >= width_seconds:
        return 0.0
    x = 1.0 - distance / max(width_seconds, 1.0e-6)
    return float(x * x * (3.0 - 2.0 * x))


def _draw_lightning_bolt(
    layer: np.ndarray,
    *,
    seed: int,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    intensity: float,
) -> None:
    height, width = layer.shape[:2]
    rng_phase = seed * 12.9898
    points: list[list[int]] = []
    segments = 10
    for step in range(segments + 1):
        t = step / float(segments)
        wobble = math.sin(rng_phase + step * 2.13) * width * 0.018 * (1.0 - abs(0.5 - t) * 1.4)
        branch = math.sin(rng_phase * 0.31 + step * 5.7) * width * 0.006
        x = int(round(start_x * (1.0 - t) + end_x * t + wobble + branch))
        y = int(round(start_y * (1.0 - t) + end_y * t))
        points.append([max(0, min(width - 1, x)), max(0, min(height - 1, y))])
    polyline = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    core = (255, int(220 + 35 * intensity), int(150 + 85 * intensity))
    glow = (255, int(84 + 110 * intensity), int(24 + 70 * intensity))
    cv2.polylines(layer, [polyline], False, glow, max(5, int(11 * intensity)), cv2.LINE_AA)
    cv2.polylines(layer, [polyline], False, core, max(1, int(3 + 3 * intensity)), cv2.LINE_AA)

    for branch_index in range(2):
        attach = 3 + branch_index * 3
        if attach >= len(points) - 2:
            continue
        bx, by = points[attach]
        direction = -1 if (seed + branch_index) % 2 else 1
        branch_end = [
            max(0, min(width - 1, bx + int(direction * width * (0.035 + 0.025 * intensity)))),
            max(0, min(height - 1, by + int(height * (0.045 + 0.025 * branch_index)))),
        ]
        branch_points = np.asarray([points[attach], branch_end], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(layer, [branch_points], False, glow, max(2, int(5 * intensity)), cv2.LINE_AA)
        cv2.polylines(layer, [branch_points], False, core, 1, cv2.LINE_AA)


def _draw_signature_lightning(
    layer: np.ndarray,
    *,
    signature: dict[str, Any],
    intensity: float,
    frame_index: int,
) -> int:
    hot_cells = signature.get("hot_cells") or []
    if not hot_cells:
        return 0
    height, width = layer.shape[:2]
    points: list[tuple[int, int, float]] = []
    for cell in hot_cells[:320]:
        try:
            x = int(float(cell["x_norm"]) * width)
            y = int(float(cell["y_norm"]) * height)
            value = float(cell.get("intensity", 0.0))
        except (TypeError, ValueError, KeyError):
            continue
        if value <= 0.08:
            continue
        x += int(round(math.sin(frame_index * 0.07 + y * 0.013) * width * 0.004 * intensity))
        y += int(round(math.cos(frame_index * 0.05 + x * 0.009) * height * 0.003 * intensity))
        points.append((max(0, min(width - 1, x)), max(0, min(height - 1, y)), value))
    if not points:
        return 0
    points.sort(key=lambda item: (item[1], item[0]))
    glow = (255, int(70 + 120 * intensity), int(20 + 80 * intensity))
    core = (255, int(215 + 40 * intensity), int(150 + 80 * intensity))
    for x, y, value in points:
        radius = max(1, int(1 + 4 * value * intensity))
        cv2.circle(layer, (x, y), radius + 2, glow, -1, cv2.LINE_AA)
        cv2.circle(layer, (x, y), radius, core, -1, cv2.LINE_AA)
    stride = max(1, len(points) // 26)
    for left, right in zip(points[::stride], points[stride::stride]):
        if abs(left[0] - right[0]) < width * 0.16 and abs(left[1] - right[1]) < height * 0.16:
            cv2.line(layer, (left[0], left[1]), (right[0], right[1]), glow, max(1, int(3 * intensity)), cv2.LINE_AA)
            cv2.line(layer, (left[0], left[1]), (right[0], right[1]), core, 1, cv2.LINE_AA)
    return len(points)


def apply_projection_visual_style(
    frame: np.ndarray,
    *,
    visual_style: str,
    audio: dict[str, float],
    frame_index: int,
    fps: int,
    duration_seconds: float,
    lightning_signature: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply optional cinematic style after 6-1-6 state projection."""
    if visual_style in {"", "projection", "raw_projection"}:
        return frame, {"visual_style": "raw_projection", "style_applied": False}
    if visual_style != "hell_power_walk":
        raise ValueError(f"unknown projection visual style: {visual_style}")

    height, width = frame.shape[:2]
    rms = _audio_value(audio, "rms")
    bass = _audio_value(audio, "bass")
    high = _audio_value(audio, "high")
    beat = _audio_value(audio, "beat")
    time_seconds = float(audio.get("time_seconds", frame_index / max(1, fps)))
    norm_time = 0.0 if duration_seconds <= 0 else max(0.0, min(1.0, time_seconds / duration_seconds))
    strike_pressure = max(0.0, min(1.0, (beat - 0.56) * 1.65 + high * 0.28 + bass * 0.12))
    section_seconds = max(5.0, duration_seconds / 7.0)
    transition_centers = [section_seconds * index for index in range(1, 7)]
    transition_flash = max((_smooth_pulse(time_seconds, center, 0.42) for center in transition_centers), default=0.0)
    peak_flash = _smooth_pulse((time_seconds * 2.0) % 1.0, 0.0, 0.12) * strike_pressure
    flash = min(1.0, transition_flash * 0.52 + peak_flash * 0.38)

    base = frame.astype(np.float32) / 255.0
    gray_u8 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur_small = cv2.GaussianBlur(gray_u8, (0, 0), 1.1)
    blur_large = cv2.GaussianBlur(gray_u8, (0, 0), 5.5)
    dog = cv2.absdiff(blur_small, blur_large)
    canny = cv2.Canny(gray_u8, 34, 118)
    scharr_x = cv2.Scharr(gray_u8, cv2.CV_32F, 1, 0)
    scharr_y = cv2.Scharr(gray_u8, cv2.CV_32F, 0, 1)
    scharr = np.clip(cv2.magnitude(scharr_x, scharr_y) / 255.0, 0.0, 1.0)
    lap = np.abs(cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)) / 255.0
    edge = np.clip(
        (canny.astype(np.float32) / 255.0) * 0.52
        + _normalize(dog.astype(np.float32), 94.0) * 0.35
        + scharr * 0.30
        + np.clip(lap, 0.0, 1.0) * 0.24,
        0.0,
        1.0,
    )
    edge_glow = cv2.GaussianBlur(edge, (0, 0), 3.0 + 3.0 * beat)

    luma = gray_u8.astype(np.float32) / 255.0
    ember = np.zeros_like(base)
    ember[:, :, 0] = np.clip(luma * (1.25 + 0.45 * beat) + edge_glow * (1.7 + high), 0.0, 1.0)
    ember[:, :, 1] = np.clip(luma * (0.30 + 0.24 * rms) + edge * (0.46 + 0.28 * beat), 0.0, 1.0)
    ember[:, :, 2] = np.clip(luma * 0.08 + edge * 0.12, 0.0, 1.0)
    shadows = np.clip((1.0 - luma) ** (1.25 + 0.5 * bass), 0.0, 1.0)
    styled = np.clip(ember * (0.74 + 0.30 * rms) + base * np.dstack([0.15 * luma, 0.05 * luma, 0.03 * luma]), 0.0, 1.0)
    styled *= np.dstack([1.0 - shadows * 0.06, 1.0 - shadows * 0.32, 1.0 - shadows * 0.54])

    # Heat haze: deterministic horizontal shimmer, strongest near the lower half.
    rows = np.arange(height, dtype=np.float32)
    heat_mask = np.clip((rows - height * 0.28) / max(1.0, height * 0.72), 0.0, 1.0)[:, None]
    distorted = np.empty_like(styled)
    for y in range(height):
        offset = int(round(math.sin(y * 0.035 + time_seconds * (2.2 + rms)) * (1.0 + 5.0 * heat_mask[y, 0] * (0.5 + beat))))
        distorted[y] = np.roll(styled[y], offset, axis=0)
    styled = distorted * 0.82 + styled * 0.18

    out = np.clip(styled * 255.0, 0, 255).astype(np.uint8)

    overlay = np.zeros_like(out, dtype=np.uint8)
    figure_scale = 0.78 - 0.18 * norm_time
    foot_y = int(height * (0.88 - 0.06 * norm_time))
    center_x = int(width * (0.50 + 0.018 * math.sin(time_seconds * 0.38)))
    body_h = max(24, int(height * 0.30 * figure_scale))
    head_r = max(6, int(height * 0.030 * figure_scale))
    torso_w = max(10, int(height * 0.060 * figure_scale))
    shoulder_y = foot_y - int(body_h * 0.68)
    hip_y = foot_y - int(body_h * 0.36)
    head_y = shoulder_y - int(head_r * 1.55)
    gait = math.sin(time_seconds * 3.4)

    aura = np.zeros_like(out, dtype=np.uint8)
    power = 0.55 + 0.45 * max(rms, beat)
    for strand in range(20):
        angle = (strand - 9.5) / 9.5
        base_x = center_x + angle * torso_w * (1.2 + 0.4 * math.sin(strand))
        length = height * (0.18 + 0.20 * power) * (0.72 + 0.25 * math.sin(strand * 1.7))
        points = _flame_polyline(
            width=width,
            height=height,
            seed=strand,
            base_x=base_x,
            base_y=foot_y - body_h * 0.10,
            length=length,
            sway=time_seconds * (1.4 + 0.4 * high),
        )
        color = (255, int(42 + 96 * beat), int(4 + 30 * high))
        cv2.polylines(aura, [points], False, color, max(2, int(3 + 5 * power)), cv2.LINE_AA)
    aura = cv2.GaussianBlur(aura, (0, 0), 10.0 + 8.0 * power)

    cv2.circle(overlay, (center_x, head_y), head_r, (0, 0, 0), -1, cv2.LINE_AA)
    torso = np.asarray(
        [
            [center_x - torso_w, shoulder_y],
            [center_x + torso_w, shoulder_y],
            [center_x + int(torso_w * 0.72), hip_y],
            [center_x - int(torso_w * 0.72), hip_y],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(overlay, torso, (0, 0, 0), cv2.LINE_AA)
    arm_swing = int(torso_w * 0.85 * gait)
    leg_swing = int(torso_w * 1.08 * gait)
    cv2.line(overlay, (center_x - torso_w, shoulder_y + 5), (center_x - torso_w - arm_swing, hip_y + 4), (0, 0, 0), max(4, int(8 * figure_scale)), cv2.LINE_AA)
    cv2.line(overlay, (center_x + torso_w, shoulder_y + 5), (center_x + torso_w + arm_swing, hip_y + 4), (0, 0, 0), max(4, int(8 * figure_scale)), cv2.LINE_AA)
    cv2.line(overlay, (center_x - int(torso_w * 0.42), hip_y), (center_x - int(torso_w * 0.78) - leg_swing, foot_y), (0, 0, 0), max(5, int(9 * figure_scale)), cv2.LINE_AA)
    cv2.line(overlay, (center_x + int(torso_w * 0.42), hip_y), (center_x + int(torso_w * 0.78) + leg_swing, foot_y), (0, 0, 0), max(5, int(9 * figure_scale)), cv2.LINE_AA)

    silhouette_mask = cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY)
    rim = cv2.dilate(silhouette_mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
    rim = cv2.GaussianBlur(rim, (0, 0), 4.0)
    rim_rgb = np.zeros_like(out)
    rim_rgb[:, :, 0] = np.clip(rim.astype(np.float32) * (1.3 + beat), 0, 255).astype(np.uint8)
    rim_rgb[:, :, 1] = np.clip(rim.astype(np.float32) * 0.24, 0, 255).astype(np.uint8)
    out = cv2.addWeighted(out, 1.0, aura, 0.62, 0)
    out = cv2.addWeighted(out, 1.0, rim_rgb, 0.42, 0)
    out[silhouette_mask > 10] = 0

    signature_points_applied = 0
    if strike_pressure > 0.24:
        lightning = np.zeros_like(out, dtype=np.uint8)
        if lightning_signature:
            signature_points_applied = _draw_signature_lightning(
                lightning,
                signature=lightning_signature,
                intensity=strike_pressure,
                frame_index=frame_index,
            )
        if signature_points_applied == 0:
            strike_count = 1 + int(strike_pressure > 0.72)
            for bolt in range(strike_count):
                seed = int(frame_index * 17 + bolt * 101 + round(strike_pressure * 1000))
                start_x = int(width * (0.18 + 0.64 * ((math.sin(seed) + 1.0) * 0.5)))
                end_x = int(width * (0.40 + 0.20 * bolt + 0.08 * math.sin(seed * 0.17)))
                _draw_lightning_bolt(
                    lightning,
                    seed=seed,
                    start_x=start_x,
                    start_y=0,
                    end_x=end_x,
                    end_y=int(height * (0.30 + 0.10 * beat)),
                    intensity=strike_pressure,
                )
        lightning = cv2.GaussianBlur(lightning, (0, 0), 0.65)
        out = cv2.addWeighted(out, 1.0, lightning, 0.42 + 0.40 * strike_pressure, 0)

    if flash > 0.0:
        red_transition = np.full_like(out, (210, 18, 10), dtype=np.uint8)
        orange_peak = np.full_like(out, (255, 148, 54), dtype=np.uint8)
        flash_color = cv2.addWeighted(red_transition, 0.70 + 0.20 * bass, orange_peak, 0.30 - 0.20 * bass, 0)
        out = cv2.addWeighted(out, 1.0 - 0.30 * flash, flash_color, 0.30 * flash, 0)

    vignette_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    vignette_y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    vignette = np.clip(1.0 - 0.42 * (vignette_x * vignette_x + vignette_y * vignette_y), 0.38, 1.0)
    out = np.clip(out.astype(np.float32) * vignette[:, :, None], 0, 255).astype(np.uint8)
    saturation_scale = 0.75
    out = _scale_rgb_saturation(out, saturation_scale)
    meta = {
        "visual_style": "hell_power_walk",
        "style_applied": True,
        "edge_filters": ["canny", "scharr_gradient", "laplacian", "difference_of_gaussians", "edge_glow"],
        "grade": "black_red_orange_ember",
        "saturation_scale": saturation_scale,
        "smooth_transition_flash": round(float(transition_flash), 6),
        "music_lighting_strike_pressure": round(float(strike_pressure), 6),
        "flash_intensity": round(float(flash), 6),
        "red_rhythm_transition": True,
        "lightning_signature_applied": bool(lightning_signature),
        "lightning_signature_points_applied": int(signature_points_applied),
        "silhouette": "walking_away_power_projection",
        "synthetic_overlay": True,
    }
    return out, meta


def project_capture_to_audio(
    *,
    capture_run_dir: Path,
    audio_path: Path,
    lyrics_path: Path | None,
    output_root: Path,
    run_id: str,
    width: int,
    height: int,
    fps: int = 12,
    sample_rate: int = 16000,
    radius: int = 6,
    max_seconds: float | None = None,
    mux_audio: bool = True,
    max_source_frames: int | None = None,
    visual_style: str = "projection",
    lightning_signature_path: Path | None = None,
) -> dict[str, Any]:
    """Project a TrueVision atmospheric capture across a song-length audio timeline."""
    started = utc_now()
    output_dir = output_root / slug(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_cells, frame_numbers, feature_names, source_manifest, source_summary = load_capture_cells(
        capture_run_dir,
        max_frames=max_source_frames,
    )
    profile = build_temporal_projection_profile(source_cells, frame_numbers, feature_names=feature_names, radius=radius)
    audio_features, audio_summary = audio_soundprint(audio_path, fps=fps, sample_rate=sample_rate, max_seconds=max_seconds)
    if not audio_features:
        raise ValueError("audio soundprint produced no frames")
    duration = min(float(audio_summary["duration_seconds"]), float(max_seconds)) if max_seconds is not None else float(audio_summary["duration_seconds"])
    lyrics = read_edge_lyrics_block(lyrics_path)
    lightning_signature = _read_json(lightning_signature_path) if lightning_signature_path and lightning_signature_path.exists() else None

    video_path = output_dir / f"{slug(run_id)}_temporal_projection.mp4"
    trace_path = output_dir / f"{slug(run_id)}_temporal_616_projection_trace.jsonl"
    soundprint_path = output_dir / f"{slug(run_id)}_audio_soundprint.json"
    profile_path = output_dir / f"{slug(run_id)}_projection_profile_summary.json"
    report_path = output_dir / f"{slug(run_id)}_projection_report.md"
    manifest_path = output_dir / f"{slug(run_id)}_manifest.json"

    _write_json(soundprint_path, {"audio_path": str(audio_path), "summary": audio_summary, "sample_frames": audio_features[:24]})
    _write_json(profile_path, profile.summary)

    frame_shape = tuple(int(value) for value in source_summary["geometry"]["frame_shape"])
    output_shape = (height, width)
    if output_shape[0] % source_cells.shape[1] != 0 or output_shape[1] % source_cells.shape[2] != 0:
        raise ValueError("output width/height must divide evenly by the TrueVision cell grid")

    proc = _ffmpeg_writer(video_path, width=width, height=height, fps=fps, audio_path=audio_path, duration=duration, mux_audio=mux_audio)
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin was not opened")

    trace_count = 0
    frame_count = 0
    style_metadata: dict[str, Any] = {"visual_style": visual_style, "style_applied": False}
    try:
        with trace_path.open("w", encoding="utf-8") as trace_handle:
            for audio, (cells, trace) in zip(audio_features, project_state_sequence(profile, audio_features, trace_every=max(1, fps * 5))):
                frame = build_rgb_replay_frame(cells, feature_names=feature_names, output_shape=output_shape)
                frame, style_metadata = apply_projection_visual_style(
                    frame,
                    visual_style=visual_style,
                    audio=audio,
                    frame_index=frame_count,
                    fps=fps,
                    duration_seconds=duration,
                    lightning_signature=lightning_signature,
                )
                proc.stdin.write(frame.tobytes())
                frame_count += 1
                if trace:
                    trace["visual_style"] = style_metadata
                    trace_handle.write(json.dumps(trace, allow_nan=False) + "\n")
                    trace_count += 1
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg projection encode failed with exit code {return_code}")

    completed = utc_now()
    manifest = {
        "schema_version": 1,
        "kind": "truevision_temporal_causality_projection_manifest",
        "run_id": slug(run_id),
        "created_at_utc": started,
        "completed_at_utc": completed,
        "claim": "TrueVision capture dynamics projected across Edge Of The World audio using 6-1-6 temporal causality.",
        "law": [
            "Forward TrueVision witnesses observed state.",
            "6-1-6 explains temporal causality from observed state.",
            "This output projects learned state dynamics; it is synthetic state media, not evidence.",
        ],
        "source": {
            "capture_run_dir": str(capture_run_dir),
            "source_run_id": source_manifest.get("run_id"),
            "source_record_kind": source_manifest.get("record_kind"),
            "source_frame_count_loaded": int(source_cells.shape[0]),
            "source_frame_shape": list(frame_shape),
            "source_cell_grid": [int(source_cells.shape[1]), int(source_cells.shape[2])],
            "source_raw_frames_available": bool(source_manifest.get("boundary", {}).get("raw_frame_saved", False)),
        },
        "audio": {
            "path": str(audio_path),
            "sha256": sha256_file(audio_path),
            "soundprint_summary": audio_summary,
        },
        "lyrics": lyrics,
        "projection": {
            "method": "deterministic_recurrent_projection_from_mixed_6_1_6_delta_fields",
            "not_clone": True,
            "not_copy": True,
            "not_source_frame_loop": True,
            "radius": radius,
            "fps": fps,
            "duration_seconds": round(duration, 6),
            "frame_count": frame_count,
            "trace_rows": trace_count,
            "teacher_profile": profile.summary,
        },
        "render": {
            "width": width,
            "height": height,
            "mux_audio": mux_audio,
            "encoder": "ffmpeg libx264 raw RGB pipe",
            "visual_style": style_metadata,
            "lightning_signature": {
                "enabled": bool(lightning_signature),
                "path": str(lightning_signature_path) if lightning_signature_path else None,
                "signature_id": lightning_signature.get("signature_id") if lightning_signature else None,
                "hot_cell_count": lightning_signature.get("hot_cell_count") if lightning_signature else None,
            },
        },
        "outputs": {
            "video_mp4": str(video_path),
            "video_sha256": sha256_file(video_path),
            "trace_jsonl": str(trace_path),
            "soundprint_json": str(soundprint_path),
            "projection_profile_json": str(profile_path),
            "report_md": str(report_path),
        },
    }
    _write_json(manifest_path, manifest)
    report_path.write_text(
        "\n".join(
            [
                f"# {slug(run_id)} TrueVision 6-1-6 Projection Report",
                "",
                "## Boundary",
                "",
                "This is synthetic projection from observed TrueVision state. It is not evidence and it is not a raw-video clone.",
                "",
                "## Lyric Reading",
                "",
                f"- Track: `{lyrics['track_title']}`",
                f"- Anchors: `{', '.join(lyrics.get('anchors') or [])}`",
                f"- Visual arc: `{ ' -> '.join(lyrics.get('visual_arc') or []) }`",
                "",
                "## Sound Print",
                "",
                f"- Duration: `{audio_summary['duration_seconds']}s`",
                f"- Frame count: `{audio_summary['frame_count']}` at `{fps}fps`",
                f"- Average level: `{audio_summary['average_level']}`",
                f"- Peaks: `{audio_summary['peak_count']}`",
                f"- Valleys: `{audio_summary['valley_count']}`",
                "",
                "## Projection",
                "",
                f"- Source capture: `{capture_run_dir}`",
                f"- Source frames loaded: `{source_cells.shape[0]}`",
                f"- 6-1-6 delta frames: `{profile.summary['delta_frames']}`",
                f"- Output video: `{video_path}`",
                f"- Visual style: `{style_metadata['visual_style']}`",
                "",
                "## Hard Rule",
                "",
                "Do not clone/copy/loop source frames. Project prior state using mixed 6-1-6 delta fields and audio pressure.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"manifest_json": str(manifest_path), **manifest["outputs"]}
