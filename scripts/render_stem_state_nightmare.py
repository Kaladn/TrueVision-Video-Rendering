from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import time
import wave
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "stem_state_nightmare"
SPECTRUM_BAND_COUNT = 16
STEM_ORDER = ("Drums", "Bass", "Guitar", "Vocals", "Synth", "FX")


def build_generation_banner() -> str:
    return (
        "CORTEX EVOLVED  /  TRUEVISION STATE GENERATION  /  LOCAL FIRST  /  "
        "AUDIO STATE DRIVES GRAPHICS  /  RECEIPT-BACKED CREATION"
    )


def build_stem_control_map() -> dict[str, list[str]]:
    return {
        "Drums": ["paired_spectrum_impulse", "edge_frame_hits", "floor_pressure"],
        "Bass": ["low_band_pair", "sub_depth_shadow", "bottom_gradient_mass"],
        "Guitar": ["mid_high_band_pair", "side_chiclets", "edge_meter"],
        "Vocals": ["center_meter", "focus_bloom", "vocal_band_pair"],
        "Synth": ["upper_band_pair", "gradient_field", "mirror_balance"],
        "FX": ["transient_chiclets", "scanline_meter", "spark_ticks"],
    }


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (values - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            data[:, 0].astype(np.int32)
            | (data[:, 1].astype(np.int32) << 8)
            | (data[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values | ~0xFFFFFF, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def _read_wav_bytes(payload: bytes, *, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        if max_seconds is not None:
            frame_count = min(frame_count, max(1, int(sample_rate * max_seconds)))
        raw = wav.readframes(frame_count)

    samples = _decode_pcm(raw, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.clip(samples.astype(np.float32), -1.0, 1.0), sample_rate


def read_wav_file(path: Path, *, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    return _read_wav_bytes(path.read_bytes(), max_seconds=max_seconds)


def infer_wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return float(wav.getnframes()) / float(wav.getframerate())


def _resample_to_rate(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(samples) == 0:
        return samples.astype(np.float32)
    duration = len(samples) / float(source_rate)
    target_len = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, len(samples), endpoint=False)
    target_x = np.linspace(0.0, duration, target_len, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def _fit_length(samples: np.ndarray, target_len: int) -> np.ndarray:
    if len(samples) >= target_len:
        return samples[:target_len].astype(np.float32)
    out = np.zeros(target_len, dtype=np.float32)
    out[: len(samples)] = samples.astype(np.float32)
    return out


def load_stem_samples(
    stems_zip: Path,
    *,
    target_rate: int,
    target_len: int,
    max_seconds: float | None = None,
) -> dict[str, np.ndarray]:
    stems: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(stems_zip, "r") as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".wav") and "__macosx" not in name.lower()
        ]
        for stem_name in STEM_ORDER:
            chosen = next((name for name in members if stem_name.lower() in Path(name).stem.lower()), None)
            if not chosen:
                stems[stem_name] = np.zeros(target_len, dtype=np.float32)
                continue
            payload = archive.read(chosen)
            samples, sample_rate = _read_wav_bytes(payload, max_seconds=max_seconds)
            samples = _resample_to_rate(samples, sample_rate, target_rate)
            stems[stem_name] = _fit_length(samples, target_len)
    return stems


def _band_edges(sample_rate: int) -> np.ndarray:
    nyquist = sample_rate / 2.0
    high = min(16000.0, max(120.0, nyquist * 0.92))
    low = min(35.0, high * 0.25)
    return np.geomspace(low, high, SPECTRUM_BAND_COUNT + 1)


def _frame_spectrum(chunk: np.ndarray, sample_rate: int, edges: np.ndarray) -> list[float]:
    if len(chunk) < 4:
        return [0.0] * SPECTRUM_BAND_COUNT
    window = np.hanning(len(chunk)).astype(np.float32)
    fft = np.fft.rfft(chunk * window)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sample_rate)
    bands: list[float] = []
    for index in range(SPECTRUM_BAND_COUNT):
        low = edges[index]
        high = edges[index + 1]
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            bands.append(0.0)
        else:
            bands.append(float(np.mean(mag[mask])))
    return bands


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    scale = float(np.percentile(values, 95))
    if scale <= 1e-8:
        scale = float(np.max(values))
    if scale <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / scale, 0.0, 1.35).astype(np.float32)


def _source_metrics(
    samples: np.ndarray,
    *,
    sample_rate: int,
    fps: int,
    frame_count: int,
) -> dict[str, Any]:
    hop = max(1, int(round(sample_rate / fps)))
    edges = _band_edges(sample_rate)
    rms_values = np.zeros(frame_count, dtype=np.float32)
    raw_bands = np.zeros((frame_count, SPECTRUM_BAND_COUNT), dtype=np.float32)

    for frame_index in range(frame_count):
        start = frame_index * hop
        end = min(len(samples), start + hop)
        chunk = samples[start:end]
        if len(chunk) == 0:
            continue
        rms_values[frame_index] = float(np.sqrt(np.mean(np.square(chunk))))
        raw_bands[frame_index] = np.asarray(_frame_spectrum(chunk, sample_rate, edges), dtype=np.float32)

    onset_values = np.maximum(0.0, np.diff(rms_values, prepend=rms_values[0]))
    return {
        "rms": _normalize(rms_values),
        "onset": _normalize(onset_values),
        "bands": _normalize(raw_bands),
    }


def compute_frame_metrics(
    samples: dict[str, np.ndarray],
    *,
    sample_rate: int,
    fps: int,
    duration: float,
) -> dict[str, Any]:
    frame_count = max(1, int(round(duration * fps)))
    master_samples = samples.get("Master")
    if master_samples is None:
        raise ValueError("samples must include a Master source")

    master_metrics = _source_metrics(master_samples, sample_rate=sample_rate, fps=fps, frame_count=frame_count)
    stem_metrics = {
        stem_name: _source_metrics(
            samples.get(stem_name, np.zeros_like(master_samples)),
            sample_rate=sample_rate,
            fps=fps,
            frame_count=frame_count,
        )
        for stem_name in STEM_ORDER
    }

    frames: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        stems: dict[str, Any] = {}
        for stem_name, metrics in stem_metrics.items():
            bands = metrics["bands"][frame_index].astype(float).tolist()
            stems[stem_name] = {
                "rms": float(metrics["rms"][frame_index]),
                "onset": float(metrics["onset"][frame_index]),
                "bass": float(np.mean(bands[:4])),
                "mid": float(np.mean(bands[4:11])),
                "high": float(np.mean(bands[11:])),
                "bands": [float(value) for value in bands],
            }
        master_bands = master_metrics["bands"][frame_index].astype(float).tolist()
        frames.append(
            {
                "frame_index": frame_index,
                "time_seconds": float(frame_index / fps),
                "master": {
                    "rms": float(master_metrics["rms"][frame_index]),
                    "onset": float(master_metrics["onset"][frame_index]),
                    "bands": [float(value) for value in master_bands],
                },
                "stems": stems,
            }
        )

    return {
        "schema": "truevision_music_spectrum_metrics_v1",
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "band_count": SPECTRUM_BAND_COUNT,
        "pairing": "master_wave_vs_stem",
        "frames": frames,
    }


def _mix_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _band_color(index: int, value: float) -> tuple[int, int, int]:
    stops = [
        (130, 40, 255),   # magenta in BGR
        (255, 70, 180),   # violet-blue
        (255, 220, 40),   # cyan/gold bridge
        (65, 255, 120),   # green
        (0, 235, 255),    # yellow
        (0, 125, 255),    # orange
    ]
    position = index / max(1, SPECTRUM_BAND_COUNT - 1)
    scaled = position * (len(stops) - 1)
    left = int(math.floor(scaled))
    right = min(len(stops) - 1, left + 1)
    color = _mix_color(stops[left], stops[right], scaled - left)
    boost = 0.45 + min(1.0, value) * 0.55
    return tuple(max(0, min(255, int(channel * boost))) for channel in color)


def _draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_lower_banner(frame: np.ndarray, text: str, time_seconds: float) -> dict[str, Any]:
    height, width = frame.shape[:2]
    banner_height = max(32, int(height * 0.065))
    y0 = height - banner_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (width, height), (5, 8, 12), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0.0, frame)

    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
    travel = width + text_width + 120
    x = int(width - ((time_seconds * 92.0) % travel))
    _draw_text(frame, text, (x, y0 + int(banner_height * 0.62)), scale=0.48, color=(210, 235, 245))
    return {"text": text, "x": x, "y": y0, "height": banner_height}


def _draw_edge_frame(
    frame: np.ndarray,
    frame_state: dict[str, Any],
    *,
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    master = frame_state["master"]
    stem_mean = float(np.mean([pair["stem_value"] for pair in pairs])) if pairs else 0.0
    intensity = float(np.clip(master["rms"] * 0.55 + master["onset"] * 0.35 + stem_mean * 0.35, 0.0, 1.0))
    thickness = max(2, int(2 + intensity * 8))
    segment_width = max(8, width // SPECTRUM_BAND_COUNT)

    for index, pair in enumerate(pairs):
        x0 = index * segment_width
        x1 = width if index == SPECTRUM_BAND_COUNT - 1 else min(width, x0 + segment_width - 2)
        value = float(np.clip(max(pair["master_value"], pair["stem_value"]), 0.0, 1.0))
        color = _band_color(index, value)
        alpha_value = 0.15 + value * 0.55
        y_top = thickness + int((1.0 - value) * 9)
        y_bottom = height - thickness - int((1.0 - value) * 9)
        cv2.rectangle(overlay, (x0, y_top), (x1, y_top + thickness), color, -1)
        cv2.rectangle(overlay, (x0, y_bottom - thickness), (x1, y_bottom), color, -1)

        chiclet_h = max(8, int(16 + value * 42))
        cy = int((height * 0.18) + (index / max(1, SPECTRUM_BAND_COUNT - 1)) * height * 0.64)
        cv2.rectangle(overlay, (thickness, cy), (thickness + 8 + int(value * 16), cy + chiclet_h), color, -1)
        cv2.rectangle(overlay, (width - thickness - 8 - int(value * 16), cy), (width - thickness, cy + chiclet_h), color, -1)

        if alpha_value > 0.55:
            cv2.line(overlay, (x0, y_top + thickness), (x1, y_top + thickness + int(value * 12)), color, 1, cv2.LINE_AA)
            cv2.line(overlay, (x0, y_bottom - thickness), (x1, y_bottom - thickness - int(value * 12)), color, 1, cv2.LINE_AA)

    cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (235, 245, 255), max(1, thickness // 3))
    cv2.addWeighted(overlay, 0.18 + intensity * 0.34, frame, 0.82 - intensity * 0.34, 0.0, frame)
    return {
        "mode": "spectrum_reactive_perimeter",
        "intensity": intensity,
        "thickness": thickness,
        "source": "master_and_stem_spectrum_pairs",
    }


def _draw_spectrum_pairs(frame: np.ndarray, frame_state: dict[str, Any]) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    top = int(height * 0.12)
    bottom = int(height * 0.86)
    left = int(width * 0.06)
    right = int(width * 0.94)
    usable_width = right - left
    usable_height = bottom - top
    slot = usable_width / SPECTRUM_BAND_COUNT
    pairs: list[dict[str, Any]] = []

    for band_index in range(SPECTRUM_BAND_COUNT):
        stem_name = STEM_ORDER[band_index % len(STEM_ORDER)]
        master_value = float(np.clip(frame_state["master"]["bands"][band_index], 0.0, 1.0))
        stem_value = float(np.clip(frame_state["stems"][stem_name]["bands"][band_index], 0.0, 1.0))
        x_center = int(left + slot * band_index + slot * 0.5)
        bar_w = max(5, int(slot * 0.22))
        gap = max(2, int(slot * 0.06))
        color = _band_color(band_index, max(master_value, stem_value))
        master_color = tuple(min(255, int(channel * 1.08 + 24)) for channel in color)
        stem_color = tuple(max(20, int(channel * 0.72)) for channel in color)

        for value, x0, draw_color, label in (
            (master_value, x_center - gap - bar_w, master_color, "M"),
            (stem_value, x_center + gap, stem_color, "S"),
        ):
            bar_h = int(usable_height * (0.03 + value * 0.97))
            y0 = bottom - bar_h
            cv2.rectangle(frame, (x0, y0), (x0 + bar_w, bottom), draw_color, -1)
            cv2.rectangle(frame, (x0, y0), (x0 + bar_w, bottom), (245, 245, 245), 1)
            glow = frame.copy()
            cv2.rectangle(glow, (x0 - 2, max(top, y0 - 2)), (x0 + bar_w + 2, bottom), draw_color, 2)
            cv2.addWeighted(glow, 0.16 + value * 0.14, frame, 0.84 - value * 0.14, 0.0, frame)
            if band_index in (0, 5, 10, 15):
                _draw_text(frame, label, (x0, min(height - 48, bottom + 16)), scale=0.28, color=(180, 205, 220))

        pairs.append(
            {
                "band_index": band_index,
                "stem_name": stem_name,
                "master_value": master_value,
                "stem_value": stem_value,
            }
        )

    for grid_index in range(5):
        y = bottom - int(usable_height * grid_index / 4.0)
        cv2.line(frame, (left, y), (right, y), (28, 34, 44), 1, cv2.LINE_AA)
    _draw_text(frame, "MASTER WAV", (left, top - 20), scale=0.42, color=(225, 235, 245))
    _draw_text(frame, "STEMS", (left + 130, top - 20), scale=0.42, color=(160, 210, 230))
    return pairs


def _draw_background(frame: np.ndarray, frame_state: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    master = frame_state["master"]
    energy = float(np.clip(master["rms"] + master["onset"], 0.0, 1.0))
    for y in range(height):
        t = y / max(1, height - 1)
        base = int(8 + t * 20 + energy * 16)
        frame[y, :, :] = (base + int(t * 16), base, base // 2)

    center = (int(width * 0.5), int(height * 0.48))
    radius = int(max(width, height) * (0.18 + energy * 0.14))
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, (42, 36, 58), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.26, frame, 0.74, 0.0, frame)


def render_frame(
    frame_state: dict[str, Any],
    *,
    width: int,
    height: int,
    stem_map: dict[str, list[str]] | None = None,
    banner_text: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    stem_map = stem_map or build_stem_control_map()
    banner_text = banner_text or build_generation_banner()
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    _draw_background(frame, frame_state)
    pairs = _draw_spectrum_pairs(frame, frame_state)
    edge_frame = _draw_edge_frame(frame, frame_state, pairs=pairs)
    banner = _draw_lower_banner(frame, banner_text, frame_state["time_seconds"])

    lane_log = {
        "schema": "truevision_stem_spectrum_frame_log_v1",
        "frame_index": frame_state["frame_index"],
        "time_seconds": frame_state["time_seconds"],
        "analyzer": {
            "band_count": SPECTRUM_BAND_COUNT,
            "pairing": "master_wave_vs_stem",
            "orientation": "bottom_up",
            "palette": "gradient",
            "pairs": pairs,
        },
        "edge_frame": edge_frame,
        "stem_controls": {
            stem_name: {
                "visual_lanes": lanes,
                "rms": frame_state["stems"][stem_name]["rms"],
                "onset": frame_state["stems"][stem_name]["onset"],
            }
            for stem_name, lanes in stem_map.items()
        },
        "boundary": {
            "lyrics_used": False,
            "center_lasers_used": False,
            "edge_frame_used": True,
            "spectrum_analyzer": True,
            "external_visual_assets_used": False,
            "openai_generation_used": False,
            "generated_media_is_evidence": False,
        },
        "banner": banner,
    }
    return frame, lane_log


def _video_command(output_path: Path, *, width: int, height: int, fps: int, encoder: str) -> list[str]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
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
        encoder,
    ]
    if encoder == "libx264":
        cmd.extend(["-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"])
    else:
        cmd.extend(["-b:v", "24M"])
    cmd.append(str(output_path))
    return cmd


def _mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ],
        check=True,
    )


def render_video(
    *,
    audio_path: Path,
    stems_zip: Path,
    output_root: Path,
    run_id: str,
    duration: float,
    fps: int,
    width: int,
    height: int,
    encoder: str,
    banner_text: str | None = None,
    state_log_every: int = 30,
) -> dict[str, Any]:
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    master_samples, sample_rate = read_wav_file(audio_path, max_seconds=duration)
    target_len = int(round(duration * sample_rate))
    master_samples = _fit_length(master_samples, target_len)
    stems = load_stem_samples(
        stems_zip,
        target_rate=sample_rate,
        target_len=target_len,
        max_seconds=duration,
    )
    samples = {"Master": master_samples, **stems}
    metrics = compute_frame_metrics(samples, sample_rate=sample_rate, fps=fps, duration=duration)
    stem_map = build_stem_control_map()

    silent_video = output_dir / f"{run_id}.silent.mp4"
    final_video = output_dir / f"{run_id}.mp4"
    state_log_path = output_dir / f"{run_id}_state_lanes.jsonl"
    manifest_path = output_dir / f"{run_id}_manifest.json"
    receipt_path = output_dir / f"{run_id}_receipt.json"
    metrics_path = output_dir / f"{run_id}_spectrum_metrics.json"

    proc = subprocess.Popen(
        _video_command(silent_video, width=width, height=height, fps=fps, encoder=encoder),
        stdin=subprocess.PIPE,
    )
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin was not available")

    with state_log_path.open("w", encoding="utf-8") as state_log:
        for frame_state in metrics["frames"]:
            frame, lane_log = render_frame(
                frame_state,
                width=width,
                height=height,
                stem_map=stem_map,
                banner_text=banner_text,
            )
            proc.stdin.write(frame.tobytes())
            if int(frame_state["frame_index"]) % max(1, state_log_every) == 0:
                state_log.write(json.dumps(lane_log, sort_keys=True) + "\n")

    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg video writer failed")
    _mux_audio(silent_video, audio_path, final_video)
    silent_video.unlink(missing_ok=True)

    metrics_summary = {
        "schema": metrics["schema"],
        "frame_count": metrics["frame_count"],
        "fps": metrics["fps"],
        "duration_seconds": metrics["duration_seconds"],
        "sample_rate": metrics["sample_rate"],
        "band_count": metrics["band_count"],
        "pairing": metrics["pairing"],
        "frame_samples_logged": max(1, metrics["frame_count"] // max(1, state_log_every)),
    }
    metrics_path.write_text(json.dumps(metrics_summary, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "schema": "truevision_stem_spectrum_render_manifest_v1",
        "run_id": run_id,
        "audio_path": str(audio_path),
        "stems_zip": str(stems_zip),
        "output_video": str(final_video),
        "state_log": str(state_log_path),
        "metrics_summary": str(metrics_path),
        "width": width,
        "height": height,
        "fps": fps,
        "duration_seconds": duration,
        "stem_controls": stem_map,
        "spectrum_band_count": SPECTRUM_BAND_COUNT,
        "pairing": "master_wave_first_meter_stem_second_meter",
        "lyrics_used": False,
        "center_lasers_used": False,
        "edge_frame_used": True,
        "raw_stems_copied_to_output": False,
        "generated_media_is_evidence": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    receipt = {
        "schema": "truevision_generation_receipt_v1",
        "run_id": run_id,
        "status": "complete",
        "artifact_kind": "generated_media",
        "output_video": str(final_video),
        "manifest": str(manifest_path),
        "state_log": str(state_log_path),
        "elapsed_seconds": round(time.time() - started, 3),
        "boundaries": {
            "lyrics_used": False,
            "center_lasers_used": False,
            "edge_frame_used": True,
            "spectrum_analyzer": True,
            "openai_generation_used": False,
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "video": final_video,
        "manifest": manifest_path,
        "receipt": receipt_path,
        "state_log": state_log_path,
        "metrics": metrics_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a local stem-driven spectrum analyzer proof.")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--stems-zip", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="becoming_the_wolf_spectrum_pairs_30s_720p")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--encoder", default="libx264")
    parser.add_argument("--banner-text", default=build_generation_banner())
    parser.add_argument("--state-log-every", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = render_video(
        audio_path=args.audio,
        stems_zip=args.stems_zip,
        output_root=args.output_root,
        run_id=args.run_id,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        encoder=args.encoder,
        banner_text=args.banner_text,
        state_log_every=args.state_log_every,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
