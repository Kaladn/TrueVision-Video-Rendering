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


def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value)).strip("_")
    return safe or "stem_state_nightmare"


def build_stem_control_map() -> dict[str, list[str]]:
    return {
        "Drums": ["impact_flash", "cut_shards", "glitch_gate"],
        "Bass": ["depth_grid_pressure", "occlusion_core_breath", "floor_warp"],
        "Guitar": ["laser_ribbons", "angular_state_transform", "edge_warp"],
        "Vocals": ["central_shadow_axis", "halo_pressure", "focus_pull"],
        "Synth": ["volumetric_color_field", "orbit_shells", "mirror_prism"],
        "FX": ["spark_noise", "scanline_tears", "color_inversion_hits"],
    }


def _read_wav_bytes(payload: bytes, *, duration: float | None = None) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(payload), "rb") as handle:
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        frame_count = int(handle.getnframes())
        if duration is not None:
            frame_count = min(frame_count, int(round(duration * sample_rate)))
        raw = handle.readframes(frame_count)

    if sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported wav sample width: {sample_width}")

    if channels > 1:
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), sample_rate


def load_stem_samples(stems_zip: Path, *, duration: float) -> tuple[dict[str, np.ndarray], int, list[dict[str, Any]]]:
    stem_map = build_stem_control_map()
    samples: dict[str, np.ndarray] = {}
    stem_sources: list[dict[str, Any]] = []
    sample_rate: int | None = None
    with zipfile.ZipFile(stems_zip, "r") as archive:
        names = archive.namelist()
        for stem_name in stem_map:
            candidates = [
                name for name in names
                if name.lower().endswith(".wav") and f"({stem_name.lower()})" in name.lower()
            ]
            if not candidates:
                raise FileNotFoundError(f"missing wav stem in zip: {stem_name}")
            entry_name = candidates[0]
            data = archive.read(entry_name)
            stem_samples, rate = _read_wav_bytes(data, duration=duration)
            if sample_rate is None:
                sample_rate = rate
            elif sample_rate != rate:
                raise ValueError(f"stem sample-rate mismatch: {stem_name} {rate} != {sample_rate}")
            samples[stem_name] = stem_samples
            stem_sources.append({"stem_name": stem_name, "zip_entry": entry_name, "bytes": len(data)})
    if sample_rate is None:
        raise ValueError("no stems loaded")
    return samples, sample_rate, stem_sources


def _band_energy(chunk: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    if chunk.size < 8:
        return 0.0
    window = np.hanning(chunk.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(chunk * window))
    frequencies = np.fft.rfftfreq(chunk.size, d=1.0 / sample_rate)
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.mean(spectrum[mask]))


def _normalize_series(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    scale = float(np.percentile(arr, 95))
    if scale <= 1e-8:
        return [0.0 for _ in values]
    return [float(np.clip(value / scale, 0.0, 1.0)) for value in values]


def compute_frame_metrics(samples: dict[str, np.ndarray], *, sample_rate: int, fps: int, duration: float) -> dict[str, Any]:
    frame_count = max(1, int(round(duration * fps)))
    samples_per_frame = max(1, int(round(sample_rate / fps)))
    raw: dict[str, dict[str, list[float]]] = {}
    for stem_name, stem_samples in samples.items():
        raw[stem_name] = {"rms": [], "bass": [], "mid": [], "high": []}
        for frame_index in range(frame_count):
            start = frame_index * samples_per_frame
            end = start + samples_per_frame
            chunk = stem_samples[start:end]
            if chunk.size < samples_per_frame:
                chunk = np.pad(chunk, (0, samples_per_frame - chunk.size))
            raw[stem_name]["rms"].append(float(np.sqrt(np.mean(chunk * chunk))))
            raw[stem_name]["bass"].append(_band_energy(chunk, sample_rate, 35.0, 180.0))
            raw[stem_name]["mid"].append(_band_energy(chunk, sample_rate, 180.0, 2400.0))
            raw[stem_name]["high"].append(_band_energy(chunk, sample_rate, 2400.0, 12000.0))

    normalized: dict[str, dict[str, list[float]]] = {}
    summaries: dict[str, Any] = {}
    for stem_name, lanes in raw.items():
        normalized[stem_name] = {lane: _normalize_series(values) for lane, values in lanes.items()}
        rms = normalized[stem_name]["rms"]
        onset = [0.0]
        for index in range(1, frame_count):
            onset.append(float(np.clip((rms[index] - rms[index - 1]) * 3.0, 0.0, 1.0)))
        normalized[stem_name]["onset"] = onset
        summaries[stem_name] = {
            "rms_mean": round(float(np.mean(rms)), 6),
            "rms_peak": round(float(np.max(rms)), 6),
            "onset_peak": round(float(np.max(onset)), 6),
            "bass_mean": round(float(np.mean(normalized[stem_name]["bass"])), 6),
            "mid_mean": round(float(np.mean(normalized[stem_name]["mid"])), 6),
            "high_mean": round(float(np.mean(normalized[stem_name]["high"])), 6),
        }

    frames: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        frames.append(
            {
                "frame_index": frame_index,
                "time_seconds": round(frame_index / fps, 9),
                "stems": {
                    stem_name: {
                        "rms": round(normalized[stem_name]["rms"][frame_index], 6),
                        "onset": round(normalized[stem_name]["onset"][frame_index], 6),
                        "bass": round(normalized[stem_name]["bass"][frame_index], 6),
                        "mid": round(normalized[stem_name]["mid"][frame_index], 6),
                        "high": round(normalized[stem_name]["high"][frame_index], 6),
                    }
                    for stem_name in normalized
                },
            }
        )
    return {
        "schema_version": "truevision_stem_frame_metrics_v1",
        "sample_rate": sample_rate,
        "fps": fps,
        "duration_seconds": duration,
        "frame_count": frame_count,
        "summary": summaries,
        "frames": frames,
    }


def _meter(frame_state: dict[str, Any], stem_name: str, lane: str) -> float:
    return float(((frame_state.get("stems") or {}).get(stem_name) or {}).get(lane, 0.0))


def _add_glow_line(layer: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: tuple[float, float, float], strength: float, width: int = 1) -> None:
    cv2.line(layer, start, end, tuple((np.asarray(color) * strength).tolist()), max(1, width), cv2.LINE_AA)


def _draw_depth_grid(frame: np.ndarray, bass: float, drums: float, t: float) -> None:
    height, width = frame.shape[:2]
    horizon = int(height * (0.54 + 0.035 * math.sin(t * 0.9)))
    vanishing = (width // 2, horizon)
    color = (0.06 + bass * 0.12, 0.14 + bass * 0.20, 0.18 + bass * 0.30)
    for x in np.linspace(-width * 0.25, width * 1.25, 23):
        cv2.line(frame, vanishing, (int(x), height), color, 1, cv2.LINE_AA)
    for row in range(16):
        y_norm = row / 15.0
        y = int(horizon + (y_norm ** (1.7 + bass * 0.5)) * (height - horizon))
        cv2.line(frame, (0, y), (width, y), tuple((np.asarray(color) * (0.55 + drums)).tolist()), 1, cv2.LINE_AA)


def _draw_laser_ribbons(frame: np.ndarray, guitar: float, guitar_mid: float, synth: float, t: float) -> dict[str, float]:
    height, width = frame.shape[:2]
    layer = np.zeros_like(frame)
    origins = [
        (int(width * 0.08), int(height * 0.18)),
        (int(width * 0.92), int(height * 0.18)),
        (int(width * 0.18), int(height * 0.48)),
        (int(width * 0.82), int(height * 0.48)),
        (int(width * 0.50), int(height * 0.12)),
    ]
    colors = [
        (0.15, 1.00, 0.92),
        (0.25, 0.48, 1.00),
        (0.92, 0.22, 1.00),
        (1.00, 0.22, 0.52),
        (1.00, 0.80, 0.16),
        (0.32, 1.00, 0.35),
    ]
    beam_energy = 0.0
    beam_count = 36
    for index in range(beam_count):
        origin = origins[index % len(origins)]
        angle = t * (0.6 + guitar_mid * 2.2) + index * 0.39 + math.sin(t * 1.7 + index) * 0.8
        radius = 0.42 + 0.28 * math.sin(t * 0.31 + index * 0.27)
        target = (
            int(width * (0.50 + math.cos(angle) * radius)),
            int(height * (0.52 + math.sin(angle * 0.73) * (0.30 + synth * 0.15))),
        )
        strength = 0.22 + guitar * 1.05 + 0.30 * math.sin(t * 4.0 + index)
        color = colors[index % len(colors)]
        _add_glow_line(layer, origin, target, color, strength, width=1 + int(guitar * 3))
        beam_energy += max(0.0, strength)
    bloom = cv2.GaussianBlur(layer, (0, 0), sigmaX=8 + guitar * 14, sigmaY=8 + synth * 10)
    sharp = cv2.GaussianBlur(layer, (0, 0), sigmaX=1.2, sigmaY=1.2)
    frame[:] = np.clip(frame + bloom * 0.82 + sharp * 1.35, 0.0, 1.0)
    return {"laser_beam_count": float(beam_count), "laser_energy": round(float(beam_energy / beam_count), 6)}


def _draw_center_state(frame: np.ndarray, vocals: float, bass: float, synth: float, t: float) -> None:
    height, width = frame.shape[:2]
    cx, cy = width // 2, int(height * 0.52)
    layer = np.zeros_like(frame)
    core_height = int(height * (0.19 + vocals * 0.14))
    core_width = int(width * (0.018 + bass * 0.025))
    cv2.ellipse(layer, (cx, cy), (core_width, core_height), 0, 0, 360, (0.0, 0.0, 0.0), -1, cv2.LINE_AA)
    cv2.ellipse(layer, (cx, cy - core_height // 2), (core_width * 2, core_width * 2), 0, 0, 360, (0.0, 0.0, 0.0), -1, cv2.LINE_AA)
    halo = np.zeros_like(frame)
    halo_color = (0.10 + vocals * 0.50, 0.35 + synth * 0.35, 0.50 + vocals * 0.42)
    cv2.circle(halo, (cx, cy), int(70 + 190 * vocals + 90 * bass), halo_color, -1, cv2.LINE_AA)
    halo = cv2.GaussianBlur(halo, (0, 0), sigmaX=30 + vocals * 30, sigmaY=30 + synth * 42)
    frame[:] = np.clip(frame + halo * 0.42, 0.0, 1.0)
    frame[:] = np.minimum(frame, 1.0 - layer * (0.45 + vocals * 0.25))
    for ring in range(3):
        radius = int((84 + ring * 62) * (1.0 + vocals * 0.24 + 0.08 * math.sin(t * 2.0 + ring)))
        cv2.ellipse(frame, (cx, cy), (radius, int(radius * 0.42)), t * 14 + ring * 38, 0, 360, (0.22, 0.75, 0.90), 1, cv2.LINE_AA)


def _draw_shards_and_tears(frame: np.ndarray, drums: float, fx: float, t: float, frame_index: int) -> None:
    height, width = frame.shape[:2]
    rng = np.random.default_rng(frame_index * 17 + 444)
    shard_layer = np.zeros_like(frame)
    shard_count = int(10 + drums * 42 + fx * 34)
    center = np.array([width * 0.5, height * 0.52])
    for index in range(shard_count):
        angle = (index / max(1, shard_count)) * math.tau + t * (0.5 + fx) + rng.normal(0, 0.08)
        inner = 34 + drums * 80 + rng.random() * 40
        outer = inner + 45 + rng.random() * (140 + fx * 190)
        p1 = center + np.array([math.cos(angle), math.sin(angle)]) * inner
        p2 = center + np.array([math.cos(angle + 0.018), math.sin(angle + 0.018)]) * outer
        color = (0.80 + fx * 0.20, 0.35 + drums * 0.55, 1.0)
        cv2.line(shard_layer, tuple(p1.astype(int)), tuple(p2.astype(int)), color, 1 + int(drums * 3), cv2.LINE_AA)
    shard_layer = cv2.GaussianBlur(shard_layer, (0, 0), sigmaX=1.0 + drums * 3.0)
    frame[:] = np.clip(frame + shard_layer * (0.7 + drums), 0.0, 1.0)

    tear_count = int(2 + fx * 9)
    for _ in range(tear_count):
        y = int(rng.integers(0, height))
        h = int(rng.integers(1, max(2, int(6 + fx * 14))))
        shift = int(rng.normal(0, 18 + fx * 80))
        frame[y : min(height, y + h)] = np.roll(frame[y : min(height, y + h)], shift, axis=1)


def _apply_mirror_prism(frame: np.ndarray, synth: float, fx: float, t: float) -> None:
    height, width = frame.shape[:2]
    if synth + fx < 0.08:
        return
    left = frame[:, : width // 2].copy()
    right = np.flip(left, axis=1)
    mix = 0.12 + synth * 0.22 + fx * 0.10
    frame[:, width // 2 :] = np.clip(frame[:, width // 2 :] * (1.0 - mix) + right * mix, 0.0, 1.0)
    chroma_shift = int((synth * 12 + fx * 8) * math.sin(t * 1.9))
    frame[:, :, 0] = np.roll(frame[:, :, 0], chroma_shift, axis=1)
    frame[:, :, 2] = np.roll(frame[:, :, 2], -chroma_shift, axis=1)


def render_frame(frame_state: dict[str, Any], *, width: int, height: int, stem_map: dict[str, list[str]] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    stem_map = stem_map or build_stem_control_map()
    t = float(frame_state.get("time_seconds") or 0.0)
    frame_index = int(frame_state.get("frame_index") or 0)
    drums = max(_meter(frame_state, "Drums", "rms"), _meter(frame_state, "Drums", "onset"))
    bass = max(_meter(frame_state, "Bass", "rms"), _meter(frame_state, "Bass", "bass"))
    guitar = max(_meter(frame_state, "Guitar", "rms"), _meter(frame_state, "Guitar", "mid"))
    vocals = max(_meter(frame_state, "Vocals", "rms"), _meter(frame_state, "Vocals", "mid"))
    synth = max(_meter(frame_state, "Synth", "rms"), _meter(frame_state, "Synth", "high"))
    fx = max(_meter(frame_state, "FX", "rms"), _meter(frame_state, "FX", "onset"))

    frame = np.zeros((height, width, 3), dtype=np.float32)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    frame[:, :, 0] = 0.010 + (1.0 - yy) * (0.018 + synth * 0.018)
    frame[:, :, 1] = 0.012 + (1.0 - yy) * (0.030 + guitar * 0.020)
    frame[:, :, 2] = 0.020 + (1.0 - yy) * (0.052 + fx * 0.018)

    _draw_depth_grid(frame, bass, drums, t)
    _draw_center_state(frame, vocals, bass, synth, t)
    laser_stats = _draw_laser_ribbons(frame, guitar, _meter(frame_state, "Guitar", "mid"), synth, t)
    _draw_shards_and_tears(frame, drums, fx, t, frame_index)
    _apply_mirror_prism(frame, synth, fx, t)

    if _meter(frame_state, "FX", "onset") > 0.62 or _meter(frame_state, "Drums", "onset") > 0.72:
        frame[:] = np.clip(frame + 0.12 * max(_meter(frame_state, "FX", "onset"), _meter(frame_state, "Drums", "onset")), 0.0, 1.0)

    image = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    lane_log = {
        "stem_controls": {
            stem_name: {
                "visual_lanes": lanes,
                "meters": (frame_state.get("stems") or {}).get(stem_name, {}),
            }
            for stem_name, lanes in stem_map.items()
        },
        "render_lanes": {
            "master_global_intensity": round(float(np.mean([drums, bass, guitar, vocals, synth, fx])), 6),
            **laser_stats,
        },
        "boundary": {
            "external_visual_assets_used": False,
            "openai_generation_used": False,
            "stems_drive_visual_lanes": True,
            "generated_media_is_evidence": False,
        },
    }
    return image, lane_log


def _video_command(output_path: Path, *, width: int, height: int, fps: int, encoder: str) -> list[str]:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
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
        "-an",
    ]
    if encoder == "h264_qsv":
        command += ["-c:v", "h264_qsv", "-global_quality", "16", "-look_ahead", "0"]
    else:
        command += ["-c:v", "libx264", "-preset", "medium", "-crf", "15"]
    command += ["-pix_fmt", "yuv420p", str(output_path)]
    return command


def _write_video_only(path: Path, metrics: dict[str, Any], *, width: int, height: int, encoder: str, state_path: Path, stem_map: dict[str, list[str]]) -> str:
    fps = int(metrics["fps"])
    process = subprocess.Popen(_video_command(path, width=width, height=height, fps=fps, encoder=encoder), stdin=subprocess.PIPE)
    assert process.stdin is not None
    with state_path.open("w", encoding="utf-8") as state_file:
        for frame_state in metrics["frames"]:
            frame, lane_log = render_frame(frame_state, width=width, height=height, stem_map=stem_map)
            record = dict(frame_state)
            record["schema_version"] = "truevision_stem_state_nightmare_frame_v1"
            record["lane_log"] = lane_log
            state_file.write(json.dumps(record, allow_nan=False) + "\n")
            process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg video encode failed using {encoder}")
    return encoder


def _mux_audio(video_path: Path, audio_path: Path, output_path: Path, *, duration: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-t",
        str(duration),
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
        "320k",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def render_video(
    *,
    master_audio: Path,
    stems_zip: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = "becoming_the_wolf_stem_state_nightmare_30s",
    duration: float = 30.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    encoder: str = "libx264",
    style_reference_path: Path | None = None,
) -> dict[str, Any]:
    stem_map = build_stem_control_map()
    run_id = _safe_id(run_id)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / f"{run_id}_frame_state.jsonl"
    meter_path = run_dir / f"{run_id}_stem_meter_summary.json"
    video_only_path = run_dir / f"{run_id}_silent_video.mp4"
    output_path = run_dir / f"{run_id}.mp4"
    manifest_path = run_dir / f"{run_id}_manifest.json"
    receipt_path = run_dir / f"{run_id}_receipt.json"

    start = time.perf_counter()
    stem_samples, sample_rate, stem_sources = load_stem_samples(stems_zip, duration=duration)
    metrics = compute_frame_metrics(stem_samples, sample_rate=sample_rate, fps=fps, duration=duration)
    meter_payload = {
        "schema_version": "truevision_stem_meter_summary_v1",
        "master_audio": str(master_audio),
        "stems_zip": str(stems_zip),
        "stem_sources": stem_sources,
        "stem_control_map": stem_map,
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "fps": fps,
        "frame_count": metrics["frame_count"],
        "summary": metrics["summary"],
        "boundary": {
            "stems_are_control_sources": True,
            "master_audio_global_timing_only": True,
            "raw_stems_copied_to_output": False,
        },
    }
    meter_path.write_text(json.dumps(meter_payload, indent=2, allow_nan=False), encoding="utf-8")

    used_encoder = encoder
    try:
        _write_video_only(video_only_path, metrics, width=width, height=height, encoder=used_encoder, state_path=state_path, stem_map=stem_map)
    except Exception:
        if encoder == "libx264":
            raise
        used_encoder = "libx264"
        _write_video_only(video_only_path, metrics, width=width, height=height, encoder=used_encoder, state_path=state_path, stem_map=stem_map)
    _mux_audio(video_only_path, master_audio, output_path, duration=duration)

    manifest = {
        "schema_version": "truevision_stem_state_nightmare_manifest_v1",
        "run_id": run_id,
        "created_at_unix": time.time(),
        "source": {
            "master_audio": str(master_audio),
            "stems_zip": str(stems_zip),
            "style_reference_path": str(style_reference_path) if style_reference_path else None,
            "style_reference_frames_ingested": False,
        },
        "output": {
            "mp4": str(output_path),
            "silent_video_mp4": str(video_only_path),
            "frame_state_jsonl": str(state_path),
            "stem_meter_summary_json": str(meter_path),
            "width": width,
            "height": height,
            "fps": fps,
            "duration_seconds": duration,
            "frame_count": metrics["frame_count"],
            "encoder": used_encoder,
            "wall_seconds": round(time.perf_counter() - start, 6),
        },
        "stem_control_map": stem_map,
        "boundary": {
            "external_visual_assets_used": False,
            "openai_generation_used": False,
            "art_imports_used": False,
            "stems_drive_visual_lanes": True,
            "master_audio_drives_global_timing": True,
            "generated_media_is_evidence": False,
        },
    }
    manifest["manifest_json"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")

    receipt = {
        "schema_version": "truevision_stem_state_nightmare_receipt_v1",
        "run_id": run_id,
        "output_mp4": str(output_path),
        "manifest_json": str(manifest_path),
        "frame_state_jsonl": str(state_path),
        "stem_meter_summary_json": str(meter_path),
        "stem_count": len(stem_map),
        "visual_lane_count": sum(len(lanes) for lanes in stem_map.values()),
        "accepted_boundary": manifest["boundary"],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    manifest["receipt_json"] = str(receipt_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a stem-driven abstract state nightmare proof.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--stems-zip", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="becoming_the_wolf_stem_state_nightmare_30s")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--encoder", choices=["libx264", "h264_qsv"], default="libx264")
    parser.add_argument("--style-reference-path", default="")
    args = parser.parse_args()
    manifest = render_video(
        master_audio=Path(args.audio),
        stems_zip=Path(args.stems_zip),
        output_root=Path(args.output_root),
        run_id=args.run_id,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        encoder=args.encoder,
        style_reference_path=Path(args.style_reference_path) if args.style_reference_path else None,
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
