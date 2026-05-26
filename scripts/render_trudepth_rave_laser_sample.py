from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from truevision_runtime.learning_intake.trudepth_contracts import build_trudepth_contract_bundle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "trudepth_rave_laser_show"

LASER_PALETTE_RGB = [
    (34, 255, 240),
    (50, 140, 255),
    (178, 66, 255),
    (255, 55, 206),
    (255, 92, 48),
    (255, 230, 40),
    (64, 255, 104),
]


def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_")
    return safe or "trudepth_rave_laser_show"


@lru_cache(maxsize=16)
def _noise_bytes(width: int, height: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    low = rng.random((max(4, height // 28), max(4, width // 28)), dtype=np.float32)
    noise = cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=10, sigmaY=10)
    return noise.astype(np.float32).tobytes()


def _noise(width: int, height: int, seed: int) -> np.ndarray:
    return np.frombuffer(_noise_bytes(width, height, seed), dtype=np.float32).reshape(height, width).copy()


def build_laser_show_plan() -> dict[str, Any]:
    return {
        "schema_version": "truevision_trudepth_rave_laser_show_plan_v1",
        "duration_seconds": 15,
        "effect_type": "rave_laser",
        "state_layers": [
            "black_club_depth",
            "volumetric_haze_medium",
            "collimated_laser_beams",
            "beam_bloom_scatter",
            "wet_floor_reflections",
            "center_warp_pulse",
            "crowd_silhouette_low_edge",
        ],
        "boundary": {
            "source_video_frames_used": False,
            "external_visual_assets_used": False,
            "synthetic_state_media": True,
            "pixels_are_final_output": True,
        },
    }


def _add_weighted_float(base: np.ndarray, layer: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(base + layer * alpha, 0.0, 1.0)


def _draw_floor(frame: np.ndarray, t: float, pulse: float) -> None:
    height, width = frame.shape[:2]
    horizon = int(height * 0.56)
    floor_color = np.array([0.020, 0.018, 0.025], dtype=np.float32)
    frame[horizon:, :, :] = frame[horizon:, :, :] * 0.52 + floor_color * 0.48
    vanishing = (width // 2, horizon)
    for index, x in enumerate(np.linspace(-width * 0.20, width * 1.20, 15)):
        color = (0.045, 0.030 + 0.018 * pulse, 0.070 + 0.035 * pulse)
        cv2.line(
            frame,
            vanishing,
            (int(x), height),
            color,
            1,
            cv2.LINE_AA,
        )
    for row in range(9):
        y_norm = row / 8.0
        y = int(horizon + (y_norm**1.85) * (height - horizon))
        strength = 0.12 * (1.0 - y_norm) + 0.035
        cv2.line(frame, (0, y), (width, y), (strength * 0.50, strength * 0.34, strength), 1, cv2.LINE_AA)


def _draw_crowd(frame: np.ndarray, t: float, pulse: float) -> None:
    height, width = frame.shape[:2]
    base_y = int(height * 0.82)
    layer = np.zeros_like(frame)
    for index in range(38):
        x = int((index + 0.5) / 38.0 * width)
        bob = int(math.sin(t * 7.0 + index * 0.91) * 5.0 * (0.4 + pulse))
        h = int(height * (0.035 + 0.025 * ((index * 17) % 7) / 7.0))
        cv2.ellipse(layer, (x, base_y + bob), (max(3, width // 180), h), 0, 0, 360, (0.008, 0.007, 0.011), -1, cv2.LINE_AA)
    frame[:] = np.maximum(frame * 0.98, layer)


def _draw_lasers(frame: np.ndarray, t: float, pulse: float, intensity: float) -> dict[str, float]:
    height, width = frame.shape[:2]
    beam_layer = np.zeros_like(frame)
    cone_layer = np.zeros_like(frame)
    reflection_layer = np.zeros_like(frame)
    beam_count = 26
    active_energy = 0.0
    horizon_y = int(height * 0.55)
    origins = [
        (int(width * 0.50), int(height * 0.30)),
        (int(width * 0.20), int(height * 0.35)),
        (int(width * 0.80), int(height * 0.35)),
        (int(width * 0.08), int(height * 0.18)),
        (int(width * 0.92), int(height * 0.18)),
    ]

    for index in range(beam_count):
        origin = origins[index % len(origins)]
        sweep = math.sin(t * (0.92 + 0.043 * index) + index * 0.73)
        wobble = math.sin(t * (2.1 + index * 0.021) + index * 1.9)
        target_x = int(width * (0.50 + 0.52 * sweep))
        target_y = int(height * (0.16 + 0.66 * ((wobble + 1.0) * 0.5) ** 1.12))
        color = np.array(LASER_PALETTE_RGB[index % len(LASER_PALETTE_RGB)], dtype=np.float32) / 255.0
        beam_gate = 0.45 + 0.55 * max(0.0, math.sin(t * 3.3 + index * 0.54))
        beam_strength = intensity * beam_gate * (0.58 + 0.62 * pulse)
        width_px = max(1, int(1 + 2 * pulse + (index % 3 == 0)))
        active_energy += beam_strength

        cone_width = int(width * (0.020 + 0.018 * pulse))
        polygon = np.array(
            [
                origin,
                (target_x - cone_width, target_y),
                (target_x + cone_width, target_y),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(cone_layer, polygon, tuple((color * beam_strength * 0.08).tolist()), cv2.LINE_AA)
        cv2.line(beam_layer, origin, (target_x, target_y), tuple((color * beam_strength).tolist()), width_px, cv2.LINE_AA)

        if target_y > horizon_y:
            reflected_y = int(horizon_y + (target_y - horizon_y) * 0.55)
            cv2.line(
                reflection_layer,
                (origin[0], horizon_y + int((origin[1] - horizon_y) * -0.24)),
                (target_x, reflected_y),
                tuple((color * beam_strength * 0.34).tolist()),
                max(1, width_px),
                cv2.LINE_AA,
            )

    beam_bloom = cv2.GaussianBlur(beam_layer, (0, 0), sigmaX=4 + 4 * pulse, sigmaY=4 + 4 * pulse)
    wide_bloom = cv2.GaussianBlur(beam_layer, (0, 0), sigmaX=16 + 8 * pulse, sigmaY=16 + 8 * pulse)
    cone_bloom = cv2.GaussianBlur(cone_layer, (0, 0), sigmaX=18, sigmaY=8)
    reflection_bloom = cv2.GaussianBlur(reflection_layer, (0, 0), sigmaX=9, sigmaY=5)

    frame[:] = _add_weighted_float(frame, cone_bloom, 1.0)
    frame[:] = _add_weighted_float(frame, wide_bloom, 0.60)
    frame[:] = _add_weighted_float(frame, beam_bloom, 1.10)
    frame[:] = _add_weighted_float(frame, beam_layer, 1.45)
    frame[:] = _add_weighted_float(frame, reflection_bloom, 0.55)

    return {
        "beam_count": float(beam_count),
        "beam_energy": float(active_energy / beam_count),
        "bloom_pressure": float(np.mean(wide_bloom)),
    }


def _apply_haze(frame: np.ndarray, t: float, pulse: float) -> None:
    height, width = frame.shape[:2]
    haze_a = _noise(width, height, 37)
    haze_b = _noise(width, height, 91)
    shift_x = int(math.sin(t * 0.42) * width * 0.055)
    shift_y = int(math.cos(t * 0.33) * height * 0.024)
    haze = np.clip(np.roll(haze_a, (shift_y, shift_x), axis=(0, 1)) * 0.58 + np.roll(haze_b, (-shift_y, -shift_x), axis=(0, 1)) * 0.42, 0.0, 1.0)
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    density = haze * (0.28 + 0.30 * (1.0 - yy) + 0.18 * pulse)
    haze_color = np.dstack([density * 0.18, density * 0.22, density * 0.30])
    frame[:] = np.clip(frame * (1.0 - density[:, :, None] * 0.22) + haze_color, 0.0, 1.0)


def render_frame(*, frame_index: int, total_frames: int, width: int, height: int, label: bool = True) -> tuple[np.ndarray, dict[str, float]]:
    t_norm = frame_index / max(1, total_frames - 1)
    seconds = t_norm * 15.0
    pulse = 0.5 + 0.5 * math.sin(seconds * math.tau * 1.65)
    pulse = pulse**3
    sweep_intensity = 0.70 + 0.25 * math.sin(seconds * math.tau * 0.23) + 0.20 * math.sin(seconds * math.tau * 0.71)
    frame = np.zeros((height, width, 3), dtype=np.float32)
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    frame[:, :, 2] = 0.025 + (1.0 - yy) * 0.035
    frame[:, :, 1] = 0.010 + (1.0 - yy) * 0.012
    frame[:, :, 0] = 0.012

    _draw_floor(frame, seconds, pulse)
    _apply_haze(frame, seconds, pulse)
    stats = _draw_lasers(frame, seconds, pulse, max(0.28, sweep_intensity))
    _draw_crowd(frame, seconds, pulse)
    _apply_haze(frame, seconds + 3.0, pulse * 0.4)

    core = np.zeros_like(frame)
    center = (width // 2, int(height * 0.34))
    cv2.circle(core, center, int(10 + 30 * pulse), (0.9, 0.42 + 0.35 * pulse, 1.0), -1, cv2.LINE_AA)
    core = cv2.GaussianBlur(core, (0, 0), sigmaX=16 + 10 * pulse)
    frame = _add_weighted_float(frame, core, 0.72)

    image = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    if label:
        cv2.putText(image, "TruDepth / rave_laser_show", (28, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (210, 225, 232), 1, cv2.LINE_AA)
    stats.update({"pulse": float(pulse), "haze_mean": float(np.mean(frame))})
    return image, stats


def _ffmpeg_command(path: Path, *, width: int, height: int, fps: int, encoder: str) -> list[str]:
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
        command += ["-c:v", "h264_qsv", "-global_quality", "18", "-look_ahead", "0"]
    else:
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16"]
    command += ["-pix_fmt", "yuv420p", str(path)]
    return command


def render_video(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = "rave_laser_show_15s_60fps",
    duration: float = 15.0,
    fps: int = 60,
    width: int = 960,
    height: int = 540,
    encoder: str = "h264_qsv",
    label: bool = True,
) -> dict[str, Any]:
    run_id = _safe_id(run_id)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{run_id}.mp4"
    state_path = run_dir / f"{run_id}_frame_state.jsonl"
    frame_count = int(round(duration * fps))
    command = _ffmpeg_command(output_path, width=width, height=height, fps=fps, encoder=encoder)
    frame_stats: list[dict[str, float]] = []
    start = time.perf_counter()
    used_encoder = encoder
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert process.stdin is not None
        with state_path.open("w", encoding="utf-8") as state_file:
            for frame_index in range(frame_count):
                frame, stats = render_frame(frame_index=frame_index, total_frames=frame_count, width=width, height=height, label=label)
                state_record = {
                    "schema_version": "truevision_trudepth_rave_laser_frame_state_v1",
                    "frame_index": frame_index,
                    "time_seconds": round(frame_index / max(fps, 1), 9),
                    "fps": fps,
                    **stats,
                }
                state_file.write(json.dumps(state_record, allow_nan=False) + "\n")
                if frame_index % max(1, fps // 2) == 0:
                    frame_stats.append({"frame_index": float(frame_index), "time_seconds": state_record["time_seconds"], **stats})
                process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg render failed")
    except Exception:
        if encoder == "libx264":
            raise
        used_encoder = "libx264"
        process = subprocess.Popen(_ffmpeg_command(output_path, width=width, height=height, fps=fps, encoder=used_encoder), stdin=subprocess.PIPE)
        assert process.stdin is not None
        frame_stats = []
        with state_path.open("w", encoding="utf-8") as state_file:
            for frame_index in range(frame_count):
                frame, stats = render_frame(frame_index=frame_index, total_frames=frame_count, width=width, height=height, label=label)
                state_record = {
                    "schema_version": "truevision_trudepth_rave_laser_frame_state_v1",
                    "frame_index": frame_index,
                    "time_seconds": round(frame_index / max(fps, 1), 9),
                    "fps": fps,
                    **stats,
                }
                state_file.write(json.dumps(state_record, allow_nan=False) + "\n")
                if frame_index % max(1, fps // 2) == 0:
                    frame_stats.append({"frame_index": float(frame_index), "time_seconds": state_record["time_seconds"], **stats})
                process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg fallback render failed")
    wall = time.perf_counter() - start
    manifest = {
        "schema_version": "truevision_trudepth_rave_laser_show_manifest_v1",
        "run_id": run_id,
        "plan": build_laser_show_plan(),
        "trudepth_contract_bundle": build_trudepth_contract_bundle("rave_laser"),
        "output": {
            "path": str(output_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frame_count,
            "duration_seconds": duration,
            "encoder": used_encoder,
            "frame_state_jsonl": str(state_path),
            "state_log_every": 1,
            "wall_seconds": round(wall, 6),
        },
        "state_samples": frame_stats,
        "boundary": {
            "source_video_frames_used": False,
            "teacher_video_used": False,
            "internal_pulse_driver": True,
            "synthetic_state_media": True,
        },
    }
    manifest_path = run_dir / f"{run_id}_manifest.json"
    manifest["manifest_json"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a 15-second TruDepth rave laser show proof.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="rave_laser_show_15s_60fps")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--encoder", choices=["h264_qsv", "libx264"], default="h264_qsv")
    parser.add_argument("--no-label", action="store_true")
    args = parser.parse_args()
    manifest = render_video(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        encoder=args.encoder,
        label=not args.no_label,
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
