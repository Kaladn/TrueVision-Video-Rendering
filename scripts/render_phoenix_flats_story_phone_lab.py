from __future__ import annotations

import argparse
import json
import math
import subprocess
import wave
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


FPS = 30
SR = 22050
DEFAULT_W = 1080
DEFAULT_H = 1920
STEM_NAMES = [
    "Lead Vocals",
    "Backing Vocals",
    "Drums",
    "Bass",
    "Keyboard",
    "Percussion",
    "Synth",
    "Other",
]
SCENE_NAMES = [
    "she_was_just_a_baby",
    "running_crazy",
    "empty_years",
    "pressure_made_me",
    "memory_turns_hazy",
    "the_question",
    "figure_it_out",
    "she_rescued_me",
    "same_blood",
    "together_again",
    "phoenix_rising",
    "full_circle",
]
SCENE_THRESHOLDS = [0.00, 0.07, 0.15, 0.23, 0.31, 0.40, 0.50, 0.60, 0.70, 0.80, 0.88, 0.94]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Phoenix From The Flats phone story proof from cropped storyboard state plates."
    )
    parser.add_argument("--audio", default=r"C:\Users\mydyi\Downloads\Phoenix From The Flats.wav")
    parser.add_argument("--stems", default=r"C:\Users\mydyi\Downloads\Phoenix From The Flats Stems.zip")
    parser.add_argument(
        "--storyboard-sheet",
        default=r"C:\Users\mydyi\Downloads\story images must be cropped to separate.png",
    )
    parser.add_argument("--output-root", default="outputs/phoenix_flats_story_phone")
    parser.add_argument("--run-id", default="phoenix_flats_story_phone_30s")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=DEFAULT_W)
    parser.add_argument("--height", type=int, default=DEFAULT_H)
    parser.add_argument("--crop-only", action="store_true")
    parser.add_argument("--visual-mode", choices=["story_plates", "geometry_phoenix"], default="story_plates")
    parser.add_argument("--style-phrase-mix", type=float, default=0.34)
    parser.add_argument("--camera-drift", type=float, default=0.035)
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _load_gray(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _storyboard_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    base_w, base_h = 1536.0, 1024.0
    xs = [28, 286, 535, 779, 1015, 1271]
    ys = [37, 499]
    box_w, box_h = 244, 275
    boxes: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            x0 = round(x / base_w * width)
            y0 = round(y / base_h * height)
            x1 = round((x + box_w) / base_w * width)
            y1 = round((y + box_h) / base_h * height)
            boxes.append((x0, y0, x1, y1))
    return boxes


def _phrase_box(width: int, height: int) -> tuple[int, int, int, int]:
    base_w, base_h = 1536.0, 1024.0
    x0, y0, x1, y1 = 540, 930, 1260, 985
    return (
        round(x0 / base_w * width),
        round(y0 / base_h * height),
        round(x1 / base_w * width),
        round(y1 / base_h * height),
    )


def _crop_storyboard_assets(sheet_path: Path, run_dir: Path) -> dict[str, Any]:
    sheet = Image.open(sheet_path).convert("L")
    width, height = sheet.size
    plate_dir = run_dir / "story_plates"
    plate_dir.mkdir(parents=True, exist_ok=True)
    plate_rows = []
    for index, (x0, y0, x1, y1) in enumerate(_storyboard_boxes(width, height), start=1):
        crop = sheet.crop((x0, y0, x1, y1))
        path = plate_dir / f"scene_{index:02d}_{SCENE_NAMES[index - 1]}.png"
        crop.save(path)
        plate_rows.append(
            {
                "scene_index": index,
                "scene_name": SCENE_NAMES[index - 1],
                "path": str(path),
                "source_box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "caption_policy": "captions_excluded_from_plate_crops",
            }
        )

    px0, py0, px1, py1 = _phrase_box(width, height)
    phrase = sheet.crop((px0, py0, px1, py1))
    phrase_path = plate_dir / "what_feels_right_phrase_style_strip.png"
    phrase.save(phrase_path)
    manifest = {
        "schema_version": "truevision_phoenix_flats_story_plate_manifest_v1",
        "source_sheet": str(sheet_path),
        "plate_count": len(plate_rows),
        "crop_policy": "art_plate_only_no_caption_blocks",
        "caption_policy": "captions_excluded_from_plate_crops",
        "plates": plate_rows,
        "style_sources": [
            {
                "style_id": "what_feels_right_phrase_style_strip",
                "path": str(phrase_path),
                "source_box": {"x0": px0, "y0": py0, "x1": px1, "y1": py1},
                "text": "WHAT FEELS RIGHT, IS RIGHT. HEART FOR MY DAUGHTER.",
                "usage": "handwritten_line_energy_mixed_as_texture_not_subtitle",
            }
        ],
    }
    manifest_path = run_dir / "story_plates_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def _fit_cover(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    ih, iw = gray.shape[:2]
    scale = max(width / max(iw, 1), height / max(ih, 1))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = max(0, (nw - width) // 2)
    y0 = max(0, (nh - height) // 2)
    out = resized[y0 : y0 + height, x0 : x0 + width]
    if out.shape[:2] != (height, width):
        out = cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)
    return out


def _fit_contain(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    ih, iw = gray.shape[:2]
    scale = min(width / max(iw, 1), height / max(ih, 1))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _decode_stems(stems_zip: Path, work_dir: Path, seconds: float) -> dict[str, Path]:
    extract_dir = work_dir / "stem_source"
    wav_dir = work_dir / "stem_wav"
    extract_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    decoded: dict[str, Path] = {}
    with zipfile.ZipFile(stems_zip, "r") as archive:
        for entry in archive.infolist():
            if not entry.filename.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg")):
                continue
            name = Path(entry.filename).name
            target = extract_dir / name
            target.write_bytes(archive.read(entry))
            stem_name = next((stem for stem in STEM_NAMES if stem.lower() in name.lower()), Path(name).stem)
            if stem_name not in STEM_NAMES:
                continue
            wav_path = wav_dir / f"{stem_name}.wav"
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-t",
                    f"{seconds:.3f}",
                    "-i",
                    str(target),
                    "-ac",
                    "1",
                    "-ar",
                    str(SR),
                    "-sample_fmt",
                    "s16",
                    str(wav_path),
                ]
            )
            decoded[stem_name] = wav_path
    return decoded


def _read_wav(path: Path, samples: int) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(min(samples, handle.getnframes()))
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if data.size < samples:
        data = np.pad(data, (0, samples - data.size))
    return data[:samples]


def _stem_envelopes(decoded: dict[str, Path], seconds: float) -> dict[str, np.ndarray]:
    frame_count = int(seconds * FPS)
    sample_count = int(seconds * SR)
    envelopes: dict[str, np.ndarray] = {}
    window = int(SR * 0.075)
    half = window // 2
    for stem in STEM_NAMES:
        if stem not in decoded:
            envelopes[stem] = np.zeros(frame_count, dtype=np.float32)
            continue
        audio = _read_wav(decoded[stem], sample_count)
        values = []
        for frame_index in range(frame_count):
            center = int((frame_index / FPS) * SR)
            lo = max(0, center - half)
            hi = min(audio.size, center + half)
            chunk = audio[lo:hi]
            values.append(float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0)
        arr = np.asarray(values, dtype=np.float32)
        pct = float(np.percentile(arr, 95)) if arr.size else 1.0
        if pct > 1.0e-6:
            arr = np.clip(arr / pct, 0.0, 1.35)
        arr = cv2.GaussianBlur(arr.reshape(1, -1), (1, 9), 0).reshape(-1)
        envelopes[stem] = arr.astype(np.float32)
    return envelopes


def _scene_index(progress: float) -> int:
    index = 0
    for threshold_index, threshold in enumerate(SCENE_THRESHOLDS):
        if progress >= threshold:
            index = threshold_index
    return min(index, 11)


def _plate_surface(plate: np.ndarray, phrase: np.ndarray, env: dict[str, float], progress: float, width: int, height: int, style_mix: float, t: float) -> np.ndarray:
    lead = env["Lead Vocals"]
    backing = env["Backing Vocals"]
    drums = env["Drums"]
    bass = env["Bass"]
    keyboard = env["Keyboard"]
    percussion = env["Percussion"]
    synth = env["Synth"]
    other = env["Other"]
    final_color = np.clip((progress - 0.90) / 0.10, 0.0, 1.0)

    margin = int(width * 0.055)
    top_h = int(height * 0.76)
    contained = _fit_contain(plate, width - margin * 2, top_h)
    canvas = np.zeros((height, width), dtype=np.float32)
    canvas[:top_h, margin : width - margin] = contained.astype(np.float32)

    edges = cv2.Canny(canvas.astype(np.uint8), 48, 138).astype(np.float32)
    line = canvas * (0.78 + keyboard * 0.22)
    line = np.maximum(line, edges * (0.38 + drums * 0.52 + percussion * 0.20))
    line += cv2.GaussianBlur(edges, (0, 0), 1.8 + synth * 1.2) * (0.05 + lead * 0.12)

    phrase_fit = _fit_contain(phrase, width - margin * 2, int(height * 0.11))
    phrase_plane = np.zeros_like(canvas)
    phrase_y = int(height * 0.82)
    phrase_plane[phrase_y : phrase_y + phrase_fit.shape[0], margin : width - margin] = phrase_fit.astype(np.float32)
    phrase_edges = cv2.Canny(phrase_plane.astype(np.uint8), 35, 115).astype(np.float32)
    phrase_pressure = style_mix * (0.34 + lead * 0.24 + backing * 0.18 + other * 0.18)
    line = np.maximum(line, phrase_edges * 255.0 * phrase_pressure)
    line *= 1.0 - (phrase_plane < 80).astype(np.float32) * phrase_pressure * 0.10

    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    memory_fog = cv2.GaussianBlur(line, (0, 0), 7.0 + synth * 5.0) * (0.045 + synth * 0.10)
    floor_shadow = np.clip((yy - 0.68) / 0.32, 0.0, 1.0) * (22.0 + bass * 38.0)
    image = np.clip(line + memory_fog - floor_shadow, 0, 255)

    drift = int(math.sin(t * 0.24) * width * 0.018)
    image = np.roll(image, drift, axis=1)
    if drums > 0.62:
        image = np.where(edges > 0, 255.0 - image * 0.25, image)

    rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
    color_wash = np.zeros_like(rgb)
    color_wash[..., 0] = 42.0 + synth * 26.0
    color_wash[..., 1] = 98.0 + bass * 32.0 + final_color * 80.0
    color_wash[..., 2] = 156.0 + lead * 48.0
    bottom_rise = np.clip((yy - (1.0 - final_color * 1.10)) * 3.2, 0.0, 1.0)
    rgb = rgb * (1.0 - bottom_rise[..., None] * 0.42) + color_wash * bottom_rise[..., None] * 0.42
    rgb += final_color * np.dstack((yy * 22.0, yy * 55.0, yy * 28.0))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _geometry_phoenix_surface(
    phrase: np.ndarray,
    env: dict[str, float],
    progress: float,
    width: int,
    height: int,
    style_mix: float,
    t: float,
) -> np.ndarray:
    lead = env["Lead Vocals"]
    backing = env["Backing Vocals"]
    drums = env["Drums"]
    bass = env["Bass"]
    keyboard = env["Keyboard"]
    percussion = env["Percussion"]
    synth = env["Synth"]
    other = env["Other"]
    impact = max(drums, percussion * 0.75)
    final = float(np.clip((progress - 0.82) / 0.18, 0.0, 1.0))
    healing = float(np.clip((progress - 0.70) / 0.30, 0.0, 1.0))

    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    field = np.zeros((height, width, 3), dtype=np.float32)
    field[..., 0] = 4.0 + yy * 8.0 + synth * 8.0
    field[..., 1] = 5.0 + yy * (10.0 + healing * 26.0)
    field[..., 2] = 8.0 + (1.0 - yy) * 16.0 + lead * 8.0
    center_glow = np.exp(-(((xx - 0.5) / (0.18 + synth * 0.08)) ** 2 + ((yy - 0.48) / (0.22 + bass * 0.04)) ** 2))
    field[..., 0] += center_glow * (26.0 + lead * 18.0 + final * 72.0)
    field[..., 1] += center_glow * (14.0 + backing * 18.0 + final * 112.0)
    field[..., 2] += center_glow * (8.0 + synth * 22.0 + final * 58.0)

    canvas = np.clip(field, 0, 255).astype(np.uint8)
    _draw_sound_geometry(canvas, env, progress, t)
    _draw_glyph_energy(canvas, phrase, style_mix, other, synth, t)
    _draw_city_reflection_state(canvas, env, progress, t)
    _draw_impact_rings(canvas, impact, bass, progress, t)
    if final > 0.02:
        _draw_generated_phoenix(canvas, final, env, t)
    return canvas


def _draw_sound_geometry(frame: np.ndarray, env: dict[str, float], progress: float, t: float) -> None:
    height, width = frame.shape[:2]
    lead = env["Lead Vocals"]
    backing = env["Backing Vocals"]
    bass = env["Bass"]
    keyboard = env["Keyboard"]
    synth = env["Synth"]
    other = env["Other"]
    cx = width * 0.5
    cy = height * (0.52 - 0.12 * progress)
    for lane in range(9):
        points = []
        phase = t * (0.42 + lead * 0.24 + lane * 0.018) + lane * 0.73
        span = width * (0.18 + lane * 0.035)
        amp = height * (0.012 + backing * 0.030 + synth * 0.018)
        y = cy + (lane - 4) * height * 0.038 + math.sin(phase) * height * 0.016
        for step in range(90):
            n = step / 89.0
            x = cx - span + n * span * 2.0
            wave = math.sin(n * math.tau * (1.5 + lane * 0.22) + phase) * amp
            pull = math.sin(n * math.tau + t * 0.31) * bass * height * 0.020
            points.append((int(x), int(y + wave + pull)))
        color = (
            int(70 + lead * 90 + progress * 30),
            int(96 + keyboard * 80 + progress * 60),
            int(130 + synth * 85 + backing * 30),
        )
        cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)

    for spoke in range(16):
        angle = spoke / 16.0 * math.tau + t * (0.05 + other * 0.06)
        radius = width * (0.12 + keyboard * 0.045 + progress * 0.10)
        x0 = int(cx + math.cos(angle) * radius * 0.22)
        y0 = int(cy + math.sin(angle) * radius * 0.36)
        x1 = int(cx + math.cos(angle) * radius)
        y1 = int(cy + math.sin(angle) * radius * 0.72)
        cv2.line(frame, (x0, y0), (x1, y1), (42, 76 + int(keyboard * 80), 98 + int(synth * 70)), 1, cv2.LINE_AA)


def _draw_glyph_energy(frame: np.ndarray, phrase: np.ndarray, style_mix: float, other: float, synth: float, t: float) -> None:
    height, width = frame.shape[:2]
    if style_mix <= 0.0:
        return
    strip = _fit_contain(phrase, int(width * 0.86), int(height * 0.065))
    edges = cv2.Canny(strip, 30, 120)
    energy = np.zeros((height, width), dtype=np.uint8)
    for row in range(5):
        y = int(height * (0.16 + row * 0.155 + math.sin(t * 0.19 + row) * 0.015))
        x = int(width * 0.07 + math.sin(t * 0.27 + row * 1.7) * width * 0.035)
        h, w = edges.shape[:2]
        if y + h < 0 or y >= height:
            continue
        y0 = max(0, y)
        y1 = min(height, y + h)
        x0 = max(0, x)
        x1 = min(width, x + w)
        sy0 = y0 - y
        sx0 = x0 - x
        energy[y0:y1, x0:x1] = np.maximum(energy[y0:y1, x0:x1], edges[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)])
    energy = cv2.GaussianBlur(energy, (0, 0), 0.8 + synth * 1.4)
    color = np.zeros_like(frame)
    color[..., 0] = 28
    color[..., 1] = 90 + int(other * 50)
    color[..., 2] = 110 + int(synth * 70)
    alpha = (energy.astype(np.float32) / 255.0) * (0.10 + style_mix * 0.22)
    frame[:] = np.clip(frame.astype(np.float32) * (1.0 - alpha[..., None]) + color.astype(np.float32) * alpha[..., None], 0, 255).astype(np.uint8)


def _draw_impact_rings(frame: np.ndarray, impact: float, bass: float, progress: float, t: float) -> None:
    if impact < 0.18 and progress < 0.55:
        return
    height, width = frame.shape[:2]
    cx = width // 2
    cy = int(height * (0.54 - progress * 0.12))
    pulse = max(impact, max(0.0, progress - 0.56) * 0.85)
    for ring in range(4):
        radius = int(width * (0.045 + ring * 0.045 + pulse * 0.080))
        color = (
            42 + int(pulse * 65),
            86 + int(bass * 80),
            135 + int(pulse * 95),
        )
        cv2.ellipse(frame, (cx, cy), (radius, int(radius * 0.56)), t * 8.0 + ring * 23.0, 0, 360, color, 1, cv2.LINE_AA)


def _draw_city_reflection_state(frame: np.ndarray, env: dict[str, float], progress: float, t: float) -> None:
    height, width = frame.shape[:2]
    bass = env["Bass"]
    keyboard = env["Keyboard"]
    synth = env["Synth"]
    reveal = float(np.clip((progress - 0.46) / 0.54, 0.0, 1.0))
    if reveal <= 0.01:
        return
    horizon = int(height * 0.78)
    water_top = int(height * 0.82)
    color = (
        24 + int(62 * reveal),
        70 + int(94 * reveal + keyboard * 35),
        122 + int(92 * reveal + synth * 24),
    )
    dark = (4, 7, 10)
    cv2.line(frame, (0, horizon), (width, horizon), color, 1, cv2.LINE_AA)
    for index in range(18):
        n = index / 17.0
        x = int(width * (0.06 + n * 0.88))
        bw = int(width * (0.018 + ((index * 11) % 7) * 0.002))
        bh = int(height * (0.045 + ((index * 23) % 13) * 0.006 + reveal * 0.025))
        y0 = horizon - bh
        cv2.rectangle(frame, (x - bw, y0), (x + bw, horizon), dark, -1)
        cv2.rectangle(frame, (x - bw, y0), (x + bw, horizon), color, 1)
        if reveal > 0.45:
            for win in range(3):
                wy = y0 + 6 + win * max(4, bh // 4)
                wx = x - bw + 4
                cv2.line(frame, (wx, wy), (x + bw - 4, wy), (48, 110, 160), 1, cv2.LINE_AA)
    water_h = height - water_top
    if water_h <= 0:
        return
    columns = 13
    for col in range(columns):
        n = (col - (columns - 1) / 2.0) / ((columns - 1) / 2.0)
        base_x = int(width * (0.5 + n * 0.18 * reveal))
        strength = reveal * max(0.0, 1.0 - abs(n) * 0.72)
        if strength <= 0.02:
            continue
        for row in range(0, water_h, 9):
            fade = max(0.0, 1.0 - row / max(1, water_h)) * strength
            y0 = water_top + row
            y1 = min(height - 1, y0 + int(4 + 18 * fade))
            wave = int(math.sin(row * 0.062 + col * 0.9 + t * (1.7 + bass)) * (2.0 + bass * 9.0) * fade)
            glint = int((3.0 + 16.0 * fade) * (0.65 + 0.35 * math.sin(t * 1.1 + row * 0.05 + col)))
            x = base_x + wave
            color = (12 + int(22 * fade), 82 + int(116 * fade), 168 + int(86 * fade))
            cv2.line(frame, (x, y0), (x, y1), color, 1, cv2.LINE_AA)
            if row % 27 == 0:
                cv2.line(frame, (x - glint, y0), (x + glint, y0), color, 1, cv2.LINE_AA)


def _draw_generated_phoenix(frame: np.ndarray, final: float, env: dict[str, float], t: float) -> None:
    height, width = frame.shape[:2]
    lead = env["Lead Vocals"]
    backing = env["Backing Vocals"]
    drums = env["Drums"]
    synth = env["Synth"]
    cx = width // 2
    cy = int(height * (0.46 - final * 0.10))
    scale = width * (0.28 + final * 0.38)
    alpha_color = (
        18 + int(final * 34),
        118 + int(final * 108 + backing * 30),
        200 + int(final * 55 + lead * 30),
    )
    hot_color = (
        22 + int(drums * 34),
        188 + int(final * 50),
        255,
    )
    deep_color = (8, 38 + int(final * 58), 168 + int(final * 52))
    for side in [-1, 1]:
        spine = []
        for i in range(72):
            n = i / 71.0
            sweep = n * math.pi * 0.92
            x = cx + side * scale * (0.12 + n * 0.96) * math.sin(sweep * 0.82)
            y = cy - scale * (0.44 + 0.18 * synth) * math.sin(sweep) + scale * (n - 0.48) * 0.36
            y += math.sin(t * 0.8 + n * 8.0 + side) * final * 9.0
            spine.append((int(x), int(y)))
        cv2.polylines(frame, [np.asarray(spine, dtype=np.int32)], False, hot_color, 3, cv2.LINE_AA)
        cv2.polylines(frame, [np.asarray(spine, dtype=np.int32)], False, alpha_color, 1, cv2.LINE_AA)
        for feather in range(18):
            n = feather / 17.0
            base = spine[min(len(spine) - 1, int(n * (len(spine) - 1)))]
            length = scale * (0.20 + n * 0.30)
            drop = scale * (0.05 + n * 0.22)
            curl = math.sin(t * 0.7 + feather * 0.73) * scale * 0.025
            tip = (int(base[0] + side * length + curl), int(base[1] + drop))
            cv2.line(frame, base, tip, deep_color, 3, cv2.LINE_AA)
            cv2.line(frame, base, tip, hot_color, 1 + int(final), cv2.LINE_AA)
            flame_tip = (int(tip[0] + side * scale * 0.08), int(tip[1] - scale * 0.08 * (1.0 - n)))
            cv2.line(frame, tip, flame_tip, (0, 220, 255), 1, cv2.LINE_AA)
    body_axis = (cx, cy + int(scale * 0.08))
    cv2.ellipse(frame, body_axis, (int(scale * 0.10), int(scale * 0.24)), 0, 0, 360, deep_color, 3, cv2.LINE_AA)
    cv2.ellipse(frame, body_axis, (int(scale * 0.07), int(scale * 0.20)), 0, 0, 360, hot_color, 1, cv2.LINE_AA)
    head = (cx, int(cy - scale * 0.26))
    cv2.circle(frame, head, max(3, int(scale * 0.045)), hot_color, 2, cv2.LINE_AA)
    beak = np.asarray(
        [
            (head[0] + int(scale * 0.035), head[1]),
            (head[0] + int(scale * 0.115), head[1] + int(scale * 0.020)),
            (head[0] + int(scale * 0.035), head[1] + int(scale * 0.045)),
        ],
        dtype=np.int32,
    )
    cv2.polylines(frame, [beak], True, (0, 210, 255), 1, cv2.LINE_AA)
    for tail in range(9):
        offset = (tail - 4) * scale * 0.035
        points = []
        for i in range(42):
            n = i / 41.0
            x = cx + offset * (1.0 + n * 1.4) + math.sin(t * 0.5 + n * 5.0 + tail) * final * 9.0
            y = cy + n * scale * (0.72 + final * 0.34)
            points.append((int(x), int(y)))
        cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, hot_color, 1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(frame, (0, 0), 2.2 + final * 2.0)
    cv2.addWeighted(glow, 0.12 + final * 0.16, frame, 1.0, 0, frame)


def _render(args: argparse.Namespace, run_dir: Path, crop_manifest: dict[str, Any]) -> dict[str, Any]:
    width = int(args.width)
    height = int(args.height)
    work_dir = run_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / f"{args.run_id}_silent_raw.mp4"
    silent_path = run_dir / f"{args.run_id}_silent.mp4"
    final_path = run_dir / f"{args.run_id}.mp4"
    manifest_path = run_dir / f"{args.run_id}_manifest.json"

    decoded = _decode_stems(Path(args.stems), work_dir, args.seconds)
    envelopes = _stem_envelopes(decoded, args.seconds)
    plates = [_load_gray(row["path"]) for row in crop_manifest["plates"]]
    phrase = _load_gray(crop_manifest["style_sources"][0]["path"])
    frame_count = max(1, int(args.seconds * FPS))
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not write {raw_path}")

    for frame_index in range(frame_count):
        t = frame_index / FPS
        progress = frame_index / max(frame_count - 1, 1)
        scene = _scene_index(progress)
        env = {stem: float(envelopes[stem][frame_index]) for stem in STEM_NAMES}
        if args.visual_mode == "geometry_phoenix":
            frame = _geometry_phoenix_surface(
                phrase,
                env,
                progress,
                width,
                height,
                args.style_phrase_mix,
                t,
            )
        else:
            frame = _plate_surface(
                plates[scene],
                phrase,
                env,
                progress,
                width,
                height,
                args.style_phrase_mix,
                t,
            )
        writer.write(frame)
    writer.release()

    used_encoder = "h264_qsv"
    try:
        _run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(raw_path),
                "-c:v",
                "h264_qsv",
                "-global_quality",
                "23",
                "-look_ahead",
                "0",
                "-pix_fmt",
                "nv12",
                str(silent_path),
            ]
        )
    except Exception:
        used_encoder = "libx264"
        _run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(raw_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                str(silent_path),
            ]
        )

    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(silent_path),
            "-t",
            f"{args.seconds:.3f}",
            "-i",
            str(args.audio),
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
    )
    manifest = {
        "schema_version": "truevision_phoenix_flats_story_phone_manifest_v1",
        "run_id": args.run_id,
        "output_video": str(final_path),
        "audio": str(args.audio),
        "stems_zip": str(args.stems),
        "storyboard_sheet": str(args.storyboard_sheet),
        "crop_manifest": str(run_dir / "story_plates_manifest.json"),
        "width": width,
        "height": height,
        "fps": FPS,
        "duration_seconds": args.seconds,
        "encoder": used_encoder,
        "visual_mode": args.visual_mode,
        "final_phoenix_style_contract": "fiery_wing_arc_over_city_reflection_state",
        "scene_count": 12,
        "style_sources": ["what_feels_right_phrase_style_strip"],
        "style_phrase_usage": "handwritten phrase mixed into line energy and texture, not rendered as a subtitle",
        "boundary": {
            "finalized_cleveland_tool_modified": False,
            "source_assets_used": True,
            "openai_generation_used": False,
            "captions_excluded_from_plate_crops": True,
            "generated_media_is_visualization": True,
            "story_plates_visible_in_main_render": args.visual_mode == "story_plates",
            "sound_drives_generated_geometry": args.visual_mode == "geometry_phoenix",
            "final_phoenix_generated": args.visual_mode == "geometry_phoenix",
            "only_pictorial_figure": "generated_phoenix_final_state"
            if args.visual_mode == "geometry_phoenix"
            else "storyboard_source_plates",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {"output_video": str(final_path), "manifest_json": str(manifest_path), "encoder": used_encoder}


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.output_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    crop_manifest = _crop_storyboard_assets(Path(args.storyboard_sheet), run_dir)
    if args.crop_only:
        print(json.dumps({"status": "cropped", "manifest": str(run_dir / "story_plates_manifest.json")}, indent=2))
        return 0
    print(json.dumps(_render(args, run_dir, crop_manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
