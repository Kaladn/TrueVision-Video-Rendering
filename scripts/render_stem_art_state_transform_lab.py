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


W, H = 1280, 720
FPS = 30
SR = 22050
STEM_NAMES = [
    "Lead Vocals",
    "Backing Vocals",
    "Drums",
    "Bass",
    "Guitar",
    "Keyboard",
    "Synth",
    "Other",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a stem-driven art state transform lab proof.")
    parser.add_argument("--audio", default=r"C:\Users\mydyi\Downloads\Lower the Room x Mind Scrape (Mashup).wav")
    parser.add_argument("--stems", default=r"C:\Users\mydyi\Downloads\Lower the Room x Mind Scrape (Mashup) Stems.zip")
    parser.add_argument("--skyline", default=r"C:\Users\mydyi\Downloads\Clea skyline lineart.png")
    parser.add_argument("--proof-a", default=r"C:\Users\mydyi\Downloads\B&W proofs2 no kings.png")
    parser.add_argument("--proof-b", default=r"C:\Users\mydyi\Downloads\B7W graffiti proofs 1.png")
    parser.add_argument("--output-root", default="outputs/cleveland_graffiti_state/lower_room_mind_scrape_bw_graffiti_30s")
    parser.add_argument("--run-id", default="lower_room_mind_scrape_bw_graffiti_30s")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--layout", choices=["landscape_graffiti", "phone_water_reflection"], default="landscape_graffiti")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--waterline", type=float, default=0.68)
    parser.add_argument("--camera-drift", type=float, default=0.18, help="Slow non-music side drift cycles per second.")
    parser.add_argument("--run-instruction", default="", help="Run-only visual instruction; not a preset promotion.")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _load_gray(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8)


def _fit_cover(gray: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = gray.shape[:2]
    scale = max(w / max(iw, 1), h / max(ih, 1))
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = max(0, (nw - w) // 2)
    y0 = max(0, (nh - h) // 2)
    out = resized[y0 : y0 + h, x0 : x0 + w]
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    return out


def _fit_contain(gray: np.ndarray, w: int, h: int) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    ih, iw = gray.shape[:2]
    scale = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _fit_contain_with_margin(gray: np.ndarray, w: int, h: int, margin_x: int) -> np.ndarray:
    usable_w = max(1, w - margin_x * 2)
    contained = _fit_contain(gray, usable_w, h)
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[:, margin_x : margin_x + usable_w] = contained
    return canvas


def _fit_contain_bottom_with_margin(gray: np.ndarray, w: int, h: int, margin_x: int) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    usable_w = max(1, w - margin_x * 2)
    ih, iw = gray.shape[:2]
    scale = min(usable_w / max(iw, 1), h / max(ih, 1))
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = margin_x + (usable_w - nw) // 2
    y0 = h - nh
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _fit_contain_bottom_info(gray: np.ndarray, w: int, h: int, margin_x: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.zeros((h, w), dtype=np.uint8)
    usable_w = max(1, w - margin_x * 2)
    ih, iw = gray.shape[:2]
    scale = min(usable_w / max(iw, 1), h / max(ih, 1))
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = margin_x + (usable_w - nw) // 2
    y0 = h - nh
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, (x0, y0, nw, nh)


def _decode_stems(stems_zip: Path, work_dir: Path, seconds: float) -> dict[str, Path]:
    extract_dir = work_dir / "stem_mp3"
    wav_dir = work_dir / "stem_wav"
    extract_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    decoded: dict[str, Path] = {}
    with zipfile.ZipFile(stems_zip, "r") as archive:
        for entry in archive.infolist():
            if not entry.filename.lower().endswith((".mp3", ".wav", ".flac")):
                continue
            name = Path(entry.filename).name
            target = extract_dir / name
            target.write_bytes(archive.read(entry))
            stem_name = next((stem for stem in STEM_NAMES if stem.lower() in name.lower()), Path(name).stem)
            wav_path = wav_dir / f"{stem_name}.wav"
            _run(
                [
                    "ffmpeg",
                    "-y",
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
        for frame_idx in range(frame_count):
            center = int((frame_idx / FPS) * SR)
            lo = max(0, center - half)
            hi = min(audio.size, center + half)
            chunk = audio[lo:hi]
            rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
            values.append(rms)
        arr = np.asarray(values, dtype=np.float32)
        pct = float(np.percentile(arr, 95)) if arr.size else 1.0
        if pct > 1.0e-6:
            arr = np.clip(arr / pct, 0.0, 1.35)
        arr = cv2.GaussianBlur(arr.reshape(1, -1), (1, 9), 0).reshape(-1)
        envelopes[stem] = arr.astype(np.float32)
    return envelopes


def _crop_plate(gray: np.ndarray, t: float, seed: int, w: int, h: int) -> np.ndarray:
    ih, iw = gray.shape
    cw = min(iw, int(iw * (0.42 + 0.08 * math.sin(t * 0.9 + seed))))
    ch = min(ih, int(ih * (0.36 + 0.05 * math.cos(t * 0.7 + seed))))
    x = int((iw - cw) * (0.5 + 0.46 * math.sin(t * 0.23 + seed * 1.7)))
    y = int((ih - ch) * (0.5 + 0.45 * math.cos(t * 0.19 + seed * 2.1)))
    crop = gray[max(0, y) : max(0, y) + ch, max(0, x) : max(0, x) + cw]
    return _fit_cover(crop, w, h)


def _apply_camera(gray: np.ndarray, zoom: float, pan_x: float, pan_y: float, angle: float) -> np.ndarray:
    center = (W / 2 + pan_x, H / 2 + pan_y)
    matrix = cv2.getRotationMatrix2D(center, angle, zoom)
    return cv2.warpAffine(gray, matrix, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _ink_mix(base: np.ndarray, proof_a: np.ndarray, proof_b: np.ndarray, env: dict[str, float], t: float) -> np.ndarray:
    lead = env["Lead Vocals"]
    backing = env["Backing Vocals"]
    drums = env["Drums"]
    bass = env["Bass"]
    guitar = env["Guitar"]
    keyboard = env["Keyboard"]
    synth = env["Synth"]
    other = env["Other"]

    a = _crop_plate(proof_a, t, 3, W, H)
    b = _crop_plate(proof_b, t, 9, W, H)
    a_edges = cv2.Canny(a, 60, 150)
    b_edges = cv2.Canny(b, 50, 145)
    skyline_edges = cv2.Canny(base, 45, 130)

    image = base.astype(np.float32)
    image = image * (0.82 + keyboard * 0.16)
    image = np.maximum(image, skyline_edges.astype(np.float32) * (0.85 + drums * 0.55))
    image = np.maximum(image, a_edges.astype(np.float32) * (0.25 + guitar * 0.75))
    image = np.maximum(image, b_edges.astype(np.float32) * (0.18 + backing * 0.65))

    # Proof-sheet ink fields, masked so source art drives the texture instead of invented overlays.
    thresh_a = (a < 120).astype(np.float32)
    thresh_b = (b < 125).astype(np.float32)
    left_mask = np.zeros((H, W), dtype=np.float32)
    right_mask = np.zeros((H, W), dtype=np.float32)
    left_mask[:, : W // 2] = 1.0
    right_mask[:, W // 2 :] = 1.0
    image *= 1.0 - thresh_a * left_mask * (0.12 + other * 0.16)
    image *= 1.0 - thresh_b * right_mask * (0.10 + synth * 0.14)

    # Bass pulls the skyline floor into heavier black; vocals let the center breathe.
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    floor_pressure = np.clip((yy - 0.62) / 0.38, 0, 1)
    image *= 1.0 - floor_pressure * (0.14 + bass * 0.22)
    center = np.exp(-(((np.arange(W)[None, :] - W / 2) / (W * 0.24)) ** 2 + ((np.arange(H)[:, None] - H * 0.46) / (H * 0.30)) ** 2))
    image += center * (lead * 34.0 + backing * 18.0)

    # Drum shutters and guitar slashes are derived from proof edges, not newly drawn graffiti.
    if drums > 0.58:
        image = np.where(skyline_edges > 0, 255 - image * 0.28, image)
    slash_shift = int((guitar - 0.5) * 20)
    shifted = np.roll(a_edges.astype(np.float32), slash_shift, axis=1)
    image = np.maximum(image, shifted * (0.10 + guitar * 0.35))

    grain = ((np.sin(np.arange(W)[None, :] * 0.071 + t * 6.2) + np.cos(np.arange(H)[:, None] * 0.047 + t * 4.1)) * 8.0)
    image += grain * (0.25 + other * 0.65)
    image = cv2.GaussianBlur(image, (0, 0), 0.35 + synth * 0.55)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _shift_horizontal(gray: np.ndarray, dx: int) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _ink_mix_phone_water(
    skyline: np.ndarray,
    proof_a: np.ndarray,
    proof_b: np.ndarray,
    env: dict[str, float],
    t: float,
    waterline: float,
    camera_drift: float,
) -> np.ndarray:
    water_y = int(np.clip(waterline, 0.52, 0.82) * H)
    water_h = max(1, H - water_y)
    max_drift = max(8, int(W * 0.035))
    drift = int(math.sin(t * math.tau * max(camera_drift, 0.01)) * max_drift)
    lead = env["Lead Vocals"]
    backing = env["Backing Vocals"]
    drums = env["Drums"]
    bass = env["Bass"]
    guitar = env["Guitar"]
    keyboard = env["Keyboard"]
    synth = env["Synth"]
    other = env["Other"]

    top_base, city_rect = _fit_contain_bottom_info(skyline, W, water_y, max_drift + 6)
    proof_a_top = _fit_contain_bottom_with_margin(proof_a, W, water_y, max_drift + 6)
    proof_b_top = _fit_contain_bottom_with_margin(proof_b, W, water_y, max_drift + 6)
    city_mask = (top_base > 10).astype(np.float32)
    top = top_base.astype(np.float32)
    top = _shift_horizontal(top.astype(np.uint8), drift).astype(np.float32)
    proof_a_top = _shift_horizontal(proof_a_top, drift)
    proof_b_top = _shift_horizontal(proof_b_top, drift)

    a_edges = cv2.Canny(proof_a_top, 58, 148).astype(np.float32)
    b_edges = cv2.Canny(proof_b_top, 50, 138).astype(np.float32)
    sky_edges = cv2.Canny(top.astype(np.uint8), 44, 125).astype(np.float32)
    object_plane = np.maximum(a_edges * (0.18 + guitar * 0.50), b_edges * (0.13 + backing * 0.42 + synth * 0.18))
    object_plane *= 0.18 + 0.82 * np.clip(city_mask + (top > 80).astype(np.float32) * 0.25, 0.0, 1.0)
    graffiti_fade = 0.16 + guitar * 0.28 + backing * 0.20
    top = np.maximum(top * (0.80 + keyboard * 0.17), sky_edges * (0.70 + drums * 0.35))
    top = np.maximum(top, object_plane * (0.85 + graffiti_fade))
    proof_dark = ((proof_a_top < 120).astype(np.float32) * (0.06 + other * 0.11)) + ((proof_b_top < 125).astype(np.float32) * (0.05 + synth * 0.08))
    top *= 1.0 - np.clip(proof_dark, 0.0, 0.26)

    center_x = np.arange(W, dtype=np.float32)[None, :]
    center_y = np.arange(water_y, dtype=np.float32)[:, None]
    breath = np.exp(-(((center_x - W / 2) / (W * 0.28)) ** 2 + ((center_y - water_y * 0.45) / (water_y * 0.32)) ** 2))
    top += breath * (lead * 24.0 + backing * 12.0)
    top += ((np.sin(center_x * 0.035 + t * 3.0) + np.cos(center_y * 0.029 + t * 2.2)) * (1.5 + other * 5.0))
    top = np.clip(top, 0, 255).astype(np.uint8)

    reflection_source = np.maximum(top * city_mask, object_plane * (0.90 + lead * 0.22))
    x0, y0, cw, ch = city_rect
    reflected_slice = cv2.flip(reflection_source[y0 : y0 + ch, :], 0) if ch > 0 else cv2.flip(reflection_source, 0)
    reflected = np.zeros((water_h, W), dtype=np.uint8)
    landing_h = min(water_h, max(1, int(ch * 0.55)))
    reflected_near = cv2.resize(reflected_slice.astype(np.uint8), (W, landing_h), interpolation=cv2.INTER_AREA)
    reflected[:landing_h, :] = reflected_near
    rows = np.arange(water_h, dtype=np.float32)
    ripple_strength = 2.0 + bass * 10.0 + synth * 6.0
    perspective = np.linspace(1.0, 0.12, water_h, dtype=np.float32)
    rippled = np.zeros_like(reflected)
    for y, row in enumerate(rows):
        shift = int((math.sin(row * 0.055 + t * 2.4) * ripple_strength + math.sin(row * 0.135 + t * 4.1) * (1.5 + drums * 3.0)) * perspective[y])
        rippled[y] = np.roll(reflected[y], shift)
    fade = np.linspace(0.50 + lead * 0.10, 0.045, water_h, dtype=np.float32)[:, None]
    water = (rippled.astype(np.float32) * fade).astype(np.uint8)
    water = cv2.GaussianBlur(water, (0, 0), 0.75 + synth * 1.0)
    water_noise = ((np.sin(np.arange(W)[None, :] * 0.043 + t * 4.0) + np.cos(rows[:, None] * 0.081 + t * 1.9)) * (3.0 + bass * 8.0)).astype(np.float32)
    flat_plane = np.clip((1.0 - rows[:, None] / max(water_h, 1)) * 9.0 + water_noise, 0, 28)
    water = np.clip(water.astype(np.float32) + flat_plane, 0, 255).astype(np.uint8)

    full = np.zeros((H, W), dtype=np.uint8)
    full[:water_y, :] = top
    full[water_y:, :] = water
    cv2.line(full, (0, water_y), (W, water_y), int(80 + 70 * (0.3 + bass * 0.7)), 2)
    return full


def _put(frame: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.5, color=(235, 235, 235), thickness: int = 1) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _render(args: argparse.Namespace) -> dict[str, Any]:
    global W, H
    W = int(args.width)
    H = int(args.height)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_path = out_root / f"{args.run_id}_silent_raw.mp4"
    silent_path = out_root / f"{args.run_id}_silent.mp4"
    final_path = out_root / f"{args.run_id}.mp4"
    manifest_path = out_root / f"{args.run_id}_manifest.json"
    work_dir = out_root / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    decoded = _decode_stems(Path(args.stems), work_dir, args.seconds)
    envelopes = _stem_envelopes(decoded, args.seconds)
    skyline_source = _load_gray(args.skyline)
    skyline = _fit_cover(skyline_source, W, H)
    proof_a = _load_gray(args.proof_a)
    proof_b = _load_gray(args.proof_b)
    frame_count = int(args.seconds * FPS)
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"could not write {raw_path}")

    for frame_idx in range(frame_count):
        t = frame_idx / FPS
        env = {stem: float(envelopes[stem][frame_idx]) for stem in STEM_NAMES}
        if args.layout == "phone_water_reflection":
            image = _ink_mix_phone_water(skyline_source, proof_a, proof_b, env, t, args.waterline, args.camera_drift)
        else:
            zoom = 1.035 + env["Bass"] * 0.035 + env["Drums"] * 0.018
            pan_x = math.sin(t * 0.34) * (16 + env["Guitar"] * 22)
            pan_y = math.cos(t * 0.27) * (8 + env["Synth"] * 14)
            angle = math.sin(t * 0.42) * (0.45 + env["Drums"] * 0.85)
            plate = _apply_camera(skyline, zoom, pan_x, pan_y, angle)
            image = _ink_mix(plate, proof_a, proof_b, env, t)
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Minimal source/tech banner. It labels the proof without becoming the image.
        overlay = rgb.copy()
        cv2.rectangle(overlay, (0, H - 46), (W, H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.48, rgb, 0.52, 0, rgb)
        banner = "TRUEVISION STATE LAB | PHONE WATER REFLECTION | FULL ART FIT + SLOW CAMERA DRIFT" if args.layout == "phone_water_reflection" else "TRUEVISION STATE PROOF | CLEVELAND LINEART + B/W GRAFFITI PROOFS | STEM-DRIVEN INK / CONTRAST / CAMERA PRESSURE"
        _put(rgb, banner, (24, H - 18), 0.43)
        writer.write(rgb)
    writer.release()

    used_encoder = "h264_qsv"
    try:
        _run(
            [
                "ffmpeg",
                "-y",
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
        _run(["ffmpeg", "-y", "-i", str(raw_path), "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", str(silent_path)])

    _run(
        [
            "ffmpeg",
            "-y",
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
        "schema_version": "truevision_cleveland_bw_graffiti_state_proof_manifest_v1",
        "run_id": args.run_id,
        "output_video": str(final_path),
        "silent_video": str(silent_path),
        "audio": str(args.audio),
        "stems_zip": str(args.stems),
        "assets": {
            "cleveland_skyline_lineart": str(args.skyline),
            "bw_proof_set_a": str(args.proof_a),
            "bw_proof_set_b": str(args.proof_b),
        },
        "stem_controls": {
            "Lead Vocals": "center ink breath and clarity pressure",
            "Backing Vocals": "echo linework from proof sheets",
            "Drums": "hard contrast shutters and micro-impact zoom",
            "Bass": "skyline floor weight and forward pressure",
            "Guitar": "proof-edge slash movement",
            "Keyboard": "plate luminance and line clarity",
            "Synth": "blur/fog pressure and atmospheric smear",
            "Other": "grain, wall grit, proof texture darkening",
        },
        "duration_seconds": args.seconds,
        "fps": FPS,
        "width": W,
        "height": H,
        "layout": args.layout,
        "run_instruction": args.run_instruction or ("city_vertical_water_plane_reflection_run_only" if args.layout == "phone_water_reflection" else "default_lab_run"),
        "promotion_status": "run_only_not_preset" if args.layout == "phone_water_reflection" else "lab_output_not_preset",
        "preset_promoted": False,
        "water_plane_contract": {
            "city_plane": "vertical",
            "water_plane": "horizontal",
            "foreground_light_objects_reflect": True,
            "reflection_lands_on_water_plane": True,
        }
        if args.layout == "phone_water_reflection"
        else None,
        "waterline": args.waterline if args.layout == "phone_water_reflection" else None,
        "camera_drift": args.camera_drift,
        "encoder": used_encoder,
        "boundary": {
            "source_assets_used": True,
            "openai_generation_used": False,
            "new_graffiti_language_added": False,
            "lyrics_rendered": False,
            "generated_media_is_visualization": True,
            "run_specific_instruction": args.layout == "phone_water_reflection",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {"output_video": str(final_path), "manifest_json": str(manifest_path), "encoder": used_encoder}


def main() -> int:
    args = _parse_args()
    print(json.dumps(_render(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
