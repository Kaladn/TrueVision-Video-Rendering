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
    parser = argparse.ArgumentParser(description="Render a Cleveland B/W graffiti proof driven by stems.")
    parser.add_argument("--audio", default=r"C:\Users\mydyi\Downloads\Lower the Room x Mind Scrape (Mashup).wav")
    parser.add_argument("--stems", default=r"C:\Users\mydyi\Downloads\Lower the Room x Mind Scrape (Mashup) Stems.zip")
    parser.add_argument("--skyline", default=r"C:\Users\mydyi\Downloads\Clea skyline lineart.png")
    parser.add_argument("--proof-a", default=r"C:\Users\mydyi\Downloads\B&W proofs2 no kings.png")
    parser.add_argument("--proof-b", default=r"C:\Users\mydyi\Downloads\B7W graffiti proofs 1.png")
    parser.add_argument("--output-root", default="outputs/cleveland_graffiti_state/lower_room_mind_scrape_bw_graffiti_30s")
    parser.add_argument("--run-id", default="lower_room_mind_scrape_bw_graffiti_30s")
    parser.add_argument("--seconds", type=float, default=30.0)
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


def _put(frame: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.5, color=(235, 235, 235), thickness: int = 1) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _render(args: argparse.Namespace) -> dict[str, Any]:
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
    skyline = _fit_cover(_load_gray(args.skyline), W, H)
    proof_a = _load_gray(args.proof_a)
    proof_b = _load_gray(args.proof_b)
    frame_count = int(args.seconds * FPS)
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"could not write {raw_path}")

    for frame_idx in range(frame_count):
        t = frame_idx / FPS
        env = {stem: float(envelopes[stem][frame_idx]) for stem in STEM_NAMES}
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
        _put(rgb, "TRUEVISION STATE PROOF | CLEVELAND LINEART + B/W GRAFFITI PROOFS | STEM-DRIVEN INK / CONTRAST / CAMERA PRESSURE", (24, H - 18), 0.43)
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
        "encoder": used_encoder,
        "boundary": {
            "source_assets_used": True,
            "openai_generation_used": False,
            "new_graffiti_language_added": False,
            "lyrics_rendered": False,
            "generated_media_is_visualization": True,
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
