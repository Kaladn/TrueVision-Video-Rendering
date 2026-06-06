from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AVATAR = Path(__file__).resolve().parent / "avatar_assets" / "truvision_avatar_v1" / "poses"


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    title: str
    pose_sequence: tuple[str, ...]
    slide_indices: tuple[int, ...]
    motion: float = 0.16
    system_labels: tuple[str, ...] = ()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT_SMALL = font(20)
FONT_MED = font(30)
FONT_BIG = font(46, bold=True)


def ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fit_image(img: Image.Image, box: tuple[int, int], fill: bool = False) -> Image.Image:
    bw, bh = box
    iw, ih = img.size
    scale = max(bw / iw, bh / ih) if fill else min(bw / iw, bh / ih)
    resized = img.resize((int(iw * scale), int(ih * scale)), Image.Resampling.LANCZOS)
    if fill:
        left = max(0, (resized.width - bw) // 2)
        top = max(0, (resized.height - bh) // 2)
        return resized.crop((left, top, left + bw, top + bh))
    canvas = Image.new("RGB", (bw, bh), (8, 11, 18))
    canvas.paste(resized, ((bw - resized.width) // 2, (bh - resized.height) // 2))
    return canvas


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def paste_rounded(base: Image.Image, img: Image.Image, xy: tuple[int, int], radius: int) -> None:
    base.paste(img, xy, rounded_mask(img.size, radius))


def draw_panel(draw: ImageDraw.ImageDraw, xyxy: tuple[int, int, int, int], color: tuple[int, int, int], width: int = 2) -> None:
    draw.rounded_rectangle(xyxy, radius=18, outline=color, width=width, fill=(10, 14, 22))


def draw_fit_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_width: int,
    start_size: int,
    fill: tuple[int, int, int],
    bold: bool = False,
) -> None:
    size = start_size
    while size > 24:
        candidate = font(size, bold=bold)
        bbox = draw.textbbox(xy, text, font=candidate)
        if bbox[2] - bbox[0] <= max_width:
            draw.text(xy, text, font=candidate, fill=fill)
            return
        size -= 2
    draw.text(xy, text, font=font(size, bold=bold), fill=fill)


def blend_images(a: Image.Image, b: Image.Image, fade: float) -> Image.Image:
    if fade <= 0.001:
        return a
    if fade >= 0.999:
        return b
    return Image.blend(a, b, fade)


def item_for_time(items: tuple[Any, ...], start: float, end: float, t: float) -> tuple[Any, Any, float]:
    if len(items) == 1:
        return items[0], items[0], 1.0
    span = max(0.001, end - start)
    local = (t - start) / span
    scaled = local * len(items)
    idx = min(len(items) - 1, int(scaled))
    next_idx = min(len(items) - 1, idx + 1)
    frac = scaled - idx
    fade = ease(max(0.0, min(1.0, (frac - 0.72) / 0.28)))
    return items[idx], items[next_idx], fade


def build_background(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), (7, 9, 14))
    px = img.load()
    for y in range(height):
        for x in range(width):
            sx = x / width
            sy = y / height
            glow = int(22 * math.exp(-((sx - 0.74) ** 2 + (sy - 0.34) ** 2) / 0.08))
            px[x, y] = (7 + glow // 4, 12 + glow // 3, 18 + glow)
    return img


def draw_avatar_system_panel(
    pose: Image.Image,
    slide: Image.Image,
    slide_idx: int,
    segment_idx: int,
    seg: Segment,
    t: float,
    direction_mode: str,
) -> Image.Image:
    panel = Image.new("RGB", (476, 476), (6, 10, 16))
    draw = ImageDraw.Draw(panel)
    if direction_mode == "slide_alternate":
        facing_right = slide_idx % 2 == 0
    else:
        facing_right = segment_idx % 2 == 0
    console_x = 274 if facing_right else 22
    avatar_x = 0 if facing_right else 176

    for y in range(476):
        shade = int(18 * (y / 476))
        draw.line((0, y, 476, y), fill=(6 + shade // 3, 10 + shade // 4, 16 + shade))
    scan_y = int(46 + (math.sin(t * 0.8) * 0.5 + 0.5) * 344)
    draw.line((22, scan_y, 454, scan_y), fill=(31, 86, 110), width=1)

    crop = pose.crop((62, 8, 272, 302)).resize((292, 408), Image.Resampling.LANCZOS)
    if not facing_right:
        crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    fade = Image.new("L", crop.size, 0)
    fdraw = ImageDraw.Draw(fade)
    fdraw.rounded_rectangle((0, 0, crop.width, crop.height), radius=16, fill=245)
    fdraw.rectangle((0, crop.height - 36, crop.width, crop.height), fill=120)
    panel.paste(crop, (avatar_x, 44 + int(6 * math.sin(t * 0.11))), fade)

    draw.rounded_rectangle((console_x - 12, 20, console_x + 192, 420), radius=14, outline=(52, 73, 90), width=1, fill=(5, 9, 15))
    thumb = fit_image(slide, (160, 90), fill=True)
    draw.rounded_rectangle((console_x - 4, 32, console_x + 164, 130), radius=9, outline=(82, 170, 210), width=2, fill=(8, 13, 20))
    paste_rounded(panel, thumb, (console_x, 36), 7)
    draw.text((console_x, 138), f"SURFACE {slide_idx:02d}", font=FONT_SMALL, fill=(190, 210, 222))

    labels = seg.system_labels or ("surface", "state", "route", "receipt")
    draw.text((console_x, 174), "SYSTEM STATE", font=FONT_SMALL, fill=(126, 148, 160))
    colors = ((82, 170, 210), (212, 121, 76), (106, 179, 105), (136, 116, 214))
    for i, label in enumerate(tuple(labels)[:4]):
        y = 208 + i * 35
        color = colors[i % 4]
        pulse = int(20 + 20 * (math.sin(t * 1.2 + i) * 0.5 + 0.5))
        draw.rounded_rectangle((console_x, y, console_x + 160, y + 24), radius=6, outline=color, width=1, fill=(8 + pulse // 8, 12 + pulse // 10, 19 + pulse // 6))
        draw.text((console_x + 10, y + 2), str(label)[:16], font=FONT_SMALL, fill=(225, 232, 238))

    node_origin_x = console_x + 12
    node_origin_y = 368
    nodes = [(0, 0), (42, -18), (84, 0), (126, -18)]
    for i, (nx, ny) in enumerate(nodes):
        x = node_origin_x + nx
        y = node_origin_y + ny
        if i:
            px = node_origin_x + nodes[i - 1][0] + 14
            py = node_origin_y + nodes[i - 1][1] + 14
            draw.line((px, py, x + 14, y + 14), fill=(70, 91, 106), width=2)
        r = 10 + int(3 * (math.sin(t * 1.6 + i) * 0.5 + 0.5))
        draw.ellipse((x + 14 - r, y + 14 - r, x + 14 + r, y + 14 + r), outline=colors[i], width=2, fill=(10, 15, 23))

    draw.text((22, 430), "POSE " + ("RIGHT" if facing_right else "LEFT"), font=FONT_SMALL, fill=(143, 168, 184))
    title_x = console_x if facing_right else 274
    draw.text((title_x, 430), seg.title.upper()[:18], font=FONT_SMALL, fill=(212, 121, 76))
    return panel.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))


def parse_segments(raw: list[dict[str, Any]]) -> list[Segment]:
    return [
        Segment(
            start=float(item["start"]),
            end=float(item["end"]),
            title=str(item["title"]),
            pose_sequence=tuple(item.get("pose_sequence") or ("01_neutral_forward.png",)),
            slide_indices=tuple(int(v) for v in item.get("slide_indices", item.get("slides", (0,)))),
            motion=float(item.get("motion", 0.16)),
            system_labels=tuple(str(v) for v in item.get("system_labels", ())),
        )
        for item in raw
    ]


def render_frame(
    t: float,
    slides: list[Image.Image],
    poses: dict[str, Image.Image],
    segments: list[Segment],
    base_bg: Image.Image,
    config: dict[str, Any],
) -> Image.Image:
    width, height = base_bg.size
    seg_idx, seg = next(((i, s) for i, s in enumerate(segments) if s.start <= t < s.end), (len(segments) - 1, segments[-1]))
    local = (t - seg.start) / max(0.001, seg.end - seg.start)
    frame = base_bg.copy()
    draw = ImageDraw.Draw(frame)
    sweep_x = int((math.sin(t * 0.045) * 0.5 + 0.5) * width)
    draw.line((sweep_x, 0, sweep_x - 420, height), fill=(15, 28, 40), width=3)

    slide_idx, next_slide_idx, slide_fade = item_for_time(seg.slide_indices, seg.start, seg.end, t)
    slide = blend_images(slides[slide_idx], slides[next_slide_idx], slide_fade)
    z = 1.0 + seg.motion * 0.018 * math.sin(local * math.pi)
    if z > 1.001:
        zw = int(slide.width * z)
        zh = int(slide.height * z)
        zimg = slide.resize((zw, zh), Image.Resampling.LANCZOS)
        slide = zimg.crop(((zw - slide.width) // 2, (zh - slide.height) // 2, (zw + slide.width) // 2, (zh + slide.height) // 2))

    draw_panel(draw, (48, 60, 1288, 806), (82, 170, 210), 2)
    paste_rounded(frame, slide, (70, 92), 14)

    pose_name, next_pose_name, pose_fade = item_for_time(seg.pose_sequence, seg.start, seg.end, t)
    pose = blend_images(poses[pose_name], poses[next_pose_name], pose_fade).convert("RGB")
    pose_canvas = draw_avatar_system_panel(pose, slide, slide_idx, seg_idx, seg, t, config.get("direction_mode", "segment_alternate"))
    avatar_x = 1370 + int(12 * math.sin(t * 0.15) * seg.motion)
    avatar_y = 205 + int(8 * math.sin(t * 0.11 + 1.2) * seg.motion)
    draw_panel(draw, (1328, 156, 1858, 750), (212, 121, 76), 2)
    paste_rounded(frame, pose_canvas, (avatar_x, avatar_y), 22)

    draw.text((1328, 790), config.get("avatar_label", "TRUVISION AVATAR"), font=FONT_SMALL, fill=(143, 168, 184))
    draw_fit_text(draw, (1328, 804), seg.title.upper(), 530, 46, (238, 244, 248), bold=True)
    draw.line((1328, 892, 1858, 892), fill=(212, 121, 76), width=2)

    duration = float(config["duration_seconds"])
    progress = t / duration
    draw.rounded_rectangle((70, 864, 1858, 944), radius=18, outline=(52, 73, 90), width=2, fill=(8, 12, 18))
    draw.rectangle((94, 928, 1834, 938), fill=(26, 36, 48))
    draw.rectangle((94, 928, 94 + int(1740 * progress), 938), fill=(82, 170, 210))
    time_label = f"{int(t // 60)}:{int(t % 60):02d} / {int(duration // 60)}:{int(duration % 60):02d}"
    draw.text((94, 882), time_label, font=FONT_MED, fill=(228, 235, 240))
    draw.text((1328, 882), config.get("motion_caption", "80% stillness / 20% intentional motion"), font=FONT_MED, fill=(190, 202, 211))
    draw.text((1520, 1018), config.get("footer_credit", "FULL TRUEVISION PRODUCTION"), font=FONT_SMALL, fill=(126, 148, 160))

    if config.get("show_production_card", True) and (t < 6.5 or t >= duration - 8.2):
        alpha = 1.0 - ease(max(0.0, min(1.0, (t - 4.6) / 1.9))) if t < 6.5 else ease(max(0.0, min(1.0, (t - (duration - 8.2)) / 2.8)))
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rounded_rectangle((520, 386, 1400, 646), radius=22, fill=(4, 7, 12, int(180 * alpha)), outline=(82, 170, 210, int(210 * alpha)), width=2)
        odraw.text((628, 444), config.get("production_card_title", "A FULL TRUEVISION PRODUCTION"), font=FONT_BIG, fill=(238, 244, 248, int(245 * alpha)))
        odraw.text((704, 514), config.get("film_title", "TrueVision Avatar Film"), font=FONT_MED, fill=(198, 212, 222, int(245 * alpha)))
        odraw.text((804, 562), config.get("production_lab", "TrueVision Generation Lab"), font=FONT_SMALL, fill=(150, 178, 194, int(245 * alpha)))
        frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
    return frame


def mix_audio(config: dict[str, Any], output_dir: Path) -> Path:
    audio = config["audio"]
    if audio.get("mixed"):
        return Path(audio["mixed"])
    vocal = Path(audio["vocal"])
    instrumental = Path(audio["instrumental"])
    audio_mix = output_dir / "assets" / "audio" / "avatar_film_mix.wav"
    audio_mix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(vocal),
            "-i",
            str(instrumental),
            "-filter_complex",
            f"[0:a]volume={audio.get('vocal_volume', 1.0)}[a0];[1:a]volume={audio.get('instrumental_volume', 0.48)}[a1];[a0][a1]amix=inputs=2:duration=shortest:normalize=0[a]",
            "-map",
            "[a]",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(audio_mix),
        ],
        check=True,
    )
    return audio_mix


def render_project(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(exist_ok=True)

    width, height = config.get("resolution", [1920, 1080])
    fps = int(config.get("fps", 24))
    duration = float(config["duration_seconds"])
    segments = parse_segments(config["segments"])
    slide_paths = [Path(p) for p in config["slides"]]
    avatar_dir = Path(config.get("avatar_pose_dir", DEFAULT_AVATAR))
    pose_paths = sorted(avatar_dir.glob("*.png"))

    slides = [fit_image(Image.open(path).convert("RGB"), (1220, 686), fill=False) for path in slide_paths]
    poses = {path.name: fit_image(Image.open(path).convert("RGB"), (350, 348), fill=True) for path in pose_paths}
    base_bg = build_background(width, height)
    audio_mix = mix_audio(config, output_dir)
    final_video = output_dir / config.get("output_name", "truvision_avatar_film.mp4")

    (output_dir / "manifests" / "render_project.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    total_frames = int(math.ceil(duration * fps))
    cmd = [
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
        "-i",
        str(audio_mix),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(config.get("crf", 19)),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(final_video),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame_no in range(total_frames):
        proc.stdin.write(render_frame(frame_no / fps, slides, poses, segments, base_bg, config).tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc:
        raise SystemExit(rc)
    receipt = {
        "video": str(final_video),
        "config": str(config_path),
        "avatar_pose_dir": str(avatar_dir),
        "duration_seconds": duration,
        "fps": fps,
        "frames_rendered": total_frames,
        "resolution": [width, height],
        "source_slides": len(slide_paths),
        "source_poses": len(pose_paths),
    }
    (output_dir / "manifests" / "render_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return final_video
