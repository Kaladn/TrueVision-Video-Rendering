"""Motion-addressed avatar state sampling and proof rendering."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSES = ROOT / "truevision_runtime" / "rendering" / "avatar_assets" / "truvision_avatar_v1" / "poses"
DEFAULT_OUTPUT = ROOT / "outputs" / "truvision_avatar_motion_address_loop_10s"

CLOCK_PATH = ("10", "11", "12", "1", "2", "1", "12", "11", "10")
YAW_BY_CLOCK = {"10": -40.0, "11": -20.0, "12": 0.0, "1": 20.0, "2": 40.0}
ATTENTION_BY_CLOCK = {
    "10": "observe",
    "11": "track",
    "12": "center",
    "1": "track",
    "2": "observe",
}


@dataclass(frozen=True)
class MotionState:
    t: float
    cycle_t: float
    clock: str
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    eye_yaw_deg: float
    attention: str
    anchor_quality: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def lerp(left: float, right: float, t: float) -> float:
    return left + (right - left) * t


def forward_state(unit_t: float, *, anchor_quality: str = "limited") -> MotionState:
    """Sample the authoritative 10 -> 2 motion law for one forward turn."""
    u = max(0.0, min(1.0, float(unit_t)))
    eased = smoothstep(u)
    yaw = lerp(-40.0, 40.0, eased)
    pitch = 12.0 * math.sin(math.pi * u)
    roll = 3.0 * math.sin(2.0 * math.pi * u)
    eye_u = max(0.0, min(1.0, u + 0.08))
    eye_yaw = lerp(-40.0, 40.0, smoothstep(eye_u))
    clock = nearest_clock(yaw)
    return MotionState(
        t=u,
        cycle_t=u,
        clock=clock,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        eye_yaw_deg=eye_yaw,
        attention=ATTENTION_BY_CLOCK[clock],
        anchor_quality=anchor_quality,
    )


def loop_state(time_seconds: float, *, duration_seconds: float, anchor_quality: str = "limited") -> MotionState:
    """Sample a 10 -> 2 -> 10 loop across a full render duration."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    phase = (float(time_seconds) % duration_seconds) / duration_seconds
    if phase <= 0.5:
        forward = phase * 2.0
        state = forward_state(forward, anchor_quality=anchor_quality)
        return MotionState(**{**asdict(state), "t": phase, "cycle_t": forward})
    reverse = (phase - 0.5) * 2.0
    state = forward_state(1.0 - reverse, anchor_quality=anchor_quality)
    return MotionState(**{**asdict(state), "t": phase, "cycle_t": 1.0 - reverse})


def nearest_clock(yaw_deg: float) -> str:
    return min(YAW_BY_CLOCK, key=lambda clock: abs(YAW_BY_CLOCK[clock] - yaw_deg))


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_image(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    scale = max(width / img.width, height / img.height)
    resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def load_anchor_images(pose_dir: Path, size: tuple[int, int]) -> dict[str, Image.Image]:
    """Load approximate visual anchors for motion addresses.

    Current retained stills are not true clock-addressed captures, so the
    renderer marks anchor quality as limited and uses mirroring to make a
    readable left-center-right proof without changing the source avatar kit.
    """
    pose = {path.name: fit_image(Image.open(path).convert("RGB"), size) for path in sorted(pose_dir.glob("*.png"))}
    required = {
        "center": "01_neutral_forward.png",
        "soft": "08_continuing_right.png",
        "far": "11_near_right_edge.png",
    }
    missing = [name for name in required.values() if name not in pose]
    if missing:
        raise ValueError(f"missing retained avatar pose assets: {missing}")
    return {
        "10": pose[required["far"]].transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "11": pose[required["soft"]].transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "12": pose[required["center"]],
        "1": pose[required["soft"]],
        "2": pose[required["far"]],
    }


def blend_images(left: Image.Image, right: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(left, right, max(0.0, min(1.0, float(alpha))))


def anchor_pair_for_yaw(yaw_deg: float) -> tuple[str, str, float]:
    ordered = [("10", -40.0), ("11", -20.0), ("12", 0.0), ("1", 20.0), ("2", 40.0)]
    if yaw_deg <= ordered[0][1]:
        return "10", "10", 0.0
    if yaw_deg >= ordered[-1][1]:
        return "2", "2", 0.0
    for (left_clock, left_yaw), (right_clock, right_yaw) in zip(ordered, ordered[1:]):
        if left_yaw <= yaw_deg <= right_yaw:
            alpha = (yaw_deg - left_yaw) / (right_yaw - left_yaw)
            return left_clock, right_clock, alpha
    return "12", "12", 0.0


def render_state_frame(state: MotionState, anchors: dict[str, Image.Image], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    left_clock, right_clock, alpha = anchor_pair_for_yaw(state.yaw_deg)
    img = blend_images(anchors[left_clock], anchors[right_clock], alpha)

    # Pitch and roll are rendered as small 2D transforms for this v1 proof.
    scale = 1.05
    work = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    rotated = work.rotate(state.roll_deg, resample=Image.Resampling.BICUBIC, expand=False)
    canvas = Image.new("RGB", (width, height), (6, 9, 15))
    x = (width - rotated.width) // 2
    y = (height - rotated.height) // 2 - int(state.pitch_deg * 2.0)
    canvas.paste(rotated, (x, y))

    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, height - 118, width, height), fill=(3, 6, 12, 180))
    draw.text((24, height - 104), f"MOTION ADDRESS {state.clock} O'CLOCK", font=font(28, bold=True), fill=(236, 243, 247, 255))
    draw.text((24, height - 66), f"yaw {state.yaw_deg:+05.1f}  pitch {state.pitch_deg:+04.1f}  roll {state.roll_deg:+04.1f}", font=font(20), fill=(177, 201, 214, 255))
    draw.text((24, height - 38), f"eyes lead {state.eye_yaw_deg:+05.1f}  attention {state.attention}", font=font(20), fill=(212, 121, 76, 255))

    center_x = width - 142
    center_y = height - 68
    radius = 58
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline=(82, 170, 210, 210), width=2)
    for clock, yaw in YAW_BY_CLOCK.items():
        angle = math.radians(270.0 + yaw)
        px = center_x + int(math.sin(angle) * radius)
        py = center_y - int(math.cos(angle) * radius)
        fill = (238, 244, 248, 255) if clock == state.clock else (126, 148, 160, 210)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=fill)
        draw.text((px - 9, py + 8), clock, font=font(14, bold=clock == state.clock), fill=fill)
    eye_angle = math.radians(270.0 + state.eye_yaw_deg)
    eye_x = center_x + int(math.sin(eye_angle) * (radius - 16))
    eye_y = center_y - int(math.cos(eye_angle) * (radius - 16))
    draw.line((center_x, center_y, eye_x, eye_y), fill=(136, 116, 214, 255), width=3)
    return np.asarray(canvas, dtype=np.uint8)


def write_video(frames: list[np.ndarray], output_mp4: Path, fps: float, crf: int) -> None:
    height, width = frames[0].shape[:2]
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
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc:
        raise SystemExit(rc)


def render_motion_address_loop(
    *,
    pose_dir: Path = DEFAULT_POSES,
    output_dir: Path = DEFAULT_OUTPUT,
    duration_seconds: float = 10.0,
    fps: float = 60.0,
    size: tuple[int, int] = (700, 696),
    crf: int = 16,
    write_frame_pngs: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frames"
    anchors = load_anchor_images(pose_dir, size)
    frame_count = int(round(duration_seconds * fps))
    states = [loop_state(index / fps, duration_seconds=duration_seconds) for index in range(frame_count)]
    frames = [render_state_frame(state, anchors, size) for state in states]
    output_mp4 = output_dir / "truvision_avatar_motion_address_loop_10s.mp4"
    write_video(frames, output_mp4, fps, crf)
    if write_frame_pngs:
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(frame_dir / f"frame_{index:04d}.png")

    key_samples = {
        "forward_t_0_0": asdict(forward_state(0.0)),
        "forward_t_0_5": asdict(forward_state(0.5)),
        "forward_t_1_0": asdict(forward_state(1.0)),
        "loop_start": asdict(loop_state(0.0, duration_seconds=duration_seconds)),
        "loop_mid": asdict(loop_state(duration_seconds / 2.0, duration_seconds=duration_seconds)),
        "loop_last": asdict(loop_state(duration_seconds - (1.0 / fps), duration_seconds=duration_seconds)),
    }
    manifest = {
        "schema": "trueframegen_avatar_motion_address_manifest_v1",
        "created_at_utc": utc_now(),
        "law": "Pose images are anchors. State(t) is authority. FrameGen fills missing samples.",
        "pose_dir": str(pose_dir),
        "anchor_quality": "limited",
        "anchor_quality_reason": "Retained avatar stills are approximate visual anchors, not exact 10/11/12/1/2 clock captures.",
        "clock_path": list(CLOCK_PATH),
        "motion_address_table": [
            {
                "clock": clock,
                "yaw_deg": YAW_BY_CLOCK[clock],
                "attention": ATTENTION_BY_CLOCK[clock],
            }
            for clock in ("10", "11", "12", "1", "2")
        ],
        "formulas": {
            "smoothstep": "t^2 * (3 - 2t)",
            "yaw": "lerp(-40, +40, smoothstep(t))",
            "pitch": "12 * sin(pi * t)",
            "roll": "3 * sin(2 * pi * t)",
            "eye_yaw": "yaw(t + 0.08)",
        },
        "duration_seconds": duration_seconds,
        "fps": fps,
        "sampled_frame_count": frame_count,
        "resolution": [size[0], size[1]],
        "key_samples": key_samples,
        "outputs": {
            "video": str(output_mp4),
            "frame_pngs": str(frame_dir) if write_frame_pngs else None,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
