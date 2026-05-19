#!/usr/bin/env python3
"""Render The Basement as a stick-figure narrative state video.

The renderer stays literal to the story scenery: storm, basement door, window
creature, Frank falling, dragging sequence, Nether World, sword awakening,
demon battles, rescue, rift sealing, and ascension. No lyric text is drawn.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from truevision_edge_audio_river import capture_hardware, decode_audio_mono, measure_audio_features, sha256_file


DEFAULT_AUDIO = Path(r"C:\Users\mydyi\Downloads\The Basement.mp3")
DEFAULT_STORY = Path(r"C:\Users\mydyi\OneDrive\Documents\Desktop\The Basement.txt")
DEFAULT_LYRICS = Path(r"C:\Users\mydyi\OneDrive\Documents\Desktop\Full Album Lyrics_sound.txt")
DEFAULT_OUTPUT_ROOT = Path("outputs/the_basement_stick_narrative")
DEFAULT_RUN_ID = "the_basement_stick_narrative"


@dataclass(frozen=True)
class SceneBeat:
    scene_id: str
    start_norm: float
    end_norm: float
    story_anchor: str
    location: str
    palette: tuple[int, int, int]
    camera: str


SCENE_BEATS: tuple[SceneBeat, ...] = (
    SceneBeat("storm_blackout", 0.00, 0.09, "storm, lightning, oak tree, blackout", "living_room", (18, 22, 36), "wide_house"),
    SceneBeat("basement_door", 0.09, 0.20, "Ashley approaches the basement door", "hallway", (28, 24, 36), "slow_push"),
    SceneBeat("window_creature", 0.20, 0.31, "creature flashes in hallway window", "hallway_window", (18, 18, 32), "snap_flash"),
    SceneBeat("frank_falls", 0.31, 0.42, "Frank opens the door and falls into silence", "basement_stairs", (22, 18, 28), "tilt_down"),
    SceneBeat("dragged_down", 0.42, 0.52, "Ashley is grabbed and dragged deeper", "basement_stairs", (30, 12, 16), "shake_descent"),
    SceneBeat("red_rift_reveal", 0.52, 0.62, "red glow from the hole and creature reveal", "rift_floor", (58, 8, 8), "low_angle"),
    SceneBeat("mirror_message", 0.62, 0.70, "morning mirror warning and mother taken", "bathroom_mirror", (34, 42, 48), "still_dread"),
    SceneBeat("sword_awakening", 0.70, 0.78, "Grandpa guidance and sword awakening", "threshold_void", (34, 52, 68), "centered_icon"),
    SceneBeat("nether_descent", 0.78, 0.85, "Ashley descends through the bore", "nether_tunnel", (50, 10, 22), "forward_run"),
    SceneBeat("demon_battle", 0.85, 0.93, "demon battles, slow-motion strikes, black smoke", "nether_cavern", (64, 10, 18), "battle_orbit"),
    SceneBeat("mother_table", 0.93, 0.965, "mother bound on stone table, chains broken", "stone_table", (48, 18, 20), "rescue_push"),
    SceneBeat("rift_escape", 0.965, 0.988, "rift closing, rescue through portal", "closing_rift", (38, 14, 38), "urgent_pull"),
    SceneBeat("seal_ascend", 0.988, 1.00, "rift sealed by heart-light, ascension", "sealed_rift", (58, 62, 80), "whiteout"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def slug(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return clean.strip("_")[:96] or DEFAULT_RUN_ID


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def build_scene_schedule(duration_seconds: float) -> list[dict[str, Any]]:
    schedule = []
    for beat in SCENE_BEATS:
        start = duration_seconds * beat.start_norm
        end = duration_seconds * beat.end_norm
        schedule.append(
            {
                "scene_id": beat.scene_id,
                "start_seconds": round(start, 6),
                "end_seconds": round(end, 6),
                "duration_seconds": round(max(0.0, end - start), 6),
                "story_anchor": beat.story_anchor,
                "location": beat.location,
                "camera": beat.camera,
                "palette_bgr": list(beat.palette),
            }
        )
    return schedule


def scene_for_time(time_seconds: float, duration_seconds: float) -> SceneBeat:
    norm = 0.0 if duration_seconds <= 0 else min(0.999999, max(0.0, time_seconds / duration_seconds))
    for beat in SCENE_BEATS:
        if beat.start_norm <= norm < beat.end_norm:
            return beat
    return SCENE_BEATS[-1]


def _line(frame: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: tuple[int, int, int], thickness: int = 3) -> None:
    cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)


def _circle(frame: np.ndarray, p: tuple[int, int], r: int, color: tuple[int, int, int], thickness: int = 2) -> None:
    cv2.circle(frame, p, r, color, thickness, cv2.LINE_AA)


def draw_stick(
    frame: np.ndarray,
    x: float,
    y: float,
    *,
    scale: float = 1.0,
    color: tuple[int, int, int] = (230, 230, 240),
    pose: str = "stand",
    glow: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    s = max(0.25, scale)
    head = (int(x), int(y - 42 * s))
    neck = (int(x), int(y - 26 * s))
    hip = (int(x), int(y + 20 * s))
    if glow:
        _circle(frame, head, int(22 * s), glow, 2)
    _circle(frame, head, int(10 * s), color, 2)
    _line(frame, neck, hip, color, int(3 * s))
    arm_swing = math.sin(x * 0.01 + y * 0.03) * 10 * s
    if pose == "run":
        left_hand = (int(x - 24 * s), int(y - 8 * s + arm_swing))
        right_hand = (int(x + 26 * s), int(y - 10 * s - arm_swing))
        left_foot = (int(x - 22 * s), int(y + 48 * s))
        right_foot = (int(x + 24 * s), int(y + 45 * s))
    elif pose == "fall":
        left_hand = (int(x - 30 * s), int(y - 22 * s))
        right_hand = (int(x + 34 * s), int(y + 4 * s))
        left_foot = (int(x - 42 * s), int(y + 8 * s))
        right_foot = (int(x + 44 * s), int(y + 30 * s))
    elif pose == "fight":
        left_hand = (int(x - 28 * s), int(y - 6 * s))
        right_hand = (int(x + 38 * s), int(y - 26 * s))
        left_foot = (int(x - 24 * s), int(y + 50 * s))
        right_foot = (int(x + 26 * s), int(y + 50 * s))
    else:
        left_hand = (int(x - 24 * s), int(y - 2 * s))
        right_hand = (int(x + 24 * s), int(y - 2 * s))
        left_foot = (int(x - 18 * s), int(y + 50 * s))
        right_foot = (int(x + 18 * s), int(y + 50 * s))
    _line(frame, neck, left_hand, color, int(3 * s))
    _line(frame, neck, right_hand, color, int(3 * s))
    _line(frame, hip, left_foot, color, int(3 * s))
    _line(frame, hip, right_foot, color, int(3 * s))
    return {"head": head, "right_hand": right_hand, "hip": hip}


def draw_house(frame: np.ndarray, beat: float, intensity: float) -> None:
    h, w = frame.shape[:2]
    ground = int(h * 0.74)
    cv2.rectangle(frame, (int(w * 0.18), int(h * 0.38)), (int(w * 0.78), ground), (28, 25, 30), -1)
    pts = np.array([[int(w * 0.15), int(h * 0.38)], [int(w * 0.48), int(h * 0.20)], [int(w * 0.82), int(h * 0.38)]], np.int32)
    cv2.fillPoly(frame, [pts], (18, 17, 24))
    cv2.rectangle(frame, (int(w * 0.49), int(h * 0.50)), (int(w * 0.58), ground), (10, 9, 14), -1)
    cv2.rectangle(frame, (int(w * 0.27), int(h * 0.49)), (int(w * 0.39), int(h * 0.58)), (20, 28, 38), -1)
    if intensity > 0.5 or beat > 0.62:
        x = int(w * (0.15 + 0.6 * ((beat + intensity) % 1.0)))
        bolts = [(x, 0), (x - 28, int(h * 0.16)), (x + 12, int(h * 0.28)), (x - 35, int(h * 0.44))]
        for a, b in zip(bolts, bolts[1:]):
            _line(frame, a, b, (245, 245, 255), 5)


def draw_hallway(frame: np.ndarray, t: float, intensity: float, creature: bool = False) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), (18, 16, 23), -1)
    cv2.rectangle(frame, (int(w * 0.58), int(h * 0.20)), (int(w * 0.76), int(h * 0.78)), (7, 6, 10), -1)
    cv2.rectangle(frame, (int(w * 0.60), int(h * 0.23)), (int(w * 0.74), int(h * 0.76)), (22, 17, 20), 3)
    knob = (int(w * 0.72), int(h * 0.51))
    _circle(frame, knob, 5, (80, 62, 34), -1)
    cv2.rectangle(frame, (int(w * 0.14), int(h * 0.25)), (int(w * 0.32), int(h * 0.48)), (12, 15, 24), -1)
    cv2.rectangle(frame, (int(w * 0.14), int(h * 0.25)), (int(w * 0.32), int(h * 0.48)), (42, 46, 62), 2)
    if creature:
        glow = int(120 + 100 * intensity)
        _circle(frame, (int(w * 0.23), int(h * 0.35)), 34, (0, 0, 60 + glow // 3), 2)
        _circle(frame, (int(w * 0.21), int(h * 0.34)), 5, (0, 80, glow), -1)
        _circle(frame, (int(w * 0.25), int(h * 0.34)), 5, (0, 80, glow), -1)
        _line(frame, (int(w * 0.20), int(h * 0.30)), (int(w * 0.17), int(h * 0.25)), (0, 60, glow), 3)
        _line(frame, (int(w * 0.26), int(h * 0.30)), (int(w * 0.29), int(h * 0.25)), (0, 60, glow), 3)
    ashley_x = int(w * (0.42 + 0.08 * math.sin(t * 1.8)))
    draw_stick(frame, ashley_x, int(h * 0.63), scale=1.0, color=(220, 222, 235), glow=(40, 42, 58))


def draw_stairs(frame: np.ndarray, t: float, intensity: float, mode: str) -> None:
    h, w = frame.shape[:2]
    frame[:] = (14, 10, 15)
    for i in range(12):
        y = int(h * 0.25 + i * h * 0.055)
        _line(frame, (int(w * 0.42 - i * 18), y), (int(w * 0.86), y + int(i * 9)), (40, 34, 42), 2)
    cv2.rectangle(frame, (int(w * 0.08), int(h * 0.18)), (int(w * 0.38), int(h * 0.92)), (7, 6, 10), -1)
    red = int(60 + 150 * intensity)
    cv2.ellipse(frame, (int(w * 0.26), int(h * 0.82)), (int(w * 0.22), int(h * 0.08)), 0, 0, 360, (0, 0, red), -1)
    if mode == "fall":
        draw_stick(frame, int(w * (0.60 + 0.08 * math.sin(t * 8))), int(h * 0.46), scale=1.0, color=(210, 210, 210), pose="fall")
    else:
        draw_stick(frame, int(w * 0.50), int(h * 0.52), scale=1.0, color=(220, 222, 235), pose="fall")
        _line(frame, (int(w * 0.46), int(h * 0.47)), (int(w * 0.23), int(h * 0.78)), (12, 12, 18), 8)


def draw_nether(frame: np.ndarray, t: float, intensity: float, scene_id: str) -> None:
    h, w = frame.shape[:2]
    frame[:] = (10, 4, 9)
    red = int(50 + 150 * intensity)
    for i in range(7):
        y = int(h * (0.30 + i * 0.08))
        cv2.ellipse(frame, (int(w * 0.50), y), (int(w * (0.22 + i * 0.04)), int(h * 0.018)), 0, 0, 360, (0, 0, max(20, red - i * 12)), 1)
    cv2.ellipse(frame, (int(w * 0.50), int(h * 0.56)), (int(w * 0.30), int(h * 0.22)), 0, 0, 360, (0, 0, red), 2)
    sword_glow = (255, 245, 210)
    ashley = draw_stick(frame, int(w * 0.44), int(h * 0.62), scale=1.05, color=(225, 228, 240), pose="fight", glow=(80, 80, 120))
    hand = ashley["right_hand"]
    sword_tip = (int(hand[0] + w * 0.12), int(hand[1] - h * 0.12))
    _line(frame, hand, sword_tip, sword_glow, 5)
    _line(frame, hand, sword_tip, (255, 255, 255), 2)
    demon_count = 3 if scene_id in {"red_rift_reveal", "nether_descent"} else 8
    for i in range(demon_count):
        x = int(w * (0.18 + (i % 5) * 0.16 + 0.02 * math.sin(t * 2 + i)))
        y = int(h * (0.62 + (i // 5) * 0.12))
        _circle(frame, (x, y - 34), 13, (0, 0, red), 2)
        _circle(frame, (x - 5, y - 36), 2, (0, 110, 255), -1)
        _circle(frame, (x + 5, y - 36), 2, (0, 110, 255), -1)
        _line(frame, (x, y - 20), (x, y + 18), (20, 20, 28), 4)
        _line(frame, (x, y), (x - 18, y + 24), (20, 20, 28), 4)
        _line(frame, (x, y), (x + 18, y + 24), (20, 20, 28), 4)
    if scene_id == "mother_table":
        cv2.rectangle(frame, (int(w * 0.58), int(h * 0.58)), (int(w * 0.84), int(h * 0.63)), (38, 38, 44), -1)
        draw_stick(frame, int(w * 0.71), int(h * 0.56), scale=0.75, color=(190, 190, 202), pose="fall")
        _line(frame, (int(w * 0.58), int(h * 0.57)), (int(w * 0.84), int(h * 0.64)), (120, 120, 140), 2)
    if intensity > 0.72:
        for i in range(6):
            a = (int(w * 0.44), int(h * 0.44))
            b = (int(w * (0.18 + i * 0.12)), int(h * (0.26 + 0.08 * math.sin(t * 3 + i))))
            _line(frame, a, b, (255, 250, 230), 2)


def draw_mirror(frame: np.ndarray, t: float, intensity: float) -> None:
    h, w = frame.shape[:2]
    frame[:] = (22, 28, 34)
    cv2.rectangle(frame, (int(w * 0.30), int(h * 0.14)), (int(w * 0.70), int(h * 0.72)), (54, 68, 72), -1)
    cv2.rectangle(frame, (int(w * 0.30), int(h * 0.14)), (int(w * 0.70), int(h * 0.72)), (120, 132, 130), 3)
    for i in range(7):
        y = int(h * (0.28 + i * 0.05))
        _line(frame, (int(w * 0.38), y), (int(w * (0.60 + 0.02 * math.sin(t + i))), y + 6), (40, 20, 30 + int(80 * intensity)), 3)
    draw_stick(frame, int(w * 0.18), int(h * 0.74), scale=0.9, color=(220, 222, 235))


def draw_sword_awakening(frame: np.ndarray, t: float, intensity: float) -> None:
    h, w = frame.shape[:2]
    frame[:] = (8, 12, 18)
    center = (int(w * 0.50), int(h * 0.50))
    for r in range(40, int(min(w, h) * 0.45), 35):
        _circle(frame, center, r, (90, 90, 130), 1)
    draw_stick(frame, int(w * 0.50), int(h * 0.70), scale=1.0 + 0.15 * intensity, color=(235, 235, 245), pose="fight", glow=(110, 110, 160))
    _line(frame, (int(w * 0.50), int(h * 0.43)), (int(w * 0.50), int(h * 0.14)), (255, 255, 235), 7)
    _line(frame, (int(w * 0.50), int(h * 0.43)), (int(w * 0.50), int(h * 0.14)), (255, 255, 255), 3)
    draw_stick(frame, int(w * 0.22), int(h * 0.65), scale=0.75, color=(120, 140, 160), glow=(40, 50, 70))


def draw_escape(frame: np.ndarray, t: float, intensity: float, final: bool = False) -> None:
    h, w = frame.shape[:2]
    frame[:] = (8, 5, 12)
    center = (int(w * 0.70), int(h * 0.50))
    radius = int(min(w, h) * (0.22 if not final else 0.16) * (1.0 + 0.08 * math.sin(t * 5)))
    cv2.ellipse(frame, center, (radius, int(radius * 1.25)), 0, 0, 360, (255, 230, 220), 3)
    cv2.ellipse(frame, center, (int(radius * 0.7), int(radius * 0.95)), 0, 0, 360, (120, 30, 160), -1)
    draw_stick(frame, int(w * 0.43), int(h * 0.65), scale=1.05, color=(230, 232, 242), pose="run", glow=(90, 90, 120))
    draw_stick(frame, int(w * 0.51), int(h * 0.62), scale=0.72, color=(190, 190, 202), pose="fall")
    if final:
        _line(frame, (int(w * 0.43), int(h * 0.44)), center, (255, 255, 245), 8)
        frame[:] = np.clip(frame.astype(np.float32) + intensity * 110, 0, 255).astype(np.uint8)


def render_frame(width: int, height: int, feature: dict[str, float], duration_seconds: float) -> tuple[np.ndarray, dict[str, Any]]:
    time_seconds = float(feature["time_seconds"])
    scene = scene_for_time(time_seconds, duration_seconds)
    rms = float(feature.get("rms", 0.0))
    beat = float(feature.get("beat", 0.0))
    intensity = min(1.0, 0.35 + 0.55 * rms + 0.35 * beat)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = scene.palette
    if scene.scene_id == "storm_blackout":
        draw_house(frame, beat, intensity)
    elif scene.scene_id == "basement_door":
        draw_hallway(frame, time_seconds, intensity, creature=False)
    elif scene.scene_id == "window_creature":
        draw_hallway(frame, time_seconds, intensity, creature=True)
    elif scene.scene_id == "frank_falls":
        draw_stairs(frame, time_seconds, intensity, "fall")
    elif scene.scene_id == "dragged_down":
        draw_stairs(frame, time_seconds, intensity, "drag")
    elif scene.scene_id in {"red_rift_reveal", "nether_descent", "demon_battle", "mother_table"}:
        draw_nether(frame, time_seconds, intensity, scene.scene_id)
    elif scene.scene_id == "mirror_message":
        draw_mirror(frame, time_seconds, intensity)
    elif scene.scene_id == "sword_awakening":
        draw_sword_awakening(frame, time_seconds, intensity)
    elif scene.scene_id == "rift_escape":
        draw_escape(frame, time_seconds, intensity, final=False)
    else:
        draw_escape(frame, time_seconds, intensity, final=True)
    vignette = np.linspace(0.50, 1.0, width, dtype=np.float32)
    vignette = np.minimum(vignette, vignette[::-1])
    frame = np.clip(frame.astype(np.float32) * vignette[np.newaxis, :, np.newaxis], 0, 255).astype(np.uint8)
    return frame, {
        "frame_index": int(feature["frame_index"]),
        "time_seconds": round(time_seconds, 6),
        "scene_id": scene.scene_id,
        "location": scene.location,
        "story_anchor": scene.story_anchor,
        "rms": round(rms, 6),
        "beat": round(beat, 6),
        "intensity": round(intensity, 6),
    }


def _extract_story_metadata(story_path: Path, lyrics_path: Path) -> dict[str, Any]:
    story_text = story_path.read_text(encoding="utf-8", errors="replace")
    lyrics_text = lyrics_path.read_text(encoding="utf-8", errors="replace")
    anchors = [
        "storm",
        "basement door",
        "window creature",
        "Frank falls",
        "dragged down",
        "red rift",
        "Nether World",
        "mirror warning",
        "sword awakening",
        "stone table",
        "rift sealing",
        "ascension",
    ]
    return {
        "story_path": str(story_path),
        "story_sha256": sha256_file(story_path),
        "story_chars": len(story_text),
        "lyrics_path": str(lyrics_path),
        "lyrics_sha256": sha256_file(lyrics_path),
        "lyrics_chars": len(lyrics_text),
        "visual_policy": "literal_story_scenery_no_lyric_text",
        "anchors": anchors,
    }


def generate_basement_stick_narrative(
    *,
    audio_path: Path,
    story_path: Path,
    lyrics_path: Path,
    output_root: Path,
    run_id: str = DEFAULT_RUN_ID,
    width: int = 1280,
    height: int = 720,
    fps: int = 12,
    sample_rate: int = 44100,
    max_seconds: float | None = None,
    mux_audio: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = utc_now()
    audio_path = audio_path.resolve()
    story_path = story_path.resolve()
    lyrics_path = lyrics_path.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    if not story_path.exists():
        raise FileNotFoundError(story_path)
    if not lyrics_path.exists():
        raise FileNotFoundError(lyrics_path)
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

    visual_path = run_dir / f"{run_id}_visual_only.mp4"
    final_path = run_dir / f"{run_id}_with_audio.mp4" if mux_audio else visual_path
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

    sampled_states: list[dict[str, Any]] = []
    thumbnail_frame: np.ndarray | None = None
    scene_counts: dict[str, int] = {}
    with state_path.open("w", encoding="utf-8") as state_handle:
        for index, feature in enumerate(features):
            frame_state = dict(feature)
            frame_state["frame_index"] = index
            frame, metadata = render_frame(width, height, frame_state, duration_seconds)
            scene_counts[metadata["scene_id"]] = scene_counts.get(metadata["scene_id"], 0) + 1
            proc.stdin.write(frame.tobytes())
            state_handle.write(json.dumps(metadata, allow_nan=False) + "\n")
            if index % max(1, fps * 2) == 0:
                sampled_states.append(metadata)
            if index == min(len(features) - 1, max(1, fps * 30)):
                thumbnail_frame = frame.copy()
    proc.stdin.close()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg video encoder failed with exit code {return_code}")

    if thumbnail_frame is None:
        thumbnail_frame = render_frame(width, height, {**features[0], "frame_index": 0}, duration_seconds)[0]
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
        "claim": "the_basement_stick_figure_narrative_state_video",
        "boundary": {
            "generated_state_media": "synthetic_not_evidence",
            "story_source": "operator_supplied_local_text",
            "no_lyric_overlay": True,
            "no_dialogue_cards": True,
            "violence_style": "symbolic_stick_figure_black_smoke_no_gore",
        },
        "inputs": {
            "audio_path": str(audio_path),
            "audio_sha256": sha256_file(audio_path),
            "sample_rate": sample_rate,
            **_extract_story_metadata(story_path, lyrics_path),
        },
        "render": {
            "width": width,
            "height": height,
            "fps": fps,
            "frames": len(features),
            "duration_seconds": round(duration_seconds, 6),
            "style": "stick_figure_horror_narrative_audio_reactive_lighting",
            "scene_counts": scene_counts,
            "scene_schedule": build_scene_schedule(duration_seconds),
        },
        "audio_feature_summary": {
            key: {
                "mean": round(float(np.mean(values)), 6),
                "max": round(float(np.max(values)), 6),
                "std": round(float(np.std(values)), 6),
            }
            for key, values in feature_arrays.items()
        },
        "sampled_frame_states": sampled_states[:300],
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
    report_lines = [
        "# The Basement Stick Narrative Render Report",
        "",
        "## Claim",
        "",
        "A synthetic stick-figure narrative video was rendered from local story/lyrics sources and audio-driven state signals.",
        "",
        "## Boundary",
        "",
        "Generated state media is synthetic, not evidence. No lyric text or dialogue cards are rendered.",
        "",
        "## Story Arc",
        "",
    ]
    for beat in manifest["render"]["scene_schedule"]:
        report_lines.append(f"- `{beat['start_seconds']:.2f}s` to `{beat['end_seconds']:.2f}s`: `{beat['scene_id']}` — {beat['story_anchor']}")
    report_lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Video: `{final_path}`",
            f"- Manifest: `{manifest_path}`",
            f"- Frame state: `{state_path}`",
            f"- Thumbnail: `{thumb_path}`",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
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
    parser = argparse.ArgumentParser(description="Render The Basement as a stick-figure narrative state video.")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO))
    parser.add_argument("--story", default=str(DEFAULT_STORY))
    parser.add_argument("--lyrics", default=str(DEFAULT_LYRICS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--visual-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_basement_stick_narrative(
        audio_path=Path(args.audio),
        story_path=Path(args.story),
        lyrics_path=Path(args.lyrics),
        output_root=Path(args.output_root),
        run_id=args.run_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
        mux_audio=not args.visual_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
