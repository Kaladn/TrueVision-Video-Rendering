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
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "truedepth_fog_reveal_samples"
DEFAULT_PROFILE_1 = ROOT / "storage" / "artifacts" / "angular_seismic" / "fog_video_project_1_virtual_middle_content_16dir_profile.json"
DEFAULT_PROFILE_2 = ROOT / "storage" / "artifacts" / "angular_seismic" / "fog_video_project_2_virtual_middle_content_16dir_profile.json"

TOOL_MODES = {
    "fog_field": "Fog Density Field",
    "volumetric_fog": "Volumetric Fog Density Slices",
    "forward_motion": "Forward Motion / Resolve Pressure",
    "truedepth": "TrueDepth Parallax Layers",
    "object_reveal": "Object Reveal Through Fog",
    "angular_drift": "16-Side Angular Drift",
    "effect_state_transform": "Copied Effect State Transform",
}


def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_")
    return safe or "truedepth_fog_reveal"


def _read_profile(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_signature(profile_1: Path = DEFAULT_PROFILE_1, profile_2: Path = DEFAULT_PROFILE_2) -> dict[str, float]:
    p1 = _read_profile(profile_1)
    p2 = _read_profile(profile_2)
    profiles = [p for p in [p1, p2] if p]
    if not profiles:
        return {
            "fog_softness": 0.74,
            "direction_angle_degrees": 315.0,
            "resolve_pressure": 0.24,
            "motion_pressure": 0.04,
            "reflection_pressure": 0.34,
            "field_coherence": 0.18,
        }

    def avg(path: list[str], default: float = 0.0) -> float:
        values = []
        for profile in profiles:
            current: Any = profile
            for key in path:
                if not isinstance(current, dict) or key not in current:
                    current = None
                    break
                current = current[key]
            if isinstance(current, (int, float)):
                values.append(float(current))
        return float(np.mean(values)) if values else default

    angle_values = [
        float(profile["angular_signature"]["dominant_angle_degrees"])
        for profile in profiles
        if "angular_signature" in profile and "dominant_angle_degrees" in profile["angular_signature"]
    ]
    angle = float(np.mean(angle_values)) if angle_values else 315.0
    return {
        "fog_softness": avg(["candidate_profiles", "walking_camera_relation", "softness_mean"], 0.74),
        "direction_angle_degrees": angle,
        "resolve_pressure": avg(["seismic_trace", "impulse_peak"], 0.24),
        "motion_pressure": avg(["candidate_profiles", "walking_camera_relation", "motion_mean"], 0.04),
        "reflection_pressure": avg(["candidate_profiles", "glass_reflections", "peak"], 0.34),
        "field_coherence": avg(["angular_signature", "field_coherence_mean"], 0.18),
    }


def build_sample_plan() -> dict[str, Any]:
    samples = [
        {"mode": "fog_field", "label": TOOL_MODES["fog_field"], "active_tools": ["fog_field"]},
        {"mode": "volumetric_fog", "label": TOOL_MODES["volumetric_fog"], "active_tools": ["volumetric_fog"]},
        {"mode": "forward_motion", "label": TOOL_MODES["forward_motion"], "active_tools": ["forward_motion"]},
        {"mode": "truedepth", "label": TOOL_MODES["truedepth"], "active_tools": ["truedepth"]},
        {"mode": "object_reveal", "label": TOOL_MODES["object_reveal"], "active_tools": ["object_reveal"]},
        {"mode": "angular_drift", "label": TOOL_MODES["angular_drift"], "active_tools": ["angular_drift"]},
        {
            "mode": "effect_state_transform",
            "label": TOOL_MODES["effect_state_transform"],
            "active_tools": ["effect_state_transform"],
        },
        {
            "mode": "combined",
            "label": "Combined Fog + Volumetrics + Movement + TrueDepth + Reveal + 16-Side Drift + Effect Transform",
            "active_tools": list(TOOL_MODES),
        },
    ]
    return {
        "schema_version": "truevision_truedepth_fog_reveal_sample_plan_v1",
        "primitive_tool_count": len(TOOL_MODES),
        "primitive_tools": [{"id": key, "label": value} for key, value in TOOL_MODES.items()],
        "samples": samples,
        "boundary": {
            "individual_samples_one_tool_each": True,
            "combined_sample_uses_all_primitives": True,
            "source_video_frames_used": False,
            "signatures_used_as_behavior_controls": True,
        },
    }


def _smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    value = np.clip((x - edge0) / max(edge1 - edge0, 1.0e-6), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


@lru_cache(maxsize=24)
def _value_noise_cached(width: int, height: int, seed: int, scale: int) -> bytes:
    rng = np.random.default_rng(seed)
    low_w = max(2, width // scale + 3)
    low_h = max(2, height // scale + 3)
    low = rng.random((low_h, low_w), dtype=np.float32)
    noise = cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(2.0, scale / 10.0), sigmaY=max(2.0, scale / 10.0))
    return noise.astype(np.float32).tobytes()


def _value_noise(width: int, height: int, *, seed: int, scale: int = 96) -> np.ndarray:
    return np.frombuffer(_value_noise_cached(width, height, seed, scale), dtype=np.float32).reshape(height, width).copy()


def _fog_layer(width: int, height: int, t: float, signature: dict[str, float], *, drift_scale: float = 1.0) -> np.ndarray:
    base = _value_noise(width, height, seed=17, scale=86)
    curl = _value_noise(width, height, seed=31, scale=42)
    shift_x = int(math.cos(math.radians(signature["direction_angle_degrees"])) * t * 54.0 * drift_scale)
    shift_y = int(math.sin(math.radians(signature["direction_angle_degrees"])) * t * 30.0 * drift_scale)
    base = np.roll(base, shift=(shift_y, shift_x), axis=(0, 1))
    curl = np.roll(curl, shift=(int(-shift_y * 0.45), int(shift_x * 0.25)), axis=(0, 1))
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    depth_density = 0.45 + 0.46 * (1.0 - yy)
    fog = np.clip(0.62 * base + 0.38 * curl, 0.0, 1.0)
    return np.clip(fog * depth_density * (0.72 + signature["fog_softness"] * 0.35), 0.0, 1.0)


def _base_world(width: int, height: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = xx / max(width - 1, 1)
    ny = yy / max(height - 1, 1)
    sky = np.dstack(
        [
            0.18 - ny * 0.08,
            0.22 - ny * 0.10,
            0.24 - ny * 0.10,
        ]
    )
    road_mask = ny > (0.54 + np.abs(nx - 0.5) * 0.28)
    road = np.dstack([np.full_like(nx, 0.12), np.full_like(nx, 0.13), np.full_like(nx, 0.12)])
    frame = sky.copy()
    frame[road_mask] = road[road_mask]
    horizon = np.exp(-((ny - 0.48) ** 2) / 0.003)
    frame += horizon[:, :, None] * np.array([0.16, 0.17, 0.15])
    return np.clip(frame, 0.0, 1.0), {"nx": nx, "ny": ny, "road_mask": road_mask}


def _draw_depth_scene(frame: np.ndarray, fields: dict[str, np.ndarray], t: float, *, depth_strength: float, reveal: float) -> np.ndarray:
    height, width = frame.shape[:2]
    nx = fields["nx"]
    ny = fields["ny"]
    # Distant tree/building bands.
    for layer, y_base, color, parallax in [
        (0, 0.42, np.array([0.10, 0.14, 0.11]), -0.012),
        (1, 0.49, np.array([0.12, 0.15, 0.13]), -0.026),
        (2, 0.58, np.array([0.09, 0.10, 0.09]), -0.048),
    ]:
        offset = depth_strength * parallax * (t - 0.5)
        wave = 0.018 * np.sin((nx + layer * 0.17 + t * 0.12) * math.tau * 4.0)
        mask = (ny > y_base + wave) & (ny < y_base + 0.14 + wave)
        alpha = (0.14 + layer * 0.08) * (0.25 + 0.75 * reveal)
        frame[mask] = frame[mask] * (1.0 - alpha) + color * alpha
        # vertical trunks/posts. Deterministic variation keeps the generated
        # scene from reading as a procedural fence.
        count = 9 + layer * 3
        for post_index, x0 in enumerate(np.linspace(0.12, 0.88, count)):
            jitter = 0.013 * math.sin(post_index * 2.17 + layer * 1.31)
            x_pos = x0 + offset + jitter
            width_var = (0.0022 + layer * 0.0010) * (0.72 + 0.55 * abs(math.sin(post_index * 1.93 + layer)))
            top_var = y_base - (0.055 + 0.075 * abs(math.cos(post_index * 1.41 + layer * 0.7)))
            bottom_var = y_base + 0.10 + 0.08 * abs(math.sin(post_index * 1.13 + layer))
            lean = (ny - y_base) * 0.010 * math.sin(post_index * 0.77 + layer)
            trunk = (np.abs(nx - (x_pos + lean)) < width_var) & (ny > top_var) & (ny < bottom_var)
            frame[trunk] = frame[trunk] * 0.55 + color * 0.45
    return frame


def _draw_reveal_object(frame: np.ndarray, fields: dict[str, np.ndarray], t: float, *, reveal: float) -> np.ndarray:
    nx = fields["nx"]
    ny = fields["ny"]
    cx = 0.55 - 0.07 * (1.0 - reveal)
    cy = 0.50 + 0.05 * (1.0 - reveal)
    scale = 0.045 + 0.075 * reveal
    sign = (np.abs(nx - cx) < scale * 0.72) & (np.abs(ny - cy) < scale * 0.34)
    post = (np.abs(nx - cx) < scale * 0.07) & (ny > cy + scale * 0.34) & (ny < cy + scale * 1.24)
    car = ((nx - (0.38 + 0.11 * reveal)) ** 2 / (scale * 1.4) ** 2 + (ny - (0.66 - 0.03 * reveal)) ** 2 / (scale * 0.42) ** 2) < 1.0
    color = np.array([0.34, 0.38, 0.36]) * (0.38 + 0.62 * reveal)
    edge_color = np.array([0.86, 0.90, 0.82])
    for mask, alpha in [(sign, 0.72 * reveal), (post, 0.62 * reveal), (car, 0.55 * reveal)]:
        frame[mask] = frame[mask] * (1.0 - alpha) + color * alpha
    edge = cv2.Canny((sign.astype(np.uint8) * 255), 10, 60).astype(bool)
    frame[edge] = frame[edge] * 0.35 + edge_color * 0.65 * reveal
    return frame


def _apply_forward_motion(frame: np.ndarray, fields: dict[str, np.ndarray], t: float, *, strength: float) -> np.ndarray:
    height, width = frame.shape[:2]
    nx = fields["nx"].astype(np.float32)
    ny = fields["ny"].astype(np.float32)
    center_x = 0.50
    center_y = 0.55
    pressure = strength * (0.15 + 0.85 * t)
    map_x = (nx - (nx - center_x) * pressure * (ny ** 1.7) * 0.20) * (width - 1)
    map_y = (ny - (ny - center_y) * pressure * (ny ** 1.7) * 0.16) * (height - 1)
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _apply_fog(frame: np.ndarray, fog: np.ndarray, *, amount: float) -> np.ndarray:
    fog_color = np.dstack([fog * 0.80 + 0.18, fog * 0.84 + 0.18, fog * 0.82 + 0.17])
    alpha = np.clip(fog * amount, 0.0, 0.92)
    return np.clip(frame * (1.0 - alpha[:, :, None]) + fog_color * alpha[:, :, None], 0.0, 1.0)


def _copy_effect_transform_state(signature: dict[str, float], t: float) -> dict[str, float]:
    """Copy learned behavior controls, then change state instead of copying frames."""
    source_angle = float(signature["direction_angle_degrees"])
    target_angle = (source_angle + 70.0) % 360.0
    blend = float(_smoothstep(0.18, 0.82, t))
    return {
        "copied_source_angle_degrees": source_angle,
        "transformed_angle_degrees": source_angle * (1.0 - blend) + target_angle * blend,
        "density_scale": 0.82 + 0.34 * math.sin(t * math.pi),
        "near_slice_weight": 0.38 + 0.34 * blend,
        "mid_slice_weight": 0.44,
        "far_slice_weight": 0.62 - 0.30 * blend,
        "reveal_gate": blend,
        "depth_collapse": float(_smoothstep(0.56, 0.92, t)),
    }


def _speed01(vehicle_speed_mph: float) -> float:
    return float(np.clip((vehicle_speed_mph - 25.0) / 35.0, 0.0, 1.0))


def _advect_for_vehicle_speed(density: np.ndarray, fields: dict[str, np.ndarray], vehicle_speed_mph: float) -> np.ndarray:
    speed = _speed01(vehicle_speed_mph)
    if speed <= 0.01:
        return density
    height, width = density.shape
    nx = fields["nx"].astype(np.float32)
    ny = fields["ny"].astype(np.float32)
    center_x = 0.50
    vanishing_y = 0.53
    stretch = speed * 0.040
    accum = density * 0.40
    weight = 0.40
    for step, step_weight in [(1.0, 0.26), (2.0, 0.18), (3.0, 0.10), (4.0, 0.06)]:
        depth_gate = np.clip((ny - vanishing_y) * 1.85, 0.0, 1.0)
        dx = (nx - center_x) * stretch * step * (0.25 + depth_gate)
        dy = (ny - vanishing_y) * stretch * step * (0.65 + 0.90 * depth_gate)
        map_x = np.clip((nx - dx) * (width - 1), 0, width - 1).astype(np.float32)
        map_y = np.clip((ny - dy) * (height - 1), 0, height - 1).astype(np.float32)
        shifted = cv2.remap(density, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        accum += shifted * step_weight
        weight += step_weight
    smeared = accum / weight
    sheet = cv2.GaussianBlur(smeared, (0, 0), sigmaX=10.0 + 14.0 * speed, sigmaY=4.0 + 7.0 * speed)
    return np.clip(smeared * (1.0 - 0.38 * speed) + sheet * (0.38 * speed), 0.0, 1.0)


def _depth_haze_density(width: int, height: int, t: float, signature: dict[str, float], *, vehicle_speed_mph: float = 45.0) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = xx.astype(np.float32) / max(width - 1, 1)
    ny = yy.astype(np.float32) / max(height - 1, 1)
    road_depth = np.clip(1.0 - np.abs(nx - 0.50) * 1.42, 0.0, 1.0) * np.clip(1.0 - ny * 0.90, 0.0, 1.0)
    horizon_bank = np.exp(-((ny - 0.49) ** 2) / 0.020)
    far_extinction = np.clip((1.0 - ny) ** 1.55, 0.0, 1.0)
    distance_haze = np.clip(0.22 + 0.58 * far_extinction + 0.42 * horizon_bank + 0.28 * road_depth, 0.0, 1.0)

    broad = _value_noise(width, height, seed=107, scale=180)
    veil = _value_noise(width, height, seed=113, scale=118)
    pocket = _value_noise(width, height, seed=119, scale=54)
    fine = _value_noise(width, height, seed=127, scale=72)
    angle = math.radians(signature["direction_angle_degrees"])
    drift_x = int(math.cos(angle) * t * width * 0.035)
    drift_y = int(math.sin(angle) * t * height * 0.018)
    broad = np.roll(broad, shift=(drift_y, drift_x), axis=(0, 1))
    veil = np.roll(veil, shift=(int(-drift_y * 0.45), int(drift_x * 0.35)), axis=(0, 1))
    pocket = np.roll(pocket, shift=(int(drift_y * 1.35), int(drift_x * -0.95)), axis=(0, 1))
    fine = np.roll(fine, shift=(int(drift_y * 0.25), int(-drift_x * 0.60)), axis=(0, 1))

    speed = _speed01(vehicle_speed_mph)
    pocket = cv2.GaussianBlur(
        pocket,
        (0, 0),
        sigmaX=8.0 + 20.0 * speed,
        sigmaY=3.0 + 8.0 * speed,
    )
    local_pockets = np.clip((pocket - (0.60 + 0.05 * speed)) * (1.25 - 0.38 * speed), 0.0, 1.0)
    bank_wave = 0.5 + 0.5 * np.sin((nx * 2.1 + ny * 3.4 + t * 0.22) * math.tau + veil * 1.4)
    foreground_drift = np.exp(-((ny - 0.82) ** 2) / 0.020) * (0.44 + 0.56 * broad)
    horizon_break = np.exp(-((ny - 0.50) ** 2) / 0.035) * (0.55 + 0.45 * bank_wave)

    organic_breakup = np.clip(
        0.70
        + 0.16 * (broad - 0.5)
        + 0.10 * (veil - 0.5)
        + 0.08 * (fine - 0.5)
        + (0.16 - 0.07 * speed) * local_pockets * (0.28 + 0.42 * far_extinction)
        + 0.18 * horizon_break
        + 0.14 * foreground_drift,
        0.42,
        0.98,
    )
    base_density = np.clip(distance_haze * organic_breakup, 0.0, 1.0).astype(np.float32)
    return _advect_for_vehicle_speed(base_density, {"nx": nx, "ny": ny}, vehicle_speed_mph).astype(np.float32)


def _apply_distance_edge_recovery(
    frame: np.ndarray,
    fields: dict[str, np.ndarray],
    *,
    reveal_gate: float,
    strength: float,
) -> np.ndarray:
    image = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 20, 70).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (0, 0), sigmaX=0.7, sigmaY=0.7)
    ny = fields["ny"]
    near_weight = np.clip((ny - 0.42) * 1.8, 0.0, 1.0)
    far_suppression = np.clip(1.0 - (1.0 - ny) * 1.10, 0.0, 1.0)
    recovery = edges * (0.28 + 0.72 * near_weight) * (0.35 + 0.65 * reveal_gate) * far_suppression
    lifted = frame + recovery[:, :, None] * strength
    return np.clip(lifted, 0.0, 1.0)


def _apply_volumetric_fog(
    frame: np.ndarray,
    fields: dict[str, np.ndarray],
    signature: dict[str, float],
    t: float,
    *,
    amount: float,
    transform_state: dict[str, float] | None = None,
    vehicle_speed_mph: float = 45.0,
) -> np.ndarray:
    state = transform_state or {
        "transformed_angle_degrees": float(signature["direction_angle_degrees"]),
        "density_scale": 1.0,
        "near_slice_weight": 0.48,
        "mid_slice_weight": 0.42,
        "far_slice_weight": 0.58,
        "reveal_gate": 0.0,
        "depth_collapse": 0.0,
    }
    angle_signature = dict(signature)
    angle_signature["direction_angle_degrees"] = float(state["transformed_angle_degrees"])
    height, width = frame.shape[:2]

    nx = fields["nx"]
    ny = fields["ny"]
    distance_haze = _depth_haze_density(width, height, t, angle_signature, vehicle_speed_mph=vehicle_speed_mph)
    far = np.clip(distance_haze * np.clip(1.15 - ny * 0.70, 0.0, 1.0), 0.0, 1.0)
    mid = np.clip(distance_haze * np.exp(-((ny - 0.50) ** 2) / 0.18), 0.0, 1.0)
    near = np.clip(distance_haze * np.clip((ny - 0.34) * 1.12, 0.0, 1.0), 0.0, 1.0)

    reveal_corridor = np.exp(-(((nx - 0.52) ** 2) / 0.030 + ((ny - 0.55) ** 2) / 0.12))
    reveal_carve = 1.0 - float(state["reveal_gate"]) * reveal_corridor * 0.76
    density = (
        far * float(state["far_slice_weight"])
        + mid * float(state["mid_slice_weight"])
        + near * float(state["near_slice_weight"])
    )
    density = np.clip(density * float(state["density_scale"]) * reveal_carve, 0.0, 1.0)

    blue_lift = 0.08 + 0.07 * float(state["depth_collapse"])
    horizon_light = np.exp(-((ny - 0.49) ** 2) / 0.010) * density
    fog_color = np.dstack(
        [
            density * 0.58 + 0.13 + horizon_light * 0.14,
            density * 0.66 + 0.15 + horizon_light * 0.16,
            density * 0.72 + 0.17 + blue_lift + horizon_light * 0.18,
        ]
    )
    alpha = np.clip(density * amount, 0.0, 0.76)
    hazed = np.clip(frame * (1.0 - alpha[:, :, None]) + fog_color * alpha[:, :, None], 0.0, 1.0)
    return _apply_distance_edge_recovery(
        hazed,
        fields,
        reveal_gate=float(state["reveal_gate"]),
        strength=0.035 + 0.055 * float(state["depth_collapse"]),
    )


def _apply_angular_drift(frame: np.ndarray, fields: dict[str, np.ndarray], signature: dict[str, float], t: float, *, amount: float) -> np.ndarray:
    height, width = frame.shape[:2]
    angle = math.radians(signature["direction_angle_degrees"])
    nx = fields["nx"].astype(np.float32)
    ny = fields["ny"].astype(np.float32)
    wave = np.sin((nx * math.cos(angle) + ny * math.sin(angle) + t * 0.65) * math.tau * 3.0)
    dx = wave * math.cos(angle) * amount * 9.0
    dy = wave * math.sin(angle) * amount * 7.0
    map_x = np.clip(nx * (width - 1) + dx, 0, width - 1).astype(np.float32)
    map_y = np.clip(ny * (height - 1) + dy, 0, height - 1).astype(np.float32)
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _overlay_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    cv2.putText(image, label, (28, image.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (188, 198, 194), 1, cv2.LINE_AA)
    return image


def render_frame(
    *,
    mode: str,
    frame_index: int,
    total_frames: int,
    width: int,
    height: int,
    signature: dict[str, float],
    label: bool = False,
    vehicle_speed_mph: float = 45.0,
) -> np.ndarray:
    t = frame_index / max(total_frames - 1, 1)
    frame, fields = _base_world(width, height)
    reveal = float(_smoothstep(0.32, 0.82, t))
    transform_state = _copy_effect_transform_state(signature, t)
    if mode in {"truedepth", "combined", "volumetric_fog", "effect_state_transform"}:
        frame = _draw_depth_scene(frame, fields, t, depth_strength=1.0, reveal=0.85 if mode == "truedepth" else reveal)
    elif mode in {"forward_motion", "fog_field", "object_reveal", "angular_drift"}:
        frame = _draw_depth_scene(frame, fields, 0.5, depth_strength=0.05, reveal=0.35)
    if mode in {"object_reveal", "combined", "effect_state_transform"}:
        frame = _draw_reveal_object(frame, fields, t, reveal=reveal)
    if mode in {"forward_motion", "combined"}:
        frame = _apply_forward_motion(frame, fields, t, strength=0.62 + signature["motion_pressure"] * 1.8)
    if mode in {"angular_drift", "combined"}:
        frame = _apply_angular_drift(frame, fields, signature, t, amount=0.75 if mode == "angular_drift" else 0.34)
    if mode == "fog_field":
        fog = _fog_layer(width, height, t, signature, drift_scale=1.0)
        frame = _apply_fog(frame, fog, amount=0.78)
    elif mode == "combined":
        frame = _apply_volumetric_fog(
            frame,
            fields,
            signature,
            t,
            amount=0.52 - reveal * 0.18,
            transform_state=transform_state,
            vehicle_speed_mph=vehicle_speed_mph,
        )
    elif mode == "volumetric_fog":
        frame = _apply_volumetric_fog(frame, fields, signature, t, amount=0.78, vehicle_speed_mph=vehicle_speed_mph)
    elif mode == "effect_state_transform":
        frame = _apply_volumetric_fog(
            frame,
            fields,
            signature,
            t,
            amount=0.86,
            transform_state=transform_state,
            vehicle_speed_mph=vehicle_speed_mph,
        )
        frame = _apply_angular_drift(
            frame,
            fields,
            {"direction_angle_degrees": transform_state["transformed_angle_degrees"]},
            t,
            amount=0.20,
        )
    elif mode in {"truedepth", "object_reveal", "forward_motion", "angular_drift"}:
        # Very light air only, so the single tested tool stays readable.
        frame = _apply_fog(frame, _fog_layer(width, height, 0.15, signature, drift_scale=0.2), amount=0.10)
    image = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    if label:
        return _overlay_label(image.astype(np.float32) / 255.0, TOOL_MODES.get(mode, mode) if mode != "combined" else "Combined 7-tool proof")
    return image


def _build_ffmpeg_command(path: Path, *, width: int, height: int, fps: int, encoder: str) -> list[str]:
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
        command += ["-c:v", "h264_qsv", "-global_quality", "20", "-look_ahead", "0"]
    else:
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    command += ["-pix_fmt", "yuv420p", str(path)]
    return command


def render_video(
    *,
    mode: str,
    output_path: Path,
    signature: dict[str, float],
    duration: float,
    fps: int,
    width: int,
    height: int,
    encoder: str,
    label: bool,
    vehicle_speed_mph: float,
) -> dict[str, Any]:
    total_frames = int(round(duration * fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = output_path.parent / f"{output_path.stem}_frame_state.jsonl"
    command = _build_ffmpeg_command(output_path, width=width, height=height, fps=fps, encoder=encoder)
    start = time.perf_counter()
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert process.stdin is not None
        with state_path.open("w", encoding="utf-8") as state_file:
            for frame_index in range(total_frames):
                frame = render_frame(
                    mode=mode,
                    frame_index=frame_index,
                    total_frames=total_frames,
                    width=width,
                    height=height,
                    signature=signature,
                    label=label,
                    vehicle_speed_mph=vehicle_speed_mph,
                )
                state_file.write(
                    json.dumps(
                        {
                            "schema_version": "truevision_truedepth_fog_reveal_frame_state_v1",
                            "frame_index": frame_index,
                            "time_seconds": round(frame_index / max(fps, 1), 9),
                            "mode": mode,
                            "fps": fps,
                            "vehicle_speed_mph": vehicle_speed_mph,
                            "signature": signature,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                process.stdin.write(frame.tobytes())
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
        used_encoder = encoder
    except Exception:
        if encoder == "libx264":
            raise
        command = _build_ffmpeg_command(output_path, width=width, height=height, fps=fps, encoder="libx264")
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert process.stdin is not None
        with state_path.open("w", encoding="utf-8") as state_file:
            for frame_index in range(total_frames):
                frame = render_frame(
                    mode=mode,
                    frame_index=frame_index,
                    total_frames=total_frames,
                    width=width,
                    height=height,
                    signature=signature,
                    label=label,
                    vehicle_speed_mph=vehicle_speed_mph,
                )
                state_file.write(
                    json.dumps(
                        {
                            "schema_version": "truevision_truedepth_fog_reveal_frame_state_v1",
                            "frame_index": frame_index,
                            "time_seconds": round(frame_index / max(fps, 1), 9),
                            "mode": mode,
                            "fps": fps,
                            "vehicle_speed_mph": vehicle_speed_mph,
                            "signature": signature,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                process.stdin.write(frame.tobytes())
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg fallback failed with exit code {return_code}")
        used_encoder = "libx264"
    wall = time.perf_counter() - start
    return {
        "mode": mode,
        "path": str(output_path),
        "frames": total_frames,
        "duration_seconds": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "encoder": used_encoder,
        "vehicle_speed_mph": vehicle_speed_mph,
        "frame_state_jsonl": str(state_path),
        "state_log_every": 1,
        "wall_seconds": round(wall, 6),
    }


def build_logging_compare(signature: dict[str, float]) -> dict[str, Any]:
    return {
        "schema_version": "truevision_new_layer_vs_old_logging_compare_v1",
        "old_logging": {
            "name": "Meter Grid",
            "answers": [
                "what changed",
                "how bright/soft/moving it was",
                "curve over time",
            ],
            "limits": [
                "weak directional structure",
                "weak propagation/arrival model",
                "harder to map into camera-space behavior",
            ],
        },
        "new_layer": {
            "name": "Angular-Seismic 16-side layer + volumetric state transform",
            "answers": [
                "where change travels",
                "which radial/director pockets carry energy",
                "how impulse rises, peaks, decays, and spreads",
                "how reveal direction maps into renderer movement",
                "how copied behavior can be reweighted into near/mid/far density slices",
                "how an effect can change state without copying source pixels",
            ],
            "signature_used": signature,
        },
        "why_it_matters": [
            "Old logging can say fog is soft and moving.",
            "The new layer can say fog/reveal energy is traveling in a measured direction.",
            "Renderers need direction, depth, and impulse to create believable motion instead of generic haze.",
            "Volumetric slices give fog a foreground, middle, and background instead of a flat veil.",
            "Effect-state transform copies measured manner, then changes density/direction/reveal state.",
            "This keeps signatures reusable without copying teacher footage.",
        ],
    }


def render_all_samples(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = "truedepth_fog_reveal_15s_60fps",
    duration: float = 15.0,
    fps: int = 60,
    width: int = 1280,
    height: int = 720,
    encoder: str = "h264_qsv",
    label: bool = True,
    modes: list[str] | None = None,
    vehicle_speed_mph: float = 45.0,
) -> dict[str, Any]:
    run_id = _safe_id(run_id)
    run_dir = output_root / run_id
    signature = load_signature()
    plan = build_sample_plan()
    selected_modes = set(modes or [sample["mode"] for sample in plan["samples"]])
    known_modes = {sample["mode"] for sample in plan["samples"]}
    unknown_modes = selected_modes.difference(known_modes)
    if unknown_modes:
        raise ValueError(f"unknown sample mode(s): {', '.join(sorted(unknown_modes))}")
    outputs = []
    for sample in plan["samples"]:
        mode = sample["mode"]
        if mode not in selected_modes:
            continue
        outputs.append(
            render_video(
                mode=mode,
                output_path=run_dir / f"{run_id}_{mode}.mp4",
                signature=signature,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                encoder=encoder,
                label=label,
                vehicle_speed_mph=vehicle_speed_mph,
            )
        )
    manifest = {
        "schema_version": "truevision_truedepth_fog_reveal_sample_manifest_v1",
        "run_id": run_id,
        "plan": plan,
        "signature": signature,
        "selected_modes": sorted(selected_modes),
        "vehicle_speed_mph": vehicle_speed_mph,
        "trudepth_contract_bundle": build_trudepth_contract_bundle("fog"),
        "outputs": outputs,
        "logging_compare": build_logging_compare(signature),
        "effect_state_transform": {
            "source": "learned fog/angular-seismic behavior signature",
            "copied_source_frames": False,
            "state_changes": [
                "direction rotates from source vector toward transformed vector",
                "density is reweighted across near/mid/far volume slices",
                "reveal corridor carves the copied density state instead of copying pixels",
                "depth collapse increases object readability while preserving fog as an effect",
            ],
        },
        "boundary": {
            "source_video_frames_used": False,
            "learned_signatures_used": True,
            "each_tool_individual_test_seconds": duration,
            "combined_sample_seconds": duration,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / f"{run_id}_manifest.json"
    compare_path = run_dir / f"{run_id}_new_layer_vs_old_logging_compare.json"
    manifest["manifest_json"] = str(manifest_path)
    manifest["compare_json"] = str(compare_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    compare_path.write_text(json.dumps(manifest["logging_compare"], indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Render 15-second TrueDepth fog reveal primitive samples.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="truedepth_fog_reveal_15s_60fps")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--encoder", default="h264_qsv", choices=["h264_qsv", "libx264"])
    parser.add_argument("--no-label", action="store_true")
    parser.add_argument("--modes", default="", help="Comma-separated sample modes. Empty renders all modes.")
    parser.add_argument("--vehicle-speed-mph", type=float, default=45.0)
    args = parser.parse_args()
    modes = [item.strip() for item in args.modes.split(",") if item.strip()] or None
    result = render_all_samples(
        output_root=Path(args.output_root),
        run_id=args.run_id,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        encoder=args.encoder,
        label=not args.no_label,
        modes=modes,
        vehicle_speed_mph=args.vehicle_speed_mph,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
