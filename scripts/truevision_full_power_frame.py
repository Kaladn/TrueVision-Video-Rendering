#!/usr/bin/env python3
"""Generate one rich TrueVision-shaped synthetic state frame.

The point is not prompt art. This creates a detailed scene, samples it through
the TrueVision 16-channel cell-state shape, then reconstructs a clean frame
using more than rgb_mean.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from truevision_resonance_recorder import CELL_FEATURE_NAMES, build_video_cell_state, sha256_file


DEFAULT_OUTPUT_ROOT = Path("storage/artifacts/truevision_generated")
DEFAULT_RUN_ID = "person_field_clean_frame_full_power"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _feature_index(feature_names: list[str] | tuple[str, ...], name: str) -> int:
    return list(feature_names).index(name)


def _stable_noise(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, shape).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), 1.15)


def _smooth_repeat(feature: np.ndarray, output_shape: tuple[int, int], interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    out_h, out_w = output_shape
    return cv2.resize(feature.astype(np.float32), (out_w, out_h), interpolation=interpolation)


def build_detailed_person_field_frame(
    *,
    frame_shape: tuple[int, int] = (540, 960),
    progress: float = 0.56,
    phase_offset: float = 0.0,
) -> np.ndarray:
    """Render a detailed source state used only to derive cell vectors."""
    height, width = frame_shape
    yy, xx = np.indices((height, width))
    y = yy / max(1, height - 1)
    x = xx / max(1, width - 1)

    sky_blend = np.clip(y / 0.52, 0, 1)
    top = np.array([78, 133, 202], dtype=np.float32)
    lower = np.array([188, 211, 229], dtype=np.float32)
    frame = top * (1 - sky_blend[:, :, None]) + lower * sky_blend[:, :, None]

    cloud = (
        np.sin((x * 11.0) + phase_offset * 0.4)
        + np.sin((x * 17.0 + y * 6.0) - 0.7)
        + np.sin((x * 5.0 - y * 13.0) + 1.2)
    )
    cloud_mask = (y < 0.42) * np.clip((cloud - 1.25) / 1.7, 0, 1)
    frame = frame * (1 - cloud_mask[:, :, None] * 0.28) + np.array([235, 239, 238]) * cloud_mask[:, :, None] * 0.28

    horizon = int(height * 0.46)
    field_mask = yy >= horizon
    field_y = np.clip((yy - horizon) / max(1, height - horizon), 0, 1)
    far = np.array([96, 156, 83], dtype=np.float32)
    near = np.array([24, 104, 38], dtype=np.float32)
    field = far * (1 - field_y[:, :, None]) + near * field_y[:, :, None]
    grass_texture = (
        np.sin(xx * 0.055 + yy * 0.02 + phase_offset)
        + np.sin(xx * 0.19 - yy * 0.013)
        + np.sin((xx + yy) * 0.031)
    )
    field[:, :, 1] += grass_texture * (8 + field_y * 18)
    field[:, :, 0] -= grass_texture * (2 + field_y * 5)
    frame[field_mask] = field[field_mask]
    cv2.line(frame, (0, horizon), (width, horizon - 3), (82, 115, 74), 3, cv2.LINE_AA)

    rng = np.random.default_rng(404)
    for _ in range(1800):
        base_x = int(rng.integers(0, width))
        base_y = int(rng.integers(horizon + 10, height))
        blade_len = int(rng.integers(6, 28) * (base_y / height))
        lean = int(rng.normal(0, 4))
        color = (
            int(rng.integers(28, 72)),
            int(rng.integers(108, 185)),
            int(rng.integers(38, 82)),
        )
        cv2.line(frame, (base_x, base_y), (base_x + lean, max(horizon, base_y - blade_len)), color, 1, cv2.LINE_AA)

    # Walking person with antialiasing on a high-res mask.
    person_layer = np.zeros_like(frame, dtype=np.float32)
    alpha = np.zeros((height, width), dtype=np.float32)
    ground_y = int(height * 0.82)
    cx = int(width * (0.18 + 0.64 * progress))
    walk = progress * math.tau * 2.0 + phase_offset
    bob = int(math.sin(walk) * height * 0.014)
    head = (cx, int(ground_y - height * 0.285 + bob))
    neck = (cx, int(ground_y - height * 0.218 + bob))
    hip = (cx, int(ground_y - height * 0.078 + bob))
    shoulder = (cx, int(ground_y - height * 0.197 + bob))
    leg = int(math.sin(walk) * width * 0.045)
    arm = int(-math.sin(walk) * width * 0.039)

    shadow = np.zeros_like(alpha)
    cv2.ellipse(shadow, (cx + int(width * 0.032), ground_y + int(height * 0.018)), (int(width * 0.074), int(height * 0.02)), -8, 0, 360, 0.45, -1, cv2.LINE_AA)
    frame = frame * (1 - shadow[:, :, None] * 0.42)

    dark = (24, 28, 34)
    cloth = (37, 57, 76)
    skin = (93, 63, 45)
    hair = (57, 34, 24)
    head_radius = int(height * 0.046)
    cv2.circle(alpha, head, head_radius, 1.0, -1, cv2.LINE_AA)
    cv2.circle(person_layer, head, head_radius, skin, -1, cv2.LINE_AA)
    cv2.circle(alpha, (head[0], head[1] - int(head_radius * 0.22)), int(head_radius * 1.02), 1.0, -1, cv2.LINE_AA)
    cv2.circle(person_layer, (head[0], head[1] - int(head_radius * 0.22)), int(head_radius * 1.02), hair, -1, cv2.LINE_AA)
    cv2.ellipse(alpha, (cx, int((neck[1] + hip[1]) / 2)), (int(width * 0.022), int(height * 0.077)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.ellipse(person_layer, (cx, int((neck[1] + hip[1]) / 2)), (int(width * 0.022), int(height * 0.077)), 0, 0, 360, cloth, -1, cv2.LINE_AA)
    cv2.line(alpha, shoulder, (cx + arm, int(ground_y - height * 0.115)), 1.0, int(width * 0.016), cv2.LINE_AA)
    cv2.line(person_layer, shoulder, (cx + arm, int(ground_y - height * 0.115)), dark, int(width * 0.016), cv2.LINE_AA)
    cv2.line(alpha, shoulder, (cx - arm, int(ground_y - height * 0.125)), 1.0, int(width * 0.016), cv2.LINE_AA)
    cv2.line(person_layer, shoulder, (cx - arm, int(ground_y - height * 0.125)), dark, int(width * 0.016), cv2.LINE_AA)
    cv2.line(alpha, hip, (cx + leg, ground_y), 1.0, int(width * 0.018), cv2.LINE_AA)
    cv2.line(person_layer, hip, (cx + leg, ground_y), dark, int(width * 0.018), cv2.LINE_AA)
    cv2.line(alpha, hip, (cx - leg, ground_y), 1.0, int(width * 0.018), cv2.LINE_AA)
    cv2.line(person_layer, hip, (cx - leg, ground_y), dark, int(width * 0.018), cv2.LINE_AA)
    frame = frame * (1 - alpha[:, :, None]) + person_layer * alpha[:, :, None]

    # Small cinematic contrast, still deterministic.
    vignette = 1.0 - 0.22 * np.clip(((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.5, 0, 1)
    frame *= vignette[:, :, None]
    return np.clip(frame, 0, 255).astype(np.uint8)


def render_full_power_frame_from_cells(
    cells: np.ndarray,
    *,
    feature_names: list[str] | tuple[str, ...],
    output_shape: tuple[int, int] = (540, 960),
    seed: int = 616,
) -> np.ndarray:
    """Render one frame using RGB, variance, texture, edge, saturation, and motion channels."""
    out_h, out_w = output_shape
    r = cells[:, :, _feature_index(feature_names, "rgb_mean_r")]
    g = cells[:, :, _feature_index(feature_names, "rgb_mean_g")]
    b = cells[:, :, _feature_index(feature_names, "rgb_mean_b")]
    base = np.dstack(
        [
            _smooth_repeat(r, output_shape),
            _smooth_repeat(g, output_shape),
            _smooth_repeat(b, output_shape),
        ]
    )

    std_rgb = np.dstack(
        [
            _smooth_repeat(cells[:, :, _feature_index(feature_names, "rgb_std_r")], output_shape),
            _smooth_repeat(cells[:, :, _feature_index(feature_names, "rgb_std_g")], output_shape),
            _smooth_repeat(cells[:, :, _feature_index(feature_names, "rgb_std_b")], output_shape),
        ]
    )
    luma_std = _smooth_repeat(cells[:, :, _feature_index(feature_names, "luma_std")], output_shape)
    texture = _smooth_repeat(cells[:, :, _feature_index(feature_names, "texture_energy")], output_shape)
    edge = _smooth_repeat(cells[:, :, _feature_index(feature_names, "edge_density")], output_shape)
    motion = _smooth_repeat(cells[:, :, _feature_index(feature_names, "motion_energy")], output_shape)
    delta = _smooth_repeat(cells[:, :, _feature_index(feature_names, "delta_luma_abs")], output_shape)
    sat = _smooth_repeat(cells[:, :, _feature_index(feature_names, "saturation_mean")], output_shape)

    noise_a = _stable_noise((out_h, out_w), seed)
    noise_b = _stable_noise((out_h, out_w), seed + 1)
    noise_c = _stable_noise((out_h, out_w), seed + 2)
    micro = np.dstack([noise_a, noise_b, noise_c])
    variation = std_rgb * 0.45 + luma_std[:, :, None] * 0.16 + texture[:, :, None] * 0.18
    frame = base + micro * variation

    luma = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.GaussianBlur(np.sqrt(grad_x * grad_x + grad_y * grad_y), (0, 0), 0.8)
    if float(grad_mag.max()) > 0:
        grad_mag /= float(grad_mag.max())
    edge_boost = edge * 120.0 * grad_mag
    frame += edge_boost[:, :, None] * np.array([0.85, 0.95, 1.0])

    motion_pressure = np.clip((motion + delta) / 95.0, 0, 1)
    shifted = np.roll(frame, shift=5, axis=1)
    frame = frame * (1 - motion_pressure[:, :, None] * 0.18) + shifted * (motion_pressure[:, :, None] * 0.18)

    hsv = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.72 + sat * 0.28
    frame = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
    smooth = cv2.bilateralFilter(frame_u8, 3, 18, 12)
    blur = cv2.GaussianBlur(smooth, (0, 0), 0.85)
    sharp = cv2.addWeighted(smooth, 1.35, blur, -0.35, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _hardware_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "compute_path": "CPU numpy/OpenCV high-detail source render, cell-state sampling, and full-power state replay",
        "gpu_acceleration_used": False,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        process = psutil.Process()
        snapshot.update(
            {
                "cpu_logical": psutil.cpu_count(logical=True),
                "cpu_physical": psutil.cpu_count(logical=False),
                "ram_total_bytes": int(vm.total),
                "ram_available_bytes": int(vm.available),
                "process_rss_bytes": int(process.memory_info().rss),
            }
        )
    except Exception as exc:  # pragma: no cover
        snapshot["psutil_error"] = str(exc)
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            gpu_data = json.loads(completed.stdout)
            snapshot["gpu_adapters_detected"] = gpu_data if isinstance(gpu_data, list) else [gpu_data]
    except Exception as exc:  # pragma: no cover
        snapshot["gpu_inventory_error"] = str(exc)
    return snapshot


def generate_full_power_frame(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
    frame_shape: tuple[int, int] = (540, 960),
    grid_shape: tuple[int, int] = (90, 160),
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    previous = build_detailed_person_field_frame(frame_shape=frame_shape, progress=0.535, phase_offset=-0.38)
    current = build_detailed_person_field_frame(frame_shape=frame_shape, progress=0.56, phase_offset=0.0)
    previous_state = build_video_cell_state(previous, grid_shape=grid_shape, previous_luma=None)
    state = build_video_cell_state(current, grid_shape=grid_shape, previous_luma=previous_state["luma"])
    cells = state["cells"].astype(np.float32)
    rendered = render_full_power_frame_from_cells(
        cells,
        feature_names=list(CELL_FEATURE_NAMES),
        output_shape=frame_shape,
        seed=616,
    )

    source_path = run_dir / f"{run_id}_source_reference.png"
    state_path = run_dir / f"{run_id}_state_full_power.png"
    npz_path = run_dir / f"{run_id}_cell_state.npz"
    manifest_path = run_dir / f"{run_id}_manifest.json"
    report_path = run_dir / f"{run_id}_report.md"

    cv2.imwrite(str(source_path), cv2.cvtColor(current, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(state_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    np.savez_compressed(
        npz_path,
        cell_state=cells,
        feature_names=np.asarray(CELL_FEATURE_NAMES),
        grid_shape=np.asarray(grid_shape, dtype=np.int32),
    )

    used_channels = [
        "rgb_mean_r/g/b",
        "rgb_std_r/g/b",
        "luma_std",
        "texture_energy",
        "edge_density",
        "motion_energy",
        "delta_luma_abs",
        "saturation_mean",
    ]
    timing = {
        "started_at_utc": utc_now(),
        "total_seconds": round(time.perf_counter() - started, 6),
    }
    manifest = {
        "schema_version": 1,
        "kind": "truevision_full_power_single_frame",
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "frame_shape": list(frame_shape),
        "grid_shape": list(grid_shape),
        "cell_shape": list(cells.shape),
        "feature_names": list(CELL_FEATURE_NAMES),
        "used_replay_channels": used_channels,
        "transition": {
            "previous_progress": 0.535,
            "current_progress": 0.56,
            "delta_luma_abs_used": True,
            "motion_energy_used": True,
        },
        "boundary": {
            "synthetic_state_media": True,
            "evidence": False,
            "prompt_generated": False,
            "raw_observation": False,
        },
        "hardware": _hardware_snapshot(),
        "timing": timing,
        "outputs": {
            "source_reference_png": str(source_path),
            "state_png": str(state_path),
            "cell_state_npz": str(npz_path),
            "report_md": str(report_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# TrueVision Full-Power Single Frame Report",
                "",
                "## Claim",
                "",
                "One clean frame was generated from a detailed scene state, sampled into TrueVision 16-channel cell vectors, then reconstructed using more than RGB means.",
                "",
                "## Channels Used",
                "",
                *[f"- `{channel}`" for channel in used_channels],
                "",
                "## Boundary",
                "",
                "```text",
                "Synthetic state media.",
                "Not evidence.",
                "Not prompt generation.",
                "State transition data is used through previous/current luma delta and motion channels.",
                "```",
                "",
                "## Outputs",
                "",
                f"- State frame: `{state_path}`",
                f"- Source reference: `{source_path}`",
                f"- Cell state: `{npz_path}`",
                f"- Manifest: `{manifest_path}`",
                "",
                "## Hardware",
                "",
                "```json",
                json.dumps(manifest["hardware"], indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "state_png": str(state_path),
        "source_reference_png": str(source_path),
        "cell_state_npz": str(npz_path),
        "manifest_json": str(manifest_path),
        "report_md": str(report_path),
        "cell_shape": list(cells.shape),
        "state_png_sha256": sha256_file(state_path),
        "source_reference_png_sha256": sha256_file(source_path),
        "cell_state_npz_sha256": sha256_file(npz_path),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one clean full-power TrueVision-shaped frame.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        json.dumps(
            generate_full_power_frame(output_root=Path(args.output_root), run_id=args.run_id),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
