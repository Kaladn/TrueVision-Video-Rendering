#!/usr/bin/env python3
"""Extract reusable video-render signatures from TrueVision capture state.

The extractor reads a completed signature capture and writes abstract profiles:
motion, camera shake, edge density, contrast/color, energy timing, and cut rhythm.
It does not read or emit raw frames.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROFILE_KIND = "truevision_signature_profile_bundle"
DEFAULT_CAPTURE_DIR = Path("storage/artifacts/signature_captures/default")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6),
        "mean": round(float(np.mean(array)), 6),
        "std": round(float(np.std(array)), 6),
        "p05": round(float(np.percentile(array, 5)), 6),
        "p50": round(float(np.percentile(array, 50)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
    }


def _norm(value: float, stat: dict[str, Any]) -> float:
    low = stat.get("p05")
    high = stat.get("p95")
    if low is None or high is None or high <= low:
        return 0.0
    return round(max(0.0, min(1.0, (value - float(low)) / (float(high) - float(low)))), 6)


def _sample_down(samples: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples < 1 or len(samples) <= max_samples:
        return samples
    step = len(samples) / max_samples
    return [samples[min(len(samples) - 1, int(round(i * step)))] for i in range(max_samples)]


def _find_manifest(capture_dir: Path) -> dict[str, Any]:
    manifests = sorted(capture_dir.glob("*_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no *_manifest.json found under {capture_dir}")
    return json.loads(manifests[0].read_text(encoding="utf-8"))


def _read_records(capture_dir: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    record_path = Path(manifest.get("records", {}).get("jsonl_path") or "")
    if not record_path.exists():
        matches = sorted(capture_dir.glob("*_records.jsonl"))
        if not matches:
            return [], {}
        record_path = matches[0]
    records: list[dict[str, Any]] = []
    by_frame: dict[int, dict[str, Any]] = {}
    with record_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            by_frame[int(record.get("frame_number", len(records)))] = record
    return records, by_frame


def _feature_index(feature_names: list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(feature_names)}


def _feature(cells: np.ndarray, indices: dict[str, int], name: str, default: float = 0.0) -> np.ndarray:
    index = indices.get(name)
    if index is None or index >= cells.shape[-1]:
        return np.full(cells.shape[:2], default, dtype=np.float32)
    return cells[:, :, index].astype(np.float32)


def _weighted_centroid(weights: np.ndarray) -> tuple[float, float]:
    rows, cols = weights.shape
    total = float(np.sum(weights))
    if total <= 1e-9:
        return 0.0, 0.0
    xs = np.linspace(-1.0, 1.0, cols, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
    cx = float(np.sum(weights * xs[np.newaxis, :]) / total)
    cy = float(np.sum(weights * ys[:, np.newaxis]) / total)
    return cx, cy


def _record_value(record: dict[str, Any], key: str) -> float:
    resonance = record.get("visual_resonance") or {}
    if key in record:
        return _safe_float(record.get(key))
    return _safe_float(resonance.get(key))


def _extract_cell_samples(
    *,
    capture_dir: Path,
    manifest: dict[str, Any],
    records_by_frame: dict[int, dict[str, Any]],
    sample_stride: int,
) -> list[dict[str, Any]]:
    cell_dir = capture_dir / "cell_state_npz"
    chunks = sorted(cell_dir.glob("*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no cell_state_npz/*.npz files found under {capture_dir}")

    fps = _safe_float(manifest.get("config", {}).get("capture_fps"), 9.0) or 9.0
    samples: list[dict[str, Any]] = []
    previous_centroid: tuple[float, float] | None = None
    for chunk_path in chunks:
        with np.load(chunk_path, allow_pickle=False) as npz:
            cell_state = np.asarray(npz["cell_state"], dtype=np.float32)
            frame_numbers = np.asarray(npz["frame_numbers"], dtype=np.int32)
            feature_names = [str(name) for name in npz["feature_names"].tolist()]
        indices = _feature_index(feature_names)
        for local_index in range(0, cell_state.shape[0], max(1, sample_stride)):
            frame_number = int(frame_numbers[local_index])
            record = records_by_frame.get(frame_number, {})
            cells = cell_state[local_index]
            delta = _feature(cells, indices, "delta_luma_abs")
            motion = _feature(cells, indices, "motion_energy")
            motion_weights = np.maximum(delta, 0.0) + np.maximum(motion, 0.0)
            centroid_x, centroid_y = _weighted_centroid(motion_weights)
            if previous_centroid is None:
                shake_x = 0.0
                shake_y = 0.0
            else:
                shake_x = centroid_x - previous_centroid[0]
                shake_y = centroid_y - previous_centroid[1]
            previous_centroid = (centroid_x, centroid_y)
            shake_magnitude = math.sqrt(shake_x * shake_x + shake_y * shake_y)
            time_seconds = _safe_float(record.get("elapsed_seconds"), (frame_number - 1) / fps)
            sample = {
                "frame_number": frame_number,
                "time_seconds": round(time_seconds, 6),
                "motion_mean": round(float(np.mean(motion)), 6),
                "motion_p95": round(float(np.percentile(motion, 95)), 6),
                "delta_luma_mean": round(float(np.mean(delta)), 6),
                "edge_mean": round(float(np.mean(_feature(cells, indices, "edge_density"))), 6),
                "texture_mean": round(float(np.mean(_feature(cells, indices, "texture_energy"))), 6),
                "luma_mean": round(float(np.mean(_feature(cells, indices, "luma_mean"))), 6),
                "luma_std": round(float(np.mean(_feature(cells, indices, "luma_std"))), 6),
                "saturation_mean": round(float(np.mean(_feature(cells, indices, "saturation_mean"))), 6),
                "hue_mean": round(float(np.mean(_feature(cells, indices, "hsv_mean_h"))), 6),
                "rgb_mean": [
                    round(float(np.mean(_feature(cells, indices, "rgb_mean_r"))), 6),
                    round(float(np.mean(_feature(cells, indices, "rgb_mean_g"))), 6),
                    round(float(np.mean(_feature(cells, indices, "rgb_mean_b"))), 6),
                ],
                "centroid_x": round(centroid_x, 6),
                "centroid_y": round(centroid_y, 6),
                "shake_x": round(shake_x, 6),
                "shake_y": round(shake_y, 6),
                "shake_magnitude": round(shake_magnitude, 6),
                "screen_energy": round(_record_value(record, "screen_energy"), 6),
                "flash": round(_record_value(record, "vis_flash_intensity"), 6),
                "contrast_shift": round(_record_value(record, "vis_contrast_shift_score"), 6),
                "jitter": round(_record_value(record, "vis_jitter_band_energy"), 6),
                "smoothness": round(_record_value(record, "vis_smoothness_index"), 6),
            }
            samples.append(sample)
    return samples


def _profile_stats(samples: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, Any]]:
    return {key: _stats([_safe_float(sample.get(key)) for sample in samples]) for key in keys}


def _events(samples: list[dict[str, Any]], key: str, stat: dict[str, Any], *, event_name: str) -> list[dict[str, Any]]:
    threshold = stat.get("p95")
    if threshold is None:
        return []
    events = []
    last_time = -999.0
    for sample in samples:
        time_seconds = _safe_float(sample.get("time_seconds"))
        if _safe_float(sample.get(key)) >= float(threshold) and time_seconds - last_time > 0.5:
            events.append(
                {
                    "event": event_name,
                    "time_seconds": round(time_seconds, 6),
                    "frame_number": int(sample.get("frame_number", 0)),
                    "value": round(_safe_float(sample.get(key)), 6),
                }
            )
            last_time = time_seconds
    return events[:500]


def _build_timeline(samples: list[dict[str, Any]], stats: dict[str, dict[str, Any]], max_timeline_samples: int) -> list[dict[str, Any]]:
    duration = max((_safe_float(sample.get("time_seconds")) for sample in samples), default=0.0)
    timeline = []
    for sample in samples:
        shake_x = max(-1.0, min(1.0, _safe_float(sample.get("shake_x")) * 3.0))
        shake_y = max(-1.0, min(1.0, _safe_float(sample.get("shake_y")) * 3.0))
        timeline.append(
            {
                "time_seconds": round(_safe_float(sample.get("time_seconds")), 6),
                "time_norm": round(_safe_float(sample.get("time_seconds")) / duration, 6) if duration > 0 else 0.0,
                "motion": _norm(_safe_float(sample.get("motion_mean")) + _safe_float(sample.get("delta_luma_mean")), stats["motion_combo"]),
                "edge": _norm(_safe_float(sample.get("edge_mean")), stats["edge_mean"]),
                "contrast": _norm(_safe_float(sample.get("luma_std")) + _safe_float(sample.get("contrast_shift")), stats["contrast_combo"]),
                "saturation": _norm(_safe_float(sample.get("saturation_mean")), stats["saturation_mean"]),
                "flash": max(0.0, min(1.0, _safe_float(sample.get("flash")))),
                "shake_x": round(shake_x, 6),
                "shake_y": round(shake_y, 6),
            }
        )
    return _sample_down(timeline, max_timeline_samples)


def extract_signature_profiles(
    *,
    capture_dir: Path,
    output_dir: Path | None = None,
    profile_id: str | None = None,
    sample_stride: int = 1,
    max_timeline_samples: int = 1800,
) -> dict[str, str]:
    capture_dir = capture_dir.resolve()
    manifest = _find_manifest(capture_dir)
    profile_id = profile_id or f"{manifest.get('run_id', capture_dir.name)}_signature_profile"
    output_dir = (output_dir or capture_dir / "signature_profiles" / profile_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, records_by_frame = _read_records(capture_dir, manifest)
    samples = _extract_cell_samples(
        capture_dir=capture_dir,
        manifest=manifest,
        records_by_frame=records_by_frame,
        sample_stride=sample_stride,
    )
    if not samples:
        raise ValueError("capture produced no signature samples")

    combo_samples = [
        {
            **sample,
            "motion_combo": _safe_float(sample.get("motion_mean")) + _safe_float(sample.get("delta_luma_mean")),
            "contrast_combo": _safe_float(sample.get("luma_std")) + _safe_float(sample.get("contrast_shift")),
        }
        for sample in samples
    ]
    common_stats = _profile_stats(
        combo_samples,
        [
            "motion_mean",
            "motion_p95",
            "delta_luma_mean",
            "motion_combo",
            "edge_mean",
            "texture_mean",
            "luma_mean",
            "luma_std",
            "contrast_combo",
            "saturation_mean",
            "hue_mean",
            "shake_x",
            "shake_y",
            "shake_magnitude",
            "screen_energy",
            "flash",
            "contrast_shift",
            "jitter",
            "smoothness",
        ],
    )
    duration = max((_safe_float(sample["time_seconds"]) for sample in samples), default=0.0)

    motion_profile = {
        "schema_version": 1,
        "kind": "motion_profile",
        "profile_id": profile_id,
        "source_capture": str(capture_dir),
        "stats": {key: common_stats[key] for key in ["motion_mean", "motion_p95", "delta_luma_mean", "motion_combo"]},
        "motion_events": _events(combo_samples, "motion_combo", common_stats["motion_combo"], event_name="motion_peak"),
        "samples": _sample_down([{k: sample[k] for k in ["time_seconds", "motion_mean", "motion_p95", "delta_luma_mean"]} for sample in samples], 1200),
    }
    camera_profile = {
        "schema_version": 1,
        "kind": "camera_shake_profile",
        "profile_id": profile_id,
        "source_capture": str(capture_dir),
        "stats": {key: common_stats[key] for key in ["centroid_x", "centroid_y", "shake_x", "shake_y", "shake_magnitude"] if key in common_stats},
        "shake_events": _events(samples, "shake_magnitude", common_stats["shake_magnitude"], event_name="shake_peak"),
        "samples": _sample_down(
            [{k: sample[k] for k in ["time_seconds", "centroid_x", "centroid_y", "shake_x", "shake_y", "shake_magnitude"]} for sample in samples],
            1200,
        ),
    }
    edge_profile = {
        "schema_version": 1,
        "kind": "edge_density_profile",
        "profile_id": profile_id,
        "source_capture": str(capture_dir),
        "stats": {key: common_stats[key] for key in ["edge_mean", "texture_mean"]},
        "edge_events": _events(samples, "edge_mean", common_stats["edge_mean"], event_name="edge_peak"),
        "samples": _sample_down([{k: sample[k] for k in ["time_seconds", "edge_mean", "texture_mean"]} for sample in samples], 1200),
    }
    color_profile = {
        "schema_version": 1,
        "kind": "contrast_color_profile",
        "profile_id": profile_id,
        "source_capture": str(capture_dir),
        "stats": {key: common_stats[key] for key in ["luma_mean", "luma_std", "contrast_combo", "saturation_mean", "hue_mean"]},
        "samples": _sample_down(
            [{k: sample[k] for k in ["time_seconds", "luma_mean", "luma_std", "saturation_mean", "hue_mean", "rgb_mean"]} for sample in samples],
            1200,
        ),
    }
    energy_profile = {
        "schema_version": 1,
        "kind": "energy_timing_profile",
        "profile_id": profile_id,
        "source_capture": str(capture_dir),
        "stats": {key: common_stats[key] for key in ["screen_energy", "flash", "contrast_shift", "jitter", "smoothness"]},
        "energy_events": _events(samples, "screen_energy", common_stats["screen_energy"], event_name="energy_peak"),
        "samples": _sample_down(
            [{k: sample[k] for k in ["time_seconds", "screen_energy", "flash", "contrast_shift", "jitter", "smoothness"]} for sample in samples],
            1200,
        ),
    }
    cut_events = []
    for event in _events(combo_samples, "motion_combo", common_stats["motion_combo"], event_name="motion_cut_candidate"):
        cut_events.append(event)
    for event in _events(samples, "contrast_shift", common_stats["contrast_shift"], event_name="contrast_cut_candidate"):
        cut_events.append(event)
    cut_events = sorted(cut_events, key=lambda event: event["time_seconds"])[:800]
    cut_profile = {
        "schema_version": 1,
        "kind": "cut_rhythm_profile",
        "profile_id": profile_id,
        "source_capture": str(capture_dir),
        "cut_events": cut_events,
        "cut_count": len(cut_events),
        "mean_seconds_between_cuts": round(duration / len(cut_events), 6) if cut_events else None,
    }
    timeline_samples = _build_timeline(combo_samples, common_stats, max_timeline_samples)

    paths = {
        "motion_profile_json": output_dir / "motion_profile.json",
        "camera_shake_profile_json": output_dir / "camera_shake_profile.json",
        "edge_density_profile_json": output_dir / "edge_density_profile.json",
        "contrast_color_profile_json": output_dir / "contrast_color_profile.json",
        "energy_timing_profile_json": output_dir / "energy_timing_profile.json",
        "cut_rhythm_profile_json": output_dir / "cut_rhythm_profile.json",
    }
    profile_payloads = {
        "motion_profile_json": motion_profile,
        "camera_shake_profile_json": camera_profile,
        "edge_density_profile_json": edge_profile,
        "contrast_color_profile_json": color_profile,
        "energy_timing_profile_json": energy_profile,
        "cut_rhythm_profile_json": cut_profile,
    }
    for key, payload in profile_payloads.items():
        _write_json(paths[key], payload)

    bundle = {
        "schema_version": 1,
        "kind": PROFILE_KIND,
        "profile_id": profile_id,
        "created_at_utc": utc_now(),
        "source": {
            "capture_dir": str(capture_dir),
            "run_id": manifest.get("run_id"),
            "frame_count": len(samples),
            "record_count": len(records),
            "duration_seconds": round(duration, 6),
            "raw_frame_saved": False,
            "signature_only": True,
        },
        "profiles": {
            "motion_profile": str(paths["motion_profile_json"]),
            "camera_shake_profile": str(paths["camera_shake_profile_json"]),
            "edge_density_profile": str(paths["edge_density_profile_json"]),
            "contrast_color_profile": str(paths["contrast_color_profile_json"]),
            "energy_timing_profile": str(paths["energy_timing_profile_json"]),
            "cut_rhythm_profile": str(paths["cut_rhythm_profile_json"]),
        },
        "profile_summaries": {
            "motion": motion_profile["stats"],
            "camera": camera_profile["stats"],
            "edge": edge_profile["stats"],
            "color": color_profile["stats"],
            "energy": energy_profile["stats"],
            "cuts": {"cut_count": cut_profile["cut_count"], "mean_seconds_between_cuts": cut_profile["mean_seconds_between_cuts"]},
        },
        "timeline_samples": timeline_samples,
        "renderer_guidance": {
            "use_as": "motion_look_signature_not_source_video",
            "allowed": ["camera shake", "motion pressure", "edge shimmer", "contrast/color pressure", "cut rhythm"],
            "forbidden": ["copy source assets", "claim evidence", "reconstruct original footage"],
        },
    }
    bundle_path = output_dir / "signature_profile_bundle.json"
    _write_json(bundle_path, bundle)

    report_path = output_dir / "SIGNATURE_PROFILE_REPORT.md"
    report_path.write_text(
        "\n".join(
            [
                "# TrueVision Signature Profile Report",
                "",
                f"- Profile: `{profile_id}`",
                f"- Source capture: `{capture_dir}`",
                f"- Samples: `{len(samples)}`",
                f"- Duration: `{duration:.2f}s`",
                f"- Cut candidates: `{cut_profile['cut_count']}`",
                "",
                "These profiles are abstract motion/look signatures only. They are not raw footage and not evidence.",
                "",
                f"- Bundle: `{bundle_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "profile_id": profile_id,
        "output_dir": str(output_dir),
        "bundle_json": str(bundle_path),
        "report_md": str(report_path),
        **{key: str(path) for key, path in paths.items()},
        "samples": str(len(samples)),
        "duration_seconds": str(round(duration, 6)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract renderer signatures from a TrueVision capture directory.")
    parser.add_argument("--capture-dir", default=str(DEFAULT_CAPTURE_DIR))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--profile-id", default="")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-timeline-samples", type=int, default=1800)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = extract_signature_profiles(
        capture_dir=Path(args.capture_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        profile_id=args.profile_id or None,
        sample_stride=args.sample_stride,
        max_timeline_samples=args.max_timeline_samples,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
