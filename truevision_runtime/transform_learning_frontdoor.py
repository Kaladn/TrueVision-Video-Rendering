from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now


PROFILE_SCHEMA = "truevision_transform_behavior_profile_v1"
COMPARISON_SCHEMA = "truevision_generated_transform_comparison_v1"
CYCLE_SCHEMA = "truevision_transform_learning_cycle_v1"
RECEIPT_SCHEMA = "truevision_transform_learning_frontdoor_receipt_v1"


TRANSFORM_METRICS: dict[str, list[str]] = {
    "lightning": [
        "luma_delta",
        "rise_time_frames",
        "falloff_frames",
        "bloom_radius_cells",
        "surrounding_exposure_lift",
        "branch_edge_density",
        "branch_direction_variance",
        "afterglow_decay_rate",
    ],
    "fog_reveal": [
        "density_gradient",
        "edge_recovery_rate",
        "contrast_recovery_rate",
        "near_reveal_speed",
        "far_occlusion_pressure",
        "motion_parallax",
        "light_scatter",
    ],
    "water_shimmer": [
        "horizontal_motion_consistency",
        "specular_flicker",
        "wave_band_repetition",
        "texture_energy",
        "edge_softness",
    ],
    "sun_sky_breath": [
        "luma_pressure",
        "warm_hue_shift",
        "saturation_pressure",
        "edge_bloom",
        "shadow_lift",
    ],
}


LOWER_IS_BETTER_WHEN_TOO_HIGH = {"rise_time_frames", "falloff_frames"}


def _safe_id(value: str | None, fallback: str = "transform_learning") -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "")).strip("_")
    return safe or fallback


def _event_metrics(event: dict[str, Any]) -> dict[str, float]:
    metrics = event.get("true_local_metrics") or event.get("metrics") or event.get("meter_peaks") or {}
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            clean[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return clean


def _metric_names(transform_kind: str, events: Iterable[dict[str, Any]]) -> list[str]:
    configured = TRANSFORM_METRICS.get(transform_kind)
    if configured:
        return configured
    names: set[str] = set()
    for event in events:
        names.update(_event_metrics(event).keys())
    return sorted(names)


def _summarize_metric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"target": 0.0, "min": 0.0, "max": 0.0, "sample_count": 0.0}
    return {
        "target": round(mean(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "sample_count": float(len(values)),
    }


def _source_refs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for event in events:
        refs.append(
            {
                "event_id": str(event.get("event_id") or event.get("id") or ""),
                "candidate_type": str(event.get("candidate_type") or event.get("event_type_candidate") or "candidate_transform_event"),
                "source_region": event.get("source_region") or event.get("time_range") or {},
            }
        )
    return refs


def build_transform_behavior_profile(
    observed_events: list[dict[str, Any]],
    *,
    transform_kind: str,
    profile_id: str | None = None,
) -> dict[str, Any]:
    metric_names = _metric_names(transform_kind, observed_events)
    behavior_metrics: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [_event_metrics(event)[name] for event in observed_events if name in _event_metrics(event)]
        behavior_metrics[name] = _summarize_metric(values)

    profile = {
        "schema_version": PROFILE_SCHEMA,
        "created_at_utc": utc_now(),
        "profile_id": _safe_id(profile_id, f"{transform_kind}_behavior_profile"),
        "transform_kind": transform_kind,
        "source_event_count": len(observed_events),
        "source_refs": _source_refs(observed_events),
        "behavior_metrics": behavior_metrics,
        "transform_law": {
            "source_state_drives_profile": True,
            "learn_behavior_not_shape": True,
            "new_generation_must_use_new_geometry": True,
            "raw_local_metrics_are_evidence": True,
            "filtered_metrics_are_interpretation": True,
        },
        "boundary": {
            "copy_behavior_not_pixels": True,
            "source_shape_copy_allowed": False,
            "source_geometry_retained": False,
            "yolo_truth_authority": False,
            "truevision_state_required": True,
            "generated_media_is_evidence": False,
            "six_one_six_mapping_enabled": False,
        },
    }
    profile["behavior_signature_sha256"] = stable_hash(
        {
            "schema_version": PROFILE_SCHEMA,
            "transform_kind": transform_kind,
            "behavior_metrics": behavior_metrics,
            "source_refs": profile["source_refs"],
        }
    )
    return profile


def _relative_error(target: float, actual: float) -> float:
    if abs(target) <= 1.0e-9:
        return abs(actual)
    return abs(actual - target) / abs(target)


def _adjustment(metric: str, target: float, actual: float) -> dict[str, Any] | None:
    delta = target - actual
    if abs(delta) <= 1.0e-9:
        return None
    direction = "increase" if delta > 0 else "decrease"
    if metric in LOWER_IS_BETTER_WHEN_TOO_HIGH and actual > target:
        direction = "decrease"
    elif metric in LOWER_IS_BETTER_WHEN_TOO_HIGH and actual < target:
        direction = "increase"
    return {
        "adjustment_id": f"{direction}_{metric}",
        "metric": metric,
        "direction": direction,
        "target": round(target, 6),
        "actual": round(actual, 6),
        "delta": round(delta, 6),
    }


def compare_generated_transform_to_profile(
    profile: dict[str, Any],
    generated_attempt: dict[str, Any],
    *,
    tolerance: float = 0.12,
) -> dict[str, Any]:
    generated_metrics = _event_metrics(generated_attempt)
    deltas: dict[str, dict[str, float | bool]] = {}
    adjustments: list[dict[str, Any]] = []
    errors: list[float] = []
    for metric, summary in profile.get("behavior_metrics", {}).items():
        target = float(summary.get("target") or 0.0)
        actual = float(generated_metrics.get(metric, 0.0))
        rel = _relative_error(target, actual)
        errors.append(rel)
        deltas[metric] = {
            "target": round(target, 6),
            "actual": round(actual, 6),
            "relative_error": round(rel, 6),
            "within_tolerance": rel <= tolerance,
        }
        if rel > tolerance:
            item = _adjustment(metric, target, actual)
            if item:
                adjustments.append(item)

    mean_error = round(mean(errors), 6) if errors else 0.0
    max_error = round(max(errors), 6) if errors else 0.0
    accepted = bool(errors) and mean_error <= tolerance and max_error <= tolerance * 2.0
    comparison = {
        "schema_version": COMPARISON_SCHEMA,
        "created_at_utc": utc_now(),
        "profile_id": profile.get("profile_id"),
        "transform_kind": profile.get("transform_kind"),
        "attempt_id": generated_attempt.get("attempt_id") or generated_attempt.get("id") or "generated_attempt",
        "score": {
            "tolerance": tolerance,
            "mean_relative_error": mean_error,
            "max_relative_error": max_error,
        },
        "accepted": accepted,
        "metric_deltas": deltas,
        "adjustments": adjustments,
        "boundary": {
            "generated_shape_may_differ": True,
            "source_shape_copy_allowed": False,
            "behavior_match_required": True,
            "raw_state_profile_is_reference": True,
            "generated_media_is_evidence": False,
            "six_one_six_mapping_enabled": False,
        },
    }
    comparison["comparison_sha256"] = stable_hash({key: value for key, value in comparison.items() if key != "comparison_sha256"})
    return comparison


def run_transform_learning_cycle(
    observed_events: list[dict[str, Any]],
    generated_attempts: list[dict[str, Any]],
    *,
    transform_kind: str,
    tolerance: float = 0.12,
    profile_id: str | None = None,
) -> dict[str, Any]:
    profile = build_transform_behavior_profile(observed_events, transform_kind=transform_kind, profile_id=profile_id)
    comparisons = [
        compare_generated_transform_to_profile(profile, attempt, tolerance=tolerance)
        for attempt in generated_attempts
    ]
    best = min(comparisons, key=lambda item: item["score"]["mean_relative_error"], default=None)
    cycle = {
        "schema_version": CYCLE_SCHEMA,
        "created_at_utc": utc_now(),
        "transform_kind": transform_kind,
        "profile": profile,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "accepted": bool(best and best.get("accepted")),
        "best_attempt_id": best.get("attempt_id") if best else "",
        "next_adjustments": [] if best and best.get("accepted") else (best.get("adjustments", []) if best else []),
        "frontdoor_contract": {
            "recognizers_propose_regions": True,
            "truevision_state_proves_metrics": True,
            "renderer_consumes_behavior_profile": True,
            "learned_transform_is_not_source_copy": True,
        },
        "boundary": {
            "capture_started": False,
            "render_started": False,
            "model_training_started": False,
            "source_shape_copy_allowed": False,
            "generated_media_is_evidence": False,
            "six_one_six_mapping_enabled": False,
        },
    }
    cycle["cycle_sha256"] = stable_hash({key: value for key, value in cycle.items() if key != "cycle_sha256"})
    return cycle


def write_transform_learning_cycle(
    *,
    cycle: dict[str, Any],
    output_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    safe_run = _safe_id(run_id)
    manifest_path = root / f"{safe_run}_transform_learning_cycle.json"
    manifest_path.write_text(json.dumps(cycle, indent=2, allow_nan=False), encoding="utf-8")
    receipt_dir = root.parent.parent / "receipts" / "transform_learning_frontdoor" if root.parts[-2:] == ("manifests", "transform_learning_frontdoor") else root / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "created_at_utc": utc_now(),
        "run_id": safe_run,
        "status": "completed",
        "manifest_json": str(manifest_path),
        "cycle_sha256": cycle.get("cycle_sha256"),
        "accepted": cycle.get("accepted", False),
        "boundary": cycle.get("boundary", {}),
    }
    receipt["receipt_sha256"] = stable_hash({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    receipt_path = receipt_dir / f"{safe_run}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "accepted": cycle.get("accepted", False),
        "best_attempt_id": cycle.get("best_attempt_id", ""),
        "status": "completed",
    }
