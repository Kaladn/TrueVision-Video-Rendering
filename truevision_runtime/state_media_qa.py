from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DEAD_MEMORY_VICE_PLAN: tuple[dict[str, Any], ...] = (
    {
        "stage": "whisper_dead_memory",
        "phase_start": 0.000,
        "phase_end": 0.105,
        "planned_beat": "almost black, low fog, weak red pulse, distant female presence",
    },
    {
        "stage": "room_wakes_bitterness",
        "phase_start": 0.105,
        "phase_end": 0.295,
        "planned_beat": "industrial chamber reveals itself; memory core begins to flicker",
    },
    {
        "stage": "vice_reveal",
        "phase_start": 0.295,
        "phase_end": 0.395,
        "planned_beat": "black iron vice jaws enter authority and begin reading as pressure",
    },
    {
        "stage": "pressure_drop_truth_cuts",
        "phase_start": 0.395,
        "phase_end": 0.585,
        "planned_beat": "vice pressure rises; white truth cuts and rain-glass distortion sharpen",
    },
    {
        "stage": "fall_from_grace",
        "phase_start": 0.585,
        "phase_end": 0.720,
        "planned_beat": "broken photos, empty silhouettes, and memory debris drift through the chamber",
    },
    {
        "stage": "collision_core",
        "phase_start": 0.720,
        "phase_end": 0.895,
        "planned_beat": "maximum pressure: fog, ash, lightning bloom, and cracked memory core collide",
    },
    {
        "stage": "final_chorus_peak",
        "phase_start": 0.895,
        "phase_end": 0.962,
        "planned_beat": "red-black ash storm and white edge bloom peak without literal gore",
    },
    {
        "stage": "outro_release",
        "phase_start": 0.962,
        "phase_end": 1.000,
        "planned_beat": "vice pressure recedes; thin gold-white survival fracture remains",
    },
)

DAUGHTER_STAR_LOCKET_PLAN: tuple[dict[str, Any], ...] = (
    {
        "stage": "dark_water_waiting",
        "phase_start": 0.000,
        "phase_end": 0.115,
        "planned_beat": "black water, distant horizon, faint daughter-star presence",
    },
    {
        "stage": "first_memory_light",
        "phase_start": 0.115,
        "phase_end": 0.250,
        "planned_beat": "star glow reveals itself and the heart locket emerges through fog",
    },
    {
        "stage": "distance_opens",
        "phase_start": 0.250,
        "phase_end": 0.380,
        "planned_beat": "water distance stretches between star and cracked heart",
    },
    {
        "stage": "heart_fracture",
        "phase_start": 0.380,
        "phase_end": 0.520,
        "planned_beat": "heart crack pulses with vocal pain and bass weight",
    },
    {
        "stage": "what_did_they_take",
        "phase_start": 0.520,
        "phase_end": 0.665,
        "planned_beat": "chain trembles and reflected memory distorts across tear ripples",
    },
    {
        "stage": "father_reaches",
        "phase_start": 0.665,
        "phase_end": 0.800,
        "planned_beat": "soft light bridge reaches from cracked heart toward daughter-star",
    },
    {
        "stage": "daughter_star_answers",
        "phase_start": 0.800,
        "phase_end": 0.935,
        "planned_beat": "daughter-star blooms with sacred restraint, not spectacle",
    },
    {
        "stage": "hope_holds",
        "phase_start": 0.935,
        "phase_end": 1.000,
        "planned_beat": "crack remains but gold-white hope fills it and the water calms",
    },
)

EDGE_NIGHTMARE_WORLD_PLAN: tuple[dict[str, Any], ...] = (
    {
        "stage": "black_edge_wake",
        "phase_start": 0.000,
        "phase_end": 0.110,
        "planned_beat": "black cliff-rim world wakes with distant abyss pressure and first storm light",
    },
    {
        "stage": "walk_to_rim",
        "phase_start": 0.110,
        "phase_end": 0.245,
        "planned_beat": "human silhouette moves toward the rim while fog and river pressure begin rising",
    },
    {
        "stage": "side_parallax_pressure",
        "phase_start": 0.245,
        "phase_end": 0.370,
        "planned_beat": "camera shifts sideways across the edge; silhouettes and cliff state carry depth",
    },
    {
        "stage": "just_looking_down",
        "phase_start": 0.370,
        "phase_end": 0.505,
        "planned_beat": "top-down look over the edge reveals the color river pulsing below",
    },
    {
        "stage": "falling_camera_spiral",
        "phase_start": 0.505,
        "phase_end": 0.650,
        "planned_beat": "camera falls and rolls through abyss state without losing deterministic structure",
    },
    {
        "stage": "river_below_answers",
        "phase_start": 0.650,
        "phase_end": 0.800,
        "planned_beat": "the river below becomes an answering energy field; storm and fog remain controlled",
    },
    {
        "stage": "storm_power_walk",
        "phase_start": 0.800,
        "phase_end": 0.925,
        "planned_beat": "silhouette returns with power as lightning and rim pressure peak",
    },
    {
        "stage": "gold_edge_release",
        "phase_start": 0.925,
        "phase_end": 1.000,
        "planned_beat": "nightmare recedes into a gold-white edge release instead of a collapse",
    },
)


SCENE_PLANS: dict[str, tuple[dict[str, Any], ...]] = {
    "dead_memory_vice_chamber": DEAD_MEMORY_VICE_PLAN,
    "vice_chamber": DEAD_MEMORY_VICE_PLAN,
    "daughter_star_locket_sea": DAUGHTER_STAR_LOCKET_PLAN,
    "edge_nightmare_world": EDGE_NIGHTMARE_WORLD_PLAN,
    "edge_nightmare": EDGE_NIGHTMARE_WORLD_PLAN,
}


def build_state_media_qa_receipt(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    render = manifest.get("render") or {}
    scene_state = manifest.get("scene_state") or {}
    scene_mode = str(scene_state.get("scene_mode") or "")
    duration = float(render.get("duration_seconds") or 0.0)
    fps = float(render.get("fps") or 0.0)
    state_log_every = int(render.get("state_log_every") or 1)
    state_path = Path(str(render.get("frame_state_jsonl") or ""))
    rows = _read_state_rows(state_path)
    plan = SCENE_PLANS.get(scene_mode, ())
    stage_results = {
        item["stage"]: _stage_result(
            item,
            [row for row in rows if row.get("stage") == item["stage"]],
            duration_seconds=duration,
            tolerance_seconds=_timing_tolerance_seconds(fps, state_log_every),
        )
        for item in plan
    }
    observed_stage_names = sorted({str(row.get("stage") or "") for row in rows if row.get("stage")})
    planned_stage_names = [item["stage"] for item in plan]
    missing_stages = [stage for stage in planned_stage_names if not stage_results[stage]["observed"]]
    timing_failures = [
        stage for stage, result in stage_results.items() if result["timing_status"] != "pass"
    ]
    structural_pass = bool(plan) and not missing_stages and not timing_failures
    receipt = {
        "schema_version": "truevision_state_media_qa_receipt_v1",
        "created_at_utc": _utc_now(),
        "run_id": manifest.get("run_id"),
        "scene_mode": scene_mode,
        "source_manifest_json": str(manifest_path),
        "frame_state_jsonl": str(state_path),
        "planned_stage_count": len(planned_stage_names),
        "observed_stage_count": len([stage for stage in planned_stage_names if stage in observed_stage_names]),
        "state_sample_count": len(rows),
        "duration_seconds": duration,
        "fps": fps,
        "state_log_every": state_log_every,
        "timing_tolerance_seconds": _timing_tolerance_seconds(fps, state_log_every),
        "missing_stages": missing_stages,
        "timing_failures": timing_failures,
        "structural_pass": structural_pass,
        "artistic_depth_pass": "manual_review_required",
        "profile_calibrated": False,
        "qa_pass": structural_pass,
        "planned_storyboard": [
            {
                "stage": item["stage"],
                "expected_time_start": round(item["phase_start"] * duration, 6),
                "expected_time_end": round(item["phase_end"] * duration, 6),
                "planned_beat": item["planned_beat"],
            }
            for item in plan
        ],
        "stage_results": stage_results,
        "receipt_notes": [
            "Compares intended storyboard states against frame-state JSONL samples.",
            "This receipt checks state arc coverage and timing; it does not judge artistic taste.",
            "Internal render mechanisms remain outside this public QA surface.",
        ],
    }
    receipt["receipt_sha256"] = _stable_hash({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    return receipt


def write_state_media_qa_receipt(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    receipt = build_state_media_qa_receipt(manifest_path)
    root = Path(output_dir) if output_dir else manifest_path.parent
    root.mkdir(parents=True, exist_ok=True)
    run_id = _safe_id(str(receipt.get("run_id") or manifest_path.stem))
    receipt_path = root / f"{run_id}_state_media_qa_receipt.json"
    report_path = root / f"{run_id}_state_media_qa_report.md"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(render_state_media_qa_markdown(receipt), encoding="utf-8")
    return {
        "receipt_json": str(receipt_path),
        "report_md": str(report_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "qa_pass": receipt["qa_pass"],
    }


def render_state_media_qa_markdown(receipt: dict[str, Any]) -> str:
    lines = [
        f"# State Media QA Report: {receipt.get('run_id')}",
        "",
        "Purpose:",
        "",
        "```text",
        "Compare intended storyboard beats against actual logged render state.",
        "```",
        "",
        "Summary:",
        "",
        "```text",
        f"scene_mode: {receipt.get('scene_mode')}",
        f"qa_pass: {receipt.get('qa_pass')}",
        f"duration_seconds: {receipt.get('duration_seconds')}",
        f"state_sample_count: {receipt.get('state_sample_count')}",
        f"planned_stage_count: {receipt.get('planned_stage_count')}",
        f"observed_stage_count: {receipt.get('observed_stage_count')}",
        f"missing_stages: {receipt.get('missing_stages')}",
        f"timing_failures: {receipt.get('timing_failures')}",
        "```",
        "",
        "## Stage Results",
        "",
    ]
    for stage, result in (receipt.get("stage_results") or {}).items():
        audio = result.get("audio") or {}
        visual = result.get("visual_state_outputs") or {}
        lines.extend(
            [
                f"### {stage}",
                "",
                "```text",
                f"planned_beat: {result.get('planned_beat')}",
                f"expected_time: {result.get('expected_time_start')} - {result.get('expected_time_end')} s",
                f"actual_time: {result.get('actual_time_start')} - {result.get('actual_time_end')} s",
                f"sample_count: {result.get('sample_count')}",
                f"timing_status: {result.get('timing_status')}",
                f"rms_mean: {_metric_mean(audio, 'rms')}",
                f"beat_mean: {_metric_mean(audio, 'beat')}",
            ]
        )
        for metric_name in sorted(visual):
            if metric_name.endswith("_mean") or metric_name == "glow_pixels":
                lines.append(f"{metric_name}: {_metric_mean(visual, metric_name)}")
        lines.extend(["```", ""])
    lines.extend(
        [
            "## Boundary",
            "",
            "```text",
            "This receipt evaluates storyboard-state alignment.",
            "It does not reveal the internal renderer backbone.",
            "It does not claim generated media is evidence.",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _stage_result(
    plan_item: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    duration_seconds: float,
    tolerance_seconds: float,
) -> dict[str, Any]:
    expected_start = float(plan_item["phase_start"]) * duration_seconds
    expected_end = float(plan_item["phase_end"]) * duration_seconds
    times = [float(row.get("time_seconds") or 0.0) for row in rows]
    actual_start = min(times) if times else None
    actual_end = max(times) if times else None
    observed = bool(rows)
    timing_status = "missing"
    if observed:
        timing_status = (
            "pass"
            if actual_start is not None
            and actual_end is not None
            and actual_start >= expected_start - tolerance_seconds
            and actual_end <= expected_end + tolerance_seconds
            else "outside_expected_window"
        )
    return {
        "planned_beat": plan_item["planned_beat"],
        "expected_time_start": round(expected_start, 6),
        "expected_time_end": round(expected_end, 6),
        "actual_time_start": round(actual_start, 6) if actual_start is not None else None,
        "actual_time_end": round(actual_end, 6) if actual_end is not None else None,
        "sample_count": len(rows),
        "observed": observed,
        "timing_status": timing_status,
        "audio": _metric_group(rows, ["rms", "bass", "high", "beat", "vocal_presence"], parent="audio"),
        "visual_state_outputs": _metric_group(rows, _visual_metric_keys(rows), parent=None),
    }


def _visual_metric_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if (key.endswith("_mean") or key == "glow_pixels") and isinstance(value, (int, float)):
                keys.add(key)
    return sorted(keys)


def _read_state_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _metric_group(rows: list[dict[str, Any]], keys: list[str], *, parent: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        values = []
        for row in rows:
            source = row.get(parent) if parent else row
            if isinstance(source, dict) and key in source:
                values.append(float(source[key]))
        result[key] = _metric_summary(values)
    return result


def _metric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(mean(values), 6),
    }


def _metric_mean(group: dict[str, Any], key: str) -> Any:
    value = group.get(key) or {}
    return value.get("mean")


def _timing_tolerance_seconds(fps: float, state_log_every: int) -> float:
    if fps <= 0:
        return 1.0
    return max(1.0, (state_log_every / fps) + 0.25)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "state_media"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a TrueVision state-media QA receipt.")
    parser.add_argument("--manifest", required=True, help="Render manifest JSON.")
    parser.add_argument("--output-dir", default="", help="Optional receipt/report output directory.")
    args = parser.parse_args(argv)
    result = write_state_media_qa_receipt(
        args.manifest,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
