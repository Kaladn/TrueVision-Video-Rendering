from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


W, H = 1280, 720
FPS = 30
BG = np.array([10, 12, 14], dtype=np.uint8)
GOLD = (48, 178, 245)
CYAN = (230, 210, 80)
GREEN = (120, 220, 130)
RED = (80, 80, 235)
WHITE = (232, 236, 238)
MUTED = (150, 158, 164)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a source-backed learned-items sample clip.")
    parser.add_argument(
        "--yolo-report",
        default="storage/manifests/yolo_event_focus/phone_fog_yolo_focus_trial_20260531_yolo_event_focus_batch.json",
    )
    parser.add_argument(
        "--transform-report",
        default="storage/manifests/transform_learning_frontdoor/fog_reveal_real_vs_generated_trial_20260531_transform_learning_cycle.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/learned_items_sample/learned_items_yolo_fog_sample_30s",
    )
    parser.add_argument("--run-id", default="learned_items_yolo_fog_sample_30s")
    parser.add_argument("--seconds", type=float, default=30.0)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _put(frame: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.55, color=WHITE, thickness: int = 1) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if len(test) > width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def _fit_frame(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = BG
    h, w = frame.shape[:2]
    scale = min(W / max(w, 1), H / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (W - nw) // 2
    y0 = (H - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, (x0, y0, nw, nh)


def _overlay_panel(frame: np.ndarray, alpha: float = 0.68) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - 132), (W, H), (7, 9, 12), -1)
    cv2.rectangle(overlay, (0, 0), (W, 72), (7, 9, 12), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.line(frame, (0, 72), (W, 72), GOLD, 1)
    cv2.line(frame, (0, H - 132), (W, H - 132), GOLD, 1)


def _event_center(event: dict[str, Any], duration: float) -> float:
    time_range = event.get("time_range") or {}
    start = float(time_range.get("start_sec") or 0.0)
    end = float(time_range.get("end_sec") or start)
    center = (start + end) * 0.5
    return max(0.0, min(max(duration - 0.1, 0.0), center))


def _draw_event_box(frame: np.ndarray, event: dict[str, Any], fit: tuple[int, int, int, int], pulse: float) -> None:
    x0, y0, fw, fh = fit
    cx, cy, bw, bh = [float(v) for v in event.get("region_xywhn") or [0.5, 0.5, 0.1, 0.1]]
    x1 = int(x0 + (cx - bw * 0.5) * fw)
    y1 = int(y0 + (cy - bh * 0.5) * fh)
    x2 = int(x0 + (cx + bw * 0.5) * fw)
    y2 = int(y0 + (cy + bh * 0.5) * fh)
    pad = int(4 + pulse * 10)
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    color = tuple(int(a * (0.65 + 0.35 * pulse)) for a in GOLD)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.line(frame, (x1, y1), (x1 + 28, y1), WHITE, 1)
    cv2.line(frame, (x1, y1), (x1, y1 + 28), WHITE, 1)
    cv2.line(frame, (x2, y2), (x2 - 28, y2), WHITE, 1)
    cv2.line(frame, (x2, y2), (x2, y2 - 28), WHITE, 1)


def _select_events(yolo: dict[str, Any], max_events: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    preferred = [
        "candidate_traffic_light",
        "candidate_vehicle",
        "candidate_person_shape",
        "candidate_yolo_train",
        "candidate_roadside_object",
    ]
    all_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for report in yolo.get("reports", []):
        video_path = Path(report.get("video", {}).get("path", ""))
        if not video_path.exists():
            continue
        for event in report.get("focus_events", []):
            all_items.append((report, event))
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_video_type: set[tuple[str, str]] = set()
    for candidate in preferred:
        for report, event in all_items:
            key = (Path(report["video"]["path"]).name, event.get("candidate_type", ""))
            if event.get("candidate_type") == candidate and key not in used_video_type:
                selected.append((report, event))
                used_video_type.add(key)
                break
        if len(selected) >= max_events:
            return selected
    for report, event in all_items:
        key = (Path(report["video"]["path"]).name, event.get("candidate_type", ""))
        if key in used_video_type:
            continue
        selected.append((report, event))
        used_video_type.add(key)
        if len(selected) >= max_events:
            break
    return selected


def _render_event_segment(
    writer: cv2.VideoWriter,
    report: dict[str, Any],
    event: dict[str, Any],
    segment_seconds: float,
    segment_index: int,
) -> None:
    video = report.get("video", {})
    metadata = video.get("metadata", {})
    path = Path(video.get("path", ""))
    cap = cv2.VideoCapture(str(path))
    fps = float(metadata.get("fps") or cap.get(cv2.CAP_PROP_FPS) or FPS)
    duration = float(metadata.get("duration_seconds") or 0.0)
    center = _event_center(event, duration)
    start = max(0.0, center - segment_seconds * 0.5)
    event_name = event.get("candidate_label", "candidate")
    candidate = event.get("candidate_type", "candidate")
    logs = ", ".join(event.get("recommended_truevision_logs", [])[:3])
    total_frames = int(segment_seconds * FPS)
    for i in range(total_frames):
        t = start + i / FPS
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, source = cap.read()
        if not ok:
            source = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame, fit = _fit_frame(source)
        pulse = 0.5 + 0.5 * np.sin(i / max(total_frames, 1) * np.pi * 4.0)
        _overlay_panel(frame)
        _draw_event_box(frame, event, fit, pulse)
        _put(frame, f"LEARNED ITEM {segment_index:02d}: YOLO FOCUS CANDIDATE", (30, 34), 0.72, WHITE, 2)
        _put(frame, "candidate-only: state loggers must prove the event", (30, 58), 0.48, MUTED, 1)
        _put(frame, f"{path.name}", (30, H - 98), 0.58, WHITE, 1)
        _put(frame, f"{candidate} / {event_name}", (30, H - 72), 0.74, GOLD, 2)
        for n, line in enumerate(_wrap(f"focus logs: {logs}", 90)[:2]):
            _put(frame, line, (30, H - 42 + n * 22), 0.48, MUTED, 1)
        support = event.get("support", {})
        conf = float(support.get("max_confidence") or 0.0)
        cv2.rectangle(frame, (W - 300, H - 82), (W - 35, H - 56), (45, 50, 56), 1)
        cv2.rectangle(frame, (W - 298, H - 80), (W - 298 + int(conf * 260), H - 58), GOLD, -1)
        _put(frame, f"max conf {conf:.2f}", (W - 300, H - 94), 0.45, MUTED, 1)
        writer.write(frame)
    cap.release()


def _bar(frame: np.ndarray, x: int, y: int, w: int, label: str, target: float, actual: float) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + 24), (38, 42, 48), 1)
    t_w = int(min(1.0, max(0.0, target)) * w)
    a_w = int(min(1.0, max(0.0, actual)) * w)
    cv2.rectangle(frame, (x, y), (x + t_w, y + 10), GREEN, -1)
    cv2.rectangle(frame, (x, y + 13), (x + a_w, y + 23), RED if abs(actual - target) > 0.12 else CYAN, -1)
    _put(frame, label, (x, y - 8), 0.42, WHITE, 1)
    _put(frame, f"target {target:.3f} / generated {actual:.3f}", (x + w + 18, y + 18), 0.42, MUTED, 1)


def _render_transform_panel(writer: cv2.VideoWriter, transform: dict[str, Any], seconds: float) -> None:
    comparison = (transform.get("comparisons") or [{}])[0]
    deltas = comparison.get("metric_deltas") or {}
    adjustments = comparison.get("adjustments") or []
    score = comparison.get("score") or {}
    total_frames = int(seconds * FPS)
    for i in range(total_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = BG
        for y in range(H):
            frame[y, :, :] = np.clip(frame[y, :, :] + int(22 * (y / H)), 0, 255)
        phase = 0.5 + 0.5 * np.sin(i / 18.0)
        cv2.circle(frame, (1030, 210), int(90 + 18 * phase), (24, 50, 68), -1)
        cv2.circle(frame, (1030, 210), int(42 + 10 * phase), (50, 102, 140), -1)
        _put(frame, "LEARNED TRANSFORM CHECK: FOG REVEAL", (48, 58), 0.84, WHITE, 2)
        _put(frame, "real road fog profile vs generated fog proof", (50, 88), 0.55, MUTED, 1)
        accepted = "ACCEPTED" if transform.get("accepted") else "REJECTED"
        color = GREEN if transform.get("accepted") else RED
        _put(frame, accepted, (980, 62), 0.82, color, 2)
        _put(frame, f"mean error {float(score.get('mean_relative_error') or 0.0):.3f}", (980, 94), 0.5, MUTED, 1)
        y = 150
        for metric, values in list(deltas.items())[:7]:
            _bar(frame, 58, y, 420, metric, float(values.get("target") or 0.0), float(values.get("actual") or 0.0))
            y += 70
        _put(frame, "adjustment pressure", (650, 150), 0.62, WHITE, 2)
        for idx, adj in enumerate(adjustments[:7]):
            direction = adj.get("direction", "")
            metric = adj.get("metric", "")
            _put(frame, f"{idx+1}. {direction} {metric}", (650, 190 + idx * 38), 0.55, GOLD, 1)
        _put(frame, "law: learn behavior, not source shape", (50, H - 70), 0.62, WHITE, 2)
        _put(frame, "YOLO chooses where to look; TrueVision state decides what the transform must do.", (50, H - 38), 0.48, MUTED, 1)
        writer.write(frame)


def main() -> int:
    args = _parse_args()
    yolo = _load_json(args.yolo_report)
    transform = _load_json(args.transform_report)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_path = out_root / f"{args.run_id}_raw.mp4"
    final_path = out_root / f"{args.run_id}.mp4"
    manifest_path = out_root / f"{args.run_id}_manifest.json"
    max_event_segments = 5
    event_seconds = 4.0
    final_seconds = max(4.0, float(args.seconds) - max_event_segments * event_seconds)
    selected = _select_events(yolo, max_event_segments)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer for {raw_path}")
    for index, (report, event) in enumerate(selected, start=1):
        _render_event_segment(writer, report, event, event_seconds, index)
    _render_transform_panel(writer, transform, final_seconds)
    writer.release()

    ffmpeg = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_path),
        "-c:v",
        "h264_qsv",
        "-global_quality",
        "24",
        "-look_ahead",
        "0",
        "-pix_fmt",
        "nv12",
        str(final_path),
    ]
    used_encoder = "h264_qsv"
    try:
        subprocess.run(ffmpeg, check=True, capture_output=True, text=True)
    except Exception:
        used_encoder = "libx264"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                str(final_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    manifest = {
        "schema_version": "truevision_learned_items_sample_manifest_v1",
        "run_id": args.run_id,
        "output_video": str(final_path),
        "raw_intermediate": str(raw_path),
        "encoder": used_encoder,
        "fps": FPS,
        "width": W,
        "height": H,
        "selected_events": [
            {
                "video": report.get("video", {}).get("path"),
                "candidate_type": event.get("candidate_type"),
                "candidate_label": event.get("candidate_label"),
                "time_range": event.get("time_range"),
                "recommended_truevision_logs": event.get("recommended_truevision_logs", []),
            }
            for report, event in selected
        ],
        "transform_cycle": {
            "path": args.transform_report,
            "accepted": transform.get("accepted"),
            "best_attempt_id": transform.get("best_attempt_id"),
        },
        "boundary": {
            "source_backed_visualization": True,
            "raw_video_copied_as_truth": False,
            "yolo_truth_authority": False,
            "generated_clip_is_evidence": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output_video": str(final_path), "manifest_json": str(manifest_path), "encoder": used_encoder}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
