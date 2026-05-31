from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
REPORT_SCHEMA = "truevision_yolo_event_focus_report_v1"
BATCH_SCHEMA = "truevision_yolo_event_focus_batch_v1"


CLASS_TO_CANDIDATE = {
    "person": "candidate_person_shape",
    "car": "candidate_vehicle",
    "truck": "candidate_vehicle",
    "bus": "candidate_vehicle",
    "motorcycle": "candidate_vehicle",
    "bicycle": "candidate_bike_motorcycle",
    "traffic light": "candidate_traffic_light",
    "stop sign": "candidate_stop_sign",
    "parking meter": "candidate_roadside_object",
    "bench": "candidate_roadside_object",
    "dog": "candidate_animal_shape",
    "cat": "candidate_animal_shape",
}


FOCUS_LOGS_BY_CANDIDATE = {
    "candidate_vehicle": ["meter_grid_from_capture", "angular_seismic_16_direction", "state_focus_lens", "driving_school_awareness"],
    "candidate_person_shape": ["meter_grid_from_capture", "state_focus_lens", "geometry_generation"],
    "candidate_traffic_light": ["meter_grid_from_capture", "state_focus_lens", "deep_pixel_transform_analysis"],
    "candidate_stop_sign": ["meter_grid_from_capture", "state_focus_lens", "geometry_generation"],
    "candidate_bike_motorcycle": ["meter_grid_from_capture", "angular_seismic_16_direction", "state_focus_lens"],
    "candidate_roadside_object": ["state_focus_lens", "geometry_generation", "driving_school_awareness"],
    "candidate_animal_shape": ["state_focus_lens", "geometry_generation"],
}


def _safe_id(value: str | None, fallback: str = "yolo_event_focus") -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "")).strip("_")
    return safe or fallback


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def discover_video_sources(roots: Iterable[str | Path], *, limit: int | None = None) -> list[dict[str, Any]]:
    videos: list[dict[str, Any]] = []
    for root_value in roots:
        root = Path(root_value)
        if root.is_file() and root.suffix.lower() in VIDEO_EXTENSIONS:
            candidates = [root]
        elif root.exists():
            candidates = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
        else:
            candidates = []
        for path in candidates:
            lowered = str(path).lower()
            if "mp4 from phone" in lowered or path.name[:8].isdigit():
                source_kind = "phone_video_candidate"
            elif "video project" in lowered or "fog" in lowered:
                source_kind = "known_local_video_candidate"
            else:
                source_kind = "local_video_candidate"
            videos.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "source_kind": source_kind,
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                    "modified_utc": utc_now(),
                    "raw_video_copied": False,
                }
            )
            if limit is not None and len(videos) >= limit:
                return videos
    return videos


def _video_metadata(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"fps": 0.0, "width": 0, "height": 0, "frame_count": 0, "duration_seconds": 0.0, "open_ok": False}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = float(frame_count / fps) if fps > 0.0 and frame_count > 0 else 0.0
    return {
        "fps": round(fps, 6),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_seconds": round(duration, 6),
        "open_ok": True,
    }


def _sample_frame_indices(metadata: dict[str, Any], *, sample_fps: float, max_frames: int) -> list[int]:
    fps = float(metadata.get("fps") or 0.0)
    frame_count = int(metadata.get("frame_count") or 0)
    if fps <= 0.0 or frame_count <= 0:
        return []
    stride = max(1, int(round(fps / max(sample_fps, 1.0e-6))))
    indices = list(range(0, frame_count, stride))
    if len(indices) > max_frames:
        pick = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
        indices = [indices[int(index)] for index in pick]
    return indices


def _xyxy_to_xywhn(xyxy: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    return [
        round(cx / max(width, 1), 6),
        round(cy / max(height, 1), 6),
        round(w / max(width, 1), 6),
        round(h / max(height, 1), 6),
    ]


def run_yolo_observations(
    video_path: str | Path,
    *,
    model_path: str | Path,
    sample_fps: float = 0.25,
    max_frames: int = 12,
    confidence: float = 0.25,
    imgsz: int = 640,
    device: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ultralytics import YOLO

    path = Path(video_path)
    metadata = _video_metadata(path)
    indices = _sample_frame_indices(metadata, sample_fps=sample_fps, max_frames=max_frames)
    if not metadata.get("open_ok"):
        return [], metadata

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(path))
    observations: list[dict[str, Any]] = []
    width = int(metadata["width"])
    height = int(metadata["height"])
    fps = float(metadata["fps"] or 1.0)
    predict_kwargs: dict[str, Any] = {"conf": confidence, "imgsz": imgsz, "verbose": False}
    if device:
        predict_kwargs["device"] = device

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        result = model.predict(frame, **predict_kwargs)[0]
        detections: list[dict[str, Any]] = []
        names = result.names or getattr(model, "names", {})
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            for box, conf_value, cls_value in zip(xyxy, confs, classes):
                class_index = int(cls_value)
                class_name = str(names.get(class_index, class_index))
                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": round(float(conf_value), 6),
                        "xywhn": _xyxy_to_xywhn(box, width, height),
                    }
                )
        observations.append(
            {
                "frame_index": int(frame_index),
                "time_sec": round(float(frame_index) / max(fps, 1.0e-9), 6),
                "detections": detections,
            }
        )
    cap.release()
    metadata["sampled_frame_count"] = len(observations)
    metadata["sampled_indices"] = indices
    return observations, metadata


def _merge_bounds(items: list[list[float]]) -> list[float]:
    if not items:
        return [0.0, 0.0, 0.0, 0.0]
    x1s, y1s, x2s, y2s = [], [], [], []
    for cx, cy, w, h in items:
        x1s.append(cx - w * 0.5)
        y1s.append(cy - h * 0.5)
        x2s.append(cx + w * 0.5)
        y2s.append(cy + h * 0.5)
    x1, y1 = max(0.0, min(x1s)), max(0.0, min(y1s))
    x2, y2 = min(1.0, max(x2s)), min(1.0, max(y2s))
    return [round((x1 + x2) * 0.5, 6), round((y1 + y2) * 0.5, 6), round(max(0.0, x2 - x1), 6), round(max(0.0, y2 - y1), 6)]


def _candidate_type(class_name: str) -> str:
    return CLASS_TO_CANDIDATE.get(class_name, f"candidate_yolo_{_safe_id(class_name, 'object')}")


def _recommended_logs(candidate_type: str) -> list[str]:
    return FOCUS_LOGS_BY_CANDIDATE.get(candidate_type, ["meter_grid_from_capture", "state_focus_lens", "geometry_generation"])


def build_yolo_event_focus_report(
    *,
    video_path: str | Path,
    model_path: str | Path,
    observations: list[dict[str, Any]],
    metadata: dict[str, Any],
    sample_policy: dict[str, Any],
    hash_video: bool = False,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        for detection in observation.get("detections", []):
            item = dict(detection)
            item["frame_index"] = int(observation.get("frame_index") or 0)
            item["time_sec"] = float(observation.get("time_sec") or 0.0)
            grouped[str(detection.get("class_name") or "unknown")].append(item)

    focus_events: list[dict[str, Any]] = []
    for class_name, items in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        candidate_type = _candidate_type(class_name)
        confidences = [float(item.get("confidence") or 0.0) for item in items]
        times = [float(item.get("time_sec") or 0.0) for item in items]
        event = {
            "event_id": f"yolo_focus_{_safe_id(class_name)}",
            "candidate_label": class_name,
            "candidate_type": candidate_type,
            "status": "candidate_yolo_supported",
            "support": {
                "detection_count": len(items),
                "sampled_frame_count": len({int(item.get("frame_index") or 0) for item in items}),
                "max_confidence": round(max(confidences) if confidences else 0.0, 6),
                "mean_confidence": round(sum(confidences) / max(len(confidences), 1), 6),
            },
            "time_range": {
                "start_sec": round(min(times) if times else 0.0, 6),
                "end_sec": round(max(times) if times else 0.0, 6),
            },
            "region_xywhn": _merge_bounds([list(item.get("xywhn") or [0.0, 0.0, 0.0, 0.0]) for item in items]),
            "recommended_truevision_logs": _recommended_logs(candidate_type),
            "truth_boundary": {
                "candidate_only": True,
                "yolo_truth_authority": False,
                "requires_state_log_support": True,
            },
        }
        focus_events.append(event)

    counts = Counter()
    for observation in observations:
        counts.update(str(detection.get("class_name") or "unknown") for detection in observation.get("detections", []))

    report = {
        "schema_version": REPORT_SCHEMA,
        "created_at_utc": utc_now(),
        "video": {
            "path": str(video_path),
            "sha256": _file_sha256(Path(video_path)) if hash_video and Path(video_path).exists() and Path(video_path).is_file() else "",
            "hash_status": "computed" if hash_video else "skipped_raw_video_hash_for_speed",
            "metadata": metadata,
        },
        "model": {
            "path": str(model_path),
            "sha256": _file_sha256(Path(model_path)) if Path(model_path).exists() else "",
        },
        "sample_policy": sample_policy,
        "content_summary": {
            "top_yolo_classes": [{"class_name": name, "count": count} for name, count in counts.most_common()],
            "observation_count": len(observations),
            "detection_count": sum(counts.values()),
        },
        "observations": observations,
        "focus_events": focus_events,
        "focus_log_contract": {
            "use_yolo_to_choose_time_region": True,
            "use_truevision_state_to_measure_event": True,
            "yolo_describes_candidate_content": True,
            "yolo_promotes_truth": False,
        },
        "boundary": {
            "candidate_only": True,
            "yolo_truth_authority": False,
            "raw_video_copied": False,
            "frames_retained": False,
            "six_one_six_mapping_enabled": False,
            "generated_media_is_evidence": False,
        },
    }
    report["report_sha256"] = stable_hash({key: value for key, value in report.items() if key != "report_sha256"})
    return report


def write_batch_report(
    *,
    video_sources: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    output_root: str | Path,
    run_id: str,
    status: str,
    model_path: str | Path,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    batch = {
        "schema_version": BATCH_SCHEMA,
        "created_at_utc": utc_now(),
        "run_id": _safe_id(run_id),
        "status": status,
        "model_path": str(model_path),
        "video_sources": video_sources,
        "reports": reports,
        "summary": {
            "video_source_count": len(video_sources),
            "processed_report_count": len(reports),
            "focus_event_count": sum(len(report.get("focus_events", [])) for report in reports),
        },
        "boundary": {
            "candidate_only": True,
            "yolo_truth_authority": False,
            "raw_video_copied": False,
            "frames_retained": False,
            "six_one_six_mapping_enabled": False,
        },
    }
    batch["batch_sha256"] = stable_hash({key: value for key, value in batch.items() if key != "batch_sha256"})
    report_path = root / f"{_safe_id(run_id)}_yolo_event_focus_batch.json"
    report_path.write_text(json.dumps(batch, indent=2, allow_nan=False), encoding="utf-8")
    receipt_dir = root.parent.parent / "receipts" / "yolo_event_focus" if root.parts[-2:] == ("manifests", "yolo_event_focus") else root / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "truevision_yolo_event_focus_receipt_v1",
        "created_at_utc": utc_now(),
        "run_id": _safe_id(run_id),
        "report_json": str(report_path),
        "status": status,
        "batch_sha256": batch["batch_sha256"],
        "boundary": batch["boundary"],
    }
    receipt["receipt_sha256"] = stable_hash({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    receipt_path = receipt_dir / f"{_safe_id(run_id)}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "report_json": str(report_path),
        "receipt_json": str(receipt_path),
        "summary": batch["summary"],
        "status": status,
    }
