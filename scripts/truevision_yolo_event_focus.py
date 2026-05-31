from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truevision_runtime.yolo_event_focus import (  # noqa: E402
    build_yolo_event_focus_report,
    discover_video_sources,
    run_yolo_observations,
    write_batch_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local YOLO as a candidate-only focus pass for TrueVision state loggers."
    )
    parser.add_argument("--video-root", action="append", default=[], help="Directory to scan for local videos.")
    parser.add_argument("--video", action="append", default=[], help="Explicit local video path.")
    parser.add_argument("--model", default="storage/models/yolo/yolo11n.pt", help="Local YOLO model path.")
    parser.add_argument("--output-root", default="storage/manifests/yolo_event_focus", help="Manifest output root.")
    parser.add_argument("--run-id", default="yolo_event_focus", help="Run id for report/receipt files.")
    parser.add_argument("--sample-fps", type=float, default=0.25, help="Sparse sample rate per video.")
    parser.add_argument("--max-frames-per-video", type=int, default=12, help="Maximum sampled frames per video.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default="", help="Optional YOLO device string.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum videos to process after discovery.")
    parser.add_argument("--dry-run", action="store_true", help="Discover and write a manifest without YOLO inference.")
    return parser.parse_args()


def _default_roots_if_empty(video_roots: list[str], videos: list[str]) -> list[str]:
    if video_roots or videos:
        return video_roots
    default = Path("C:/Users/mydyi/Videos/MP4 from phone")
    return [str(default)] if default.exists() else []


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for source in sources:
        key = str(Path(source["path"]).resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def main() -> int:
    args = _parse_args()
    limit = args.limit if args.limit > 0 else None
    roots = _default_roots_if_empty(args.video_root, args.video)
    sources = discover_video_sources(args.video, limit=limit)
    remaining_limit = None if limit is None else max(0, limit - len(sources))
    sources.extend(discover_video_sources(roots, limit=remaining_limit))
    sources = _dedupe_sources(sources)
    if limit is not None:
        sources = sources[:limit]

    if args.dry_run:
        result = write_batch_report(
            video_sources=sources,
            reports=[],
            output_root=args.output_root,
            run_id=args.run_id,
            status="dry_run_discovery",
            model_path=args.model,
        )
        print(json.dumps(result, indent=2))
        return 0

    reports: list[dict[str, Any]] = []
    sample_policy = {
        "sample_fps": args.sample_fps,
        "max_frames": args.max_frames_per_video,
        "confidence": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "raw_frames_retained": False,
        "raw_video_copied": False,
    }
    for source in sources:
        video_path = Path(source["path"])
        try:
            observations, metadata = run_yolo_observations(
                video_path,
                model_path=args.model,
                sample_fps=args.sample_fps,
                max_frames=args.max_frames_per_video,
                confidence=args.conf,
                imgsz=args.imgsz,
                device=args.device,
            )
            reports.append(
                build_yolo_event_focus_report(
                    video_path=video_path,
                    model_path=Path(args.model),
                    observations=observations,
                    metadata=metadata,
                    sample_policy=sample_policy,
                )
            )
        except Exception as exc:  # Keep batch receipts even when one source is damaged.
            reports.append(
                {
                    "schema_version": "truevision_yolo_event_focus_error_v1",
                    "video": {"path": str(video_path)},
                    "status": "yolo_focus_failed",
                    "error": str(exc),
                    "focus_events": [],
                    "boundary": {
                        "candidate_only": True,
                        "yolo_truth_authority": False,
                        "raw_video_copied": False,
                        "frames_retained": False,
                        "six_one_six_mapping_enabled": False,
                    },
                }
            )

    result = write_batch_report(
        video_sources=sources,
        reports=reports,
        output_root=args.output_root,
        run_id=args.run_id,
        status="completed",
        model_path=args.model,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
