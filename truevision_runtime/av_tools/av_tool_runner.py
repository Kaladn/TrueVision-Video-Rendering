from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from .av_recalibration import append_recalibration_event, list_recalibration_events
from .av_tool_policy import AVToolPolicyError, safe_flat_json_name, validate_tool_call
from .av_tool_receipts import stable_hash, utc_now, write_tool_receipt


def _ensure_storage(storage_root: Path) -> None:
    for lane in ["artifacts", "events", "manifests", "receipts", "reports", "templates"]:
        path = storage_root / lane
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")


def _template_path(storage_root: Path, name: str) -> Path:
    return storage_root / "templates" / safe_flat_json_name(name)


def _read_template(storage_root: Path, name: str) -> dict[str, Any]:
    path = _template_path(storage_root, name)
    if not path.exists():
        raise FileNotFoundError(path.name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("template file must contain an object")
    return payload


def _write_template(storage_root: Path, name: str, template: dict[str, Any]) -> dict[str, Any]:
    path = _template_path(storage_root, name)
    path.write_text(json.dumps(template, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "name": path.name,
        "path": str(path),
        "sha256": stable_hash(template),
        "template": template,
    }


def _set_json_path(payload: dict[str, Any], json_path: str, value: Any) -> dict[str, Any]:
    if not json_path or any(part in {"", ".."} for part in json_path.split(".")):
        raise ValueError("json_path must be a dot-separated object path")
    target = payload
    parts = json_path.split(".")
    for part in parts[:-1]:
        existing = target.get(part)
        if not isinstance(existing, dict):
            existing = {}
            target[part] = existing
        target = existing
    target[parts[-1]] = value
    return payload


def _probe_duration(path: str) -> float | None:
    if not path:
        return None
    media_path = Path(path)
    if not media_path.exists():
        return None
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return round(float(completed.stdout.strip()), 6)
    except (subprocess.SubprocessError, ValueError):
        return None


def _create_template(args: dict[str, Any]) -> dict[str, Any]:
    fps = int(args.get("fps") or 30)
    duration = float(args.get("duration_seconds") or args.get("audio_duration_seconds") or 60)
    frame_count = max(1, int(round(duration * fps)))
    return {
        "schema_version": 1,
        "name": str(args.get("name") or "TrueVision AV template")[:120],
        "renderer": str(args.get("renderer") or "edge_audio_river"),
        "prompt": str(args.get("prompt") or ""),
        "media": {
            "audio_path": str(args.get("audio_path") or ""),
            "audio_duration_seconds": args.get("audio_duration_seconds"),
            "sync_to_audio": bool(args.get("sync_to_audio", True)),
        },
        "timeline": {
            "duration_seconds": round(duration, 6),
            "fps": fps,
            "frame_count": frame_count,
            "start_seconds": 0,
            "end_seconds": round(duration, 6),
        },
        "time_distance": {
            "source": str(args.get("duration_source") or "manual_duration"),
            "seconds_per_frame": round(1 / fps, 9),
            "frames_per_second": fps,
            "total_frames": frame_count,
        },
        "visual_parameters": deepcopy(args.get("visual_parameters") or {}),
        "state_plan": deepcopy(args.get("state_plan") or {}),
        "boundary": {
            "synthetic_state_media": True,
            "evidence": False,
            "audio_video_only": True,
            "renderer_executes_validated_state": True,
        },
    }


def _write_manifest(storage_root: Path, prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    manifests = storage_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    safe_prefix = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in prefix).strip("_") or "manifest"
    manifest = {
        "manifest_kind": "truevision_av_manifest_v1",
        "written_at_utc": now,
        **payload,
    }
    path = manifests / f"{now.replace(':', '').replace('.', '_')}_{safe_prefix}.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {"name": path.name, "path": str(path), "sha256": stable_hash(manifest), "manifest": manifest}


def _prepare_render_job(storage_root: Path, args: dict[str, Any], *, preview: bool) -> dict[str, Any]:
    template = deepcopy(args.get("template") or {})
    if not template and args.get("template_name"):
        template = _read_template(storage_root, str(args["template_name"]))
    if not template:
        template = _create_template(args)
    timeline = template.get("timeline", {})
    fps = int(args.get("fps") or timeline.get("fps") or 30)
    duration = float(args.get("seconds") or args.get("duration_seconds") or timeline.get("duration_seconds") or 5)
    if preview:
        duration = min(duration, float(args.get("max_preview_seconds") or 15))
    frame_count = max(1, int(round(duration * fps)))
    job = {
        "job_kind": "preview_render" if preview else "full_render",
        "job_id": str(args.get("job_id") or f"{'preview' if preview else 'full'}_{utc_now().replace(':', '').replace('.', '_')}"),
        "status": "prepared_preview" if preview else "prepared_requires_human_execute",
        "renderer": template.get("renderer", "edge_audio_river"),
        "timeline": {
            "duration_seconds": round(duration, 6),
            "fps": fps,
            "frame_count": frame_count,
        },
        "template_hash": stable_hash(template),
        "template": template,
        "boundary": {
            "synthetic_state_media": True,
            "evidence": False,
            "execute_requires_human_confirmation": not preview,
        },
    }
    manifest = _write_manifest(storage_root, f"{job['job_kind']}_{job['job_id']}", {"job": job})
    return {"job": job, "manifest": manifest}


def _list_media_artifacts(storage_root: Path, lane: str | None = None) -> list[dict[str, Any]]:
    lanes = [lane] if lane else ["artifacts", "manifests", "reports", "templates", "receipts", "events"]
    files: list[dict[str, Any]] = []
    for item_lane in lanes:
        root = storage_root / item_lane
        if not root.exists():
            continue
        for path in root.glob("*"):
            if path.is_file() and path.name != ".gitkeep":
                files.append({"name": path.name, "lane": item_lane, "path": str(path), "size_bytes": path.stat().st_size})
    return sorted(files, key=lambda item: item["name"])


def _execute_validated_tool(validated: dict[str, Any], storage_root: Path) -> dict[str, Any]:
    tool = validated["tool"]
    args = validated["args"]
    if tool == "audio_probe_duration":
        duration = _probe_duration(str(args.get("path") or args.get("audio_path") or ""))
        return {"duration_seconds": duration, "path": str(args.get("path") or args.get("audio_path") or "")}
    if tool == "template_create":
        return {"template": _create_template(args)}
    if tool == "template_save":
        template = args.get("template")
        if not isinstance(template, dict):
            template = _create_template(args)
        name = str(args.get("name") or f"{template.get('name', 'template')}.json")
        return _write_template(storage_root, name, template)
    if tool == "template_load":
        name = str(args.get("name") or "")
        return {"name": safe_flat_json_name(name), "template": _read_template(storage_root, name)}
    if tool == "template_patch":
        name = str(args.get("name") or "")
        template = _read_template(storage_root, name)
        patched = _set_json_path(deepcopy(template), str(args.get("json_path") or ""), args.get("value"))
        patched.setdefault("patch_history", []).append(
            {
                "patched_at_utc": utc_now(),
                "json_path": str(args.get("json_path") or ""),
                "reason": str(args.get("reason") or ""),
            }
        )
        return _write_template(storage_root, name, patched)
    if tool == "template_create_variant":
        source = _read_template(storage_root, str(args.get("source_name") or args.get("name") or ""))
        variant = deepcopy(source)
        variant["name"] = str(args.get("variant_name") or f"{source.get('name', 'template')} variant")[:120]
        for path, value in (args.get("changes") or {}).items():
            _set_json_path(variant, str(path), value)
        variant.setdefault("variant_of", source.get("template_id") or source.get("name"))
        return _write_template(storage_root, str(args.get("variant_file") or f"{variant['name']}.json"), variant)
    if tool == "template_delete":
        path = _template_path(storage_root, str(args.get("name") or ""))
        existed = path.exists()
        if existed:
            path.unlink()
        return {"name": path.name, "deleted": existed}
    if tool == "storage_list_templates":
        return {"templates": _list_media_artifacts(storage_root, "templates")}
    if tool == "storage_list_artifacts":
        return {"files": _list_media_artifacts(storage_root, args.get("lane"))}
    if tool == "time_marker_add":
        event = {
            "kind": "time_marker",
            "template_id": str(args.get("template_id") or ""),
            "source_artifact": str(args.get("source_artifact") or ""),
            "time_seconds": float(args.get("time_seconds") or 0),
            "note": str(args.get("note") or ""),
            "target": str(args.get("target") or ""),
            "direction": str(args.get("direction") or ""),
            "confidence": float(args.get("confidence") or 1.0),
        }
        return {"marker": append_recalibration_event(storage_root=storage_root, event=event)["event"]}
    if tool == "time_marker_list":
        return {"markers": list_recalibration_events(storage_root=storage_root, template_id=args.get("template_id"), kind="time_marker")}
    if tool == "recalibration_add_note":
        event = {
            "kind": "recalibration_note",
            "template_id": str(args.get("template_id") or ""),
            "source_artifact": str(args.get("source_artifact") or ""),
            "time_seconds": float(args.get("time_seconds") or 0),
            "note": str(args.get("note") or ""),
            "target": str(args.get("target") or ""),
            "direction": str(args.get("direction") or ""),
            "confidence": float(args.get("confidence") or 1.0),
        }
        return {"note": append_recalibration_event(storage_root=storage_root, event=event)["event"]}
    if tool == "recalibration_apply":
        notes = list_recalibration_events(storage_root=storage_root, template_id=args.get("template_id"), kind="recalibration_note")
        return {"patch_proposal": {"template_id": args.get("template_id"), "notes_used": len(notes), "notes": notes}}
    if tool == "video_render_preview":
        return _prepare_render_job(storage_root, args, preview=True)
    if tool == "video_prepare_full_render":
        return _prepare_render_job(storage_root, args, preview=False)
    if tool == "video_execute_full_render":
        return {"status": "execution_gated", "message": "Full render execution is not wired here; prepare job was approved only."}
    if tool == "manifest_generate":
        return _write_manifest(storage_root, str(args.get("name") or "av_manifest"), {"payload": args.get("payload") or args})
    if tool == "learning_record_save":
        event = {"kind": "learning_record", **args}
        return {"learning_record": append_recalibration_event(storage_root=storage_root, event=event)["event"]}
    if tool == "receipt_create":
        return {"status": "receipt_create_is_internal"}
    raise AVToolPolicyError(f"tool has no runner implementation: {tool}")


def run_av_tool_call(call: dict[str, Any], *, storage_root: Path) -> dict[str, Any]:
    _ensure_storage(storage_root)
    tool = str(call.get("tool") or "unknown") if isinstance(call, dict) else "invalid"
    try:
        validated = validate_tool_call(call)
        result = _execute_validated_tool(validated, storage_root)
        receipt = write_tool_receipt(storage_root=storage_root, tool=validated["tool"], status="ok", call=validated, result=result)
        return {"ok": True, "tool": validated["tool"], "result": result, "receipt": receipt}
    except Exception as exc:
        receipt = write_tool_receipt(storage_root=storage_root, tool=tool, status="rejected", call=call if isinstance(call, dict) else {}, error=str(exc))
        return {"ok": False, "tool": tool, "error": str(exc), "receipt": receipt}
