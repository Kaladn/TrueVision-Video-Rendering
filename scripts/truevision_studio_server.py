from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from truevision_region_snip import build_recorder_command


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "ui" / "truevision_state_media_studio.html"
STORAGE_ROOT = ROOT / "storage"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MODEL = "qwen3-coder:30b"
STORAGE_LANES = {
    "inbox",
    "outbox",
    "events",
    "state_chunks",
    "artifacts",
    "chats",
    "manifests",
    "reports",
    "receipts",
    "presets",
    "templates",
    "tmp",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def local_day() -> str:
    return datetime.now().date().isoformat()


def slug(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return clean.strip("_")[:80] or "artifact"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def ensure_storage_layout(storage_root: Path = STORAGE_ROOT) -> None:
    for lane in STORAGE_LANES:
        path = storage_root / lane
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")


def write_json_artifact(
    *,
    storage_root: Path,
    lane: str,
    prefix: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if lane not in STORAGE_LANES:
        raise ValueError(f"Unknown storage lane: {lane}")
    ensure_storage_layout(storage_root)
    now = utc_now()
    filename = f"{now.replace(':', '').replace('.', '_')}_{slug(prefix)}.json"
    path = storage_root / lane / filename
    envelope = {
        "written_at_utc": now,
        "storage_lane": lane,
        "payload": payload,
    }
    path.write_text(json.dumps(envelope, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "name": path.name,
        "path": str(path),
        "lane": lane,
        "kind": "json",
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "written_at_utc": now,
    }


def list_storage_files(storage_root: Path = STORAGE_ROOT) -> list[dict[str, Any]]:
    ensure_storage_layout(storage_root)
    files: list[dict[str, Any]] = []
    for path in storage_root.rglob("*"):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        relative = path.relative_to(storage_root)
        lane = relative.parts[0] if relative.parts else "storage"
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "relative_path": str(relative),
                "lane": lane,
                "kind": path.suffix.lstrip(".") or "file",
                "size_bytes": path.stat().st_size,
                "modified_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
            }
        )
    return sorted(files, key=lambda item: item["modified_at_utc"], reverse=True)


def append_chat_message(
    *,
    storage_root: Path,
    message: dict[str, Any],
    day: str | None = None,
) -> dict[str, Any]:
    ensure_storage_layout(storage_root)
    chat_day = day or local_day()
    if not chat_day or any(char not in "0123456789-" for char in chat_day):
        raise ValueError("day must use YYYY-MM-DD characters")
    path = storage_root / "chats" / f"{chat_day}.jsonl"
    entry = {
        "written_at_utc": utc_now(),
        "source": str(message.get("source") or "unknown")[:80],
        "text": str(message.get("text") or ""),
        "operator": bool(message.get("operator", False)),
        "kind": str(message.get("kind") or "chat")[:80],
    }
    if message.get("meta") is not None:
        entry["meta"] = message["meta"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    return {
        "name": path.name,
        "path": str(path),
        "lane": "chats",
        "kind": "jsonl",
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "day": chat_day,
    }


def read_chat_log(*, storage_root: Path, day: str | None = None) -> list[dict[str, Any]]:
    ensure_storage_layout(storage_root)
    chat_day = day or local_day()
    path = storage_root / "chats" / f"{chat_day}.jsonl"
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def _template_path(storage_root: Path, name: str) -> Path:
    filename = Path(name).name
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    if filename in {"", ".json"} or filename != slug(filename.removesuffix(".json")) + ".json":
        raise ValueError("template name must be a flat safe JSON filename")
    return storage_root / "templates" / filename


def save_template(
    *,
    storage_root: Path,
    template: dict[str, Any],
    name: str | None = None,
) -> dict[str, Any]:
    ensure_storage_layout(storage_root)
    now = utc_now()
    template_name = str(template.get("name") or name or "truevision_template")
    filename = name or f"{now.replace(':', '').replace('.', '_')}_{slug(template_name)}.json"
    path = _template_path(storage_root, filename)
    payload = {
        "template_id": path.stem,
        "created_at_utc": str(template.get("created_at_utc") or now),
        "updated_at_utc": now,
        **template,
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "name": path.name,
        "path": str(path),
        "lane": "templates",
        "kind": "json",
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "template": payload,
    }


def list_templates(storage_root: Path = STORAGE_ROOT) -> list[dict[str, Any]]:
    ensure_storage_layout(storage_root)
    templates: list[dict[str, Any]] = []
    for path in sorted((storage_root / "templates").glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            template = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            template = {"name": path.stem, "read_error": "invalid_json"}
        templates.append(
            {
                "name": path.name,
                "path": str(path),
                "lane": "templates",
                "kind": "json",
                "size_bytes": path.stat().st_size,
                "modified_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                "template": template,
            }
        )
    return templates


def delete_template(*, storage_root: Path, name: str) -> dict[str, Any]:
    ensure_storage_layout(storage_root)
    path = _template_path(storage_root, name)
    existed = path.exists()
    if existed:
        path.unlink()
    return {
        "name": path.name,
        "path": str(path),
        "lane": "templates",
        "deleted": existed,
    }


def probe_media_duration(path: str) -> float | None:
    if not path:
        return None
    media_path = Path(path)
    if not media_path.exists():
        return None
    try:
        result = subprocess.run(
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
        return round(float(result.stdout.strip()), 6)
    except (subprocess.SubprocessError, ValueError):
        return None


def build_generation_template_from_request(request: dict[str, Any]) -> dict[str, Any]:
    media = request.get("media") if isinstance(request.get("media"), dict) else {}
    capture = request.get("capture_shape") if isinstance(request.get("capture_shape"), dict) else {}
    renderer_payload = request.get("renderer") if isinstance(request.get("renderer"), dict) else {}
    fps = int(capture.get("fps") or request.get("fps") or 30)
    audio_path = str(media.get("audio_path") or "")
    sync_to_audio = bool(media.get("sync_to_audio", True))
    audio_duration = media.get("audio_duration_seconds")
    duration_source = "manual_duration"
    if sync_to_audio and audio_duration not in {None, ""}:
        duration_seconds = float(audio_duration)
        duration_source = "audio_duration"
    elif sync_to_audio and audio_path:
        probed = probe_media_duration(audio_path)
        if probed:
            duration_seconds = float(probed)
            duration_source = "audio_probe"
        else:
            duration_seconds = float(capture.get("duration_seconds") or 60)
    else:
        duration_seconds = float(capture.get("duration_seconds") or float(capture.get("duration_minutes", 1)) * 60)
    frame_count = max(1, int(round(duration_seconds * fps)))
    renderer = str(renderer_payload.get("name") or request.get("renderer_name") or "state_formula")
    return {
        "schema_version": 1,
        "name": str(request.get("template_name") or request.get("prompt") or "TrueVision template")[:120],
        "renderer": renderer,
        "prompt": str(request.get("prompt") or ""),
        "media": {
            "audio_path": audio_path,
            "audio_duration_seconds": duration_seconds if duration_source.startswith("audio") else audio_duration,
            "sync_to_audio": sync_to_audio,
        },
        "timeline": {
            "duration_seconds": round(duration_seconds, 6),
            "fps": fps,
            "frame_count": frame_count,
            "start_seconds": 0,
            "end_seconds": round(duration_seconds, 6),
        },
        "time_distance": {
            "source": duration_source,
            "seconds_per_frame": round(1 / fps, 9),
            "frames_per_second": fps,
            "total_frames": frame_count,
        },
        "visual_parameters": {
            "geometry": request.get("geometry", {}),
            "trigonometry": request.get("trigonometry", {}),
            "linear_algebra": request.get("linear_algebra", {}),
            "physics": request.get("physics", {}),
            "electronics": request.get("electronics", {}),
            "path_tracing": request.get("path_tracing", {}),
            "computer_vision": request.get("computer_vision", {}),
        },
        "state_plan": request.get("qwen_state_plan") or {},
        "boundary": {
            "synthetic_state_media": True,
            "evidence": False,
            "renderer_executes_validated_state": True,
        },
    }


def build_recording_command_from_request(
    request: dict[str, Any],
    *,
    storage_root: Path = STORAGE_ROOT,
) -> dict[str, Any]:
    capture = request.get("capture_shape", {})
    record_zone = request.get("record_start_zone", {})
    duration_seconds = int(round(float(capture.get("duration_minutes", 1)) * 60))
    fps = int(capture.get("fps", 9))
    resolution = [
        int(capture.get("resolution_width", 960)),
        int(capture.get("resolution_height", 540)),
    ]
    grid = [
        int(capture.get("grid_width", 160)),
        int(capture.get("grid_height", 90)),
    ]
    snapped_region = record_zone.get("snapped_region") or [0, 0, resolution[0], resolution[1]]
    selected_region = record_zone.get("selected_region") or snapped_region
    preset = {
        "preset_id": request.get("run_id", "studio_region"),
        "selected_region": [int(value) for value in selected_region],
        "snapped_region": [int(value) for value in snapped_region],
        "capture_resolution": resolution,
        "grid": grid,
        "blocks": [16, 9],
        "monitor": int(record_zone.get("monitor", 0)),
    }
    run_id = slug(str(request.get("run_id") or f"studio_{utc_now()}"))
    command = build_recorder_command(
        preset,
        duration=duration_seconds,
        fps=fps,
        output_root=storage_root / "artifacts",
        run_id=run_id,
        python_exe=sys.executable,
    )
    start_delay_seconds = int(round(float(record_zone.get("start_delay_minutes", 0)) * 60))
    countdown_seconds = int(record_zone.get("countdown_seconds", 0))
    return {
        "run_id": run_id,
        "duration_seconds": duration_seconds,
        "fps": fps,
        "start_delay_seconds": start_delay_seconds,
        "countdown_seconds": countdown_seconds,
        "preset": preset,
        "command": command,
        "command_text": " ".join(f'"{part}"' if " " in part else part for part in command),
    }


def _append_action(actions: list[str], action: str) -> None:
    if action not in actions:
        actions.append(action)


def resolve_assistant_actions(message: str, request: dict[str, Any]) -> list[str]:
    text = message.lower()
    actions: list[str] = []

    wants_files = any(word in text for word in ["files", "list", "refresh", "show artifacts", "what exists"])
    wants_save = any(word in text for word in ["save", "persist", "write", "store"])
    wants_record = any(word in text for word in ["prepare", "record", "capture", "recorder", "command"])
    looks_like_visual_prompt = any(
        word in text
        for word in [
            "animate",
            "camera",
            "clip",
            "field",
            "frame",
            "generate",
            "image",
            "lighting",
            "motion",
            "person",
            "photo",
            "render",
            "scene",
            "shot",
            "sunset",
            "video",
            "visual",
            "walk",
        ]
    )
    wants_compile = any(word in text for word in ["compile", "generate", "draft", "state", "qwen", "catbot", "do it"])
    wants_compile = wants_compile or looks_like_visual_prompt

    if wants_files:
        _append_action(actions, "refresh_files")
    if wants_save or wants_record or wants_compile:
        _append_action(actions, "save_request")
    if wants_compile and request.get("local_llm", {}).get("enabled"):
        _append_action(actions, "qwen_compile")
    if wants_record:
        _append_action(actions, "prepare_record")

    if not actions:
        if request.get("local_llm", {}).get("enabled"):
            _append_action(actions, "qwen_chat")
        else:
            _append_action(actions, "save_request")

    return actions


def handle_assistant_message(
    payload: dict[str, Any],
    *,
    storage_root: Path = STORAGE_ROOT,
) -> dict[str, Any]:
    message = str(payload.get("message") or "").strip()
    request = payload.get("request")
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    actions = resolve_assistant_actions(message, request)
    results: dict[str, Any] = {}

    if "save_request" in actions:
        results["request"] = write_json_artifact(
            storage_root=storage_root,
            lane="outbox",
            prefix="assistant_state_request",
            payload=request,
        )
    if "prepare_record" in actions:
        recording = build_recording_command_from_request(request, storage_root=storage_root)
        results["recording"] = recording
        results["recording_artifact"] = write_json_artifact(
            storage_root=storage_root,
            lane="manifests",
            prefix="assistant_record_command",
            payload=recording,
        )
    files = list_storage_files(storage_root)

    pending_actions = {"qwen_chat", "qwen_compile"}
    completed = [action for action in actions if action not in pending_actions]
    pending = [action for action in actions if action in pending_actions]
    parts = []
    if completed:
        parts.append("ran " + ", ".join(completed))
    if pending:
        parts.append("queued " + ", ".join(pending))
    if not parts:
        parts.append("no action")

    return {
        "ok": True,
        "assistant": "Catbot " + "; ".join(parts) + ".",
        "actions": actions,
        "results": results,
        "files": files,
    }


def normalize_provider(provider: str | None) -> str:
    if provider == "openai_compatible":
        return "openai_compatible"
    return "ollama_native"


def validate_local_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "http":
        raise ValueError("Only local http endpoints are allowed")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Only loopback model endpoints are allowed")
    if not parsed.port:
        raise ValueError("Model endpoint must include a port")
    return endpoint


def build_messages(system_prompt: str, request: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(request, indent=2, sort_keys=True)},
    ]


def build_downstream_payload(
    provider: str,
    model: str,
    system_prompt: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    provider = normalize_provider(provider)
    messages = build_messages(system_prompt, request)
    if provider == "openai_compatible":
        return {
            "model": model or DEFAULT_MODEL,
            "temperature": 0.1,
            "stream": False,
            "messages": messages,
        }
    return {
        "model": model or DEFAULT_MODEL,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": 0.1,
        },
    }


def extract_model_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
    message = payload.get("message")
    if isinstance(message, dict) and message.get("content"):
        return str(message["content"])
    if payload.get("response"):
        return str(payload["response"])
    raise ValueError("Model response did not include message content")


def fetch_json(
    endpoint: str,
    method: str,
    payload: dict[str, Any] | None = None,
    api_key: str = "",
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(endpoint, data=data, headers=headers, method=method)
    with urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "TrueVisionStudio/0.1"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_file(HTML_PATH, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/health":
            self._send_json({"ok": True, "service": "truevision_studio_server"})
            return
        if parsed.path == "/api/local-llm/models":
            self._handle_models(parsed.query)
            return
        if parsed.path == "/api/files":
            self._send_json({"ok": True, "files": list_storage_files(STORAGE_ROOT)})
            return
        if parsed.path == "/api/chat/today":
            values = parse_qs(parsed.query)
            day = values.get("day", [None])[0]
            self._send_json({"ok": True, "day": day or local_day(), "messages": read_chat_log(storage_root=STORAGE_ROOT, day=day)})
            return
        if parsed.path == "/api/templates":
            self._send_json({"ok": True, "templates": list_templates(STORAGE_ROOT)})
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/local-llm/chat":
            self._handle_local_llm_chat()
            return
        if parsed.path == "/api/state/request":
            self._handle_state_request()
            return
        if parsed.path == "/api/state/plan":
            self._handle_state_plan()
            return
        if parsed.path == "/api/record/prepare":
            self._handle_record_prepare()
            return
        if parsed.path == "/api/assistant/message":
            self._handle_assistant_message()
            return
        if parsed.path == "/api/chat/log":
            self._handle_chat_log()
            return
        if parsed.path == "/api/templates/save":
            self._handle_template_save()
            return
        if parsed.path == "/api/templates/delete":
            self._handle_template_delete()
            return
        if parsed.path == "/api/media/probe":
            self._handle_media_probe()
            return
        self.send_error(404, "Not found")

    def _handle_local_llm_chat(self) -> None:
        try:
            payload = self._read_json()
            endpoint = validate_local_endpoint(str(payload.get("endpoint", "")))
            provider = normalize_provider(str(payload.get("provider", "")))
            model = str(payload.get("model") or DEFAULT_MODEL)
            api_key = str(payload.get("api_key") or "")
            system_prompt = str(payload.get("system_prompt") or "")
            request_body = payload.get("request")
            if not isinstance(request_body, dict):
                raise ValueError("request must be an object")
            downstream = build_downstream_payload(provider, model, system_prompt, request_body)
            upstream = fetch_json(endpoint, "POST", downstream, api_key)
            self._send_json(
                {
                    "ok": True,
                    "provider": provider,
                    "model": model,
                    "content": extract_model_content(upstream),
                    "raw": upstream,
                }
            )
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._send_json({"ok": False, "error": detail or str(exc), "status": exc.code}, exc.code)
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _handle_state_request(self) -> None:
        try:
            payload = self._read_json()
            request = payload.get("request", payload)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            artifact = write_json_artifact(
                storage_root=STORAGE_ROOT,
                lane="outbox",
                prefix="state_request",
                payload=request,
            )
            event = write_json_artifact(
                storage_root=STORAGE_ROOT,
                lane="events",
                prefix="state_request_saved",
                payload={"event": "state_request_saved", "artifact": artifact},
            )
            self._send_json({"ok": True, "artifact": artifact, "event": event, "files": list_storage_files(STORAGE_ROOT)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _handle_state_plan(self) -> None:
        try:
            payload = self._read_json()
            plan = payload.get("plan", payload)
            if not isinstance(plan, dict):
                raise ValueError("plan must be an object")
            artifact = write_json_artifact(
                storage_root=STORAGE_ROOT,
                lane="manifests",
                prefix="state_plan",
                payload=plan,
            )
            self._send_json({"ok": True, "artifact": artifact, "files": list_storage_files(STORAGE_ROOT)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _handle_record_prepare(self) -> None:
        try:
            payload = self._read_json()
            request = payload.get("request", payload)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            prepared = build_recording_command_from_request(request, storage_root=STORAGE_ROOT)
            artifact = write_json_artifact(
                storage_root=STORAGE_ROOT,
                lane="manifests",
                prefix="record_command",
                payload=prepared,
            )
            self._send_json(
                {
                    "ok": True,
                    "recording": prepared,
                    "artifact": artifact,
                    "files": list_storage_files(STORAGE_ROOT),
                }
            )
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _handle_assistant_message(self) -> None:
        try:
            payload = self._read_json()
            result = handle_assistant_message(payload, storage_root=STORAGE_ROOT)
            self._send_json(result)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _handle_chat_log(self) -> None:
        try:
            payload = self._read_json()
            message = payload.get("message", payload)
            if not isinstance(message, dict):
                raise ValueError("message must be an object")
            day = payload.get("day")
            artifact = append_chat_message(
                storage_root=STORAGE_ROOT,
                message=message,
                day=str(day) if day else None,
            )
            self._send_json({"ok": True, "artifact": artifact})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _handle_template_save(self) -> None:
        try:
            payload = self._read_json()
            request = payload.get("request")
            template = payload.get("template")
            if isinstance(request, dict):
                template = build_generation_template_from_request(request)
            if not isinstance(template, dict):
                raise ValueError("template or request must be an object")
            artifact = save_template(
                storage_root=STORAGE_ROOT,
                template=template,
                name=str(payload.get("name")) if payload.get("name") else None,
            )
            self._send_json(
                {
                    "ok": True,
                    "artifact": artifact,
                    "templates": list_templates(STORAGE_ROOT),
                    "files": list_storage_files(STORAGE_ROOT),
                }
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _handle_template_delete(self) -> None:
        try:
            payload = self._read_json()
            name = str(payload.get("name") or "")
            if not name:
                raise ValueError("name is required")
            result = delete_template(storage_root=STORAGE_ROOT, name=name)
            self._send_json({"ok": True, "result": result, "templates": list_templates(STORAGE_ROOT)})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _handle_media_probe(self) -> None:
        try:
            payload = self._read_json()
            path = str(payload.get("path") or "")
            duration = probe_media_duration(path)
            self._send_json({"ok": True, "path": path, "duration_seconds": duration})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _handle_models(self, query: str) -> None:
        values = parse_qs(query)
        endpoint = values.get("endpoint", ["http://127.0.0.1:11434/api/tags"])[0]
        try:
            endpoint = validate_local_endpoint(endpoint)
            payload = fetch_json(endpoint, "GET")
            self._send_json({"ok": True, "raw": payload})
        except (HTTPError, URLError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body or "{}")
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[studio] {self.address_string()} - {fmt % args}")


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), StudioHandler)
    print(f"TrueVision Studio: http://{host}:{port}/")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve TrueVision Studio with a local LLM CORS proxy.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
