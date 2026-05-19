from __future__ import annotations

import argparse
import hashlib
import json
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
    "manifests",
    "reports",
    "receipts",
    "presets",
    "tmp",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


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
