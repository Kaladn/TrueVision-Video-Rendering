from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "ui" / "truevision_state_media_studio.html"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MODEL = "qwen3-coder:30b"


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
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/local-llm/chat":
            self.send_error(404, "Not found")
            return
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
