from __future__ import annotations

import base64
import json
import os
import socket
import struct
import time
import urllib.request
from typing import Any
from urllib.parse import urlparse


def build_video_play_expression(start_seconds: float) -> str:
    start = max(0.0, float(start_seconds))
    return f"""
(async () => {{
  const v = document.querySelector('video');
  if (!v) return {{ ok: false, reason: 'missing_video', href: location.href, title: document.title }};
  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  if (v.readyState < 1) {{
    await new Promise(resolve => {{
      const done = () => resolve();
      v.addEventListener('loadedmetadata', done, {{ once: true }});
      setTimeout(done, 10000);
    }});
  }}
  v.muted = true;
  v.currentTime = {start:.3f};
  const deadline = Date.now() + 25000;
  while (Date.now() < deadline) {{
    if (v.readyState >= 2 && Math.abs(v.currentTime - {start:.3f}) <= 3.0) break;
    await new Promise(resolve => {{
      const done = () => resolve();
      v.addEventListener('seeked', done, {{ once: true }});
      v.addEventListener('canplay', done, {{ once: true }});
      v.addEventListener('canplaythrough', done, {{ once: true }});
      setTimeout(done, 1000);
    }});
  }}
  let playError = null;
  for (let attempt = 0; attempt < 3; attempt++) {{
    try {{
      await v.play();
      playError = null;
      break;
    }} catch (error) {{
      playError = String(error);
      await sleep(1500);
    }}
  }}
  await sleep(750);
  const targetReached = Math.abs(v.currentTime - {start:.3f}) <= 5.0 || v.currentTime >= {start:.3f};
  const playable = v.readyState >= 2 && !v.paused && targetReached;
  return {{ ok: playable, reason: playable ? null : (playError || 'target_not_playable'), href: location.href, title: document.title, currentTime: v.currentTime, paused: v.paused, readyState: v.readyState, duration: v.duration }};
}})()
""".strip()


def build_video_state_expression() -> str:
    return """
(() => {
  const v = document.querySelector('video');
  if (!v) return { ok: false, reason: 'missing_video', href: location.href, title: document.title };
  return { ok: true, href: location.href, title: document.title, currentTime: v.currentTime, paused: v.paused, readyState: v.readyState, duration: v.duration };
})()
""".strip()


class DevToolsClient:
    def __init__(self, websocket_url: str, *, timeout_seconds: float = 5.0):
        parsed = urlparse(websocket_url)
        if parsed.scheme != "ws":
            raise ValueError("only ws:// DevTools endpoints are supported")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.path = parsed.path
        if parsed.query:
            self.path += "?" + parsed.query
        self.timeout_seconds = timeout_seconds
        self._id = 0
        self._socket = socket.create_connection((self.host, self.port), timeout=timeout_seconds)
        self._socket.settimeout(timeout_seconds)
        self._handshake()

    def close(self) -> None:
        try:
            self._socket.close()
        except Exception:
            pass

    def _handshake(self) -> None:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._socket.sendall(request.encode("ascii"))
        response = self._socket.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"DevTools websocket handshake failed: {response[:120]!r}")

    def command(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        message = {"id": self._id, "method": method, "params": params or {}}
        self._send_text(json.dumps(message, separators=(",", ":")))
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            payload = self._recv_text()
            if not payload:
                continue
            event = json.loads(payload)
            if event.get("id") == self._id:
                if "error" in event:
                    raise RuntimeError(json.dumps(event["error"], allow_nan=False))
                return event.get("result") or {}
        raise TimeoutError(f"DevTools command timed out: {method}")

    def evaluate(self, expression: str, *, await_promise: bool = False) -> Any:
        result = self.command(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": bool(await_promise),
                "returnByValue": True,
            },
        )
        value = ((result.get("result") or {}).get("value"))
        return value

    def _send_text(self, text: str) -> None:
        payload = text.encode("utf-8")
        header = bytearray()
        header.append(0x81)
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask = os.urandom(4)
        header.extend(mask)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self._socket.sendall(bytes(header) + masked)

    def _recv_exact(self, length: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            chunk = self._socket.recv(length - len(chunks))
            if not chunk:
                raise ConnectionError("DevTools websocket closed")
            chunks.extend(chunk)
        return bytes(chunks)

    def _recv_text(self) -> str:
        first = self._recv_exact(2)
        opcode = first[0] & 0x0F
        masked = bool(first[1] & 0x80)
        length = first[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else b""
        payload = self._recv_exact(length)
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        if opcode == 0x8:
            raise ConnectionError("DevTools websocket closed")
        if opcode != 0x1:
            return ""
        return payload.decode("utf-8", errors="replace")


def select_devtools_page(pages: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        page
        for page in pages
        if page.get("type") == "page" and page.get("webSocketDebuggerUrl")
    ]
    if not candidates:
        return None
    for page in candidates:
        url = str(page.get("url") or "")
        if "youtube.com/watch" in url or "youtube.com/shorts" in url:
            return page
    for page in candidates:
        url = str(page.get("url") or "")
        if url.startswith("http://") or url.startswith("https://"):
            if "chrome-extension" not in url and "dc-chrome-extension" not in url:
                return page
    return candidates[0]


def wait_for_devtools_page(port: int, *, timeout_seconds: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1.5) as response:
                pages = json.loads(response.read().decode("utf-8", errors="replace"))
            page = select_devtools_page(pages)
            if page is not None:
                return page
        except Exception as exc:  # noqa: BLE001 - retry until browser exposes the endpoint.
            last_error = exc
        time.sleep(0.25)
    raise TimeoutError(f"DevTools page was not available on port {port}: {last_error}")


def command_on_first_page(port: int, method: str, params: dict[str, Any] | None = None, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
    page = wait_for_devtools_page(port, timeout_seconds=timeout_seconds)
    client = DevToolsClient(page["webSocketDebuggerUrl"], timeout_seconds=timeout_seconds)
    try:
        return client.command(method, params or {})
    finally:
        client.close()


def navigate_first_page(port: int, url: str, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
    return command_on_first_page(port, "Page.navigate", {"url": url}, timeout_seconds=timeout_seconds)


def evaluate_on_first_page(port: int, expression: str, *, await_promise: bool = False, timeout_seconds: float = 10.0) -> Any:
    page = wait_for_devtools_page(port, timeout_seconds=timeout_seconds)
    client = DevToolsClient(page["webSocketDebuggerUrl"], timeout_seconds=timeout_seconds)
    try:
        return client.evaluate(expression, await_promise=await_promise)
    finally:
        client.close()
