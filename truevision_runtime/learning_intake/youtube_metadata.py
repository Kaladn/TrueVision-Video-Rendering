from __future__ import annotations

import html
import json
import re
import urllib.request
from typing import Any

from truevision_runtime.learning_intake.source_surface import canonicalize_approved_source_url


def _clean_title(value: str) -> str:
    title = html.unescape(value or "").strip()
    if title.endswith(" - YouTube"):
        title = title[: -len(" - YouTube")].strip()
    return title


def extract_youtube_metadata_from_html(html_text: str, *, fallback_title: str = "") -> dict[str, Any]:
    duration_match = re.search(r'"lengthSeconds"\s*:\s*"?([0-9]+)"?', html_text)
    if not duration_match:
        raise ValueError("YouTube duration was not detected")
    duration = float(duration_match.group(1))
    title = ""
    json_title_match = re.search(r'"title"\s*:\s*"([^"]+)"', html_text)
    if json_title_match:
        try:
            title = json.loads(f'"{json_title_match.group(1)}"')
        except json.JSONDecodeError:
            title = json_title_match.group(1)
    if not title:
        html_title_match = re.search(r"<title>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
        if html_title_match:
            title = html_title_match.group(1)
    title = _clean_title(title or fallback_title)
    if not title:
        raise ValueError("YouTube title was not detected")
    return {
        "video_title": title,
        "duration_seconds": duration,
    }


def fetch_youtube_metadata(source_url: str, *, timeout_seconds: float = 15.0) -> dict[str, Any]:
    canonical = canonicalize_approved_source_url(source_url)
    request = urllib.request.Request(
        canonical["address_bar_url"],
        headers={
            "User-Agent": "Mozilla/5.0 TrueVisionLearningIntake/1.0",
            "Accept-Language": "en-US,en;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8", errors="replace")
    metadata = extract_youtube_metadata_from_html(body, fallback_title=canonical["video_id"])
    metadata.update(
        {
            "source_url": source_url,
            "address_bar_url": canonical["address_bar_url"],
            "video_id": canonical["video_id"],
        }
    )
    return metadata
