"""Public dataset crawling and lightweight training utilities for TrueVision."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HF_DATASET_RE = re.compile(r"https://huggingface\.co/datasets/([^)\s]+)")
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+\-.]*")


@dataclass(frozen=True)
class DatasetTarget:
    title: str
    dataset_id: str
    priority: str
    tags: list[str]
    truevision_use: str
    source_links: list[str]


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return slug[:160] or "dataset"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_hf_dataset_id(links: Iterable[str]) -> str | None:
    for link in links:
        match = HF_DATASET_RE.search(link)
        if match:
            return match.group(1).rstrip("/")
    return None


def choose_dataset_targets(catalog_path: Path, limit: int = 10) -> list[DatasetTarget]:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    candidates: list[DatasetTarget] = []
    seen: set[str] = set()
    for row in load_jsonl(catalog_path):
        dataset_id = extract_hf_dataset_id(row.get("links", []))
        if not dataset_id or dataset_id in seen:
            continue
        if row.get("item_type") not in {"dataset", "benchmark"}:
            continue
        seen.add(dataset_id)
        candidates.append(
            DatasetTarget(
                title=row.get("title", dataset_id),
                dataset_id=dataset_id,
                priority=row.get("priority", "low"),
                tags=list(row.get("tags", [])),
                truevision_use=row.get("truevision_use", ""),
                source_links=list(row.get("links", [])),
            )
        )

    candidates.sort(
        key=lambda item: (
            priority_order.get(item.priority, 9),
            0 if any(tag in item.tags for tag in ("motion_address", "physics_state", "post_training_alignment")) else 1,
            item.title.lower(),
        )
    )
    return candidates[:limit]


def fetch_text(url: str, timeout_seconds: int = 30) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "TrueVisionResearchCrawler/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            return int(response.status), body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")


def head_content_length(url: str, timeout_seconds: int = 30) -> tuple[int, int | None]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "TrueVisionResearchCrawler/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            length = response.headers.get("Content-Length")
            return int(response.status), int(length) if length and length.isdigit() else None
    except urllib.error.HTTPError as exc:
        return int(exc.code), None


def download_binary_with_limit(url: str, output_path: Path, max_bytes: int, timeout_seconds: int = 60) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "TrueVisionResearchCrawler/1.0"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            with output_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        handle.close()
                        output_path.unlink(missing_ok=True)
                        return {"status": "skipped_over_budget_during_download", "bytes_seen": bytes_written}
                    handle.write(chunk)
        return {"status": "downloaded", "bytes_written": bytes_written, "path": str(output_path)}
    except urllib.error.HTTPError as exc:
        return {"status": "http_error", "http_status": int(exc.code), "bytes_written": bytes_written}


def crawl_huggingface_dataset(target: DatasetTarget, output_root: Path, timeout_seconds: int = 30) -> dict:
    dataset_dir = output_root / safe_slug(target.dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    encoded_id = urllib.parse.quote(target.dataset_id, safe="/")
    api_url = f"https://huggingface.co/api/datasets/{encoded_id}"
    readme_url = f"https://huggingface.co/datasets/{target.dataset_id}/raw/main/README.md"

    api_status, api_text = fetch_text(api_url, timeout_seconds)
    readme_status, readme_text = fetch_text(readme_url, timeout_seconds)

    api_path = dataset_dir / "hf_api.json"
    readme_path = dataset_dir / "README.md"
    target_path = dataset_dir / "truevision_dataset_target.json"

    api_path.write_text(api_text, encoding="utf-8")
    readme_path.write_text(readme_text, encoding="utf-8")
    target_path.write_text(json.dumps(target.__dict__, indent=2, sort_keys=True), encoding="utf-8")

    file_count = None
    gated = None
    size_categories: list[str] = []
    if api_status == 200:
        try:
            api = json.loads(api_text)
            siblings = api.get("siblings")
            if isinstance(siblings, list):
                file_count = len(siblings)
            gated = api.get("gated")
            card_data = api.get("cardData") or {}
            size_categories = list(card_data.get("dataset_info", {}).keys()) if isinstance(card_data, dict) else []
        except json.JSONDecodeError:
            pass

    return {
        "title": target.title,
        "dataset_id": target.dataset_id,
        "priority": target.priority,
        "tags": target.tags,
        "truevision_use": target.truevision_use,
        "api_url": api_url,
        "readme_url": readme_url,
        "api_status": api_status,
        "readme_status": readme_status,
        "local_dir": str(dataset_dir),
        "api_path": str(api_path),
        "readme_path": str(readme_path),
        "file_count": file_count,
        "gated": gated,
        "size_categories": size_categories,
        "download_scope": "metadata_and_dataset_card",
        "crawl_time_unix": time.time(),
    }


def crawl_targets(targets: list[DatasetTarget], output_root: Path, throttle_seconds: float = 0.25) -> list[dict]:
    receipts: list[dict] = []
    for target in targets:
        receipts.append(crawl_huggingface_dataset(target, output_root))
        if throttle_seconds:
            time.sleep(throttle_seconds)
    return receipts


def sample_dataset_payloads(
    crawl_receipt: Path,
    output_root: Path,
    max_total_bytes: int,
    max_files_per_dataset: int = 2,
) -> dict:
    receipt = json.loads(crawl_receipt.read_text(encoding="utf-8"))
    remaining = max_total_bytes
    sampled: list[dict] = []
    allowed_suffixes = {
        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".txt",
        ".md",
        ".parquet",
        ".arrow",
    }

    for dataset in receipt.get("receipts", []):
        api_path = Path(dataset["api_path"])
        dataset_id = dataset["dataset_id"]
        dataset_slug = safe_slug(dataset_id)
        try:
            api = json.loads(api_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        dataset_count = 0
        for sibling in api.get("siblings", []):
            rfilename = sibling.get("rfilename", "")
            suffix = Path(rfilename).suffix.lower()
            if suffix not in allowed_suffixes or rfilename.upper() in {"README.MD", "LICENSE"}:
                continue
            if dataset_count >= max_files_per_dataset or remaining <= 0:
                break

            quoted_file = urllib.parse.quote(rfilename, safe="/")
            url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{quoted_file}"
            head_status, length = head_content_length(url)
            record = {
                "dataset_id": dataset_id,
                "rfilename": rfilename,
                "url": url,
                "head_status": head_status,
                "content_length": length,
            }
            if head_status != 200:
                record["status"] = "skipped_head_not_ok"
            elif length is None:
                record["status"] = "skipped_unknown_size"
            elif length > remaining:
                record["status"] = "skipped_over_remaining_budget"
            else:
                out_path = output_root / dataset_slug / safe_slug(rfilename)
                result = download_binary_with_limit(url, out_path, remaining)
                record.update(result)
                if result.get("status") == "downloaded":
                    remaining -= int(result.get("bytes_written", 0))
                    dataset_count += 1
            sampled.append(record)

    return {
        "kind": "truevision_dataset_payload_sample_receipt",
        "crawl_receipt": str(crawl_receipt),
        "max_total_bytes": max_total_bytes,
        "remaining_bytes": remaining,
        "downloaded_bytes": max_total_bytes - remaining,
        "max_files_per_dataset": max_files_per_dataset,
        "sampled": sampled,
        "law": [
            "Only public Hugging Face dataset files are sampled.",
            "Large or unknown-size files are skipped unless the caller raises the budget/policy.",
            "This is a payload sampler, not a license bypasser.",
        ],
    }


def tokenize(*values: str) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(TOKEN_RE.findall(value.lower()))
    return [token for token in tokens if len(token) > 2]


def row_label(row: dict) -> int:
    tags = set(row.get("tags", []))
    priority = row.get("priority")
    if priority == "high":
        return 1
    if {"motion_address", "physics_state", "post_training_alignment"} & tags and priority == "medium":
        return 1
    return 0


def train_relevance_model(catalog_rows: list[dict], crawled_rows: list[dict] | None = None) -> dict:
    positive = Counter()
    negative = Counter()
    label_counts = Counter()

    for row in catalog_rows:
        label = row_label(row)
        label_counts[str(label)] += 1
        text_parts = [
            row.get("title", ""),
            row.get("truevision_use", ""),
            " ".join(row.get("tags", [])),
            " ".join(row.get("links", [])),
        ]
        counter = positive if label else negative
        counter.update(set(tokenize(*text_parts)))

    vocab = sorted(set(positive) | set(negative))
    pos_total = sum(positive.values()) + len(vocab)
    neg_total = sum(negative.values()) + len(vocab)
    weights = {}
    for token in vocab:
        pos_p = (positive[token] + 1) / pos_total
        neg_p = (negative[token] + 1) / neg_total
        weights[token] = math.log(pos_p / neg_p)

    crawled_scores = []
    for row in crawled_rows or []:
        text = " ".join(
            [
                row.get("title", ""),
                row.get("dataset_id", ""),
                row.get("truevision_use", ""),
                " ".join(row.get("tags", [])),
                row.get("readme_excerpt", ""),
            ]
        )
        score = sum(weights.get(token, 0.0) for token in set(tokenize(text)))
        crawled_scores.append(
            {
                "dataset_id": row.get("dataset_id"),
                "title": row.get("title"),
                "score": round(score, 4),
                "recommended_use": row.get("truevision_use"),
            }
        )

    crawled_scores.sort(key=lambda item: item["score"], reverse=True)
    return {
        "model_kind": "truevision_research_relevance_naive_bayes_log_ratio",
        "training_rows": len(catalog_rows),
        "positive_rows": label_counts["1"],
        "negative_rows": label_counts["0"],
        "feature_count": len(weights),
        "top_positive_features": sorted(weights.items(), key=lambda item: item[1], reverse=True)[:40],
        "top_negative_features": sorted(weights.items(), key=lambda item: item[1])[:20],
        "crawled_dataset_scores": crawled_scores,
        "law": [
            "This model is a research triage model, not a video generator.",
            "It may use dataset metadata, links, tags, and text signals, not only state.",
            "It does not claim ownership or license clearance for full dataset payloads.",
        ],
    }


def write_json(path: Path, payload: dict | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
