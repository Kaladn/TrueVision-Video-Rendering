#!/usr/bin/env python3
"""Train a small TrueVision research relevance model from catalog + crawled metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from truevision_runtime.research_dataset_intake import load_jsonl, train_relevance_model, write_json


def load_crawled_receipts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("receipts", []))
    for row in rows:
        readme_path = Path(row.get("readme_path", ""))
        if readme_path.exists():
            text = readme_path.read_text(encoding="utf-8", errors="replace")
            row["readme_excerpt"] = text[:12000]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("external_research/truevision_video_post_training_catalog.jsonl"),
    )
    parser.add_argument(
        "--crawl-receipt",
        type=Path,
        default=Path("storage/research_datasets/video_post_training_10/crawl_receipt.json"),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("storage/research_datasets/video_post_training_10/truevision_research_relevance_model.json"),
    )
    args = parser.parse_args()

    catalog_rows = load_jsonl(args.catalog)
    crawled_rows = load_crawled_receipts(args.crawl_receipt)
    model = train_relevance_model(catalog_rows, crawled_rows)
    model["catalog"] = str(args.catalog)
    model["crawl_receipt"] = str(args.crawl_receipt)
    write_json(args.model_out, model)

    print(f"training_rows={model['training_rows']}")
    print(f"positive_rows={model['positive_rows']}")
    print(f"feature_count={model['feature_count']}")
    print(f"model={args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
