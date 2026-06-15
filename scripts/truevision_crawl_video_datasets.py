#!/usr/bin/env python3
"""Crawl public metadata for selected video-generation datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from truevision_runtime.research_dataset_intake import choose_dataset_targets, crawl_targets, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("external_research/truevision_video_post_training_catalog.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("storage/research_datasets/video_post_training_10"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("storage/research_datasets/video_post_training_10/crawl_receipt.json"),
    )
    args = parser.parse_args()

    targets = choose_dataset_targets(args.catalog, limit=args.limit)
    receipts = crawl_targets(targets, args.output_root)
    payload = {
        "kind": "truevision_public_dataset_metadata_crawl_receipt",
        "catalog": str(args.catalog),
        "limit": args.limit,
        "target_count": len(targets),
        "download_scope": "metadata_and_dataset_cards_only",
        "full_payload_note": "Use dataset licenses and storage budget checks before downloading full video payloads.",
        "targets": [target.__dict__ for target in targets],
        "receipts": receipts,
    }
    write_json(args.receipt, payload)
    print(f"targets={len(targets)}")
    print(f"receipt={args.receipt}")
    print(f"output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
