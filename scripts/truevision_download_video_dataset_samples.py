#!/usr/bin/env python3
"""Download small public payload samples from crawled dataset manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from truevision_runtime.research_dataset_intake import sample_dataset_payloads, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crawl-receipt",
        type=Path,
        default=Path("storage/research_datasets/video_post_training_10/crawl_receipt.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("storage/research_datasets/video_post_training_10/payload_samples"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("storage/research_datasets/video_post_training_10/payload_sample_receipt.json"),
    )
    parser.add_argument("--max-mb", type=int, default=64)
    parser.add_argument("--max-files-per-dataset", type=int, default=2)
    args = parser.parse_args()

    payload = sample_dataset_payloads(
        args.crawl_receipt,
        args.output_root,
        max_total_bytes=args.max_mb * 1024 * 1024,
        max_files_per_dataset=args.max_files_per_dataset,
    )
    write_json(args.receipt, payload)
    print(f"downloaded_bytes={payload['downloaded_bytes']}")
    print(f"remaining_bytes={payload['remaining_bytes']}")
    print(f"receipt={args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
