#!/usr/bin/env python3
"""Check a pasted video post-training list against the TrueVision catalog."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


KNOWN_VENUES = {
    "AAAI",
    "ACL",
    "ACM MM",
    "COLM",
    "CVPR",
    "ECCV",
    "EMNLP",
    "ICCV",
    "ICLR",
    "ICML",
    "IJCAI",
    "MICCAI",
    "NeurIPS",
    "SIGGRAPH Asia",
    "WACV",
    "arXiv",
}


@dataclass(frozen=True)
class PastedItem:
    venue: str
    title: str
    year: str


def normalize_title(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_pasted_items(path: Path) -> list[PastedItem]:
    items: list[PastedItem] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line or line in {"Title Year Links", "arXiv Papers", "Conference Papers"}:
            continue

        match = re.match(r"^(?P<head>.+?)\s+(?P<year>20\d{2})\s+(?P<links>Paper.*)$", line)
        if not match:
            continue

        head = match.group("head").strip()
        year = match.group("year")
        venue = ""
        title = head
        for candidate in sorted(KNOWN_VENUES, key=len, reverse=True):
            prefix = f"{candidate} "
            if head.startswith(prefix):
                venue = candidate
                title = head[len(prefix) :].strip()
                break

        if title:
            items.append(PastedItem(venue=venue, title=title, year=year))
    return items


def load_catalog_titles(path: Path) -> set[str]:
    titles: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            title = row.get("title", "")
            if title:
                titles.add(normalize_title(title))
    return titles


def is_covered(item: PastedItem, catalog_titles: set[str]) -> bool:
    title = normalize_title(item.title)
    if title in catalog_titles:
        return True
    return any(title in catalog_title or catalog_title in title for catalog_title in catalog_titles)


def write_report(path: Path, pasted: list[PastedItem], missing: list[PastedItem]) -> None:
    lines = [
        "# Pasted Video Post-Training Coverage",
        "",
        "This report checks the attached pasted research list against the generated TrueVision video post-training catalog.",
        "",
        f"- pasted rows parsed: {len(pasted)}",
        f"- rows already covered: {len(pasted) - len(missing)}",
        f"- rows not matched: {len(missing)}",
        "",
    ]
    if missing:
        lines.extend(["## Not Matched", ""])
        for item in missing:
            venue = f"{item.venue} " if item.venue else ""
            lines.append(f"- {venue}{item.title} ({item.year})")
    else:
        lines.append("All parsed pasted rows are covered by the current catalog.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pasted_text", type=Path)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("external_research/truevision_video_post_training_catalog.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("external_research/TRUEVISION_VIDEO_POST_TRAINING_PASTED_COVERAGE.md"),
    )
    args = parser.parse_args()

    pasted = parse_pasted_items(args.pasted_text)
    catalog_titles = load_catalog_titles(args.catalog)
    missing = [item for item in pasted if not is_covered(item, catalog_titles)]
    write_report(args.report, pasted, missing)

    print(f"pasted_rows={len(pasted)}")
    print(f"covered_rows={len(pasted) - len(missing)}")
    print(f"missing_rows={len(missing)}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
