#!/usr/bin/env python3
"""Build a TrueVision-oriented catalog from video post-training research lists."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


SECTION_HEADINGS = {
    "### Datasets": "dataset",
    "### Benchmarks": "benchmark",
    "### Conference Papers": "paper",
    "### arXiv Papers": "paper",
}

TRUEVISION_SIGNALS = {
    "motion_address": [
        "motion",
        "trajectory",
        "track",
        "tracklet",
        "pose",
        "camera",
        "control",
        "controllable",
        "interpolation",
        "in-between",
        "flow",
        "temporal",
    ],
    "avatar_identity": [
        "avatar",
        "portrait",
        "talking",
        "face",
        "human",
        "identity",
        "character",
        "listener",
        "gesture",
        "animation",
    ],
    "physics_state": [
        "physics",
        "physical",
        "geometry",
        "geometric",
        "3d",
        "4d",
        "world",
        "gravity",
        "causal",
        "robot",
        "manipulation",
    ],
    "post_training_alignment": [
        "post-training",
        "preference",
        "dpo",
        "reward",
        "alignment",
        "align",
        "rl",
        "grpo",
        "feedback",
        "distillation",
        "fine-tuning",
        "lora",
        "adaptation",
    ],
    "audio_video": [
        "audio",
        "music",
        "lip",
        "sync",
        "speech",
        "sound",
    ],
    "evaluation": [
        "bench",
        "benchmark",
        "eval",
        "test",
        "metric",
        "rewardbench",
    ],
}


@dataclass(frozen=True)
class CatalogItem:
    item_type: str
    venue: str
    title: str
    year: str
    links: list[str]
    tags: list[str]
    priority: str
    truevision_use: str
    source_file: str


def strip_markdown(value: str) -> str:
    value = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = value.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", value).strip()


def parse_links(value: str) -> list[str]:
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", value)
    return [f"{label}: {url}" for label, url in links]


def parse_name_cell(value: str) -> tuple[str, str]:
    clean = strip_markdown(value)
    parts = clean.split(" ", 1)
    if len(parts) == 2 and parts[0].isupper():
        return parts[0], parts[1].strip()
    if len(parts) == 2 and parts[0] in {"arXiv", "ICLR", "ICCV", "CVPR", "NeurIPS", "ICML", "AAAI", "ECCV", "WACV", "ACM", "ACL", "EMNLP", "PMLR", "IJCAI", "SIGGRAPH", "MICCAI"}:
        return parts[0], parts[1].strip()
    return "", clean


def score_item(title: str, item_type: str) -> tuple[list[str], str, str]:
    title_lower = title.lower()
    tags: list[str] = []
    score = 0

    for tag, needles in TRUEVISION_SIGNALS.items():
        hits = [needle for needle in needles if needle in title_lower]
        if hits:
            tags.append(tag)
            score += 2 if tag in {"motion_address", "physics_state", "post_training_alignment"} else 1

    if item_type in {"benchmark", "dataset"}:
        score += 1
        if "evaluation" not in tags and item_type == "benchmark":
            tags.append("evaluation")

    if score >= 5:
        priority = "high"
    elif score >= 3:
        priority = "medium"
    else:
        priority = "low"

    use_parts = []
    if "motion_address" in tags:
        use_parts.append("motion-address control")
    if "physics_state" in tags:
        use_parts.append("state/physics verification")
    if "avatar_identity" in tags:
        use_parts.append("avatar identity and pose behavior")
    if "post_training_alignment" in tags:
        use_parts.append("post-training or reward alignment")
    if "audio_video" in tags:
        use_parts.append("audio/video synchronization")
    if "evaluation" in tags:
        use_parts.append("benchmarking generated clips")

    truevision_use = "; ".join(use_parts) if use_parts else "background reference"
    return tags, priority, truevision_use


def parse_markdown_table(path: Path) -> list[CatalogItem]:
    items: list[CatalogItem] = []
    section = ""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line in SECTION_HEADINGS:
            section = SECTION_HEADINGS[line]
            continue
        if not section or not line.startswith("|") or "---" in line:
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        first_cell = strip_markdown(cells[0]).lower()
        if first_cell in {"title", "dataset", "benchmark"}:
            continue

        venue, title = parse_name_cell(cells[0])
        year = strip_markdown(cells[1])
        links = parse_links(cells[2])
        if not title or not year.isdigit():
            continue

        tags, priority, truevision_use = score_item(title, section)
        items.append(
            CatalogItem(
                item_type=section,
                venue=venue,
                title=title,
                year=year,
                links=links,
                tags=tags,
                priority=priority,
                truevision_use=truevision_use,
                source_file=str(path),
            )
        )

    return items


def write_jsonl(path: Path, items: list[CatalogItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.__dict__, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(path: Path, items: list[CatalogItem]) -> None:
    high = [item for item in items if item.priority == "high"]
    medium = [item for item in items if item.priority == "medium"]
    by_type = {}
    for item in items:
        by_type[item.item_type] = by_type.get(item.item_type, 0) + 1

    lines = [
        "# TrueVision Video Post-Training Research Intake",
        "",
        "This catalog converts the downloaded video post-training research list into a TrueVision-oriented training and evaluation map.",
        "",
        "## Counts",
        "",
    ]
    for item_type in sorted(by_type):
        lines.append(f"- {item_type}: {by_type[item_type]}")
    lines.extend(
        [
            f"- high priority: {len(high)}",
            f"- medium priority: {len(medium)}",
            f"- total cataloged: {len(items)}",
            "",
            "## High-Priority Starting Points",
            "",
        ]
    )

    for item in high[:40]:
        venue = f"{item.venue} " if item.venue else ""
        lines.append(f"- {venue}{item.title} ({item.year}) - {item.truevision_use}")

    lines.extend(
        [
            "",
            "## TrueVision Training Reading Order",
            "",
            "1. Motion-address control: trajectory, pose, camera, interpolation, and track-guided generation.",
            "2. Post-training alignment: preference, DPO, reward, GRPO, RL, and fine-tuning methods.",
            "3. Physics/state verification: geometry, world modeling, physical plausibility, and causal dynamics.",
            "4. Avatar-specific work: identity preservation, portrait animation, talking/listening behavior, and audio sync.",
            "5. Benchmarks: VBench-style quality checks, physics benchmarks, reward benchmarks, and sequential consistency tests.",
            "",
            "## Local Law",
            "",
            "TrueVision should use these papers as references, not authority. Authority remains observed state, motion address, temporal profile, and receipt-backed output provenance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("external_research/Awesome-Video-Generation-Post-Training"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("external_research/truevision_video_post_training_catalog.jsonl"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("external_research/TRUEVISION_VIDEO_POST_TRAINING_RESEARCH_INTAKE.md"),
    )
    args = parser.parse_args()

    readme = args.source_root / "README.md"
    if not readme.exists():
        raise FileNotFoundError(readme)

    items = parse_markdown_table(readme)
    items.sort(key=lambda item: ({"high": 0, "medium": 1, "low": 2}[item.priority], item.item_type, item.year, item.title))
    write_jsonl(args.output_jsonl, items)
    write_report(args.output_report, items)
    print(f"catalog_items={len(items)}")
    print(f"jsonl={args.output_jsonl}")
    print(f"report={args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
