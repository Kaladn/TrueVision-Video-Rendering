from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from truevision_runtime.document_state import record_document_state_movie


SCHEMA_VERSION = "truevision_markdown_frame_set@1"
DEFAULT_FRAME_SIZE = (1600, 1000)
DEFAULT_GRID_SHAPE = (100, 160)

BG = (12, 18, 25)
INK = (235, 241, 244)
MUTED = (156, 169, 177)
BLUE = (32, 83, 117)
TEAL = (33, 126, 111)
GOLD = (183, 132, 47)
RED = (148, 63, 69)
CARD = (18, 26, 34)
LINE = (86, 105, 116)
DARK = (22, 28, 34)
PANEL = (15, 22, 30)
PANEL_ALT = (18, 27, 36)
PALETTE = [BLUE, TEAL, GOLD, RED, (89, 78, 133), (75, 119, 64)]


def render_markdown_frame_set(
    *,
    source_markdown: str | Path,
    output_root: str | Path,
    run_id: str | None = None,
    title: str | None = None,
    frames_per_page: int = 1,
    fps: float = 1.0,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
    frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
) -> dict[str, Any]:
    source_path = Path(source_markdown).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    frame_dir = output_path / "frames"
    state_root = output_path / "truevision_state_movie"
    frame_dir.mkdir(parents=True, exist_ok=True)

    markdown = source_path.read_text(encoding="utf-8")
    doc_title = title or parse_markdown_title(markdown) or source_path.stem.replace("_", " ").replace("-", " ").title()
    sections = parse_markdown_sections(markdown) or [{"title": doc_title, "body": markdown}]
    actual_run_id = slug(run_id or source_path.stem)

    images: list[Image.Image] = []
    frame_rows: list[dict[str, Any]] = []
    cover = draw_cover_frame(doc_title, source_path, len(sections), frame_size=frame_size)
    cover_path = frame_dir / "00_cover.png"
    cover.save(cover_path)
    images.append(cover)
    frame_rows.append(_frame_row(0, "Cover", cover_path))

    total = len(sections) + 1
    for index, section in enumerate(sections, start=1):
        image = draw_section_frame(
            section,
            source_path=source_path,
            frame_index=index,
            frame_total=total,
            frame_size=frame_size,
        )
        path = frame_dir / f"{index:02d}_{slug(section['title'])[:64]}.png"
        image.save(path)
        images.append(image)
        frame_rows.append(_frame_row(index, section["title"], path))

    pdf_path = output_path / f"{actual_run_id}_frames.pdf"
    images[0].save(pdf_path, save_all=True, append_images=images[1:], resolution=150.0)

    page_frames = [np.asarray(image.convert("RGB")) for image in images]
    state_movie = record_document_state_movie(
        source_id=str(source_path),
        page_frames=page_frames,
        output_root=state_root,
        run_id=actual_run_id,
        frames_per_page=frames_per_page,
        fps=fps,
        grid_shape=grid_shape,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "run_id": actual_run_id,
        "title": doc_title,
        "source_markdown": str(source_path),
        "source_sha256": sha256_file(source_path),
        "output_root": str(output_path),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "frame_count": len(frame_rows),
        "frames": frame_rows,
        "pdf": {"path": str(pdf_path), "sha256": sha256_file(pdf_path)},
        "truevision_document_state_movie": state_movie,
        "boundary": {
            "generated_media_is_presentation": True,
            "source_markdown_remains_receipt": True,
            "html_css_js_used": False,
            "raw_frames_saved": False,
            "generated_media_is_evidence": False,
        },
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return {
        "run_id": actual_run_id,
        "output_root": str(output_path),
        "frames_dir": str(frame_dir),
        "frame_count": len(frame_rows),
        "pdf_path": str(pdf_path),
        "manifest_json": str(manifest_path),
        "truevision_document_state_movie": state_movie,
    }


def parse_markdown_title(markdown: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_markdown_sections(markdown: str) -> list[dict[str, str]]:
    matches = list(re.finditer(r"^##\s+(.+)$", markdown, re.MULTILINE))
    if not matches:
        body = re.sub(r"^#\s+.+$", "", markdown, count=1, flags=re.MULTILINE).strip()
        title = parse_markdown_title(markdown) or "Document"
        return [{"title": title, "body": body}] if body else []
    sections: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections.append({"title": match.group(1).strip(), "body": markdown[start:end].strip()})
    return sections


def draw_cover_frame(title: str, source_path: Path, section_count: int, *, frame_size: tuple[int, int]) -> Image.Image:
    width, height = frame_size
    fonts = FontSet.for_size(frame_size)
    image = Image.new("RGB", frame_size, DARK)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, height - int(height * 0.22), width, height), fill=(37, 47, 54))
    stripe_y = int(height * 0.13)
    stripe_h = max(6, height // 100)
    for index, color in enumerate(PALETTE):
        y = stripe_y + index * stripe_h * 2
        draw.rectangle((0, y, width, y + stripe_h), fill=color)
    draw.text((width * 0.06, height * 0.17), title, font=fonts.title, fill=(255, 255, 250))
    draw.text((width * 0.064, height * 0.30), "TrueVision markdown frame set", font=fonts.body, fill=(214, 224, 225))
    pills = ["source markdown receipt", "visual page frames", "PDF surface", "document state movie", "no HTML/CSS/JS", "pixels last"]
    x = int(width * 0.06)
    y = int(height * 0.42)
    pill_w = int(width * 0.28)
    pill_h = int(height * 0.058)
    for index, pill in enumerate(pills):
        column = index // 3
        row = index % 3
        px = x + column * int(width * 0.32)
        py = y + row * int(height * 0.075)
        rounded(draw, (px, py, px + pill_w, py + pill_h), 8, fill=(50, 62, 71), outline=(89, 101, 109))
        draw.rectangle((px, py, px + 10, py + pill_h), fill=PALETTE[index % len(PALETTE)])
        draw.text((px + 28, py + pill_h * 0.24), pill, font=fonts.body_bold, fill=(252, 250, 244))
    receipt = f"{section_count} source sections -> {section_count + 1} visual frames"
    box = (x, height - int(height * 0.145), x + int(width * 0.42), height - int(height * 0.085))
    rounded(draw, box, 8, fill=(245, 244, 238))
    draw.text((box[0] + 24, box[1] + 16), receipt, font=fonts.body_bold, fill=DARK)
    draw.text((x, height - int(height * 0.05)), f"Source: {source_path.name}", font=fonts.small, fill=(196, 203, 204))
    return image


def draw_section_frame(
    section: dict[str, str],
    *,
    source_path: Path,
    frame_index: int,
    frame_total: int,
    frame_size: tuple[int, int],
) -> Image.Image:
    width, height = frame_size
    fonts = FontSet.for_size(frame_size)
    image = Image.new("RGB", frame_size, BG)
    draw = ImageDraw.Draw(image)
    draw_header(draw, section["title"], f"Generated from {source_path.name}", frame_index, frame_total, fonts, frame_size)
    body = section["body"]
    mermaid = extract_mermaid(body)
    bullets = extract_bullets(body)
    code_blocks = extract_non_mermaid_code_blocks(body)
    text_summary = extract_plain_text(body)

    margin = int(width * 0.045)
    top = int(height * 0.154)
    bottom = int(height * 0.806)
    right = width - margin
    side_w = int(width * 0.30)
    gap = int(width * 0.02)
    main_box = (margin, top, right - side_w - gap, bottom)
    side_box = (right - side_w, top, right, top + int((bottom - top) * 0.66))
    lower_side_box = (right - side_w, side_box[3] + int(height * 0.032), right, bottom)

    if mermaid:
        draw_graph(draw, mermaid, main_box, fonts)
    else:
        draw_text_panel(draw, "Document Text", text_summary or ["No body text."], main_box, fonts)
    draw_bullet_panel(draw, "Highlights", bullets or text_summary[:5] or ["Source section retained as a visual frame."], side_box, fonts)
    if code_blocks:
        draw_code_panel(draw, "State / Code", code_blocks, lower_side_box, fonts)
    else:
        draw_bullet_panel(
            draw,
            "Boundary",
            ["Frame surface is generated presentation.", "Source markdown and manifest remain the receipt."],
            lower_side_box,
            fonts,
            accent=TEAL,
        )
    draw_footer(draw, section["title"], fonts, frame_size)
    return image


def extract_mermaid(body: str) -> str:
    match = re.search(r"```mermaid\s*\n(.*?)\n```", body, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_non_mermaid_code_blocks(body: str) -> list[str]:
    blocks: list[str] = []
    for match in re.finditer(r"```([a-zA-Z0-9_-]*)\s*\n(.*?)\n```", body, re.DOTALL):
        if match.group(1).strip().lower() != "mermaid":
            blocks.append(match.group(2).strip())
    return blocks


def extract_bullets(body: str) -> list[str]:
    text = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    return [line.strip()[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")][:8]


def extract_plain_text(body: str) -> list[str]:
    text = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and not line.startswith("- "):
            lines.append(line)
    return lines[:8]


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str, index: int, total: int, fonts: "FontSet", frame_size: tuple[int, int]) -> None:
    width, height = frame_size
    header_h = int(height * 0.118)
    draw.rectangle((0, 0, width, header_h), fill=DARK)
    draw.rectangle((0, header_h - 6, width, header_h), fill=GOLD)
    draw.text((width * 0.045, height * 0.034), title, font=fonts.heading, fill=(252, 250, 244))
    draw.text((width * 0.046, height * 0.088), subtitle, font=fonts.small, fill=(203, 213, 216))
    bx = width - int(width * 0.12)
    by = int(height * 0.036)
    rounded(draw, (bx, by, bx + int(width * 0.073), by + int(height * 0.04)), 8, fill=(44, 55, 65), outline=(91, 101, 111))
    draw.text((bx + int(width * 0.018), by + int(height * 0.009)), f"{index:02d}/{total:02d}", font=fonts.small_bold, fill=(245, 244, 238))


def draw_footer(draw: ImageDraw.ImageDraw, section_name: str, fonts: "FontSet", frame_size: tuple[int, int]) -> None:
    width, height = frame_size
    margin = int(width * 0.045)
    y = height - int(height * 0.058)
    draw.line((margin, y, width - margin, y), fill=(51, 66, 76), width=2)
    draw.text((margin, y + 14), "TrueVision markdown frame set", font=fonts.small, fill=MUTED)
    text_w = text_width(draw, section_name, fonts.small)
    draw.text((width - margin - text_w, y + 14), section_name, font=fonts.small, fill=MUTED)


def draw_graph(draw: ImageDraw.ImageDraw, code: str, box: tuple[int, int, int, int], fonts: "FontSet") -> None:
    x0, y0, x1, y1 = box
    nodes, edges = parse_mermaid_graph(code)
    rounded(draw, box, 10, fill=PANEL, outline=(58, 73, 84), width=2)
    if not nodes:
        draw.text((x0 + 24, y0 + 24), "No supported graph structure found.", font=fonts.body, fill=MUTED)
        return
    layout = graph_layout_boxes(nodes, edges, box, frame_scale=fonts.scale)
    for left, right in edges:
        if left not in layout or right not in layout:
            continue
        draw_connector(draw, layout[left], layout[right])
    for index, node_id in enumerate(layout):
        item = layout[node_id]
        fill = PALETTE[index % len(PALETTE)]
        rounded(draw, (item["x0"], item["y0"], item["x1"], item["y1"]), 8, fill=(18, 26, 34), outline=fill, width=3)
        ty = item["y0"] + (item["height"] - len(item["lines"]) * int(fonts.tiny_size * 1.25)) / 2
        for line in item["lines"]:
            label_font = fonts.tiny_bold if item["compact"] else fonts.small_bold
            tw = text_width(draw, line, label_font)
            draw.text((item["cx"] - tw / 2, ty), line, font=label_font, fill=(244, 248, 250))
            ty += int(fonts.tiny_size * 1.35)


def draw_connector(draw: ImageDraw.ImageDraw, source: dict[str, Any], target: dict[str, Any]) -> None:
    start = (source["x1"] + 8, source["cy"])
    end = (target["x0"] - 8, target["cy"])
    if end[0] <= start[0]:
        start = (source["cx"], source["y1"] + 8)
        end = (target["cx"], target["y0"] - 8)
    mid_x = (start[0] + end[0]) / 2
    points = [start, (mid_x, start[1]), (mid_x, end[1]), end]
    draw.line(points, fill=LINE, width=2, joint="curve")
    angle = math.atan2(end[1] - points[-2][1], end[0] - points[-2][0])
    arrow = [
        end,
        (end[0] - 10 * math.cos(angle - 0.45), end[1] - 10 * math.sin(angle - 0.45)),
        (end[0] - 10 * math.cos(angle + 0.45), end[1] - 10 * math.sin(angle + 0.45)),
    ]
    draw.polygon(arrow, fill=LINE)


def display_label_for_graph_node(label: str) -> str:
    exact = {
        "POST /api/lexicon/intake/map": "POST intake/map",
        "POST /api/lexicon/mapping/run": "POST mapping/run",
        "preview_document_intake": "Preview",
        "build_intake_mapping facade": "Intake facade",
        "map_intake_content_to_user_counts_native": "Native inline map",
        "build_observed_map": "Observed map",
        "observed/symbolic map artifacts": "Observed map",
        "store.py: LexiconStore facade": "LexiconStore",
        "store_runtime mixins": "Runtime",
        "store_modules powers": "Powers",
        "store_support.py shared imports/helpers": "Support hub",
        "intake.py reserved/light wrapper": "intake reserved",
        "visual_flat.py": "visual flat",
        "State/user/user_lexicon/anchors.json": "User lexicon",
        "Canonical/*.json": "Canonical JSON",
        "_all_known_anchors": "Known anchors",
        "_canonical_symbol_by_anchor": "Canonical symbol",
        "_symbol_authority_by_anchor": "Authority lookup",
        "authority_snapshot.json": "Authority snapshot",
        "ConversationEngine": "Conversation Engine",
    }
    if label in exact:
        return exact[label]
    if label.startswith("POST /api/lexicon/"):
        return "POST " + label.removeprefix("POST /api/lexicon/")
    return label


def needs_compact_graph_label(label: str) -> bool:
    return len(label) > 12 or "/" in label or "_" in label


def parse_mermaid_graph(code: str) -> tuple[dict[str, str], list[tuple[str, str]]]:
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    token_re = re.compile(r'([A-Za-z0-9_]+)(?:\["([^"]+)"\])?')
    for raw in code.splitlines():
        line = raw.strip()
        if not line or line.startswith("flowchart") or line.startswith("graph"):
            continue
        parts = [part.strip() for part in re.split(r"-->|---", line)]
        ids: list[str] = []
        for part in parts:
            match = token_re.match(part)
            if not match:
                continue
            node_id = match.group(1)
            nodes[node_id] = match.group(2) or node_id
            ids.append(node_id)
        for left, right in zip(ids, ids[1:]):
            edges.append((left, right))
    return nodes, edges


def graph_positions(nodes: dict[str, str], edges: list[tuple[str, str]], box: tuple[int, int, int, int]) -> dict[str, tuple[float, float]]:
    layout = graph_layout_boxes(nodes, edges, box, frame_scale=1.0)
    return {node_id: (item["cx"], item["cy"]) for node_id, item in layout.items()}


def graph_layout_boxes(
    nodes: dict[str, str],
    edges: list[tuple[str, str]],
    box: tuple[int, int, int, int],
    *,
    frame_scale: float,
) -> dict[str, dict[str, Any]]:
    x0, y0, x1, y1 = box
    incoming = {node_id: 0 for node_id in nodes}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for left, right in edges:
        if right in incoming:
            incoming[right] += 1
        outgoing.setdefault(left, []).append(right)
    depths = {node_id: 0 for node_id in nodes}
    queue = [node_id for node_id, count in incoming.items() if count == 0] or list(nodes)[:1]
    seen: set[str] = set()
    while queue:
        node_id = queue.pop(0)
        seen.add(node_id)
        for child in outgoing.get(node_id, []):
            depths[child] = max(depths.get(child, 0), depths[node_id] + 1)
            if child not in seen:
                queue.append(child)
    max_depth = max(depths.values()) if depths else 0
    columns: list[list[str]] = [[] for _ in range(max_depth + 1)]
    for node_id, depth in sorted(depths.items(), key=lambda item: (item[1], item[0])):
        columns[depth].append(node_id)
    columns = split_dense_columns(columns, max_nodes_per_column=7)
    layout: dict[str, dict[str, Any]] = {}
    pad_x = 72 * frame_scale
    pad_y = 72 * frame_scale
    usable_w = x1 - x0 - pad_x * 2
    usable_h = y1 - y0 - pad_y * 2
    col_w = usable_w / max(1, len(columns))
    for col_index, column in enumerate(columns):
        if not column:
            continue
        dense = len(column) > 6
        node_w = min((182 if dense else 210) * frame_scale, max((142 if dense else 128) * frame_scale, col_w * (0.52 if dense else 0.58)))
        row_gap = usable_h / max(1, len(column))
        stagger = (row_gap * 0.28) if col_index % 2 else 0
        for row_index, node_id in enumerate(column):
            label = display_label_for_graph_node(nodes[node_id])
            compact = dense or needs_compact_graph_label(label)
            char_limit = max(10, int(node_w / (8.5 * frame_scale)))
            lines = wrap_label_to_chars(label, char_limit)[:3]
            line_height = (16 if dense else 20) * frame_scale if compact else 22 * frame_scale
            node_h = max((42 if dense else 58) * frame_scale, (14 if dense else 22) * frame_scale + len(lines) * line_height)
            fish_scale = ((row_index % 2) - 0.5) * min(44 * frame_scale, col_w * 0.18) if dense else 0
            cx = x0 + pad_x + col_w * col_index + col_w / 2 + fish_scale
            cy = y0 + pad_y + row_gap * row_index + row_gap / 2 + stagger
            cy = min(y1 - pad_y - node_h / 2, max(y0 + pad_y + node_h / 2, cy))
            layout[node_id] = {
                "cx": cx,
                "cy": cy,
                "x0": cx - node_w / 2,
                "y0": cy - node_h / 2,
                "x1": cx + node_w / 2,
                "y1": cy + node_h / 2,
                "width": node_w,
                "height": node_h,
                "lines": lines,
                "compact": compact,
            }
    return avoid_layout_collisions(layout, (x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y), min_gap=18 * frame_scale)


def split_dense_columns(columns: list[list[str]], *, max_nodes_per_column: int) -> list[list[str]]:
    expanded: list[list[str]] = []
    for column in columns:
        if len(column) <= max_nodes_per_column:
            expanded.append(column)
            continue
        lane_count = math.ceil(len(column) / max_nodes_per_column)
        lanes = [[] for _ in range(lane_count)]
        for index, node_id in enumerate(column):
            lanes[index % lane_count].append(node_id)
        expanded.extend(lanes)
    return expanded


def wrap_label_to_chars(label: str, max_chars: int) -> list[str]:
    words = str(label).split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(word) > max_chars and not current:
            lines.append(word[:max_chars])
            remainder = word[max_chars:]
            if remainder:
                current = remainder
            continue
        trial = word if not current else current + " " + word
        if len(trial) <= max_chars:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [label]


def avoid_layout_collisions(
    layout: dict[str, dict[str, Any]],
    bounds: tuple[float, float, float, float],
    *,
    min_gap: float,
) -> dict[str, dict[str, Any]]:
    _left, top, _right, bottom = bounds
    ordered = sorted(layout, key=lambda node_id: (layout[node_id]["x0"], layout[node_id]["y0"]))
    for _pass in range(6):
        moved = False
        for index, node_id in enumerate(ordered):
            node = layout[node_id]
            for other_id in ordered[:index]:
                other = layout[other_id]
                if boxes_overlap(node, other, min_gap):
                    shift = other["y1"] + min_gap - node["y0"]
                    move_box(node, 0, shift)
                    moved = True
            if node["y1"] > bottom:
                move_box(node, 0, bottom - node["y1"])
            if node["y0"] < top:
                move_box(node, 0, top - node["y0"])
        if not moved:
            break
    return layout


def boxes_overlap(left: dict[str, Any], right: dict[str, Any], gap: float) -> bool:
    return not (
        left["x1"] + gap <= right["x0"]
        or right["x1"] + gap <= left["x0"]
        or left["y1"] + gap <= right["y0"]
        or right["y1"] + gap <= left["y0"]
    )


def move_box(box: dict[str, Any], dx: float, dy: float) -> None:
    for key in ("x0", "x1", "cx"):
        box[key] += dx
    for key in ("y0", "y1", "cy"):
        box[key] += dy


def draw_bullet_panel(
    draw: ImageDraw.ImageDraw,
    title: str,
    bullets: list[str],
    box: tuple[int, int, int, int],
    fonts: "FontSet",
    *,
    accent: tuple[int, int, int] = BLUE,
) -> None:
    x0, y0, x1, y1 = box
    rounded(draw, box, 10, fill=PANEL_ALT, outline=(60, 76, 88), width=2)
    draw.text((x0 + 24, y0 + 22), title, font=fonts.subheading, fill=accent)
    y = y0 + 68
    line_step = int(fonts.body_size * 1.32)
    for bullet in bullets:
        if y + line_step > y1 - 28:
            break
        draw.ellipse((x0 + 28, y + 10, x0 + 38, y + 20), fill=GOLD)
        for line in wrap_text(draw, bullet, fonts.body, x1 - x0 - 76)[:3]:
            if y + line_step > y1 - 24:
                return
            draw.text((x0 + 54, y), line, font=fonts.body, fill=INK)
            y += line_step
        y += 8
        if y + line_step > y1 - 28:
            break


def draw_text_panel(draw: ImageDraw.ImageDraw, title: str, lines: list[str], box: tuple[int, int, int, int], fonts: "FontSet") -> None:
    x0, y0, x1, y1 = box
    rounded(draw, box, 10, fill=PANEL, outline=(58, 73, 84), width=2)
    draw.text((x0 + 28, y0 + 24), title, font=fonts.subheading, fill=BLUE)
    y = y0 + 78
    for paragraph in lines:
        for line in wrap_text(draw, paragraph, fonts.body, x1 - x0 - 56):
            draw.text((x0 + 28, y), line, font=fonts.body, fill=INK)
            y += int(fonts.body_size * 1.35)
        y += 12
        if y > y1 - 40:
            break


def draw_code_panel(draw: ImageDraw.ImageDraw, title: str, blocks: list[str], box: tuple[int, int, int, int], fonts: "FontSet") -> None:
    x0, y0, x1, y1 = box
    rounded(draw, box, 10, fill=(18, 25, 32), outline=(73, 91, 104), width=2)
    draw.text((x0 + 24, y0 + 18), title, font=fonts.small_bold, fill=(242, 225, 166))
    y = y0 + 54
    panel_lines = prepare_code_panel_lines(blocks)
    max_chars = max(30, int((x1 - x0) / max(8, fonts.mono_size * 0.58)))
    for raw in panel_lines:
        for line in wrap_monospace_line(raw, max_chars=max_chars):
            if y + int(fonts.mono_size * 1.35) > y1 - 24:
                return
            draw.text((x0 + 24, y), line, font=fonts.mono, fill=(231, 235, 230))
            y += int(fonts.mono_size * 1.35)
        if not raw.strip():
            y += int(fonts.mono_size * 0.6)


def prepare_code_panel_lines(blocks: list[str]) -> list[str]:
    raw_lines = "\n\n".join(block.strip() for block in blocks[:2]).splitlines()
    expanded = expand_code_panel_lines(raw_lines)
    if "Not reached by:" not in expanded:
        return expanded

    path_lines: list[str] = []
    route_lines: list[str] = []
    detail_lines: list[str] = []
    for line in expanded:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("src/") and len(path_lines) < 1:
            path_lines.append(line)
        elif stripped.startswith("build_intake_mapping"):
            detail_lines.append(line)
        elif stripped.startswith("build_observed_map"):
            detail_lines.append(line)
        elif stripped == "Not reached by:":
            route_lines.append(line)
        elif stripped.startswith("/api/lexicon/intake/map") or stripped.startswith("/intake-map"):
            route_lines.append(line)
    selected = path_lines[:1] + route_lines + detail_lines
    return selected or expanded


def expand_code_panel_lines(lines: list[str]) -> list[str]:
    expanded: list[str] = []
    not_reached_re = re.compile(r"^\s*Not reached by (?P<first>/\S+) or (?P<second>/\S+)(?P<rest>.*)$")
    for line in lines:
        match = not_reached_re.match(line)
        if not match:
            expanded.append(line)
            continue
        rest = match.group("rest").strip()
        expanded.extend(["Not reached by:", f"  {match.group('first')}", f"  {match.group('second')}"])
        if rest:
            expanded.append(f"  {rest}")
    return expanded


def wrap_monospace_line(line: str, *, max_chars: int) -> list[str]:
    stripped = line.strip()
    if len(stripped) <= max_chars:
        return [stripped]
    chunks: list[str] = []
    current = ""
    for word in stripped.split():
        trial = word if not current else current + " " + word
        if len(trial) <= max_chars:
            current = trial
            continue
        if current:
            chunks.append(current)
        current = word
    if current:
        chunks.append(current)
    return chunks or [stripped[:max_chars]]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.ImageFont, max_width: int | float) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    line = ""
    for word in words:
        trial = word if not line else line + " " + word
        if text_width(draw, trial, font_obj) <= max_width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [""]


def text_width(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    return int(bbox[2] - bbox[0])


def rounded(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int | float, int | float, int | float, int | float],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "markdown_frames"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _frame_row(index: int, title: str, path: Path) -> dict[str, Any]:
    return {"frame_index": index, "title": title, "path": str(path), "sha256": sha256_file(path)}


class FontSet:
    def __init__(self, *, scale: float) -> None:
        self.scale = scale
        self.title = load_font("segoeuib.ttf", int(56 * scale))
        self.heading = load_font("segoeuib.ttf", int(42 * scale))
        self.subheading = load_font("segoeuib.ttf", int(27 * scale))
        self.body_size = max(12, int(24 * scale))
        self.body = load_font("segoeui.ttf", self.body_size)
        self.body_bold = load_font("segoeuib.ttf", self.body_size)
        self.small = load_font("segoeui.ttf", max(10, int(18 * scale)))
        self.small_bold = load_font("segoeuib.ttf", max(10, int(18 * scale)))
        self.tiny_size = max(9, int(16 * scale))
        self.tiny_bold = load_font("segoeuib.ttf", self.tiny_size)
        self.mono_size = max(9, int(15 * scale))
        self.mono = load_font("consola.ttf", self.mono_size)

    @classmethod
    def for_size(cls, frame_size: tuple[int, int]) -> "FontSet":
        return cls(scale=max(0.5, min(frame_size[0] / 1600, frame_size[1] / 1000)))


def load_font(name: str, size: int) -> ImageFont.ImageFont:
    font_dir = Path("C:/Windows/Fonts")
    for candidate in (font_dir / name, font_dir / "segoeui.ttf", font_dir / "arial.ttf"):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a markdown document as TrueVision page frames.")
    parser.add_argument("markdown", help="Markdown document to render.")
    parser.add_argument("--output-root", default="", help="Output root. Defaults to outputs/<markdown-stem>_frames.")
    parser.add_argument("--run-id", default="", help="Stable run id for receipts and state movie.")
    parser.add_argument("--title", default="", help="Display title override.")
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_SIZE[1])
    parser.add_argument("--grid-rows", type=int, default=DEFAULT_GRID_SHAPE[0])
    parser.add_argument("--grid-cols", type=int, default=DEFAULT_GRID_SHAPE[1])
    parser.add_argument("--frames-per-page", type=int, default=1)
    parser.add_argument("--fps", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    source = Path(args.markdown)
    output_root = Path(args.output_root) if args.output_root else Path("outputs") / f"{source.stem}_frames"
    result = render_markdown_frame_set(
        source_markdown=source,
        output_root=output_root,
        run_id=args.run_id or None,
        title=args.title or None,
        frames_per_page=args.frames_per_page,
        fps=args.fps,
        grid_shape=(args.grid_rows, args.grid_cols),
        frame_size=(args.width, args.height),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
