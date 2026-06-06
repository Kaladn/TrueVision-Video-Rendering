import json
import tempfile
import unittest
from pathlib import Path

from truevision_runtime.rendering.markdown_frame_renderer import (
    display_label_for_graph_node,
    prepare_code_panel_lines,
    wrap_monospace_line,
    parse_markdown_sections,
    render_markdown_frame_set,
)


class MarkdownFrameRendererTests(unittest.TestCase):
    def test_parse_markdown_sections_uses_h2_sections(self):
        source = """# Lab Map

Intro text.

## First Section

```mermaid
flowchart TD
    A["Start"] --> B["End"]
```

Why:

- One thing.

## Second Section

Plain notes.
"""

        sections = parse_markdown_sections(source)

        self.assertEqual([section["title"] for section in sections], ["First Section", "Second Section"])
        self.assertIn("Start", sections[0]["body"])

    def test_render_markdown_frame_set_writes_frames_pdf_manifest_and_state_movie(self):
        markdown = """# Sample System

Generated test document.

## Runtime Path

```mermaid
flowchart TD
    Main["main.py"] --> CLI["cli.py"]
    CLI --> Store["Store"]
```

Why:

- Main dispatches into CLI.
- Store is the public state boundary.

## Receipt

```text
state first
pixels last
```
"""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.md"
            output = Path(tmp) / "frames"
            source.write_text(markdown, encoding="utf-8")

            result = render_markdown_frame_set(
                source_markdown=source,
                output_root=output,
                run_id="sample_system",
                frames_per_page=1,
                fps=1.0,
                grid_shape=(20, 32),
                frame_size=(640, 400),
            )

            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))

            self.assertEqual(manifest["schema_version"], "truevision_markdown_frame_set@1")
            self.assertEqual(manifest["frame_count"], 3)
            self.assertTrue(Path(result["pdf_path"]).exists())
            self.assertTrue(Path(result["truevision_document_state_movie"]["manifest_json"]).exists())
            self.assertEqual(len(list((output / "frames").glob("*.png"))), 3)
            self.assertFalse(manifest["boundary"]["html_css_js_used"])
            self.assertTrue(manifest["boundary"]["source_markdown_remains_receipt"])

    def test_display_label_shortens_route_and_function_graph_nodes(self):
        self.assertEqual(display_label_for_graph_node("POST /api/lexicon/intake/map"), "POST intake/map")
        self.assertEqual(display_label_for_graph_node("POST /api/lexicon/mapping/run"), "POST mapping/run")
        self.assertEqual(display_label_for_graph_node("preview_document_intake"), "Preview")
        self.assertEqual(display_label_for_graph_node("build_intake_mapping facade"), "Intake facade")
        self.assertEqual(display_label_for_graph_node("map_intake_content_to_user_counts_native"), "Native inline map")
        self.assertEqual(display_label_for_graph_node("build_observed_map"), "Observed map")
        self.assertEqual(display_label_for_graph_node("observed/symbolic map artifacts"), "Observed map")

    def test_wrap_monospace_line_preserves_long_code_panel_content(self):
        lines = wrap_monospace_line("Not reached by /api/lexicon/intake/map or /intake-map", max_chars=32)

        self.assertEqual(lines, ["Not reached by", "/api/lexicon/intake/map or", "/intake-map"])

    def test_prepare_code_panel_lines_keeps_not_reached_routes_visible(self):
        block = """src/AnchorWorks/store_runtime/intake.py
  build_intake_mapping is now a compatibility facade into native inline-content mapping.
  build_observed_map still exists for legacy observed/symbolic map artifacts.
  Not reached by /api/lexicon/intake/map or /intake-map after the native route change."""

        lines = prepare_code_panel_lines([block])

        self.assertIn("Not reached by:", lines)
        self.assertIn("  /api/lexicon/intake/map", lines)
        self.assertIn("  /intake-map", lines)
        stripped = [line.strip() for line in lines]
        self.assertLess(stripped.index("Not reached by:"), stripped.index("build_intake_mapping is now a compatibility facade into native inline-content mapping."))


if __name__ == "__main__":
    unittest.main()
