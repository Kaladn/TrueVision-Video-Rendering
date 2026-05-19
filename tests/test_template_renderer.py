import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from truevision_runtime.rendering.template_renderer import (
    load_render_template,
    render_mirror_maze_frame,
    render_template,
    scene_for_time,
)


class TemplateRendererTests(unittest.TestCase):
    def _write_tiny_audio(self, path: Path, sample_rate: int = 8000) -> None:
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        samples = (
            0.5 * np.sin(2 * np.pi * 90 * t)
            + 0.25 * np.sin(2 * np.pi * 700 * t)
            + 0.1 * np.sin(2 * np.pi * 2200 * t)
        ).astype(np.float32)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes())

    def _template(self, root: Path) -> Path:
        audio_path = root / "tiny.wav"
        self._write_tiny_audio(audio_path)
        template_path = root / "template.json"
        template_path.write_text(
            json.dumps(
                {
                    "template_id": "unit_mirror",
                    "run_id": "unit_mirror",
                    "title": "Unit Mirror",
                    "audio": {"path": str(audio_path), "sample_rate": 8000},
                    "render": {"visual_mode": "mirror_maze_realism"},
                    "output": {
                        "root": str(root / "out"),
                        "width": 160,
                        "height": 90,
                        "fps": 6,
                        "max_seconds": 0.6,
                        "encoder": "libx264",
                    },
                    "style": {"bloom_strength": 0.4, "vignette_strength": 1.2},
                    "scenes": [
                        {"scene_id": "mirror_start", "start_norm": 0.0, "end_norm": 0.5},
                        {"scene_id": "mirror_build", "start_norm": 0.5, "end_norm": 1.0},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return template_path

    def test_load_template_uses_reusable_visual_mode_not_song_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = load_render_template(self._template(Path(tmp)))

            self.assertEqual(template.visual_mode, "mirror_maze_realism")
            self.assertEqual(template.width, 160)
            self.assertEqual(template.encoder, "libx264")

    def test_scene_for_time_uses_template_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = load_render_template(self._template(Path(tmp)))

            self.assertEqual(scene_for_time(template, 0.1, 1.0)["scene_id"], "mirror_start")
            self.assertEqual(scene_for_time(template, 0.7, 1.0)["scene_id"], "mirror_build")

    def test_render_mirror_maze_frame_has_realism_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = load_render_template(self._template(Path(tmp)))
            frame, metadata = render_mirror_maze_frame(
                template=template,
                frame_state={
                    "frame_index": 3,
                    "time_seconds": 0.5,
                    "rms": 0.7,
                    "bass": 0.8,
                    "mid": 0.6,
                    "high": 0.5,
                    "beat": 0.9,
                },
                duration_seconds=1.0,
            )

            self.assertEqual(frame.shape, (90, 160, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertGreater(np.count_nonzero(frame), 1000)
            self.assertIn("mirror_shards", metadata["layers"])
            self.assertIn("volumetric_smoke", metadata["layers"])
            self.assertTrue(metadata["boundary"]["template_driven"])

    def test_render_template_writes_manifest_with_encoder_and_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = render_template(self._template(Path(tmp)), mux_audio=False)
            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            report = Path(result["report_md"]).read_text(encoding="utf-8")

            self.assertTrue(Path(result["video_mp4"]).exists())
            self.assertTrue(Path(result["thumbnail_jpg"]).exists())
            self.assertEqual(manifest["render"]["visual_mode"], "mirror_maze_realism")
            self.assertEqual(manifest["render"]["encoder_used"], "libx264")
            self.assertFalse(manifest["render"]["hardware_encode"])
            self.assertIn("machine_cost", manifest)
            self.assertIn("frame_synthesis_and_encode_seconds", manifest["component_timing_seconds"])
            self.assertIn("Template-driven mirror-maze realism", report)


if __name__ == "__main__":
    unittest.main()

