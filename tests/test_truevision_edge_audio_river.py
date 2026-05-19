import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from truevision_edge_audio_river import (
    build_edge_theme,
    generate_edge_audio_river,
    measure_audio_features,
    render_river_frame,
)


class TrueVisionEdgeAudioRiverTests(unittest.TestCase):
    def test_measure_audio_features_returns_normalized_bands(self):
        sample_rate = 8000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        samples = (
            0.55 * np.sin(2 * np.pi * 90 * t)
            + 0.25 * np.sin(2 * np.pi * 900 * t)
            + 0.18 * np.sin(2 * np.pi * 2600 * t)
        ).astype(np.float32)

        features = measure_audio_features(samples, sample_rate=sample_rate, fps=10)

        self.assertGreaterEqual(len(features), 10)
        for feature in features:
            for key in ["rms", "bass", "mid", "high", "beat"]:
                self.assertGreaterEqual(feature[key], 0.0)
                self.assertLessEqual(feature[key], 1.0)
        self.assertGreater(max(feature["bass"] for feature in features), 0.2)

    def test_render_river_frame_has_black_field_and_no_glyph_layer(self):
        frame_state = {
            "frame_index": 12,
            "time_seconds": 0.4,
            "rms": 0.45,
            "bass": 0.8,
            "mid": 0.4,
            "high": 0.3,
            "beat": 0.7,
        }
        frame, metadata = render_river_frame(
            width=320,
            height=180,
            fps=30,
            frame_state=frame_state,
            trail=None,
        )

        self.assertEqual(frame.shape, (180, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(frame), 100)
        self.assertTrue(metadata["visual_rules"]["no_lettering"])
        self.assertTrue(metadata["visual_rules"]["no_glyphs"])

    def test_render_river_frame_can_stamp_program_in_black_band(self):
        frame_state = {
            "frame_index": 12,
            "time_seconds": 0.4,
            "rms": 0.45,
            "bass": 0.8,
            "mid": 0.4,
            "high": 0.3,
            "beat": 0.7,
        }
        frame, metadata = render_river_frame(
            width=320,
            height=180,
            fps=30,
            frame_state=frame_state,
            trail=None,
            river_height_ratio=0.35,
            program_stamp="TrueVision Generation Lab",
        )

        self.assertTrue(metadata["visual_rules"]["program_stamp"])
        self.assertFalse(metadata["visual_rules"]["no_lettering"])
        self.assertGreater(np.count_nonzero(frame[150:179, 0:180]), 0)

    def test_build_edge_theme_uses_lyric_river_language(self):
        text = """Edge Of The World
Lyrics:
There is a river of life floating in the abyss
You! Me! Together we fight
---
Next Track
"""
        with tempfile.TemporaryDirectory() as tmp:
            lyric_path = Path(tmp) / "lyrics.txt"
            lyric_path.write_text(text, encoding="utf-8")

            theme = build_edge_theme(lyric_path)

        self.assertEqual(theme["track_title"], "Edge Of The World")
        self.assertIn("river of life", theme["theme_phrases"])
        self.assertIn("you me together", theme["theme_phrases"])
        self.assertTrue(theme["visual_rules"]["no_lettering"])

    def test_generate_edge_audio_river_writes_tiny_bundle(self):
        sample_rate = 8000
        samples = (0.5 * np.sin(2 * np.pi * 120 * np.arange(sample_rate // 2) / sample_rate)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "tiny.wav"
            with wave.open(str(audio_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes())

            result = generate_edge_audio_river(
                audio_path=audio_path,
                lyrics_path=None,
                output_root=root / "out",
                run_id="unit_edge_river",
                width=160,
                height=90,
                fps=6,
                max_seconds=0.5,
                mux_audio=False,
            )

            self.assertTrue(Path(result["video_mp4"]).exists())
            self.assertTrue(Path(result["manifest_json"]).exists())
            self.assertTrue(Path(result["frame_state_jsonl"]).exists())
            self.assertTrue(Path(result["report_md"]).exists())
            self.assertFalse(result["audio_muxed"])


if __name__ == "__main__":
    unittest.main()
