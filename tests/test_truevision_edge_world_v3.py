import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from truevision_edge_world_v3 import (
    build_edge_world_v3_schedule,
    generate_edge_world_v3,
    render_edge_world_v3_frame,
    scene_for_time,
)


class TrueVisionEdgeWorldV3Tests(unittest.TestCase):
    def _signature(self) -> dict:
        return {
            "kind": "truevision_signature_profile_bundle",
            "profile_id": "unit_cod_signature",
            "timeline_samples": [
                {
                    "time_norm": 0.0,
                    "motion": 0.1,
                    "edge": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.25,
                    "flash": 0.0,
                    "shake_x": 0.0,
                    "shake_y": 0.0,
                },
                {
                    "time_norm": 0.44,
                    "motion": 0.92,
                    "edge": 0.85,
                    "contrast": 0.7,
                    "saturation": 0.8,
                    "flash": 0.45,
                    "shake_x": 0.38,
                    "shake_y": -0.22,
                },
            ],
        }

    def test_scene_schedule_includes_edge_smoke_and_looking_down(self):
        schedule = build_edge_world_v3_schedule(278.0)
        scene_ids = [entry["scene_id"] for entry in schedule]

        self.assertEqual(scene_ids[0], "edge_horizon_smoke")
        self.assertIn("looking_down_over_edge", scene_ids)
        self.assertIn("river_below_energy", scene_ids)
        self.assertEqual(scene_ids[-1], "return_to_black_edge")
        self.assertEqual(schedule[0]["start_seconds"], 0.0)
        self.assertEqual(schedule[-1]["end_seconds"], 278.0)

    def test_scene_for_time_maps_mid_song_to_looking_down(self):
        self.assertEqual(scene_for_time(122.0, 278.0).scene_id, "looking_down_over_edge")

    def test_render_frame_has_smoke_river_metadata_and_no_lyrics(self):
        feature = {
            "frame_index": 44,
            "time_seconds": 122.0,
            "rms": 0.72,
            "bass": 0.68,
            "mid": 0.5,
            "high": 0.55,
            "beat": 0.7,
        }

        frame, metadata = render_edge_world_v3_frame(
            width=320,
            height=180,
            fps=12,
            frame_state=feature,
            duration_seconds=278.0,
            signature_profile=self._signature(),
        )

        self.assertEqual(frame.shape, (180, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(frame), 500)
        self.assertEqual(metadata["scene_id"], "looking_down_over_edge")
        self.assertTrue(metadata["visual_rules"]["no_lyric_overlay"])
        self.assertTrue(metadata["signature_style"]["applied"])
        self.assertIn("river_below", metadata["layers"])

    def test_signature_profile_changes_frame_style(self):
        feature = {
            "frame_index": 44,
            "time_seconds": 122.0,
            "rms": 0.72,
            "bass": 0.68,
            "mid": 0.5,
            "high": 0.55,
            "beat": 0.7,
        }
        plain, _ = render_edge_world_v3_frame(
            width=320,
            height=180,
            fps=12,
            frame_state=feature,
            duration_seconds=278.0,
            signature_profile=None,
        )
        styled, styled_meta = render_edge_world_v3_frame(
            width=320,
            height=180,
            fps=12,
            frame_state=feature,
            duration_seconds=278.0,
            signature_profile=self._signature(),
        )

        self.assertFalse(np.array_equal(plain, styled))
        self.assertEqual(styled_meta["signature_style"]["profile_id"], "unit_cod_signature")

    def test_generate_edge_world_v3_writes_tiny_bundle(self):
        sample_rate = 8000
        samples = (0.5 * np.sin(2 * np.pi * 100 * np.arange(sample_rate) / sample_rate)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "edge_tiny.wav"
            lyrics_path = root / "lyrics.txt"
            signature_path = root / "signature.json"
            lyrics_path.write_text("Edge Of The World\nwake up you me together river of life", encoding="utf-8")
            signature_path.write_text(json.dumps(self._signature()), encoding="utf-8")
            with wave.open(str(audio_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes())

            result = generate_edge_world_v3(
                audio_path=audio_path,
                lyrics_path=lyrics_path,
                output_root=root / "out",
                run_id="unit_edge_v3",
                width=160,
                height=90,
                fps=6,
                sample_rate=sample_rate,
                max_seconds=0.6,
                mux_audio=False,
                signature_profile_path=signature_path,
            )
            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))

            self.assertTrue(Path(result["video_mp4"]).exists())
            self.assertTrue(Path(result["frame_state_jsonl"]).exists())
            self.assertTrue(Path(result["thumbnail_jpg"]).exists())
            self.assertFalse(result["audio_muxed"])
            self.assertEqual(manifest["render"]["style"], "edge_world_v3_edge_smoke_river_below")
            self.assertTrue(manifest["render"]["signature_profile"]["enabled"])
            self.assertIn("machine_cost", manifest)
            self.assertGreaterEqual(manifest["machine_cost"]["process_cpu_seconds"], 0.0)
            self.assertGreaterEqual(manifest["machine_cost"]["avg_process_logical_cpu_percent"], 0.0)
            self.assertIn("memory_start", manifest["machine_cost"])
            self.assertIn("memory_end", manifest["machine_cost"])
            self.assertIn("working_set_bytes", manifest["machine_cost"]["memory_end"])
            self.assertIn("component_timing_seconds", manifest)
            self.assertIn("audio_decode_feature_extract_seconds", manifest["component_timing_seconds"])
            self.assertIn("frame_synthesis_and_video_encode_seconds", manifest["component_timing_seconds"])
            self.assertIn("manifest_report_hash_seconds", manifest["component_timing_seconds"])
            self.assertIn("gpu", manifest["hardware"])
            report_text = Path(result["report_md"]).read_text(encoding="utf-8")
            self.assertIn("## Component Timing", report_text)
            self.assertIn("## System Components", report_text)
            self.assertIn("GPU acceleration used", report_text)


if __name__ == "__main__":
    unittest.main()
