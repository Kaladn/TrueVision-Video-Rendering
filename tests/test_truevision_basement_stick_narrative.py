import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from truevision_basement_stick_narrative import (
    SCENE_BEATS,
    build_scene_schedule,
    generate_basement_stick_narrative,
    render_frame,
    scene_for_time,
)


class TrueVisionBasementStickNarrativeTests(unittest.TestCase):
    def test_scene_schedule_covers_literal_basement_arc(self):
        schedule = build_scene_schedule(267.75)
        scene_ids = [entry["scene_id"] for entry in schedule]

        self.assertEqual(scene_ids[0], "storm_blackout")
        self.assertEqual(scene_ids[-1], "seal_ascend")
        self.assertIn("basement_door", scene_ids)
        self.assertIn("window_creature", scene_ids)
        self.assertIn("frank_falls", scene_ids)
        self.assertIn("red_rift_reveal", scene_ids)
        self.assertIn("sword_awakening", scene_ids)
        self.assertIn("mother_table", scene_ids)
        self.assertAlmostEqual(schedule[0]["start_seconds"], 0.0)
        self.assertAlmostEqual(schedule[-1]["end_seconds"], 267.75)
        self.assertEqual(len(schedule), len(SCENE_BEATS))

    def test_scene_for_time_maps_music_arc_to_expected_beats(self):
        duration = 100.0

        self.assertEqual(scene_for_time(1.0, duration).scene_id, "storm_blackout")
        self.assertEqual(scene_for_time(22.0, duration).scene_id, "window_creature")
        self.assertEqual(scene_for_time(37.0, duration).scene_id, "frank_falls")
        self.assertEqual(scene_for_time(75.0, duration).scene_id, "sword_awakening")
        self.assertEqual(scene_for_time(94.0, duration).scene_id, "mother_table")

    def test_render_frame_returns_nonblank_story_metadata(self):
        feature = {
            "frame_index": 17,
            "time_seconds": 73.0,
            "rms": 0.65,
            "bass": 0.9,
            "mid": 0.4,
            "high": 0.3,
            "beat": 0.8,
        }

        frame, metadata = render_frame(320, 180, feature, 100.0)

        self.assertEqual(frame.shape, (180, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(frame), 500)
        self.assertEqual(metadata["scene_id"], "sword_awakening")
        self.assertIn("Grandpa guidance", metadata["story_anchor"])

    def test_generate_basement_stick_narrative_writes_tiny_bundle_without_text_cards(self):
        sample_rate = 8000
        samples = (0.45 * np.sin(2 * np.pi * 120 * np.arange(sample_rate) / sample_rate)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "basement_tiny.wav"
            story_path = root / "The Basement.txt"
            lyrics_path = root / "Full Album Lyrics_sound.txt"
            story_path.write_text("storm basement door creature Frank Nether World sword mother rift", encoding="utf-8")
            lyrics_path.write_text("The Basement\nwake up join together descend ascend", encoding="utf-8")
            with wave.open(str(audio_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes())

            result = generate_basement_stick_narrative(
                audio_path=audio_path,
                story_path=story_path,
                lyrics_path=lyrics_path,
                output_root=root / "out",
                run_id="unit_basement",
                width=160,
                height=90,
                fps=5,
                sample_rate=sample_rate,
                max_seconds=0.6,
                mux_audio=False,
            )

            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            self.assertTrue(Path(result["video_mp4"]).exists())
            self.assertTrue(Path(result["manifest_json"]).exists())
            self.assertTrue(Path(result["frame_state_jsonl"]).exists())
            self.assertTrue(Path(result["thumbnail_jpg"]).exists())
            self.assertFalse(result["audio_muxed"])
            self.assertTrue(manifest["boundary"]["no_lyric_overlay"])
            self.assertTrue(manifest["boundary"]["no_dialogue_cards"])
            self.assertEqual(manifest["inputs"]["visual_policy"], "literal_story_scenery_no_lyric_text")
            self.assertEqual(manifest["render"]["scene_schedule"][0]["scene_id"], "storm_blackout")


if __name__ == "__main__":
    unittest.main()
