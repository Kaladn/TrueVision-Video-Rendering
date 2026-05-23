import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trueaudio_runtime.replayable import write_replayable_audio_state
from trueaudio_runtime.lyrics import align_lyrics_to_speech_segments
from trueaudio_runtime.speech import detect_speech_segments_from_replayable_state


class TrueSpeechDetectionTests(unittest.TestCase):
    def test_detects_speech_region_without_transcript_claims(self):
        sample_rate = 8000
        seconds = 3.0
        t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
        rng = np.random.default_rng(616)
        noise = rng.normal(0.0, 0.012, t.shape[0]).astype(np.float32)

        speech_like = np.zeros_like(t, dtype=np.float32)
        speech_window = (t >= 1.0) & (t < 2.0)
        local = t[speech_window] - 1.0
        envelope = np.sin(np.pi * local).astype(np.float32)
        voiced = (
            np.sin(math.tau * 145.0 * local) * 0.22
            + np.sin(math.tau * 290.0 * local) * 0.12
            + np.sin(math.tau * 580.0 * local) * 0.07
            + np.sin(math.tau * 1160.0 * local) * 0.04
        ).astype(np.float32)
        speech_like[speech_window] = voiced * envelope

        mono = noise + speech_like
        stereo = np.column_stack([mono, mono * 0.92]).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logged = write_replayable_audio_state(
                stereo,
                sample_rate=sample_rate,
                storage_root=root,
                run_id="unit_speech_state",
                frame_size=512,
                hop_size=128,
            )

            result = detect_speech_segments_from_replayable_state(
                logged["state_npz"],
                storage_root=root,
                run_id="unit_speech_detect",
            )

            self.assertEqual(result["schema_version"], "truespeech_detection_result_v1")
            self.assertTrue(Path(result["frames_jsonl"]).exists())
            self.assertTrue(Path(result["segments_json"]).exists())
            self.assertTrue(Path(result["manifest_json"]).exists())
            self.assertGreaterEqual(result["summary"]["speech_segment_count"], 1)
            segment = result["segments"][0]
            self.assertLess(segment["start_seconds"], 1.25)
            self.assertGreater(segment["end_seconds"], 1.75)
            self.assertGreater(segment["mean_confidence"], 0.45)

            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "truespeech_detection_manifest_v1")
            self.assertFalse(manifest["boundary"]["asr_claim"])
            self.assertFalse(manifest["boundary"]["transcript_claim"])
            self.assertTrue(manifest["boundary"]["speech_detection_only"])
            self.assertEqual(manifest["source_state"]["schema"], "trueaudio_replayable_spectral_state_v1")

    def test_aligns_provided_lyrics_as_candidates_not_transcript_truth(self):
        lyrics = """
[Verse]
Baby don't let me fall again
I was hanging by a thread back then

[Chorus]
Baby come and rescue me
"""
        segments_payload = {
            "schema_version": "truespeech_segments_v1",
            "segments": [
                {
                    "start_seconds": 10.0,
                    "end_seconds": 14.0,
                    "duration_seconds": 4.0,
                    "mean_confidence": 0.71,
                    "max_confidence": 0.82,
                    "frame_count": 100,
                },
                {
                    "start_seconds": 20.0,
                    "end_seconds": 26.0,
                    "duration_seconds": 6.0,
                    "mean_confidence": 0.83,
                    "max_confidence": 0.93,
                    "frame_count": 150,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segments_path = root / "segments.json"
            segments_path.write_text(json.dumps(segments_payload), encoding="utf-8")

            result = align_lyrics_to_speech_segments(
                segments_path,
                lyrics_text=lyrics,
                storage_root=root,
                run_id="unit_lyrics_align",
            )

            self.assertEqual(result["schema_version"], "truespeech_lyric_alignment_result_v1")
            self.assertEqual(result["line_count"], 3)
            self.assertTrue(Path(result["alignment_json"]).exists())
            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            self.assertFalse(manifest["boundary"]["asr_claim"])
            self.assertFalse(manifest["boundary"]["transcript_claim"])
            self.assertTrue(manifest["boundary"]["provided_lyrics_used"])
            self.assertTrue(manifest["boundary"]["candidate_alignment_only"])
            alignment = json.loads(Path(result["alignment_json"]).read_text(encoding="utf-8"))
            self.assertEqual(alignment["lines"][0]["section"], "Verse")
            self.assertLess(alignment["lines"][0]["start_seconds"], alignment["lines"][-1]["start_seconds"])


if __name__ == "__main__":
    unittest.main()
