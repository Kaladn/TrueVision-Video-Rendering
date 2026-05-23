import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from trueaudio_runtime.logging import log_machine_pre_sound_state, log_pre_sound_state
from trueaudio_runtime.replay import replay_trueaudio_state
from trueaudio_runtime.replayable import (
    log_file_replayable_audio_state,
    log_machine_replayable_audio_state,
    replay_replayable_audio_state,
    write_replayable_audio_state,
)


class TrueAudioLoggingTests(unittest.TestCase):
    def _write_stereo_wav(self, path: Path, *, sample_rate: int = 8000, seconds: float = 0.5) -> None:
        frame_count = int(sample_rate * seconds)
        frames = bytearray()
        for index in range(frame_count):
            t = index / sample_rate
            left = int(math.sin(math.tau * 220.0 * t) * 18000)
            right = int(math.sin(math.tau * 440.0 * t) * 9000)
            frames.extend(left.to_bytes(2, "little", signed=True))
            frames.extend(right.to_bytes(2, "little", signed=True))
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(bytes(frames))

    def test_logs_pre_sound_state_without_saving_raw_pcm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "tiny_stereo.wav"
            self._write_stereo_wav(audio)

            result = log_pre_sound_state(
                audio,
                storage_root=root,
                run_id="unit_trueaudio",
                fps=10,
                sample_rate=8000,
                max_seconds=0.5,
            )

            self.assertEqual(result["schema_version"], "trueaudio_pre_sound_log_result_v1")
            self.assertEqual(result["frame_count"], 5)
            self.assertTrue(Path(result["state_jsonl"]).exists())
            self.assertTrue(Path(result["manifest_json"]).exists())
            self.assertTrue(Path(result["receipt_json"]).exists())
            self.assertGreater(result["summary"]["max_rms"], 0.0)
            self.assertIn("stereo_balance_mean", result["summary"])

            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "trueaudio_pre_sound_manifest_v1")
            self.assertEqual(manifest["decode_stage"], "decoded_pcm_pre_output")
            self.assertEqual(manifest["boundary"]["system_role"], "TrueAudio sibling sensor/state system")
            self.assertFalse(manifest["boundary"]["raw_audio_saved"])
            self.assertFalse(manifest["boundary"]["pcm_saved"])
            self.assertTrue(manifest["boundary"]["derived_state_only"])

            rows = [
                json.loads(line)
                for line in Path(result["state_jsonl"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["schema_version"], "trueaudio_state_frame_v1")
            self.assertIn("rms_left", rows[0]["channels"])
            self.assertIn("rms_right", rows[0]["channels"])
            self.assertIn("bass", rows[0]["bands"])
            self.assertIn("attack", rows[0]["dynamics"])

    def test_state_hash_is_deterministic_for_same_audio_and_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "tiny_stereo.wav"
            self._write_stereo_wav(audio)

            first = log_pre_sound_state(audio, storage_root=root, run_id="first", fps=8, sample_rate=8000)
            second = log_pre_sound_state(audio, storage_root=root, run_id="second", fps=8, sample_rate=8000)

            self.assertEqual(first["state_sha256"], second["state_sha256"])
            self.assertEqual(first["summary"], second["summary"])

    def test_logs_machine_pre_sound_state_from_loopback_provider(self):
        def fake_capture(*, duration_seconds: float):
            sample_rate = 8000
            frame_count = int(sample_rate * duration_seconds)
            t = np.arange(frame_count, dtype=np.float32) / sample_rate
            left = np.sin(t * math.tau * 110.0).astype(np.float32) * 0.25
            right = np.sin(t * math.tau * 220.0).astype(np.float32) * 0.125
            return np.column_stack([left, right]).astype(np.float32), {
                "backend": "test_loopback",
                "sample_rate": sample_rate,
                "channels": 2,
                "device_role": "render_loopback",
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            result = log_machine_pre_sound_state(
                storage_root=root,
                run_id="unit_machine",
                duration_seconds=0.5,
                fps=10,
                capture_provider=fake_capture,
            )

            self.assertEqual(result["schema_version"], "trueaudio_machine_pre_sound_log_result_v1")
            self.assertEqual(result["frame_count"], 5)
            self.assertTrue(Path(result["state_jsonl"]).exists())
            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "trueaudio_machine_pre_sound_manifest_v1")
            self.assertEqual(manifest["decode_stage"], "machine_loopback_pre_output")
            self.assertEqual(manifest["machine_capture"]["backend"], "test_loopback")
            self.assertEqual(manifest["boundary"]["capture_scope"], "local_machine_output_mix")
            self.assertFalse(manifest["boundary"]["raw_audio_saved"])
            self.assertFalse(manifest["boundary"]["pcm_saved"])
            self.assertFalse(manifest["boundary"]["replayable_audio"])

    def test_replays_state_log_as_bounded_sonification_not_original_audio(self):
        def fake_capture(*, duration_seconds: float):
            sample_rate = 8000
            frame_count = int(sample_rate * duration_seconds)
            t = np.arange(frame_count, dtype=np.float32) / sample_rate
            left = np.sin(t * math.tau * 110.0).astype(np.float32) * 0.3
            right = np.sin(t * math.tau * 330.0).astype(np.float32) * 0.2
            return np.column_stack([left, right]).astype(np.float32), {
                "backend": "test_loopback",
                "sample_rate": sample_rate,
                "channels": 2,
                "device_role": "render_loopback",
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logged = log_machine_pre_sound_state(
                storage_root=root,
                run_id="unit_machine_for_replay",
                duration_seconds=0.5,
                fps=10,
                capture_provider=fake_capture,
            )

            replay = replay_trueaudio_state(
                logged["state_jsonl"],
                storage_root=root,
                run_id="unit_replay",
                sample_rate=8000,
            )

            self.assertEqual(replay["schema_version"], "trueaudio_state_replay_result_v1")
            self.assertEqual(replay["frame_count"], 5)
            self.assertTrue(Path(replay["wav_path"]).exists())
            self.assertTrue(Path(replay["manifest_json"]).exists())
            with wave.open(replay["wav_path"], "rb") as handle:
                self.assertEqual(handle.getnchannels(), 2)
                self.assertEqual(handle.getframerate(), 8000)
                self.assertGreater(handle.getnframes(), 0)
            manifest = json.loads(Path(replay["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "trueaudio_state_replay_manifest_v1")
            self.assertEqual(manifest["boundary"]["replay_kind"], "state_sonification")
            self.assertFalse(manifest["boundary"]["source_audio_recovered"])
            self.assertFalse(manifest["boundary"]["claims_original_audio"])

    def test_replayable_audio_state_reconstructs_close_without_raw_pcm(self):
        sample_rate = 8000
        duration = 1.0
        t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
        left = (
            np.sin(t * math.tau * 220.0) * 0.30
            + np.sin(t * math.tau * 660.0) * 0.08
            + np.sin(t * math.tau * (180.0 + t * 80.0)) * 0.04
        )
        right = (
            np.sin(t * math.tau * 330.0) * 0.20
            + np.sin(t * math.tau * 990.0) * 0.06
            + np.sin(t * math.tau * (120.0 + t * 40.0)) * 0.03
        )
        original = np.column_stack([left, right]).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logged = write_replayable_audio_state(
                original,
                sample_rate=sample_rate,
                storage_root=root,
                run_id="unit_replayable",
                frame_size=512,
                hop_size=128,
            )
            replay = replay_replayable_audio_state(
                logged["state_npz"],
                storage_root=root,
                run_id="unit_replayable_replay",
            )

            self.assertEqual(logged["schema_version"], "trueaudio_replayable_state_log_result_v1")
            self.assertEqual(replay["schema_version"], "trueaudio_replayable_state_replay_result_v1")
            self.assertTrue(Path(logged["state_npz"]).exists())
            self.assertTrue(Path(replay["wav_path"]).exists())
            manifest = json.loads(Path(logged["manifest_json"]).read_text(encoding="utf-8"))
            self.assertFalse(manifest["boundary"]["raw_audio_saved"])
            self.assertFalse(manifest["boundary"]["pcm_saved"])
            self.assertTrue(manifest["boundary"]["replayable_audio_state"])

            restored = replay["samples"]
            compare = min(original.shape[0], restored.shape[0])
            error = original[:compare] - restored[:compare]
            signal_power = float(np.mean(original[:compare] * original[:compare]))
            error_power = float(np.mean(error * error))
            snr_db = 10.0 * math.log10(signal_power / max(error_power, 1.0e-12))
            self.assertGreater(snr_db, 35.0)

    def test_machine_replayable_audio_state_uses_loopback_provider(self):
        def fake_capture(*, duration_seconds: float):
            sample_rate = 8000
            frame_count = int(sample_rate * duration_seconds)
            t = np.arange(frame_count, dtype=np.float32) / sample_rate
            left = np.sin(t * math.tau * 220.0).astype(np.float32) * 0.2
            right = np.sin(t * math.tau * 440.0).astype(np.float32) * 0.15
            return np.column_stack([left, right]).astype(np.float32), {
                "backend": "test_loopback",
                "sample_rate": sample_rate,
                "channels": 2,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = log_machine_replayable_audio_state(
                storage_root=root,
                run_id="unit_machine_replayable",
                duration_seconds=0.5,
                frame_size=512,
                hop_size=128,
                capture_provider=fake_capture,
            )

            self.assertEqual(result["schema_version"], "trueaudio_machine_replayable_state_log_result_v1")
            self.assertTrue(Path(result["state_npz"]).exists())
            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["capture"]["backend"], "test_loopback")
            self.assertEqual(manifest["boundary"]["capture_scope"], "local_machine_output_mix")
            self.assertTrue(manifest["boundary"]["replayable_audio_state"])

    def test_file_replayable_audio_state_uses_ffmpeg_decode_without_raw_pcm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wav_path = root / "voice_file.wav"
            sample_rate = 8000
            samples = bytearray()
            for index in range(sample_rate // 2):
                value = int(math.sin(index / sample_rate * math.tau * 220.0) * 12000)
                samples.extend(value.to_bytes(2, "little", signed=True))
                samples.extend(value.to_bytes(2, "little", signed=True))
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(bytes(samples))

            result = log_file_replayable_audio_state(
                wav_path,
                storage_root=root,
                run_id="unit_file_replayable",
                sample_rate=sample_rate,
                frame_size=512,
                hop_size=128,
            )

            self.assertEqual(result["schema_version"], "trueaudio_file_replayable_state_log_result_v1")
            self.assertTrue(Path(result["state_npz"]).exists())
            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["capture"]["capture_scope"], "source_audio_file")
            self.assertEqual(manifest["capture"]["capture_stage"], "ffmpeg_decoded_pre_output")
            self.assertEqual(manifest["capture"]["source_audio_sha256"], result["source_audio_sha256"])
            self.assertFalse(manifest["boundary"]["raw_audio_saved"])
            self.assertFalse(manifest["boundary"]["pcm_saved"])
            self.assertTrue(manifest["boundary"]["replayable_audio_state"])


if __name__ == "__main__":
    unittest.main()
