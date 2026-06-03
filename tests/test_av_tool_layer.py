import json
import math
import wave
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from truevision_runtime.av_tools.av_tool_policy import AVToolPolicyError, validate_tool_call
from truevision_runtime.av_tools.av_tool_registry import list_av_tools
from truevision_runtime.av_tools.av_tool_runner import run_av_tool_call


class AVToolLayerTests(unittest.TestCase):
    def test_registry_is_audio_video_only(self):
        tools = list_av_tools()
        names = {tool["name"] for tool in tools}

        self.assertIn("audio_probe_duration", names)
        self.assertIn("audio_analyze_levels", names)
        self.assertIn("audio_extract_features", names)
        self.assertIn("trueaudio_log_pre_sound", names)
        self.assertIn("trueaudio_log_machine_pre_sound", names)
        self.assertIn("trueaudio_replay_state", names)
        self.assertIn("trueaudio_log_file_replayable", names)
        self.assertIn("trueaudio_log_machine_replayable", names)
        self.assertIn("trueaudio_replay_replayable", names)
        self.assertIn("truespeech_detect_segments", names)
        self.assertIn("truespeech_align_lyrics_candidate", names)
        self.assertIn("template_from_audio_signals", names)
        self.assertIn("video_render_preview", names)
        self.assertIn("template_patch", names)
        self.assertIn("meter_grid_from_capture", names)
        self.assertNotIn("filesystem_delete", names)
        self.assertNotIn("browser_open", names)
        self.assertNotIn("security_enforce", names)
        self.assertTrue(all(tool["domain"] == "audio_video" for tool in tools))

    def test_registry_exposes_state_language_metadata(self):
        tools = {tool["name"]: tool for tool in list_av_tools()}

        atmosphere = tools["atmosphere_profile_from_capture"]
        self.assertEqual(atmosphere["behavior_family"], "fog_reveal")
        self.assertTrue(atmosphere["can_witness"])
        self.assertTrue(atmosphere["can_profile"])
        self.assertFalse(atmosphere["can_surface"])
        self.assertIn("witness", atmosphere["state_language"]["supported_stages"])
        self.assertIn("profile", atmosphere["state_language"]["supported_stages"])

        preview = tools["video_render_preview"]
        self.assertTrue(preview["can_plan"])
        self.assertTrue(preview["can_surface"])
        self.assertFalse(preview["copies_source_media"])
        self.assertFalse(preview["raw_media_saved"])
        self.assertFalse(preview["state_language"]["media_is_source_truth"])

    def test_policy_rejects_unknown_and_path_escape(self):
        with self.assertRaises(AVToolPolicyError):
            validate_tool_call({"tool": "filesystem_delete", "args": {"path": "D:/"}})

        with self.assertRaises(AVToolPolicyError):
            validate_tool_call({"tool": "template_load", "args": {"name": "../README.md"}})

    def test_policy_requires_confirmation_for_delete_and_execute(self):
        with self.assertRaises(AVToolPolicyError):
            validate_tool_call({"tool": "template_delete", "args": {"name": "edge.json"}})

        with self.assertRaises(AVToolPolicyError):
            validate_tool_call({"tool": "video_execute_full_render", "args": {"job_id": "edge"}})

        validated = validate_tool_call(
            {
                "tool": "video_execute_full_render",
                "args": {"job_id": "edge"},
                "human_confirmed": True,
            }
        )
        self.assertEqual(validated["tool"], "video_execute_full_render")

    def test_template_tools_and_receipts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            created = run_av_tool_call(
                {
                    "tool": "template_create",
                    "args": {
                        "name": "Edge River",
                        "prompt": "thin river on black",
                        "renderer": "edge_audio_river",
                        "duration_seconds": 278.32,
                        "fps": 30,
                    },
                },
                storage_root=storage,
            )
            self.assertTrue(created["ok"])
            self.assertEqual(created["result"]["template"]["timeline"]["frame_count"], 8350)
            self.assertTrue(Path(created["receipt"]["path"]).exists())

            saved = run_av_tool_call(
                {
                    "tool": "template_save",
                    "args": {
                        "name": "edge_river.json",
                        "template": created["result"]["template"],
                    },
                },
                storage_root=storage,
            )
            self.assertTrue(saved["ok"])
            self.assertEqual(saved["result"]["name"], "edge_river.json")

            patched = run_av_tool_call(
                {
                    "tool": "template_patch",
                    "args": {
                        "name": "edge_river.json",
                        "json_path": "visual_parameters.geometry.river_height_ratio",
                        "value": 0.28,
                        "reason": "river too thick at chorus",
                    },
                },
                storage_root=storage,
            )
            self.assertTrue(patched["ok"])
            self.assertEqual(
                patched["result"]["template"]["visual_parameters"]["geometry"]["river_height_ratio"],
                0.28,
            )

            receipt_files = list((storage / "receipts").glob("*.json"))
            self.assertEqual(len(receipt_files), 3)

    def test_markers_recalibration_and_render_prepare_are_structured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            marker = run_av_tool_call(
                {
                    "tool": "time_marker_add",
                    "args": {
                        "template_id": "edge_river",
                        "source_artifact": "edge.mp4",
                        "time_seconds": 72,
                        "note": "river too thick",
                        "target": "river_height_ratio",
                        "direction": "decrease",
                    },
                },
                storage_root=storage,
            )
            self.assertTrue(marker["ok"])
            self.assertEqual(marker["result"]["marker"]["time_seconds"], 72)

            note = run_av_tool_call(
                {
                    "tool": "recalibration_add_note",
                    "args": {
                        "template_id": "edge_river",
                        "source_artifact": "edge.mp4",
                        "time_seconds": 204,
                        "note": "colors should calm down",
                    },
                },
                storage_root=storage,
            )
            self.assertTrue(note["ok"])

            prepare = run_av_tool_call(
                {
                    "tool": "video_prepare_full_render",
                    "args": {
                        "template": {
                            "name": "Edge River",
                            "renderer": "edge_audio_river",
                            "timeline": {"duration_seconds": 278.32, "fps": 30},
                        }
                    },
                },
                storage_root=storage,
            )
            self.assertTrue(prepare["ok"])
            self.assertEqual(prepare["result"]["job"]["status"], "prepared_requires_human_execute")
            self.assertTrue(Path(prepare["result"]["manifest"]["path"]).exists())

            event_lines = (storage / "events" / "av_recalibration.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(event_lines), 2)
            self.assertEqual(json.loads(event_lines[0])["kind"], "time_marker")

    def test_audio_extract_features_reads_wav_and_writes_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            wav_path = storage / "tiny.wav"
            sample_rate = 8000
            samples = bytearray()
            for index in range(sample_rate // 2):
                value = int(math.sin(index / sample_rate * math.tau * 220) * 16000)
                samples.extend(value.to_bytes(2, "little", signed=True))
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(bytes(samples))

            result = run_av_tool_call(
                {
                    "tool": "audio_extract_features",
                    "args": {
                        "audio_path": str(wav_path),
                        "fps": 10,
                        "max_feature_frames": 3,
                    },
                },
                storage_root=storage,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["result"]["feature_count"], 5)
            self.assertEqual(result["result"]["features_returned"], 3)
            self.assertTrue(Path(result["result"]["feature_path"]).exists())
            self.assertGreater(result["result"]["summary"]["max_rms"], 0)

    def test_audio_analyze_levels_uses_ffmpeg_and_template_library(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            wav_path = storage / "peaks_and_valleys.wav"
            sample_rate = 8000
            samples = bytearray()
            for index in range(sample_rate * 2):
                second = index / sample_rate
                amp = 0.02 if second < 0.5 else 0.9 if second < 1.0 else 0.08 if second < 1.5 else 0.65
                value = int(math.sin(index / sample_rate * math.tau * 110) * amp * 30000)
                samples.extend(value.to_bytes(2, "little", signed=True))
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(bytes(samples))

            signals = run_av_tool_call(
                {
                    "tool": "audio_analyze_levels",
                    "args": {
                        "audio_path": str(wav_path),
                        "fps": 10,
                        "sample_rate": sample_rate,
                        "section_seconds": 0.5,
                        "max_signal_frames": 20,
                    },
                },
                storage_root=storage,
            )

            self.assertTrue(signals["ok"])
            self.assertTrue(Path(signals["result"]["signal_path"]).exists())
            self.assertGreater(signals["result"]["summary"]["peak_count"], 0)
            self.assertGreater(signals["result"]["summary"]["valley_count"], 0)
            self.assertGreater(signals["result"]["section_count"], 1)
            pattern_ids = {pattern["pattern_id"] for pattern in signals["result"]["recommended_patterns"]}
            self.assertIn("random_geometry_shards", pattern_ids)
            self.assertIn("quiet_valley_drift", pattern_ids)

            template = run_av_tool_call(
                {
                    "tool": "template_from_audio_signals",
                    "args": {
                        "name": "signal_template.json",
                        "signal_path": signals["result"]["signal_path"],
                        "prompt": "peaks make random geometry",
                    },
                },
                storage_root=storage,
            )

            self.assertTrue(template["ok"])
            self.assertEqual(template["result"]["template"]["renderer"], "audio_geometry_field")
            self.assertEqual(template["result"]["template"]["time_distance"]["source"], "ffmpeg_audio_signal")
            self.assertIn("signal_source", template["result"]["template"])
            self.assertTrue(Path(template["result"]["path"]).exists())

    def test_trueaudio_log_pre_sound_writes_state_manifest_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            wav_path = storage / "pre_sound.wav"
            sample_rate = 8000
            samples = bytearray()
            for index in range(sample_rate // 2):
                left = int(math.sin(index / sample_rate * math.tau * 220) * 16000)
                right = int(math.sin(index / sample_rate * math.tau * 330) * 8000)
                samples.extend(left.to_bytes(2, "little", signed=True))
                samples.extend(right.to_bytes(2, "little", signed=True))
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(bytes(samples))

            result = run_av_tool_call(
                {
                    "tool": "trueaudio_log_pre_sound",
                    "args": {
                        "audio_path": str(wav_path),
                        "run_id": "tool_trueaudio",
                        "fps": 10,
                        "sample_rate": sample_rate,
                    },
                },
                storage_root=storage,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["result"]["frame_count"], 5)
            self.assertTrue(Path(result["result"]["state_jsonl"]).exists())
            self.assertTrue(Path(result["result"]["manifest_json"]).exists())
            manifest = json.loads(Path(result["result"]["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["decode_stage"], "decoded_pcm_pre_output")
            self.assertFalse(manifest["boundary"]["raw_audio_saved"])
            self.assertTrue(Path(result["receipt"]["path"]).exists())

    def test_trueaudio_machine_pre_sound_routes_through_tool_bus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            with patch("truevision_runtime.av_tools.av_tool_runner.log_machine_pre_sound_state") as mocked:
                mocked.return_value = {
                    "schema_version": "trueaudio_machine_pre_sound_log_result_v1",
                    "run_id": "machine_tool",
                    "frame_count": 3,
                    "manifest_json": str(storage / "manifests" / "machine.json"),
                    "receipt_json": str(storage / "receipts" / "machine.json"),
                }

                result = run_av_tool_call(
                    {
                        "tool": "trueaudio_log_machine_pre_sound",
                        "args": {
                            "run_id": "machine_tool",
                            "duration_seconds": 0.1,
                            "fps": 30,
                        },
                    },
                    storage_root=storage,
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["result"]["schema_version"], "trueaudio_machine_pre_sound_log_result_v1")
            mocked.assert_called_once()
            self.assertEqual(mocked.call_args.kwargs["duration_seconds"], 0.1)

    def test_truespeech_detect_segments_routes_through_tool_bus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            state_path = storage / "voice.trueaudio.npz"
            state_path.write_bytes(b"placeholder")
            with patch("truevision_runtime.av_tools.av_tool_runner.detect_speech_segments_from_replayable_state") as mocked:
                mocked.return_value = {
                    "schema_version": "truespeech_detection_result_v1",
                    "run_id": "voice_detect",
                    "segments": [],
                    "summary": {"speech_segment_count": 0},
                    "manifest_json": str(storage / "manifests" / "voice.json"),
                    "receipt_json": str(storage / "receipts" / "voice.json"),
                }

                result = run_av_tool_call(
                    {
                        "tool": "truespeech_detect_segments",
                        "args": {
                            "state": str(state_path),
                            "run_id": "voice_detect",
                            "speech_threshold": 0.52,
                            "min_segment_seconds": 0.2,
                        },
                    },
                    storage_root=storage,
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["result"]["schema_version"], "truespeech_detection_result_v1")
            mocked.assert_called_once()
            self.assertEqual(mocked.call_args.args[0], state_path)
            self.assertEqual(mocked.call_args.kwargs["speech_threshold"], 0.52)
            self.assertEqual(mocked.call_args.kwargs["min_segment_seconds"], 0.2)

    def test_replayable_file_and_lyric_alignment_route_through_tool_bus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            audio_path = storage / "song.wav"
            audio_path.write_bytes(b"placeholder")
            with patch("truevision_runtime.av_tools.av_tool_runner.log_file_replayable_audio_state") as mocked_file:
                mocked_file.return_value = {
                    "schema_version": "trueaudio_file_replayable_state_log_result_v1",
                    "run_id": "song_state",
                    "state_npz": str(storage / "song.trueaudio.npz"),
                }

                result = run_av_tool_call(
                    {
                        "tool": "trueaudio_log_file_replayable",
                        "args": {
                            "audio_path": str(audio_path),
                            "run_id": "song_state",
                            "sample_rate": 48000,
                            "max_seconds": 12,
                        },
                    },
                    storage_root=storage,
                )

            self.assertTrue(result["ok"])
            mocked_file.assert_called_once()
            self.assertEqual(mocked_file.call_args.args[0], audio_path)
            self.assertEqual(mocked_file.call_args.kwargs["max_seconds"], 12.0)

            segments_path = storage / "segments.json"
            segments_path.write_text('{"segments":[]}', encoding="utf-8")
            with patch("truevision_runtime.av_tools.av_tool_runner.align_lyrics_to_speech_segments") as mocked_align:
                mocked_align.return_value = {
                    "schema_version": "truespeech_lyric_alignment_result_v1",
                    "run_id": "song_align",
                    "line_count": 0,
                }

                aligned = run_av_tool_call(
                    {
                        "tool": "truespeech_align_lyrics_candidate",
                        "args": {
                            "segments": str(segments_path),
                            "lyrics_text": "Baby come and rescue me",
                            "run_id": "song_align",
                        },
                    },
                    storage_root=storage,
                )

            self.assertTrue(aligned["ok"])
            mocked_align.assert_called_once()
            self.assertEqual(mocked_align.call_args.args[0], segments_path)
            self.assertEqual(mocked_align.call_args.kwargs["lyrics_text"], "Baby come and rescue me")


if __name__ == "__main__":
    unittest.main()
