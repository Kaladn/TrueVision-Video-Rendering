from __future__ import annotations

import json
import math
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "truevision_capture_rs"
BIN = NATIVE / "target" / "release" / "truevision_phoenix_flats_state_rs.exe"


def write_wav(path: Path, *, frequency: float, sample_rate: int = 8000, seconds: float = 0.5) -> None:
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        payload = bytearray()
        for index in range(frames):
            sample = int(math.sin(index * math.tau * frequency / sample_rate) * 15000)
            payload.extend(sample.to_bytes(2, "little", signed=True))
        handle.writeframes(bytes(payload))


class NativePhoenixFlatsStateRsTests(unittest.TestCase):
    def test_native_phoenix_flats_renderer_uses_stems_without_external_artifacts(self):
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "truevision_phoenix_flats_state_rs"],
            cwd=NATIVE,
            check=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            master = work / "master.wav"
            write_wav(master, frequency=220.0)
            stems_dir = work / "stems"
            stems_dir.mkdir()
            stem_names = [
                "Lead Vocals",
                "Backing Vocals",
                "Drums",
                "Bass",
                "Keyboard",
                "Percussion",
                "Synth",
                "Other",
            ]
            for offset, stem_name in enumerate(stem_names):
                write_wav(stems_dir / f"{offset} {stem_name}.wav", frequency=110.0 + offset * 31.0)

            output_root = work / "out"
            run_id = "phoenix_flats_smoke"
            subprocess.run(
                [
                    str(BIN),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    run_id,
                    "--audio",
                    str(master),
                    "--stems-dir",
                    str(stems_dir),
                    "--width",
                    "160",
                    "--height",
                    "90",
                    "--fps",
                    "5",
                    "--duration",
                    "0.5",
                    "--video-encoder",
                    "libx264",
                    "--state-log-every",
                    "1",
                ],
                cwd=ROOT,
                check=True,
            )

            run_dir = output_root / run_id
            video = run_dir / f"{run_id}.mp4"
            manifest_path = run_dir / f"{run_id}_manifest.json"
            state_path = run_dir / f"{run_id}_frame_state.jsonl"
            self.assertTrue(video.exists(), video)
            self.assertGreater(video.stat().st_size, 1000)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["renderer"], "rust")
            self.assertEqual(manifest["preset"], "phoenix_from_the_flats_state_v0")
            self.assertTrue(manifest["boundary"]["stems_drive_visual_lanes"])
            self.assertFalse(manifest["boundary"]["external_visual_assets_used"])
            self.assertFalse(manifest["boundary"]["openai_generation_used"])
            self.assertFalse(manifest["boundary"]["literal_phoenix_spam"])
            self.assertTrue(manifest["boundary"]["state_transform_arc_logged"])
            self.assertEqual(
                manifest["state_transform_arc"]["phase_names"],
                [
                    "lineart_damage_state",
                    "witness_expansion",
                    "dual_descent",
                    "impact_transform",
                    "regrowth_wave",
                    "healed_forest_state",
                ],
            )

            state_lines = [
                json.loads(line)
                for line in state_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(state_lines), 2)
            self.assertEqual(state_lines[0]["state_transform_arc"]["phase_name"], "lineart_damage_state")
            self.assertEqual(state_lines[-1]["state_transform_arc"]["phase_name"], "healed_forest_state")
            self.assertAlmostEqual(state_lines[-1]["state_transform_arc"]["healing_color_ratio"], 1.0)
            self.assertAlmostEqual(state_lines[-1]["state_transform_arc"]["regrowth_ratio"], 1.0)
            self.assertIn("lineart_world_mask", state_lines[0]["state_layers"])
            self.assertIn("damaged_city_silhouette", state_lines[0]["state_layers"])
            self.assertIn("dual_phoenix_vector_field", state_lines[-1]["state_layers"])
            self.assertIn("forest_regrowth_mask", state_lines[-1]["state_layers"])
            self.assertIn("clear_water_reflection_return", state_lines[-1]["state_layers"])
            state_line = json.dumps(state_lines[0], separators=(",", ":"))
            self.assertIn('"main_gold_ember_thread"', state_line)
            self.assertIn('"answering_rose_ember_thread"', state_line)
            self.assertIn('"phoenix_heat_veil"', state_line)
            self.assertIn('"generated_media_is_evidence":false', state_line)


if __name__ == "__main__":
    unittest.main()
