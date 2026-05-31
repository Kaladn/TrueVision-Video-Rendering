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
BIN = NATIVE / "target" / "release" / "truevision_distant_love_state_soft_rs.exe"


def write_wav(path: Path, *, frequency: float, sample_rate: int = 8000, seconds: float = 0.6) -> None:
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        payload = bytearray()
        for index in range(frames):
            sample = int(math.sin(index * math.tau * frequency / sample_rate) * 16000)
            payload.extend(sample.to_bytes(2, "little", signed=True))
        handle.writeframes(bytes(payload))


class NativeDistantLoveStateSoftRsTests(unittest.TestCase):
    def test_native_soft_state_renderer_uses_stem_directory_without_laser_preset(self):
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "truevision_distant_love_state_soft_rs"],
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
                "Guitar",
                "Keyboard",
                "Percussion",
                "Strings",
                "Synth",
                "Other",
                "Brass",
                "Woodwinds",
            ]
            for offset, stem_name in enumerate(stem_names):
                write_wav(stems_dir / f"{offset} {stem_name}.wav", frequency=110.0 + offset * 37.0)

            output_root = work / "out"
            run_id = "distant_love_soft_smoke"
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
                    "0.6",
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
            self.assertEqual(manifest["preset"], "distant_love_state_soft")
            self.assertEqual(manifest["renderer"], "rust")
            self.assertTrue(manifest["boundary"]["stem_directory_controls"])
            self.assertFalse(manifest["boundary"]["laser_show_preset"])
            self.assertTrue(manifest["boundary"]["soft_terror_balance"])

            state_line = state_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn('"lead_vocal_ribbon"', state_line)
            self.assertIn('"strings_fate_arcs"', state_line)
            self.assertIn('"keyboard_harmonic_lattice"', state_line)
            self.assertIn('"soft_terror_balance"', state_line)
            self.assertIn('"activity_scale"', state_line)


if __name__ == "__main__":
    unittest.main()
