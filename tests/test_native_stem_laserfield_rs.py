from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "truevision_capture_rs"
BIN = NATIVE / "target" / "release" / "truevision_stem_laserfield_rs.exe"


def write_wav(path: Path, *, frequency: float, sample_rate: int = 8000, seconds: float = 0.4) -> None:
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        payload = bytearray()
        for index in range(frames):
            sample = int(math.sin(index * math.tau * frequency / sample_rate) * 18000)
            payload.extend(sample.to_bytes(2, "little", signed=True))
        handle.writeframes(bytes(payload))


class NativeStemLaserfieldRsTests(unittest.TestCase):
    def test_native_stem_laserfield_renders_stem_driven_receipted_video_without_python_loop(self):
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "truevision_stem_laserfield_rs"],
            cwd=NATIVE,
            check=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            master = work / "master.wav"
            write_wav(master, frequency=220.0)
            stems_zip = work / "stems.zip"
            stem_names = ["Drums", "Bass", "Guitar", "Vocals", "Synth", "FX"]
            with zipfile.ZipFile(stems_zip, "w") as archive:
                for offset, stem_name in enumerate(stem_names):
                    stem_path = work / f"Becoming the Wolf ({stem_name}).wav"
                    write_wav(stem_path, frequency=110.0 + offset * 73.0)
                    archive.write(stem_path, arcname=stem_path.name)

            output_root = work / "out"
            run_id = "native_stem_smoke"
            subprocess.run(
                [
                    str(BIN),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    run_id,
                    "--audio",
                    str(master),
                    "--stems-zip",
                    str(stems_zip),
                    "--width",
                    "160",
                    "--height",
                    "90",
                    "--fps",
                    "5",
                    "--duration",
                    "0.4",
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
            self.assertFalse(manifest["boundary"]["python_render_loop"])
            self.assertTrue(manifest["boundary"]["stems_drive_visual_lanes"])
            self.assertTrue(manifest["boundary"]["visible_stem_meter_overlay"])
            self.assertTrue(manifest["boundary"]["vocal_stem_visual_lane"])
            self.assertEqual(manifest["boundary"]["guitar_laser_alpha"], 0.35)
            self.assertEqual(manifest["banner"]["position"], "lower_scrolling")
            state_line = state_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn('"stem_controls"', state_line)
            self.assertIn('"Guitar"', state_line)
            self.assertIn('"visible_stem_lanes"', state_line)
            self.assertIn('"vocal_lane"', state_line)
            self.assertIn('"voice_pressure"', state_line)


if __name__ == "__main__":
    unittest.main()
