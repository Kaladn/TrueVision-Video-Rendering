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
BIN = NATIVE / "target" / "release" / "truevision_cortex_photo_state_rs.exe"


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


def write_ppm(path: Path, width: int = 64, height: int = 96) -> None:
    data = bytearray(f"P6\n{width} {height}\n255\n", "ascii")
    for y in range(height):
        for x in range(width):
            gold = x > width // 3 and x < width * 2 // 3 and y > height // 5 and y < height * 4 // 5
            text = y in {12, 13, 70, 71}
            if gold:
                data.extend([210, 150, 46])
            elif text:
                data.extend([220, 218, 204])
            else:
                data.extend([8 + x % 8, 9 + y % 8, 11 + (x + y) % 10])
    path.write_bytes(bytes(data))


class NativeCortexPhotoStateRsTests(unittest.TestCase):
    def test_native_photo_state_renderer_uses_artifact_groups_and_artwork_glyph_schema(self):
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "truevision_cortex_photo_state_rs"],
            cwd=NATIVE,
            check=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            master = work / "master.wav"
            write_wav(master, frequency=220.0)
            image = work / "poster.ppm"
            write_ppm(image)
            stems_dir = work / "stems"
            stems_dir.mkdir()
            for offset, stem_name in enumerate(
                ["Lead Vocals", "Backing Vocals", "Drums", "Bass", "Guitar", "Percussion", "Synth", "Other"]
            ):
                write_wav(stems_dir / f"{offset} {stem_name}.wav", frequency=110.0 + offset * 51.0)

            output_root = work / "out"
            run_id = "cortex_photo_state_smoke"
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
                    "--image",
                    str(image),
                    "--plate-mode",
                    "portrait_fit",
                    "--width",
                    "108",
                    "--height",
                    "192",
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
            self.assertEqual(manifest["preset"], "cortex_photo_state_transform")
            self.assertTrue(manifest["boundary"]["photo_state_transform"])
            self.assertTrue(manifest["boundary"]["glyph_schema_from_artwork"])
            self.assertTrue(manifest["boundary"]["technical_identity_banner"])
            self.assertFalse(manifest["boundary"]["moving_objects"])

            state_line = state_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn('"artifact_groups"', state_line)
            self.assertIn('"gold_edge_flame"', state_line)
            self.assertIn('"line_art_negative_trace"', state_line)
            self.assertIn('"camera_pressure"', state_line)
            self.assertIn('"banner"', state_line)
            self.assertIn('"hardware"', state_line)
            self.assertIn('"artwork_color_schema"', state_line)


if __name__ == "__main__":
    unittest.main()
