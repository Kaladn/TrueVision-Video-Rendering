from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_phoenix_flats_story_phone_lab.py"


def _make_storyboard_sheet(path: Path) -> None:
    image = Image.new("L", (1536, 1024), 255)
    draw = ImageDraw.Draw(image)
    xs = [28, 286, 535, 779, 1015, 1271]
    top_y = 37
    bottom_y = 499
    w = 244
    h = 275
    for row, y in enumerate([top_y, bottom_y]):
        for col, x in enumerate(xs):
            draw.rectangle((x, y, x + w, y + h), outline=0, width=3)
            draw.line((x + 8, y + 8, x + w - 8, y + h - 8), fill=0, width=2)
            draw.text((x + 16, y + h + 24), f"{row * 6 + col + 1}. caption must not crop", fill=0)
    draw.text((545, 932), "WHAT FEELS RIGHT, IS RIGHT.  <3  FOR MY DAUGHTER.", fill=0)
    image.save(path)


def _write_wav(path: Path, *, frequency: float, sample_rate: int = 8000, seconds: float = 0.5) -> None:
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        payload = bytearray()
        for index in range(frames):
            sample = int(__import__("math").sin(index * __import__("math").tau * frequency / sample_rate) * 14000)
            payload.extend(sample.to_bytes(2, "little", signed=True))
        handle.writeframes(bytes(payload))


class PhoenixFlatsStoryPhoneLabTests(unittest.TestCase):
    def test_cli_exposes_storyboard_phone_controls(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--storyboard-sheet", result.stdout)
        self.assertIn("--crop-only", result.stdout)
        self.assertIn("--width", result.stdout)
        self.assertIn("--height", result.stdout)
        self.assertIn("--style-phrase-mix", result.stdout)
        self.assertIn("--visual-mode", result.stdout)
        self.assertIn("geometry_phoenix", result.stdout)

    def test_crop_only_writes_twelve_art_plates_and_phrase_style_strip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            sheet = work / "storyboard.png"
            _make_storyboard_sheet(sheet)
            out = work / "out"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--storyboard-sheet",
                    str(sheet),
                    "--output-root",
                    str(out),
                    "--run-id",
                    "crop_test",
                    "--crop-only",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            crop_dir = out / "crop_test" / "story_plates"
            plates = sorted(crop_dir.glob("scene_*.png"))
            self.assertEqual(len(plates), 12)
            first = Image.open(plates[0])
            self.assertEqual(first.size, (244, 275))
            first_pixels = first.load()
            self.assertEqual(first_pixels[2, 2], 0)
            manifest = (out / "crop_test" / "story_plates_manifest.json").read_text(encoding="utf-8")
            self.assertIn("what_feels_right_phrase_style_strip", manifest)
            self.assertIn("captions_excluded_from_plate_crops", manifest)
            self.assertTrue((crop_dir / "what_feels_right_phrase_style_strip.png").exists())

    def test_geometry_phoenix_mode_uses_no_story_pictures_until_final_phoenix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            sheet = work / "storyboard.png"
            _make_storyboard_sheet(sheet)
            master = work / "master.wav"
            _write_wav(master, frequency=220.0)
            stems_zip = work / "stems.zip"
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
            with zipfile.ZipFile(stems_zip, "w") as archive:
                for offset, stem in enumerate(stem_names):
                    stem_path = work / f"{offset} {stem}.wav"
                    _write_wav(stem_path, frequency=110.0 + offset * 33.0)
                    archive.write(stem_path, arcname=stem_path.name)
            out = work / "out"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--storyboard-sheet",
                    str(sheet),
                    "--audio",
                    str(master),
                    "--stems",
                    str(stems_zip),
                    "--output-root",
                    str(out),
                    "--run-id",
                    "geometry_test",
                    "--seconds",
                    "0.5",
                    "--width",
                    "180",
                    "--height",
                    "320",
                    "--visual-mode",
                    "geometry_phoenix",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            manifest_path = out / "geometry_test" / "geometry_test_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["visual_mode"], "geometry_phoenix")
            self.assertFalse(manifest["boundary"]["story_plates_visible_in_main_render"])
            self.assertTrue(manifest["boundary"]["sound_drives_generated_geometry"])
            self.assertTrue(manifest["boundary"]["final_phoenix_generated"])
            self.assertEqual(manifest["boundary"]["only_pictorial_figure"], "generated_phoenix_final_state")
            self.assertEqual(
                manifest["final_phoenix_style_contract"],
                "fiery_wing_arc_over_city_reflection_state",
            )


if __name__ == "__main__":
    unittest.main()
