import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from truevision_exact_photo_state_snap import write_exact_photo_state_snap


class TrueVisionExactPhotoStateSnapTests(unittest.TestCase):
    def test_exact_snap_reconstructs_decoded_pixels_without_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.png"
            rgb = np.zeros((36, 48, 3), dtype=np.uint8)
            rgb[:, :, 0] = np.arange(48, dtype=np.uint8)[None, :]
            rgb[:, :, 1] = np.arange(36, dtype=np.uint8)[:, None]
            rgb[10:28, 12:34, 2] = 240
            cv2.imwrite(str(source_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            result = write_exact_photo_state_snap(
                image_path=source_path,
                output_root=root / "out",
                run_id="snap-test",
                max_grid_shape=(18, 24),
            )

            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            with np.load(result["pixel_state_npz"], allow_pickle=False) as data:
                stored = data["rgb"]
            replay_bgr = cv2.imread(result["reconstructed_png"], cv2.IMREAD_COLOR)
            replay_rgb = cv2.cvtColor(replay_bgr, cv2.COLOR_BGR2RGB)

            self.assertTrue(np.array_equal(stored, rgb))
            self.assertTrue(np.array_equal(replay_rgb, rgb))
            self.assertEqual(manifest["exact_reconstruction"]["max_abs_error"], 0)
            self.assertEqual(manifest["exact_reconstruction"]["pixel_exact"], True)
            self.assertEqual(manifest["pixel_state"]["format"], "rgb_u8_lossless_decoded_pixels")
            self.assertEqual(manifest["cell_state"]["role"], "derived_truevision_telemetry_not_exact_photo")
            self.assertTrue(Path(result["source_copy"]).exists())


if __name__ == "__main__":
    unittest.main()
