import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from truevision_still_image_capture import capture_still_image


class TrueVisionStillImageCaptureTests(unittest.TestCase):
    def test_capture_still_image_writes_video_shaped_state_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "source.jpg"
            image = np.zeros((300, 400, 3), dtype=np.uint8)
            image[:, :, 0] = 30
            image[80:220, 120:280, 1] = 180
            image[120:180, 160:240, 2] = 240
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            result = capture_still_image(
                image_path=image_path,
                output_root=tmp / "out",
                run_id="still-test",
                frame_shape=(540, 960),
                grid_shape=(90, 160),
                block_shape=(9, 16),
                frames=3,
                fps=9.0,
            )

            run_dir = Path(result["run_dir"])
            manifest = json.loads((run_dir / "still-test_manifest.json").read_text())
            summary = json.loads((run_dir / "still-test_summary.json").read_text())
            records = (run_dir / "still-test_records.jsonl").read_text().splitlines()

            self.assertEqual(manifest["config"]["source_kind"], "still_image_as_video_state")
            self.assertEqual(manifest["records"]["frame_count"], 3)
            self.assertEqual(summary["frame_count"], 3)
            self.assertEqual(len(records), 3)
            self.assertEqual(summary["geometry"]["frame_shape"], [540, 960])
            self.assertEqual(summary["geometry"]["grid_shape"], [90, 160])
            self.assertEqual(manifest["cell_state"]["chunks"][0]["shape"], [3, 90, 160, 16])

            with np.load(manifest["cell_state"]["chunks"][0]["path"], allow_pickle=False) as data:
                self.assertEqual(tuple(data["cell_state"].shape), (3, 90, 160, 16))
                self.assertEqual(data["frame_numbers"].tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
