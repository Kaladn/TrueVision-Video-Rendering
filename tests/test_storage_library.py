import tempfile
import unittest
from pathlib import Path

from truevision_runtime.storage_library import ensure_storage_library, storage_report


class StorageLibraryTests(unittest.TestCase):
    def test_ensure_storage_library_creates_tidy_media_lanes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_storage_library(Path(tmpdir))
            root = Path(result["root"])

            self.assertEqual(result["clip_unit_minutes"], 20)
            self.assertTrue((root / "library" / "README.md").exists())
            self.assertTrue((root / "library" / "indexes" / "library_index.json").exists())
            self.assertTrue((root / "library" / "source_audio" / "wav").is_dir())
            self.assertTrue((root / "library" / "source_video" / "mp4").is_dir())
            self.assertTrue((root / "library" / "source_stills" / "jpg").is_dir())
            self.assertTrue((root / "library" / "capture_units" / "20_minute" / "runs").is_dir())
            self.assertTrue((root / "library" / "signature_profiles" / "fog").is_dir())
            self.assertTrue((root / "library" / "renders" / "full").is_dir())

    def test_storage_report_lists_created_lanes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ensure_storage_library(root)

            rows = storage_report(root)

            self.assertTrue(any(row["lane"] == "library" for row in rows))
            self.assertTrue(any(row["lane"] == "receipts" for row in rows))


if __name__ == "__main__":
    unittest.main()
