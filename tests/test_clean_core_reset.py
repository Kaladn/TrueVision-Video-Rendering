import unittest
from pathlib import Path

from scripts import truevision_studio_server as studio


class TrueVisionCleanCoreResetTests(unittest.TestCase):
    def test_chat_runtime_is_not_installed(self):
        self.assertNotIn("chats", studio.STORAGE_LANES)
        self.assertFalse(hasattr(studio, "append_chat_message"))
        self.assertFalse(hasattr(studio, "read_chat_log"))

        status = studio.core_runtime_status()
        self.assertEqual(status.get("ui_runtime"), "not_installed")
        self.assertEqual(status.get("chat_runtime"), "not_installed")
        self.assertEqual(status.get("memory_runtime"), "not_installed")
        self.assertTrue(status.get("future_port"))

    def test_embedded_model_contract_folder_is_not_installed(self):
        root = Path(__file__).resolve().parents[1]

        self.assertFalse((root / "docs" / ("q" + "wen")).exists())


if __name__ == "__main__":
    unittest.main()
