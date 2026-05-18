from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "ui" / "truevision_state_media_studio.html"


class TrueVisionStateMediaStudioHtmlTests(unittest.TestCase):
    def test_html_exposes_local_qwen_adapter_controls(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("local-qwen-32b", html)
        self.assertIn("qwenModel", html)
        self.assertIn("qwenEndpoint", html)
        self.assertIn("http://127.0.0.1:11434/v1/chat/completions", html)

    def test_html_can_call_local_qwen_without_backend_wiring(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("async function callLocalQwen", html)
        self.assertIn("fetch(endpoint", html)
        self.assertIn("Authorization", html)
        self.assertIn("parseQwenStateResponse", html)

    def test_local_qwen_prompt_preserves_securecore_boundary(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("Generated state media is synthetic, not evidence.", html)
        self.assertIn("Return only JSON", html)
        self.assertIn("truevision_state_media_draft", html)


if __name__ == "__main__":
    unittest.main()
