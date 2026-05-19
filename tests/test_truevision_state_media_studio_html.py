from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "ui" / "truevision_state_media_studio.html"


class TrueVisionStateMediaStudioHtmlTests(unittest.TestCase):
    def test_html_exposes_local_qwen_adapter_controls(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("local-qwen-32b", html)
        self.assertIn("qwenProvider", html)
        self.assertIn("qwenModel", html)
        self.assertIn("qwenEndpoint", html)
        self.assertIn("qwenUseProxy", html)
        self.assertIn("qwenProxyEndpoint", html)
        self.assertIn("qwen3-coder:30b", html)
        self.assertIn("http://127.0.0.1:8765/api/local-llm/chat", html)
        self.assertIn("http://127.0.0.1:11434/api/chat", html)
        self.assertIn("ollama_native", html)
        self.assertIn("openai_compatible", html)

    def test_html_uses_backend_wiring_for_state_operations(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("/api/state/request", html)
        self.assertIn("/api/state/plan", html)
        self.assertIn("/api/record/prepare", html)
        self.assertIn("/api/assistant/message", html)
        self.assertIn("/api/files", html)
        self.assertIn("saveRequestToServer", html)
        self.assertIn("sendAssistantMessage", html)
        self.assertIn("runAssistantAction", html)
        self.assertIn("prepareRecordCommand", html)
        self.assertIn("refreshServerFiles", html)

    def test_html_can_call_local_qwen_through_studio_proxy(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("async function callLocalQwen", html)
        self.assertIn("fetch(endpoint", html)
        self.assertIn("if (apiKey && !useProxy)", html)
        self.assertIn("buildOllamaChatBody", html)
        self.assertIn("buildOpenAiChatBody", html)
        self.assertIn("parseQwenStateResponse", html)

    def test_center_panel_is_an_operations_log_not_generic_chat(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("Run Log", html)
        self.assertIn("Catbot", html)
        self.assertIn("Tell Catbot", html)
        self.assertIn("Compile", html)
        self.assertIn("Clear Log", html)
        self.assertIn("server-backed", html)

    def test_local_qwen_prompt_preserves_securecore_boundary(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("Generated state media is synthetic, not evidence.", html)
        self.assertIn("Return only JSON", html)
        self.assertIn("truevision_state_media_draft", html)

    def test_record_start_zone_uses_manual_minutes(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("Record Start Zone", html)
        self.assertIn('id="durationMinutes"', html)
        self.assertIn('type="number"', html)
        self.assertIn("duration_minutes", html)
        self.assertIn('duration_seconds: Math.round(number("durationMinutes") * 60)', html)
        self.assertIn("start_delay_minutes", html)
        self.assertNotIn('id="duration" type="range"', html)

    def test_countdown_overlay_is_state_aware_and_manifested(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("Countdown Overlay", html)
        self.assertIn('id="countdownEnabled"', html)
        self.assertIn('id="countdownSeconds"', html)
        self.assertIn("countdown_enabled", html)
        self.assertIn("countdown_seconds", html)
        self.assertIn("overlay_position", html)
        self.assertIn("contrast_mode_used", html)
        self.assertIn("record_start_time", html)
        self.assertIn("chooseCountdownContrastMode", html)


if __name__ == "__main__":
    unittest.main()
