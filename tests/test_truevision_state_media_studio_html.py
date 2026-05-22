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

    def test_html_defaults_to_working_local_qwen_route_only(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn('llm: "local-qwen-32b"', html)
        self.assertIn('class="active" data-value="local-qwen-32b">Qwen 32B</button>', html)
        self.assertNotIn('data-value="api_llm"', html)
        self.assertNotIn('data-value="clearspeak"', html)
        self.assertNotIn('data-group="remote"', html)

    def test_html_redirects_file_protocol_to_local_server(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn('window.location.protocol === "file:"', html)
        self.assertIn('http://127.0.0.1:8765/', html)
        self.assertIn("setBusy", html)
        self.assertIn("Catbot is working", html)

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
        self.assertIn("async function callLocalQwenText", html)
        self.assertIn("fetch(endpoint", html)
        self.assertIn("if (apiKey && !useProxy)", html)
        self.assertIn("buildOllamaChatBody", html)
        self.assertIn("buildOpenAiChatBody", html)
        self.assertIn("parseQwenStateResponse", html)

    def test_catbot_has_conversation_lane_not_only_action_log(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("buildCatbotSystemPrompt", html)
        self.assertIn('action === "qwen_chat"', html)
        self.assertIn("callCatbotQwen", html)
        self.assertIn("Qwen Project Chat", html)
        self.assertIn("Talk with Qwen", html)
        self.assertIn("available_tools", html)

    def test_center_panel_is_an_operations_log_not_generic_chat(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("Qwen Project Chat", html)
        self.assertIn("Daily chat log", html)
        self.assertIn("Talk with Qwen", html)
        self.assertIn("Compile", html)
        self.assertIn("Clear Log", html)
        self.assertIn("server-backed", html)

    def test_local_qwen_prompt_preserves_truevision_boundary(self):
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

    def test_html_has_daily_chat_and_template_workspace(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("/api/chat/today", html)
        self.assertIn("/api/chat/log", html)
        self.assertIn("loadDailyChat", html)
        self.assertIn("logChatMessage", html)
        self.assertIn("Daily chat log", html)
        self.assertIn("templateJson", html)
        self.assertIn("Render Template", html)

    def test_html_exposes_template_crud_and_song_duration_match(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("/api/templates", html)
        self.assertIn("/api/templates/save", html)
        self.assertIn("/api/templates/delete", html)
        self.assertIn("/api/media/probe", html)
        self.assertIn("audioPath", html)
        self.assertIn("syncDurationToAudio", html)
        self.assertIn("saveTemplate", html)
        self.assertIn("deleteTemplate", html)
        self.assertIn("duration_source", html)

    def test_html_tells_qwen_about_av_only_tool_calls(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("/api/av-tools", html)
        self.assertIn("/api/av-tools/call", html)
        self.assertIn("AV-only validated tools", html)
        self.assertIn("video_render_preview", html)
        self.assertIn("audio_analyze_levels", html)
        self.assertIn("audio_extract_features", html)
        self.assertIn("template_from_audio_signals", html)
        self.assertIn("template_patch", html)
        self.assertIn("AI cannot execute directly", html)
        self.assertIn("extractAvToolRequest", html)
        self.assertIn("runAvToolRequest", html)
        self.assertIn("available_av_tools", html)
        self.assertIn("human_confirmed: false", html)

    def test_html_exposes_reusable_studio_tools_and_presets(self):
        html = HTML.read_text(encoding="utf-8")

        self.assertIn("Render Preset Library", html)
        self.assertIn("/api/render-presets", html)
        self.assertIn("refreshPresets", html)
        self.assertIn("loadPreset", html)
        self.assertIn("selectedPreset", html)
        self.assertIn("source_snap_tool", html)
        self.assertIn("existing_state_animator", html)
        self.assertIn("electric_glow_intensity_animator", html)
        self.assertIn("spectrum_audio_reactive_city", html)
        self.assertIn("frame_diff_replay_accuracy", html)
        self.assertIn("manifest_browser", html)
        self.assertIn("render_preset_library", html)
        self.assertIn("local_qwen_controller", html)


if __name__ == "__main__":
    unittest.main()
