import unittest

from truevision_studio_server import (
    build_downstream_payload,
    normalize_provider,
    validate_local_endpoint,
)


class TrueVisionStudioServerTests(unittest.TestCase):
    def test_validate_local_endpoint_allows_loopback_only(self):
        self.assertEqual(
            validate_local_endpoint("http://127.0.0.1:11434/api/chat"),
            "http://127.0.0.1:11434/api/chat",
        )
        self.assertEqual(
            validate_local_endpoint("http://localhost:11434/v1/chat/completions"),
            "http://localhost:11434/v1/chat/completions",
        )
        with self.assertRaises(ValueError):
            validate_local_endpoint("https://example.com/api/chat")

    def test_normalize_provider_defaults_to_ollama(self):
        self.assertEqual(normalize_provider(""), "ollama_native")
        self.assertEqual(normalize_provider("ollama_native"), "ollama_native")
        self.assertEqual(normalize_provider("openai_compatible"), "openai_compatible")

    def test_build_downstream_payload_for_ollama(self):
        payload = build_downstream_payload(
            provider="ollama_native",
            model="qwen3-coder:30b",
            system_prompt="system",
            request={"prompt": "make state"},
        )

        self.assertEqual(payload["model"], "qwen3-coder:30b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn("options", payload)

    def test_build_downstream_payload_for_openai_compatible(self):
        payload = build_downstream_payload(
            provider="openai_compatible",
            model="qwen3-coder:30b",
            system_prompt="system",
            request={"prompt": "make state"},
        )

        self.assertEqual(payload["model"], "qwen3-coder:30b")
        self.assertFalse(payload["stream"])
        self.assertIn("temperature", payload)
        self.assertNotIn("options", payload)


if __name__ == "__main__":
    unittest.main()
