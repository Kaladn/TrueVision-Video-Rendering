import unittest

from truevision_runtime.learning_intake.youtube_cdp import (
    build_video_play_expression,
    build_video_state_expression,
)


class YouTubeCdpControlTests(unittest.TestCase):
    def test_play_expression_sets_time_mutes_and_calls_play(self):
        expression = build_video_play_expression(444.0)

        self.assertIn("document.querySelector('video')", expression)
        self.assertIn("v.currentTime = 444.000", expression)
        self.assertIn("v.muted = true", expression)
        self.assertIn("await v.play()", expression)
        self.assertIn("for (let attempt = 0; attempt < 3; attempt++)", expression)
        self.assertIn("targetReached", expression)
        self.assertIn("currentTime", expression)

    def test_state_expression_reports_time_paused_ready_and_url(self):
        expression = build_video_state_expression()

        self.assertIn("document.querySelector('video')", expression)
        self.assertIn("location.href", expression)
        self.assertIn("paused", expression)
        self.assertIn("readyState", expression)


if __name__ == "__main__":
    unittest.main()
