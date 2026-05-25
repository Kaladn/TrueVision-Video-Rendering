import unittest

from truevision_runtime.learning_intake.youtube_metadata import extract_youtube_metadata_from_html


class YouTubeLearningMetadataTests(unittest.TestCase):
    def test_extracts_length_seconds_and_title_from_watch_html(self):
        html = """
        <html><head><title>Smoke Teacher - YouTube</title></head>
        <script>{"lengthSeconds":"3600","title":"Fire Particles Overlay 4K"}</script>
        </html>
        """

        metadata = extract_youtube_metadata_from_html(html, fallback_title="Fallback")

        self.assertEqual(metadata["duration_seconds"], 3600.0)
        self.assertEqual(metadata["video_title"], "Fire Particles Overlay 4K")

    def test_falls_back_to_html_title_when_json_title_missing(self):
        html = '<html><head><title>Rain Teacher - YouTube</title></head><body>"lengthSeconds": "42"</body></html>'

        metadata = extract_youtube_metadata_from_html(html)

        self.assertEqual(metadata["duration_seconds"], 42.0)
        self.assertEqual(metadata["video_title"], "Rain Teacher")


if __name__ == "__main__":
    unittest.main()
