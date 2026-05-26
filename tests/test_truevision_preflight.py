import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from truevision_preflight import (
    build_preflight_report,
    check_executable,
    check_gitignore_outputs,
    check_python_module,
    check_python_version,
    compute_exit_code,
    report_as_text,
)


class TrueVisionPreflightTests(unittest.TestCase):
    def test_python_version_passes_for_supported_runtime(self):
        check = check_python_version((3, 11, 2), minimum=(3, 11))

        self.assertEqual(check["status"], "pass")
        self.assertEqual(check["id"], "python.version")

    def test_python_version_fails_for_old_runtime(self):
        check = check_python_version((3, 10, 9), minimum=(3, 11))

        self.assertEqual(check["status"], "fail")
        self.assertIn("Python 3.11+", check["message"])

    def test_executable_check_reports_download_hint_when_missing(self):
        check = check_executable(
            "ffmpeg",
            lookup=lambda name: None,
            install_url="https://ffmpeg.org/download.html",
        )

        self.assertEqual(check["status"], "fail")
        self.assertEqual(check["install_url"], "https://ffmpeg.org/download.html")

    def test_executable_check_passes_when_found(self):
        check = check_executable("cargo", lookup=lambda name: "C:/Rust/bin/cargo.exe")

        self.assertEqual(check["status"], "pass")
        self.assertEqual(check["path"], "C:/Rust/bin/cargo.exe")

    def test_python_module_check_uses_import_spec(self):
        check = check_python_module(
            "opencv-python",
            module_name="cv2",
            find_spec=lambda name: object(),
            install_url="https://pypi.org/project/opencv-python/",
        )

        self.assertEqual(check["status"], "pass")
        self.assertEqual(check["module"], "cv2")

    def test_python_module_check_fails_with_package_hint(self):
        check = check_python_module(
            "mss",
            module_name="mss",
            find_spec=lambda name: None,
            install_url="https://pypi.org/project/mss/",
        )

        self.assertEqual(check["status"], "fail")
        self.assertIn("pip install mss", check["message"])

    def test_gitignore_outputs_requires_generated_dirs_ignored(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".gitignore").write_text("outputs/**\nstorage/artifacts/**\n", encoding="utf-8")

            check = check_gitignore_outputs(root)

        self.assertEqual(check["status"], "pass")

    def test_gitignore_outputs_warns_when_missing_generated_dirs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".gitignore").write_text("*.pyc\n", encoding="utf-8")

            check = check_gitignore_outputs(root)

        self.assertEqual(check["status"], "warn")
        self.assertIn("outputs/**", check["missing_patterns"])

    def test_build_preflight_report_contains_summary_and_checks(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".gitignore").write_text(
                "outputs/**\nstorage/artifacts/**\nstorage/receipts/**\n",
                encoding="utf-8",
            )
            for relative in (
                "truevision_runtime",
                "trueaudio_runtime",
                "trueframegen",
                "native/truevision_capture_rs",
                "scripts",
                "tests",
            ):
                (root / relative).mkdir(parents=True)

            report = build_preflight_report(
                root,
                executable_lookup=lambda name: f"C:/bin/{name}.exe",
                find_spec=lambda name: object(),
                version_info=(3, 11, 8),
            )

        self.assertEqual(report["schema"], "truevision_preflight_report.v1")
        self.assertEqual(report["summary"]["fail"], 0)
        self.assertGreaterEqual(report["summary"]["pass"], 1)
        self.assertTrue(any(check["id"] == "tool.ffmpeg" for check in report["checks"]))

    def test_compute_exit_code_fails_only_on_fail_status(self):
        self.assertEqual(compute_exit_code([{"status": "pass"}, {"status": "warn"}]), 0)
        self.assertEqual(compute_exit_code([{"status": "pass"}, {"status": "fail"}]), 1)

    def test_text_report_includes_install_links_for_failures(self):
        text = report_as_text(
            {
                "summary": {"pass": 0, "warn": 0, "fail": 1},
                "checks": [
                    {
                        "id": "tool.ffmpeg",
                        "status": "fail",
                        "message": "ffmpeg not found",
                        "install_url": "https://ffmpeg.org/download.html",
                    }
                ],
            }
        )

        self.assertIn("tool.ffmpeg", text)
        self.assertIn("https://ffmpeg.org/download.html", text)


if __name__ == "__main__":
    unittest.main()
