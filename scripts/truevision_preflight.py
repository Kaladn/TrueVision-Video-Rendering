"""Local TrueVision prerequisite and repo health preflight.

This script reports what is present or missing. It does not install, patch,
open browsers, or mutate runtime state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]


def _check(
    *,
    check_id: str,
    status: str,
    message: str,
    **extra: Any,
) -> dict[str, Any]:
    payload = {"id": check_id, "status": status, "message": message}
    payload.update(extra)
    return payload


def check_python_version(
    version_info: tuple[int, int, int] | tuple[int, int],
    *,
    minimum: tuple[int, int] = (3, 11),
) -> dict[str, Any]:
    major, minor = int(version_info[0]), int(version_info[1])
    current = f"{major}.{minor}"
    required = f"{minimum[0]}.{minimum[1]}+"
    if (major, minor) >= minimum:
        return _check(
            check_id="python.version",
            status="pass",
            message=f"Python {current} satisfies Python {required}.",
            current=current,
            required=required,
        )
    return _check(
        check_id="python.version",
        status="fail",
        message=f"Python {required} is required; found Python {current}.",
        current=current,
        required=required,
        install_url="https://www.python.org/downloads/",
    )


def check_executable(
    name: str,
    *,
    lookup: Callable[[str], str | None] = shutil.which,
    install_url: str | None = None,
    required: bool = True,
) -> dict[str, Any]:
    path = lookup(name)
    if path:
        return _check(
            check_id=f"tool.{name}",
            status="pass",
            message=f"{name} found.",
            path=path,
        )
    status = "fail" if required else "warn"
    return _check(
        check_id=f"tool.{name}",
        status=status,
        message=f"{name} not found on PATH.",
        install_url=install_url,
    )


def check_python_module(
    package_name: str,
    *,
    module_name: str,
    find_spec: Callable[[str], Any | None] = importlib.util.find_spec,
    install_url: str | None = None,
) -> dict[str, Any]:
    if find_spec(module_name) is not None:
        return _check(
            check_id=f"python_module.{module_name}",
            status="pass",
            message=f"Python module {module_name} is importable.",
            package=package_name,
            module=module_name,
        )
    return _check(
        check_id=f"python_module.{module_name}",
        status="fail",
        message=f"Python module {module_name} is missing. Install with: pip install {package_name}",
        package=package_name,
        module=module_name,
        install_url=install_url,
    )


def check_path_exists(root: Path, relative: str, *, kind: str = "path") -> dict[str, Any]:
    path = root / relative
    check_id = "repo." + relative.replace("/", ".").replace("\\", ".")
    if path.exists():
        return _check(
            check_id=check_id,
            status="pass",
            message=f"{relative} exists.",
            path=str(path),
        )
    return _check(
        check_id=check_id,
        status="fail",
        message=f"Required {kind} missing: {relative}",
        path=str(path),
    )


def check_gitignore_outputs(root: Path) -> dict[str, Any]:
    gitignore = root / ".gitignore"
    required_patterns = ["outputs/**", "storage/artifacts/**"]
    if not gitignore.exists():
        return _check(
            check_id="gitignore.generated_outputs",
            status="warn",
            message=".gitignore is missing; generated media may be accidentally tracked.",
            missing_patterns=required_patterns,
        )
    text = gitignore.read_text(encoding="utf-8", errors="replace")
    missing = [pattern for pattern in required_patterns if pattern not in text]
    if missing:
        return _check(
            check_id="gitignore.generated_outputs",
            status="warn",
            message="Generated output ignore patterns are incomplete.",
            missing_patterns=missing,
        )
    return _check(
        check_id="gitignore.generated_outputs",
        status="pass",
        message="Generated output folders are ignored.",
        required_patterns=required_patterns,
    )


def summarize_checks(checks: Iterable[dict[str, Any]]) -> dict[str, int]:
    summary = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        status = str(check.get("status", "fail"))
        summary[status] = summary.get(status, 0) + 1
    return summary


def compute_exit_code(checks: Iterable[dict[str, Any]]) -> int:
    return 1 if any(check.get("status") == "fail" for check in checks) else 0


def build_preflight_report(
    root: str | Path = ROOT,
    *,
    executable_lookup: Callable[[str], str | None] = shutil.which,
    find_spec: Callable[[str], Any | None] = importlib.util.find_spec,
    version_info: tuple[int, int, int] | tuple[int, int] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    version = version_info or (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    checks = [
        check_python_version(version),
        check_executable("ffmpeg", lookup=executable_lookup, install_url="https://ffmpeg.org/download.html"),
        check_executable("cargo", lookup=executable_lookup, install_url="https://www.rust-lang.org/tools/install"),
        check_executable("code", lookup=executable_lookup, install_url="https://code.visualstudio.com/", required=False),
        check_python_module("numpy", module_name="numpy", find_spec=find_spec, install_url="https://pypi.org/project/numpy/"),
        check_python_module(
            "opencv-python",
            module_name="cv2",
            find_spec=find_spec,
            install_url="https://pypi.org/project/opencv-python/",
        ),
        check_python_module("mss", module_name="mss", find_spec=find_spec, install_url="https://pypi.org/project/mss/"),
        check_path_exists(root, "truevision_runtime", kind="directory"),
        check_path_exists(root, "trueaudio_runtime", kind="directory"),
        check_path_exists(root, "trueframegen", kind="directory"),
        check_path_exists(root, "native/truevision_capture_rs", kind="directory"),
        check_path_exists(root, "scripts", kind="directory"),
        check_path_exists(root, "tests", kind="directory"),
        check_gitignore_outputs(root),
    ]
    summary = summarize_checks(checks)
    return {
        "schema": "truevision_preflight_report.v1",
        "repo_root": str(root),
        "summary": summary,
        "exit_code": compute_exit_code(checks),
        "boundary": {
            "non_mutating": True,
            "installs_nothing": True,
            "opens_no_browser": True,
            "local_first": True,
        },
        "checks": checks,
    }


def report_as_text(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "TrueVision Preflight",
        f"pass={summary.get('pass', 0)} warn={summary.get('warn', 0)} fail={summary.get('fail', 0)}",
        "",
    ]
    for check in report.get("checks", []):
        status = str(check.get("status", "fail")).upper()
        lines.append(f"[{status}] {check.get('id')}: {check.get('message')}")
        path = check.get("path")
        if path:
            lines.append(f"  path: {path}")
        install_url = check.get("install_url")
        if install_url:
            lines.append(f"  link: {install_url}")
        missing = check.get("missing_patterns")
        if missing:
            lines.append(f"  missing: {', '.join(missing)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local TrueVision prerequisites and repo health.")
    parser.add_argument("--json", action="store_true", help="Write machine-readable JSON.")
    parser.add_argument("--root", default=str(ROOT), help="TrueVision repo root.")
    args = parser.parse_args(argv)

    report = build_preflight_report(args.root)
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(report_as_text(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
