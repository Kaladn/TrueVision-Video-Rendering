from __future__ import annotations

import argparse

from truevision_runtime.rendering.avatar_film_renderer import render_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a TrueVision avatar film from a project JSON.")
    parser.add_argument("project", help="Path to a TrueVision avatar film project JSON.")
    args = parser.parse_args()
    video = render_project(args.project)
    print(video)


if __name__ == "__main__":
    main()
