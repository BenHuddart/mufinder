"""The command line: ``python scripts/render_docs_screenshots.py [--example NAME] [NAME ...]``.

With no names and no ``--example`` every screenshot is rendered, and the
output directory's ``.stamp`` is touched once they all are.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# The render module configures Qt and VTK, so it is imported before them.
from .render import DEFAULT_OUTPUT, IMAGE_SUFFIX, install_offscreen_viewers, remove_scratch, render
from .registry import EXAMPLES, SCREENSHOTS

from PySide6.QtWidgets import QApplication

STAMP_NAME = ".stamp"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the documentation screenshots from the running application.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="directory to write the images to")
    parser.add_argument("--list", action="store_true", help="print the screenshot names and exit")
    parser.add_argument(
        "--example",
        choices=sorted(EXAMPLES),
        help="render (or, with --list, list) every screenshot of one example",
    )
    parser.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help="screenshots to render (default: all; see --list)",
    )
    return parser


def _run(argv: list[str] | None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.example and args.names:
        parser.error("name screenshots or pass --example, not both")
    unknown = sorted(set(args.names) - set(SCREENSHOTS))
    if unknown:
        parser.error(f"unknown screenshots: {', '.join(unknown)}")
    if args.example:
        names = list(EXAMPLES[args.example].SCENES)
    else:
        names = args.names or list(SCREENSHOTS)
    if args.list:
        print("\n".join(names))
        return 0

    app = QApplication(sys.argv[:1])
    app.setApplicationName("MuFinder")
    app.setOrganizationName("MuFinder")
    from mufinder.app.theme import THEME_LIGHT, apply_theme

    apply_theme(app, THEME_LIGHT)
    install_offscreen_viewers()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        render(app, SCREENSHOTS[name], output / f"{name}{IMAGE_SUFFIX}")
    if not args.example and not args.names:
        # A full render marks the output complete; ``make docs`` renders again
        # only once a source is newer than this stamp.
        (output / STAMP_NAME).touch()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    finally:
        remove_scratch()


if __name__ == "__main__":
    raise SystemExit(main())
