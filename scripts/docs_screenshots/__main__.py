"""The command line: ``python scripts/render_docs_screenshots.py [--example NAME] [--shard K/N] [--jobs N] [--scene-timeout S] [NAME ...]``.

With no names, no ``--example`` and no ``--shard`` every screenshot is
rendered, and the output directory's ``.stamp`` is touched once they all are.
``--shard K/N`` renders the K-th of N shares of the selection, the split CI's
screenshot jobs use. Several screenshots are shared out between ``--jobs``
worker processes the same way, each rendering its share as a run with names
does; one job renders them in this process. The shares come from
:func:`docs_screenshots.shards.partition`. Each scene, its project's build
included, must finish within ``--scene-timeout`` seconds, or the renderer
names it, dumps every thread's stack and fails (see
:mod:`docs_screenshots.watchdog`).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

# The render module configures Qt and VTK, so it is imported before them.
from .render import DEFAULT_OUTPUT, IMAGE_SUFFIX, install_offscreen_viewers, remove_scratch, render
from .registry import EXAMPLES, SCREENSHOTS
from .shards import load_durations, parse_shard, partition, store_durations
from .watchdog import scene_deadline

from PySide6.QtWidgets import QApplication

STAMP_NAME = ".stamp"
SCRIPT = Path(__file__).resolve().parents[1] / "render_docs_screenshots.py"
DEFAULT_JOBS = min(4, os.cpu_count() or 1)
# The slowest scene takes about 15 s and a worker's first scene of a project
# also builds it; a slow CI runner takes a few times longer.
DEFAULT_SCENE_TIMEOUT_S = 300.0
# A day; a longer limit (or infinity) overflows the platform's timers.
MAX_SCENE_TIMEOUT_S = 86_400.0


def _scene_timeout(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number of seconds: {value!r}") from None
    if not 0 < seconds <= MAX_SCENE_TIMEOUT_S:
        raise argparse.ArgumentTypeError(f"must be more than 0 and at most {MAX_SCENE_TIMEOUT_S:g} seconds, not {value}")
    return seconds


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
        "--shard",
        metavar="K/N",
        help="render (or, with --list, list) the K-th of N shares of equal measured duration",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=f"worker processes to share the screenshots between (default: {DEFAULT_JOBS})",
    )
    parser.add_argument(
        "--scene-timeout",
        type=_scene_timeout,
        default=DEFAULT_SCENE_TIMEOUT_S,
        metavar="SECONDS",
        help=f"fail a scene that takes longer than this (default: {DEFAULT_SCENE_TIMEOUT_S:g})",
    )
    parser.add_argument(
        "--store-durations",
        action="store_true",
        help="render in this process and record each screenshot's seconds in durations.json",
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
    if args.shard:
        try:
            shard, shards = parse_shard(args.shard)
        except ValueError as exc:
            parser.error(f"--shard: {exc}")
        names = _shares(names, shards)[shard - 1]
    if args.list:
        print("\n".join(names))
        return 0

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    jobs = 1 if args.store_durations else min(args.jobs, len(names))
    if jobs > 1:
        if not _render_in_workers(names, output, jobs, args.scene_timeout):
            return 1
    else:
        seconds = _render_here(names, output, args.scene_timeout)
        if args.store_durations:
            store_durations(seconds, list(SCREENSHOTS))
    if not (args.example or args.names or args.shard):
        # A full render marks the output complete; ``make docs`` renders again
        # only once a source is newer than this stamp.
        (output / STAMP_NAME).touch()
    return 0


def _shares(names: list[str], count: int) -> list[list[str]]:
    """``names`` in ``count`` shares, each project's scenes kept together where they fit."""

    projects = {name: id(SCREENSHOTS[name].project) for name in names}
    return partition(names, projects, load_durations(), count)


def _render_in_workers(names: list[str], output: Path, jobs: int, scene_timeout: float) -> bool:
    """Render ``names`` in ``jobs`` worker processes, one share each; ``True`` if all succeed."""

    command = [sys.executable, str(SCRIPT), "--jobs", "1", "--scene-timeout", str(scene_timeout), "--output", str(output)]
    workers = [subprocess.Popen([*command, *share]) for share in _shares(names, jobs) if share]
    codes = [worker.wait() for worker in workers]
    return all(code == 0 for code in codes)


def _render_here(names: list[str], output: Path, scene_timeout: float) -> dict[str, float]:
    """Render ``names`` in this process; the seconds each took, its project's build included."""

    app = QApplication(sys.argv[:1])
    from mufinder.app.gui_garbage_collector import install_gui_garbage_collector

    # As in the application, reference cycles of Qt objects are freed on this
    # thread only, never on a worker thread the window starts.
    install_gui_garbage_collector(app)
    app.setApplicationName("MuFinder")
    app.setOrganizationName("MuFinder")
    from mufinder.app.theme import THEME_LIGHT, apply_theme

    apply_theme(app, THEME_LIGHT)
    install_offscreen_viewers()
    seconds: dict[str, float] = {}
    for name in names:
        start = time.perf_counter()
        with scene_deadline(name, scene_timeout, cleanup=remove_scratch):
            render(app, SCREENSHOTS[name], output / f"{name}{IMAGE_SUFFIX}")
        seconds[name] = time.perf_counter() - start
    return seconds


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    finally:
        remove_scratch()


if __name__ == "__main__":
    raise SystemExit(main())
