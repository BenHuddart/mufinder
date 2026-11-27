"""Render worker processes: start them, report the ones that fail, and re-run a crash once.

The renderer shares its screenshots out between worker processes, each one a
run of the renderer with names (see :mod:`docs_screenshots.__main__`). Each
worker appends a scene's name to its *done file* once the scene's image is
written, so the scenes a worker left unrendered are known however it ended.

A worker that exits non-zero is reported on stderr: its exit status (the
signal's name when a signal ended it), the scenes in its share, the scene it
was on and the scenes it had rendered. Its own stderr, which the fault handler
writes the Python stack of a crash to, is the worker's and goes straight to
the same stream.

A worker a crash signal ended (a segmentation fault inside Qt, say) has its
unrendered scenes run once more, in a fresh process started as soon as the
crash is seen; a second failure fails the run, and a crash after the worker's
last scene was written fails nothing. A worker that exits with a status of its
own (a scene raised, or ran past its time limit) failed for a reason a re-run
repeats, and one that ``SIGINT`` or ``SIGTERM`` ended was stopped, so neither
is re-run. The module imports no Qt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import signal
import subprocess
import itertools
import sys
import tempfile
import time

#: The option a worker is given its done file with.
DONE_FILE_OPTION = "--done-file"
#: Signals that stop a worker rather than crash it; its scenes are not re-run.
STOP_SIGNALS = frozenset({signal.SIGINT, signal.SIGTERM})
#: How often the running workers are checked for an exit.
POLL_INTERVAL_S = 0.2


@dataclass
class _Worker:
    title: str
    share: list[str]
    done_file: Path
    process: subprocess.Popen[bytes]
    is_rerun: bool

    def rendered(self) -> list[str]:
        try:
            lines = self.done_file.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [name for name in lines if name in self.share]

    def unrendered(self) -> list[str]:
        rendered = set(self.rendered())
        return [name for name in self.share if name not in rendered]


def record_done(done_file: Path | None, name: str) -> None:
    """Append ``name`` to ``done_file``, if there is one; called once a scene's image is written."""

    if done_file is None:
        return
    with done_file.open("a", encoding="utf-8") as stream:
        stream.write(f"{name}\n")


def describe_exit(code: int) -> str:
    """``code`` as a person reads it: ``signal SIGSEGV (-11)`` or ``status 3``."""

    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = str(-code)
        return f"signal {name} ({code})"
    return f"status {code}"


def render_shares(shares: Sequence[Sequence[str]], command: Sequence[str]) -> bool:
    """Render each share in a worker running ``command`` with its names; ``True`` if all succeed.

    A worker a crash signal ends has its unrendered scenes re-run once.
    """

    with tempfile.TemporaryDirectory(prefix="mufinder-screenshot-workers-") as scratch:
        directory = Path(scratch)
        numbers = itertools.count(1)

        def start(title: str, share: list[str], *, is_rerun: bool) -> _Worker:
            done_file = directory / f"{next(numbers)}.done"
            process = subprocess.Popen([*command, DONE_FILE_OPTION, str(done_file), *share])
            return _Worker(title, share, done_file, process, is_rerun)

        running = [
            start(f"Render worker {index}", list(share), is_rerun=False)
            for index, share in enumerate([share for share in shares if share], start=1)
        ]
        succeeded = True
        while running:
            time.sleep(POLL_INTERVAL_S)
            for worker in list(running):
                code = worker.process.poll()
                if code is None:
                    continue
                running.remove(worker)
                if code == 0:
                    continue
                _report(worker, code)
                unrendered = worker.unrendered()
                if not unrendered and code < 0:
                    _say("  Every scene in its share was written first; nothing to re-run.")
                elif code > 0 or -code in STOP_SIGNALS or worker.is_rerun:
                    succeeded = False
                else:
                    _say(
                        f"  Re-running its {len(unrendered)} unrendered scenes in a fresh process."
                    )
                    title = worker.title.replace("Render worker", "Re-run worker")
                    running.append(start(title, unrendered, is_rerun=True))
        return succeeded


def _report(worker: _Worker, code: int) -> None:
    rendered = worker.rendered()
    unrendered = worker.unrendered()
    lines = [
        f"{worker.title} exited with {describe_exit(code)}.",
        f"  Its scenes: {', '.join(worker.share)}.",
    ]
    if unrendered:
        lines.append(f"  It was on: {unrendered[0]}.")
    lines.append(f"  Rendered before it ended: {', '.join(rendered) if rendered else 'none'}.")
    _say("\n".join(lines))


def _say(text: str) -> None:
    sys.stderr.write(f"{text}\n")
    sys.stderr.flush()
