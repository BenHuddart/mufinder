"""A time limit for each screenshot scene.

A scene that never finishes would stall ``make docs`` and CI's screenshot
jobs until something kills them, with nothing to say where. Inside
:func:`scene_deadline` a scene that runs past its limit ends the renderer
instead: the scene's name and the stack of every thread go to stderr, the
``cleanup`` callback removes the renderer's scratch directory, and the
process exits with :data:`TIMEOUT_EXIT_STATUS`.

The limit is a Python timer thread, which needs the GIL to report. A hang
that holds the GIL is caught by ``faulthandler``'s own watchdog, which needs
no GIL, :data:`GIL_BACKSTOP_S` seconds later; it dumps the same stacks under a
plain ``Timeout`` heading and exits with status 1, leaving the scratch
directory behind. The module imports no Qt.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import faulthandler
import os
import sys
import threading

#: The renderer's exit status when a scene runs past its time limit.
TIMEOUT_EXIT_STATUS = 3
#: How long after the limit ``faulthandler`` ends a process whose GIL is held.
GIL_BACKSTOP_S = 30.0


@contextmanager
def scene_deadline(
    name: str, seconds: float, *, cleanup: Callable[[], None] | None = None
) -> Iterator[None]:
    """End the process, naming ``name`` and dumping every stack, if the block runs past ``seconds``.

    ``cleanup`` runs after the dump and before the exit, which skips
    ``finally`` blocks; an error it raises is reported and the exit goes ahead.
    A limit that runs out just as the block finishes does nothing: the lock
    decides which came first.
    """

    lock = threading.Lock()
    finished = False

    def expire() -> None:
        with lock:
            if finished:
                return
            _report_and_exit(name, seconds, cleanup)

    timer = threading.Timer(seconds, expire)
    timer.name = f"scene-deadline-{name}"
    timer.daemon = True
    faulthandler.dump_traceback_later(seconds + GIL_BACKSTOP_S, exit=True)
    timer.start()
    try:
        yield
    finally:
        with lock:
            finished = True
        timer.cancel()
        faulthandler.cancel_dump_traceback_later()


def _report_and_exit(name: str, seconds: float, cleanup: Callable[[], None] | None) -> None:
    sys.stderr.write(
        f"Scene {name!r} did not finish within {seconds:g} s; the stack of every thread follows.\n"
    )
    sys.stderr.flush()
    faulthandler.dump_traceback(all_threads=True)
    if cleanup is not None:
        try:
            cleanup()
        except Exception as exc:  # noqa: BLE001 - the exit must not depend on it
            sys.stderr.write(f"Cleaning up after {name!r} failed: {exc}\n")
            sys.stderr.flush()
    os._exit(TIMEOUT_EXIT_STATUS)
