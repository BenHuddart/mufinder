"""Run a blocking call off the GUI thread while keeping the window responsive.

Filesystem retries and keyring lookups can block for seconds or minutes (a
keyring may be waiting on an unlock prompt), so they must not run on the GUI
thread. :func:`run_blocking_call` runs the call on a worker thread and keeps
the caller's synchronous semantics: it returns the result or re-raises the
exception. While the call runs, user input is held back; if the call takes
noticeably long, an application-modal busy dialog appears, optionally with a
Cancel button.
"""

from __future__ import annotations

import threading
from typing import Callable, TypeVar

from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtWidgets import QProgressDialog, QWidget

T = TypeVar("T")

#: How long a call may run before the busy dialog appears, in seconds.
BUSY_DIALOG_DELAY_SECONDS = 0.4
_POLL_INTERVAL_MS = 25


class BlockingCallCancelled(Exception):
    """Raised when the user cancels a cancellable blocking call."""


def run_blocking_call(
    parent: QWidget | None,
    op: Callable[[], T],
    *,
    title: str,
    description: str,
    cancellable: bool = False,
    busy_delay_seconds: float = BUSY_DIALOG_DELAY_SECONDS,
) -> T:
    """Run ``op`` on a worker thread and return its result on the GUI thread.

    The event loop keeps running, so the window repaints, but user input is
    held back until the busy dialog appears; the dialog is application-modal,
    so other windows cannot start a re-entrant action. A cancelled call raises
    :class:`BlockingCallCancelled` at once; the worker thread is abandoned and
    finishes on its own, and its result is discarded.
    """

    outcome: dict[str, object] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            outcome["result"] = op()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the GUI thread
            outcome["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_worker, name="mufinder-blocking-call", daemon=True)
    worker.start()
    # Without the GUI thread waiting, a fast call never shows a dialog.
    if not done.wait(min(busy_delay_seconds, 0.05)):
        cancelled = _wait_with_busy_dialog(
            parent,
            done,
            title=title,
            description=description,
            cancellable=cancellable,
            busy_delay_seconds=busy_delay_seconds,
        )
        if cancelled:
            raise BlockingCallCancelled(description)
    worker.join()
    error = outcome.get("error")
    if isinstance(error, BaseException):
        raise error
    return outcome.get("result")  # type: ignore[return-value]


def _wait_with_busy_dialog(
    parent: QWidget | None,
    done: threading.Event,
    *,
    title: str,
    description: str,
    cancellable: bool,
    busy_delay_seconds: float,
) -> bool:
    """Pump the event loop until ``done`` is set; return whether the user cancelled."""

    cancelled = False

    def _pump(
        loop: QEventLoop,
        flags: QEventLoop.ProcessEventsFlag,
        *,
        deadline_ms: int | None,
        on_tick: Callable[[], None] | None = None,
    ) -> None:
        def _tick() -> None:
            if done.is_set() or cancelled:
                loop.quit()
            elif on_tick is not None:
                on_tick()

        poll = QTimer()
        poll.setInterval(_POLL_INTERVAL_MS)
        poll.timeout.connect(_tick)
        deadline = QTimer()
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        poll.start()
        if deadline_ms is not None:
            deadline.start(deadline_ms)
        try:
            if not done.is_set():
                loop.exec(flags)
        finally:
            poll.stop()
            deadline.stop()

    # Phase 1: repaint but hold back user input until the dialog is worth showing.
    _pump(
        QEventLoop(),
        QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents,
        deadline_ms=max(0, int(busy_delay_seconds * 1000)),
    )
    if done.is_set():
        return False

    # Phase 2: an application-modal busy dialog, which blocks input elsewhere.
    progress = QProgressDialog(description, "Cancel" if cancellable else None, 0, 0, parent)
    progress.setWindowTitle(title)
    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    loop = QEventLoop()

    def _watch_dialog() -> None:
        nonlocal cancelled
        # The Cancel button sets wasCanceled(); Escape only hides the dialog.
        # Closing the dialog later emits canceled, so the signal is not used.
        if cancellable and (progress.wasCanceled() or not progress.isVisible()):
            cancelled = True
            loop.quit()
        elif not progress.isVisible():
            # Keep a dialog without a Cancel button up, so input to other
            # windows stays blocked until the call finishes.
            progress.show()

    progress.show()
    try:
        _pump(loop, QEventLoop.ProcessEventsFlag.AllEvents, deadline_ms=None, on_tick=_watch_dialog)
    finally:
        progress.close()
        progress.deleteLater()
    # A call that finished as the user cancelled still returns its result.
    return cancelled and not done.is_set()
