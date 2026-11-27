"""Run a blocking call off the GUI thread while keeping the window responsive.

Filesystem retries and keyring lookups can block for seconds or minutes (a
keyring may be waiting on an unlock prompt), so they must not run on the GUI
thread. :func:`run_blocking_call` runs the call on a worker thread and keeps
the caller's synchronous semantics: it returns the result or re-raises the
exception. While the call runs, user input is held back; if the call takes
noticeably long, an application-modal busy dialog appears, optionally with a
Cancel button that can also tell the call to stop. :func:`wait_for_event` waits
the same way, up to a time limit, for work already running elsewhere.
"""

from __future__ import annotations

import threading
import time
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
    cancel_event: threading.Event | None = None,
    busy_delay_seconds: float = BUSY_DIALOG_DELAY_SECONDS,
) -> T:
    """Run ``op`` on a worker thread and return its result on the GUI thread.

    The event loop keeps running, so the window repaints, but user input is
    held back until the busy dialog appears; the dialog is application-modal,
    so other windows cannot start a re-entrant action. A cancelled call raises
    :class:`BlockingCallCancelled` at once and sets ``cancel_event``: an ``op``
    that polls it (for example as ``should_cancel=cancel_event.is_set``) stops
    early, and one that does not finishes on its own. Either way the worker's
    result or error is dropped as soon as it ends, so the memory it holds is freed.
    """

    outcome: dict[str, object] = {}
    done = threading.Event()
    abandoned = threading.Event()

    def _worker() -> None:
        try:
            result = op()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the GUI thread
            # A kept error would hold the cancelled call's frames, and their arrays,
            # in a reference cycle until the next garbage collection.
            if not abandoned.is_set():
                outcome["error"] = exc
        else:
            if not abandoned.is_set():
                outcome["result"] = result
            del result
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
            abandoned.set()
            # The worker may have ended just before it was abandoned.
            outcome.clear()
            if cancel_event is not None:
                cancel_event.set()
            raise BlockingCallCancelled(description)
    worker.join()
    error = outcome.get("error")
    if isinstance(error, BaseException):
        raise error
    return outcome.get("result")  # type: ignore[return-value]


def wait_for_event(
    parent: QWidget | None,
    done: threading.Event,
    timeout_seconds: float,
    *,
    title: str,
    description: str,
    busy_delay_seconds: float = BUSY_DIALOG_DELAY_SECONDS,
) -> bool:
    """Wait up to ``timeout_seconds`` for ``done``, keeping the window responsive.

    The event loop runs while it waits, so queued signals are delivered, and a
    busy dialog without a Cancel button appears as for :func:`run_blocking_call`.
    Returns whether ``done`` was set.
    """

    timeout_seconds = max(0.0, timeout_seconds)
    deadline = time.monotonic() + timeout_seconds
    if done.wait(max(0.0, min(busy_delay_seconds, timeout_seconds, 0.05))):
        return True
    _wait_with_busy_dialog(
        parent,
        done,
        title=title,
        description=description,
        cancellable=False,
        busy_delay_seconds=busy_delay_seconds,
        deadline=deadline,
    )
    return done.is_set()


def _wait_with_busy_dialog(
    parent: QWidget | None,
    done: threading.Event,
    *,
    title: str,
    description: str,
    cancellable: bool,
    busy_delay_seconds: float,
    deadline: float | None = None,
) -> bool:
    """Pump the event loop until ``done`` is set; return whether the user cancelled.

    With a ``deadline`` (a :func:`time.monotonic` time) it also stops then.
    """

    cancelled = False

    def _remaining_ms(limit_ms: int | None) -> int | None:
        if deadline is None:
            return limit_ms
        remaining = max(0, int((deadline - time.monotonic()) * 1000))
        return remaining if limit_ms is None else min(limit_ms, remaining)

    def _expired() -> bool:
        return deadline is not None and time.monotonic() >= deadline

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
        deadline_ms=_remaining_ms(max(0, int(busy_delay_seconds * 1000))),
    )
    if done.is_set() or _expired():
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
        _pump(
            loop,
            QEventLoop.ProcessEventsFlag.AllEvents,
            deadline_ms=_remaining_ms(None),
            on_tick=_watch_dialog,
        )
    finally:
        progress.close()
        progress.deleteLater()
    # A call that finished as the user cancelled still returns its result.
    return cancelled and not done.is_set()
