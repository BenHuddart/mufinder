"""Tests for running blocking calls off the GUI thread."""

from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QProgressDialog, QWidget

from mufinder.app.blocking_call import BlockingCallCancelled, run_blocking_call, wait_for_event


def _busy_dialogs() -> list[QProgressDialog]:
    return [
        widget
        for widget in QApplication.topLevelWidgets()
        if isinstance(widget, QProgressDialog) and widget.isVisible()
    ]


def _cancel_busy_dialog() -> None:
    dialogs = _busy_dialogs()
    if dialogs:
        dialogs[0].cancel()
    else:
        QTimer.singleShot(10, _cancel_busy_dialog)


def test_run_blocking_call_returns_result_off_the_gui_thread(qapp) -> None:
    threads: list[threading.Thread] = []

    def _op() -> str:
        threads.append(threading.current_thread())
        return "done"

    assert run_blocking_call(None, _op, title="Busy", description="Working") == "done"
    assert threads[0] is not threading.main_thread()


def test_run_blocking_call_reraises_the_operation_error(qapp) -> None:
    def _op() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_blocking_call(None, _op, title="Busy", description="Working")


def test_run_blocking_call_shows_busy_dialog_for_slow_calls(qapp) -> None:
    release = threading.Event()
    seen: list[tuple[str, str, bool]] = []
    parent = QWidget()

    def _inspect() -> None:
        dialogs = _busy_dialogs()
        if dialogs:
            dialog = dialogs[0]
            seen.append((dialog.windowTitle(), dialog.labelText(), dialog.isModal()))
            release.set()
        else:
            QTimer.singleShot(10, _inspect)

    QTimer.singleShot(0, _inspect)
    result = run_blocking_call(
        parent,
        lambda: release.wait(5) and "finished",
        title="Working on files",
        description="Deleting files",
        busy_delay_seconds=0.05,
    )

    assert result == "finished"
    assert seen == [("Working on files", "Deleting files", True)]
    assert _busy_dialogs() == []
    parent.deleteLater()


def test_run_blocking_call_cancel_returns_without_waiting_for_the_call(qapp) -> None:
    release = threading.Event()

    QTimer.singleShot(0, _cancel_busy_dialog)
    started = time.monotonic()
    try:
        with pytest.raises(BlockingCallCancelled):
            run_blocking_call(
                None,
                lambda: release.wait(5),
                title="System keyring passwords",
                description="Reading passwords",
                cancellable=True,
                busy_delay_seconds=0.05,
            )
        assert time.monotonic() - started < 2
        assert not release.is_set()
        assert _busy_dialogs() == []
    finally:
        release.set()


class _Held:
    """Stands for the arrays a cancelled calculation holds; it can be weakly referenced."""


def test_run_blocking_call_cancel_tells_the_call_to_stop_and_drops_what_it_held(qapp) -> None:
    stop = threading.Event()
    workers: list[threading.Thread] = []
    held: list[weakref.ref[_Held]] = []

    def calculation() -> None:
        workers.append(threading.current_thread())
        arrays = _Held()
        held.append(weakref.ref(arrays))
        if not stop.wait(5):
            raise AssertionError("the call was never told to stop")
        raise RuntimeError("stopped", arrays)

    QTimer.singleShot(0, _cancel_busy_dialog)
    collecting = gc.isenabled()
    # MuFinder collects cycles only on a GUI-thread timer, so a cycle would keep them.
    gc.disable()
    try:
        with pytest.raises(BlockingCallCancelled):
            run_blocking_call(
                None,
                calculation,
                title="Advanced Demagnetizing Field",
                description="Calculating",
                cancellable=True,
                cancel_event=stop,
                busy_delay_seconds=0.05,
            )
        assert stop.is_set()
        workers[0].join(5)
        assert not workers[0].is_alive()
        assert held[0]() is None
    finally:
        stop.set()
        if collecting:
            gc.enable()


def test_run_blocking_call_leaves_the_cancel_event_clear_when_the_call_finishes(qapp) -> None:
    stop = threading.Event()

    result = run_blocking_call(
        None,
        lambda: "finished",
        title="Advanced Demagnetizing Field",
        description="Calculating",
        cancellable=True,
        cancel_event=stop,
        busy_delay_seconds=0.05,
    )

    assert result == "finished"
    assert not stop.is_set()


def test_run_blocking_call_slow_cancellable_call_returns_its_result(qapp) -> None:
    release = threading.Event()
    QTimer.singleShot(150, release.set)

    result = run_blocking_call(
        None,
        lambda: release.wait(5) and "finished",
        title="System keyring passwords",
        description="Reading passwords",
        cancellable=True,
        busy_delay_seconds=0.05,
    )

    assert result == "finished"
    assert _busy_dialogs() == []


def test_run_blocking_call_escape_cancels_a_cancellable_call(qapp) -> None:
    release = threading.Event()

    def _escape() -> None:
        dialogs = _busy_dialogs()
        if dialogs:
            dialogs[0].reject()
        else:
            QTimer.singleShot(10, _escape)

    QTimer.singleShot(0, _escape)
    try:
        with pytest.raises(BlockingCallCancelled):
            run_blocking_call(
                None,
                lambda: release.wait(5),
                title="System keyring passwords",
                description="Reading passwords",
                cancellable=True,
                busy_delay_seconds=0.05,
            )
    finally:
        release.set()


def test_run_blocking_call_without_cancel_keeps_waiting_after_escape(qapp) -> None:
    release = threading.Event()
    rejected: list[bool] = []

    def _reject_then_release() -> None:
        dialogs = _busy_dialogs()
        if not dialogs:
            QTimer.singleShot(10, _reject_then_release)
            return
        if not rejected:
            dialogs[0].reject()
            rejected.append(True)
            QTimer.singleShot(50, _reject_then_release)
            return
        release.set()

    QTimer.singleShot(0, _reject_then_release)
    result = run_blocking_call(
        None,
        lambda: release.wait(5) and "finished",
        title="Working on files",
        description="Deleting files",
        busy_delay_seconds=0.05,
    )

    assert result == "finished"
    assert rejected == [True]


def test_wait_for_event_returns_at_once_when_the_event_is_set(qapp) -> None:
    done = threading.Event()
    done.set()

    assert wait_for_event(None, done, 5.0, title="Closing", description="Waiting") is True
    assert _busy_dialogs() == []


def test_wait_for_event_shows_a_busy_dialog_and_delivers_queued_signals(qapp) -> None:
    done = threading.Event()
    seen: list[tuple[str, str]] = []
    delivered: list[bool] = []

    def _inspect() -> None:
        dialogs = _busy_dialogs()
        if dialogs:
            seen.append((dialogs[0].windowTitle(), dialogs[0].labelText()))
            done.set()
        else:
            QTimer.singleShot(10, _inspect)

    QTimer.singleShot(0, lambda: delivered.append(True))
    QTimer.singleShot(0, _inspect)

    assert wait_for_event(
        None, done, 5.0, title="Closing", description="Waiting", busy_delay_seconds=0.05
    )
    assert seen == [("Closing", "Waiting")]
    assert delivered == [True]
    assert _busy_dialogs() == []


@pytest.mark.parametrize("busy_delay_seconds", [0.05, 5.0], ids=["dialog", "no-dialog"])
def test_wait_for_event_gives_up_at_its_time_limit(qapp, busy_delay_seconds: float) -> None:
    started = time.monotonic()

    finished = wait_for_event(
        None,
        threading.Event(),
        0.3,
        title="Closing",
        description="Waiting",
        busy_delay_seconds=busy_delay_seconds,
    )

    assert finished is False
    assert 0.25 <= time.monotonic() - started < 2.0
    assert _busy_dialogs() == []
