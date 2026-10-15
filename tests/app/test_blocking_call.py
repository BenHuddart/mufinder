"""Tests for running blocking calls off the GUI thread."""

from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QProgressDialog, QWidget

from mufinder.app.blocking_call import BlockingCallCancelled, run_blocking_call


def _busy_dialogs() -> list[QProgressDialog]:
    return [
        widget
        for widget in QApplication.topLevelWidgets()
        if isinstance(widget, QProgressDialog) and widget.isVisible()
    ]


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

    def _cancel() -> None:
        dialogs = _busy_dialogs()
        if dialogs:
            dialogs[0].cancel()
        else:
            QTimer.singleShot(10, _cancel)

    QTimer.singleShot(0, _cancel)
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
