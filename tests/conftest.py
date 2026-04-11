"""Shared pytest fixtures for MuFinder tests."""

from __future__ import annotations

import atexit
from collections.abc import Generator
import gc
import os
import sys

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from mufinder.app.theme import cleanup_theme_runtime


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_PYTEST_EXITSTATUS = 0


@pytest.fixture(scope="session")
def qapp() -> Generator[QApplication, None, None]:
    """Provide one QApplication instance for all UI tests."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setQuitOnLastWindowClosed(False)

    yield app

    cleanup_theme_runtime(app)
    for widget in list(app.topLevelWidgets()):
        try:
            widget.close()
        except (RuntimeError, SystemError):
            continue
    for widget in [w for w in list(app.allWidgets()) if w.parent() is None]:
        try:
            widget.deleteLater()
        except (RuntimeError, SystemError):
            continue
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, 0)
    app.processEvents()
    gc.collect()
    app.quit()
    app.processEvents()


@pytest.fixture(autouse=True)
def _cleanup_qt_state_per_test(request: pytest.FixtureRequest):
    """Ensure Qt widgets and filters from one test do not leak into the next."""

    yield

    if "qapp" not in request.fixturenames:
        return

    app = QApplication.instance()
    if app is None:
        return

    cleanup_theme_runtime(app)
    for widget in list(app.topLevelWidgets()):
        try:
            widget.close()
            widget.deleteLater()
        except (RuntimeError, SystemError):
            continue
    QCoreApplication.sendPostedEvents(None, 0)
    app.processEvents()
    gc.collect()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Record final status and perform last Qt cleanup before interpreter shutdown."""

    global _PYTEST_EXITSTATUS
    _PYTEST_EXITSTATUS = int(exitstatus)

    app = QApplication.instance()
    if app is not None:
        cleanup_theme_runtime(app)
        for widget in list(app.topLevelWidgets()):
            try:
                widget.close()
                widget.deleteLater()
            except (RuntimeError, SystemError):
                continue
        QCoreApplication.sendPostedEvents(None, 0)
        app.processEvents()
        gc.collect()


@atexit.register
def _force_exit_before_extension_teardown() -> None:
    """Bypass crash-prone C-extension finalization after pytest completes."""

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_PYTEST_EXITSTATUS)
