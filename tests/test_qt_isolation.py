"""Guard that Qt state left by one test does not reach the next.

``tests/conftest.py`` cleans the shared ``QApplication`` after every test. A
mouse button a test pressed and never released stays recorded as held in
``QGuiApplication.mouseButtons()``; later code then sees a drag that is not
there (the focus-ring tracker, for one, shows no ring). The clean-up must
also leave alone the widgets Qt itself owns for the application's lifetime.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
import shiboken6

from tests.conftest import _cleanup_qt_application


def test_the_per_test_clean_up_releases_a_mouse_button_left_held(qapp) -> None:
    widget = QWidget()
    widget.resize(40, 40)
    widget.show()
    QTest.mousePress(widget, Qt.MouseButton.LeftButton)
    assert QGuiApplication.mouseButtons() & Qt.MouseButton.LeftButton

    _cleanup_qt_application(qapp)

    assert QGuiApplication.mouseButtons() == Qt.MouseButton.NoButton


def _desktop_widgets() -> list[QWidget]:
    return [
        widget
        for widget in QApplication.allWidgets()
        if widget.windowType() == Qt.WindowType.Desktop
    ]


def test_the_per_test_clean_up_keeps_the_desktop_widget_qt_made(qapp) -> None:
    """Qt keeps its own desktop widget for the application's lifetime.

    It makes the widget the first time the pointer leaves a child widget with
    a cursor of its own (a plot canvas, say, hidden while under the pointer),
    and keeps a plain pointer to it. Deleting it, as the clean-up deletes
    every other parentless widget, leaves that pointer dangling: the next
    such leave reads freed memory and the process crashes, in whichever later
    test it happens.
    """

    def leave_a_child_with_a_cursor() -> None:
        window = QWidget()
        window.resize(200, 200)
        child = QWidget(window)
        child.setGeometry(0, 0, 200, 200)
        child.setCursor(Qt.CursorShape.CrossCursor)
        window.show()
        QTest.mouseMove(child, QPoint(100, 100))
        qapp.processEvents()
        assert child.underMouse()
        child.hide()

    leave_a_child_with_a_cursor()
    # An earlier test in the same process may have made it already; either way
    # there is exactly one, and it is Qt's.
    desktop = _desktop_widgets()
    assert len(desktop) == 1

    _cleanup_qt_application(qapp)

    assert shiboken6.isValid(desktop[0])
    # Qt goes on using the same widget, rather than reading a deleted one.
    leave_a_child_with_a_cursor()
    assert _desktop_widgets() == desktop
