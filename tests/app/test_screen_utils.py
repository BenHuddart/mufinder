"""Tests for screen-aware window sizing helpers."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QWidget

from mufinder.app.screen_utils import clamp_to_screen, clamped_size


def _available_geometry(widget: QWidget):
    screen = widget.screen()
    if screen is None:
        screen = QApplication.instance().primaryScreen()
    return screen.availableGeometry()


def test_clamped_size_returns_requested_size_when_it_fits(qapp) -> None:
    widget = QWidget()
    try:
        avail = _available_geometry(widget)
        width = max(1, min(200, int(avail.width() * 0.1)))
        height = max(1, min(200, int(avail.height() * 0.1)))

        size = clamped_size(widget, width, height)

        assert size.width() == width
        assert size.height() == height
    finally:
        widget.deleteLater()


def test_clamped_size_clamps_when_requested_size_exceeds_fraction(qapp) -> None:
    widget = QWidget()
    try:
        avail = _available_geometry(widget)
        fraction = 0.9
        oversized_width = avail.width() * 10
        oversized_height = avail.height() * 10

        size = clamped_size(widget, oversized_width, oversized_height, fraction=fraction)

        expected_width = int(avail.width() * fraction)
        expected_height = int(avail.height() * fraction)
        assert size.width() == expected_width
        assert size.height() == expected_height
    finally:
        widget.deleteLater()


def test_clamped_size_headless_widget_without_screen_returns_requested_size(qapp, monkeypatch) -> None:
    widget = QWidget()
    try:
        monkeypatch.setattr(QWidget, "screen", lambda self: None)
        monkeypatch.setattr(QApplication, "primaryScreen", staticmethod(lambda: None))

        size = clamped_size(widget, 1234, 5678)

        assert size.width() == 1234
        assert size.height() == 5678
    finally:
        widget.deleteLater()


def test_clamp_to_screen_brings_oversized_widget_fully_on_screen(qapp) -> None:
    widget = QWidget()
    try:
        avail = _available_geometry(widget)

        # Move/resize the widget far off-screen and much larger than the display.
        widget.setGeometry(
            avail.right() + 5000,
            avail.bottom() + 5000,
            avail.width() * 5,
            avail.height() * 5,
        )
        widget.show()

        clamp_to_screen(widget)

        frame = widget.frameGeometry()
        assert frame.width() <= avail.width()
        assert frame.height() <= avail.height()
        assert frame.left() >= avail.left()
        assert frame.top() >= avail.top()
        assert frame.right() <= avail.right()
        assert frame.bottom() <= avail.bottom()
    finally:
        widget.close()
        widget.deleteLater()


def test_clamp_to_screen_noop_without_screen(qapp, monkeypatch) -> None:
    widget = QWidget()
    try:
        widget.setGeometry(10, 10, 100, 100)
        monkeypatch.setattr(QWidget, "screen", lambda self: None)
        monkeypatch.setattr(QApplication, "primaryScreen", staticmethod(lambda: None))

        clamp_to_screen(widget)

        # No screen geometry available: geometry is left untouched.
        assert widget.geometry().width() == 100
        assert widget.geometry().height() == 100
    finally:
        widget.deleteLater()
