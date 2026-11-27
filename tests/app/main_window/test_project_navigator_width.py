"""The project navigator's splitter: default width, dragging and the remembered width."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFrame

from mufinder.app.main_window import MainWindow
from mufinder.app.navigator_panel import WIDTH_SETTINGS_KEY
from mufinder.app.project_navigator import ProjectNavigator
from mufinder.services.project_service import ProjectService

_WINDOW_SIZE = (1280, 840)
DEFAULT = ProjectNavigator.DEFAULT_WIDTH
MINIMUM = ProjectNavigator.MINIMUM_WIDTH


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return ProjectService().create_project(tmp_path / "project.mufinder", "Project").project_root


def _in_project(project: Path) -> MainWindow:
    """A window with ``project`` open: the navigator is hidden until one is."""

    window = MainWindow()
    assert window.open_project(project)
    return window


def _shown(window: MainWindow, size: tuple[int, int] = _WINDOW_SIZE) -> MainWindow:
    window.show()
    window.resize(*size)
    QApplication.processEvents()
    return window


def _save_navigator_width(value: object) -> None:
    QSettings("MuFinder", "MuFinder").setValue(WIDTH_SETTINGS_KEY, value)


def _saved_navigator_width() -> object:
    return QSettings("MuFinder", "MuFinder").value(WIDTH_SETTINGS_KEY)


def _drag_navigator_edge(window: MainWindow, delta: int) -> None:
    handle = window._navigator_splitter.handle(1)
    start = handle.rect().center()
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(handle, start + QPoint(delta, 0))
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=start + QPoint(delta, 0))
    QApplication.processEvents()


def test_default_width_gives_the_pages_the_rest_of_the_window(qapp, project: Path) -> None:
    window = _shown(_in_project(project))
    try:
        assert window.findChild(QFrame, "projectTreePanel").width() == DEFAULT
        assert window._navigator_splitter.sizes() == [DEFAULT, window._main_stack.width()]
        assert window._main_stack.width() > window._main_stack.minimumSizeHint().width()
        assert window._navigator_splitter.handle(1).accessibleName() == "Resize project navigator"
    finally:
        window.close()


def test_splitter_restores_a_saved_width(qapp, project: Path) -> None:
    _save_navigator_width(260)

    window = _shown(_in_project(project))
    try:
        assert window._project_navigator.width() == 260
    finally:
        window.close()


def test_dragging_saves_the_width_for_the_next_window(qapp, project: Path) -> None:
    window = _shown(_in_project(project))
    try:
        _drag_navigator_edge(window, 45)
        assert window._project_navigator.width() == DEFAULT + 45
    finally:
        window.close()

    assert int(_saved_navigator_width()) == DEFAULT + 45
    reopened = _shown(_in_project(project))
    try:
        assert reopened._project_navigator.width() == DEFAULT + 45
    finally:
        reopened.close()


def test_navigator_cannot_be_dragged_past_its_minimum_or_over_the_page_minimum(
    qapp, project: Path
) -> None:
    window = _shown(_in_project(project))
    try:
        _drag_navigator_edge(window, -1000)
        assert window._project_navigator.width() == MINIMUM

        _drag_navigator_edge(window, 2000)
        assert window._main_stack.width() == window._main_stack.minimumSizeHint().width()
        assert window._project_navigator.width() > DEFAULT
    finally:
        window.close()


def test_resizing_the_window_changes_the_pages_and_keeps_the_navigator_width(
    qapp, project: Path
) -> None:
    window = _shown(_in_project(project))
    try:
        stack_width = window._main_stack.width()

        window.resize(_WINDOW_SIZE[0] + 200, _WINDOW_SIZE[1])
        QApplication.processEvents()

        assert window._project_navigator.width() == DEFAULT
        assert window._main_stack.width() == stack_width + 200
    finally:
        window.close()


def test_a_window_too_narrow_for_the_width_keeps_the_saved_choice(qapp, project: Path) -> None:
    _save_navigator_width(3000)

    window = _shown(_in_project(project))
    try:
        assert window._project_navigator.width() < 3000
        assert window._main_stack.width() == window._main_stack.minimumSizeHint().width()
    finally:
        window.close()

    assert int(_saved_navigator_width()) == 3000


@pytest.mark.parametrize(
    "saved",
    ["not a width", "", "0", "-40", "179", "99999999999"],
)
def test_an_unusable_saved_width_falls_back_to_the_default(qapp, project: Path, saved: str) -> None:
    _save_navigator_width(saved)

    window = _shown(_in_project(project))
    try:
        assert window._project_navigator.width() == DEFAULT
    finally:
        window.close()


def test_window_minimum_is_the_navigator_minimum_plus_the_page_minimum(qapp, project: Path) -> None:
    window = _shown(_in_project(project))
    try:
        page_minimum = window._main_stack.minimumSizeHint().width()
        assert window.minimumSizeHint().width() == MINIMUM + page_minimum + 24
    finally:
        window.close()


def test_a_blocked_drag_leaves_the_saved_width_alone(qapp, project: Path) -> None:
    _save_navigator_width(300)

    window = _in_project(project)
    window.show()
    window.resize(window.minimumSizeHint().width(), _WINDOW_SIZE[1])
    QApplication.processEvents()
    try:
        squeezed = window._project_navigator.width()
        assert squeezed < 300

        _drag_navigator_edge(window, 40)

        assert window._project_navigator.width() == squeezed
    finally:
        window.close()

    assert int(_saved_navigator_width()) == 300


def test_a_drag_out_and_back_saves_the_width_it_ended_on(qapp, project: Path) -> None:
    _save_navigator_width(260)

    window = _shown(_in_project(project))
    try:
        handle = window._navigator_splitter.handle(1)
        start = handle.rect().center()
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(handle, start + QPoint(20, 0))
        QApplication.processEvents()
        assert window._project_navigator.width() == 280
        # Positions are relative to the handle, which has moved with the drag.
        QTest.mouseMove(handle, start + QPoint(-20, 0))
        QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=start)
        QApplication.processEvents()
        assert window._project_navigator.width() == 260
    finally:
        window.close()

    assert int(_saved_navigator_width()) == 260
