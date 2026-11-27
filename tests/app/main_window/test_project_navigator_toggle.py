"""View > Project Navigator: hiding and showing the navigator, Ctrl+B and the remembered choice."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu, QWidget
import pytest

from mufinder.app.main_window import MainWindow
from mufinder.app.navigator_panel import SHOWN_SETTINGS_KEY, WIDTH_SETTINGS_KEY
from mufinder.app.project_navigator import ProjectNavigator
from tests.support.main_window import empty_project
from tests.support.qt import press_key


def _view_action(window: MainWindow) -> QAction:
    for menu in window.menuBar().findChildren(QMenu):
        if menu.title() == "View":
            for action in menu.actions():
                if action.text() == "Project Navigator":
                    return action
    raise AssertionError("No View > Project Navigator")


def _navigator(window: MainWindow) -> ProjectNavigator:
    return window.findChild(ProjectNavigator)


def _navigator_list(window: MainWindow) -> QWidget:
    return window.findChild(QWidget, "projectNavigatorList")


def _shown(window: MainWindow) -> MainWindow:
    window.show()
    window.resize(1280, 840)
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    QApplication.processEvents()
    return window


def _trigger(window: MainWindow) -> None:
    _view_action(window).trigger()
    QApplication.processEvents()


def _focus_is_in_page(window: MainWindow) -> bool:
    focused = QApplication.focusWidget()
    page = window.findChild(QWidget, "projectNavigatorSplitter").widget(1)
    return focused is not None and page.isAncestorOf(focused)


@pytest.fixture
def window(main_window: MainWindow) -> Iterator[MainWindow]:
    yield _shown(main_window)
    # Hidden, a window closes without asking, so the Qt clean-up shows no real prompt.
    main_window.hide()


@pytest.fixture
def open_window(window: MainWindow, tmp_path: Path) -> MainWindow:
    assert window.open_project(empty_project(tmp_path))
    QApplication.processEvents()
    return window


def test_the_action_is_first_in_view_checkable_with_ctrl_b(main_window: MainWindow) -> None:
    action = _view_action(main_window)
    view_menu = next(m for m in main_window.menuBar().findChildren(QMenu) if m.title() == "View")

    assert view_menu.actions()[0] is action
    assert action.isCheckable()
    assert action.isChecked()
    # Qt shows and reads Ctrl as Cmd on macOS.
    assert action.shortcut() == QKeySequence("Ctrl+B")
    assert not action.autoRepeat()


def test_ctrl_b_is_bound_to_no_other_action(main_window: MainWindow) -> None:
    ctrl_b = QKeySequence("Ctrl+B")
    bound = [
        action.text()
        for action in main_window.findChildren(QAction)
        if ctrl_b in action.shortcuts()
    ]

    assert bound == ["Project Navigator"]


def test_without_a_project_the_action_is_disabled_and_the_navigator_hidden(
    window: MainWindow,
) -> None:
    assert not _view_action(window).isEnabled()
    assert not _navigator(window).isVisible()

    window.findChild(QWidget, "projectNavigatorSplitter").setFocus()
    press_key(Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)

    assert not _navigator(window).isVisible()
    assert _view_action(window).isChecked()
    assert QSettings("MuFinder", "MuFinder").value(SHOWN_SETTINGS_KEY) is None


def test_the_action_hides_and_shows_the_navigator(open_window: MainWindow) -> None:
    action = _view_action(open_window)
    assert action.isEnabled()
    assert _navigator(open_window).isVisible()

    _trigger(open_window)
    assert not _navigator(open_window).isVisible()
    assert not action.isChecked()

    _trigger(open_window)
    assert _navigator(open_window).isVisible()
    assert action.isChecked()


def test_ctrl_b_hides_and_shows_the_navigator(open_window: MainWindow) -> None:
    press_key(Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
    assert not _navigator(open_window).isVisible()
    assert not _view_action(open_window).isChecked()

    press_key(Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
    assert _navigator(open_window).isVisible()
    assert _view_action(open_window).isChecked()


def test_hiding_with_focus_in_the_navigator_moves_focus_to_the_page(
    open_window: MainWindow,
) -> None:
    assert QApplication.focusWidget() is _navigator_list(open_window)

    press_key(Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)

    assert _focus_is_in_page(open_window)


def test_hiding_with_focus_in_the_page_leaves_focus_where_it_is(open_window: MainWindow) -> None:
    # Hiding once finds a control in the page; showing again focuses the navigator.
    focused = press_key(Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
    press_key(Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
    assert _navigator(open_window).isVisible()
    focused.setFocus(Qt.FocusReason.OtherFocusReason)
    QApplication.processEvents()

    press_key(Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)

    assert QApplication.focusWidget() is focused


def test_showing_the_navigator_gives_it_focus(open_window: MainWindow) -> None:
    press_key(Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
    assert _focus_is_in_page(open_window)

    press_key(Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)

    assert QApplication.focusWidget() is _navigator_list(open_window)


def test_the_choice_is_saved_and_restored_after_a_restart(
    open_window: MainWindow, tmp_path: Path
) -> None:
    _trigger(open_window)
    open_window.close()
    QApplication.processEvents()

    restarted = _shown(MainWindow())
    try:
        assert not _view_action(restarted).isChecked()
        assert not _view_action(restarted).isEnabled()

        assert restarted.open_project(tmp_path / "Alpha.mufinder")
        QApplication.processEvents()
        assert not _navigator(restarted).isVisible()
        assert _view_action(restarted).isEnabled()
        assert not _view_action(restarted).isChecked()
        # With the navigator hidden, a project opens with focus in the page.
        assert _focus_is_in_page(restarted)

        _trigger(restarted)
        assert _navigator(restarted).isVisible()
    finally:
        restarted.hide()


def test_a_project_opened_before_the_window_shows_starts_with_focus_in_the_page(
    qapp: QApplication, tmp_path: Path
) -> None:
    QSettings("MuFinder", "MuFinder").setValue(SHOWN_SETTINGS_KEY, False)
    window = MainWindow()
    try:
        # As the command line opens a project, before the window is on screen.
        assert window.open_project(empty_project(tmp_path))
        _shown(window)

        assert not _navigator(window).isVisible()
        assert _focus_is_in_page(window)
    finally:
        window.hide()


def test_setting_the_action_from_code_hides_the_navigator_and_saves_the_choice(
    open_window: MainWindow,
) -> None:
    _view_action(open_window).setChecked(False)
    QApplication.processEvents()

    assert not _navigator(open_window).isVisible()
    assert QSettings("MuFinder", "MuFinder").value(SHOWN_SETTINGS_KEY) in (False, "false")


def test_a_hidden_navigator_stays_hidden_when_another_project_opens(
    open_window: MainWindow, tmp_path: Path
) -> None:
    _trigger(open_window)

    assert open_window.open_project(empty_project(tmp_path, "Beta"))
    QApplication.processEvents()

    assert not _navigator(open_window).isVisible()
    assert not _view_action(open_window).isChecked()


def test_pages_still_open_with_the_navigator_hidden(open_window: MainWindow) -> None:
    _trigger(open_window)

    assert open_window.show_page("host_structure")
    QApplication.processEvents()

    assert _navigator(open_window).current_route() == "host_structure"
    assert not _navigator(open_window).isVisible()


def test_hiding_keeps_the_saved_width_and_showing_restores_it(
    qapp: QApplication, tmp_path: Path
) -> None:
    QSettings("MuFinder", "MuFinder").setValue(WIDTH_SETTINGS_KEY, 260)
    window = _shown(MainWindow())
    try:
        assert window.open_project(empty_project(tmp_path))
        QApplication.processEvents()
        assert _navigator(window).width() == 260

        _trigger(window)
        _trigger(window)

        assert _navigator(window).width() == 260
    finally:
        window.close()
    assert int(QSettings("MuFinder", "MuFinder").value(WIDTH_SETTINGS_KEY)) == 260


def test_a_window_with_the_navigator_hidden_measures_its_minimum_without_it(
    qapp: QApplication,
) -> None:
    shown = MainWindow()
    QSettings("MuFinder", "MuFinder").setValue(SHOWN_SETTINGS_KEY, False)
    hidden = MainWindow()
    try:
        # A narrow window saved with the navigator hidden reopens as narrow.
        assert hidden.minimumWidth() <= shown.minimumWidth() - ProjectNavigator.MINIMUM_WIDTH
    finally:
        shown.close()
        hidden.close()
