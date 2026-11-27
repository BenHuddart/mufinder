"""Menu-bar mnemonics: Alt+letter opens a menu and a letter then chooses its entry."""

from __future__ import annotations

from collections.abc import Iterator

from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible, QAction, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox
import pytest

from mufinder.app import main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.app.recent_projects import RecentProjectsMenu
from tests.support.qt import pump_until, release_keyboard_modifiers, shown_text


def _mnemonic(text: str) -> str:
    """Return the letter Qt binds for ``text``'s mnemonic, upper case, or ``""``."""

    return QKeySequence.mnemonic(text).toString().removeprefix("Alt+").upper()


def _menu_entries(menu: QMenu) -> list[QAction]:
    """The entries of ``menu`` that are commands or submenus: no separators,
    no recent-project paths and not the disabled ``(No recent projects)`` note."""

    if isinstance(menu, RecentProjectsMenu):
        return [
            action
            for action in menu.actions()
            if shown_text(action.text()) == "Clear Recent Projects"
        ]
    return [action for action in menu.actions() if not action.isSeparator()]


def _all_menus(window: MainWindow) -> list[QMenu]:
    """Every menu and submenu on the menu bar.

    Found as children rather than through ``QAction.menu()``, whose wrapper
    PySide can let go of while the menu is still in use.
    """

    # The untitled one is the menu bar's own overflow menu.
    return [menu for menu in window.menuBar().findChildren(QMenu) if menu.title()]


@pytest.fixture(params=["linux", "win32"])
def window(
    request: pytest.FixtureRequest, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MainWindow]:
    # ``Exit`` on Windows takes a different letter from ``Quit``.
    monkeypatch.setattr(main_window_module.sys, "platform", request.param)
    window = MainWindow()
    window._recent_projects.set_paths(["/p/a&b", "/p/c"])
    yield window
    window.close()


def test_every_menu_bar_title_has_its_own_mnemonic(window: MainWindow) -> None:
    titles = [action.text() for action in window.menuBar().actions()]
    letters = [_mnemonic(title) for title in titles]

    assert [shown_text(title) for title in titles] == ["File", "Setup", "View", "Help"]
    assert letters == ["F", "S", "V", "H"]


def test_every_menu_entry_has_a_mnemonic_unique_in_its_menu(window: MainWindow) -> None:
    menus = list(_all_menus(window))
    assert {shown_text(menu.title()) for menu in menus} == {
        "File",
        "Open Recent",
        "Setup",
        "View",
        "Appearance",
        "Help",
    }
    for menu in menus:
        entries = _menu_entries(menu)
        assert entries, menu.title()
        letters = [_mnemonic(action.text()) for action in entries]
        assert all(letters), [action.text() for action in entries]
        assert len(set(letters)) == len(letters), [action.text() for action in entries]


def test_a_recent_project_path_has_no_mnemonic_and_shows_its_ampersand(
    window: MainWindow,
) -> None:
    paths = [
        action for action in window._recent_projects_menu.actions() if action.text().startswith("/")
    ]

    assert [shown_text(action.text()) for action in paths] == ["/p/a&b", "/p/c"]
    assert [_mnemonic(action.text()) for action in paths] == ["", ""]


def test_menu_bar_accessible_names_have_no_ampersand(window: MainWindow) -> None:
    bar = QAccessible.queryAccessibleInterface(window.menuBar())
    names = [bar.child(index).text(QAccessible.Text.Name) for index in range(bar.childCount())]

    assert names == ["File", "Setup", "View", "Help"]


@pytest.mark.skipif(
    QKeySequence.mnemonic("&F").isEmpty(), reason="this platform shows no mnemonics"
)
def test_alt_f_opens_file_and_alt_h_a_opens_about(
    main_window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "about", lambda _parent, title, _text: shown.append(title))
    main_window.show()
    main_window.activateWindow()
    assert QTest.qWaitForWindowActive(main_window)

    QTest.keyClick(main_window, Qt.Key.Key_F, Qt.KeyboardModifier.AltModifier)
    release_keyboard_modifiers()
    assert pump_until(lambda: isinstance(QApplication.activePopupWidget(), QMenu))
    assert shown_text(QApplication.activePopupWidget().title()) == "File"

    QTest.keyClick(QApplication.activePopupWidget(), Qt.Key.Key_Escape)
    assert pump_until(lambda: QApplication.activePopupWidget() is None)

    QTest.keyClick(main_window, Qt.Key.Key_H, Qt.KeyboardModifier.AltModifier)
    release_keyboard_modifiers()
    assert pump_until(lambda: isinstance(QApplication.activePopupWidget(), QMenu))
    help_menu = QApplication.activePopupWidget()
    assert shown_text(help_menu.title()) == "Help"

    QTest.keyClick(help_menu, Qt.Key.Key_A)
    assert pump_until(lambda: bool(shown))

    assert shown == ["About MuFinder"]
    assert QApplication.activePopupWidget() is None
    main_window.hide()
