"""Menu-bar mnemonics: Alt+letter opens a menu and a letter then chooses its entry."""

from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible, QAction, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox
import pytest

from mufinder.app import main_window as main_window_module
from mufinder.app.main_window import MainWindow
from tests.support.qt import pump_until, release_keyboard_modifiers, shown_text

RECENT_PATHS = ["/p/a&b", "/p/c"]


def _mnemonic(text: str) -> str:
    """Return the letter after ``text``'s mnemonic ``&``, upper case, or ``""``.

    Read from the text rather than with ``QKeySequence.mnemonic``, which is
    empty on macOS, where Qt turns mnemonics off.
    """

    index = 0
    while index < len(text) - 1:
        if text[index] == "&":
            if text[index + 1] != "&":
                return text[index + 1].upper()
            index += 1
        index += 1
    return ""


def _menus(window: MainWindow) -> dict[str, QMenu]:
    """Every menu and submenu on the menu bar, by shown title.

    Found as children rather than through ``QAction.menu()``, whose wrapper
    PySide can let go of while the menu is still in use. The untitled one is
    the menu bar's own overflow menu.
    """

    return {
        shown_text(menu.title()): menu
        for menu in window.menuBar().findChildren(QMenu)
        if menu.title()
    }


def _entries(menu: QMenu) -> list[QAction]:
    """The commands and submenus in ``menu``: no separators and no recent-project paths."""

    return [
        action
        for action in menu.actions()
        if not action.isSeparator() and shown_text(action.text()) not in RECENT_PATHS
    ]


def _assert_unique_mnemonics(actions: list[QAction]) -> None:
    texts = [action.text() for action in actions]
    letters = [_mnemonic(text) for text in texts]
    assert all(letters), texts
    assert len(set(letters)) == len(letters), texts


@pytest.fixture
def window(main_window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> MainWindow:
    """A window whose Open Recent lists ``RECENT_PATHS``, with no settings written."""

    monkeypatch.setattr(main_window._recent_projects, "paths", lambda: list(RECENT_PATHS))
    main_window._recent_projects_menu.refresh()
    return main_window


def test_every_menu_bar_title_has_its_own_mnemonic(window: MainWindow) -> None:
    titles = [action.text() for action in window.menuBar().actions()]

    assert [shown_text(title) for title in titles] == ["File", "Setup", "View", "Help"]
    assert [_mnemonic(title) for title in titles] == ["F", "S", "V", "H"]


def test_every_menu_entry_has_a_mnemonic_unique_in_its_menu(window: MainWindow) -> None:
    menus = _menus(window)

    assert set(menus) == {"File", "Open Recent", "Setup", "View", "Appearance", "Help"}
    for menu in menus.values():
        _assert_unique_mnemonics(_entries(menu))
    assert [shown_text(action.text()) for action in _entries(menus["Open Recent"])] == [
        "Clear Recent Projects"
    ]


def test_exit_on_windows_keeps_the_file_menu_letters_unique(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only the menu code sees Windows; the theme and fonts keep this platform's.
    monkeypatch.setattr(main_window_module, "sys", SimpleNamespace(platform="win32"))
    window = MainWindow()

    entries = _entries(_menus(window)["File"])
    _assert_unique_mnemonics(entries)
    assert entries[-1].text() == "E&xit"


def test_a_recent_project_path_has_no_mnemonic_and_shows_its_ampersand(
    window: MainWindow,
) -> None:
    paths = [
        action
        for action in _menus(window)["Open Recent"].actions()
        if shown_text(action.text()) in RECENT_PATHS
    ]

    assert [shown_text(action.text()) for action in paths] == RECENT_PATHS
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
