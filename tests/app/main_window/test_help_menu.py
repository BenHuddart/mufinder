"""Help menu in the main window: the user guide link and the About box."""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QMessageBox

import mufinder.app.main_window as main_window_module
from mufinder._version import current_version
from mufinder.branding import DOCUMENTATION_URL, REPOSITORY_URL


def _help_actions(window) -> list[QAction]:
    menus = [menu for menu in window.menuBar().findChildren(QMenu) if menu.title() == "Help"]
    assert len(menus) == 1
    return menus[0].actions()


def test_help_menu_follows_view_and_lists_user_guide_then_about(main_window) -> None:
    titles = [action.text() for action in main_window.menuBar().actions()]

    assert titles[-1] == "Help"
    assert titles[titles.index("Help") - 1] == "View"
    assert [action.text() for action in _help_actions(main_window)] == ["User Guide", "About MuFinder"]


def test_user_guide_opens_documentation_url_in_system_browser(main_window, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        main_window_module.QDesktopServices,
        "openUrl",
        lambda url: opened.append(url.toString()) or True,
    )

    _help_actions(main_window)[0].trigger()

    assert opened == [DOCUMENTATION_URL]


def test_about_box_shows_version_description_and_links(main_window, monkeypatch) -> None:
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "about",
        lambda _parent, title, text: shown.append((title, text)),
    )

    _help_actions(main_window)[1].trigger()

    assert len(shown) == 1
    title, text = shown[0]
    assert title == "About MuFinder"
    assert current_version() in text
    assert "CASTEP" in text
    assert f'href="{DOCUMENTATION_URL}"' in text
    assert f'href="{REPOSITORY_URL}"' in text
