"""The recent-project list in the settings and the File menu's Open Recent submenu."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMainWindow, QMenu

from mufinder.app import workflow_table_utils
from mufinder.app.recent_projects import RecentProjects, RecentProjectsMenu
from mufinder.app.workflow_table_utils import confirm_destructive as _real_confirm_destructive
from tests.support.qt import answer_next_message_box, shown_text


def _settings() -> QSettings:
    return QSettings("MuFinder", "MuFinder")


def _menu_texts(menu: RecentProjectsMenu) -> list[str]:
    return [action.text() for action in menu.actions() if not action.isSeparator()]


def test_the_list_reads_the_setting_older_versions_wrote(qapp) -> None:
    settings = _settings()
    settings.setValue("ui/recent_projects", ["/a/One.mufinder", "/b/Two.mufinder"])

    assert RecentProjects(settings).paths() == ["/a/One.mufinder", "/b/Two.mufinder"]


def test_a_single_stored_path_is_a_list_of_one(qapp) -> None:
    settings = _settings()
    settings.setValue("ui/recent_projects", "/a/Only.mufinder")

    assert RecentProjects(settings).paths() == ["/a/Only.mufinder"]


def test_blanks_and_repeats_are_dropped_and_at_most_ten_are_kept(qapp) -> None:
    settings = _settings()
    stored = [" /p/0 ", "", "/p/0", *(f"/p/{index}" for index in range(1, 14))]
    settings.setValue("ui/recent_projects", stored)

    assert RecentProjects(settings).paths() == [f"/p/{index}" for index in range(10)]


def test_record_puts_the_project_first_once(qapp, tmp_path: Path) -> None:
    recent = RecentProjects(_settings())
    first = tmp_path / "first.mufinder"
    second = tmp_path / "second.mufinder"

    recent.record(first)
    recent.record(second)
    recent.record(first)

    assert recent.paths() == [str(first), str(second)]
    assert _settings().value("ui/recent_projects") == [str(first), str(second)]


def test_remove_and_clear_change_the_list_and_announce_it(qapp) -> None:
    recent = RecentProjects(_settings())
    recent.set_paths(["/p/a", "/p/b", "/p/c"])
    announced: list[None] = []
    recent.changed.connect(lambda: announced.append(None))

    recent.remove("/p/b")
    assert recent.paths() == ["/p/a", "/p/c"]

    recent.clear()
    assert recent.paths() == []
    assert len(announced) == 2


def test_the_menu_lists_each_project_then_clear(qapp) -> None:
    recent = RecentProjects(_settings())
    recent.set_paths(["/p/a", "/p/b"])

    menu = RecentProjectsMenu(recent)

    assert shown_text(menu.title()) == "Open Recent"
    assert _menu_texts(menu) == ["/p/a", "/p/b", "Clear Recent Projects"]


def test_the_menu_says_when_there_are_none(qapp) -> None:
    menu = RecentProjectsMenu(RecentProjects(_settings()))

    actions = menu.actions()
    assert [shown_text(action.text()) for action in actions] == ["(No recent projects)"]
    assert not actions[0].isEnabled()


def test_the_menu_follows_the_list_and_asks_to_open_an_entry(qapp) -> None:
    recent = RecentProjects(_settings())
    menu = RecentProjectsMenu(recent)
    requested: list[str] = []
    menu.open_requested.connect(requested.append)

    recent.record(Path("/p/new.mufinder"))
    menu.actions()[0].trigger()

    assert requested == ["/p/new.mufinder"]


def _clear_action(menu: RecentProjectsMenu):
    return next(
        action for action in menu.actions() if shown_text(action.text()) == "Clear Recent Projects"
    )


def test_clear_recent_projects_asks_first_then_empties_the_list(
    qapp, confirmed_destructive_actions: list[dict[str, str]]
) -> None:
    recent = RecentProjects(_settings())
    recent.set_paths(["/p/a", "/p/b"])
    menu = RecentProjectsMenu(recent)

    _clear_action(menu).trigger()

    assert confirmed_destructive_actions == [
        {
            "title": "Clear recent projects",
            "text": "Clear 2 recent projects from the list?",
            "detail": "Open Recent and the welcome page will list no projects. "
            "No project or file is deleted.",
            "accept_label": "Clear list",
            "reject_label": "Cancel",
        }
    ]
    assert recent.paths() == []
    assert _menu_texts(menu) == ["(No recent projects)"]


def test_the_question_counts_a_single_project(
    qapp, confirmed_destructive_actions: list[dict[str, str]]
) -> None:
    recent = RecentProjects(_settings())
    recent.set_paths(["/p/a"])

    _clear_action(RecentProjectsMenu(recent)).trigger()

    assert confirmed_destructive_actions[0]["text"] == "Clear 1 recent project from the list?"


def test_declining_the_question_keeps_the_list(qapp, monkeypatch) -> None:
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *a, **k: False)
    recent = RecentProjects(_settings())
    recent.set_paths(["/p/a", "/p/b"])
    menu = RecentProjectsMenu(recent)

    _clear_action(menu).trigger()

    assert recent.paths() == ["/p/a", "/p/b"]
    assert _menu_texts(menu) == ["/p/a", "/p/b", "Clear Recent Projects"]


def test_the_question_defaults_to_cancel_and_escape_keeps_the_list(qapp, monkeypatch) -> None:
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", _real_confirm_destructive)
    parent = QMainWindow()
    recent = RecentProjects(_settings())
    recent.set_paths(["/p/a"])
    menu = RecentProjectsMenu(recent, QMenu("File", parent))

    seen = answer_next_message_box()
    _clear_action(menu).trigger()

    assert seen["parent"] is parent
    assert seen["default"] == "Cancel"
    assert seen["escape"] == "Cancel"
    assert sorted(seen["buttons"]) == ["Cancel", "Clear list"]
    assert recent.paths() == ["/p/a"]
    parent.close()


def test_clear_is_not_offered_when_the_list_is_empty(
    qapp, confirmed_destructive_actions: list[dict[str, str]]
) -> None:
    menu = RecentProjectsMenu(RecentProjects(_settings()))

    assert "Clear Recent Projects" not in [shown_text(action.text()) for action in menu.actions()]
    assert confirmed_destructive_actions == []
