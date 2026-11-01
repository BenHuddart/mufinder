"""The welcome page: its cards, its recent entries and the shell around it."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QGridLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from mufinder.app import main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.app.recent_project_button import RecentProjectButton
from mufinder.services.project_service import ProjectService

_WINDOW_SIZE = (1280, 840)


def _shown(window: MainWindow) -> MainWindow:
    window.show()
    window.resize(*_WINDOW_SIZE)
    QApplication.processEvents()
    return window


@pytest.fixture
def window(main_window: MainWindow) -> Iterator[MainWindow]:
    """A shown window, closed while ``confirm_choice`` still answers its save question."""

    yield _shown(main_window)
    main_window.close()


def _dock(window: MainWindow) -> QDockWidget:
    return window.findChild(QDockWidget, "jobDashboardDock")


def _recent_buttons(window: MainWindow) -> list[RecentProjectButton]:
    return window._home_welcome_page.findChildren(RecentProjectButton)


def _set_recents(window: MainWindow, paths: list[str]) -> None:
    window._set_recent_projects(paths)
    window._refresh_home_recent_projects()
    QApplication.processEvents()


def test_no_demo_action_exists(main_window: MainWindow) -> None:
    button_texts = [button.text() for button in main_window.findChildren(QPushButton)]
    action_texts = [action.text() for action in main_window.findChildren(QAction)]
    card_titles = [
        label.text() for label in main_window._home_welcome_page.findChildren(QLabel, "cardTitle")
    ]

    assert card_titles == ["New Project", "Open Project", "Recent Projects"]
    assert not [text for text in button_texts + action_texts if "demo" in text.lower()]


def test_a_long_recent_path_leaves_the_window_minimum_width_unchanged(
    window: MainWindow, tmp_path: Path
) -> None:
    minimum = window.minimumSizeHint().width()
    width = window.width()

    long_path = str(tmp_path / ("d" * 300) / "Deep.mufinder")
    _set_recents(window, [long_path, str(tmp_path / ("e" * 300 + ".mufinder"))])

    assert window.minimumSizeHint().width() == minimum
    assert window.width() == width
    for button in _recent_buttons(window):
        assert button.minimumSizeHint().width() <= button.parentWidget().width()
        assert button.width() <= button.parentWidget().width()


def test_clicking_a_recent_entry_opens_that_project(window: MainWindow, tmp_path: Path) -> None:
    created = ProjectService().create_project(tmp_path / "CoF2.mufinder", "CoF2")
    _set_recents(window, [str(created.project_root)])

    _recent_buttons(window)[0].click()

    assert window._document is not None
    assert window._document.project_root == created.project_root


def test_recent_entries_list_at_most_five_projects_and_rebuild(
    window: MainWindow, tmp_path: Path
) -> None:

    _set_recents(window, [str(tmp_path / f"P{index}.mufinder") for index in range(7)])
    layout = window._home_recent_projects_layout
    shown = [layout.itemAt(row).widget() for row in range(layout.count())]
    assert [button.accessibleName().rsplit(" ", 1)[1] for button in shown] == [
        f"P{index}" for index in range(5)
    ]

    window._clear_recent_projects()
    assert layout.count() == 1
    assert layout.itemAt(0).widget().text() == "No recent projects yet."


def test_card_titles_share_a_height_when_one_body_wraps_further(
    main_window: MainWindow,
) -> None:
    # A short card beside a taller one takes on the extra height; none of it
    # may go to the title.
    window = main_window
    host = QWidget()
    grid = QGridLayout(host)
    short, _short_layout = window._make_home_card("Short", "One line.")
    tall, _tall_layout = window._make_home_card("Tall", "Many words that wrap. " * 12)
    grid.addWidget(short, 0, 0)
    grid.addWidget(tall, 0, 1)
    host.resize(420, 400)
    host.show()
    QApplication.processEvents()

    short_title = short.findChild(QLabel, "cardTitle")
    tall_title = tall.findChild(QLabel, "cardTitle")

    assert short.height() == tall.height()
    assert short_title.height() == tall_title.height() == short_title.sizeHint().height()
    host.close()


@pytest.mark.parametrize("size", [(1280, 840), (1800, 1000)])
def test_the_three_welcome_card_titles_share_a_height(
    window: MainWindow, size: tuple[int, int]
) -> None:
    window.resize(*size)
    QApplication.processEvents()

    titles = window._home_welcome_page.findChildren(QLabel, "cardTitle")

    assert len(titles) == 3
    assert len({title.height() for title in titles}) == 1


def test_navigator_and_dashboard_hidden_with_no_project(window: MainWindow) -> None:

    assert not window._project_tree_panel.isVisible()
    assert not _dock(window).isVisible()
    assert window._home_content_stack.currentWidget() is window._home_welcome_page
    # The welcome page fills the window's central area, with no splitter handle beside it.
    assert not window._navigator_splitter.handle(1).isVisible()
    assert window._main_stack.width() == window._navigator_splitter.width()


def test_navigator_and_dashboard_shown_after_open_project(
    window: MainWindow, tmp_path: Path
) -> None:
    created = ProjectService().create_project(tmp_path / "shown.mufinder", "Shown")

    assert window.open_project(created.project_root)
    QApplication.processEvents()

    assert window._project_tree_panel.isVisible()
    assert _dock(window).isVisible()
    assert window._navigator_splitter.handle(1).isVisible()
    assert window._home_content_stack.currentWidget() is window._project_home_page


def test_navigator_and_dashboard_shown_after_new_project(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main_window_module.QInputDialog, "getText", lambda *args, **kwargs: ("Fresh", True)
    )
    monkeypatch.setattr(window, "_get_existing_directory", lambda **kwargs: str(tmp_path))

    window._new_project()
    QApplication.processEvents()

    assert window._document is not None
    assert window._project_tree_panel.isVisible()
    assert _dock(window).isVisible()


def test_a_failed_open_leaves_the_welcome_page_alone(window: MainWindow, tmp_path: Path) -> None:

    assert not window._open_project_from_path(tmp_path / "missing.mufinder", show_errors=False)
    QApplication.processEvents()

    assert not window._project_tree_panel.isVisible()
    assert not _dock(window).isVisible()


def test_opening_a_project_brings_back_a_dashboard_closed_with_its_button(
    window: MainWindow, tmp_path: Path
) -> None:
    first = ProjectService().create_project(tmp_path / "first.mufinder", "First")
    second = ProjectService().create_project(tmp_path / "second.mufinder", "Second")
    assert window.open_project(first.project_root)
    QApplication.processEvents()

    _dock(window).close()
    QApplication.processEvents()
    assert not _dock(window).isVisible()

    assert window.open_project(second.project_root)
    QApplication.processEvents()
    assert _dock(window).isVisible()


def test_navigator_opens_at_its_saved_width_after_the_welcome_page(qapp, tmp_path: Path) -> None:
    created = ProjectService().create_project(tmp_path / "wide.mufinder", "Wide")
    settings = QSettings("MuFinder", "MuFinder")
    settings.setValue(MainWindow._PROJECT_NAVIGATOR_WIDTH_KEY, 280)

    window = _shown(MainWindow())
    try:
        assert window.open_project(created.project_root)
        QApplication.processEvents()

        assert window._project_tree_panel.width() == 280
    finally:
        window.close()
    # Hiding and showing the panel is not a drag, so it saves nothing new.
    assert int(settings.value(MainWindow._PROJECT_NAVIGATOR_WIDTH_KEY)) == 280


def test_opening_a_project_does_not_resize_the_window(window: MainWindow, tmp_path: Path) -> None:
    created = ProjectService().create_project(tmp_path / "same.mufinder", "Same")
    window.resize(window.minimumSize())
    QApplication.processEvents()
    size = window.size()

    assert window.open_project(created.project_root)
    QApplication.processEvents()

    assert window.size() == size


def test_panels_appear_even_if_a_refresh_fails(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail() -> None:
        raise RuntimeError("bad job entry")

    window._document = ProjectService().create_project(tmp_path / "bad.mufinder", "Bad")
    monkeypatch.setattr(window, "_refresh_jobs_table", fail)

    with pytest.raises(RuntimeError):
        window._refresh_ui_after_project_change()

    assert window._project_tree_panel.isVisible()
    assert _dock(window).isVisible()


def test_welcome_window_minimum_is_the_page_minimum_alone(window: MainWindow) -> None:

    assert window.minimumSizeHint().width() == window._main_stack.minimumSizeHint().width() + 18
