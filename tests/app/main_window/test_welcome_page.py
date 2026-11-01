"""The welcome page: its cards, its recent entries and the shell around it."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QGridLayout,
    QLabel,
    QLineEdit,
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


def _active(window: QWidget) -> QWidget:
    """Activate ``window`` and wait until the application reports it as active."""

    window.activateWindow()
    for _attempt in range(200):
        if QApplication.activeWindow() is window:
            return window
        QTest.qWait(10)
    raise AssertionError("the window never became active")


def _assert_down_walks_the_navigator(
    window: MainWindow, rows: tuple[str, str] = ("Host Structure", "Convergence Testing")
) -> None:
    """Down from where the user is reaches ``rows`` in turn, and each opens its page."""

    tree = window._project_tree
    assert QApplication.focusWidget() is tree

    walked = []
    for _press in range(2):
        QTest.keyClick(QApplication.focusWidget(), Qt.Key.Key_Down)
        QApplication.processEvents()
        assert tree.selectedItems() == [tree.currentItem()]
        walked.append(tree.currentItem().text(0))

    assert QApplication.focusWidget() is tree
    assert tuple(walked) == rows
    assert window._main_stack.currentWidget() is not window._home_page


def _open_through_the_file_menu(window: MainWindow, root: Path, monkeypatch) -> None:
    monkeypatch.setattr(window, "_get_existing_directory", lambda **kwargs: str(root))
    window._open_project()


def _open_from_the_command_line(window: MainWindow, root: Path, monkeypatch) -> None:
    assert window.open_project(root)


def _open_from_the_command_line_on_a_page(window: MainWindow, root: Path, monkeypatch) -> None:
    assert window.open_project(root)
    window.show_page("uep")


def _open_a_recent_entry(window: MainWindow, root: Path, monkeypatch) -> None:
    _set_recents(window, [str(root)])
    _recent_buttons(window)[0].click()


@pytest.mark.parametrize(
    "open_route",
    [_open_through_the_file_menu, _open_from_the_command_line, _open_a_recent_entry],
)
def test_keyboard_focus_lands_on_the_navigator_after_opening(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, open_route
) -> None:
    root = ProjectService().create_project(tmp_path / "keys.mufinder", "Keys").project_root
    _active(window)

    open_route(window, root, monkeypatch)
    QApplication.processEvents()

    assert window._document is not None
    # On the project home no row is the user's yet, not even as a pale current row.
    assert window._main_stack.currentWidget() is window._home_page
    assert window._project_tree.currentItem() is None
    _assert_down_walks_the_navigator(window)


def test_up_from_the_project_home_also_starts_at_the_first_row(
    window: MainWindow, tmp_path: Path
) -> None:
    root = ProjectService().create_project(tmp_path / "up.mufinder", "Up").project_root
    _active(window)
    assert window.open_project(root)
    QApplication.processEvents()

    QTest.keyClick(QApplication.focusWidget(), Qt.Key.Key_Up)
    QApplication.processEvents()

    assert window._project_tree.currentItem().text(0) == "Host Structure"
    assert window._project_tree.selectedItems() == [window._project_tree.currentItem()]


def test_opening_on_a_page_walks_down_from_that_page(window: MainWindow, tmp_path: Path) -> None:
    root = ProjectService().create_project(tmp_path / "page.mufinder", "Page").project_root
    _active(window)
    assert window.open_project(root)
    assert window.show_page("uep")
    QApplication.processEvents()

    _assert_down_walks_the_navigator(window, ("Muon Sites", "Generation"))


def test_opening_a_second_project_starts_at_the_first_row_again(
    window: MainWindow, tmp_path: Path
) -> None:
    first = ProjectService().create_project(tmp_path / "first.mufinder", "First").project_root
    second = ProjectService().create_project(tmp_path / "second.mufinder", "Second").project_root
    _active(window)
    assert window.open_project(first)
    assert window.show_page("uep")

    assert window.open_project(second)
    QApplication.processEvents()

    _assert_down_walks_the_navigator(window)


def test_returning_to_the_window_after_an_open_keeps_down_on_the_first_row(
    window: MainWindow, tmp_path: Path
) -> None:
    root = ProjectService().create_project(tmp_path / "away.mufinder", "Away").project_root
    _active(window)
    assert window.open_project(root)
    QApplication.processEvents()

    # Focus coming back to a tree with no current row must not pick one.
    window._project_tree.clearFocus()
    window._project_tree.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
    QApplication.processEvents()

    assert window._project_tree.currentItem() is None
    _assert_down_walks_the_navigator(window)


def _missing_recents(window: MainWindow, tmp_path: Path, count: int) -> list[str]:
    paths = [str(tmp_path / f"Missing{index}.mufinder") for index in range(count)]
    _set_recents(window, paths)
    return paths


def _click_recent_row(window: MainWindow, row: int) -> None:
    layout = window._home_recent_projects_layout
    button = layout.itemAt(row).widget()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()


@pytest.fixture
def warned(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Answer warning boxes at once, recording the widget holding focus at the time."""

    focused: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: focused.append(QApplication.focusWidget().accessibleName()),
    )
    return focused


def test_a_missing_recent_entry_passes_focus_to_the_next_entry(
    window: MainWindow, tmp_path: Path, warned: list[str]
) -> None:
    _active(window)
    _missing_recents(window, tmp_path, 3)

    _click_recent_row(window, 0)

    assert window._recent_projects() == [
        str(tmp_path / "Missing1.mufinder"),
        str(tmp_path / "Missing2.mufinder"),
    ]
    assert warned == ["Open recent project Missing1"]
    assert QApplication.focusWidget() is window._home_recent_projects_layout.itemAt(0).widget()


def test_the_last_missing_recent_entry_passes_focus_to_the_entry_above(
    window: MainWindow, tmp_path: Path, warned: list[str]
) -> None:
    _active(window)
    _missing_recents(window, tmp_path, 2)

    _click_recent_row(window, 1)

    assert QApplication.focusWidget() is window._home_recent_projects_layout.itemAt(0).widget()
    assert warned == ["Open recent project Missing0"]


def test_the_only_missing_recent_entry_passes_focus_to_create_new_project(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: None)
    _active(window)
    _missing_recents(window, tmp_path, 1)

    _click_recent_row(window, 0)

    focused = QApplication.focusWidget()
    assert isinstance(focused, QPushButton)
    assert focused.text() == "Create New Project"


def test_a_failed_open_from_the_menu_leaves_focus_on_the_welcome_page_alone(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: None)
    _active(window)
    paths = _missing_recents(window, tmp_path, 2)
    open_button = window._home_welcome_page.findChildren(QPushButton)[1]
    open_button.setFocus()

    window._open_recent_project(paths[0])
    QApplication.processEvents()

    assert QApplication.focusWidget() is open_button


def test_keyboard_focus_lands_on_the_navigator_after_new_project(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _active(window)
    monkeypatch.setattr(
        main_window_module.QInputDialog, "getText", lambda *args, **kwargs: ("Keys", True)
    )
    monkeypatch.setattr(window, "_get_existing_directory", lambda **kwargs: str(tmp_path))

    window._new_project()
    QApplication.processEvents()

    assert window._document is not None
    _assert_down_walks_the_navigator(window)


def test_a_failed_open_leaves_keyboard_focus_where_it_was(
    window: MainWindow, tmp_path: Path
) -> None:
    _active(window)
    button = window._home_welcome_page.findChildren(QPushButton)[0]
    button.setFocus()
    assert QApplication.focusWidget() is button

    assert not window._open_project_from_path(tmp_path / "missing.mufinder", show_errors=False)
    QApplication.processEvents()

    assert QApplication.focusWidget() is button


def test_a_refresh_of_the_open_project_leaves_keyboard_focus_alone(
    window: MainWindow, tmp_path: Path
) -> None:
    root = ProjectService().create_project(tmp_path / "keep.mufinder", "Keep").project_root
    _active(window)
    assert window.open_project(root)
    window._jobs_table.setFocus()
    assert QApplication.focusWidget() is window._jobs_table

    window._refresh_ui_after_project_change()
    QApplication.processEvents()

    assert QApplication.focusWidget() is window._jobs_table


def test_opening_under_a_dialog_leaves_its_focus_and_lands_on_the_navigator_after(
    window: MainWindow, tmp_path: Path
) -> None:
    root = ProjectService().create_project(tmp_path / "under.mufinder", "Under").project_root
    _active(window)
    dialog = QDialog(window)
    field = QLineEdit(dialog)
    dialog.setModal(True)
    dialog.show()
    _active(dialog)
    field.setFocus()
    assert QApplication.focusWidget() is field

    assert window.open_project(root)
    QApplication.processEvents()
    assert QApplication.focusWidget() is field

    dialog.close()
    _active(window)
    QApplication.processEvents()
    _assert_down_walks_the_navigator(window)


def test_integrity_warning_keeps_focus_until_dismissed_then_the_navigator_has_it(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ProjectService()
    root = service.create_project(tmp_path / "warned.mufinder", "Warned").project_root
    real_load = service.load_project
    focus_during_warning: list[bool] = []

    def load_with_warning(path: Path):
        document = real_load(path)
        document.ui_state["_integrity_warnings"] = ["A warning."]
        return document

    monkeypatch.setattr(window._project_service, "load_project", load_with_warning)
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: focus_during_warning.append(window._project_tree.hasFocus()),
    )
    _active(window)

    assert window.open_project(root)
    QApplication.processEvents()

    assert focus_during_warning == [False]
    _assert_down_walks_the_navigator(window)


def test_welcome_window_minimum_is_the_page_minimum_alone(window: MainWindow) -> None:

    assert window.minimumSizeHint().width() == window._main_stack.minimumSizeHint().width() + 18
