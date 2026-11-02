"""The File and View menus, the window title and what closing asks."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QDockWidget, QMenu, QMessageBox
import pytest

from mufinder.app import main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.services.project_service import ProjectService
from tests.support.main_window import answer_new_project


def _menu(window: MainWindow, title: str) -> QMenu:
    for menu in window.menuBar().findChildren(QMenu):
        if menu.title() == title:
            return menu
    raise AssertionError(f"No menu {title}")


def _action(window: MainWindow, menu: str, text: str) -> QAction:
    for action in _menu(window, menu).actions():
        if action.text() == text:
            return action
    raise AssertionError(f"No action {text} in {menu}")


def _dock(window: MainWindow) -> QDockWidget:
    return window.findChild(QDockWidget, "jobDashboardDock")


def _project(tmp_path: Path, name: str = "Alpha") -> Path:
    return ProjectService().create_project(tmp_path / f"{name}.mufinder", name).project_root


@pytest.fixture
def window(main_window: MainWindow) -> Iterator[MainWindow]:
    main_window.show()
    QApplication.processEvents()
    yield main_window
    # Hidden, a window closes without asking, so the Qt clean-up shows no real prompt.
    main_window.hide()


@pytest.fixture
def open_window(window: MainWindow, tmp_path: Path) -> MainWindow:
    assert window.open_project(_project(tmp_path))
    QApplication.processEvents()
    return window


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("New Project…", QKeySequence.StandardKey.New),
        ("Open Project…", QKeySequence.StandardKey.Open),
        ("Save Project", QKeySequence.StandardKey.Save),
    ],
)
def test_file_actions_have_the_platform_shortcut(
    main_window: MainWindow, text: str, key: QKeySequence.StandardKey
) -> None:
    action = _action(main_window, "File", text)

    assert action.shortcuts() == QKeySequence.keyBindings(key)
    assert action.shortcuts()


def test_quit_has_the_platform_shortcut_and_role(main_window: MainWindow) -> None:
    action = _action(main_window, "File", "Quit")

    assert action.shortcuts() == QKeySequence.keyBindings(QKeySequence.StandardKey.Quit)
    assert action.menuRole() == QAction.MenuRole.QuitRole


def test_quit_is_exit_on_windows(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_window_module.sys, "platform", "win32")

    action = _action(MainWindow(), "File", "Exit")

    assert action.menuRole() == QAction.MenuRole.QuitRole


@pytest.mark.parametrize(
    ("menu", "text"),
    [
        ("File", "Save Project"),
        ("File", "Import Legacy Relaxations…"),
        ("File", "Remove Imported Relaxations…"),
        ("View", "Job Dashboard"),
    ],
)
def test_project_actions_need_a_project(
    window: MainWindow, tmp_path: Path, menu: str, text: str
) -> None:
    action = _action(window, menu, text)
    assert not action.isEnabled()

    assert window.open_project(_project(tmp_path))

    assert action.isEnabled()


def test_a_project_change_leaves_import_disabled_during_a_scan(
    open_window: MainWindow, tmp_path: Path
) -> None:
    window = open_window
    window._legacy_scan_root = tmp_path
    window._import_legacy_action.setEnabled(False)

    assert window.open_project(_project(tmp_path, "Beta"))
    assert not window._import_legacy_action.isEnabled()

    window._finish_legacy_scan()
    assert window._import_legacy_action.isEnabled()


def test_the_title_follows_the_project(window: MainWindow, tmp_path: Path) -> None:
    assert window.windowTitle() == "MuFinder"

    assert window.open_project(_project(tmp_path))
    assert window.windowTitle() == "Alpha — MuFinder"

    assert window.open_project(_project(tmp_path, "Beta"))
    assert window.windowTitle() == "Beta — MuFinder"


def test_a_failed_open_leaves_the_title(window: MainWindow, tmp_path: Path) -> None:
    assert not window._open_project_from_path(tmp_path / "missing.mufinder", show_errors=False)

    assert window.windowTitle() == "MuFinder"


# --- View > Job Dashboard -------------------------------------------------------


def test_the_view_action_brings_back_a_closed_dashboard(open_window: MainWindow) -> None:
    window = open_window
    action = _action(window, "View", "Job Dashboard")
    assert action.isChecked()

    _dock(window).close()
    QApplication.processEvents()
    assert not _dock(window).isVisible()
    assert not action.isChecked()

    action.trigger()
    QApplication.processEvents()
    assert _dock(window).isVisible()
    assert action.isChecked()


def test_a_closed_dashboard_stays_closed_when_another_project_opens(
    open_window: MainWindow, tmp_path: Path
) -> None:
    window = open_window
    _action(window, "View", "Job Dashboard").trigger()
    QApplication.processEvents()
    assert not _dock(window).isVisible()

    assert window.open_project(_project(tmp_path, "Beta"))
    QApplication.processEvents()

    assert not _dock(window).isVisible()
    assert not _action(window, "View", "Job Dashboard").isChecked()


def test_minimising_the_window_is_not_closing_the_dashboard(open_window: MainWindow) -> None:
    window = open_window

    window.showMinimized()
    QApplication.processEvents()
    assert window._jobs_dock_wanted

    window.showNormal()
    QApplication.processEvents()
    assert _dock(window).isVisible()


def _restarted(window: MainWindow) -> MainWindow:
    window.close()
    QApplication.processEvents()
    restarted = MainWindow()
    restarted.show()
    QApplication.processEvents()
    return restarted


def test_the_dashboard_height_survives_a_restart_and_skips_the_default(
    open_window: MainWindow, tmp_path: Path
) -> None:
    window = open_window
    window.resize(1280, 1000)
    QApplication.processEvents()
    default_height = _dock(window).height()
    window.resizeDocks(
        [_dock(window)], [default_height + 40], main_window_module.Qt.Orientation.Vertical
    )
    QApplication.processEvents()
    height = _dock(window).height()
    assert height > default_height

    restarted = _restarted(window)
    try:
        # With no project it stays hidden, whatever was saved.
        assert not _dock(restarted).isVisible()

        # The offscreen screen is too short for the restored geometry to keep this.
        restarted.resize(1280, 1000)
        assert restarted.open_project(tmp_path / "Alpha.mufinder")
        QApplication.processEvents()

        assert _dock(restarted).isVisible()
        assert _dock(restarted).height() == height
    finally:
        restarted.hide()


def test_a_closed_dashboard_stays_closed_after_a_restart(
    open_window: MainWindow, tmp_path: Path
) -> None:
    _action(open_window, "View", "Job Dashboard").trigger()
    QApplication.processEvents()

    restarted = _restarted(open_window)
    try:
        # Decided before the pages fill, so it never appears on the way.
        assert restarted.open_project(tmp_path / "Alpha.mufinder")
        assert not _dock(restarted).isVisible()

        _action(restarted, "View", "Job Dashboard").trigger()
        QApplication.processEvents()
        assert _dock(restarted).isVisible()
        assert _dock(restarted).height() > 0
    finally:
        restarted.hide()


def test_closing_from_the_welcome_page_keeps_the_saved_dashboard(
    open_window: MainWindow, tmp_path: Path
) -> None:
    welcome = _restarted(open_window)
    assert not _dock(welcome).isVisible()

    restarted = _restarted(welcome)
    try:
        assert restarted.open_project(tmp_path / "Alpha.mufinder")
        QApplication.processEvents()
        assert _dock(restarted).isVisible()
    finally:
        restarted.hide()


def test_a_first_run_saves_no_window_state_from_the_welcome_page(window: MainWindow) -> None:
    window.close()

    assert not QSettings("MuFinder", "MuFinder").contains(MainWindow._MAIN_WINDOW_STATE_KEY)


def test_a_corrupt_saved_state_falls_back_to_the_default(
    qapp: QApplication, tmp_path: Path
) -> None:
    QSettings("MuFinder", "MuFinder").setValue(
        MainWindow._MAIN_WINDOW_STATE_KEY, main_window_module.QByteArray(b"not a window state")
    )

    window = MainWindow()
    window.show()
    try:
        assert window.open_project(_project(tmp_path))
        QApplication.processEvents()
        assert _dock(window).isVisible()
    finally:
        window.hide()


# --- Closing --------------------------------------------------------------------


def _change_a_viewer_choice(window: MainWindow) -> None:
    window._document.ui_state["uep_analysis_view"] = {"layout": "changed"}


def test_close_right_after_opening_does_not_ask_or_write(
    open_window: MainWindow, answered_choices: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = open_window
    saves: list[object] = []
    monkeypatch.setattr(window._project_service, "save_project", saves.append)

    assert window.close() is True

    assert answered_choices == []
    assert saves == []


def test_close_after_a_page_tidied_a_saved_form_on_open_does_not_ask(
    window: MainWindow, tmp_path: Path, answered_choices: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    fill = window._refresh_journal_view

    def _tidy() -> None:
        window._document.ui_state["uep_form"] = {"tidied": True}
        fill()

    monkeypatch.setattr(window, "_refresh_journal_view", _tidy)
    assert window.open_project(root)

    assert window.close() is True

    assert answered_choices == []


def test_close_after_save_does_not_ask(open_window: MainWindow, answered_choices: list) -> None:
    window = open_window
    _change_a_viewer_choice(window)
    window._save_project()

    assert window.close() is True

    assert answered_choices == []


def test_close_after_a_ui_state_change_asks_and_saves(
    open_window: MainWindow, answered_choices: list
) -> None:
    window = open_window
    root = window._document.project_root
    _change_a_viewer_choice(window)

    assert window.close() is True

    assert [prompt["title"] for prompt in answered_choices] == ["Save project?"]
    assert "uep_analysis_view" in ProjectService().load_project(root).ui_state


def test_an_autosave_takes_the_change_so_close_does_not_ask(
    open_window: MainWindow, answered_choices: list
) -> None:
    window = open_window
    _change_a_viewer_choice(window)
    window._persist_document_ui_state(failure_message="failed")

    assert window.close() is True

    assert answered_choices == []


def test_undoing_a_change_means_close_does_not_ask(
    open_window: MainWindow, answered_choices: list
) -> None:
    window = open_window
    _change_a_viewer_choice(window)
    del window._document.ui_state["uep_analysis_view"]

    assert window.close() is True

    assert answered_choices == []


def test_moving_between_pages_is_kept_without_asking(
    open_window: MainWindow, answered_choices: list
) -> None:
    window = open_window
    root = window._document.project_root
    window.show_page("host_structure")
    QApplication.processEvents()
    assert window._document.ui_state.get("project_home_stage") == "host_structure"

    assert window.close() is True

    assert answered_choices == []
    assert ProjectService().load_project(root).ui_state["project_home_stage"] == "host_structure"


def test_a_page_move_and_a_viewer_choice_still_ask(
    open_window: MainWindow, answered_choices: list
) -> None:
    window = open_window
    window.show_page("host_structure")
    _change_a_viewer_choice(window)

    assert window.close() is True

    assert len(answered_choices) == 1


def test_job_state_waiting_for_its_autosave_is_written_without_asking(
    open_window: MainWindow, answered_choices: list
) -> None:
    window = open_window
    root = window._document.project_root
    window._document.project.name = "Renamed"
    window._request_dashboard_flush(failure_message="failed")

    assert window.close() is True

    assert answered_choices == []
    assert ProjectService().load_project(root).project.name == "Renamed"


def test_a_failed_quiet_save_keeps_the_window_open_and_the_next_close_asks(
    open_window: MainWindow, answered_choices: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = open_window
    window._document.project.name = "Renamed"
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda _parent, title, text: messages.append((title, text))
    )

    def _full_disk(_document) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(window._project_service, "save_project", _full_disk)

    assert window.close() is False

    assert window.isVisible()
    assert answered_choices == []
    assert messages == [("Save blocked", str(OSError(28, "No space left on device")))]
    assert window._job_watchdog_timer.isActive()

    # The way out of a folder that stays unwritable: the next close offers Don't save.
    monkeypatch.setattr(
        main_window_module.workflow_table_utils, "confirm_choice", lambda *_a, **_k: "discard"
    )
    assert window.close() is True


@pytest.mark.parametrize("save", ["manual", "autosave"])
def test_a_later_good_save_clears_the_failed_close(
    open_window: MainWindow, answered_choices: list, save: str
) -> None:
    window = open_window
    window._close_save_failed = True

    if save == "manual":
        window._save_project()
    else:
        window._persist_document_ui_state(failure_message="failed")

    assert window.close() is True
    assert answered_choices == []


def test_an_unchanged_project_in_a_read_only_folder_closes(
    open_window: MainWindow, answered_choices: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = open_window

    def _refuse(_document) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(window._project_service, "save_project", _refuse)

    assert window.close() is True
    assert answered_choices == []


# --- New Project ----------------------------------------------------------------


def test_new_project_creates_the_shown_folder_and_remembers_the_location(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts = answer_new_project(monkeypatch, "  CoF2 ", location=tmp_path)

    window._new_project()

    assert window._document is not None
    assert window._document.project.name == "CoF2"
    assert window._document.project_root == tmp_path / "CoF2.mufinder"
    assert (tmp_path / "CoF2.mufinder" / "project.json").exists()
    assert window.windowTitle() == "CoF2 — MuFinder"
    assert window._recent_projects() == [str(tmp_path / "CoF2.mufinder")]

    assert starts != [str(tmp_path)]
    second = answer_new_project(monkeypatch, "Other", accept=False)
    window._new_project()
    assert second == [str(tmp_path)]


def test_cancelling_new_project_changes_nothing(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_new_project(monkeypatch, "CoF2", accept=False, location=tmp_path)

    window._new_project()

    assert window._document is None
    assert not (tmp_path / "CoF2.mufinder").exists()
    assert window._dialog_start_dir("new_project_parent") != str(tmp_path)
