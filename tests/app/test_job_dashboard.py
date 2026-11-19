"""The Job Dashboard widget: its rows, status cells, runtime, menus, keys and confirmations.

The tests drive it inside a main window, as the user meets it.
"""

from __future__ import annotations

from mufinder.services.job_monitoring import job_status_word
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QItemSelectionModel, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QHeaderView,
    QLineEdit,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import mufinder.app.main_window as main_window_module
import mufinder.app.job_dashboard as job_dashboard_module
import mufinder.app.workflow_table_utils as workflow_table_utils
from mufinder.app import selection_snapshot as selection_snapshot_module
from mufinder.app.selection_snapshot import SelectionScope
from mufinder.app.job_dashboard import JobDashboard
from mufinder.app.main_window import MainWindow
from mufinder.app.theme import _FOCUS_RING_NAME, apply_theme, muted_text_brush
from mufinder.domain.enums import EndReason, JobActivity, JobStatus, WorkflowStage
from mufinder.services.project_service import ProjectService
from tests.support.jobs import set_job_state
from tests.support.dashboard import (
    action_button,
    job_row,
    more_button,
    open_context_menu,
    open_more_menu,
)
from tests.support.reset_menu import FakeMenu
from tests.support.dashboard import add_dashboard_job, orphaned_cluster_job


def test_main_window_runtime_column_updates_for_running_jobs(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "runtime_tick.mufinder", "RuntimeTick")

    job = service.add_job(
        doc,
        label="uep_runtime_job",
        workflow_stage=main_window_module.WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    job.started_at = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(
        job_dashboard_module,
        "utc_now",
        lambda: datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC),
    )
    window._document = doc
    window._refresh_jobs_table()

    assert window._job_dashboard.runtime_timer.isActive() is True
    assert (
        window._job_dashboard.table.item(0, window._job_dashboard.RUNTIME_COLUMN).text() == "00:00"
    )

    monkeypatch.setattr(
        job_dashboard_module,
        "utc_now",
        lambda: datetime(2026, 4, 20, 16, 1, 5, tzinfo=UTC),
    )
    window._job_dashboard.refresh_runtime_cells()

    assert (
        window._job_dashboard.table.item(0, window._job_dashboard.RUNTIME_COLUMN).text() == "01:05"
    )

    job.status = JobStatus.COMPLETED
    job.completed_at = datetime(2026, 4, 20, 16, 1, 5, tzinfo=UTC)
    window._job_dashboard.refresh_runtime_cells()

    assert window._job_dashboard.runtime_timer.isActive() is False
    window.close()


def test_main_window_cluster_runtime_uses_queue_elapsed_and_ticks_between_polls(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_runtime_tick.mufinder", "ClusterRuntimeTick")

    job = service.add_job(
        doc,
        label="uep_cluster_runtime_job",
        workflow_stage=main_window_module.WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    job.execution_target = "redwood"

    window._document = doc
    window._refresh_jobs_table()

    snapshot_time = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(job_dashboard_module, "utc_now", lambda: snapshot_time)
    window._on_cluster_queue_checked(
        SimpleNamespace(
            checked_at=snapshot_time,
            elapsed_by_job_id={job.id: "00:12:34"},
        )
    )

    assert (
        window._job_dashboard.table.item(0, window._job_dashboard.RUNTIME_COLUMN).text() == "12:34"
    )
    assert window._job_dashboard.runtime_timer.isActive() is True

    monkeypatch.setattr(
        job_dashboard_module,
        "utc_now",
        lambda: datetime(2026, 4, 20, 16, 1, 4, tzinfo=UTC),
    )
    window._job_dashboard.refresh_runtime_cells()

    assert (
        window._job_dashboard.table.item(0, window._job_dashboard.RUNTIME_COLUMN).text() == "13:38"
    )

    # A poll that no longer reports the job clears its scheduler runtime.
    window._on_cluster_queue_checked(
        SimpleNamespace(checked_at=snapshot_time, elapsed_by_job_id={})
    )
    assert window._job_dashboard.table.item(0, window._job_dashboard.RUNTIME_COLUMN).text() == ""
    window.close()


def test_main_window_runtime_timer_pauses_while_job_dock_hidden(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dock_pause.mufinder", "DockPause")

    job = service.add_job(
        doc,
        label="uep_dock_job",
        workflow_stage=main_window_module.WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    job.started_at = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(
        job_dashboard_module, "utc_now", lambda: datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    )

    window._document = doc
    window._refresh_jobs_table()
    assert window._job_dashboard.runtime_timer.isActive() is True

    # Dock becoming not-visible (closed, obscured, or window minimized — all of
    # which emit visibilityChanged(False)) should pause the per-second updates.
    window._job_dashboard.set_visible_to_user(False)
    assert window._job_dashboard._visible_to_user is False
    assert window._job_dashboard.runtime_timer.isActive() is False

    # A live-runtime resync while not visible must not re-arm the timer.
    window._job_dashboard.sync_runtime_timer()
    assert window._job_dashboard.runtime_timer.isActive() is False

    # Becoming visible again re-arms it and refreshes the (now stale) cells.
    window._job_dashboard.set_visible_to_user(True)
    assert window._job_dashboard._visible_to_user is True
    assert window._job_dashboard.runtime_timer.isActive() is True

    window.close()


def _cluster_job_with_retrieval_failure(service, doc, label="r0"):
    job = add_dashboard_job(service, doc, label, "failed")
    job.end_reason = EndReason.OUTPUT_UNAVAILABLE
    job.execution_target = "SCARF"
    job.remote_job_id = "4127731"
    job.remote_directory = "/remote/r0"
    return job


@pytest.mark.parametrize(
    ("status", "expected_text"),
    [
        ("submitting", "Cancel"),
        ("queued", "Cancel"),
        ("running", "Cancel"),
        ("retrieving", "Cancel"),
        ("failed", "Inspect…"),
        ("stopped", "Inspect…"),
        ("cancelled", "Resubmit…"),
    ],
)
def test_dashboard_action_button_fits_the_state(main_window, tmp_path, status, expected_text):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "primary.mufinder", "Primary")
    job = add_dashboard_job(service, doc, "j0", status)
    window._document = doc

    window._refresh_jobs_table()

    assert action_button(window).text() == expected_text
    assert action_button(window).accessibleName() == f"{expected_text.rstrip('…')} {job.label}"
    assert more_button(window).text() == "⋯"
    assert more_button(window).accessibleName() == f"More actions for {job.label}"
    window.close()


def test_dashboard_action_button_retries_a_failed_retrieval_only_while_connected(
    main_window, tmp_path, monkeypatch
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "primary_retry.mufinder", "PrimaryRetry")
    _cluster_job_with_retrieval_failure(service, doc)
    window._document = doc

    window._refresh_jobs_table()
    assert action_button(window).text() == "Inspect…"
    assert open_more_menu(window, monkeypatch).action("Check now") is None

    window._cluster_runner._connected = True
    window._cluster_runner._connected_cluster_name = "SCARF"
    window._refresh_jobs_table()
    assert action_button(window).text() == "Check now"
    menu = open_more_menu(window, monkeypatch)
    assert menu.action("Check now") is None
    assert menu.action("Inspect…") is not None
    window.close()


def test_dashboard_action_button_runs_its_action_for_its_job(main_window, tmp_path, monkeypatch):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "primary_click.mufinder", "PrimaryClick")
    jobs = [
        add_dashboard_job(service, doc, label, status)
        for label, status in (
            ("running", "running"),
            ("failed", "failed"),
            ("cancelled", "cancelled"),
        )
    ]
    window._document = doc
    calls: list[tuple[str, list[object]]] = []
    for kind, handler in (
        ("cancel", "_on_cancel_from_dashboard"),
        ("inspect", "_on_inspect_from_dashboard"),
        ("resubmit", "_on_resubmit_from_dashboard"),
    ):
        monkeypatch.setattr(window, handler, lambda ids, k=kind: calls.append((k, list(ids))))
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: True)

    window._refresh_jobs_table()
    for row in range(3):
        action_button(window, row).click()

    assert calls == [
        ("cancel", [jobs[0].id]),
        ("inspect", [jobs[1].id]),
        ("resubmit", [jobs[2].id]),
    ]
    window.close()


def test_dashboard_action_button_follows_the_job_without_being_rebuilt(
    main_window, tmp_path, monkeypatch
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "primary_change.mufinder", "PrimaryChange")
    job = add_dashboard_job(service, doc, "j0", "running")
    window._document = doc
    inspected: list[list[object]] = []
    monkeypatch.setattr(
        window, "_on_inspect_from_dashboard", lambda ids: inspected.append(list(ids))
    )
    window._refresh_jobs_table()
    button = action_button(window)

    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    window._refresh_jobs_table()

    assert action_button(window) is button
    assert button.text() == "Inspect…"
    button.click()
    assert inspected == [[job.id]]
    window.close()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("queued", ["Open in Cutoff Energy"]),
        ("running", ["Open in Cutoff Energy"]),
        (
            "failed",
            ["Resubmit…", "Reset…", "Dismiss", "Open in Cutoff Energy"],
        ),
        (
            "stopped",
            ["Resubmit…", "Reset…", "Dismiss", "Open in Cutoff Energy"],
        ),
        ("cancelled", ["Reset…", "Dismiss", "Open in Cutoff Energy"]),
    ],
)
def test_dashboard_more_menu_lists_the_rest_for_the_state(
    main_window, tmp_path, monkeypatch, status, expected
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "more.mufinder", "More")
    add_dashboard_job(service, doc, "j0", status)
    window._document = doc
    window._refresh_jobs_table()

    menu = open_more_menu(window, monkeypatch)

    assert [action.text for action in menu.actions] == expected
    assert menu.separators_before == [len(expected) - 1]
    window.close()


def test_dashboard_more_menu_on_a_selected_row_asks_about_the_selection(
    main_window, tmp_path, monkeypatch
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "more_scope.mufinder", "MoreScope")
    failed = add_dashboard_job(service, doc, "j0", "failed")
    stopped = add_dashboard_job(service, doc, "j1", "stopped")
    window._document = doc
    window._refresh_jobs_table()
    acted: list[list[object]] = []
    monkeypatch.setattr(window, "_on_reset_failed_from_dashboard", lambda ids: acted.append(ids))
    asked: list[str] = []
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda _parent, _title, body, *_a: asked.append(body) or SelectionScope.ALL_SELECTED,
    )

    more_button(window).selection_snapshot_rows = (0, 1)
    open_more_menu(window, monkeypatch).action("Reset…").triggered.emit()

    assert asked == ["Reset all 2 selected jobs?"]
    assert [set(ids) for ids in acted] == [{failed.id, stopped.id}]
    window.close()


def test_dashboard_more_menu_keeps_the_jobs_read_when_it_opened(main_window, tmp_path, monkeypatch):
    """A refresh while the menu is open moves rows, but not the jobs an entry acts on."""

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "more_moved.mufinder", "MoreMoved")
    gone = add_dashboard_job(service, doc, "j0", "failed")
    first = add_dashboard_job(service, doc, "j1", "failed")
    second = add_dashboard_job(service, doc, "j2", "failed")
    window._document = doc
    window._refresh_jobs_table()
    acted: list[list[object]] = []
    monkeypatch.setattr(window, "_on_reset_failed_from_dashboard", lambda ids: acted.append(ids))
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *_a, **_k: SelectionScope.ALL_SELECTED,
    )
    more_button(window, 1).selection_snapshot_rows = (1, 2)
    menu = open_more_menu(window, monkeypatch, 1)

    gone.dismissed_at = utc_now_for_test()
    window._refresh_jobs_table()
    menu.action("Reset…").triggered.emit()

    assert [set(ids) for ids in acted] == [{first.id, second.id}]
    window.close()


def test_dashboard_more_menu_entry_does_nothing_once_its_row_is_gone(
    main_window, tmp_path, monkeypatch
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "more_gone.mufinder", "MoreGone")
    job = add_dashboard_job(service, doc, "j0", "failed")
    window._document = doc
    window._refresh_jobs_table()
    acted: list[list[object]] = []
    monkeypatch.setattr(window, "_on_reset_failed_from_dashboard", lambda ids: acted.append(ids))
    menu = open_more_menu(window, monkeypatch)

    job.dismissed_at = utc_now_for_test()
    window._refresh_jobs_table()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    menu.action("Reset…").triggered.emit()

    assert acted == []
    assert window.statusBar().currentMessage() == "j0 has left the dashboard; nothing changed"
    window.close()


def test_dashboard_scope_prompt_survives_the_row_going_away(main_window, tmp_path, monkeypatch):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "prompt_gone.mufinder", "PromptGone")
    first = add_dashboard_job(service, doc, "j0", "failed")
    second = add_dashboard_job(service, doc, "j1", "failed")
    window._document = doc
    window._refresh_jobs_table()
    acted: list[list[object]] = []
    monkeypatch.setattr(window, "_on_reset_failed_from_dashboard", lambda ids: acted.append(ids))

    def row_goes_while_asking(*_args, **_kwargs):
        first.dismissed_at = utc_now_for_test()
        window._refresh_jobs_table()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        return SelectionScope.ALL_SELECTED

    monkeypatch.setattr(selection_snapshot_module, "ask_selection_scope", row_goes_while_asking)
    more_button(window).selection_snapshot_rows = (0, 1)
    open_more_menu(window, monkeypatch).action("Reset…").triggered.emit()

    assert [set(ids) for ids in acted] == [{first.id, second.id}]
    window.close()


def test_dashboard_more_menu_restores_the_selection_where_the_jobs_moved(
    main_window, tmp_path, monkeypatch
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "more_moved_rows.mufinder", "MoreMovedRows")
    gone = add_dashboard_job(service, doc, "j0", "failed")
    add_dashboard_job(service, doc, "j1", "failed")
    add_dashboard_job(service, doc, "j2", "failed")
    add_dashboard_job(service, doc, "j3", "failed")
    window._document = doc
    window._refresh_jobs_table()
    monkeypatch.setattr(
        selection_snapshot_module, "ask_selection_scope", lambda *_a, **_k: SelectionScope.CANCEL
    )
    more_button(window, 2).selection_snapshot_rows = (2, 3)
    menu = open_more_menu(window, monkeypatch, 2)

    gone.dismissed_at = utc_now_for_test()
    window._refresh_jobs_table()
    menu.action("Reset…").triggered.emit()

    selected = sorted(
        index.row() for index in window._job_dashboard.table.selectionModel().selectedRows()
    )
    assert selected == [1, 2]
    window.close()


def test_dashboard_rows_show_stage_status_where_and_cores(main_window, tmp_path):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cells.mufinder", "Cells")
    job = service.add_job(
        doc,
        label="site_1",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_geom_opt",
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.CASTEP_ERROR
    job.error_message = "SCF did not converge in 100 cycles\nmore detail"
    job.execution_target = "SCARF"
    job.remote_job_id = "4127731"
    job.num_processes = 64
    window._document = doc

    window._refresh_jobs_table()

    cell = lambda column: window._job_dashboard.table.item(0, column).text()  # noqa: E731
    assert cell(window._job_dashboard.LABEL_COLUMN) == "site_1"
    assert cell(window._job_dashboard.STAGE_COLUMN) == "Relaxations"
    assert (
        cell(window._job_dashboard.STATUS_COLUMN) == "failed · SCF did not converge in 100 cycles"
    )
    assert cell(window._job_dashboard.WHERE_COLUMN) == "SCARF #4127731"
    assert cell(window._job_dashboard.CORES_COLUMN) == "64"
    window.close()


def _press(key, *modifiers) -> None:
    """Send a real key press to whatever has keyboard focus now."""

    target = QApplication.focusWidget()
    assert target is not None
    combined = Qt.KeyboardModifier.NoModifier
    for modifier in modifiers:
        combined |= modifier
    QTest.keyClick(target, key, combined)
    QApplication.processEvents()


def _shown_dashboard(window, tmp_path, statuses):
    service = ProjectService()
    doc = service.create_project(tmp_path / "focus.mufinder", "Focus")
    jobs = [
        add_dashboard_job(service, doc, f"j{index}", status)
        for index, status in enumerate(statuses)
    ]
    window._document = doc
    window._sync_project_panels()
    window._refresh_jobs_table()
    window.resize(1400, 1000)
    window.show()
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    QApplication.processEvents()
    return jobs


def _selected_rows(window) -> list[int]:
    return sorted(
        index.row() for index in window._job_dashboard.table.selectionModel().selectedRows()
    )


def _focus_cell(window, row, column) -> None:
    table = window._job_dashboard.table
    table.setCurrentCell(row, column)
    table.setFocus(Qt.FocusReason.OtherFocusReason)
    QApplication.processEvents()


def _focus(window):
    QApplication.processEvents()
    return QApplication.focusWidget()


def test_dashboard_tab_goes_through_the_current_row_and_out_of_the_table(main_window, tmp_path):
    """The table is one stop on the way round the window, however many rows it has."""

    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"] * 20)
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 3, window._job_dashboard.CORES_COLUMN)

        _press(Qt.Key.Key_Tab)
        assert _focus(window) is action_button(window, 3)

        _press(Qt.Key.Key_Tab)
        assert _focus(window) is more_button(window, 3)

        _press(Qt.Key.Key_Tab)
        assert _focus(window) is not table
        assert not table.isAncestorOf(_focus(window))

        # Round the window and back, with no stop on any other row.
        stops = []
        for _press_number in range(60):
            _press(Qt.Key.Key_Tab)
            stops.append(_focus(window))
            if stops[-1] is table:
                break
        assert stops[-1] is table
        assert not any(table.isAncestorOf(stop) for stop in stops[:-1])
        assert table.currentRow() == 3
    finally:
        window.hide()


def test_dashboard_shift_tab_comes_back_through_the_row_and_out_of_the_table(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed", "failed"])
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is more_button(window, 1)

        _press(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
        assert _focus(window) is action_button(window, 1)

        _press(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
        assert _focus(window) is table
        assert table.currentRow() == 1

        _press(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
        assert not table.isAncestorOf(_focus(window))
        assert _focus(window) is not table
    finally:
        window.hide()


def test_dashboard_shift_tab_from_after_the_table_lands_on_the_table(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed", "failed"])
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)
        for _control_or_out in range(3):
            _press(Qt.Key.Key_Tab)

        _press(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)

        assert _focus(window) is table
    finally:
        window.hide()


def test_dashboard_tab_leaves_an_empty_table(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, [])
    table = window._job_dashboard.table
    try:
        table.setFocus(Qt.FocusReason.OtherFocusReason)
        assert _focus(window) is table

        _press(Qt.Key.Key_Tab)

        assert _focus(window) is not table
    finally:
        window.hide()


def test_dashboard_tab_leaves_a_clicked_control_from_its_own_row(
    main_window, tmp_path, monkeypatch
):
    """A click inside a selection focuses a button without moving the table's current cell."""

    window = main_window
    _shown_dashboard(window, tmp_path, ["failed", "failed", "failed"])
    table = window._job_dashboard.table
    monkeypatch.setattr(window._job_dashboard, "dispatch_action", lambda *_args, **_kwargs: None)
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        for row in (0, 1):
            table.selectionModel().select(
                table.model().index(row, 0),
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
            )
        QTest.mouseClick(action_button(window, 1), Qt.MouseButton.LeftButton)
        assert _focus(window) is action_button(window, 1)
        assert table.currentRow() == 0

        _press(Qt.Key.Key_Tab)
        assert _focus(window) is more_button(window, 1)

        _press(Qt.Key.Key_Tab)
        assert not table.isAncestorOf(_focus(window))
        assert table.currentRow() == 1
    finally:
        window.hide()


def test_dashboard_tab_through_a_row_keeps_a_multi_row_selection(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"] * 3)
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)
        for row in (1, 2):
            table.selectionModel().select(
                table.model().index(row, 0),
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
            )

        for _stop in range(3):
            _press(Qt.Key.Key_Tab)
            assert _selected_rows(window) == [1, 2]
    finally:
        window.hide()


def test_dashboard_tab_past_a_row_outside_the_selection_leaves_the_selection(
    main_window, tmp_path, monkeypatch
):
    """Passing through changes nothing; pressing the button moves the highlight as a click does."""

    window = main_window
    (_first, _second, third) = _shown_dashboard(window, tmp_path, ["failed"] * 3)
    table = window._job_dashboard.table
    inspected: list[list[object]] = []
    monkeypatch.setattr(
        window, "_on_inspect_from_dashboard", lambda ids: inspected.append(list(ids))
    )
    try:
        _focus_cell(window, 2, window._job_dashboard.LABEL_COLUMN)
        table.selectRow(0)
        table.setCurrentCell(
            2, window._job_dashboard.LABEL_COLUMN, QItemSelectionModel.SelectionFlag.NoUpdate
        )

        _press(Qt.Key.Key_Tab)
        assert _focus(window) is action_button(window, 2)
        assert _selected_rows(window) == [0]

        _press(Qt.Key.Key_Return)
        assert inspected == [[third.id]]
        assert _selected_rows(window) == [2]
    finally:
        window.hide()


@pytest.mark.parametrize(
    "key",
    [Qt.Key.Key_PageDown, Qt.Key.Key_PageUp, Qt.Key.Key_Home, Qt.Key.Key_End, Qt.Key.Key_A],
)
def test_dashboard_other_keys_on_a_button_do_not_move_the_table(main_window, tmp_path, key):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"] * 20)
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 5, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is action_button(window, 5)

        _press(key)

        assert _focus(window) is action_button(window, 5)
        assert table.currentRow() == 5
        assert _selected_rows(window) == [5]
    finally:
        window.hide()


@pytest.mark.parametrize("key", [Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right])
def test_dashboard_shift_arrow_on_a_button_does_nothing(main_window, tmp_path, key):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"] * 3)
    try:
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)

        _press(key, Qt.KeyboardModifier.ShiftModifier)

        assert _focus(window) is action_button(window, 1)
        assert _selected_rows(window) == [1]
    finally:
        window.hide()


def test_dashboard_arrow_onto_the_action_cell_stays_in_the_table(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"])
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 0, window._job_dashboard.CORES_COLUMN)

        _press(Qt.Key.Key_Right)

        assert table.currentColumn() == window._job_dashboard.ACTION_COLUMN
        assert _focus(window) is table
    finally:
        window.hide()


def test_dashboard_up_and_down_on_a_button_move_between_rows(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"] * 3)
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is action_button(window, 0)

        _press(Qt.Key.Key_Up)
        assert _focus(window) is action_button(window, 0)

        _press(Qt.Key.Key_Down)
        assert _focus(window) is action_button(window, 1)
        assert table.currentRow() == 1
        assert _selected_rows(window) == [1]

        _press(Qt.Key.Key_Tab)
        assert _focus(window) is more_button(window, 1)

        _press(Qt.Key.Key_Down)
        assert _focus(window) is more_button(window, 2)

        _press(Qt.Key.Key_Down)
        assert _focus(window) is more_button(window, 2)

        _press(Qt.Key.Key_Up)
        assert _focus(window) is more_button(window, 1)
        assert _selected_rows(window) == [1]
    finally:
        window.hide()


def test_dashboard_down_inside_a_multi_row_selection_keeps_it(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"] * 3)
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        for row in (0, 1):
            table.selectionModel().select(
                table.model().index(row, 0),
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
            )

        _press(Qt.Key.Key_Down)
        assert _focus(window) is action_button(window, 1)
        assert _selected_rows(window) == [0, 1]

        _press(Qt.Key.Key_Down)
        assert _focus(window) is action_button(window, 2)
        assert _selected_rows(window) == [2]
    finally:
        window.hide()


def test_dashboard_left_and_right_stay_on_the_row(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"] * 2)
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)

        _press(Qt.Key.Key_Left)
        assert _focus(window) is action_button(window, 0)

        _press(Qt.Key.Key_Right)
        assert _focus(window) is more_button(window, 0)

        _press(Qt.Key.Key_Right)
        assert _focus(window) is more_button(window, 0)

        _press(Qt.Key.Key_Left)
        assert _focus(window) is action_button(window, 0)
    finally:
        window.hide()


def _put_out_of_reach(control, hide: bool) -> None:
    if hide:
        control.setVisible(False)
    else:
        control.setEnabled(False)


@pytest.mark.parametrize("hide", [False, True], ids=["disabled", "hidden"])
def test_dashboard_tab_skips_a_control_that_cannot_take_focus(main_window, tmp_path, hide):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"] * 2)
    table = window._job_dashboard.table
    try:
        _put_out_of_reach(action_button(window, 0), hide)
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)

        _press(Qt.Key.Key_Tab)
        assert _focus(window) is more_button(window, 0)

        # Shift+Tab goes past it too, rather than sticking in the cell.
        _press(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
        assert _focus(window) is table

        _put_out_of_reach(more_button(window, 1), hide)
        table.setCurrentCell(1, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is action_button(window, 1)

        _press(Qt.Key.Key_Tab)
        assert not table.isAncestorOf(_focus(window))
        assert _focus(window) is not table
    finally:
        window.hide()


@pytest.mark.parametrize("hide", [False, True], ids=["disabled", "hidden"])
def test_dashboard_tab_goes_past_a_row_with_no_control_to_take_focus(main_window, tmp_path, hide):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"])
    table = window._job_dashboard.table
    try:
        for control in (action_button(window, 0), more_button(window, 0)):
            _put_out_of_reach(control, hide)
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)

        _press(Qt.Key.Key_Tab)

        assert _focus(window) is not table
        assert not table.isAncestorOf(_focus(window))
    finally:
        window.hide()


def test_dashboard_control_disabled_while_focused_passes_focus_along_the_row(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"])
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)

        action_button(window, 0).setEnabled(False)

        assert _focus(window) is more_button(window, 0)
    finally:
        window.hide()


def test_dashboard_last_control_disabled_while_focused_stays_on_the_row(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"])
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is more_button(window, 0)

        more_button(window, 0).setEnabled(False)

        assert _focus(window) is action_button(window, 0)
    finally:
        window.hide()


def test_dashboard_down_goes_past_a_row_with_no_control_to_take_focus(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"] * 3)
    try:
        for control in (action_button(window, 1), more_button(window, 1)):
            control.setEnabled(False)
        more_button(window, 2).setEnabled(False)
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is more_button(window, 0)

        _press(Qt.Key.Key_Down)

        # Row 2's own ⋯ is disabled, so its other control takes focus.
        assert _focus(window) is action_button(window, 2)
    finally:
        window.hide()


def _host_structure_field(window):
    window._refresh_ui_after_project_change()
    window.show_page("host_structure")
    QApplication.processEvents()
    return next(field for field in window.findChildren(QLineEdit) if field.isVisible())


def _leave(window, *jobs) -> None:
    for job in jobs:
        job.dismissed_at = utc_now_for_test()
    window._refresh_jobs_table()
    QApplication.processEvents()


def test_dashboard_field_keeps_focus_when_a_job_leaves(main_window, tmp_path):
    """Focus on a row's button, then in a field: the table's current cell is still that row."""

    window = main_window
    jobs = _shown_dashboard(window, tmp_path, ["running"] * 3)
    table = window._job_dashboard.table
    field = _host_structure_field(window)
    try:
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is action_button(window, 1)
        QTest.mouseClick(field, Qt.MouseButton.LeftButton)
        assert _focus(window) is field

        _leave(window, jobs[1])

        assert table.rowCount() == 2
        assert _focus(window) is field
    finally:
        window.hide()


def test_dashboard_table_keeps_focus_when_a_job_leaves(main_window, tmp_path):
    window = main_window
    jobs = _shown_dashboard(window, tmp_path, ["running"] * 3)
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)

        _leave(window, jobs[1])

        assert table.rowCount() == 2
        assert _focus(window) is table
    finally:
        window.hide()


def test_dashboard_clicked_button_passes_focus_to_the_next_row_when_its_row_leaves(
    main_window, tmp_path, monkeypatch
):
    window = main_window
    jobs = _shown_dashboard(window, tmp_path, ["running"] * 3)
    table = window._job_dashboard.table
    monkeypatch.setattr(window._job_dashboard, "dispatch_action", lambda *_args, **_kwargs: None)
    try:
        # The current cell is on another row, so nothing but the hand-over moves focus.
        _focus_cell(window, 2, window._job_dashboard.LABEL_COLUMN)
        QTest.mouseClick(action_button(window, 0), Qt.MouseButton.LeftButton)
        assert _focus(window) is action_button(window, 0)

        _leave(window, jobs[0])

        assert table.rowCount() == 2
        assert _focus(window) is action_button(window, 0)
        assert job_row(window, jobs[1].id) == 0
        assert table.currentRow() == 0
        assert _visible_rings(action_button(window, 0)) == []
    finally:
        window.hide()


def test_dashboard_tabbed_menu_button_passes_focus_to_the_same_control_with_one_ring(
    main_window, tmp_path
):
    window = main_window
    apply_theme(QApplication.instance(), "light")
    jobs = _shown_dashboard(window, tmp_path, ["running"] * 3)
    try:
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is more_button(window, 1)

        _leave(window, jobs[1])

        assert job_row(window, jobs[2].id) == 1
        assert _focus(window) is more_button(window, 1)
        assert len(_visible_rings(more_button(window, 1))) == 1
    finally:
        window.hide()


def test_dashboard_last_row_passes_focus_up_when_it_leaves(main_window, tmp_path):
    window = main_window
    jobs = _shown_dashboard(window, tmp_path, ["running"] * 3)
    try:
        _focus_cell(window, 2, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)

        _leave(window, jobs[2])

        assert _focus(window) is action_button(window, 1)
    finally:
        window.hide()


def test_dashboard_focus_passes_over_rows_leaving_at_the_same_time(main_window, tmp_path):
    window = main_window
    jobs = _shown_dashboard(window, tmp_path, ["running"] * 4)
    try:
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)

        _leave(window, jobs[1], jobs[2])

        assert job_row(window, jobs[3].id) == 1
        assert _focus(window) is action_button(window, 1)
    finally:
        window.hide()


def test_dashboard_focus_skips_a_neighbour_with_no_control_to_take_it(main_window, tmp_path):
    window = main_window
    jobs = _shown_dashboard(window, tmp_path, ["running"] * 3)
    try:
        for control in (action_button(window, 2), more_button(window, 2)):
            control.setEnabled(False)
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)

        _leave(window, jobs[1])

        assert _focus(window) is action_button(window, 0)
    finally:
        window.hide()


def test_dashboard_focus_stays_in_the_table_when_the_last_job_leaves(main_window, tmp_path):
    window = main_window
    jobs = _shown_dashboard(window, tmp_path, ["running"])
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)

        _leave(window, jobs[0])

        assert window._job_dashboard.table.rowCount() == 0
        assert _focus(window) is window._job_dashboard.table
    finally:
        window.hide()


def test_dashboard_hand_over_leaves_the_selection_alone(main_window, tmp_path):
    """A job leaving in the background does not replace what the user selected."""

    window = main_window
    jobs = _shown_dashboard(window, tmp_path, ["running"] * 4)
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        table.clearSelection()
        for row in (2, 3):
            table.selectionModel().select(
                table.model().index(row, 0),
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
            )

        _leave(window, jobs[0])

        assert _focus(window) is action_button(window, 0)
        assert _selected_rows(window) == [1, 2]
        assert {job_row(window, jobs[2].id), job_row(window, jobs[3].id)} == {1, 2}
    finally:
        window.hide()


def test_dashboard_row_leaving_under_another_window_hands_over_on_return(main_window, tmp_path):
    """A prompt has the application's focus; the window's own focus still has to move."""

    window = main_window
    jobs = _shown_dashboard(window, tmp_path, ["running"] * 2)
    other = QDialog(window)
    other.setLayout(QVBoxLayout())
    other.layout().addWidget(QLineEdit())
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        other.show()
        other.activateWindow()
        QTest.qWaitForWindowActive(other)
        assert window.focusWidget() is action_button(window, 0)

        _leave(window, jobs[0])

        assert window.focusWidget() is action_button(window, 0)
        assert job_row(window, jobs[1].id) == 0

        other.close()
        window.activateWindow()
        QTest.qWaitForWindowActive(window)
        assert _focus(window) is action_button(window, 0)
    finally:
        window.hide()


def test_dashboard_focus_survives_the_project_closing(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["running"] * 2)
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is action_button(window, 0)

        window._document = None
        window._refresh_jobs_table()
        window._sync_project_panels()

        assert table.rowCount() == 0
        focused = _focus(window)
        assert focused is not None
        assert focused.isVisible()
    finally:
        window.hide()


@pytest.mark.parametrize("key", [Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space])
def test_dashboard_keyboard_runs_the_action_button_and_the_menu_button(
    main_window, tmp_path, monkeypatch, key
):
    window = main_window
    (job,) = _shown_dashboard(window, tmp_path, ["failed"])
    inspected: list[list[object]] = []
    monkeypatch.setattr(
        window, "_on_inspect_from_dashboard", lambda ids: inspected.append(list(ids))
    )
    FakeMenu.instances = []
    monkeypatch.setattr(job_dashboard_module, "QMenu", FakeMenu)
    try:
        _focus_cell(window, 0, window._job_dashboard.CORES_COLUMN)

        _press(Qt.Key.Key_Tab)
        _press(key)
        assert inspected == [[job.id]]
        assert FakeMenu.instances == []

        _press(Qt.Key.Key_Tab)
        _press(key)
        assert len(FakeMenu.instances) == 1
    finally:
        window.hide()


def _visible_rings(control) -> list[QWidget]:
    return [child for child in control.findChildren(QWidget, _FOCUS_RING_NAME) if child.isVisible()]


def test_dashboard_focus_ring_shows_once_on_both_controls(main_window, tmp_path):
    window = main_window
    apply_theme(QApplication.instance(), "light")
    _shown_dashboard(window, tmp_path, ["failed"])
    try:
        _focus_cell(window, 0, window._job_dashboard.CORES_COLUMN)
        for button_at in (action_button, more_button):
            _press(Qt.Key.Key_Tab)
            assert len(_visible_rings(button_at(window, 0))) == 1, button_at.__name__
    finally:
        window.hide()


def test_dashboard_down_after_a_click_shows_the_focus_ring(main_window, tmp_path, monkeypatch):
    """The arrow is keyboard use even though it moves focus from a handler."""

    window = main_window
    apply_theme(QApplication.instance(), "light")
    _shown_dashboard(window, tmp_path, ["failed"] * 2)
    monkeypatch.setattr(window._job_dashboard, "dispatch_action", lambda *_args, **_kwargs: None)
    try:
        QTest.mouseClick(action_button(window, 0), Qt.MouseButton.LeftButton)

        _press(Qt.Key.Key_Down)

        assert _focus(window) is action_button(window, 1)
        assert len(_visible_rings(action_button(window, 1))) == 1
    finally:
        window.hide()


def test_dashboard_table_is_not_editable(main_window):
    assert (
        main_window._job_dashboard.table.editTriggers()
        == QAbstractItemView.EditTrigger.NoEditTriggers
    )


def test_main_window_dashboard_status_uses_color_cues(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "jobs_status_colors.mufinder", "JobsStatusColors")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    window._document = doc
    window._refresh_jobs_table()

    status_item = window._job_dashboard.table.item(0, window._job_dashboard.STATUS_COLUMN)
    assert status_item is not None
    assert status_item.foreground().color().name() == "#b91c1c"
    window.close()


def test_main_window_dashboard_shows_job_process_count(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "jobs_process_count.mufinder", "JobsProcessCount")
    job = service.add_job(
        doc,
        label="job_running",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    job.num_processes = 8
    window._document = doc

    window._refresh_jobs_table()

    processes_item = window._job_dashboard.table.item(0, window._job_dashboard.CORES_COLUMN)
    assert processes_item is not None
    assert processes_item.text() == "8"
    window.close()


def _dashboard_with_jobs(window, service, doc, statuses):
    jobs = [
        add_dashboard_job(service, doc, f"job_{index}_{status}", status)
        for index, status in enumerate(statuses)
    ]
    window._document = doc
    window._refresh_jobs_table()
    window._job_dashboard.table.selectAll()
    return jobs


@pytest.fixture(params=["delete", "backspace"])
def delete_key_shortcut(request):
    """Return the name of each shortcut the dashboard binds to its delete action."""

    return f"{request.param}_shortcut"


def test_dashboard_delete_keys_cancel_active_jobs_and_dismiss_the_rest(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delete_key_shortcut
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "delete_keys.mufinder", "DeleteKeys")
    running, queued, failed, stopped, cancelled = _dashboard_with_jobs(
        window,
        service,
        doc,
        [
            "running",
            "queued",
            "failed",
            "stopped",
            "cancelled",
        ],
    )
    prompts: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or True,
    )
    events: list[object] = []

    def fake_cancel(jobs):
        events.append(("cancel", jobs))
        window.statusBar().showMessage(f"Cancellation requested for {len(jobs)} jobs")

    monkeypatch.setattr(window, "_cancel_jobs", fake_cancel)
    reset: list[object] = []
    monkeypatch.setattr(window, "_reset_failed_jobs", lambda *a, **k: reset.append(a))

    shortcut = getattr(window._job_dashboard, delete_key_shortcut)
    assert shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    shortcut.activated.emit()

    # One summary before anything is touched, then every job handled.
    assert prompts == [
        (
            "Cancel and dismiss jobs",
            "Cancel 2 jobs and dismiss 3 jobs from the dashboard?",
            "Dismissed jobs stay in the project. Show them again with Show dismissed.",
            "Cancel 2, dismiss 3",
            "Keep them",
        )
    ]
    assert events == [("cancel", [running, queued])]
    assert all(job.dismissed_at is not None for job in (failed, stopped, cancelled))
    assert (
        window.statusBar().currentMessage()
        == "Cancellation requested for 2 jobs; dismissed 3 jobs from the dashboard"
    )
    # Delete never resets a job or deletes its records.
    assert reset == []
    assert all(job.id in doc.jobs for job in (running, queued, failed, stopped, cancelled))
    window.close()


def test_dashboard_delete_keys_are_bound_to_both_keys(main_window) -> None:
    window = main_window
    assert window._job_dashboard.delete_shortcut.key() == QKeySequence(
        QKeySequence.StandardKey.Delete
    )
    assert window._job_dashboard.backspace_shortcut.key() == QKeySequence(Qt.Key.Key_Backspace)
    # Bound on the dock, so they also work while its title-bar controls have focus.
    assert window._job_dashboard.delete_shortcut.parent() is window._jobs_dock
    assert window._job_dashboard.backspace_shortcut.parent() is window._jobs_dock


def test_dashboard_confirmations_centre_on_the_main_window_when_the_dock_floats(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    window._jobs_dock.setFloating(True)
    parents: list[object] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda parent, *_args, **_kwargs: parents.append(parent) or False,
    )

    window._job_dashboard.confirm_cancel_and_dismiss(["job"], [])

    assert parents == [window]


@pytest.mark.parametrize(
    ("status", "expected_prompt"),
    [
        (
            "running",
            ("Cancel job", "Cancel job 'job_0_running'?", "", "Cancel job", "Keep running"),
        ),
        (
            "failed",
            (
                "Dismiss job",
                "Dismiss job 'job_0_failed' from the dashboard?",
                "The job stays in the project. Show it again with Show dismissed.",
                "Dismiss job",
            ),
        ),
    ],
)
def test_dashboard_delete_key_refused_touches_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status, expected_prompt
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "delete_refused.mufinder", "DeleteRefused")
    (job,) = _dashboard_with_jobs(window, service, doc, [status])
    prompts: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or False,
    )
    touched: list[str] = []
    monkeypatch.setattr(window, "_cancel_jobs", lambda *_a, **_k: touched.append("cancel"))

    window._job_dashboard.delete_selected()

    assert prompts == [expected_prompt]
    assert touched == []
    assert job.dismissed_at is None
    window.close()


def test_dashboard_delete_key_with_nothing_selected_asks_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "delete_none.mufinder", "DeleteNone")
    add_dashboard_job(service, doc, "j0", "failed")
    window._document = doc
    window._refresh_jobs_table()
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda *_a, **_k: pytest.fail("nothing is selected, so nothing to ask"),
    )

    window._job_dashboard.delete_selected()
    window.close()


def _dashboard_row_widgets(window, row):
    """Snapshot the cell widgets + items for a dashboard row (identity checks)."""
    return {
        "label": window._job_dashboard.table.item(row, window._job_dashboard.LABEL_COLUMN),
        "status": window._job_dashboard.table.item(row, window._job_dashboard.STATUS_COLUMN),
        "action": action_button(window, row),
        "more": more_button(window, row),
    }


def test_dashboard_reconcile_reuses_widgets_when_jobs_unchanged(
    main_window, tmp_path: Path
) -> None:
    """Two consecutive refreshes with the same jobs must reuse every cell widget."""
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_reuse.mufinder", "ReconcileReuse")
    add_dashboard_job(service, doc, "r0", "running")
    add_dashboard_job(service, doc, "r1", "failed")
    window._document = doc

    window._refresh_jobs_table()
    before = [
        _dashboard_row_widgets(window, row) for row in range(window._job_dashboard.table.rowCount())
    ]
    assert len(before) == 2

    window._refresh_jobs_table()
    after = [
        _dashboard_row_widgets(window, row) for row in range(window._job_dashboard.table.rowCount())
    ]
    assert len(after) == 2

    # Every item and every button is the identical object after re-refresh.
    for row_before, row_after in zip(before, after, strict=True):
        for key in row_before:
            assert row_before[key] is row_after[key], f"row widget {key} was recreated"
    window.close()


def test_dashboard_reconcile_status_change_updates_in_place(main_window, tmp_path: Path) -> None:
    """A status change updates the status cell + action button without widget churn."""
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_status.mufinder", "ReconcileStatus")
    job = add_dashboard_job(service, doc, "s0", "running")
    window._document = doc

    window._refresh_jobs_table()
    before = _dashboard_row_widgets(window, 0)
    assert before["status"].text() == "running"
    assert before["action"].text() == "Cancel"

    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    window._refresh_jobs_table()
    after = _dashboard_row_widgets(window, 0)

    # Same widget objects, updated text.
    for key in before:
        assert before[key] is after[key], f"row widget {key} was recreated on status change"
    assert after["status"].text() == "failed"
    assert after["action"].text() == "Inspect…"
    window.close()


def test_dashboard_reconcile_remove_job_removes_only_its_row(main_window, tmp_path: Path) -> None:
    """Dismissing a job removes exactly its row; the surviving row keeps its widgets."""
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_remove.mufinder", "ReconcileRemove")
    job0 = add_dashboard_job(service, doc, "x0", "failed")
    add_dashboard_job(service, doc, "x1", "running")
    window._document = doc

    window._refresh_jobs_table()
    assert window._job_dashboard.table.rowCount() == 2
    # Capture the surviving (running) row's widgets before removal.
    survivor_before = _dashboard_row_widgets(window, 1)

    job0.dismissed_at = utc_now_for_test()
    window._refresh_jobs_table()

    assert window._job_dashboard.table.rowCount() == 1
    remaining_id = window._job_dashboard.table.item(0, window._job_dashboard.LABEL_COLUMN).data(
        Qt.ItemDataRole.UserRole
    )
    assert remaining_id != str(job0.id)
    # The survivor's widgets are preserved (moved up to row 0 identity intact).
    survivor_after = _dashboard_row_widgets(window, 0)
    for key in survivor_before:
        assert survivor_before[key] is survivor_after[key], f"survivor widget {key} was recreated"
    window.close()


def test_dashboard_reconcile_add_job_inserts_row_preserving_others(
    main_window, tmp_path: Path
) -> None:
    """Adding a job appends a row; existing rows keep their widgets."""
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_add.mufinder", "ReconcileAdd")
    add_dashboard_job(service, doc, "a0", "running")
    window._document = doc

    window._refresh_jobs_table()
    assert window._job_dashboard.table.rowCount() == 1
    first_before = _dashboard_row_widgets(window, 0)

    new_job = add_dashboard_job(service, doc, "a1", "failed")
    window._refresh_jobs_table()

    assert window._job_dashboard.table.rowCount() == 2
    first_after = _dashboard_row_widgets(window, 0)
    for key in first_before:
        assert first_before[key] is first_after[key], (
            f"existing widget {key} was recreated on insert"
        )
    # New job landed in the appended row (document iteration order).
    assert window._job_dashboard.table.item(1, window._job_dashboard.LABEL_COLUMN).data(
        Qt.ItemDataRole.UserRole
    ) == str(new_job.id)
    window.close()


def test_dashboard_reconcile_button_dispatches_correct_job_id_after_churn(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After several reconciliations, a button still dispatches its own job id."""
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_closure.mufinder", "ReconcileClosure")
    job0 = add_dashboard_job(service, doc, "c0", "cancelled")
    job1 = add_dashboard_job(service, doc, "c1", "cancelled")
    window._document = doc

    resubmitted: list[list[object]] = []
    monkeypatch.setattr(
        window, "_on_resubmit_from_dashboard", lambda ids: resubmitted.append(list(ids))
    )

    window._refresh_jobs_table()
    # Churn: remove job0 (dismiss), refresh, re-add nothing but refresh again.
    job0.dismissed_at = utc_now_for_test()
    window._refresh_jobs_table()
    job0.dismissed_at = None
    window._refresh_jobs_table()  # job0 comes back (fresh row/closure)

    # Locate each job's action button by id and click it — must dispatch that job.
    action_button(window, job_row(window, job1.id)).click()
    action_button(window, job_row(window, job0.id)).click()

    assert resubmitted == [[job1.id], [job0.id]]
    window.close()


def utc_now_for_test():
    from datetime import datetime

    return datetime.now(UTC)


def test_main_window_dashboard_scope_prompt_cancel_does_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "snap_scope_cancel.mufinder", "SnapScopeCancel")
    for label in ("j0", "j1"):
        job = service.add_job(
            doc,
            label=label,
            workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        job.status = JobStatus.ACTIVE
        job.activity = JobActivity.RUNNING
    window._document = doc
    cancelled: list[list[object]] = []
    monkeypatch.setattr(
        window, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids))
    )
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda _parent, title, body, *a: asked.append((title, body)) or SelectionScope.CANCEL,
    )
    window._refresh_jobs_table()

    cancel_btn = action_button(window, 0)
    cancel_btn.selection_snapshot_rows = (0, 1)
    cancel_btn.clicked.emit()

    assert asked == [("Apply to selected jobs?", "Cancel all 2 selected jobs?")]
    assert cancelled == []
    window.close()


def test_main_window_dashboard_single_row_cancel_button_is_confirmed(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "single_cancel.mufinder", "Single")
    job = add_dashboard_job(service, doc, "j0", "running")
    window._document = doc
    acted: list[list[object]] = []
    monkeypatch.setattr(window, "_on_cancel_from_dashboard", lambda ids: acted.append(list(ids)))
    prompts: list[tuple[str, ...]] = []
    answers = [False, True]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or answers.pop(0),
    )
    window._refresh_jobs_table()

    button = action_button(window)
    button.selection_snapshot_rows = (0,)
    button.clicked.emit()
    assert acted == []

    button.selection_snapshot_rows = (0,)
    button.clicked.emit()
    assert acted == [[job.id]]
    assert prompts[0] == ("Cancel job", "Cancel job 'j0'?", "", "Cancel job", "Keep running")
    window.close()


def test_main_window_dashboard_more_menu_dismiss_is_confirmed(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "single_dismiss.mufinder", "Single")
    job = add_dashboard_job(service, doc, "j0", "failed")
    window._document = doc
    acted: list[list[object]] = []
    monkeypatch.setattr(window, "_on_dismiss_from_dashboard", lambda ids: acted.append(list(ids)))
    prompts: list[tuple[str, ...]] = []
    answers = [False, True]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or answers.pop(0),
    )
    window._refresh_jobs_table()

    open_more_menu(window, monkeypatch).action("Dismiss").triggered.emit()
    assert acted == []

    open_more_menu(window, monkeypatch).action("Dismiss").triggered.emit()
    assert acted == [[job.id]]
    assert prompts[0] == (
        "Dismiss job",
        "Dismiss job 'j0' from the dashboard?",
        "The job stays in the project. Show it again with Show dismissed.",
        "Dismiss job",
    )
    window.close()


def test_main_window_dashboard_multi_row_cancel_is_confirmed_by_the_scope_prompt_alone(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed_destructive_actions
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "multi_cancel.mufinder", "MultiCancel")
    jobs = []
    for label in ("j0", "j1"):
        job = service.add_job(
            doc,
            label=label,
            workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        job.status = JobStatus.ACTIVE
        job.activity = JobActivity.RUNNING
        jobs.append(job)
    window._document = doc
    cancelled: list[list[object]] = []
    monkeypatch.setattr(
        window, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids))
    )
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )
    window._refresh_jobs_table()

    button = action_button(window, 0)
    button.selection_snapshot_rows = (0, 1)
    button.clicked.emit()

    assert [set(ids) for ids in cancelled] == [{job.id for job in jobs}]
    assert confirmed_destructive_actions == []
    window.close()


def _dashboard_window_with_statuses(tmp_path: Path, name: str, statuses):
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / f"{name}.mufinder", name)
    jobs = []
    for index, status in enumerate(statuses):
        job = service.add_job(
            doc,
            label=f"j{index}",
            workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        set_job_state(job, status)
        jobs.append(job)
    window._document = doc
    window._refresh_jobs_table()
    return window, jobs


def _select_dashboard_rows(window, rows) -> None:
    selection_model = window._job_dashboard.table.selectionModel()
    selection_model.clearSelection()
    for row in rows:
        selection_model.select(
            window._job_dashboard.table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )


def test_main_window_dashboard_hint_appears_only_for_a_multi_row_selection(
    qapp, tmp_path: Path
) -> None:
    window, _jobs = _dashboard_window_with_statuses(
        tmp_path, "hint", ["failed", "failed", "running"]
    )
    hint = window._job_dashboard.selection_hint
    dock = window.findChild(main_window_module.QDockWidget, "jobDashboardDock")
    assert window._job_dashboard.table.parent() is dock.widget()
    assert hint.parent() is dock.widget()
    assert hint.isHidden()

    _select_dashboard_rows(window, [0])
    assert hint.isHidden()

    _select_dashboard_rows(window, [0, 1])
    assert not hint.isHidden()
    assert hint.text() == "2 selected \u00b7 row actions apply to all selected"
    prop = workflow_table_utils.MULTI_ROW_TARGET_PROPERTY
    for button_at in (action_button, more_button):
        assert bool(button_at(window, 0).property(prop)) is True
        assert bool(button_at(window, 2).property(prop)) is False

    _select_dashboard_rows(window, [])
    assert hint.isHidden()
    window.close()


def test_main_window_dashboard_menu_acts_on_the_eligible_selected_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, (running, failed, cancelled_job) = _dashboard_window_with_statuses(
        tmp_path, "menu", ["running", "failed", "cancelled"]
    )
    calls: list[tuple[str, list[object]]] = []
    for kind, handler in (
        ("resubmit", "_on_resubmit_from_dashboard"),
        ("inspect", "_on_inspect_failed_from_dashboard"),
        ("reset", "_on_reset_failed_from_dashboard"),
    ):
        monkeypatch.setattr(window, handler, lambda ids, k=kind: calls.append((k, list(ids))))
    _select_dashboard_rows(window, [0, 1, 2])

    menu = open_context_menu(window, monkeypatch)

    # Entries that apply to only some of the three selected jobs say how many.
    assert [action.text for action in menu.actions] == [
        "Cancel (1 of 3)",
        "Inspect… (1 of 3)",
        "Check now",
        "Resubmit… (2 of 3)",
        "Continue relaxation…",
        "Reset… (2 of 3)",
        "Dismiss (2 of 3)",
        "Open in Cutoff Energy",
    ]
    assert menu.separators_before == [7]
    assert {action.text for action in menu.actions if not action.enabled} == {
        "Check now",
        "Continue relaxation…",
    }

    menu.action("Resubmit… (2 of 3)").triggered.emit()
    menu.action("Inspect… (1 of 3)").triggered.emit()
    menu.action("Reset… (2 of 3)").triggered.emit()
    assert calls == [
        ("resubmit", [failed.id, cancelled_job.id]),
        ("inspect", [failed.id]),
        ("reset", [failed.id, cancelled_job.id]),
    ]
    window.close()


def test_main_window_dashboard_menu_confirms_cancel_and_dismiss(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, (running, failed) = _dashboard_window_with_statuses(
        tmp_path, "menu_confirm", ["running", "failed"]
    )
    calls: list[tuple[str, list[object]]] = []
    monkeypatch.setattr(
        window, "_on_cancel_from_dashboard", lambda ids: calls.append(("cancel", list(ids)))
    )
    monkeypatch.setattr(
        window, "_on_dismiss_from_dashboard", lambda ids: calls.append(("dismiss", list(ids)))
    )
    prompts: list[tuple[str, ...]] = []
    answers = [False, True, True]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or answers.pop(0),
    )
    _select_dashboard_rows(window, [0, 1])
    menu = open_context_menu(window, monkeypatch)

    menu.action("Cancel (1 of 2)").triggered.emit()
    assert calls == []
    menu.action("Cancel (1 of 2)").triggered.emit()
    menu.action("Dismiss (1 of 2)").triggered.emit()

    assert calls == [("cancel", [running.id]), ("dismiss", [failed.id])]
    assert prompts[0] == ("Cancel job", "Cancel job 'j0'?", "", "Cancel job", "Keep running")
    assert prompts[2][1] == "Dismiss job 'j1' from the dashboard?"
    window.close()


def test_main_window_dashboard_menu_selects_the_clicked_row(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _jobs = _dashboard_window_with_statuses(
        tmp_path, "menu_select", ["failed", "failed", "failed"]
    )
    _select_dashboard_rows(window, [0, 1])

    menu = open_context_menu(window, monkeypatch, row=2)

    selected = sorted(
        index.row() for index in window._job_dashboard.table.selectionModel().selectedRows()
    )
    assert selected == [2]
    # Records are not deleted from the dashboard.
    assert not any("Delete" in action.text for action in menu.actions)
    window.close()


def test_main_window_dashboard_menu_opens_a_shared_stage_only(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, (first, second) = _dashboard_window_with_statuses(
        tmp_path, "menu_open", ["failed", "failed"]
    )
    opened: list[list[object]] = []
    monkeypatch.setattr(
        window, "_on_open_in_stage_from_dashboard", lambda ids: opened.append(list(ids))
    )
    _select_dashboard_rows(window, [0, 1])

    open_context_menu(window, monkeypatch).action("Open in Cutoff Energy").triggered.emit()
    assert opened == [[first.id, second.id]]

    second.workflow_stage = main_window_module.WorkflowStage.UEP
    window._refresh_jobs_table()
    _select_dashboard_rows(window, [0, 1])
    action = open_context_menu(window, monkeypatch).action("Open in stage")
    assert action.enabled is False
    assert action.tooltip
    window.close()


def test_main_window_dashboard_menu_continues_one_relaxation_at_a_time(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, (first, second, running) = _dashboard_window_with_statuses(
        tmp_path, "menu_continue", ["failed", "stopped", "running"]
    )
    continued: list[object] = []
    monkeypatch.setattr(
        window,
        "_continue_relaxation_for_job",
        lambda job: (lambda j=job: continued.append(j.id)) if job is not running else None,
    )

    _select_dashboard_rows(window, [0, 1])
    action = open_context_menu(window, monkeypatch).action("Continue relaxation…")
    assert action.enabled is False
    assert action.tooltip == "Continue one relaxation at a time."

    _select_dashboard_rows(window, [0, 2])
    open_context_menu(window, monkeypatch).action("Continue relaxation… (1 of 2)").triggered.emit()
    assert continued == [first.id]

    open_more_menu(window, monkeypatch, 1).action("Continue relaxation…").triggered.emit()
    assert continued == [first.id, second.id]
    window.close()


_DASHBOARD_SCALE_ROWS = 300


def _add_scaled_dashboard_jobs(service, doc, *, now, count=_DASHBOARD_SCALE_ROWS):
    """Fill ``doc`` with ``count`` cluster jobs: running, queued, failed, stopped, running..."""

    statuses = (
        "running",
        "queued",
        "failed",
        "stopped",
        "running",
    )
    jobs = []
    for index in range(count):
        job = add_dashboard_job(service, doc, f"scaled_{index:03d}", statuses[index % 5])
        job.execution_target = "SCARF"
        job.remote_job_id = str(5_000_000 + index)
        job.num_processes = 64
        job.started_at = now - timedelta(hours=1)
        job.last_checked_at = now - timedelta(minutes=1)
        jobs.append(job)
    return jobs


def _record_dashboard_cell_writes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, int]]:
    """Record every text, brush and tooltip write to a table item as ``(kind, row, column)``."""

    writes: list[tuple[str, int, int]] = []
    for name in ("setText", "setForeground", "setToolTip"):
        original = getattr(QTableWidgetItem, name)

        def recorder(item, value, _original=original, _name=name):
            writes.append((_name, item.row(), item.column()))
            return _original(item, value)

        monkeypatch.setattr(QTableWidgetItem, name, recorder)
    return writes


def _freeze_dashboard_clock(monkeypatch: pytest.MonkeyPatch, moment: datetime) -> None:
    monkeypatch.setattr(job_dashboard_module, "utc_now", lambda: moment)


def _rows_in_jobs_viewport(window) -> set[int]:
    table = window._job_dashboard.table
    viewport = table.viewport().rect()
    return {
        row
        for row in range(table.rowCount())
        if viewport.intersects(table.visualRect(table.model().index(row, 0)))
    }


def test_dashboard_refresh_with_no_job_changes_writes_no_cells(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At 300 rows a refresh with nothing changed sets no text, brush or tooltip."""

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "scaled_refresh.mufinder", "ScaledRefresh")
    now = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    _freeze_dashboard_clock(monkeypatch, now)
    _add_scaled_dashboard_jobs(service, doc, now=now)
    window._document = doc
    window._refresh_jobs_table()
    assert window._job_dashboard.table.rowCount() == _DASHBOARD_SCALE_ROWS

    writes = _record_dashboard_cell_writes(monkeypatch)
    window._refresh_jobs_table()

    assert writes == []


def test_dashboard_runtime_tick_writes_only_visible_active_rows_that_changed(
    main_window, qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 1 s tick touches only the runtime cells of running rows in the viewport."""

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "scaled_tick.mufinder", "ScaledTick")
    now = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    _freeze_dashboard_clock(monkeypatch, now)
    _add_scaled_dashboard_jobs(service, doc, now=now)
    window._document = doc
    window._sync_project_panels()
    window.resize(1400, 900)
    window.show()
    try:
        qapp.processEvents()
        window._refresh_jobs_table()
        qapp.processEvents()

        writes = _record_dashboard_cell_writes(monkeypatch)
        window._job_dashboard.refresh_runtime_cells()
        assert writes == [], "an unchanged runtime must not be rewritten"

        job_by_row = {
            row: doc.jobs[
                window._job_dashboard.row_job_id(window._job_dashboard.table.item(row, 0))
            ]
            for row in range(window._job_dashboard.table.rowCount())
        }
        for scroll_to_bottom in (False, True):
            if scroll_to_bottom:
                window._job_dashboard.table.scrollToBottom()
                qapp.processEvents()
            visible = _rows_in_jobs_viewport(window)
            expected = {row for row in visible if job_status_word(job_by_row[row]) == "running"}
            assert expected, "the viewport must show at least one running row"
            assert len(visible) < _DASHBOARD_SCALE_ROWS

            now += timedelta(seconds=1)
            _freeze_dashboard_clock(monkeypatch, now)
            writes.clear()
            window._job_dashboard.refresh_runtime_cells()

            assert {kind for kind, _row, _column in writes} == {"setText"}
            assert {column for _kind, _row, column in writes} == {
                window._job_dashboard.RUNTIME_COLUMN
            }
            assert sorted(row for _kind, row, _column in writes) == sorted(expected)

        assert window._job_dashboard.runtime_timer.isActive() is True
    finally:
        # Hidden, the window closes at teardown without the save prompt.
        window.hide()


def test_dashboard_columns_are_sized_once_not_per_cell(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No column re-measures itself per cell change; the runtime column still widens."""

    window = main_window
    header = window._job_dashboard.table.horizontalHeader()
    assert all(
        header.sectionResizeMode(column) != QHeaderView.ResizeMode.ResizeToContents
        for column in range(window._job_dashboard.table.columnCount())
    )

    service = ProjectService()
    doc = service.create_project(tmp_path / "runtime_width.mufinder", "RuntimeWidth")
    now = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    _freeze_dashboard_clock(monkeypatch, now)
    job = add_dashboard_job(service, doc, "long_runner", "running")
    job.started_at = now - timedelta(hours=9999, minutes=59, seconds=59)
    window._document = doc
    window._refresh_jobs_table()
    runtime_item = window._job_dashboard.table.item(0, window._job_dashboard.RUNTIME_COLUMN)
    assert runtime_item.text() == "9999:59:59"
    width_before = window._job_dashboard.table.columnWidth(window._job_dashboard.RUNTIME_COLUMN)

    _freeze_dashboard_clock(monkeypatch, now + timedelta(seconds=1))
    window._job_dashboard.refresh_runtime_cells()

    assert runtime_item.text() == "10000:00:00"
    width_grown = window._job_dashboard.table.columnWidth(window._job_dashboard.RUNTIME_COLUMN)
    assert width_grown > width_before

    # A refresh that shortens the text fits the column back down.
    job.started_at = now
    window._refresh_jobs_table()

    assert runtime_item.text() == "00:01"
    assert (
        window._job_dashboard.table.columnWidth(window._job_dashboard.RUNTIME_COLUMN) < width_grown
    )


def _dashboard_labels(window) -> list[str]:
    return [
        window._job_dashboard.table.item(row, window._job_dashboard.LABEL_COLUMN).text()
        for row in range(window._job_dashboard.table.rowCount())
    ]


def test_show_dismissed_brings_dismissed_jobs_back_greyed_with_their_actions(main_window, tmp_path):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "show_dismissed.mufinder", "ShowDismissed")
    kept = add_dashboard_job(service, doc, "kept", "failed")
    hidden = orphaned_cluster_job(service, doc)
    service.dismiss_job(hidden)
    window._document = doc
    window._refresh_jobs_table()
    box = window._job_dashboard.show_dismissed_box

    assert box.text() == "Show dismissed"
    assert box.isChecked() is False
    assert _dashboard_labels(window) == [kept.label]

    box.setChecked(True)

    assert sorted(_dashboard_labels(window)) == sorted([kept.label, hidden.label])
    row = _dashboard_labels(window).index(hidden.label)
    label = window._job_dashboard.table.item(row, window._job_dashboard.LABEL_COLUMN)
    assert label.foreground() == muted_text_brush()
    assert label.toolTip().startswith("Dismissed ")
    kept_label = window._job_dashboard.table.item(
        _dashboard_labels(window).index(kept.label), window._job_dashboard.LABEL_COLUMN
    )
    assert kept_label.toolTip() == ""
    # A shown dismissed job keeps every action but Dismiss itself.
    assert window._job_dashboard.action_applies("reset", hidden) is True
    assert window._job_dashboard.action_applies("dismiss", hidden) is False
    assert window._job_dashboard.action_applies("dismiss", kept) is True

    box.setChecked(False)

    assert _dashboard_labels(window) == [kept.label]
    window.close()


def test_a_dismissed_job_made_active_again_shows_without_show_dismissed(main_window, tmp_path):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reappeared.mufinder", "Reappeared")
    job = orphaned_cluster_job(service, doc)
    service.dismiss_job(job)
    window._document = doc
    window._refresh_jobs_table()
    assert _dashboard_labels(window) == []

    service.mark_job_active(job, JobActivity.RETRIEVING)
    window._refresh_jobs_table()

    assert _dashboard_labels(window) == [job.label]
    window.close()


def test_show_dismissed_starts_unticked_in_each_project(main_window, tmp_path):
    window = main_window
    service = ProjectService()
    first = service.create_project(tmp_path / "first.mufinder", "First")
    hidden = orphaned_cluster_job(service, first)
    service.dismiss_job(hidden)
    service.save_project(first)
    second = service.create_project(tmp_path / "second.mufinder", "Second")
    other = orphaned_cluster_job(service, second, label="o1")
    service.dismiss_job(other)
    service.save_project(second)

    assert window.open_project(first.project_root)
    window._job_dashboard.show_dismissed_box.setChecked(True)
    assert _dashboard_labels(window) == [hidden.label]

    assert window.open_project(second.project_root)

    assert window._job_dashboard.show_dismissed_box.isChecked() is False
    assert _dashboard_labels(window) == []
    window.close()


def _add_job(service, doc, *, label: str):
    return service.add_job(
        doc,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )


def _status_item(window):
    return window._job_dashboard.table.item(0, window._job_dashboard.STATUS_COLUMN)


def test_format_relative_age_branches(qapp) -> None:
    now = datetime.now(UTC)
    assert JobDashboard.format_relative_age(None, now=now) == "—"
    assert JobDashboard.format_relative_age(now, now=now) == "just now"
    assert JobDashboard.format_relative_age(now - timedelta(minutes=3), now=now) == "3 min ago"
    assert JobDashboard.format_relative_age(now - timedelta(hours=5), now=now) == "5 hr ago"
    assert JobDashboard.format_relative_age(now - timedelta(days=1), now=now) == "1 day ago"
    assert JobDashboard.format_relative_age(now - timedelta(days=3), now=now) == "3 days ago"
    # Naive stamps are treated as UTC (older-build compatibility).
    naive = (now - timedelta(minutes=2)).replace(tzinfo=None)
    assert JobDashboard.format_relative_age(naive, now=now) == "2 min ago"


def test_status_tooltip_is_empty_without_metadata(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "nometa.mufinder", "NoMeta")
    job = _add_job(service, doc, label="cancelled")
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.CANCELLED
    window._document = doc
    window._refresh_jobs_table()

    assert _status_item(window).toolTip() == ""
    assert _status_item(window).icon().isNull()
    window.close()


def test_status_tooltip_carries_the_last_check(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "meta.mufinder", "Meta")
    job = _add_job(service, doc, label="running")
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    job.execution_target = "scarf"
    job.last_checked_at = datetime.now(UTC) - timedelta(minutes=2)
    job.last_status_source = "poll"
    job.last_status_message = "queued on scarf"
    window._document = doc
    job.remote_job_id = "4127731"
    # Connected to the job's cluster and checked recently, so the row is NOT stale.
    window._cluster_runner._connected = True
    window._cluster_runner._connected_cluster_name = "scarf"
    window._refresh_jobs_table()

    item = _status_item(window)
    assert item.text() == "running"
    assert item.toolTip() == "Last checked: 2 min ago\nSource: poll\nMessage: queued on scarf"
    assert item.icon().isNull()
    window.close()


def test_disconnected_active_cluster_job_gets_the_stale_icon(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "stale.mufinder", "Stale")
    job = _add_job(service, doc, label="queued")
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.QUEUED
    job.execution_target = "scarf"
    job.last_checked_at = datetime.now(UTC) - timedelta(minutes=1)
    window._document = doc
    # Cluster runner is NOT connected -> active cluster job is stale.
    window._cluster_runner._connected = False
    window._cluster_runner._connected_cluster_name = None
    window._refresh_jobs_table()

    item = _status_item(window)
    # The state reads as it always does: no amber text, no "stale" in the cell.
    assert item.text() == "queued"
    assert not item.icon().isNull()
    assert "Stale: cluster disconnected" in item.toolTip()

    # The row is updated in place: once the job is no longer stale its cell
    # drops the icon and the stale line.
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    window._refresh_jobs_table()

    assert _status_item(window) is item
    assert item.icon().isNull()
    assert "Stale" not in item.toolTip()
    window.close()
