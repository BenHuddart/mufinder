"""The Job Dashboard widget: its rows, status cells, runtime, menus, keys and confirmations.

The tests drive it inside a main window, as the user meets it.
"""

from __future__ import annotations

from mufinder.services.job_monitoring import (
    CANCEL_NOT_SENT_SOURCE,
    RESULT_NOT_RECORDED_PREFIX,
    SUBMISSION_CUT_OFF_SOURCE,
    job_status_word,
    submission_cut_off_message,
)
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
    QDockWidget,
    QHeaderView,
    QLineEdit,
    QPushButton,
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
from mufinder.domain.constants import IMPORTED_JOB_TAG
from mufinder.domain.enums import EndReason, JobActivity, JobStatus, WorkflowStage
from mufinder.services.cluster_runner import QueueCheckSnapshot
from mufinder.services.project_service import ProjectService
from tests.support.jobs import set_job_state
from tests.support.dashboard import (
    dismiss_button,
    give_jobs_rows,
    job_row,
    look_button,
    open_context_menu,
    row_actions,
    run_button,
    slot_button,
)
from tests.support.qt import release_keyboard_modifiers
from tests.support.row_menu import FakeMenu
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
    window._jobs._on_cluster_queue_checked(
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
    window._jobs._on_cluster_queue_checked(
        SimpleNamespace(checked_at=snapshot_time, elapsed_by_job_id={})
    )
    assert window._job_dashboard.table.item(0, window._job_dashboard.RUNTIME_COLUMN).text() == ""
    window.close()


def _running_cluster_job(service: ProjectService, doc, label: str):
    job = service.add_job(
        doc,
        label=label,
        workflow_stage=main_window_module.WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    job.execution_target = "redwood"
    return job


def _runtime_cells(window: MainWindow) -> dict[str, str]:
    table = window._job_dashboard.table
    return {
        table.item(row, 0).text(): table.item(row, window._job_dashboard.RUNTIME_COLUMN).text()
        for row in range(table.rowCount())
    }


def test_main_window_a_check_of_one_job_keeps_the_other_jobs_scheduler_runtime(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Check now on one job replaced every job's scheduler elapsed time,
    # so another running job's runtime cell lost its scheduler time.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "partial_check.mufinder", "PartialCheck")
    first = _running_cluster_job(service, doc, "first_running")
    second = _running_cluster_job(service, doc, "second_running")
    window._document = doc
    window._refresh_jobs_table()
    full_check = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    one_job_check = full_check + timedelta(minutes=1)
    monkeypatch.setattr(job_dashboard_module, "utc_now", lambda: full_check)
    window._jobs._on_cluster_queue_checked(
        QueueCheckSnapshot(
            checked_at=full_check,
            elapsed_by_job_id={first.id: "00:10:00", second.id: "00:20:00"},
            job_ids=(first.id, second.id),
        )
    )

    monkeypatch.setattr(job_dashboard_module, "utc_now", lambda: one_job_check)
    window._jobs._on_cluster_queue_checked(
        QueueCheckSnapshot(
            checked_at=one_job_check,
            elapsed_by_job_id={first.id: "00:11:30"},
            job_ids=(first.id,),
        )
    )

    # The checked job counts from its new time; the other from its own check.
    assert _runtime_cells(window) == {"first_running": "11:30", "second_running": "21:00"}
    window.close()


def test_main_window_a_checked_job_without_a_scheduler_time_loses_its_runtime(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "partial_check_drop.mufinder", "PartialCheckDrop")
    first = _running_cluster_job(service, doc, "first_running")
    second = _running_cluster_job(service, doc, "second_running")
    window._document = doc
    window._refresh_jobs_table()
    checked = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(job_dashboard_module, "utc_now", lambda: checked)
    window._jobs._on_cluster_queue_checked(
        QueueCheckSnapshot(
            checked_at=checked,
            elapsed_by_job_id={first.id: "00:10:00", second.id: "00:20:00"},
            job_ids=(first.id, second.id),
        )
    )

    window._jobs._on_cluster_queue_checked(
        QueueCheckSnapshot(checked_at=checked, elapsed_by_job_id={}, job_ids=(first.id,))
    )

    assert _runtime_cells(window) == {"first_running": "", "second_running": "20:00"}
    window.close()


def test_main_window_a_resubmitted_job_does_not_take_its_last_runs_scheduler_runtime(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A job drops out of the checks once it ends, so the time its last check
    # gave is still known when it is submitted again; the new run must not use it.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "resubmitted.mufinder", "Resubmitted")
    job = _running_cluster_job(service, doc, "resubmitted_running")
    window._document = doc
    window._refresh_jobs_table()
    checked = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(job_dashboard_module, "utc_now", lambda: checked)
    window._jobs._on_cluster_queue_checked(
        QueueCheckSnapshot(
            checked_at=checked,
            elapsed_by_job_id={job.id: "01:30:00"},
            job_ids=(job.id,),
        )
    )
    assert _runtime_cells(window) == {"resubmitted_running": "1:30:00"}

    # Submitted again and running before any check has reported on the new run.
    job.submitted_at = checked + timedelta(minutes=5)
    job.started_at = checked + timedelta(minutes=6)
    monkeypatch.setattr(job_dashboard_module, "utc_now", lambda: checked + timedelta(minutes=7))
    window._refresh_jobs_table()

    assert _runtime_cells(window) == {"resubmitted_running": "01:00"}

    # A stamp an older build saved without a time zone compares as UTC.
    job.submitted_at = job.submitted_at.replace(tzinfo=None)
    window._refresh_jobs_table()
    assert _runtime_cells(window) == {"resubmitted_running": "01:00"}
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
    ("status", "expected"),
    [
        ("submitting", ("Cancel", "", "")),
        ("queued", ("Cancel", "", "")),
        ("running", ("Cancel", "", "")),
        ("retrieving", ("Cancel", "", "")),
        ("failed", ("Resubmit…", "Inspect…", "✕")),
        ("stopped", ("Resubmit…", "Inspect…", "✕")),
        ("cancelled", ("Resubmit…", "", "✕")),
    ],
)
def test_dashboard_row_slots_fit_the_state(main_window, tmp_path, monkeypatch, status, expected):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "slots.mufinder", "Slots")
    job = add_dashboard_job(service, doc, "j0", status)
    window._document = doc
    give_jobs_rows(window, monkeypatch)

    window._refresh_jobs_table()

    assert row_actions(window) == expected
    names = {
        "Cancel": "Cancel j0",
        "Resubmit…": "Resubmit j0",
        "Inspect…": "Inspect j0",
        "✕": f"Dismiss {job.label}",
    }
    for slot, shown in enumerate(expected):
        if shown:
            assert slot_button(window, 0, slot).accessibleName() == names[shown]
    window.close()


def test_dashboard_row_offers_check_now_in_place_of_inspect_connected_or_not(
    main_window, tmp_path, monkeypatch
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "check_slot.mufinder", "CheckSlot")
    job = _cluster_job_with_retrieval_failure(service, doc)
    window._document = doc
    give_jobs_rows(window, monkeypatch)

    window._refresh_jobs_table()
    assert row_actions(window) == ("Resubmit…", "Check now", "✕")
    assert look_button(window).accessibleName() == f"Check now {job.label}"

    window._cluster_runner._connected = True
    window._cluster_runner._connected_cluster_name = "SCARF"
    window._refresh_jobs_table()
    assert row_actions(window) == ("Resubmit…", "Check now", "✕")

    # Once its output has been read to a verdict there is nothing left to fetch.
    job.end_reason = EndReason.CASTEP_ERROR
    window._refresh_jobs_table()
    assert row_actions(window) == ("Resubmit…", "Inspect…", "✕")
    window.close()


#: Each case, the row's slots, and the reason the status tooltip gives for Check now.
_CHECK_NOW_CASES = {
    "cancel-not-sent": (
        ("Resubmit…", "Check now", "✕"),
        "Cancelled in the project only: SCARF was not told, so the job may still be running "
        "there. Use 'Check now' to find out.",
    ),
    "cut-off": (
        ("Resubmit…", "Check now", "✕"),
        "Its submission was cut off while MuFinder was closing; it may be running on SCARF. "
        "Use 'Check now' to find out.",
    ),
    "result-not-recorded": (
        ("Resubmit…", "Check now", "✕"),
        "The job finished, but MuFinder could not record its result. Use 'Check now' to "
        "record it from the output on SCARF.",
    ),
    "cancel-confirmed": (("Resubmit…", "", "✕"), None),
    "cut-off-settled": (("Resubmit…", "", "✕"), None),
    "castep-error": (("Resubmit…", "Inspect…", "✕"), None),
}


@pytest.mark.parametrize("case", list(_CHECK_NOW_CASES))
def test_dashboard_row_offers_check_now_while_cluster_state_is_still_to_fetch(
    main_window, tmp_path, monkeypatch, case
):
    expected, reason = _CHECK_NOW_CASES[case]
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "check_cases.mufinder", "CheckCases")
    job = _cluster_job_with_retrieval_failure(service, doc)
    job.last_status_source = "retrieval"
    # A later check that found nothing left this message behind.
    job.last_status_message = "checked; no complete output found"
    if case.startswith(("cancel", "cut-off")):
        set_job_state(job, "cancelled")
    if case == "cancel-not-sent":
        job.last_status_source = CANCEL_NOT_SENT_SOURCE
    elif case.startswith("cut-off"):
        job.remote_job_id = None
        job.error_message = submission_cut_off_message(job)
        if case == "cut-off":
            job.last_status_source = SUBMISSION_CUT_OFF_SOURCE
        else:
            job.last_status_source = "scheduler_poll"
    elif case == "result-not-recorded":
        job.end_reason = EndReason.UNKNOWN
        job.error_message = f"{RESULT_NOT_RECORDED_PREFIX}: unexpected output layout"
    elif case == "castep-error":
        job.end_reason = EndReason.CASTEP_ERROR
    window._document = doc
    give_jobs_rows(window, monkeypatch)

    window._refresh_jobs_table()

    assert row_actions(window) == expected
    tooltip = window._job_dashboard.table.item(0, window._job_dashboard.STATUS_COLUMN).toolTip()
    if reason is None:
        assert "Use 'Check now'" not in tooltip
    else:
        assert reason in tooltip
        assert "Use Inspect" not in tooltip
    # A stamp still awaiting an answer is explained in words, not as raw fields.
    assert ("Source: " in tooltip) is (case not in {"cancel-not-sent", "cut-off"})
    window.close()


def test_dashboard_tooltip_keeps_the_raw_stamp_where_it_gives_no_check_now_reason(
    main_window, tmp_path, monkeypatch
):
    # Without a remote folder the row offers no Check now, so the tooltip has no
    # reason to give and keeps the stamp's own lines.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "no_folder.mufinder", "NoFolder")
    job = _cluster_job_with_retrieval_failure(service, doc)
    set_job_state(job, "cancelled")
    job.remote_directory = None
    job.last_status_source = CANCEL_NOT_SENT_SOURCE
    job.last_status_message = "cancelled in the project only; the scheduler was not told"
    window._document = doc
    give_jobs_rows(window, monkeypatch)

    window._refresh_jobs_table()

    assert row_actions(window) == ("Resubmit…", "", "✕")
    tooltip = window._job_dashboard.table.item(0, window._job_dashboard.STATUS_COLUMN).toolTip()
    assert "Source: cancelled_in_project" in tooltip
    assert "Message: cancelled in the project only; the scheduler was not told" in tooltip
    window.close()


@pytest.mark.parametrize(
    ("status", "imported", "expected"),
    [
        ("failed", False, ("", "Inspect…", "✕")),
        ("cancelled", False, ("", "", "✕")),
        ("failed", True, ("", "Inspect…", "✕")),
        ("cancelled", True, ("", "", "✕")),
    ],
    ids=["rowless-failed", "rowless-cancelled", "imported-failed", "imported-cancelled"],
)
def test_dashboard_run_slot_is_blank_for_a_job_that_cannot_be_resubmitted(
    main_window, tmp_path, monkeypatch, status, imported, expected
):
    """A job with no row, or an imported one, shows no Resubmit…, not even a greyed one."""

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "blank_run.mufinder", "BlankRun")
    job = add_dashboard_job(service, doc, "j0", status)
    if imported:
        # Its row still holds it, but MuFinder did not run it.
        job.tags = [*job.tags, IMPORTED_JOB_TAG]
        monkeypatch.setattr(window._jobs, "_row_for_job", lambda _job: object())
    window._document = doc

    window._refresh_jobs_table()

    assert row_actions(window) == expected
    assert run_button(window).isHidden()
    window.close()


@pytest.mark.parametrize(
    ("unretrieved", "expected"),
    [(False, ("Resubmit…", "Inspect…", "")), (True, ("Resubmit…", "Check now", ""))],
)
def test_dashboard_dismissed_row_keeps_its_actions_but_dismiss(
    main_window, tmp_path, monkeypatch, unretrieved, expected
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismissed_slots.mufinder", "DismissedSlots")
    job = (
        _cluster_job_with_retrieval_failure(service, doc)
        if unretrieved
        else add_dashboard_job(service, doc, "j0", "failed")
    )
    job.dismissed_at = utc_now_for_test()
    window._document = doc
    give_jobs_rows(window, monkeypatch)
    window._job_dashboard.show_dismissed_box.setChecked(True)

    window._refresh_jobs_table()

    assert row_actions(window) == expected
    window.close()


def test_dashboard_slots_line_up_down_the_table_whatever_each_row_shows(
    main_window, tmp_path, monkeypatch
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "aligned.mufinder", "Aligned")
    for label, status in (("q", "queued"), ("f", "failed"), ("c", "cancelled")):
        add_dashboard_job(service, doc, label, status)
    _cluster_job_with_retrieval_failure(service, doc)
    window._document = doc
    give_jobs_rows(window, monkeypatch)
    window._sync_project_panels()
    window._refresh_jobs_table()
    window.resize(1400, 1000)
    window.show()
    QApplication.processEvents()
    try:
        table = window._job_dashboard.table
        assert [row_actions(window, row) for row in range(4)] == [
            ("Cancel", "", ""),
            ("Resubmit…", "Inspect…", "✕"),
            ("Resubmit…", "", "✕"),
            ("Resubmit…", "Check now", "✕"),
        ]
        for slot in range(3):
            geometries = {slot_button(window, row, slot).geometry() for row in range(4)}
            assert len(geometries) == 1, slot
            assert (
                slot_button(window, 0, slot).width()
                >= slot_button(window, 0, slot).sizeHint().width()
            )
        # Each slot is as wide as the widest label it can show.
        for slot, texts in ((0, ("Cancel", "Resubmit…")), (1, ("Inspect…", "Check now"))):
            button = slot_button(window, 1, slot)
            widest = max(button.fontMetrics().horizontalAdvance(text) for text in texts)
            assert button.contentsRect().width() >= widest
        # The column holds the three slots without cutting any of them off.
        cell = table.cellWidget(0, window._job_dashboard.ACTION_COLUMN)
        assert table.columnWidth(window._job_dashboard.ACTION_COLUMN) >= cell.sizeHint().width()
    finally:
        window.hide()


def test_dashboard_has_no_more_actions_button(main_window, tmp_path, monkeypatch):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "no_more.mufinder", "NoMore")
    for status in ("running", "failed", "cancelled"):
        add_dashboard_job(service, doc, status, status)
    window._document = doc
    give_jobs_rows(window, monkeypatch)
    window._refresh_jobs_table()

    buttons = window._job_dashboard.findChildren(QPushButton)
    assert not [button for button in buttons if "⋯" in button.text()]
    assert not [button for button in buttons if "more" in button.accessibleName().casefold()]
    for row in range(3):
        assert len(window._job_dashboard.table.cellWidget(row, 6).row_controls) == 3
    window.close()


def test_dashboard_dismiss_button_is_an_icon_with_a_name_and_a_tooltip(main_window, tmp_path):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismiss_icon.mufinder", "DismissIcon")
    add_dashboard_job(service, doc, "j0", "failed")
    window._document = doc
    window._refresh_jobs_table()

    button = dismiss_button(window)
    assert button.text() == ""
    assert not button.icon().isNull()
    assert button.accessibleName() == "Dismiss j0"
    assert button.toolTip() == (
        "Dismiss this job from the dashboard. Show dismissed lists it again."
    )
    window.close()


def test_dashboard_dismiss_button_confirms_before_dismissing(main_window, tmp_path, monkeypatch):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismiss_click.mufinder", "DismissClick")
    job = add_dashboard_job(service, doc, "j0", "failed")
    window._document = doc
    window._refresh_jobs_table()
    dismissed: list[list[object]] = []
    monkeypatch.setattr(
        window._jobs, "_on_dismiss_from_dashboard", lambda ids: dismissed.append(list(ids))
    )
    asked: list[tuple] = []
    answer = [False]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda *args, **_kwargs: asked.append(args[1:]) or answer[0],
    )

    dismiss_button(window).click()
    assert dismissed == []

    answer[0] = True
    dismiss_button(window).click()

    assert dismissed == [[job.id]]
    assert asked[0] == (
        "Dismiss job",
        "Dismiss job 'j0' from the dashboard?",
        "The job stays in the project. Show it again with Show dismissed.",
        "Dismiss job",
    )
    window.close()


def test_dashboard_row_buttons_run_their_action_for_their_job(main_window, tmp_path, monkeypatch):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "slot_click.mufinder", "SlotClick")
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
        monkeypatch.setattr(window._jobs, handler, lambda ids, k=kind: calls.append((k, list(ids))))
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: True)
    give_jobs_rows(window, monkeypatch)

    window._refresh_jobs_table()
    run_button(window, 0).click()
    look_button(window, 1).click()
    run_button(window, 1).click()
    run_button(window, 2).click()

    assert calls == [
        ("cancel", [jobs[0].id]),
        ("inspect", [jobs[1].id]),
        ("resubmit", [jobs[1].id]),
        ("resubmit", [jobs[2].id]),
    ]
    window.close()


def test_dashboard_row_buttons_follow_the_job_without_being_rebuilt(
    main_window, tmp_path, monkeypatch
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "slot_change.mufinder", "SlotChange")
    job = add_dashboard_job(service, doc, "j0", "running")
    window._document = doc
    inspected: list[list[object]] = []
    monkeypatch.setattr(
        window._jobs, "_on_inspect_from_dashboard", lambda ids: inspected.append(list(ids))
    )
    window._refresh_jobs_table()
    buttons = [slot_button(window, 0, slot) for slot in range(3)]
    assert row_actions(window) == ("Cancel", "", "")

    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    window._refresh_jobs_table()

    assert [slot_button(window, 0, slot) for slot in range(3)] == buttons
    assert row_actions(window) == ("", "Inspect…", "✕")
    look_button(window).click()
    assert inspected == [[job.id]]
    window.close()


@pytest.mark.parametrize("status", ["failed", "stopped", "cancelled"])
def test_dashboard_context_menu_of_an_ended_job_offers_no_reset(
    main_window, tmp_path, monkeypatch, status
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "context_ended.mufinder", "ContextEnded")
    add_dashboard_job(service, doc, "j0", status)
    window._document = doc
    give_jobs_rows(window, monkeypatch)
    window._refresh_jobs_table()

    menu = open_context_menu(window, monkeypatch)

    assert menu.texts() == [
        "Cancel",
        "Inspect…",
        "Check now",
        "Resubmit…",
        "Dismiss",
        "Open in Cutoff Energy",
    ]
    enabled = {action.text for action in menu.actions if action.enabled}
    assert enabled == {"Resubmit…", "Dismiss", "Open in Cutoff Energy"} | (
        set() if status == "cancelled" else {"Inspect…"}
    )
    window.close()


def test_dashboard_context_menu_offers_check_now_only_with_output_to_fetch(
    main_window, tmp_path, monkeypatch
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "context_check.mufinder", "ContextCheck")
    add_dashboard_job(service, doc, "running", "running")
    add_dashboard_job(service, doc, "failed", "failed")
    _cluster_job_with_retrieval_failure(service, doc)
    window._document = doc
    window._refresh_jobs_table()

    def check_entry(row):
        menu = open_context_menu(window, monkeypatch, row)
        return next(action for action in menu.actions if action.text.startswith("Check now"))

    assert check_entry(0).enabled is False
    assert check_entry(1).enabled is False
    assert check_entry(2).enabled is True

    window._job_dashboard.table.selectAll()
    menu = open_context_menu(window, monkeypatch, 2)
    entries = {action.text: action.enabled for action in menu.actions}
    assert entries["Check now (1 of 3)"] is True
    assert entries["Cancel (1 of 3)"] is True
    assert entries["Dismiss (2 of 3)"] is True
    assert entries["Inspect… (2 of 3)"] is True
    window.close()


def test_dashboard_row_button_on_a_selected_row_asks_about_the_selection(
    main_window, tmp_path, monkeypatch
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "slot_scope.mufinder", "SlotScope")
    failed = add_dashboard_job(service, doc, "j0", "failed")
    stopped = add_dashboard_job(service, doc, "j1", "stopped")
    window._document = doc
    acted: list[list[object]] = []
    give_jobs_rows(window, monkeypatch)
    window._refresh_jobs_table()
    monkeypatch.setattr(window._jobs, "_on_resubmit_from_dashboard", lambda ids: acted.append(ids))
    asked: list[str] = []
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda _parent, _title, body, *_a: asked.append(body) or SelectionScope.ALL_SELECTED,
    )

    run_button(window).selection_snapshot_rows = (0, 1)
    run_button(window).click()

    assert asked == ["Resubmit all 2 selected jobs?"]
    assert [set(ids) for ids in acted] == [{failed.id, stopped.id}]
    window.close()


def test_dashboard_scope_prompt_survives_the_row_going_away(main_window, tmp_path, monkeypatch):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "prompt_gone.mufinder", "PromptGone")
    first = add_dashboard_job(service, doc, "j0", "failed")
    second = add_dashboard_job(service, doc, "j1", "failed")
    window._document = doc
    acted: list[list[object]] = []
    give_jobs_rows(window, monkeypatch)
    window._refresh_jobs_table()
    monkeypatch.setattr(window._jobs, "_on_resubmit_from_dashboard", lambda ids: acted.append(ids))

    def row_goes_while_asking(*_args, **_kwargs):
        first.dismissed_at = utc_now_for_test()
        window._refresh_jobs_table()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        return SelectionScope.ALL_SELECTED

    monkeypatch.setattr(selection_snapshot_module, "ask_selection_scope", row_goes_while_asking)
    run_button(window).selection_snapshot_rows = (0, 1)
    run_button(window).click()

    assert [set(ids) for ids in acted] == [{first.id, second.id}]
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
    # The modifier is let go, as a real key would be, before anything reads it.
    release_keyboard_modifiers()
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
        assert _focus(window) is look_button(window, 3)

        _press(Qt.Key.Key_Tab)
        assert _focus(window) is dismiss_button(window, 3)

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
        assert _focus(window) is dismiss_button(window, 1)

        _press(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
        assert _focus(window) is look_button(window, 1)

        _press(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
        assert _focus(window) is table
        assert table.currentRow() == 1

        _press(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
        assert not table.isAncestorOf(_focus(window))
        assert _focus(window) is not table
    finally:
        window.hide()


def test_a_shifted_key_press_leaves_no_modifier_held(main_window, tmp_path):
    """Shift+Tab lets Shift go, so a later selection replaces rather than extends."""

    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"] * 3)
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 2, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
        assert QApplication.keyboardModifiers() == Qt.KeyboardModifier.NoModifier

        table.selectRow(0)
        assert _selected_rows(window) == [0]
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
        QTest.mouseClick(look_button(window, 1), Qt.MouseButton.LeftButton)
        assert _focus(window) is look_button(window, 1)
        assert table.currentRow() == 0

        _press(Qt.Key.Key_Tab)
        assert _focus(window) is dismiss_button(window, 1)

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
        window._jobs, "_on_inspect_from_dashboard", lambda ids: inspected.append(list(ids))
    )
    try:
        _focus_cell(window, 2, window._job_dashboard.LABEL_COLUMN)
        table.selectRow(0)
        table.setCurrentCell(
            2, window._job_dashboard.LABEL_COLUMN, QItemSelectionModel.SelectionFlag.NoUpdate
        )

        _press(Qt.Key.Key_Tab)
        assert _focus(window) is look_button(window, 2)
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
        assert _focus(window) is look_button(window, 5)

        _press(key)

        assert _focus(window) is look_button(window, 5)
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

        assert _focus(window) is look_button(window, 1)
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
        assert _focus(window) is look_button(window, 0)

        _press(Qt.Key.Key_Up)
        assert _focus(window) is look_button(window, 0)

        _press(Qt.Key.Key_Down)
        assert _focus(window) is look_button(window, 1)
        assert table.currentRow() == 1
        assert _selected_rows(window) == [1]

        _press(Qt.Key.Key_Tab)
        assert _focus(window) is dismiss_button(window, 1)

        _press(Qt.Key.Key_Down)
        assert _focus(window) is dismiss_button(window, 2)

        _press(Qt.Key.Key_Down)
        assert _focus(window) is dismiss_button(window, 2)

        _press(Qt.Key.Key_Up)
        assert _focus(window) is dismiss_button(window, 1)
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
        assert _focus(window) is look_button(window, 1)
        assert _selected_rows(window) == [0, 1]

        _press(Qt.Key.Key_Down)
        assert _focus(window) is look_button(window, 2)
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
        assert _focus(window) is look_button(window, 0)

        _press(Qt.Key.Key_Right)
        assert _focus(window) is dismiss_button(window, 0)

        _press(Qt.Key.Key_Right)
        assert _focus(window) is dismiss_button(window, 0)

        _press(Qt.Key.Key_Left)
        assert _focus(window) is look_button(window, 0)
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
        _put_out_of_reach(look_button(window, 0), hide)
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)

        _press(Qt.Key.Key_Tab)
        assert _focus(window) is dismiss_button(window, 0)

        # Shift+Tab goes past it too, rather than sticking in the cell.
        _press(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
        assert _focus(window) is table

        _put_out_of_reach(dismiss_button(window, 1), hide)
        table.setCurrentCell(1, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is look_button(window, 1)

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
        for control in (look_button(window, 0), dismiss_button(window, 0)):
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

        look_button(window, 0).setEnabled(False)

        assert _focus(window) is dismiss_button(window, 0)
    finally:
        window.hide()


def test_dashboard_last_control_disabled_while_focused_stays_on_the_row(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"])
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is dismiss_button(window, 0)

        dismiss_button(window, 0).setEnabled(False)

        assert _focus(window) is look_button(window, 0)
    finally:
        window.hide()


def test_dashboard_down_goes_past_a_row_with_no_control_to_take_focus(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed"] * 3)
    try:
        for control in (look_button(window, 1), dismiss_button(window, 1)):
            control.setEnabled(False)
        dismiss_button(window, 2).setEnabled(False)
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is dismiss_button(window, 0)

        _press(Qt.Key.Key_Down)

        # Row 2's own ✕ is disabled, so its other control takes focus.
        assert _focus(window) is look_button(window, 2)
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
        assert _focus(window) is look_button(window, 1)
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
        QTest.mouseClick(run_button(window, 0), Qt.MouseButton.LeftButton)
        assert _focus(window) is run_button(window, 0)

        _leave(window, jobs[0])

        assert table.rowCount() == 2
        assert _focus(window) is run_button(window, 0)
        assert job_row(window, jobs[1].id) == 0
        assert table.currentRow() == 0
        assert _visible_rings(run_button(window, 0)) == []
    finally:
        window.hide()


def test_dashboard_tabbed_menu_button_passes_focus_to_the_same_control_with_one_ring(
    main_window, tmp_path
):
    window = main_window
    apply_theme(QApplication.instance(), "light")
    jobs = _shown_dashboard(window, tmp_path, ["failed"] * 3)
    try:
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is dismiss_button(window, 1)

        _leave(window, jobs[1])

        assert job_row(window, jobs[2].id) == 1
        assert _focus(window) is dismiss_button(window, 1)
        assert len(_visible_rings(dismiss_button(window, 1))) == 1
    finally:
        window.hide()


def test_dashboard_last_row_passes_focus_up_when_it_leaves(main_window, tmp_path):
    window = main_window
    jobs = _shown_dashboard(window, tmp_path, ["running"] * 3)
    try:
        _focus_cell(window, 2, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)

        _leave(window, jobs[2])

        assert _focus(window) is run_button(window, 1)
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
        assert _focus(window) is run_button(window, 1)
    finally:
        window.hide()


def test_dashboard_focus_skips_a_neighbour_with_no_control_to_take_it(main_window, tmp_path):
    window = main_window
    jobs = _shown_dashboard(window, tmp_path, ["failed"] * 3)
    try:
        for control in (look_button(window, 2), dismiss_button(window, 2)):
            control.setEnabled(False)
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)

        _leave(window, jobs[1])

        assert _focus(window) is look_button(window, 0)
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

        assert _focus(window) is run_button(window, 0)
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
        assert window.focusWidget() is run_button(window, 0)

        _leave(window, jobs[0])

        assert window.focusWidget() is run_button(window, 0)
        assert job_row(window, jobs[1].id) == 0

        other.close()
        window.activateWindow()
        QTest.qWaitForWindowActive(window)
        assert _focus(window) is run_button(window, 0)
    finally:
        window.hide()


def test_dashboard_focus_survives_the_project_closing(main_window, tmp_path):
    window = main_window
    _shown_dashboard(window, tmp_path, ["running"] * 2)
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is run_button(window, 0)

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
def test_dashboard_keyboard_presses_every_button_on_the_row(
    main_window, tmp_path, monkeypatch, key
):
    window = main_window
    give_jobs_rows(window, monkeypatch)
    (job,) = _shown_dashboard(window, tmp_path, ["failed"])
    calls: list[tuple[str, list[object]]] = []
    for kind, handler in (
        ("resubmit", "_on_resubmit_from_dashboard"),
        ("inspect", "_on_inspect_from_dashboard"),
        ("dismiss", "_on_dismiss_from_dashboard"),
    ):
        monkeypatch.setattr(window._jobs, handler, lambda ids, k=kind: calls.append((k, list(ids))))
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: True)
    try:
        _focus_cell(window, 0, window._job_dashboard.CORES_COLUMN)

        for slot in range(3):
            _press(Qt.Key.Key_Tab)
            assert _focus(window) is slot_button(window, 0, slot)
            _press(key)

        assert calls == [
            ("resubmit", [job.id]),
            ("inspect", [job.id]),
            ("dismiss", [job.id]),
        ]
    finally:
        window.hide()


def test_dashboard_tab_goes_through_the_three_slots_in_order(main_window, tmp_path, monkeypatch):
    window = main_window
    give_jobs_rows(window, monkeypatch)
    _shown_dashboard(window, tmp_path, ["failed", "queued"])
    table = window._job_dashboard.table
    try:
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        stops = []
        for _stop in range(3):
            _press(Qt.Key.Key_Tab)
            stops.append(_focus(window).accessibleName())
        assert stops == ["Resubmit j0", "Inspect j0", "Dismiss j0"]
        _press(Qt.Key.Key_Tab)
        assert not table.isAncestorOf(_focus(window))

        # A row with one button has one stop; its blank slots are skipped.
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is run_button(window, 1)
        _press(Qt.Key.Key_Tab)
        assert not table.isAncestorOf(_focus(window))
    finally:
        window.hide()


@pytest.mark.parametrize(
    ("key", "modifier"),
    [
        (Qt.Key.Key_Menu, Qt.KeyboardModifier.NoModifier),
        (Qt.Key.Key_F10, Qt.KeyboardModifier.ShiftModifier),
    ],
    ids=["menu-key", "shift-f10"],
)
def test_dashboard_menu_key_opens_the_context_menu_on_the_current_row(
    main_window, tmp_path, monkeypatch, key, modifier
):
    window = main_window
    _shown_dashboard(window, tmp_path, ["failed", "running"])
    FakeMenu.instances = []
    monkeypatch.setattr(job_dashboard_module, "QMenu", FakeMenu)
    try:
        # From the table: the current row.
        _focus_cell(window, 1, window._job_dashboard.LABEL_COLUMN)
        _press(key, modifier)
        (menu,) = FakeMenu.instances
        assert menu.action("Cancel").enabled is True
        assert _selected_rows(window) == [1]

        # From a row's button: that button's row, even outside the selection.
        _focus_cell(window, 0, window._job_dashboard.LABEL_COLUMN)
        _press(Qt.Key.Key_Tab)
        assert _focus(window) is look_button(window, 0)
        table = window._job_dashboard.table
        table.selectRow(1)
        table.setCurrentCell(
            1, window._job_dashboard.LABEL_COLUMN, QItemSelectionModel.SelectionFlag.NoUpdate
        )
        look_button(window, 0).setFocus(Qt.FocusReason.ShortcutFocusReason)
        _press(key, modifier)
        menu = FakeMenu.instances[-1]
        assert menu.action("Inspect…").enabled is True
        assert menu.action("Cancel").enabled is False
        assert _selected_rows(window) == [0]
    finally:
        window.hide()


def test_dashboard_double_click_opens_the_jobs_stage(main_window, tmp_path, monkeypatch):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "double_click.mufinder", "DoubleClick")
    job = add_dashboard_job(service, doc, "j0", "failed")
    window._document = doc
    window._refresh_jobs_table()
    opened: list[list[object]] = []
    monkeypatch.setattr(
        window, "_on_open_in_stage_from_dashboard", lambda ids: opened.append(list(ids))
    )

    table = window._job_dashboard.table
    table.doubleClicked.emit(table.model().index(0, window._job_dashboard.STATUS_COLUMN))

    assert opened == [[job.id]]
    window.close()


def _visible_rings(control) -> list[QWidget]:
    return [child for child in control.findChildren(QWidget, _FOCUS_RING_NAME) if child.isVisible()]


def test_dashboard_focus_ring_shows_once_on_both_controls(main_window, tmp_path):
    window = main_window
    apply_theme(QApplication.instance(), "light")
    _shown_dashboard(window, tmp_path, ["failed"])
    try:
        _focus_cell(window, 0, window._job_dashboard.CORES_COLUMN)
        for button_at in (look_button, dismiss_button):
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
        QTest.mouseClick(look_button(window, 0), Qt.MouseButton.LeftButton)

        _press(Qt.Key.Key_Down)

        assert _focus(window) is look_button(window, 1)
        assert len(_visible_rings(look_button(window, 1))) == 1
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

    monkeypatch.setattr(window._jobs, "cancel_jobs", fake_cancel)

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
    # Delete never runs a job again or deletes its records.
    assert all(job.id in doc.jobs for job in (running, queued, failed, stopped, cancelled))
    window.close()


def test_dashboard_delete_keys_are_bound_to_both_keys(main_window) -> None:
    window = main_window
    assert window._job_dashboard.delete_shortcut.key() == QKeySequence(
        QKeySequence.StandardKey.Delete
    )
    assert window._job_dashboard.backspace_shortcut.key() == QKeySequence(Qt.Key.Key_Backspace)
    # Bound on the dock, so they also work while its header strip's controls have focus.
    assert window._job_dashboard.delete_shortcut.parent() is window._jobs_dock
    assert window._job_dashboard.backspace_shortcut.parent() is window._jobs_dock


def test_dashboard_confirmations_centre_on_the_main_window(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
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
    monkeypatch.setattr(window._jobs, "cancel_jobs", lambda *_a, **_k: touched.append("cancel"))

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
        "run": run_button(window, row),
        "look": look_button(window, row),
        "dismiss": dismiss_button(window, row),
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
    assert row_actions(window) == ("Cancel", "", "")

    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    window._refresh_jobs_table()
    after = _dashboard_row_widgets(window, 0)

    # Same widget objects, updated text.
    for key in before:
        assert before[key] is after[key], f"row widget {key} was recreated on status change"
    assert after["status"].text() == "failed"
    assert row_actions(window) == ("", "Inspect…", "✕")
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
    give_jobs_rows(window, monkeypatch)

    resubmitted: list[list[object]] = []
    monkeypatch.setattr(
        window._jobs, "_on_resubmit_from_dashboard", lambda ids: resubmitted.append(list(ids))
    )

    window._refresh_jobs_table()
    # Churn: remove job0 (dismiss), refresh, re-add nothing but refresh again.
    job0.dismissed_at = utc_now_for_test()
    window._refresh_jobs_table()
    job0.dismissed_at = None
    window._refresh_jobs_table()  # job0 comes back (fresh row/closure)

    # Locate each job's action button by id and click it — must dispatch that job.
    run_button(window, job_row(window, job1.id)).click()
    run_button(window, job_row(window, job0.id)).click()

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
        window._jobs, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids))
    )
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda _parent, title, body, *a: asked.append((title, body)) or SelectionScope.CANCEL,
    )
    window._refresh_jobs_table()

    cancel_btn = run_button(window, 0)
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
    monkeypatch.setattr(
        window._jobs, "_on_cancel_from_dashboard", lambda ids: acted.append(list(ids))
    )
    prompts: list[tuple[str, ...]] = []
    answers = [False, True]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or answers.pop(0),
    )
    window._refresh_jobs_table()

    button = run_button(window)
    button.selection_snapshot_rows = (0,)
    button.clicked.emit()
    assert acted == []

    button.selection_snapshot_rows = (0,)
    button.clicked.emit()
    assert acted == [[job.id]]
    assert prompts[0] == ("Cancel job", "Cancel job 'j0'?", "", "Cancel job", "Keep running")
    window.close()


def test_main_window_dashboard_dismiss_button_is_confirmed_on_its_own_row(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "single_dismiss.mufinder", "Single")
    job = add_dashboard_job(service, doc, "j0", "failed")
    window._document = doc
    acted: list[list[object]] = []
    monkeypatch.setattr(
        window._jobs, "_on_dismiss_from_dashboard", lambda ids: acted.append(list(ids))
    )
    prompts: list[tuple[str, ...]] = []
    answers = [False, True]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or answers.pop(0),
    )
    window._refresh_jobs_table()

    dismiss_button(window).selection_snapshot_rows = (0,)
    dismiss_button(window).click()
    assert acted == []

    dismiss_button(window).selection_snapshot_rows = (0,)
    dismiss_button(window).click()
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
        window._jobs, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids))
    )
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )
    window._refresh_jobs_table()

    button = run_button(window, 0)
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
    dock = window.findChild(QDockWidget, "jobDashboardDock")
    assert window._job_dashboard.table.parent() is dock.widget()
    assert dock.titleBarWidget().isAncestorOf(hint)
    assert hint.isHidden()

    _select_dashboard_rows(window, [0])
    assert hint.isHidden()

    _select_dashboard_rows(window, [0, 1])
    assert not hint.isHidden()
    assert hint.text() == "2 selected \u00b7 row actions apply to all selected"
    prop = workflow_table_utils.MULTI_ROW_TARGET_PROPERTY
    for button_at in (look_button, dismiss_button):
        assert bool(button_at(window, 0).property(prop)) is True
    assert bool(run_button(window, 2).property(prop)) is False

    _select_dashboard_rows(window, [])
    assert hint.isHidden()
    window.close()


def test_main_window_dashboard_menu_acts_on_the_eligible_selected_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, (running, failed, cancelled_job) = _dashboard_window_with_statuses(
        tmp_path, "menu", ["running", "failed", "cancelled"]
    )
    give_jobs_rows(window, monkeypatch)
    calls: list[tuple[str, list[object]]] = []
    for kind, handler in (
        ("resubmit", "_on_resubmit_from_dashboard"),
        ("inspect", "_on_inspect_failed_from_dashboard"),
    ):
        monkeypatch.setattr(window._jobs, handler, lambda ids, k=kind: calls.append((k, list(ids))))
    _select_dashboard_rows(window, [0, 1, 2])

    menu = open_context_menu(window, monkeypatch)

    # Entries that apply to only some of the three selected jobs say how many.
    assert [action.text for action in menu.actions] == [
        "Cancel (1 of 3)",
        "Inspect… (1 of 3)",
        "Check now",
        "Resubmit… (2 of 3)",
        "Dismiss (2 of 3)",
        "Open in Cutoff Energy",
    ]
    assert menu.separators_before == [5]
    assert {action.text for action in menu.actions if not action.enabled} == {"Check now"}

    menu.action("Resubmit… (2 of 3)").triggered.emit()
    menu.action("Inspect… (1 of 3)").triggered.emit()
    assert calls == [
        ("resubmit", [failed.id, cancelled_job.id]),
        ("inspect", [failed.id]),
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
        window._jobs, "_on_cancel_from_dashboard", lambda ids: calls.append(("cancel", list(ids)))
    )
    monkeypatch.setattr(
        window._jobs, "_on_dismiss_from_dashboard", lambda ids: calls.append(("dismiss", list(ids)))
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


def test_main_window_dashboard_menu_offers_resubmit_only_for_jobs_with_a_row(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, (with_row, rowless) = _dashboard_window_with_statuses(
        tmp_path, "menu_rows", ["failed", "stopped"]
    )
    monkeypatch.setattr(window._jobs, "can_resubmit", lambda job: job is with_row)
    resubmitted: list[list[object]] = []
    monkeypatch.setattr(
        window._jobs, "_on_resubmit_from_dashboard", lambda ids: resubmitted.append(list(ids))
    )
    _select_dashboard_rows(window, [0, 1])

    open_context_menu(window, monkeypatch).action("Resubmit… (1 of 2)").triggered.emit()

    assert resubmitted == [[with_row.id]]
    assert row_actions(window, 1)[0] == ""
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
    assert window._job_dashboard.action_applies("inspect", hidden) is True
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
