"""Jobs-dashboard behaviour in the main window: row buttons, menus, flushes and status cells."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QHeaderView, QTableWidgetItem, QTreeWidgetItem

import mufinder.app.main_window as main_window_module
import mufinder.app.workflow_table_utils as workflow_table_utils
from mufinder.app import selection_snapshot as selection_snapshot_module
from mufinder.app.selection_snapshot import SelectionScope
from mufinder.app.journal_widget import JournalWidget
from mufinder.app.main_window import MainWindow
from mufinder.castep_setup import CastepSetup
from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import CalculationRun, Job, ProvenanceRecord
from mufinder.services.project_service import ProjectService
from mufinder.services.ssh_client import SSHAuthenticationError, SSHCredentials
from tests.support.main_window import select_route


def test_dashboard_flush_is_coalesced_across_a_completion_burst(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "autosave_coalesce.mufinder", "AutosaveCoalesce")
    window._document = doc

    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    # A burst of lifecycle autosave requests does not write per request.
    window._request_dashboard_flush(failure_message="x")
    window._request_dashboard_flush(failure_message="x")
    window._request_dashboard_flush(failure_message="x")
    assert save_calls == []

    # The flush (called after a queue poll / by the backstop timer) writes once.
    window._flush_dashboard()
    assert save_calls == ["save"]

    # Nothing pending -> no further writes.
    window._flush_dashboard()
    assert save_calls == ["save"]
    window.close()


def test_dashboard_flush_builds_the_journal_only_for_new_records(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "journal_flush.mufinder", "JournalFlush")
    for minute in range(3):
        service.append_provenance(
            doc,
            ProvenanceRecord(
                timestamp=datetime(2026, 4, 21, 9, minute, tzinfo=UTC),
                action="generate_candidates",
                parameters={"method": "random", "count": 2},
            ),
        )
    window._document = doc
    window._refresh_ui_after_project_change()
    select_route(window, "provenance_log")

    rebuilds: list[str] = []
    cards: list[str] = []
    original_rebuild = JournalWidget._rebuild_view
    original_card = JournalWidget._build_entry_card

    def _rebuild(self: JournalWidget) -> None:
        rebuilds.append("rebuild")
        original_rebuild(self)

    def _card(self: JournalWidget, entry):
        cards.append(entry.title)
        return original_card(self, entry)

    monkeypatch.setattr(JournalWidget, "_rebuild_view", _rebuild)
    monkeypatch.setattr(JournalWidget, "_build_entry_card", _card)

    window._request_dashboard_flush(failure_message="x")
    window._flush_dashboard()

    assert rebuilds == [] and cards == []

    service.append_provenance(
        doc,
        ProvenanceRecord(
            timestamp=datetime(2026, 4, 21, 9, 30, tzinfo=UTC),
            action="generate_candidates",
            parameters={"method": "random", "count": 4},
        ),
    )
    window._request_dashboard_flush(failure_message="x")
    window._flush_dashboard()

    assert rebuilds == []
    assert cards == ["Muon sites generated"]

    # While another page is shown, a flush leaves the journal for its next show.
    select_route(window, "host_structure")
    service.append_provenance(
        doc,
        ProvenanceRecord(
            timestamp=datetime(2026, 4, 21, 9, 45, tzinfo=UTC),
            action="generate_candidates",
            parameters={"method": "random", "count": 6},
        ),
    )
    window._request_dashboard_flush(failure_message="x")
    window._flush_dashboard()

    assert len(cards) == 1

    select_route(window, "provenance_log")

    assert rebuilds == [] and len(cards) == 2


def test_dashboard_flush_trace_is_logged_after_the_event_loop_pass(
    main_window, qapp, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    window = main_window
    window._document = ProjectService().create_project(tmp_path / "trace.mufinder", "Trace")
    caplog.set_level(logging.DEBUG, logger="mufinder.app.main_window")

    window._request_dashboard_flush(failure_message="x")
    window._flush_dashboard()

    assert not [r for r in caplog.records if r.getMessage().startswith("dashboard flush")]

    qapp.processEvents()

    messages = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("dashboard flush")
    ]
    assert len(messages) == 1
    assert "event loop (deferred deletes, repaint)=" in messages[0]


def test_close_persists_pending_dashboard_flush(main_window, tmp_path: Path) -> None:
    # A coalesced flush pending at close must be persisted, not dropped — before
    # autosave coalescing these completions were already written to disk per job.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "close_flush.mufinder", "CloseFlush")
    window._document = doc

    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    window._request_dashboard_flush(failure_message="x")
    assert save_calls == []

    window.close()
    assert save_calls == ["save"]


def test_cluster_queue_checked_schedules_deferred_flush(main_window, tmp_path: Path) -> None:
    # _on_cluster_queue_checked used to call _flush_dashboard() directly,
    # bypassing the debounce/rate-limit entirely. It must now route through
    # _request_dashboard_flush like every other lifecycle handler: the queue
    # check drains any already-pending flush, but does not write synchronously.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "queue_flush.mufinder", "QueueFlush")
    window._document = doc

    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    window._request_dashboard_flush(failure_message="x")
    assert save_calls == []

    window._on_cluster_queue_checked(SimpleNamespace(checked_at=None, elapsed_by_job_id={}))
    # Not synchronous: the write is still only scheduled.
    assert save_calls == []
    assert window._dashboard_flush_pending is True

    # The metadata stamp landed in the in-memory document regardless of when
    # the flush itself runs.
    window._flush_dashboard()
    assert save_calls == ["save"]
    window.close()


def test_cluster_queue_checked_with_no_activity_does_not_schedule_flush(
    main_window, tmp_path: Path
) -> None:
    # A routine queue poll that stamps nothing and has no pending completion
    # must not start scheduling flushes on its own — only actual job activity
    # (via mark_queue_checked stamping a job, or a completion handler already
    # having marked pending) should arm the timer.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "queue_no_activity.mufinder", "QueueNoActivity")
    window._document = doc

    window._on_cluster_queue_checked(SimpleNamespace(checked_at=None, elapsed_by_job_id={}))
    assert window._dashboard_flush_pending is False
    assert window._dashboard_flush_timer.isActive() is False
    window.close()


def test_dashboard_flush_isolated_event_uses_short_debounce(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An isolated lifecycle event (nothing nearby) should still arm the short
    # debounce, not the longer min-flush-interval floor, preserving the
    # existing prompt-flush feel.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "isolated_event.mufinder", "IsolatedEvent")
    window._document = doc

    clock = [1_000.0]
    monkeypatch.setattr(main_window_module.time, "monotonic", lambda: clock[0])

    window._request_dashboard_flush(failure_message="x")
    assert window._dashboard_flush_timer.isActive() is True
    assert window._dashboard_flush_timer.interval() == window._DASHBOARD_FLUSH_DEBOUNCE_MS
    window.close()


def test_dashboard_flush_stream_is_rate_limited_to_min_interval(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A steady stream of requests spaced closer than the min flush interval
    # (but further apart than the debounce) must flush on a bounded ~2s
    # cadence during the stream, not only once the stream stops, and must not
    # have each new request push the deadline further into the future.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "flush_stream.mufinder", "FlushStream")
    window._document = doc

    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    clock = [0.0]
    monkeypatch.setattr(main_window_module.time, "monotonic", lambda: clock[0])

    # First request: isolated so far -> short debounce deadline at t=0.3s.
    window._request_dashboard_flush(failure_message="x")
    first_deadline = window._dashboard_flush_deadline
    assert first_deadline == pytest.approx(0.3)

    # Fire the debounce timer as Qt would once its interval elapses.
    clock[0] = 0.3
    window._dashboard_flush_timer.timeout.emit()
    assert save_calls == ["save"]
    assert window._dashboard_last_flush_at == pytest.approx(0.3)

    # A request 0.5s after the last flush: within the 2s floor, so the
    # deadline should land 2s after the last flush (t=2.3s), not 0.3s later.
    clock[0] = 0.8
    window._request_dashboard_flush(failure_message="x")
    assert window._dashboard_flush_deadline == pytest.approx(2.3)

    # Further requests inside the same armed window must not push the
    # deadline further out.
    clock[0] = 1.3
    window._request_dashboard_flush(failure_message="x")
    assert window._dashboard_flush_deadline == pytest.approx(2.3)

    clock[0] = 1.8
    window._request_dashboard_flush(failure_message="x")
    assert window._dashboard_flush_deadline == pytest.approx(2.3)

    # The armed deadline fires on schedule, bounding the burst to one flush
    # per ~2s rather than starving until the stream ends.
    clock[0] = 2.3
    window._dashboard_flush_timer.timeout.emit()
    assert save_calls == ["save", "save"]
    assert window._dashboard_last_flush_at == pytest.approx(2.3)

    # Trailing request after the stream ends still schedules a guaranteed
    # final flush.
    clock[0] = 2.5
    window._request_dashboard_flush(failure_message="x")
    assert window._dashboard_flush_pending is True
    window._flush_dashboard()
    assert save_calls == ["save", "save", "save"]
    window.close()


def _cluster_widgets(window) -> tuple:
    return (
        window._cluster_connection_label,
        window._cluster_queue_checked_label,
        window._cluster_refresh_button,
    )


def _shown(window) -> list[bool]:
    return [not widget.isHidden() for widget in _cluster_widgets(window)]


def _project_with_job(
    window,
    tmp_path: Path,
    *,
    target: str = "SCARF",
    status: JobStatus = JobStatus.RUNNING,
    remote_job_id: str | None = "12345",
):
    service = ProjectService()
    doc = service.create_project(tmp_path / "status_bar.mufinder", "StatusBar")
    job = service.add_job(
        doc,
        label="job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target=target,
    )
    job.status = status
    job.remote_job_id = remote_job_id
    window._document = doc
    window._refresh_jobs_table()
    return job


def _connect_status_bar(window, monkeypatch: pytest.MonkeyPatch, name: str | None) -> None:
    monkeypatch.setattr(
        type(window._cluster_runner), "connected_cluster_name", property(lambda _self: name)
    )
    monkeypatch.setattr(
        type(window._cluster_runner), "is_connected", property(lambda _self: name is not None)
    )
    window._on_cluster_connection_status_changed(name is not None)


def test_status_bar_hides_cluster_widgets_with_no_project(main_window) -> None:
    assert main_window._document is None
    assert _shown(main_window) == [False, False, False]
    assert main_window._cluster_refresh_button.isEnabled() is False
    assert "nothing to poll" in main_window._cluster_refresh_button.toolTip()


def test_status_bar_hides_cluster_widgets_for_a_local_only_project(
    main_window, tmp_path: Path
) -> None:
    _project_with_job(main_window, tmp_path, target="local")
    assert _shown(main_window) == [False, False, False]


def test_status_bar_shows_cluster_widgets_for_an_active_cluster_job(
    main_window, tmp_path: Path
) -> None:
    _project_with_job(main_window, tmp_path)

    assert _shown(main_window) == [True, True, True]
    assert main_window._cluster_connection_label.text() == "Not connected to SCARF"
    assert main_window._cluster_queue_checked_label.text() == "Queue last checked: never"
    assert main_window._cluster_refresh_button.isEnabled() is True


def test_status_bar_hides_cluster_widgets_when_the_last_cluster_job_ends(
    main_window, tmp_path: Path
) -> None:
    job = _project_with_job(main_window, tmp_path)
    job.status = JobStatus.COMPLETED
    main_window._refresh_jobs_table()
    assert _shown(main_window) == [False, False, False]


def test_status_bar_connection_colour_follows_the_state(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    palette = main_window_module.current_palette()
    label = window._cluster_connection_label
    monkeypatch.setattr(main_window_module, "show_detailed_warning", lambda *_a, **_k: None)
    _project_with_job(window, tmp_path)

    # Idle: reopened project, nothing has dropped.
    assert label.text() == "Not connected to SCARF"
    assert palette.text_muted in label.styleSheet()

    _connect_status_bar(window, monkeypatch, "SCARF")
    assert label.text() == "Connected to SCARF"
    assert palette.success in label.styleSheet()

    # A deliberate disconnect is not a drop.
    _connect_status_bar(window, monkeypatch, None)
    assert label.text() == "Not connected to SCARF"
    assert palette.text_muted in label.styleSheet()

    # The runner signals the disconnect, then the failed poll.
    _connect_status_bar(window, monkeypatch, "SCARF")
    _connect_status_bar(window, monkeypatch, None)
    window._on_cluster_poll_failed("Queue polling failed for SCARF: timeout")
    assert label.text() == "Connection to SCARF lost"
    assert palette.danger in label.styleSheet()

    # Reconnecting clears the drop.
    _connect_status_bar(window, monkeypatch, "SCARF")
    assert label.text() == "Connected to SCARF"
    _connect_status_bar(window, monkeypatch, None)
    assert palette.text_muted in label.styleSheet()


def test_status_bar_opening_a_project_forgets_the_previous_drop_and_check(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    monkeypatch.setattr(main_window_module, "show_detailed_warning", lambda *_a, **_k: None)
    _project_with_job(window, tmp_path)
    _connect_status_bar(window, monkeypatch, "SCARF")
    window._on_cluster_queue_checked("12:34:56")
    _connect_status_bar(window, monkeypatch, None)
    window._on_cluster_poll_failed("Queue polling failed for SCARF: timeout")
    assert window._cluster_queue_checked_label.text() == "Queue last checked: 12:34:56"

    window._reset_job_tracking_for_project_change()
    window._refresh_jobs_table()

    assert window._cluster_connection_label.text() == "Not connected to SCARF"
    assert window._cluster_queue_checked_label.text() == "Queue last checked: never"


def test_status_bar_refresh_disabled_with_a_tooltip_when_no_job_can_be_polled(
    main_window, tmp_path: Path
) -> None:
    job = _project_with_job(main_window, tmp_path, status=JobStatus.SUBMITTING, remote_job_id=None)
    button = main_window._cluster_refresh_button

    assert _shown(main_window) == [True, True, True]
    assert button.isEnabled() is False
    assert "no active cluster job has a remote job ID" in button.toolTip()

    job.remote_job_id = "12345"
    main_window._refresh_jobs_table()
    assert button.isEnabled() is True
    assert "Check the cluster queue now" in button.toolTip()


def test_status_bar_reports_the_result_of_a_queue_refresh(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    job = _project_with_job(window, tmp_path)
    _connect_status_bar(window, monkeypatch, "SCARF")
    calls: list[str] = []
    monkeypatch.setattr(window._cluster_runner, "refresh_queue", lambda: calls.append("refresh"))

    # A periodic poll says nothing in the status bar.
    window.statusBar().showMessage("Ready")
    snapshot = SimpleNamespace(
        checked_at="12:34:56",
        elapsed_by_job_id={},
        observations_by_job_id={job.id: "scheduler reports RUNNING"},
    )
    window._on_cluster_queue_checked(snapshot)
    assert window.statusBar().currentMessage() == "Ready"

    window._cluster_refresh_button.click()
    assert calls == ["refresh"]
    window._on_cluster_queue_checked(snapshot)
    assert window.statusBar().currentMessage() == "Refreshed queue: checked 1 job at 12:34:56"

    window.statusBar().showMessage("Ready")
    window._on_cluster_queue_checked(snapshot)
    assert window.statusBar().currentMessage() == "Ready"


def test_status_bar_refresh_reconnects_when_connected_to_another_cluster(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _project_with_job(window, tmp_path)
    _connect_status_bar(window, monkeypatch, "PSI")
    assert window._cluster_connection_label.text() == "Not connected to SCARF"

    scarf = ClusterConfig(
        name="SCARF",
        hostname="ui1.scarf.rl.ac.uk",
        username="alice",
        auth_method="key",
        key_path="",
        default_working_directory="/work4/isis/alice",
        queue_system="SLURM",
        cores_per_node=24,
        default_walltime="12:00:00",
        template_script_path="builtin:scarf-slurm",
    )
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [scarf])
    connected: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda config, password=None: connected.append(config.name),
    )
    polls: list[str] = []
    monkeypatch.setattr(window._cluster_runner, "refresh_queue", lambda: polls.append("poll"))

    window._cluster_refresh_button.click()

    assert connected == ["SCARF"]
    assert polls == ["poll"]


def test_status_bar_reports_a_failed_queue_refresh(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    monkeypatch.setattr(main_window_module, "show_detailed_warning", lambda *_a, **_k: None)
    _project_with_job(window, tmp_path)
    _connect_status_bar(window, monkeypatch, "SCARF")
    monkeypatch.setattr(window._cluster_runner, "refresh_queue", lambda: None)

    window._cluster_refresh_button.click()
    _connect_status_bar(window, monkeypatch, None)
    window._on_cluster_poll_failed("Queue polling failed for SCARF: timeout")

    assert window.statusBar().currentMessage() == "Failed to check queue on SCARF"


def test_main_window_cluster_queue_checked_formats_timezone_aware_times_locally(
    main_window,
) -> None:
    window = main_window

    class _LocalTime:
        def strftime(self, _format: str) -> str:
            return "12:34:56"

    class _AwareTime:
        tzinfo = object()

        def astimezone(self):
            return _LocalTime()

        def strftime(self, _format: str) -> str:
            return "11:34:56"

    window._on_cluster_queue_checked(_AwareTime())

    assert "12:34:56" in window._cluster_queue_checked_label.text()
    window.close()


def test_main_window_cluster_queue_checked_uses_snapshot_timestamp(main_window) -> None:
    window = main_window

    snapshot = SimpleNamespace(
        checked_at=datetime(2026, 4, 16, 12, 34, 56, tzinfo=UTC),
        elapsed_by_job_id={},
    )

    window._on_cluster_queue_checked(snapshot)

    expected = snapshot.checked_at.astimezone().strftime("%H:%M:%S")
    assert window._cluster_queue_checked_label.text() == f"Queue last checked: {expected}"
    window.close()


def test_main_window_cluster_queue_checked_unknown_object_stays_compact(main_window) -> None:
    window = main_window

    window._on_cluster_queue_checked(SimpleNamespace(checked_at=object(), elapsed_by_job_id={}))

    assert window._cluster_queue_checked_label.text() == "Queue last checked: unknown"
    window.close()


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
    job.status = JobStatus.RUNNING
    job.started_at = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(
        main_window_module,
        "utc_now",
        lambda: datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC),
    )
    window._document = doc
    window._refresh_jobs_table()

    assert window._job_runtime_timer.isActive() is True
    assert window._jobs_table.item(0, window._JOB_RUNTIME_COLUMN).text() == "00:00"

    monkeypatch.setattr(
        main_window_module,
        "utc_now",
        lambda: datetime(2026, 4, 20, 16, 1, 5, tzinfo=UTC),
    )
    window._refresh_active_job_runtime_cells()

    assert window._jobs_table.item(0, window._JOB_RUNTIME_COLUMN).text() == "01:05"

    job.status = JobStatus.COMPLETED
    job.completed_at = datetime(2026, 4, 20, 16, 1, 5, tzinfo=UTC)
    window._refresh_active_job_runtime_cells()

    assert window._job_runtime_timer.isActive() is False
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
    job.status = JobStatus.RUNNING
    job.execution_target = "redwood"

    window._document = doc
    window._refresh_jobs_table()

    snapshot_time = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(main_window_module, "utc_now", lambda: snapshot_time)
    window._on_cluster_queue_checked(
        SimpleNamespace(
            checked_at=snapshot_time,
            elapsed_by_job_id={job.id: "00:12:34"},
        )
    )

    assert window._jobs_table.item(0, window._JOB_RUNTIME_COLUMN).text() == "12:34"
    assert window._job_runtime_timer.isActive() is True

    monkeypatch.setattr(
        main_window_module,
        "utc_now",
        lambda: datetime(2026, 4, 20, 16, 1, 4, tzinfo=UTC),
    )
    window._refresh_active_job_runtime_cells()

    assert window._jobs_table.item(0, window._JOB_RUNTIME_COLUMN).text() == "13:38"

    # A poll that no longer reports the job clears its scheduler runtime.
    window._on_cluster_queue_checked(
        SimpleNamespace(checked_at=snapshot_time, elapsed_by_job_id={})
    )
    assert window._jobs_table.item(0, window._JOB_RUNTIME_COLUMN).text() == ""
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
    job.status = JobStatus.RUNNING
    job.started_at = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(
        main_window_module, "utc_now", lambda: datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    )

    window._document = doc
    window._refresh_jobs_table()
    assert window._job_runtime_timer.isActive() is True

    # Dock becoming not-visible (closed, obscured, or window minimized — all of
    # which emit visibilityChanged(False)) should pause the per-second updates.
    window._on_jobs_dock_visibility_changed(False)
    assert window._jobs_dock_visible is False
    assert window._job_runtime_timer.isActive() is False

    # A live-runtime resync while not visible must not re-arm the timer.
    window._sync_job_runtime_timer()
    assert window._job_runtime_timer.isActive() is False

    # Becoming visible again re-arms it and refreshes the (now stale) cells.
    window._on_jobs_dock_visibility_changed(True)
    assert window._jobs_dock_visible is True
    assert window._job_runtime_timer.isActive() is True

    window.close()


def test_main_window_tab_navigation_does_not_recompute_stage_icons(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "nav_icons.mufinder", "NavIcons")
    window._document = doc
    window._refresh_ui_after_project_change()

    # Switching tabs cannot change workflow completion, so navigation must not
    # re-run the (job/result-scanning) completion checks.
    calls = 0
    original = window._refresh_workflow_stage_icons

    def _counting() -> None:
        nonlocal calls
        calls += 1
        original()

    monkeypatch.setattr(window, "_refresh_workflow_stage_icons", _counting)

    item = QTreeWidgetItem(["Candidates"])
    item.setData(0, 0x0100, "muon_candidates")  # Qt.UserRole
    window._project_tree.clearSelection()
    window._project_tree.addTopLevelItem(item)
    window._project_tree.setCurrentItem(item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._muon_generation_view
    assert calls == 0

    window.close()


def test_main_window_disconnected_refresh_attempts_active_cluster_targets(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_refresh_reconnect.mufinder", "ClusterRefresh")

    redwood_job = service.add_job(
        doc,
        label="redwood_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="Redwood",
    )
    redwood_job.status = JobStatus.RUNNING

    scarf_job = service.add_job(
        doc,
        label="scarf_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="SCARF",
    )
    scarf_job.status = JobStatus.QUEUED

    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="alice",
        auth_method="key",
        key_path="",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="ADDQUEUE",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
    )
    scarf = ClusterConfig(
        name="SCARF",
        hostname="ui1.scarf.rl.ac.uk",
        username="alice",
        auth_method="key",
        key_path="",
        default_working_directory="/work4/isis/alice",
        queue_system="SLURM",
        cores_per_node=24,
        default_walltime="12:00:00",
        template_script_path="builtin:scarf-slurm",
    )

    monkeypatch.setattr(
        main_window_module, "load_cluster_configs", lambda _settings: [redwood, scarf]
    )

    connect_calls: list[str] = []
    refresh_calls: list[str] = []

    def _connect(config, password=None):
        _ = password
        connect_calls.append(config.name)

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(
        window._cluster_runner, "refresh_queue", lambda: refresh_calls.append("refresh")
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._on_refresh_cluster_queue()

    assert connect_calls == ["Redwood", "SCARF"]
    assert refresh_calls == ["refresh", "refresh"]
    assert warnings == []
    window.close()


def test_main_window_disconnected_refresh_prompts_password_for_password_cluster(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "cluster_refresh_password_prompt.mufinder", "ClusterRefreshPwd"
    )

    redwood_job = service.add_job(
        doc,
        label="redwood_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="Redwood",
    )
    redwood_job.status = JobStatus.QUEUED
    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="alice",
        auth_method="password",
        key_path="",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="ADDQUEUE",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
    )
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])

    connect_calls: list[tuple[str, str | None]] = []

    def _connect(config, password=None):
        connect_calls.append((config.name, password))
        if password is None:
            raise RuntimeError("Password is required for password authentication.")

    refresh_calls: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(
        window._cluster_runner, "refresh_queue", lambda: refresh_calls.append("refresh")
    )
    monkeypatch.setattr(
        main_window_module,
        "prompt_for_cluster_credentials",
        lambda *_args, **_kwargs: SSHCredentials(target_secret="secret"),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._on_refresh_cluster_queue()

    assert connect_calls == [("Redwood", None), ("Redwood", "secret")]
    assert refresh_calls == ["refresh"]
    assert warnings == []
    window.close()


def test_main_window_disconnected_refresh_prompts_passphrase_for_key_cluster(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "cluster_refresh_key_prompt.mufinder", "ClusterRefreshKey"
    )

    redwood_job = service.add_job(
        doc,
        label="redwood_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="Redwood",
    )
    redwood_job.status = JobStatus.QUEUED
    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="alice",
        auth_method="key",
        key_path="/Users/alice/.ssh/id_ed25519",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="ADDQUEUE",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
    )
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])

    connect_calls: list[tuple[str, str | None]] = []

    def _connect(config, password=None):
        connect_calls.append((config.name, password))
        if password is None:
            raise SSHAuthenticationError("SSH key authentication failed: private key is encrypted")

    refresh_calls: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(
        window._cluster_runner, "refresh_queue", lambda: refresh_calls.append("refresh")
    )
    monkeypatch.setattr(
        main_window_module,
        "prompt_for_cluster_credentials",
        lambda *_args, **_kwargs: SSHCredentials(target_secret="secret-passphrase"),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._on_refresh_cluster_queue()

    assert connect_calls == [("Redwood", None), ("Redwood", "secret-passphrase")]
    assert refresh_calls == ["refresh"]
    assert warnings == []
    window.close()


def test_main_window_cluster_poll_failed_shows_detailed_warning(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module,
        "show_detailed_warning",
        lambda _parent, title, _message, **_kwargs: warnings.append(title),
    )

    window._on_cluster_poll_failed("Queue polling failed for alpha: timeout")

    assert warnings == ["Queue polling failed"]
    window.close()


def test_main_window_cancel_jobs_only_active_states(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cancel_local.mufinder", "CancelLocal")
    active_job = service.add_job(
        doc,
        label="job_running",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    inactive_job = service.add_job(
        doc,
        label="job_defined",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    active_job.status = JobStatus.RUNNING
    inactive_job.status = JobStatus.DEFINED
    window._document = doc

    cancelled_ids: list[object] = []
    monkeypatch.setattr(window._local_runner, "cancel", lambda job_id: cancelled_ids.append(job_id))

    window._cancel_jobs([active_job, inactive_job])

    assert cancelled_ids == [active_job.id]
    window.close()


def test_main_window_run_lookup_prefers_active_mapping(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "run_lookup.mufinder", "RunLookup")
    job = service.add_job(
        doc,
        label="job_a",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    run_1 = CalculationRun(job_id=job.id, run_index=1)
    run_2 = CalculationRun(job_id=job.id, run_index=2)
    doc.runs[run_1.id] = run_1
    doc.runs[run_2.id] = run_2
    window._document = doc

    window._active_run_by_job[job.id] = run_1.id
    assert window._run_for_job(job.id) is run_1

    window._active_run_by_job.clear()
    assert window._run_for_job(job.id) is run_2
    window.close()


def test_main_window_dashboard_uses_row_buttons_and_no_context_menu(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "jobs_buttons.mufinder", "JobsButtons")
    service.add_job(
        doc,
        label="job_defined",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    running_job = service.add_job(
        doc,
        label="job_running",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    failed_job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    completed_job = service.add_job(
        doc,
        label="job_completed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    cancelled_job = service.add_job(
        doc,
        label="job_cancelled",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    running_job.status = JobStatus.RUNNING
    failed_job.status = JobStatus.FAILED
    completed_job.status = JobStatus.COMPLETED
    cancelled_job.status = JobStatus.CANCELLED
    window._document = doc
    resubmitted: list[list[object]] = []
    cancelled: list[list[object]] = []
    inspected: list[list[object]] = []
    reset: list[list[object]] = []
    monkeypatch.setattr(
        window, "_on_resubmit_from_dashboard", lambda ids: resubmitted.append(list(ids))
    )
    monkeypatch.setattr(
        window, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids))
    )
    monkeypatch.setattr(
        window, "_on_inspect_failed_from_dashboard", lambda ids: inspected.append(list(ids))
    )
    monkeypatch.setattr(
        window, "_on_reset_failed_from_dashboard", lambda ids: reset.append(list(ids))
    )

    window._refresh_jobs_table()

    assert (
        window._jobs_table.contextMenuPolicy()
        == main_window_module.Qt.ContextMenuPolicy.CustomContextMenu
    )
    assert window._jobs_table.rowCount() == 3

    running_resubmit_button = window._jobs_table.cellWidget(0, window._JOB_SUBMIT_COLUMN)
    failed_resubmit_button = window._jobs_table.cellWidget(1, window._JOB_SUBMIT_COLUMN)
    cancelled_resubmit_button = window._jobs_table.cellWidget(2, window._JOB_SUBMIT_COLUMN)
    cancel_button = window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN)
    inspect_button = window._jobs_table.cellWidget(1, window._JOB_INSPECT_COLUMN)
    reset_button = window._jobs_table.cellWidget(1, window._JOB_RESET_COLUMN)
    cancelled_reset_button = window._jobs_table.cellWidget(2, window._JOB_RESET_COLUMN)

    assert isinstance(running_resubmit_button, main_window_module.QPushButton)
    assert isinstance(failed_resubmit_button, main_window_module.QPushButton)
    assert isinstance(cancelled_resubmit_button, main_window_module.QPushButton)
    assert isinstance(cancel_button, main_window_module.QPushButton)
    assert isinstance(inspect_button, main_window_module.QPushButton)
    assert isinstance(reset_button, main_window_module.QPushButton)
    assert isinstance(cancelled_reset_button, main_window_module.QPushButton)

    assert running_resubmit_button.text() == "Resubmit"
    assert failed_resubmit_button.text() == "Resubmit"
    assert cancelled_resubmit_button.text() == "Resubmit"
    assert running_resubmit_button.isEnabled() is False
    assert failed_resubmit_button.isEnabled() is True
    assert cancelled_resubmit_button.isEnabled() is True
    assert window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN).isEnabled() is True
    assert cancel_button.isEnabled() is True
    assert inspect_button.isEnabled() is True
    assert reset_button.isEnabled() is True
    assert cancelled_reset_button.isEnabled() is True

    failed_resubmit_button.click()
    cancelled_resubmit_button.click()
    cancel_button.click()
    inspect_button.click()
    reset_button.click()

    assert resubmitted == [[failed_job.id], [cancelled_job.id]]
    assert cancelled == [[running_job.id]]
    assert inspected == [[failed_job.id]]
    assert reset == [[failed_job.id]]
    window.close()


def test_main_window_dashboard_cancel_applies_to_snapshot_rows(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the clicked row is in a multi-row snapshot, prompt and apply to all selected rows."""
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "snap_cancel.mufinder", "SnapCancel")
    job0 = service.add_job(
        doc,
        label="j0",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job1 = service.add_job(
        doc,
        label="j1",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job0.status = JobStatus.RUNNING
    job1.status = JobStatus.RUNNING
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

    cancel_btn = window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN)
    assert isinstance(cancel_btn, SelectionSnapshotPushButton)
    cancel_btn.selection_snapshot_rows = (0, 1)
    cancel_btn.clicked.emit()

    assert len(cancelled) == 1
    assert set(cancelled[0]) == {job0.id, job1.id}
    window.close()


def test_main_window_dashboard_cancel_single_row_no_prompt(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "snap_cancel_single.mufinder", "SnapCancelSingle")
    job0 = service.add_job(
        doc,
        label="j0",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job0.status = JobStatus.RUNNING
    window._document = doc
    cancelled: list[list[object]] = []
    monkeypatch.setattr(
        window, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids))
    )
    prompted: list[bool] = []
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: prompted.append(True) or SelectionScope.ALL_SELECTED,
    )
    window._refresh_jobs_table()

    cancel_btn = window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN)
    assert isinstance(cancel_btn, SelectionSnapshotPushButton)
    cancel_btn.selection_snapshot_rows = (0,)
    cancel_btn.clicked.emit()

    assert cancelled == [[job0.id]]
    assert prompted == []
    window.close()


def test_main_window_dashboard_cancel_unselected_row_no_prompt(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "snap_cancel_unsel.mufinder", "SnapCancelUnsel")
    job0 = service.add_job(
        doc,
        label="j0",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job1 = service.add_job(
        doc,
        label="j1",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job0.status = JobStatus.RUNNING
    job1.status = JobStatus.RUNNING
    window._document = doc
    cancelled: list[list[object]] = []
    monkeypatch.setattr(
        window, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids))
    )
    prompted: list[bool] = []
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: prompted.append(True) or SelectionScope.ALL_SELECTED,
    )
    window._refresh_jobs_table()

    # Snapshot only contains row 0; click Cancel on row 1 (not in snapshot)
    cancel_btn = window._jobs_table.cellWidget(1, window._JOB_CANCEL_COLUMN)
    assert isinstance(cancel_btn, SelectionSnapshotPushButton)
    cancel_btn.selection_snapshot_rows = (0,)
    cancel_btn.clicked.emit()

    assert cancelled == [[job1.id]]
    assert prompted == []
    window.close()


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
    job.status = JobStatus.FAILED
    window._document = doc
    window._refresh_jobs_table()

    status_item = window._jobs_table.item(0, window._JOB_STATUS_COLUMN)
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
    job.status = JobStatus.RUNNING
    job.num_processes = 8
    window._document = doc

    window._refresh_jobs_table()

    processes_item = window._jobs_table.item(0, window._JOB_PROCESSES_COLUMN)
    assert processes_item is not None
    assert processes_item.text() == "8"
    window.close()


def test_main_window_dashboard_handlers_no_document_are_noops(main_window) -> None:
    window = main_window

    window._on_submit_local_from_dashboard([uuid4()])
    window._on_resubmit_from_dashboard([uuid4()])
    window._on_cancel_from_dashboard([uuid4()])
    window._on_inspect_failed_from_dashboard([uuid4()])
    window._on_reset_failed_from_dashboard([uuid4()])

    window.close()


def test_main_window_dashboard_handlers_forward_selected_jobs(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dashboard_forward.mufinder", "DashboardForward")
    job = service.add_job(
        doc,
        label="job_dash",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    submitted: list[list[object]] = []
    resubmitted: list[tuple[list[object], dict[str, object]]] = []
    cancelled: list[list[object]] = []
    inspected: list[list[object]] = []
    reset: list[list[object]] = []
    monkeypatch.setattr(window, "_submit_jobs_locally", lambda jobs: submitted.append(jobs))
    monkeypatch.setattr(window, "_cancel_jobs", lambda jobs: cancelled.append(jobs))
    monkeypatch.setattr(window, "_inspect_failed_jobs", lambda jobs: inspected.append(jobs))
    monkeypatch.setattr(window, "_reset_failed_jobs", lambda jobs: reset.append(jobs))

    window._on_submit_local_from_dashboard([job.id])
    window._on_cancel_from_dashboard([job.id])
    window._on_inspect_failed_from_dashboard([job.id])
    window._on_reset_failed_from_dashboard([job.id])

    assert submitted == [[job]]
    assert cancelled == [[job]]
    assert inspected == [[job]]
    assert reset == [[job]]

    job.status = JobStatus.CANCELLED
    monkeypatch.setattr(
        window,
        "_submit_jobs_locally",
        lambda jobs, **kwargs: resubmitted.append((jobs, kwargs)),
    )
    # Resubmit checks for a CASTEP executable before it asks anything.
    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/opt/castep", mpirun_path=""),
    )
    window._on_resubmit_from_dashboard([job.id])

    assert resubmitted
    assert resubmitted[0][0] == [job]
    assert resubmitted[0][1]["eligible_statuses"] == {
        JobStatus.FAILED,
        JobStatus.STOPPED,
        JobStatus.CANCELLED,
    }
    assert "cancelled" in resubmitted[0][1]["no_eligible_message"]
    window.close()


def _dashboard_with_jobs(window, service, doc, statuses):
    jobs = []
    for index, status in enumerate(statuses):
        job = service.add_job(
            doc,
            label=f"job_{index}_{status.value}",
            workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        job.status = status
        jobs.append(job)
    window._document = doc
    window._jobs_table.setRowCount(len(jobs))
    selection_model = window._jobs_table.selectionModel()
    for row, job in enumerate(jobs):
        label_item = main_window_module.QTableWidgetItem(job.label)
        label_item.setData(main_window_module.Qt.ItemDataRole.UserRole, str(job.id))
        window._jobs_table.setItem(row, window._JOB_LABEL_COLUMN, label_item)
    for row in range(len(jobs)):
        selection_model.select(
            window._jobs_table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
    return jobs


def test_main_window_dashboard_delete_key_summarises_then_applies_each_action(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dashboard_delete_key.mufinder", "DashboardDeleteKey")
    running_job, failed_job, completed_job = _dashboard_with_jobs(
        window, service, doc, [JobStatus.RUNNING, JobStatus.FAILED, JobStatus.COMPLETED]
    )

    events: list[object] = []
    monkeypatch.setattr(
        window,
        "_prompt_dashboard_bulk_action",
        lambda **counts: events.append(("prompt", counts)) or True,
    )
    monkeypatch.setattr(window, "_cancel_jobs", lambda jobs: events.append(("cancel", jobs)))
    monkeypatch.setattr(
        window,
        "_reset_failed_jobs",
        lambda jobs, cleanup_choice=None: events.append(("reset", jobs, cleanup_choice)),
    )
    monkeypatch.setattr(
        window,
        "_delete_dashboard_jobs",
        lambda jobs, confirmed=False: events.append(("delete", jobs, confirmed)),
    )

    window._on_dashboard_delete_key_requested()

    # One prompt, before anything is touched, then each action already confirmed.
    assert events == [
        (
            "prompt",
            {"cancel_count": 1, "reset_count": 1, "delete_count": 1, "skipped_imported_count": 0},
        ),
        ("cancel", [running_job]),
        ("reset", [failed_job], True),
        ("delete", [completed_job], True),
    ]
    window.close()


def test_main_window_dashboard_delete_key_cancelled_summary_touches_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dashboard_delete_none.mufinder", "DashboardDeleteNone")
    jobs = _dashboard_with_jobs(
        window, service, doc, [JobStatus.RUNNING, JobStatus.FAILED, JobStatus.COMPLETED]
    )

    touched: list[str] = []
    monkeypatch.setattr(window, "_prompt_dashboard_bulk_action", lambda **_counts: None)
    monkeypatch.setattr(window, "_cancel_jobs", lambda *_a, **_k: touched.append("cancel"))
    monkeypatch.setattr(window, "_reset_failed_jobs", lambda *_a, **_k: touched.append("reset"))
    monkeypatch.setattr(
        window, "_delete_dashboard_jobs", lambda *_a, **_k: touched.append("delete")
    )

    window._on_dashboard_delete_key_requested()

    assert touched == []
    assert all(job.id in doc.jobs for job in jobs)
    window.close()


@pytest.mark.parametrize(
    ("counts", "expected_lines", "has_cleanup_box"),
    [
        (
            {"cancel_count": 2, "reset_count": 1, "delete_count": 3},
            [
                "\u2022 Cancel 2 running jobs",
                "\u2022 Reset 1 failed job to ready",
                "\u2022 Delete 3 finished job records and linked results",
            ],
            True,
        ),
        (
            {"cancel_count": 0, "reset_count": 0, "delete_count": 1},
            ["\u2022 Delete 1 finished job record and linked results"],
            False,
        ),
    ],
)
def test_main_window_dashboard_bulk_prompt_lists_only_what_applies(
    main_window, monkeypatch: pytest.MonkeyPatch, counts, expected_lines, has_cleanup_box
) -> None:
    window = main_window
    seen: dict[str, object] = {}

    def _fake_exec(box):
        seen["text"] = box.text()
        seen["lines"] = box.informativeText().splitlines()
        seen["checkbox"] = box.checkBox()
        seen["default"] = box.defaultButton().text()
        if box.checkBox() is not None:
            box.checkBox().setChecked(True)
        apply_button = next(button for button in box.buttons() if button.text() == "Apply")
        monkeypatch.setattr(
            main_window_module.QMessageBox, "clickedButton", lambda _self: apply_button
        )
        return 0

    monkeypatch.setattr(main_window_module.QMessageBox, "exec", _fake_exec)

    result = window._prompt_dashboard_bulk_action(**counts)

    total = sum(counts.values())
    assert seen["text"] == f"Apply to {total} selected job{'' if total == 1 else 's'}?"
    assert seen["lines"] == expected_lines
    assert (seen["checkbox"] is not None) is has_cleanup_box
    assert seen["default"] != "Apply"
    assert result is has_cleanup_box
    window.close()


def test_main_window_dashboard_bulk_prompt_cancel_returns_none(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window

    def _fake_exec(box):
        cancel_button = next(button for button in box.buttons() if button.text() != "Apply")
        monkeypatch.setattr(
            main_window_module.QMessageBox, "clickedButton", lambda _self: cancel_button
        )
        return 0

    monkeypatch.setattr(main_window_module.QMessageBox, "exec", _fake_exec)

    assert (
        window._prompt_dashboard_bulk_action(cancel_count=1, reset_count=1, delete_count=0) is None
    )
    window.close()


def test_main_window_dashboard_backspace_shortcut_uses_delete_handler(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "dashboard_backspace_key.mufinder", "DashboardBackspaceKey"
    )
    running_job = service.add_job(
        doc,
        label="job_running",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    running_job.status = JobStatus.RUNNING
    window._document = doc
    window._refresh_jobs_table()
    window._jobs_table.selectRow(0)
    cancelled: list[list[Job]] = []
    monkeypatch.setattr(window, "_cancel_jobs", lambda jobs: cancelled.append(jobs))
    monkeypatch.setattr(window, "_prompt_dashboard_bulk_action", lambda **_counts: False)

    assert (
        window._jobs_backspace_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    )
    window._jobs_backspace_shortcut.activated.emit()

    assert cancelled == [[running_job]]
    window.close()


def test_main_window_dashboard_delete_key_confirms_before_deleting_jobs(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "dashboard_delete_confirm.mufinder", "DashboardDeleteConfirm"
    )
    (job,) = _dashboard_with_jobs(window, service, doc, [JobStatus.DEFINED])

    monkeypatch.setattr(window, "_prompt_dashboard_bulk_action", lambda **_counts: None)
    window._on_dashboard_delete_key_requested()
    assert job.id in doc.jobs

    # The cancelled prompt left the row and its selection in place.
    monkeypatch.setattr(window, "_prompt_dashboard_bulk_action", lambda **_counts: False)
    window._on_dashboard_delete_key_requested()
    assert job.id not in doc.jobs
    window.close()


def _add_dashboard_job(service, doc, label, status):
    job = service.add_job(
        doc,
        label=label,
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = status
    return job


def _dashboard_row_widgets(window, row):
    """Snapshot the cell widgets + items for a dashboard row (identity checks)."""
    return {
        "label": window._jobs_table.item(row, window._JOB_LABEL_COLUMN),
        "status": window._jobs_table.item(row, window._JOB_STATUS_COLUMN),
        "resubmit": window._jobs_table.cellWidget(row, window._JOB_SUBMIT_COLUMN),
        "retry": window._jobs_table.cellWidget(row, window._JOB_RETRY_RETRIEVAL_COLUMN),
        "cancel": window._jobs_table.cellWidget(row, window._JOB_CANCEL_COLUMN),
        "inspect": window._jobs_table.cellWidget(row, window._JOB_INSPECT_COLUMN),
        "reset": window._jobs_table.cellWidget(row, window._JOB_RESET_COLUMN),
        "dismiss": window._jobs_table.cellWidget(row, window._JOB_DISMISS_COLUMN),
    }


def test_dashboard_reconcile_reuses_widgets_when_jobs_unchanged(
    main_window, tmp_path: Path
) -> None:
    """Two consecutive refreshes with the same jobs must reuse every cell widget."""
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_reuse.mufinder", "ReconcileReuse")
    _add_dashboard_job(service, doc, "r0", JobStatus.RUNNING)
    _add_dashboard_job(service, doc, "r1", JobStatus.FAILED)
    window._document = doc

    window._refresh_jobs_table()
    before = [_dashboard_row_widgets(window, row) for row in range(window._jobs_table.rowCount())]
    assert len(before) == 2

    window._refresh_jobs_table()
    after = [_dashboard_row_widgets(window, row) for row in range(window._jobs_table.rowCount())]
    assert len(after) == 2

    # Every item and every button is the identical object after re-refresh.
    for row_before, row_after in zip(before, after, strict=True):
        for key in row_before:
            assert row_before[key] is row_after[key], f"row widget {key} was recreated"
    window.close()


def test_dashboard_reconcile_status_change_updates_in_place(main_window, tmp_path: Path) -> None:
    """A status change updates the status cell + button enablement without widget churn."""
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_status.mufinder", "ReconcileStatus")
    job = _add_dashboard_job(service, doc, "s0", JobStatus.RUNNING)
    window._document = doc

    window._refresh_jobs_table()
    before = _dashboard_row_widgets(window, 0)
    assert before["status"].text() == JobStatus.RUNNING.value
    # Running: cancel enabled, resubmit/inspect/reset disabled.
    assert before["cancel"].isEnabled() is True
    assert before["resubmit"].isEnabled() is False
    assert before["inspect"].isEnabled() is False
    assert before["reset"].isEnabled() is False

    job.status = JobStatus.FAILED
    window._refresh_jobs_table()
    after = _dashboard_row_widgets(window, 0)

    # Same widget objects, updated text + enablement.
    for key in before:
        assert before[key] is after[key], f"row widget {key} was recreated on status change"
    assert after["status"].text() == JobStatus.FAILED.value
    assert after["cancel"].isEnabled() is False
    assert after["resubmit"].isEnabled() is True
    assert after["inspect"].isEnabled() is True
    assert after["reset"].isEnabled() is True
    window.close()


def test_dashboard_reconcile_remove_job_removes_only_its_row(main_window, tmp_path: Path) -> None:
    """Dismissing a job removes exactly its row; the surviving row keeps its widgets."""
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reconcile_remove.mufinder", "ReconcileRemove")
    job0 = _add_dashboard_job(service, doc, "x0", JobStatus.FAILED)
    _add_dashboard_job(service, doc, "x1", JobStatus.RUNNING)
    window._document = doc

    window._refresh_jobs_table()
    assert window._jobs_table.rowCount() == 2
    # Capture the surviving (running) row's widgets before removal.
    survivor_before = _dashboard_row_widgets(window, 1)

    job0.dismissed_at = utc_now_for_test()
    window._refresh_jobs_table()

    assert window._jobs_table.rowCount() == 1
    remaining_id = window._jobs_table.item(0, window._JOB_LABEL_COLUMN).data(
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
    _add_dashboard_job(service, doc, "a0", JobStatus.RUNNING)
    window._document = doc

    window._refresh_jobs_table()
    assert window._jobs_table.rowCount() == 1
    first_before = _dashboard_row_widgets(window, 0)

    new_job = _add_dashboard_job(service, doc, "a1", JobStatus.FAILED)
    window._refresh_jobs_table()

    assert window._jobs_table.rowCount() == 2
    first_after = _dashboard_row_widgets(window, 0)
    for key in first_before:
        assert first_before[key] is first_after[key], (
            f"existing widget {key} was recreated on insert"
        )
    # New job landed in the appended row (document iteration order).
    assert window._jobs_table.item(1, window._JOB_LABEL_COLUMN).data(
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
    job0 = _add_dashboard_job(service, doc, "c0", JobStatus.FAILED)
    job1 = _add_dashboard_job(service, doc, "c1", JobStatus.FAILED)
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

    # Locate job1's resubmit button by id and click it — must dispatch job1.
    def resubmit_btn_for(job_id):
        for row in range(window._jobs_table.rowCount()):
            item = window._jobs_table.item(row, window._JOB_LABEL_COLUMN)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == str(job_id):
                return window._jobs_table.cellWidget(row, window._JOB_SUBMIT_COLUMN)
        return None

    resubmit_btn_for(job1.id).click()
    resubmit_btn_for(job0.id).click()

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
        job.status = JobStatus.RUNNING
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

    cancel_btn = window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN)
    cancel_btn.selection_snapshot_rows = (0, 1)
    cancel_btn.clicked.emit()

    assert asked == [("Apply to selected jobs?", "Cancel all 2 selected jobs?")]
    assert cancelled == []
    window.close()


@pytest.mark.parametrize(
    ("kind", "status", "column_attr", "handler", "expected_prompt"),
    [
        (
            "cancel",
            JobStatus.RUNNING,
            "_JOB_CANCEL_COLUMN",
            "_on_cancel_from_dashboard",
            ("Cancel job", "Cancel job 'j0'?", "", "Cancel job", "Keep running"),
        ),
        (
            "dismiss",
            JobStatus.FAILED,
            "_JOB_DISMISS_COLUMN",
            "_on_dismiss_from_dashboard",
            (
                "Dismiss job",
                "Dismiss job 'j0' from the dashboard?",
                "The job stays in the project; its workflow row can still resubmit it.",
                "Dismiss job",
            ),
        ),
    ],
)
def test_main_window_dashboard_single_row_destructive_button_is_confirmed(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind,
    status,
    column_attr,
    handler,
    expected_prompt,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / f"single_{kind}.mufinder", "Single")
    job = service.add_job(
        doc,
        label="j0",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = status
    window._document = doc
    acted: list[list[object]] = []
    monkeypatch.setattr(window, handler, lambda ids: acted.append(list(ids)))
    prompts: list[tuple[str, ...]] = []
    answers = [False, True]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or answers.pop(0),
    )
    window._refresh_jobs_table()

    button = window._jobs_table.cellWidget(0, getattr(window, column_attr))
    button.selection_snapshot_rows = (0,)
    button.clicked.emit()
    assert acted == []

    button.selection_snapshot_rows = (0,)
    button.clicked.emit()
    assert acted == [[job.id]]
    assert prompts[0] == expected_prompt
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
        job.status = JobStatus.RUNNING
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

    button = window._jobs_table.cellWidget(0, window._JOB_CANCEL_COLUMN)
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
        job.status = status
        jobs.append(job)
    window._document = doc
    window._refresh_jobs_table()
    return window, jobs


def _select_dashboard_rows(window, rows) -> None:
    selection_model = window._jobs_table.selectionModel()
    selection_model.clearSelection()
    for row in rows:
        selection_model.select(
            window._jobs_table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )


def test_main_window_dashboard_hint_appears_only_for_a_multi_row_selection(
    qapp, tmp_path: Path
) -> None:
    window, _jobs = _dashboard_window_with_statuses(
        tmp_path, "hint", [JobStatus.FAILED, JobStatus.FAILED, JobStatus.RUNNING]
    )
    hint = window._jobs_selection_hint
    dock = window.findChild(main_window_module.QDockWidget, "jobDashboardDock")
    assert window._jobs_table.parent() is dock.widget()
    assert hint.parent() is dock.widget()
    assert hint.isHidden()

    _select_dashboard_rows(window, [0])
    assert hint.isHidden()

    _select_dashboard_rows(window, [0, 1])
    assert not hint.isHidden()
    assert hint.text() == "2 selected \u00b7 row buttons apply to all selected"
    prop = workflow_table_utils.MULTI_ROW_TARGET_PROPERTY
    assert bool(window._jobs_table.cellWidget(0, window._JOB_RESET_COLUMN).property(prop)) is True
    assert bool(window._jobs_table.cellWidget(2, window._JOB_RESET_COLUMN).property(prop)) is False

    _select_dashboard_rows(window, [])
    assert hint.isHidden()
    window.close()


def _open_dashboard_menu(window, monkeypatch, row: int = 0):
    from tests.support.reset_menu import FakeMenu

    FakeMenu.instances = []
    monkeypatch.setattr(main_window_module, "QMenu", FakeMenu)
    rect = window._jobs_table.visualRect(window._jobs_table.model().index(row, 0))
    window._jobs_table.customContextMenuRequested.emit(rect.center())
    return FakeMenu.instances[-1]


def test_main_window_dashboard_menu_acts_on_the_eligible_selected_jobs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, (running, failed, cancelled_job) = _dashboard_window_with_statuses(
        tmp_path, "menu", [JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED]
    )
    calls: list[tuple[str, list[object]]] = []
    for kind, handler in (
        ("resubmit", "_on_resubmit_from_dashboard"),
        ("inspect", "_on_inspect_failed_from_dashboard"),
        ("reset", "_on_reset_failed_from_dashboard"),
    ):
        monkeypatch.setattr(window, handler, lambda ids, k=kind: calls.append((k, list(ids))))
    _select_dashboard_rows(window, [0, 1, 2])

    menu = _open_dashboard_menu(window, monkeypatch)

    # Entries that apply to only some of the three selected jobs say how many.
    assert [action.text for action in menu.actions] == [
        "Resubmit (2 of 3)",
        "Retry Retrieval",
        "Cancel (1 of 3)",
        "Inspect (1 of 3)",
        "Reset (2 of 3)",
        "Dismiss (2 of 3)",
        "Delete…",
    ]
    assert menu.separators_before == [6]
    assert {action.text for action in menu.actions if not action.enabled} == {"Retry Retrieval"}

    menu.action("Resubmit (2 of 3)").triggered.emit()
    menu.action("Inspect (1 of 3)").triggered.emit()
    menu.action("Reset (2 of 3)").triggered.emit()
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
        tmp_path, "menu_confirm", [JobStatus.RUNNING, JobStatus.FAILED]
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
    menu = _open_dashboard_menu(window, monkeypatch)

    menu.action("Cancel (1 of 2)").triggered.emit()
    assert calls == []
    menu.action("Cancel (1 of 2)").triggered.emit()
    menu.action("Dismiss (1 of 2)").triggered.emit()

    assert calls == [("cancel", [running.id]), ("dismiss", [failed.id])]
    assert prompts[0] == ("Cancel jobs", "Cancel 1 job?", "", "Cancel 1 job", "Keep running")
    assert prompts[2][1] == "Dismiss 1 job from the dashboard?"
    window.close()


def test_main_window_dashboard_menu_selects_the_clicked_row_and_offers_delete(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _jobs = _dashboard_window_with_statuses(
        tmp_path, "menu_delete", [JobStatus.FAILED, JobStatus.FAILED, JobStatus.FAILED]
    )
    deleted: list[int] = []
    monkeypatch.setattr(window, "_on_dashboard_delete_key_requested", lambda: deleted.append(1))
    _select_dashboard_rows(window, [0, 1])

    menu = _open_dashboard_menu(window, monkeypatch, row=2)

    selected = sorted(index.row() for index in window._jobs_table.selectionModel().selectedRows())
    assert selected == [2]
    menu.action("Delete…").triggered.emit()
    assert deleted == [1]
    window.close()


_DASHBOARD_SCALE_ROWS = 300


def _add_scaled_dashboard_jobs(service, doc, *, now, count=_DASHBOARD_SCALE_ROWS):
    """Fill ``doc`` with ``count`` cluster jobs: running, queued, failed, stopped, running..."""

    statuses = (
        JobStatus.RUNNING,
        JobStatus.QUEUED,
        JobStatus.FAILED,
        JobStatus.STOPPED,
        JobStatus.RUNNING,
    )
    jobs = []
    for index in range(count):
        job = _add_dashboard_job(service, doc, f"scaled_{index:03d}", statuses[index % 5])
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
    monkeypatch.setattr(main_window_module, "utc_now", lambda: moment)


def _rows_in_jobs_viewport(window) -> set[int]:
    table = window._jobs_table
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
    assert window._jobs_table.rowCount() == _DASHBOARD_SCALE_ROWS

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
    window.resize(1400, 900)
    window.show()
    try:
        qapp.processEvents()
        window._refresh_jobs_table()
        qapp.processEvents()

        writes = _record_dashboard_cell_writes(monkeypatch)
        window._refresh_active_job_runtime_cells()
        assert writes == [], "an unchanged runtime must not be rewritten"

        job_by_row = {
            row: doc.jobs[window._jobs_table_row_job_id(window._jobs_table.item(row, 0))]
            for row in range(window._jobs_table.rowCount())
        }
        for scroll_to_bottom in (False, True):
            if scroll_to_bottom:
                window._jobs_table.scrollToBottom()
                qapp.processEvents()
            visible = _rows_in_jobs_viewport(window)
            expected = {row for row in visible if job_by_row[row].status == JobStatus.RUNNING}
            assert expected, "the viewport must show at least one running row"
            assert len(visible) < _DASHBOARD_SCALE_ROWS

            now += timedelta(seconds=1)
            _freeze_dashboard_clock(monkeypatch, now)
            writes.clear()
            window._refresh_active_job_runtime_cells()

            assert {kind for kind, _row, _column in writes} == {"setText"}
            assert {column for _kind, _row, column in writes} == {window._JOB_RUNTIME_COLUMN}
            assert sorted(row for _kind, row, _column in writes) == sorted(expected)

        assert window._job_runtime_timer.isActive() is True
    finally:
        # Hidden, the window closes at teardown without the save prompt.
        window.hide()


def test_dashboard_columns_are_sized_once_not_per_cell(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No column re-measures itself per cell change; the runtime column still widens."""

    window = main_window
    header = window._jobs_table.horizontalHeader()
    assert all(
        header.sectionResizeMode(column) != QHeaderView.ResizeMode.ResizeToContents
        for column in range(window._jobs_table.columnCount())
    )

    service = ProjectService()
    doc = service.create_project(tmp_path / "runtime_width.mufinder", "RuntimeWidth")
    now = datetime(2026, 4, 20, 16, 0, 0, tzinfo=UTC)
    _freeze_dashboard_clock(monkeypatch, now)
    job = _add_dashboard_job(service, doc, "long_runner", JobStatus.RUNNING)
    job.started_at = now - timedelta(hours=9999, minutes=59, seconds=59)
    window._document = doc
    window._refresh_jobs_table()
    runtime_item = window._jobs_table.item(0, window._JOB_RUNTIME_COLUMN)
    assert runtime_item.text() == "9999:59:59"
    width_before = window._jobs_table.columnWidth(window._JOB_RUNTIME_COLUMN)

    _freeze_dashboard_clock(monkeypatch, now + timedelta(seconds=1))
    window._refresh_active_job_runtime_cells()

    assert runtime_item.text() == "10000:00:00"
    width_grown = window._jobs_table.columnWidth(window._JOB_RUNTIME_COLUMN)
    assert width_grown > width_before

    # A refresh that shortens the text fits the column back down.
    job.started_at = now
    window._refresh_jobs_table()

    assert runtime_item.text() == "00:01"
    assert window._jobs_table.columnWidth(window._JOB_RUNTIME_COLUMN) < width_grown
