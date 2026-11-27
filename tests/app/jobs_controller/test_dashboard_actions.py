"""Job Dashboard actions carried out by the jobs controller: flushes, the status bar, queue checks and what each action does."""

from __future__ import annotations

from mufinder.services.job_monitoring import job_status_word
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

import mufinder.app.main_window as main_window_module
import mufinder.app.jobs_controller as jobs_controller_module
import mufinder.app.workflow_table_utils as workflow_table_utils
from mufinder.app import selection_snapshot as selection_snapshot_module
from mufinder.app.selection_snapshot import SelectionScope
from mufinder.app.journal_widget import JournalWidget
from mufinder.castep_setup import CastepSetup
from mufinder.domain.cluster import ClusterConfig
from mufinder.app.convergence_types import ConvergenceConfig
from mufinder.domain.enums import EndReason, JobActivity, JobStatus
from mufinder.domain.models import CalculationRun, ProvenanceRecord
from mufinder.services.job_reconcile import JobEvidence, OutputState, SchedulerState
from mufinder.services.project_service import ProjectService
from mufinder.services.project_summary import job_stage_key, job_stage_name
from mufinder.services.ssh_client import SSHAuthenticationError, SSHCredentials
from tests.support.jobs import make_job, set_job_state
from tests.support.cluster_ssh import FakeSSH
from tests.support.qt import pump_until
from mufinder.services.cluster_status import ClusterConnectionTone
from mufinder.services.scheduler import SchedulerJobInfo, SlurmScheduler
from tests.support.dashboard import (
    job_row,
    look_button,
    row_actions,
    run_button,
)
from tests.support.main_window import open_project_with_host, select_route
from tests.support.dashboard import add_dashboard_job, orphaned_cluster_job


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

    window._jobs._on_cluster_queue_checked(SimpleNamespace(checked_at=None, elapsed_by_job_id={}))
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

    window._jobs._on_cluster_queue_checked(SimpleNamespace(checked_at=None, elapsed_by_job_id={}))
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
        window._jobs._cluster_connection_label,
        window._jobs._cluster_queue_checked_label,
        window._jobs._cluster_refresh_button,
    )


def _shown(window) -> list[bool]:
    return [not widget.isHidden() for widget in _cluster_widgets(window)]


_COMPLETE_OUTPUT = Path(__file__).parents[2] / "fixtures" / "uep" / "Si2.castep"


def _project_with_job(
    window,
    tmp_path: Path,
    *,
    target: str = "SCARF",
    status: str = "running",
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
    set_job_state(job, status)
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
    window._jobs._on_cluster_connection_status_changed(name is not None)


def test_status_bar_hides_cluster_widgets_with_no_project(main_window) -> None:
    assert main_window._document is None
    assert _shown(main_window) == [False, False, False]
    assert main_window._jobs._cluster_refresh_button.isEnabled() is False
    assert "nothing to check" in main_window._jobs._cluster_refresh_button.toolTip()


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
    assert main_window._jobs._cluster_connection_label.text() == "Not connected to SCARF"
    assert main_window._jobs._cluster_queue_checked_label.text() == "Queue last checked: never"
    assert main_window._jobs._cluster_refresh_button.isEnabled() is True


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
    palette = jobs_controller_module.current_palette()
    label = window._jobs._cluster_connection_label
    monkeypatch.setattr(main_window_module, "show_detailed_warning", lambda *_a, **_k: None)
    monkeypatch.setattr(jobs_controller_module, "show_detailed_warning", lambda *_a, **_k: None)
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
    window._jobs._on_cluster_poll_failed("Queue polling failed for SCARF: timeout")
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
    monkeypatch.setattr(jobs_controller_module, "show_detailed_warning", lambda *_a, **_k: None)
    _project_with_job(window, tmp_path)
    _connect_status_bar(window, monkeypatch, "SCARF")
    window._jobs._on_cluster_queue_checked("12:34:56")
    _connect_status_bar(window, monkeypatch, None)
    window._jobs._on_cluster_poll_failed("Queue polling failed for SCARF: timeout")
    assert window._jobs._cluster_queue_checked_label.text() == "Queue last checked: 12:34:56"

    window._jobs.reset_tracking_for_project_change()
    window._refresh_jobs_table()

    assert window._jobs._cluster_connection_label.text() == "Not connected to SCARF"
    assert window._jobs._cluster_queue_checked_label.text() == "Queue last checked: never"


def test_status_bar_ignores_a_poll_from_the_previous_project(main_window, tmp_path: Path) -> None:
    window = main_window
    job = _project_with_job(window, tmp_path)
    previous_project_poll = SimpleNamespace(
        checked_at="12:34:56",
        elapsed_by_job_id={},
        job_ids=(uuid4(),),
    )

    window._jobs._on_cluster_queue_checked(previous_project_poll)
    assert window._jobs._cluster_queue_checked_label.text() == "Queue last checked: never"

    window._jobs._on_cluster_queue_checked(
        SimpleNamespace(
            checked_at="12:35:56",
            elapsed_by_job_id={},
            job_ids=(job.id,),
        )
    )
    assert window._jobs._cluster_queue_checked_label.text() == "Queue last checked: 12:35:56"


def test_status_bar_refresh_disabled_with_a_tooltip_when_no_job_can_be_polled(
    main_window, tmp_path: Path
) -> None:
    job = _project_with_job(
        main_window,
        tmp_path,
        status="submitting",
        remote_job_id=None,
    )
    button = main_window._jobs._cluster_refresh_button

    assert _shown(main_window) == [True, True, True]
    assert button.isEnabled() is False
    assert "no active cluster job has a scheduler job ID" in button.toolTip()

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
    monkeypatch.setattr(
        window._cluster_runner, "check_jobs", lambda _targets, **_kwargs: calls.append("refresh")
    )

    # A periodic poll says nothing in the status bar.
    window.statusBar().showMessage("Ready")
    snapshot = SimpleNamespace(
        checked_at="12:34:56",
        elapsed_by_job_id={},
        job_ids=(job.id,),
    )
    window._jobs._on_cluster_queue_checked(snapshot)
    assert window.statusBar().currentMessage() == "Ready"

    window._jobs._cluster_refresh_button.click()
    assert calls == ["refresh"]
    # A periodic poll that ends before the refresh's own pass is not its answer.
    window._jobs._on_cluster_queue_checked(snapshot)
    assert window.statusBar().currentMessage() == "Ready"
    refreshed = SimpleNamespace(**vars(snapshot), refresh=True)
    window._jobs._on_cluster_queue_checked(refreshed)
    assert window.statusBar().currentMessage() == "Refreshed queue: checked 1 job at 12:34:56"

    window.statusBar().showMessage("Ready")
    window._jobs._on_cluster_queue_checked(refreshed)
    assert window.statusBar().currentMessage() == "Ready"


def test_status_bar_refresh_reconnects_when_connected_to_another_cluster(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _project_with_job(window, tmp_path)
    _connect_status_bar(window, monkeypatch, "PSI")
    assert window._jobs._cluster_connection_label.text() == "Not connected to SCARF"

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
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [scarf])
    connected: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda config, password=None: connected.append(config.name),
    )
    polls: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner, "check_jobs", lambda _targets, **_kwargs: polls.append("poll")
    )

    window._jobs._cluster_refresh_button.click()

    assert connected == ["SCARF"]
    assert polls == ["poll"]


def _ended_with_output_left(window, tmp_path: Path, *, reason: EndReason, dismissed=False):
    job = _project_with_job(window, tmp_path, status="failed", remote_job_id="777")
    job.end_reason = reason
    job.remote_directory = "/remote/job"
    if dismissed:
        ProjectService().dismiss_job(job)
    window._refresh_jobs_table()
    return job


@pytest.mark.parametrize("dismissed", [False, True])
def test_status_bar_shows_cluster_widgets_for_an_ended_job_with_output_left(
    main_window, tmp_path: Path, dismissed: bool
) -> None:
    _ended_with_output_left(
        main_window, tmp_path, reason=EndReason.OUTPUT_UNAVAILABLE, dismissed=dismissed
    )

    assert _shown(main_window) == [True, True, True]
    assert main_window._jobs._cluster_connection_label.text() == "Not connected to SCARF"
    assert main_window._jobs._cluster_refresh_button.isEnabled() is True


@pytest.mark.parametrize("status", ["completed", "judged"])
def test_status_bar_hides_cluster_widgets_with_no_output_left_to_fetch(
    main_window, tmp_path: Path, status: str
) -> None:
    if status == "completed":
        job = _project_with_job(main_window, tmp_path, status="completed")
        job.remote_directory = "/remote/job"
    else:
        job = _ended_with_output_left(main_window, tmp_path, reason=EndReason.CASTEP_ERROR)
        job.last_status_source = "retrieval"
    main_window._refresh_jobs_table()

    assert _shown(main_window) == [False, False, False]


def test_status_bar_never_warns_of_a_lost_connection_for_ended_jobs_only(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _ended_with_output_left(window, tmp_path, reason=EndReason.OUTPUT_UNAVAILABLE)
    window._jobs._cluster_last_connected_name = "SCARF"
    monkeypatch.setattr(jobs_controller_module, "show_detailed_warning", lambda *a, **k: None)

    window._jobs._on_cluster_poll_failed("Queue polling failed for SCARF: boom")

    assert window._jobs._cluster_connection_label.text() == "Not connected to SCARF"
    assert window._jobs._cluster_status_tone is ClusterConnectionTone.NEUTRAL


def test_status_bar_refresh_reconnects_to_an_ended_jobs_cluster_and_completes_it(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    job = _ended_with_output_left(window, tmp_path, reason=EndReason.OUTPUT_UNAVAILABLE)
    job.submitted_at = datetime.now(UTC)
    (tmp_path / job.label).mkdir()
    execution = window._job_execution
    monkeypatch.setattr(execution, "_job_output_directory", lambda j: tmp_path / j.label)
    monkeypatch.setattr(execution, "_job_seed_name", lambda j: j.label)
    scarf = ClusterConfig(name="SCARF", hostname="scarf.invalid", auth_method="key")
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [scarf])

    ssh = FakeSSH()
    ssh.remote_files[f"/remote/job/{job.label}.castep"] = _COMPLETE_OUTPUT.read_bytes()

    class _Scheduler(SlurmScheduler):
        def status(self, _ssh, job_ids):
            return {}

        def history(self, _ssh, job_ids):
            return {i: SchedulerJobInfo(job_id=i, state="COMPLETED") for i in job_ids}

    runner = window._cluster_runner._runner
    connected: list[str] = []

    def connect(config, password=None):  # runs on the cluster worker thread
        connected.append(config.name)
        runner._ssh = ssh
        runner._scheduler = _Scheduler()
        runner._config = config
        runner.connection_status_changed.emit(True)

    monkeypatch.setattr(runner, "connect_to_cluster", connect)
    assert not window._cluster_runner.is_connected

    window._jobs._cluster_refresh_button.click()

    assert connected == ["SCARF"]
    assert pump_until(lambda: job.status == JobStatus.COMPLETED, timeout=5.0)
    # Nothing is left for a refresh to change, so the widgets go.
    window._jobs.sync_cluster_status_widgets()
    assert _shown(window) == [False, False, False]


def test_status_bar_names_clusters_a_refresh_could_not_reach_before_the_warning(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _project_with_job(window, tmp_path)
    second = ProjectService().add_job(
        window._document,
        label="psi_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="PSI",
    )
    second.status = JobStatus.ACTIVE
    second.activity = JobActivity.RUNNING
    second.remote_job_id = "678"

    # SCARF is configured but refuses the connection; PSI is not configured.
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
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [scarf])

    def _refuse(config, password=None):
        raise ConnectionError(f"{config.name} refused the connection")

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _refuse)

    shown_during_warning: list[str] = []
    monkeypatch.setattr(
        main_window_module,
        "show_detailed_warning",
        lambda *_a, **_k: shown_during_warning.append(window.statusBar().currentMessage()),
    )
    monkeypatch.setattr(
        jobs_controller_module,
        "show_detailed_warning",
        lambda *_a, **_k: shown_during_warning.append(window.statusBar().currentMessage()),
    )

    window._jobs._on_refresh_cluster_queue()

    assert shown_during_warning == ["Failed to check queue on SCARF, PSI"]
    assert window.statusBar().currentMessage() == "Failed to check queue on SCARF, PSI"


def test_status_bar_reports_a_failed_queue_refresh(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    monkeypatch.setattr(main_window_module, "show_detailed_warning", lambda *_a, **_k: None)
    monkeypatch.setattr(jobs_controller_module, "show_detailed_warning", lambda *_a, **_k: None)
    _project_with_job(window, tmp_path)
    _connect_status_bar(window, monkeypatch, "SCARF")
    monkeypatch.setattr(window._cluster_runner, "check_jobs", lambda _targets, **_kwargs: None)

    window._jobs._cluster_refresh_button.click()
    _connect_status_bar(window, monkeypatch, None)
    window._jobs._on_cluster_poll_failed("Queue polling failed for SCARF: timeout")

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

    window._jobs._on_cluster_queue_checked(_AwareTime())

    assert "12:34:56" in window._jobs._cluster_queue_checked_label.text()
    window.close()


def test_main_window_cluster_queue_checked_uses_snapshot_timestamp(main_window) -> None:
    window = main_window

    snapshot = SimpleNamespace(
        checked_at=datetime(2026, 4, 16, 12, 34, 56, tzinfo=UTC),
        elapsed_by_job_id={},
    )

    window._jobs._on_cluster_queue_checked(snapshot)

    expected = snapshot.checked_at.astimezone().strftime("%H:%M:%S")
    assert window._jobs._cluster_queue_checked_label.text() == f"Queue last checked: {expected}"
    window.close()


def test_main_window_cluster_queue_checked_unknown_object_stays_compact(main_window) -> None:
    window = main_window

    window._jobs._on_cluster_queue_checked(
        SimpleNamespace(checked_at=object(), elapsed_by_job_id={})
    )

    assert window._jobs._cluster_queue_checked_label.text() == "Queue last checked: unknown"
    window.close()


def test_main_window_page_navigation_does_not_recompute_stage_seals(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "nav_icons.mufinder", "NavIcons")
    window._document = doc
    window._refresh_ui_after_project_change()

    # Switching pages cannot change workflow completion, so navigation must not
    # re-run the (job/result-scanning) completion checks.
    calls = 0
    original = window._project_navigator.refresh

    def _counting(*args, **kwargs) -> bool:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(window._project_navigator, "refresh", _counting)

    select_route(window, "muon_candidates")

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
    redwood_job.status = JobStatus.ACTIVE
    redwood_job.activity = JobActivity.RUNNING

    scarf_job = service.add_job(
        doc,
        label="scarf_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="SCARF",
    )
    scarf_job.status = JobStatus.ACTIVE
    scarf_job.activity = JobActivity.QUEUED

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
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_configs", lambda _settings: [redwood, scarf]
    )

    connect_calls: list[str] = []
    refresh_calls: list[str] = []

    def _connect(config, password=None):
        _ = password
        connect_calls.append(config.name)
        window._cluster_runner._connected = True
        window._cluster_runner._connected_cluster_name = config.name

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(
        window._cluster_runner,
        "check_jobs",
        lambda _targets, **_kwargs: refresh_calls.append("refresh"),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._jobs._on_refresh_cluster_queue()

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
    redwood_job.status = JobStatus.ACTIVE
    redwood_job.activity = JobActivity.QUEUED
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
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [redwood])

    connect_calls: list[tuple[str, str | None]] = []

    def _connect(config, password=None):
        connect_calls.append((config.name, password))
        if password is None:
            raise RuntimeError("Password is required for password authentication.")
        window._cluster_runner._connected = True
        window._cluster_runner._connected_cluster_name = config.name

    refresh_calls: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(
        window._cluster_runner,
        "check_jobs",
        lambda _targets, **_kwargs: refresh_calls.append("refresh"),
    )
    monkeypatch.setattr(
        jobs_controller_module,
        "prompt_for_cluster_credentials",
        lambda *_args, **_kwargs: SSHCredentials(target_secret="secret"),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._jobs._on_refresh_cluster_queue()

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
    redwood_job.status = JobStatus.ACTIVE
    redwood_job.activity = JobActivity.QUEUED
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
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [redwood])

    connect_calls: list[tuple[str, str | None]] = []

    def _connect(config, password=None):
        connect_calls.append((config.name, password))
        if password is None:
            raise SSHAuthenticationError("SSH key authentication failed: private key is encrypted")
        window._cluster_runner._connected = True
        window._cluster_runner._connected_cluster_name = config.name

    refresh_calls: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(
        window._cluster_runner,
        "check_jobs",
        lambda _targets, **_kwargs: refresh_calls.append("refresh"),
    )
    monkeypatch.setattr(
        jobs_controller_module,
        "prompt_for_cluster_credentials",
        lambda *_args, **_kwargs: SSHCredentials(target_secret="secret-passphrase"),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._jobs._on_refresh_cluster_queue()

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
    monkeypatch.setattr(
        jobs_controller_module,
        "show_detailed_warning",
        lambda _parent, title, _message, **_kwargs: warnings.append(title),
    )

    window._jobs._on_cluster_poll_failed("Queue polling failed for alpha: timeout")

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
    active_job.status = JobStatus.ACTIVE
    active_job.activity = JobActivity.RUNNING
    inactive_job.status = JobStatus.DEFINED
    window._document = doc

    cancelled_ids: list[object] = []
    monkeypatch.setattr(window._local_runner, "cancel", lambda job_id: cancelled_ids.append(job_id))

    window._jobs.cancel_jobs([active_job, inactive_job])

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
    job0.status = JobStatus.ACTIVE
    job0.activity = JobActivity.RUNNING
    job1.status = JobStatus.ACTIVE
    job1.activity = JobActivity.RUNNING
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

    cancel_btn = run_button(window, 0)
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
    job0.status = JobStatus.ACTIVE
    job0.activity = JobActivity.RUNNING
    window._document = doc
    cancelled: list[list[object]] = []
    monkeypatch.setattr(
        window._jobs, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids))
    )
    prompted: list[bool] = []
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: prompted.append(True) or SelectionScope.ALL_SELECTED,
    )
    window._refresh_jobs_table()

    cancel_btn = run_button(window, 0)
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
    job0.status = JobStatus.ACTIVE
    job0.activity = JobActivity.RUNNING
    job1.status = JobStatus.ACTIVE
    job1.activity = JobActivity.RUNNING
    window._document = doc
    cancelled: list[list[object]] = []
    monkeypatch.setattr(
        window._jobs, "_on_cancel_from_dashboard", lambda ids: cancelled.append(list(ids))
    )
    prompted: list[bool] = []
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: prompted.append(True) or SelectionScope.ALL_SELECTED,
    )
    window._refresh_jobs_table()

    # Snapshot only contains row 0; click Cancel on row 1 (not in snapshot)
    cancel_btn = run_button(window, 1)
    assert isinstance(cancel_btn, SelectionSnapshotPushButton)
    cancel_btn.selection_snapshot_rows = (0,)
    cancel_btn.clicked.emit()

    assert cancelled == [[job1.id]]
    assert prompted == []
    window.close()


def test_main_window_dashboard_handlers_no_document_are_noops(main_window) -> None:
    window = main_window

    window._jobs._on_submit_local_from_dashboard([uuid4()])
    window._jobs._on_resubmit_from_dashboard([uuid4()])
    window._jobs._on_cancel_from_dashboard([uuid4()])
    window._jobs._on_inspect_failed_from_dashboard([uuid4()])

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
    cancelled: list[list[object]] = []
    inspected: list[list[object]] = []
    monkeypatch.setattr(window._jobs, "submit_jobs_locally", lambda jobs: submitted.append(jobs))
    monkeypatch.setattr(window._jobs, "cancel_jobs", lambda jobs: cancelled.append(jobs))
    monkeypatch.setattr(window._jobs, "inspect_failed_jobs", lambda jobs: inspected.append(jobs))

    window._jobs._on_submit_local_from_dashboard([job.id])
    window._jobs._on_cancel_from_dashboard([job.id])
    window._jobs._on_inspect_failed_from_dashboard([job.id])

    assert submitted == [[job]]
    assert cancelled == [[job]]
    assert inspected == [[job]]

    job.status = JobStatus.ENDED
    job.end_reason = EndReason.CANCELLED
    routed: list[tuple[list[object], dict[str, object]]] = []
    monkeypatch.setattr(
        window._convergence_controller,
        "key_for_job",
        lambda job_id: 300.0 if job_id == job.id else None,
    )
    monkeypatch.setattr(
        window._convergence_controller,
        "submit",
        lambda keys, **kwargs: routed.append((list(keys), kwargs)),
    )
    # Resubmit checks for a CASTEP executable before the row asks anything.
    monkeypatch.setattr(
        jobs_controller_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/opt/castep", mpirun_path=""),
    )
    window._jobs._on_resubmit_from_dashboard([job.id])

    # Resubmit is the row's Run: the owning controller submits the job's row.
    assert routed == [([300.0], {"target": "local"})]
    window.close()


def test_dashboard_double_click_shows_the_stage_with_the_row_selected(
    main_window, qapp, tmp_path: Path
) -> None:
    window = main_window
    service, doc, _host = open_project_with_host(window, tmp_path)
    window._convergence_view.configure_requested.emit(
        ConvergenceConfig(
            sweep_values=(300.0, 400.0),
            shared_params={"task": "SINGLEPOINT"},
            overrides={},
            name_prefix="cutoff_convergence",
        )
    )
    job = add_dashboard_job(service, doc, "conv_ecut_400", "failed")
    definition = window._convergence_controller.active_definition()
    definition.parameters["job_ids_by_cutoff"]["400"] = str(job.id)
    window._refresh_ui_after_project_change()
    window._set_active_route("host_structure")
    window.resize(1400, 900)
    window.show()
    try:
        qapp.processEvents()
        table = window._job_dashboard.table
        rect = table.visualRect(
            table.model().index(job_row(window, job.id), window._job_dashboard.LABEL_COLUMN)
        )
        QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
        QTest.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
        qapp.processEvents()

        assert window._project_navigator.active_route() == "convergence_cutoff"
        assert window._convergence_view._selected_keys() == (400.0,)
    finally:
        window.hide()


def test_dashboard_open_finds_a_convergence_row_whose_form_has_diverged(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service, doc, _host = open_project_with_host(window, tmp_path)
    window._convergence_view.configure_requested.emit(
        ConvergenceConfig(
            sweep_values=(300.0, 400.0),
            shared_params={"task": "SINGLEPOINT"},
            overrides={},
            name_prefix="another_prefix",
        )
    )
    job = add_dashboard_job(service, doc, "another_prefix_ecut_400", "failed")
    definition = window._convergence_controller.active_definition()
    definition.parameters["job_ids_by_cutoff"]["400"] = str(job.id)
    window._refresh_ui_after_project_change()

    window._on_open_in_stage_from_dashboard([job.id])

    assert window._convergence_view._selected_keys() == (400.0,)


@pytest.mark.parametrize(
    "stage",
    [
        main_window_module.WorkflowStage.MAGNETIC_FIELD,
        main_window_module.WorkflowStage.CONVERGENCE_KPOINT,
    ],
)
def test_dashboard_open_shows_the_page_and_says_when_the_job_has_no_row(
    main_window, tmp_path: Path, stage
) -> None:
    window = main_window
    service, doc, _host = open_project_with_host(window, tmp_path)
    job = service.add_job(doc, label="orphan", workflow_stage=stage, calc_type="castep")
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN

    window._on_open_in_stage_from_dashboard([job.id])

    assert window._project_navigator.active_route() == job_stage_key(stage)
    name = job_stage_name(stage)
    assert window.statusBar().currentMessage() == f"orphan has no row on {name} now"


def test_dashboard_open_clears_an_earlier_no_row_message(main_window, tmp_path: Path) -> None:
    window = main_window
    service, doc, _host = open_project_with_host(window, tmp_path)
    window._convergence_view.configure_requested.emit(
        ConvergenceConfig(
            sweep_values=(300.0, 400.0),
            shared_params={"task": "SINGLEPOINT"},
            overrides={},
            name_prefix="cleared",
        )
    )
    with_row = add_dashboard_job(service, doc, "cleared_ecut_400", "failed")
    definition = window._convergence_controller.active_definition()
    definition.parameters["job_ids_by_cutoff"]["400"] = str(with_row.id)
    orphan = service.add_job(
        doc,
        label="orphan",
        workflow_stage=main_window_module.WorkflowStage.MAGNETIC_FIELD,
        calc_type="castep",
    )
    orphan.status = JobStatus.ENDED
    orphan.end_reason = EndReason.UNKNOWN
    window._refresh_ui_after_project_change()

    window._on_open_in_stage_from_dashboard([orphan.id])
    assert "orphan has no row" in window.statusBar().currentMessage()

    window._on_open_in_stage_from_dashboard([with_row.id])

    assert window._convergence_view._selected_keys() == (400.0,)
    assert window.statusBar().currentMessage() == ""


def test_every_workflow_page_can_select_a_job_row(main_window, tmp_path: Path) -> None:
    """Each stage with a job table hands back a controller and a view that can find rows."""

    window = main_window
    open_project_with_host(window, tmp_path)
    for stage in main_window_module.WorkflowStage:
        page = window._job_stage_page(stage)
        assert window._project_navigator.has_route(job_stage_key(stage))
        if page is None:
            assert stage in {
                main_window_module.WorkflowStage.MUON_GENERATION,
                main_window_module.WorkflowStage.SITE_ANALYSIS,
                main_window_module.WorkflowStage.MAGNETIC_FIELD,
            }
            continue
        controller, view = page
        assert controller.key_for_job(uuid4()) is None
        assert view.select_rows_for_keys(["no such row"]) is False


@pytest.mark.parametrize("open_before_showing", [False, True])
def test_an_empty_dashboard_opens_one_row_tall(
    main_window, qapp, tmp_path: Path, open_before_showing: bool
) -> None:
    # The dock stays hidden until a project opens, which may be after the first show.
    window = main_window
    root = ProjectService().create_project(tmp_path / "height.mufinder", "Height").project_root
    window.resize(1400, 1000)
    if open_before_showing:
        assert window.open_project(root)
    window.show()
    try:
        qapp.processEvents()
        if not open_before_showing:
            assert window.open_project(root)

        table = window._job_dashboard.table
        row_height = table.verticalHeader().defaultSectionSize()
        # The fit applies on a later pass of the event loop, once the show is laid out.
        for _attempt in range(100):
            if table.viewport().height() == row_height:
                break
            QTest.qWait(20)
        assert table.viewport().height() == row_height
    finally:
        window.hide()


@pytest.mark.parametrize("field_name", ["propagation", "applied", "hyperfine", "measured"])
def test_dismiss_flush_preserves_typing_and_saves_the_typed_value(
    project_with_host, monkeypatch, qapp, field_name
) -> None:
    window, service, document = project_with_host
    host = document.structures[document.project.host_structure_id]
    job = service.add_job(
        document,
        label="failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    magnetic = window._magnetic_field_view
    if field_name == "propagation":
        select_route(window, "host_structure")
        edit = window._host_structure_view._propagation_x
    else:
        select_route(window, "magnetic_fields")
        if field_name == "applied":
            edit = magnetic._applied_field_input.edits[0]
        elif field_name == "hyperfine":
            edit = magnetic._hyperfine_coupling
        else:
            magnetic._measured_magnetization_check.setChecked(True)
            edit = magnetic._measured_magnetization_input.edits[0]
    window.show()
    window.activateWindow()
    edit.setFocus()
    qapp.processEvents()
    assert edit.hasFocus()
    edit.selectAll()
    QTest.keyClicks(edit, "00.1")
    cursor = edit.cursorPosition()
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *a, **kw: True)
    window._jobs._on_dismiss_from_dashboard([job.id])
    window._flush_dashboard()
    assert job.dismissed_at is not None
    assert edit.text() == "00.1"
    assert edit.cursorPosition() == cursor
    assert edit.hasFocus()
    QTest.keyClicks(edit, "2")
    QTest.keyClick(edit, Qt.Key.Key_Tab)
    assert edit.text() == "00.12"
    window._save_project()
    loaded = service.load_project(document.project_root)
    if field_name == "propagation":
        assert host.properties["propagation_vector"][0] == 0.12
        assert loaded.structures[host.id].properties["propagation_vector"][0] == 0.12
    else:
        state = loaded.ui_state[window._MAGNETIC_FIELD_FORM_KEY]
        key = {
            "applied": "applied_field_T",
            "hyperfine": "hyperfine_coupling",
            "measured": "measured_magnetization",
        }[field_name]
        value = state[key] if field_name == "hyperfine" else state[key][0]
        assert float(value) == 0.12


def test_host_refresh_updates_an_idle_field(project_with_host) -> None:
    window, _service, document = project_with_host
    host = document.structures[document.project.host_structure_id]
    host.properties["propagation_vector"] = [0.25, 0.0, 0.0]
    window._refresh_host_structure_view()
    assert window._host_structure_view._propagation_x.text() == "0.25"


def test_reopening_the_same_project_replaces_unfinished_host_text(project_with_host, qapp):
    window, _service, document = project_with_host
    root = document.project_root
    window._save_project()
    select_route(window, "host_structure")
    window.show()
    window.activateWindow()
    edit = window._host_structure_view._propagation_x
    edit.setFocus()
    qapp.processEvents()
    edit.selectAll()
    QTest.keyClicks(edit, "-")
    assert edit.text() == "-"
    assert window.open_project(root)
    assert edit.text() == "0"
    assert not edit.isModified()


@pytest.mark.parametrize("leave_focus", [False, True])
def test_energy_barrier_draft_limit_survives_a_dashboard_flush(
    project_with_host, qapp, leave_focus
):
    window, _service, _document = project_with_host
    select_route(window, "energy_barriers")
    edit = window._energy_barriers_view._max_path_length_input
    window.show()
    window.activateWindow()
    edit.setFocus()
    qapp.processEvents()
    QTest.keyClicks(edit, "2.5")
    if leave_focus:
        QTest.keyClick(edit, Qt.Key.Key_Tab)
    window._request_dashboard_flush(failure_message="Autosave failed")
    window._flush_dashboard()
    assert edit.text() == "2.5"


def _connect_to_scarf(window) -> None:
    window._cluster_runner._connected = True
    window._cluster_runner._connected_cluster_name = "SCARF"


def test_dashboard_offers_retrieval_for_a_cluster_job_mufinder_lost_track_of(main_window, tmp_path):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "orphaned.mufinder", "Orphaned")
    orphaned_cluster_job(service, doc)
    window._document = doc
    _connect_to_scarf(window)

    window._refresh_jobs_table()

    assert row_actions(window) == ("", "Check now", "✕")
    window.close()


def test_dismissing_a_job_with_unretrieved_output_says_so(main_window, tmp_path, monkeypatch):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismiss_orphan.mufinder", "DismissOrphan")
    job = orphaned_cluster_job(service, doc)
    window._document = doc
    window._refresh_jobs_table()
    prompts: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt, **_k: prompts.append(prompt) or False,
    )

    window._job_dashboard.run_menu_action("dismiss", [job.id])

    assert prompts == [
        (
            "Dismiss job",
            "Dismiss job 'o0' from the dashboard?",
            "The job stays in the project. Show it again with Show dismissed.\n\n"
            "Its output may still be on SCARF and has not been retrieved. Dismissing "
            "does not retrieve it; use Check now or Refresh Queue first to keep the result.",
            "Dismiss job",
        )
    ]
    assert job.dismissed_at is None
    window.close()


def test_cancelling_a_cluster_job_with_no_scheduler_id_cancels_it_in_the_project(
    main_window, tmp_path
):
    # The scheduler cannot be asked to cancel a job it gave no ID for, so the
    # row would otherwise stay active with no way out.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cancel_untracked.mufinder", "CancelUntracked")
    job = add_dashboard_job(service, doc, "u0", JobStatus.DEFINED)
    job.execution_target = "SCARF"
    service.submit_job(job)
    job.submitted_at = datetime.now(UTC) - timedelta(minutes=10)
    window._document = doc
    window._refresh_jobs_table()

    window._jobs.cancel_jobs([job])

    assert job_status_word(job) == "cancelled"
    assert window.statusBar().currentMessage() == (
        "Cancelled 1 job in the project; SCARF was not contacted for jobs it could not be "
        "asked about"
    )
    window.close()


def test_cancelling_a_job_that_is_still_being_submitted_goes_to_the_cluster(
    main_window, tmp_path, monkeypatch
):
    # The scheduler has not answered yet. Cancelling the job in the project
    # alone would leave the submission to run.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cancel_submitting.mufinder", "CancelSubmitting")
    job = add_dashboard_job(service, doc, "s0", JobStatus.DEFINED)
    job.execution_target = "SCARF"
    service.submit_job(job)
    window._document = doc
    monkeypatch.setattr(window._cluster_runner, "is_submitting", lambda job_id: job_id == job.id)
    requested: list[tuple] = []
    monkeypatch.setattr(window._cluster_runner, "cancel", lambda *args: requested.append(args))

    window._jobs.cancel_jobs([job])

    assert requested == [(job.id, "")]
    assert job_status_word(job) == "submitting"
    assert window.statusBar().currentMessage() == "Cancellation requested for 1 job"
    window.close()


def test_cancel_reports_jobs_the_cluster_was_and_was_not_asked_about_separately(
    main_window, tmp_path, monkeypatch
):
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cancel_mixed.mufinder", "CancelMixed")
    old = datetime.now(UTC) - timedelta(minutes=10)
    jobs = []
    for label in ("m0", "m1", "u0"):
        job = add_dashboard_job(service, doc, label, JobStatus.DEFINED)
        job.execution_target = "SCARF"
        service.submit_job(job)
        job.submitted_at = old
        jobs.append(job)
    for index, job in enumerate(jobs[:2]):
        job.remote_job_id = f"41277{index}"
    window._cluster_runner._connected = True
    window._cluster_runner._connected_cluster_name = "SCARF"
    window._document = doc
    requested: list[tuple] = []
    monkeypatch.setattr(window._cluster_runner, "cancel", lambda *args: requested.append(args))

    window._jobs.cancel_jobs(jobs)

    assert window.statusBar().currentMessage() == (
        "Cancellation requested for 2 jobs; cancelled 1 job in the project (SCARF was not "
        "contacted for jobs it could not be asked about)"
    )
    assert requested == [(jobs[0].id, "412770"), (jobs[1].id, "412771")]
    window.close()


def test_a_finished_cluster_job_whose_result_cannot_be_recorded_fails_with_the_reason(
    main_window, tmp_path, monkeypatch
):
    # If recording the result raises, the job fails with the reason instead of
    # staying active.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "record_fails.mufinder", "RecordFails")
    job = add_dashboard_job(service, doc, "f0", JobStatus.DEFINED)
    job.execution_target = "SCARF"
    job.remote_job_id = "4127731"
    job.remote_directory = "/remote/f0"
    service.submit_job(job)
    service.mark_job_active(job, JobActivity.RUNNING)
    service.mark_job_active(job, JobActivity.RETRIEVING)
    window._document = doc
    window._refresh_jobs_table()

    def _raise(*_args, **_kwargs):
        raise RuntimeError("unexpected output layout")

    monkeypatch.setattr(jobs_controller_module, "merge_controller_result", _raise)

    window._jobs._on_cluster_job_checked(
        job.id,
        JobEvidence(
            checked_at=datetime.now(UTC),
            scheduler=SchedulerState.COMPLETED,
            output=OutputState.COMPLETE,
            parsed=jobs_controller_module.CastepResult(completed=True, total_energy=-1.0),
            remote_job_id="4127731",
            remote_directory="/remote/f0",
        ),
    )

    assert job_status_word(job) == "failed"
    assert job.error_message == (
        "The job's output was retrieved, but MuFinder could not record the result: "
        "unexpected output layout"
    )
    assert doc.results == {}
    assert all(run.result_id is None for run in doc.runs.values())
    # The cluster wrapper must not report the opposite outcome afterwards.
    assert window.statusBar().currentMessage() == "Could not record the result of f0"
    window.close()


def test_an_error_after_a_job_completed_keeps_its_result(main_window, tmp_path, monkeypatch):
    # Whatever raises once the result is stored must not strip it from a
    # completed job.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "late_error.mufinder", "LateError")
    job = add_dashboard_job(service, doc, "c0", JobStatus.DEFINED)
    service.submit_job(job)
    service.mark_job_active(job, JobActivity.RUNNING)
    window._document = doc

    def _raise(*_args, **_kwargs):
        raise RuntimeError("flush failed")

    monkeypatch.setattr(window, "_request_dashboard_flush", _raise)

    window._jobs._on_local_job_finished(
        job.id, jobs_controller_module.CastepResult(completed=True, total_energy=-1.0)
    )

    assert job.status == JobStatus.COMPLETED
    assert len(doc.results) == 1
    window.close()


def test_dashboard_check_now_stays_on_the_row_whether_or_not_the_cluster_is_connected(
    main_window, tmp_path
):
    # Pressing it on a closed connection connects first, so it never goes away.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "connects.mufinder", "Connects")
    orphaned_cluster_job(service, doc)
    window._document = doc
    window._refresh_jobs_table()
    assert row_actions(window) == ("", "Check now", "✕")

    _connect_to_scarf(window)
    window._jobs._on_cluster_connection_status_changed(True)
    assert row_actions(window) == ("", "Check now", "✕")

    window._cluster_runner._connected = False
    window._jobs._on_cluster_connection_status_changed(False)
    assert row_actions(window) == ("", "Check now", "✕")
    window.close()


def test_dashboard_check_now_connects_to_the_jobs_cluster_first_and_completes_it(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    job = _ended_with_output_left(window, tmp_path, reason=EndReason.OUTPUT_UNAVAILABLE)
    job.submitted_at = datetime.now(UTC)
    (tmp_path / job.label).mkdir()
    execution = window._job_execution
    monkeypatch.setattr(execution, "_job_output_directory", lambda j: tmp_path / j.label)
    monkeypatch.setattr(execution, "_job_seed_name", lambda j: j.label)
    scarf = ClusterConfig(name="SCARF", hostname="scarf.invalid", auth_method="key")
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [scarf])

    ssh = FakeSSH()
    ssh.remote_files[f"/remote/job/{job.label}.castep"] = _COMPLETE_OUTPUT.read_bytes()

    class _Scheduler(SlurmScheduler):
        def status(self, _ssh, job_ids):
            return {}

        def history(self, _ssh, job_ids):
            return {i: SchedulerJobInfo(job_id=i, state="COMPLETED") for i in job_ids}

    runner = window._cluster_runner._runner
    connected: list[str] = []

    def connect(config, password=None):  # runs on the cluster worker thread
        connected.append(config.name)
        runner._ssh = ssh
        runner._scheduler = _Scheduler()
        runner._config = config
        runner.connection_status_changed.emit(True)

    monkeypatch.setattr(runner, "connect_to_cluster", connect)
    assert not window._cluster_runner.is_connected
    assert row_actions(window)[1] == "Check now"

    look_button(window).click()

    assert connected == ["SCARF"]
    assert pump_until(lambda: job.status == JobStatus.COMPLETED, timeout=5.0)


def test_dashboard_check_now_that_cannot_connect_says_why_and_checks_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    job = _ended_with_output_left(window, tmp_path, reason=EndReason.INTERRUPTED)
    # SCARF is not configured, so there is nothing to connect to.
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [])
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        jobs_controller_module,
        "show_detailed_warning",
        lambda _parent, title, text, **_kwargs: warnings.append((title, text)),
    )
    infos: list[str] = []
    monkeypatch.setattr(
        jobs_controller_module.QMessageBox,
        "information",
        lambda _parent, title, *_args, **_kwargs: infos.append(title),
    )
    checks: list[object] = []
    monkeypatch.setattr(
        window._cluster_runner, "check_jobs", lambda targets, **_kwargs: checks.append(targets)
    )

    look_button(window).click()

    assert warnings == [
        (
            "Check now failed",
            "Some clusters could not be connected to:\n- SCARF: cluster is not configured. "
            "Set it in Setup > HPC Clusters.",
        )
    ]
    assert infos == []
    assert checks == []
    assert window.statusBar().currentMessage() == "Failed to connect to SCARF"
    assert job_status_word(job) == "stopped"


@pytest.mark.parametrize("backend", ["submit_jobs_locally", "submit_jobs_on_cluster"])
def test_a_run_again_plan_needs_the_accept_step_that_applies_it(main_window, backend: str) -> None:
    from mufinder.app.workflow_submission import RunAgainPlan

    with pytest.raises(ValueError, match="on_accepted"):
        getattr(main_window._jobs, backend)([make_job()], run_again=RunAgainPlan())
