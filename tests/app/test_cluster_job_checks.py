"""The main window asks the cluster about its jobs and applies what comes back.

Which jobs a check covers is decided in ``JobExecutionController`` and tested
there. These tests pin when the main window asks (project open, connect, the
once-a-minute tick, the queue button) and that evidence reaching it changes the
job.
"""

from __future__ import annotations

from mufinder.services.job_monitoring import (
    RESULT_NOT_RECORDED_PREFIX,
    cancel_not_sent_to_cluster,
    job_status_word,
    unretrieved_output_note,
)
from pathlib import Path
from types import SimpleNamespace

import pytest

from mufinder.app.main_window import MainWindow
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import EndReason, JobActivity, JobStatus, WorkflowStage
from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.models import Job, utc_now
from mufinder.services.cluster_runner import ClusterJobRunner, QueueCheckSnapshot
from mufinder.services.job_reconcile import (
    NO_JOB_ID_MARKER,
    JobEvidence,
    OutputState,
    SchedulerState,
)
from mufinder.services.project_service import ProjectService
from mufinder.services.scheduler import SchedulerJobInfo, SlurmScheduler
from tests.support.cluster_ssh import FakeSSH
from tests.support.dashboard import job_row, row_actions
from tests.support.jobs import set_job_state

_REMOTE_DIR = "/remote/work/job"


def _add_cluster_job(
    document,
    service: ProjectService,
    *,
    label: str,
    status: JobStatus,
    execution_target: str = "SCARF",
    remote_job_id: str | None = "12345",
) -> Job:
    job = service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
        execution_target=execution_target,
    )
    set_job_state(job, status)
    job.remote_job_id = remote_job_id
    job.remote_directory = _REMOTE_DIR
    return job


@pytest.fixture
def checking(main_window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return ``(service, document, window, checks)`` connected to SCARF.

    ``checks`` records each request the window sends to the cluster worker as
    ``(labels, skip_if_busy)``, or ``(labels, skip_if_busy, "refresh")`` for a
    Refresh Queue pass; nothing reaches a real cluster.
    """

    window = main_window
    service = ProjectService()
    document = service.create_project(tmp_path / "checks.mufinder", "Checks")
    window._document = document
    checks: list[tuple] = []

    def record(targets, *, skip_if_busy: bool = False, refresh: bool = False) -> bool:
        labels = [target.seed_name for target in targets]
        checks.append((labels, skip_if_busy, "refresh") if refresh else (labels, skip_if_busy))
        return True

    monkeypatch.setattr(window._cluster_runner, "check_jobs", record)
    execution = window._job_execution
    monkeypatch.setattr(execution, "_job_output_directory", lambda job: tmp_path / job.label)
    monkeypatch.setattr(execution, "_job_seed_name", lambda job: job.label)
    window._cluster_runner._connected = True
    window._cluster_runner._connected_cluster_name = "SCARF"
    return service, document, window, checks


def _evidence(job: Job, scheduler: SchedulerState, output=OutputState.NOT_LOOKED, **fields):
    return JobEvidence(
        checked_at=utc_now(),
        scheduler=scheduler,
        output=output,
        remote_job_id=job.remote_job_id or "",
        remote_directory=job.remote_directory or "",
        **fields,
    )


def test_opening_a_project_asks_about_its_active_cluster_jobs(checking) -> None:
    service, document, window, checks = checking
    _add_cluster_job(document, service, label="queued", status="queued")
    _add_cluster_job(
        document,
        service,
        label="no_id",
        status="submitting",
        remote_job_id=None,
    )
    _add_cluster_job(document, service, label="failed", status="failed")
    _add_cluster_job(
        document,
        service,
        label="elsewhere",
        status="running",
        execution_target="PSI",
    )

    window._refresh_ui_after_project_change()

    # The check that opens a project gives way to one already on the worker. A
    # job with no scheduler job ID is looked at only when the user asks.
    assert checks == [(["queued"], True)]

    no_id = next(job for job in document.jobs.values() if job.label == "no_id")
    window._jobs._on_check_now_from_dashboard([no_id.id])
    assert checks[-1] == (["no_id"], False)


def test_connecting_asks_about_the_clusters_jobs(checking) -> None:
    service, document, window, checks = checking
    _add_cluster_job(document, service, label="running", status="running")

    window._jobs._on_cluster_connection_status_changed(True)

    assert checks == [(["running"], True)]


def test_disconnecting_asks_nothing(checking) -> None:
    service, document, window, checks = checking
    _add_cluster_job(document, service, label="running", status="running")
    window._cluster_runner._connected = False

    window._jobs._on_cluster_connection_status_changed(False)
    window._jobs.check_jobs()

    assert checks == []


def test_the_minute_tick_skips_a_check_while_one_is_running(checking) -> None:
    service, document, window, checks = checking
    _add_cluster_job(document, service, label="running", status="running")

    window._jobs.check_jobs()

    assert checks == [(["running"], True)]


def test_the_queue_button_asks_about_ended_jobs_too_and_reports_its_own_pass(checking) -> None:
    service, document, window, checks = checking
    job = _add_cluster_job(document, service, label="running", status="running")
    ended = _add_cluster_job(document, service, label="failed", status="failed")
    window._refresh_jobs_table()
    window._jobs.sync_cluster_status_widgets()
    window._jobs._cluster_queue_checked_label.setText("Queue last checked: never")

    window._jobs._on_refresh_cluster_queue()

    # Queued behind any check on the worker rather than dropped, and marked as a refresh.
    assert checks == [(["running", "failed"], False, "refresh")]
    # A minute check that ends first is not taken for the refresh.
    window._jobs._on_cluster_queue_checked(
        QueueCheckSnapshot(checked_at=utc_now(), elapsed_by_job_id={}, job_ids=(job.id,))
    )
    assert not window.statusBar().currentMessage().startswith("Refreshed queue")
    window._jobs._on_cluster_queue_checked(
        QueueCheckSnapshot(
            checked_at=utc_now(),
            elapsed_by_job_id={},
            job_ids=(job.id, ended.id),
            refresh=True,
        )
    )
    assert window.statusBar().currentMessage().startswith("Refreshed queue: checked 2 jobs at ")
    assert window._jobs._cluster_queue_checked_label.text() != "Queue last checked: never"


def test_a_pass_over_another_projects_jobs_is_ignored(checking) -> None:
    service, document, window, _checks = checking
    job = _add_cluster_job(document, service, label="running", status="running")
    other = Job(label="other")
    before = window._jobs._cluster_queue_checked_label.text()

    window._jobs._on_cluster_queue_checked(
        QueueCheckSnapshot(checked_at=utc_now(), elapsed_by_job_id={}, job_ids=(other.id,))
    )
    assert window._jobs._cluster_queue_checked_label.text() == before

    window._jobs._on_cluster_queue_checked(
        QueueCheckSnapshot(checked_at=utc_now(), elapsed_by_job_id={}, job_ids=(job.id,))
    )
    assert window._jobs._cluster_queue_checked_label.text() != before


def test_evidence_moves_the_job_and_says_so(checking) -> None:
    service, document, window, _checks = checking
    job = _add_cluster_job(document, service, label="queued", status="queued")

    window._jobs._on_cluster_job_checked(job.id, _evidence(job, SchedulerState.RUNNING))

    assert job_status_word(job) == "running"
    assert window._dashboard_flush_pending
    # The status bar speaks once, when the check ends.
    assert window.statusBar().currentMessage() != "Cluster job started running: queued"
    window._jobs._on_cluster_queue_checked(
        QueueCheckSnapshot(checked_at=utc_now(), elapsed_by_job_id={}, job_ids=(job.id,))
    )
    assert window.statusBar().currentMessage() == "Cluster job started running: queued"


def test_evidence_of_complete_output_records_the_result(checking) -> None:
    service, document, window, _checks = checking
    job = _add_cluster_job(document, service, label="done", status="running")
    evidence = _evidence(
        job,
        SchedulerState.COMPLETED,
        OutputState.COMPLETE,
        parsed=CastepResult(completed=True, total_energy=-12.5),
    )

    window._jobs._on_cluster_job_checked(job.id, evidence)

    assert job.status == JobStatus.COMPLETED
    run = next(run for run in document.runs.values() if run.job_id == job.id)
    assert document.results[run.result_id].data["total_energy"] == -12.5
    assert window.statusBar().currentMessage() == "Completed cluster job done"


def test_a_job_the_scheduler_lost_stays_active_with_a_lost_track_marker(checking) -> None:
    service, document, window, _checks = checking
    job = _add_cluster_job(document, service, label="lost", status="running")

    window._jobs._on_cluster_job_checked(
        job.id, _evidence(job, SchedulerState.NOT_LISTED, OutputState.ABSENT)
    )

    assert job_status_word(job) == "running"
    assert job.end_reason is None
    marker = window._job_execution.staleness(job, now=utc_now())
    assert marker is not None and marker.startswith("lost track")


def test_a_scheduler_failure_fails_the_job(checking) -> None:
    service, document, window, _checks = checking
    job = _add_cluster_job(document, service, label="oom", status="running")

    window._jobs._on_cluster_job_checked(
        job.id,
        _evidence(
            job, SchedulerState.FAILED, OutputState.INCOMPLETE, scheduler_detail="OUT_OF_MEMORY"
        ),
    )

    assert job_status_word(job) == "failed"
    assert job.end_reason == EndReason.OUT_OF_MEMORY
    window._jobs._on_cluster_queue_checked(
        QueueCheckSnapshot(checked_at=utc_now(), elapsed_by_job_id={}, job_ids=(job.id,))
    )
    assert window.statusBar().currentMessage() == "Cluster job failed (memory): oom"


def test_evidence_with_no_open_project_is_ignored(main_window: MainWindow) -> None:
    job = Job(
        label="gone",
        execution_target="SCARF",
        status=JobStatus.ACTIVE,
        activity=JobActivity.RUNNING,
    )

    main_window._jobs._on_cluster_job_checked(job.id, _evidence(job, SchedulerState.RUNNING))


def test_a_check_that_changes_several_jobs_ends_with_one_summary(checking) -> None:
    service, document, window, _checks = checking
    done = _add_cluster_job(document, service, label="done", status="running")
    timed_out = _add_cluster_job(document, service, label="late", status="queued")
    same = _add_cluster_job(document, service, label="same", status="running")
    window._refresh_jobs_table()
    window._jobs.sync_cluster_status_widgets()
    window._jobs._on_refresh_cluster_queue()

    window._jobs._on_cluster_job_checked(
        done.id,
        _evidence(
            done,
            SchedulerState.COMPLETED,
            OutputState.COMPLETE,
            parsed=CastepResult(completed=True, total_energy=-1.0),
        ),
    )
    window._jobs._on_cluster_job_checked(
        timed_out.id, _evidence(timed_out, SchedulerState.TIMED_OUT, OutputState.ABSENT)
    )
    window._jobs._on_cluster_job_checked(same.id, _evidence(same, SchedulerState.RUNNING))
    window._jobs._on_cluster_queue_checked(
        QueueCheckSnapshot(
            checked_at=utc_now(),
            elapsed_by_job_id={},
            job_ids=(done.id, timed_out.id, same.id),
        )
    )

    assert window.statusBar().currentMessage() == (
        "Checked 3 cluster jobs: 1 completed, 1 stopped at its walltime limit"
    )


def test_a_failed_check_discards_what_it_had_changed(checking, monkeypatch) -> None:
    service, document, window, _checks = checking
    job = _add_cluster_job(document, service, label="queued", status="queued")
    monkeypatch.setattr(
        "mufinder.app.main_window.show_detailed_warning", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "mufinder.app.jobs_controller.show_detailed_warning", lambda *args, **kwargs: None
    )
    window._jobs._on_cluster_job_checked(job.id, _evidence(job, SchedulerState.RUNNING))

    window._jobs._on_cluster_poll_failed("Queue polling failed for SCARF: boom")

    assert window._job_execution.take_check_summary(1) == ""


# --- Refresh Queue against a simulated cluster -----------------------------------

_COMPLETE_OUTPUT = Path(__file__).parents[1] / "fixtures" / "uep" / "Si2.castep"


class _SimulatedScheduler(SlurmScheduler):
    """A SLURM backend answering from ``queue`` and ``accounting`` and recording each call."""

    def __init__(self) -> None:
        super().__init__()
        self.queue: dict[str, str] = {}
        self.accounting: dict[str, str] = {}
        self.calls: list[tuple[str, list[str]]] = []

    def status(self, _ssh, job_ids):
        self.calls.append(("squeue", list(job_ids)))
        return {
            job_id: SchedulerJobInfo(job_id=job_id, state=self.queue[job_id], name="job")
            for job_id in job_ids
            if job_id in self.queue
        }

    def history(self, _ssh, job_ids):
        self.calls.append(("sacct", list(job_ids)))
        return {
            job_id: SchedulerJobInfo(job_id=job_id, state=self.accounting[job_id], name="job")
            for job_id in job_ids
            if job_id in self.accounting
        }


@pytest.fixture
def simulated(main_window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return ``(service, document, window, cluster)`` wired to a simulated SCARF.

    Checks run a real ``ClusterJobRunner`` on the GUI thread against a fake
    scheduler and remote filesystem, and its signals reach the window as the
    worker's would. ``cluster`` has ``scheduler``, ``ssh`` and ``passes``, one
    ``(labels, skip_if_busy, refresh)`` per request.
    """

    window = main_window
    service = ProjectService()
    document = service.create_project(tmp_path / "refresh.mufinder", "Refresh")
    window._document = document
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _SimulatedScheduler()
    runner._config = ClusterConfig(name="SCARF", hostname="scarf.invalid")
    runner.job_checked.connect(window._jobs._on_cluster_job_checked)
    runner.queue_checked.connect(window._jobs._on_cluster_queue_checked)
    passes: list[tuple[list[str], bool, bool]] = []

    def check(targets, *, skip_if_busy: bool = False, refresh: bool = False) -> bool:
        passes.append(([target.seed_name for target in targets], skip_if_busy, refresh))
        runner.check_jobs(list(targets), refresh=refresh)
        return True

    monkeypatch.setattr(window._cluster_runner, "check_jobs", check)
    execution = window._job_execution
    monkeypatch.setattr(execution, "_job_output_directory", lambda job: tmp_path / job.label)
    monkeypatch.setattr(execution, "_job_seed_name", lambda job: job.label)
    window._cluster_runner._connected = True
    window._cluster_runner._connected_cluster_name = "SCARF"
    cluster = SimpleNamespace(scheduler=runner._scheduler, ssh=runner._ssh, passes=passes)
    return service, document, window, cluster


def _on_simulated_cluster(
    document,
    service: ProjectService,
    tmp_path: Path,
    *,
    label: str,
    status: str,
    remote_job_id: str | None,
    reason: EndReason | None = None,
    source: str | None = None,
) -> Job:
    job = _add_cluster_job(
        document, service, label=label, status=status, remote_job_id=remote_job_id
    )
    job.remote_directory = f"/remote/{label}"
    if reason is not None:
        job.end_reason = reason
    job.last_status_source = source
    job.submitted_at = utc_now()
    (tmp_path / label).mkdir(exist_ok=True)
    return job


def _put_complete_output(cluster, job: Job) -> None:
    cluster.ssh.remote_files[f"{job.remote_directory}/{job.label}.castep"] = (
        _COMPLETE_OUTPUT.read_bytes()
    )


def _downloads_from(cluster, job: Job) -> list[str]:
    folder = f"{job.remote_directory}/"
    return [
        path for verb, path in cluster.ssh.commands if verb == "get" and path.startswith(folder)
    ]


def _press_refresh_queue(window: MainWindow) -> None:
    window._refresh_jobs_table()
    window._jobs.sync_cluster_status_widgets()
    window._jobs._on_refresh_cluster_queue()


def test_refresh_completes_an_ended_job_the_minute_check_leaves_alone(
    simulated, tmp_path: Path
) -> None:
    service, document, window, cluster = simulated
    running = _on_simulated_cluster(
        document, service, tmp_path, label="running", status="running", remote_job_id="100"
    )
    unavailable = _on_simulated_cluster(
        document,
        service,
        tmp_path,
        label="unavailable",
        status="failed",
        remote_job_id="200",
        reason=EndReason.OUTPUT_UNAVAILABLE,
        source="retrieval",
    )
    cluster.scheduler.queue["100"] = "R"
    cluster.scheduler.accounting["200"] = "COMPLETED"
    _put_complete_output(cluster, unavailable)

    window._jobs.check_jobs()

    assert cluster.passes == [(["running"], True, False)]
    assert unavailable.status == JobStatus.ENDED
    assert _downloads_from(cluster, unavailable) == []

    _press_refresh_queue(window)

    assert cluster.passes[-1] == (["running", "unavailable"], False, True)
    assert unavailable.status == JobStatus.COMPLETED
    run = next(
        run for run in document.runs.values() if run.job_id == unavailable.id and run.result_id
    )
    assert run.result_id in document.results
    assert job_status_word(running) == "running"
    assert window.statusBar().currentMessage() == "Completed cluster job unavailable"


def test_refresh_treats_a_job_the_user_cancelled_like_any_other_ended_job(
    simulated, tmp_path: Path
) -> None:
    service, document, window, cluster = simulated
    _on_simulated_cluster(
        document, service, tmp_path, label="running", status="running", remote_job_id="100"
    )
    listed = _on_simulated_cluster(
        document, service, tmp_path, label="listed", status="cancelled", remote_job_id="300"
    )
    finished = _on_simulated_cluster(
        document, service, tmp_path, label="finished", status="cancelled", remote_job_id="301"
    )
    cluster.scheduler.queue.update({"100": "R", "300": "R"})
    cluster.scheduler.accounting["301"] = "CANCELLED"
    _put_complete_output(cluster, finished)

    _press_refresh_queue(window)

    assert job_status_word(listed) == "running"
    assert finished.status == JobStatus.COMPLETED
    assert window.statusBar().currentMessage() == (
        "Checked 3 cluster jobs: 1 started running, 1 completed"
    )


def test_refresh_asks_the_scheduler_but_does_not_download_output_already_judged(
    simulated, tmp_path: Path
) -> None:
    service, document, window, cluster = simulated
    _on_simulated_cluster(
        document, service, tmp_path, label="running", status="running", remote_job_id="100"
    )
    judged = _on_simulated_cluster(
        document,
        service,
        tmp_path,
        label="judged",
        status="failed",
        remote_job_id="400",
        reason=EndReason.CASTEP_ERROR,
        source="retrieval",
    )
    requeued = _on_simulated_cluster(
        document,
        service,
        tmp_path,
        label="requeued",
        status="stopped",
        remote_job_id="401",
        source="retrieval",
    )
    cluster.scheduler.queue.update({"100": "R", "401": "PD"})
    cluster.scheduler.accounting["400"] = "COMPLETED"
    # Whatever the folder holds now, a verdict was already reached on it.
    _put_complete_output(cluster, judged)

    _press_refresh_queue(window)

    assert cluster.scheduler.calls == [
        ("squeue", ["100", "400", "401"]),
        ("sacct", ["400"]),
    ]
    assert _downloads_from(cluster, judged) == []
    assert (job_status_word(judged), judged.end_reason) == ("failed", EndReason.CASTEP_ERROR)
    # Its scheduler state still counts: listed again, it is queued again.
    assert job_status_word(requeued) == "queued"


def test_refresh_completes_a_dismissed_job_and_brings_a_listed_one_back(
    simulated, tmp_path: Path
) -> None:
    service, document, window, cluster = simulated
    _on_simulated_cluster(
        document, service, tmp_path, label="running", status="running", remote_job_id="100"
    )
    retrievable = _on_simulated_cluster(
        document,
        service,
        tmp_path,
        label="retrievable",
        status="failed",
        remote_job_id="500",
        reason=EndReason.INTERRUPTED,
    )
    relisted = _on_simulated_cluster(
        document,
        service,
        tmp_path,
        label="relisted",
        status="failed",
        remote_job_id="501",
        reason=EndReason.OUTPUT_UNAVAILABLE,
    )
    for job in (retrievable, relisted):
        service.dismiss_job(job)
    cluster.scheduler.queue.update({"100": "R", "501": "R"})
    cluster.scheduler.accounting["500"] = "COMPLETED"
    _put_complete_output(cluster, retrievable)

    _press_refresh_queue(window)
    window._flush_dashboard()

    assert retrievable.status == JobStatus.COMPLETED and retrievable.dismissed_at is None
    assert job_status_word(relisted) == "running" and relisted.dismissed_at is None
    dashboard = window._jobs._deps.job_dashboard()
    listed = {dashboard.job_id_at_row(row) for row in range(dashboard.table.rowCount())}
    assert relisted.id in listed


def test_refresh_never_ends_a_job_for_want_of_evidence(simulated, tmp_path: Path) -> None:
    service, document, window, cluster = simulated
    lost = _on_simulated_cluster(
        document, service, tmp_path, label="lost", status="running", remote_job_id="600"
    )
    no_id = _on_simulated_cluster(
        document, service, tmp_path, label="no_id", status="running", remote_job_id=None
    )

    _press_refresh_queue(window)

    # Listed nowhere with nothing in its folder, and with no ID to ask about.
    assert (job_status_word(lost), job_status_word(no_id)) == ("running", "running")
    assert cluster.passes[-1] == (["lost", "no_id"], False, True)
    staleness = window._job_execution.staleness
    assert staleness(lost, now=utc_now()).startswith("lost track")
    assert staleness(no_id, now=utc_now()) == NO_JOB_ID_MARKER
    assert window.statusBar().currentMessage().startswith("Refreshed queue: checked 2 jobs at ")


# --- Jobs whose state on the cluster is still to be found out ----------------------


def _status_tooltip(window: MainWindow, job: Job) -> str:
    dashboard = window._job_dashboard
    return dashboard.table.item(job_row(window, job.id), dashboard.STATUS_COLUMN).toolTip()


def _refresh_queue_shown(window: MainWindow) -> bool:
    window._jobs.sync_cluster_status_widgets()
    return not window._jobs._cluster_refresh_button.isHidden()


def _cancel_while_disconnected(window: MainWindow, job: Job) -> None:
    window._cluster_runner._connected = False
    window._jobs.cancel_jobs([job])
    window._refresh_jobs_table()


def _reconnect(window: MainWindow) -> None:
    window._cluster_runner._connected = True
    window._refresh_jobs_table()


def test_a_cancel_the_cluster_never_received_offers_check_now(simulated, tmp_path: Path) -> None:
    service, document, window, _cluster = simulated
    job = _on_simulated_cluster(
        document, service, tmp_path, label="unsent", status="running", remote_job_id="700"
    )

    _cancel_while_disconnected(window, job)

    assert job_status_word(job) == "cancelled"
    assert row_actions(window, job_row(window, job.id)) == ("", "Check now", "✕")
    assert (
        "Cancelled in the project only: SCARF was not told, so the job may still be "
        "running there. Use 'Check now' to find out."
    ) in _status_tooltip(window, job)
    # The project has nothing else on a cluster, yet Refresh Queue stays.
    assert _refresh_queue_shown(window)
    # The dismiss confirmation warns about it too.
    assert unretrieved_output_note([job]).startswith("Its output may still be on SCARF")

    # Saved and opened again, it still asks to be checked.
    service.save_project(document)
    reopened = ProjectService().load_project(document.project_root).jobs[job.id]
    assert cancel_not_sent_to_cluster(reopened)


@pytest.mark.parametrize("press", ["check_now", "refresh_queue"])
def test_a_check_finds_an_unsent_cancel_still_running_so_cancel_reaches_the_scheduler(
    simulated, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, press: str
) -> None:
    service, document, window, cluster = simulated
    job = _on_simulated_cluster(
        document, service, tmp_path, label="unsent", status="running", remote_job_id="700"
    )
    _cancel_while_disconnected(window, job)
    _reconnect(window)
    cluster.scheduler.queue["700"] = "R"

    if press == "check_now":
        window._jobs._on_check_now_from_dashboard([job.id])
    else:
        _press_refresh_queue(window)
    window._refresh_jobs_table()

    assert job_status_word(job) == "running"
    assert not cancel_not_sent_to_cluster(job)
    assert row_actions(window, job_row(window, job.id)) == ("Cancel", "", "")

    sent: list[tuple] = []
    monkeypatch.setattr(window._cluster_runner, "cancel", lambda *args: sent.append(args))
    window._jobs.cancel_jobs([job])

    assert sent == [(job.id, "700")]


@pytest.mark.parametrize("press", ["check_now", "refresh_queue"])
def test_a_check_the_scheduler_answers_settles_an_unsent_cancel(
    simulated, tmp_path: Path, press: str
) -> None:
    service, document, window, cluster = simulated
    job = _on_simulated_cluster(
        document, service, tmp_path, label="unsent", status="running", remote_job_id="700"
    )
    _cancel_while_disconnected(window, job)
    _reconnect(window)
    cluster.scheduler.accounting["700"] = "CANCELLED"

    if press == "check_now":
        window._jobs._on_check_now_from_dashboard([job.id])
    else:
        _press_refresh_queue(window)
    window._refresh_jobs_table()

    assert job_status_word(job) == "cancelled"
    assert not cancel_not_sent_to_cluster(job)
    assert row_actions(window, job_row(window, job.id)) == ("", "", "✕")
    assert "was not told" not in _status_tooltip(window, job)
    assert not _refresh_queue_shown(window)


def test_a_cancel_the_scheduler_confirmed_never_offers_check_now(
    simulated, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, document, window, _cluster = simulated
    job = _on_simulated_cluster(
        document, service, tmp_path, label="told", status="running", remote_job_id="701"
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "cancel",
        lambda job_id, _remote: window._jobs._on_cluster_job_cancelled(job_id),
    )

    window._jobs.cancel_jobs([job])
    window._refresh_jobs_table()

    assert job_status_word(job) == "cancelled"
    assert not cancel_not_sent_to_cluster(job)
    assert row_actions(window, job_row(window, job.id)) == ("", "", "✕")
    assert not _refresh_queue_shown(window)


def test_a_late_submission_whose_cancel_failed_can_be_found_by_a_check(
    simulated, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, document, window, cluster = simulated
    job = _on_simulated_cluster(
        document, service, tmp_path, label="late", status="submitting", remote_job_id=None
    )
    window._job_execution.cancel_job(job.id)
    monkeypatch.setattr(
        "mufinder.app.jobs_controller.show_detailed_warning", lambda *_args, **_kwargs: None
    )

    window._cluster_runner.abandoned_job_cancel_failed.emit(job.id, "702", "scancel failed")
    window._refresh_jobs_table()

    assert job.remote_job_id == "702"
    assert row_actions(window, job_row(window, job.id)) == ("", "Check now", "✕")
    cluster.scheduler.queue["702"] = "PD"
    window._jobs._on_check_now_from_dashboard([job.id])
    assert job_status_word(job) == "queued"


def _unrecorded(document, service, tmp_path: Path, label: str = "unrecorded") -> Job:
    job = _on_simulated_cluster(
        document,
        service,
        tmp_path,
        label=label,
        status="failed",
        remote_job_id="800",
        reason=EndReason.UNKNOWN,
        source="retrieval",
    )
    job.error_message = f"{RESULT_NOT_RECORDED_PREFIX}: unexpected output layout"
    return job


@pytest.mark.parametrize("press", ["check_now", "refresh_queue"])
def test_a_job_whose_result_was_not_recorded_is_completed_by_a_check(
    simulated, tmp_path: Path, press: str
) -> None:
    service, document, window, cluster = simulated
    job = _unrecorded(document, service, tmp_path)
    cluster.scheduler.accounting["800"] = "COMPLETED"
    _put_complete_output(cluster, job)
    window._refresh_jobs_table()

    assert row_actions(window, job_row(window, job.id)) == ("", "Check now", "✕")
    assert _refresh_queue_shown(window)

    if press == "check_now":
        window._jobs._on_check_now_from_dashboard([job.id])
    else:
        _press_refresh_queue(window)
    window._refresh_jobs_table()

    assert _downloads_from(cluster, job)
    assert job.status == JobStatus.COMPLETED
    assert not _refresh_queue_shown(window)


def test_a_result_that_still_cannot_be_recorded_leaves_the_job_as_it_was(
    simulated, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, document, window, cluster = simulated
    job = _unrecorded(document, service, tmp_path)
    cluster.scheduler.accounting["800"] = "COMPLETED"
    _put_complete_output(cluster, job)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("still broken")

    monkeypatch.setattr("mufinder.app.jobs_controller.merge_controller_result", _raise)
    window._jobs._on_check_now_from_dashboard([job.id])
    window._refresh_jobs_table()

    assert job_status_word(job) == "failed"
    assert job.error_message == f"{RESULT_NOT_RECORDED_PREFIX}: still broken"
    assert row_actions(window, job_row(window, job.id)) == ("", "Check now", "✕")


def test_a_job_with_no_scheduler_job_id_is_cancelled_as_not_sent(simulated, tmp_path: Path) -> None:
    service, document, window, cluster = simulated
    job = _on_simulated_cluster(
        document, service, tmp_path, label="no_id", status="running", remote_job_id=None
    )

    window._jobs.cancel_jobs([job])
    window._refresh_jobs_table()

    assert cancel_not_sent_to_cluster(job)
    assert row_actions(window, job_row(window, job.id)) == ("", "Check now", "✕")
    # With no ID there is no scheduler to ask, and an empty folder does not show
    # the job is not queued: the question stays open, and the tooltip says why.
    window._jobs._on_check_now_from_dashboard([job.id])
    window._refresh_jobs_table()
    assert job_status_word(job) == "cancelled"
    assert row_actions(window, job_row(window, job.id)) == ("", "Check now", "✕")
    assert "SCARF was not told" in _status_tooltip(window, job)
    # Complete output from the run completes it.
    _put_complete_output(cluster, job)
    window._jobs._on_check_now_from_dashboard([job.id])
    assert job.status == JobStatus.COMPLETED


def _cut_off(document, service, tmp_path: Path, window: MainWindow, *, remote_job_id=None) -> Job:
    job = _on_simulated_cluster(
        document, service, tmp_path, label="cut_off", status="submitting", remote_job_id=None
    )
    window._job_execution.cancel_job(job.id)
    window._job_execution.note_submission_cut_off(job.id)
    job.remote_job_id = remote_job_id
    window._refresh_jobs_table()
    return job


_CUT_OFF_REASON = (
    "Its submission was cut off while MuFinder was closing; it may be running on SCARF. "
    "Use 'Check now' to find out."
)


def test_a_cut_off_submission_keeps_saying_why_while_a_check_finds_nothing(
    simulated, tmp_path: Path
) -> None:
    service, document, window, cluster = simulated
    job = _cut_off(document, service, tmp_path, window)
    assert row_actions(window, job_row(window, job.id)) == ("", "Check now", "✕")
    assert _CUT_OFF_REASON in _status_tooltip(window, job)

    window._jobs._on_check_now_from_dashboard([job.id])
    window._refresh_jobs_table()

    # The check ran and read the folder, which holds nothing.
    assert cluster.passes[-1][0] == ["cut_off"]
    assert job.last_checked_at is not None
    assert row_actions(window, job_row(window, job.id)) == ("", "Check now", "✕")
    assert _CUT_OFF_REASON in _status_tooltip(window, job)
    assert _refresh_queue_shown(window)


def test_a_cut_off_submission_the_scheduler_answers_for_is_settled(
    simulated, tmp_path: Path
) -> None:
    # A job whose scheduler job ID became known (a check can then ask about it).
    service, document, window, cluster = simulated
    job = _cut_off(document, service, tmp_path, window, remote_job_id="900")
    cluster.scheduler.accounting["900"] = "CANCELLED"

    window._jobs._on_check_now_from_dashboard([job.id])
    window._refresh_jobs_table()

    assert job_status_word(job) == "cancelled"
    assert row_actions(window, job_row(window, job.id)) == ("", "", "✕")
    assert "Use 'Check now'" not in _status_tooltip(window, job)
    assert not _refresh_queue_shown(window)


def test_a_job_whose_result_was_not_recorded_says_why_it_offers_check_now(
    simulated, tmp_path: Path
) -> None:
    service, document, window, _cluster = simulated
    job = _unrecorded(document, service, tmp_path)
    window._refresh_jobs_table()

    tooltip = _status_tooltip(window, job)

    assert (
        "The job finished, but MuFinder could not record its result. Use 'Check now' to "
        "record it from the output on SCARF."
    ) in tooltip
    assert "Use Inspect" not in tooltip
