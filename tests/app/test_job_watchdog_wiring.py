"""Tests for the cluster tracked-id mirror that feeds the job watchdog (REC #7).

The ``ClusterController`` maintains a GUI-thread mirror of the runner's active
tracked-job ids (``tracked_active_job_ids``). These tests exercise how that
mirror is maintained from the signals and calls the controller already sees,
using the same fake-runner/worker-thread harness as ``test_cluster_controller``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from mufinder.app import cluster_controller
from mufinder.app.cluster_controller import ClusterController
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun
from mufinder.services.project_service import ProjectService


def _process_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        QApplication.processEvents()
        time.sleep(0.005)
    return predicate()


class _FakeRunner(QObject):
    """Minimal runner stand-in that lets tests drive status_changed directly."""

    job_started = Signal(object)
    job_finished = Signal(object, object)
    job_failed = Signal(object, str)
    job_cancelled = Signal(object)
    status_changed = Signal(object, object)
    connection_status_changed = Signal(bool)
    queue_checked = Signal(object)
    poll_failed = Signal(str)

    def __init__(self, *, settings: object = None) -> None:
        super().__init__()
        self.settings = settings

    def track_existing_job(self, **kwargs) -> None:
        return None

    def clear_tracked_jobs(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


@pytest.fixture
def controller(qapp, monkeypatch) -> Iterator[tuple[ClusterController, _FakeRunner]]:
    monkeypatch.setattr(cluster_controller, "ClusterJobRunner", _FakeRunner)
    ctrl = ClusterController(settings=None)
    fake = ctrl._runner  # type: ignore[assignment]
    try:
        yield ctrl, fake
    finally:
        ctrl.shutdown()


def test_mirror_starts_empty(controller) -> None:
    ctrl, _fake = controller
    assert ctrl.tracked_active_job_ids() == frozenset()


def test_active_status_adds_and_terminal_status_removes(controller) -> None:
    ctrl, fake = controller
    job_id = uuid4()

    # A queued notification (as emitted by submit / the poll loop) adds the job.
    fake.status_changed.emit(job_id, JobStatus.QUEUED.value)
    assert _process_until(lambda: ctrl.tracked_active_job_ids() == frozenset({job_id}))

    # A running notification keeps it tracked.
    fake.status_changed.emit(job_id, JobStatus.RUNNING.value)
    assert _process_until(lambda: ctrl.tracked_active_job_ids() == frozenset({job_id}))

    # A terminal notification removes it.
    fake.status_changed.emit(job_id, JobStatus.COMPLETED.value)
    assert _process_until(lambda: ctrl.tracked_active_job_ids() == frozenset())


@pytest.mark.parametrize(
    "terminal",
    [
        JobStatus.COMPLETED.value,
        JobStatus.FAILED.value,
        JobStatus.STOPPED.value,
        JobStatus.CANCELLED.value,
    ],
)
def test_each_terminal_status_removes(controller, terminal: str) -> None:
    ctrl, fake = controller
    job_id = uuid4()
    fake.status_changed.emit(job_id, JobStatus.RUNNING.value)
    assert _process_until(lambda: job_id in ctrl.tracked_active_job_ids())
    fake.status_changed.emit(job_id, terminal)
    assert _process_until(lambda: job_id not in ctrl.tracked_active_job_ids())


def test_track_existing_job_adds_active_job_to_mirror(controller) -> None:
    ctrl, _fake = controller
    job_id = uuid4()
    ctrl.track_existing_job(
        job_id=job_id,
        remote_job_id="123",
        remote_dir="/scratch/x",
        seed_name="x",
        local_output_dir="/tmp/x",
        cluster_name="scarf",
        status=JobStatus.RUNNING,
    )
    # Added eagerly on the GUI thread (the runner emits nothing on re-track).
    assert ctrl.tracked_active_job_ids() == frozenset({job_id})


def test_track_existing_job_ignores_terminal_status(controller) -> None:
    ctrl, _fake = controller
    job_id = uuid4()
    ctrl.track_existing_job(
        job_id=job_id,
        remote_job_id="123",
        remote_dir="/scratch/x",
        seed_name="x",
        local_output_dir="/tmp/x",
        cluster_name="scarf",
        status=JobStatus.STOPPED,
    )
    assert ctrl.tracked_active_job_ids() == frozenset()


def test_clear_tracked_jobs_empties_mirror(controller) -> None:
    ctrl, fake = controller
    job_id = uuid4()
    fake.status_changed.emit(job_id, JobStatus.RUNNING.value)
    assert _process_until(lambda: job_id in ctrl.tracked_active_job_ids())

    ctrl.clear_tracked_jobs()
    assert ctrl.tracked_active_job_ids() == frozenset()


def test_mirror_survives_disconnect(controller) -> None:
    # The runner keeps its tracked jobs across a disconnect, so the GUI mirror
    # must too: clearing it here would make the watchdog stop live jobs after a
    # reconnect. The watchdog never flags cluster jobs while disconnected.
    ctrl, fake = controller
    job_id = uuid4()
    fake.status_changed.emit(job_id, JobStatus.RUNNING.value)
    assert _process_until(lambda: job_id in ctrl.tracked_active_job_ids())

    fake.connection_status_changed.emit(False)
    assert _process_until(lambda: ctrl.is_connected is False)
    assert ctrl.tracked_active_job_ids() == frozenset({job_id})


# ----------------------------------------------------------------------------
# MainWindow handler wiring (timer -> snapshots -> evaluate -> stop_stuck_job)
# ----------------------------------------------------------------------------
def _running_local_job(window: MainWindow, service: ProjectService, document) -> object:
    job = service.add_job(
        document,
        label="stuck-local",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
    )
    # Drive it to RUNNING through the primitives, exactly as a real submission
    # would, then register an active run so close-out has something to touch.
    service.submit_job(job)
    service.start_job(job)
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.RUNNING)
    document.runs[run.id] = run
    window._active_run_by_job[job.id] = run.id
    return job


def test_watchdog_handler_stops_confirmed_stuck_local_job(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    document = service.create_project(tmp_path / "watchdog_wiring.mufinder", "Watchdog Wiring")
    window._document = document
    window._refresh_ui_after_project_change()

    job = _running_local_job(window, service, document)
    # No local runner entry and no cluster tracking: the job is genuinely stuck.

    # First tick: two-tick gate means nothing is stopped yet.
    window._on_job_watchdog_tick()
    assert document.jobs[job.id].status == JobStatus.RUNNING

    # Second consecutive tick: confirmed and stopped.
    window._on_job_watchdog_tick()
    stopped = document.jobs[job.id]
    assert stopped.status == JobStatus.STOPPED
    assert stopped.last_status_source == "watchdog"
    assert stopped.last_status_message
    # The active run was closed out and unmapped, mirroring the walltime-stop path.
    assert job.id not in window._active_run_by_job
    stopped_runs = [run for run in document.runs.values() if run.job_id == job.id]
    assert stopped_runs and all(run.status == JobStatus.STOPPED for run in stopped_runs)
    window.close()


def test_watchdog_handler_noop_without_active_jobs(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    document = service.create_project(tmp_path / "watchdog_empty.mufinder", "Watchdog Empty")
    window._document = document
    window._refresh_ui_after_project_change()

    # No active jobs: the handler returns early and does not raise.
    window._on_job_watchdog_tick()
    window._on_job_watchdog_tick()
    window.close()
