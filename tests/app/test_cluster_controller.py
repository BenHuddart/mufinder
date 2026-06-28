"""Tests for the threaded cluster controller facade.

These verify the Phase 1 guarantee: cluster SSH work runs on a dedicated worker
thread (never the GUI thread), while the controller preserves the public surface
that ``main_window`` relies on — forwarded signals, mirrored connection state,
blocking return values, and exception propagation.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QApplication

from mufinder.app import cluster_controller
from mufinder.app.cluster_controller import ClusterController


def _process_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> bool:
    """Pump the GUI event loop until ``predicate`` holds or ``timeout`` elapses."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        QApplication.processEvents()
        time.sleep(0.005)
    return predicate()


class _FakeConfig:
    def __init__(self, name: str = "hpc", username: str = "alice") -> None:
        self.name = name
        self.username = username


class _FakeRunner(QObject):
    """Stand-in for ClusterJobRunner that records the thread each call runs on."""

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
        self.calls: list[tuple] = []
        self.threads: dict[str, QThread] = {}
        self.submit_should_raise = False

    def connect_to_cluster(self, config, password=None, *, credentials=None) -> None:
        self.threads["connect"] = QThread.currentThread()
        self.calls.append(("connect", config.name))
        self.connection_status_changed.emit(True)

    def submit(self, **kwargs) -> str:
        self.threads["submit"] = QThread.currentThread()
        if self.submit_should_raise:
            raise ConnectionError("boom")
        self.calls.append(("submit", kwargs.get("seed_name")))
        return "JOB-1"

    def remote_directory_exists(self, remote_directory: str) -> bool:
        self.threads["exists"] = QThread.currentThread()
        self.calls.append(("exists", remote_directory))
        return True

    def remove_remote_directory(self, remote_directory: str) -> None:
        self.threads["rmtree"] = QThread.currentThread()
        self.calls.append(("rmtree", remote_directory))

    def refresh_queue(self) -> None:
        self.threads["refresh"] = QThread.currentThread()
        self.calls.append(("refresh",))

    def cancel(self, job_id) -> None:
        self.threads["cancel"] = QThread.currentThread()
        self.calls.append(("cancel", job_id))

    def track_existing_job(self, **kwargs) -> None:
        self.threads["track"] = QThread.currentThread()
        self.calls.append(("track", kwargs.get("job_id")))

    def clear_tracked_jobs(self) -> None:
        self.threads["clear"] = QThread.currentThread()
        self.calls.append(("clear",))

    def shutdown(self) -> None:
        self.calls.append(("shutdown",))

    @staticmethod
    def remote_job_directory(*, working_directory: str, seed_name: str) -> str:
        return f"{working_directory}/{seed_name}"

    @staticmethod
    def invalid_remote_job_labels(labels: list[str]) -> list[str]:
        return [label for label in labels if "/" in label]


@pytest.fixture
def controller(qapp, monkeypatch) -> Iterator[tuple[ClusterController, _FakeRunner]]:
    monkeypatch.setattr(cluster_controller, "ClusterJobRunner", _FakeRunner)
    ctrl = ClusterController(settings=None)
    fake = ctrl._runner  # type: ignore[assignment]
    try:
        yield ctrl, fake
    finally:
        ctrl.shutdown()


def test_blocking_calls_run_on_worker_thread(controller) -> None:
    ctrl, fake = controller
    gui_thread = QThread.currentThread()

    assert ctrl.submit(seed_name="seed") == "JOB-1"
    assert ctrl.remote_directory_exists("/remote/dir") is True
    ctrl.remove_remote_directory("/remote/dir")
    ctrl.clear_tracked_jobs()

    for key in ("submit", "exists", "rmtree", "clear"):
        assert fake.threads[key] is ctrl._thread
        assert fake.threads[key] is not gui_thread


def test_async_calls_run_on_worker_thread(controller) -> None:
    ctrl, fake = controller
    gui_thread = QThread.currentThread()

    ctrl.refresh_queue()
    ctrl.cancel("job-7")
    ctrl.track_existing_job(job_id="job-9")

    assert _process_until(lambda: {"refresh", "cancel", "track"} <= set(fake.threads))
    assert fake.threads["refresh"] is ctrl._thread
    assert fake.threads["refresh"] is not gui_thread
    assert fake.threads["cancel"] is ctrl._thread
    assert fake.threads["track"] is ctrl._thread


def test_connect_mirrors_state_and_forwards_signal(controller) -> None:
    ctrl, fake = controller
    received: list[bool] = []
    ctrl.connection_status_changed.connect(received.append)
    assert ctrl.is_connected is False

    ctrl.connect_to_cluster(_FakeConfig(name="hpc", username="alice"))

    # Mirror is updated eagerly after a successful blocking connect.
    assert ctrl.is_connected is True
    assert ctrl.connected_cluster_name == "hpc"
    assert ctrl.connected_username == "alice"
    assert fake.threads["connect"] is ctrl._thread

    # The forwarded signal arrives on the GUI thread once events are pumped.
    assert _process_until(lambda: received == [True])


def test_disconnect_signal_clears_mirror(controller) -> None:
    ctrl, fake = controller
    ctrl.connect_to_cluster(_FakeConfig())
    assert ctrl.is_connected is True

    fake.connection_status_changed.emit(False)

    assert _process_until(lambda: ctrl.is_connected is False)
    assert ctrl.connected_cluster_name is None
    assert ctrl.connected_username is None


def test_blocking_call_propagates_exception(controller) -> None:
    ctrl, fake = controller
    fake.submit_should_raise = True
    with pytest.raises(ConnectionError, match="boom"):
        ctrl.submit(seed_name="seed")


def test_worker_signals_forward_to_gui(controller) -> None:
    ctrl, fake = controller
    received: list[tuple] = []
    ctrl.job_finished.connect(lambda job_id, parsed: received.append((job_id, parsed)))

    fake.job_finished.emit("job-1", "parsed")

    assert _process_until(lambda: received == [("job-1", "parsed")])


def test_pure_passthrough_helpers(controller) -> None:
    ctrl, _fake = controller
    assert ctrl.remote_job_directory(working_directory="/work", seed_name="s") == "/work/s"
    assert ctrl.invalid_remote_job_labels(["ok", "bad/label"]) == ["bad/label"]


def test_ssh_task_runner_executes_and_captures(qapp) -> None:
    runner = cluster_controller._SshTaskRunner()

    ran: list[str] = []
    runner.run_task(lambda: ran.append("task"))
    assert ran == ["task"]

    runner._pending_task = lambda: 42
    runner.run_pending()
    assert runner._pending_result == 42
    assert runner._pending_error is None

    def boom() -> None:
        raise ValueError("nope")

    runner._pending_task = boom
    runner.run_pending()
    assert isinstance(runner._pending_error, ValueError)

    runner._pending_task = None
    runner.run_pending()
    assert runner._pending_result is None


def test_blocking_call_falls_back_after_shutdown(controller) -> None:
    ctrl, fake = controller
    ctrl.shutdown()
    assert ctrl._thread.isRunning() is False
    # With the worker stopped, the blocking bridge runs the call directly so the
    # GUI does not deadlock waiting on a dead thread.
    ctrl.clear_tracked_jobs()
    assert ("clear",) in fake.calls


def test_shutdown_stops_worker_thread(qapp, monkeypatch) -> None:
    monkeypatch.setattr(cluster_controller, "ClusterJobRunner", _FakeRunner)
    ctrl = ClusterController(settings=None)
    assert ctrl._thread.isRunning() is True
    ctrl.shutdown()
    assert ctrl._thread.isRunning() is False
    # Idempotent.
    ctrl.shutdown()
