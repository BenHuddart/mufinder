"""Tests for the threaded cluster controller facade.

These verify the guarantee: cluster SSH work runs on a dedicated worker thread
(never the GUI thread), and **no GUI-thread call blocks on the worker**. The
controller preserves the public surface that ``main_window`` relies on —
forwarded signals, mirrored connection state, and now a request/completion bridge
(``operation_finished``) that delivers each op's outcome asynchronously with a
per-request id so continuations route correctly and always fire.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest
from PySide6.QtCore import QObject, QThread, Signal

from mufinder.app import cluster_controller
from mufinder.app.cluster_controller import ClusterController
from tests.support.qt import pump_until


class _FakeConfig:
    def __init__(self, name: str = "hpc", username: str = "alice") -> None:
        self.name = name
        self.username = username
        self.jump_host_enabled = False


class _FakeCredentials:
    def __init__(self, target_secret: str | None = None) -> None:
        self.target_secret = target_secret


class _FakeRunner(QObject):
    """Stand-in for ClusterJobRunner that records the thread each call runs on."""

    job_started = Signal(object)
    job_finished = Signal(object, object)
    job_failed = Signal(object, str)
    job_cancelled = Signal(object)
    status_changed = Signal(object, object)
    job_reappeared = Signal(object, object)
    connection_status_changed = Signal(bool)
    queue_checked = Signal(object)
    poll_failed = Signal(str)
    geometry_progress_read = Signal(object)

    def __init__(self, *, settings: object = None) -> None:
        super().__init__()
        self.settings = settings
        self.calls: list[tuple] = []
        self.threads: dict[str, QThread] = {}
        self.submit_should_raise = False
        self.existing_dirs: set[str] | None = None
        self.exists_should_raise: set[str] = set()
        self.existing_files: set[str] | None = None
        self.file_exists_should_raise: set[str] = set()
        self.active_scheduler_jobs: set[str] = set()
        self.on_submit = None
        self.connect_should_raise: Exception | None = None
        # When set, ``block_worker`` holds the worker thread inside a long op so a
        # test can prove a later GUI-thread call does not wait for it.
        self.block_worker: threading.Event | None = None

    def connect_to_cluster(self, config, password=None, *, credentials=None) -> None:
        self.threads["connect"] = QThread.currentThread()
        self.calls.append(("connect", config.name))
        if self.connect_should_raise is not None:
            raise self.connect_should_raise
        self.connection_status_changed.emit(True)

    def submit(self, **kwargs) -> str:
        self.threads["submit"] = QThread.currentThread()
        if self.submit_should_raise:
            raise ConnectionError("boom")
        self.calls.append(("submit", kwargs.get("seed_name")))
        if self.on_submit is not None:
            self.on_submit()
        return "JOB-1"

    def remote_directory_exists(self, remote_directory: str) -> bool:
        self.threads["exists"] = QThread.currentThread()
        self.calls.append(("exists", remote_directory))
        if remote_directory in self.exists_should_raise:
            raise ConnectionError("stat failed")
        if self.existing_dirs is None:
            return True
        return remote_directory in self.existing_dirs

    def remote_file_exists(self, remote_path: str) -> bool:
        self.threads["file_exists"] = QThread.currentThread()
        self.calls.append(("file_exists", remote_path))
        if remote_path in self.file_exists_should_raise:
            raise ConnectionError("stat failed")
        if self.existing_files is None:
            return True
        return remote_path in self.existing_files

    def remove_remote_directory(self, remote_directory: str) -> None:
        self.threads["rmtree"] = QThread.currentThread()
        self.calls.append(("rmtree", remote_directory))

    def scheduler_job_is_active(self, remote_job_id: str) -> bool:
        self.calls.append(("scheduler_check", remote_job_id))
        return remote_job_id in self.active_scheduler_jobs

    def refresh_queue(self) -> None:
        self.threads["refresh"] = QThread.currentThread()
        self.calls.append(("refresh",))

    def read_geometry_progress(self, job_ids) -> None:
        self.threads["progress"] = QThread.currentThread()
        self.calls.append(("progress", tuple(job_ids)))
        self.geometry_progress_read.emit({job_id: f"progress-{job_id}" for job_id in job_ids})

    def cancel(self, job_id) -> None:
        self.threads["cancel"] = QThread.currentThread()
        self.calls.append(("cancel", job_id))

    def track_existing_job(self, **kwargs) -> None:
        self.threads["track"] = QThread.currentThread()
        self.calls.append(("track", kwargs.get("job_id")))

    def clear_tracked_jobs(self) -> None:
        self.threads["clear"] = QThread.currentThread()
        self.calls.append(("clear",))

    def hold(self) -> str:
        """A long op that pins the worker thread until ``block_worker`` is set."""

        self.threads["hold"] = QThread.currentThread()
        self.calls.append(("hold",))
        if self.block_worker is not None:
            self.block_worker.wait(2.0)
        return "held"

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


def _collect_ops(ctrl: ClusterController) -> list[tuple]:
    """Record every ``operation_finished`` payload as ``(request_id, kind, result, error)``."""

    ops: list[tuple] = []
    ctrl.operation_finished.connect(
        lambda rid, kind, result, error: ops.append((rid, kind, result, error))
    )
    return ops


def test_bridge_ops_run_on_worker_thread_and_report_via_signal(controller) -> None:
    ctrl, fake = controller
    gui_thread = QThread.currentThread()
    ops = _collect_ops(ctrl)

    rid_submit = ctrl.submit(seed_name="seed")
    rid_exists = ctrl.remote_directory_exists("/remote/dir")
    rid_rm = ctrl.remove_remote_directory("/remote/dir")
    rid_clear = ctrl.clear_tracked_jobs()

    # Every call returned a distinct request id immediately.
    assert len({rid_submit, rid_exists, rid_rm, rid_clear}) == 4

    assert pump_until(lambda: len(ops) == 4)
    by_kind = {kind: (result, error) for _rid, kind, result, error in ops}
    assert by_kind[ctrl.OP_SUBMIT] == ("JOB-1", None)
    assert by_kind[ctrl.OP_REMOTE_DIR_EXISTS] == (True, None)
    assert by_kind[ctrl.OP_REMOVE_REMOTE_DIR][1] is None
    assert by_kind[ctrl.OP_CLEAR_TRACKED][1] is None

    for key in ("submit", "exists", "rmtree", "clear"):
        assert fake.threads[key] is ctrl._thread
        assert fake.threads[key] is not gui_thread


def test_clear_tracked_jobs_clears_mirror_eagerly(controller) -> None:
    # The GUI-thread mirror is cleared synchronously so the watchdog sees the drop
    # immediately, even before the queued worker op runs.
    ctrl, _fake = controller
    from uuid import uuid4

    job_id = uuid4()
    ctrl._tracked_active_job_ids.add(job_id)
    ctrl.clear_tracked_jobs()
    assert ctrl.tracked_active_job_ids() == frozenset()


def test_bridge_op_does_not_block_gui_while_worker_is_busy(controller) -> None:
    # A slow op pins the worker; a second GUI-thread call must return at once
    # rather than waiting for the first to finish.
    ctrl, fake = controller
    gate = threading.Event()
    fake.block_worker = gate
    ops = _collect_ops(ctrl)

    # Kick off the long op (worker will block inside fake.hold()).
    rid_slow = ctrl._run_async_op("hold", fake.hold)
    assert pump_until(lambda: "hold" in fake.threads)  # worker is now blocked

    # This call must return immediately even though the worker is busy.
    start = time.monotonic()
    rid_fast = ctrl.clear_tracked_jobs()
    assert time.monotonic() - start < 0.2
    assert rid_fast != rid_slow

    # Release the worker; both completions arrive, each keyed to its request id.
    gate.set()
    assert pump_until(lambda: {rid_slow, rid_fast} <= {op[0] for op in ops})


def test_async_calls_run_on_worker_thread(controller) -> None:
    ctrl, fake = controller
    gui_thread = QThread.currentThread()

    ctrl.refresh_queue()
    ctrl.cancel("job-7")
    ctrl.track_existing_job(job_id="job-9")

    assert pump_until(lambda: {"refresh", "cancel", "track"} <= set(fake.threads))
    assert fake.threads["refresh"] is ctrl._thread
    assert fake.threads["refresh"] is not gui_thread
    assert fake.threads["cancel"] is ctrl._thread
    assert fake.threads["track"] is ctrl._thread


def test_geometry_progress_is_read_on_the_worker_and_reported_on_the_gui_thread(controller) -> None:
    ctrl, fake = controller
    gui_thread = QThread.currentThread()
    received: list[tuple[object, QThread]] = []
    ctrl.geometry_progress_read.connect(
        lambda results: received.append((results, QThread.currentThread()))
    )

    ctrl.read_geometry_progress(["job-1", "job-2"])

    assert pump_until(lambda: bool(received))
    assert fake.threads["progress"] is ctrl._thread
    assert received == [({"job-1": "progress-job-1", "job-2": "progress-job-2"}, gui_thread)]


def test_connect_mirrors_state_and_forwards_signal(controller) -> None:
    ctrl, fake = controller
    received: list[bool] = []
    ctrl.connection_status_changed.connect(received.append)
    ops = _collect_ops(ctrl)
    assert ctrl.is_connected is False

    rid = ctrl.connect_to_cluster(_FakeConfig(name="hpc", username="alice"))
    assert rid is not None

    # The connect runs on the worker; the mirror adopts the pending identity from
    # the forwarded connection_status_changed once events are pumped (no blocking).
    assert pump_until(lambda: ctrl.is_connected is True)
    assert ctrl.connected_cluster_name == "hpc"
    assert ctrl.connected_username == "alice"
    assert fake.threads["connect"] is ctrl._thread
    assert received == [True]

    # The op completion also fires (success -> no error).
    assert pump_until(lambda: any(k == ctrl.OP_CONNECT for _r, k, _res, _e in ops))
    connect_op = next(op for op in ops if op[1] == ctrl.OP_CONNECT)
    assert connect_op[3] is None


def test_connect_async_success_mirrors_and_signals_none(controller) -> None:
    ctrl, fake = controller
    results: list = []
    ctrl.connect_finished.connect(results.append)

    ctrl.connect_async(_FakeConfig(name="hpc", username="bob"), credentials=_FakeCredentials())

    assert pump_until(lambda: results)
    assert results == [None]
    assert ctrl.is_connected is True
    assert ctrl.connected_cluster_name == "hpc"
    assert ctrl.connected_username == "bob"
    assert fake.threads["connect"] is ctrl._thread


def test_pending_connect_identity_is_consumed_not_reused(controller) -> None:
    # After a successful connect, a later status change must not re-adopt the
    # previous connect's identity (the pending identity is consumed once).
    ctrl, fake = controller
    ctrl.connect_async(_FakeConfig(name="hpc", username="bob"), credentials=_FakeCredentials())
    assert pump_until(lambda: ctrl.connected_cluster_name == "hpc")

    # A spurious reconnect signal with nothing pending keeps the current name.
    fake.connection_status_changed.emit(True)
    assert pump_until(lambda: ctrl.is_connected is True)
    assert ctrl.connected_cluster_name == "hpc"
    assert ctrl._pending_connect is None


def test_connect_async_failure_emits_exception(controller) -> None:
    ctrl, fake = controller
    fake.connect_should_raise = ConnectionError("nope")
    results: list = []
    ctrl.connect_finished.connect(results.append)

    ctrl.connect_async(_FakeConfig(), credentials=_FakeCredentials())

    assert pump_until(lambda: results)
    assert isinstance(results[0], ConnectionError)
    assert ctrl.is_connected is False


def test_disconnect_signal_clears_mirror(controller) -> None:
    ctrl, fake = controller
    ctrl.connect_to_cluster(_FakeConfig())
    assert pump_until(lambda: ctrl.is_connected is True)

    fake.connection_status_changed.emit(False)

    assert pump_until(lambda: ctrl.is_connected is False)
    assert ctrl.connected_cluster_name is None
    assert ctrl.connected_username is None


def test_bridge_op_delivers_exception_via_completion(controller) -> None:
    # A raising op does not propagate on the GUI thread; the exception is carried
    # on the completion so a pending-state UI is always restored (never stuck).
    ctrl, fake = controller
    fake.submit_should_raise = True
    ops = _collect_ops(ctrl)

    rid = ctrl.submit(seed_name="seed")

    assert pump_until(lambda: any(op[0] == rid for op in ops))
    op = next(op for op in ops if op[0] == rid)
    _rid, kind, result, error = op
    assert kind == ctrl.OP_SUBMIT
    assert result is None
    assert isinstance(error, ConnectionError)
    assert str(error) == "boom"


def test_two_ops_queued_back_to_back_route_to_own_continuation(controller) -> None:
    # Regression: distinct request ids let two in-flight ops route to their own
    # continuation even though the worker processes them serially.
    ctrl, fake = controller
    fake.existing_dirs = {"/yes"}
    routed: dict[object, tuple] = {}
    ctrl.operation_finished.connect(
        lambda rid, kind, result, error: routed.__setitem__(rid, (kind, result, error))
    )

    rid_a = ctrl.remote_directory_exists("/yes")
    rid_b = ctrl.remote_directory_exists("/no")
    assert rid_a != rid_b

    assert pump_until(lambda: {rid_a, rid_b} <= set(routed))
    assert routed[rid_a] == (ctrl.OP_REMOTE_DIR_EXISTS, True, None)
    assert routed[rid_b] == (ctrl.OP_REMOTE_DIR_EXISTS, False, None)


def test_worker_signals_forward_to_gui(controller) -> None:
    ctrl, fake = controller
    received: list[tuple] = []
    ctrl.job_finished.connect(lambda job_id, parsed: received.append((job_id, parsed)))

    fake.job_finished.emit("job-1", "parsed")

    assert pump_until(lambda: received == [("job-1", "parsed")])


def test_pure_passthrough_helpers(controller) -> None:
    ctrl, _fake = controller
    assert ctrl.remote_job_directory(working_directory="/work", seed_name="s") == "/work/s"
    assert ctrl.invalid_remote_job_labels(["ok", "bad/label"]) == ["bad/label"]


def _batch_item(index: int, *, delete: str | None = None) -> dict:
    return {
        "job_id": f"job-{index}",
        "label": f"L{index}",
        "submit_kwargs": {"seed_name": f"s{index}"},
        "remote_dir_to_delete": delete,
    }


def test_check_remote_dirs_reports_existing_and_errors(controller) -> None:
    ctrl, fake = controller
    fake.existing_dirs = {"/exists"}
    fake.exists_should_raise = {"/broken"}
    payloads: list[dict] = []
    ctrl.remote_dirs_checked.connect(payloads.append)

    ctrl.check_remote_dirs([("a", "/exists"), ("b", "/missing"), ("c", "/broken")])

    assert pump_until(lambda: payloads)
    assert payloads[0]["existing"] == ["/exists"]
    assert payloads[0]["errors"] == {"c": "stat failed"}
    assert fake.threads["exists"] is ctrl._thread


def test_check_remote_dirs_always_emits_even_on_unexpected_error(controller) -> None:
    # A malformed item makes the loop's tuple unpacking raise outside the per-item
    # try; the result must still be emitted (finally) so the GUI's nested event
    # loop — which has no Cancel button — can quit instead of hanging forever.
    ctrl, _fake = controller
    payloads: list[dict] = []
    ctrl.remote_dirs_checked.connect(payloads.append)

    # The unpacking error propagates (and is swallowed by Qt on the worker
    # thread in production), but the result is emitted first via finally.
    with pytest.raises(ValueError):
        ctrl._run_check_remote_dirs([("only-one-element",)])

    assert payloads and isinstance(payloads[0], dict)


def test_check_remote_files_reports_existing_and_errors(controller) -> None:
    ctrl, fake = controller
    fake.existing_files = {"/exists/seed.check"}
    fake.file_exists_should_raise = {"/broken/seed.check"}
    payloads: list[dict] = []
    ctrl.remote_files_checked.connect(payloads.append)

    ctrl.check_remote_files(
        [
            ("a", "/exists/seed.check"),
            ("b", "/missing/seed.check"),
            ("c", "/broken/seed.check"),
        ]
    )

    assert pump_until(lambda: payloads)
    assert payloads[0]["existing"] == ["/exists/seed.check"]
    assert payloads[0]["errors"] == {"c": "stat failed"}
    assert fake.threads["file_exists"] is ctrl._thread


def test_check_remote_files_always_emits_even_on_unexpected_error(controller) -> None:
    # Mirrors test_check_remote_dirs_always_emits_even_on_unexpected_error: a
    # malformed item makes the loop's tuple unpacking raise outside the
    # per-item try, but the result must still be emitted via finally.
    ctrl, _fake = controller
    payloads: list[dict] = []
    ctrl.remote_files_checked.connect(payloads.append)

    with pytest.raises(ValueError):
        ctrl._run_check_remote_files([("only-one-element",)])

    assert payloads and isinstance(payloads[0], dict)


def test_run_check_remote_files_direct_covers_worker_body(controller) -> None:
    ctrl, fake = controller
    fake.existing_files = {"/x/seed.check"}
    fake.file_exists_should_raise = {"/e/seed.check"}
    payloads: list[dict] = []
    ctrl.remote_files_checked.connect(payloads.append)

    ctrl._run_check_remote_files(
        [("a", "/x/seed.check"), ("b", "/y/seed.check"), ("c", "/e/seed.check")]
    )

    assert payloads[0]["existing"] == ["/x/seed.check"]
    assert payloads[0]["errors"] == {"c": "stat failed"}


def test_submit_batch_emits_progress_results_and_finished(controller) -> None:
    ctrl, fake = controller
    progress: list[tuple] = []
    results: list[tuple] = []
    finished: list[bool] = []
    ctrl.submit_batch_progress.connect(lambda i, n, label: progress.append((i, n, label)))
    ctrl.job_submit_finished.connect(lambda jid, rid, err: results.append((jid, rid, err)))
    ctrl.submit_batch_finished.connect(lambda: finished.append(True))

    ctrl.submit_batch([_batch_item(0, delete="/old"), _batch_item(1)])

    assert pump_until(lambda: finished)
    assert progress[0] == (1, 2, "L0")
    assert [r[0] for r in results] == ["job-0", "job-1"]
    assert all(r[1] == "JOB-1" and r[2] == "" for r in results)
    assert ("rmtree", "/old") in fake.calls
    assert fake.threads["submit"] is ctrl._thread


def test_submit_batch_reports_per_job_failure(controller) -> None:
    # JobRunner design decision 1: ClusterJobRunner.submit may raise; the batch
    # worker converts it per job into job_submit_finished(job_id, None, error).
    ctrl, fake = controller
    fake.submit_should_raise = True
    results: list[tuple] = []
    finished: list[bool] = []
    ctrl.job_submit_finished.connect(lambda jid, rid, err: results.append((jid, rid, err)))
    ctrl.submit_batch_finished.connect(lambda: finished.append(True))

    ctrl.submit_batch([_batch_item(0)])

    assert pump_until(lambda: finished)
    assert results == [("job-0", None, "boom")]


def test_submit_batch_stops_after_cancel(controller) -> None:
    ctrl, fake = controller
    results: list[tuple] = []
    finished: list[bool] = []
    ctrl.job_submit_finished.connect(lambda jid, rid, err: results.append((jid, rid, err)))
    ctrl.submit_batch_finished.connect(lambda: finished.append(True))
    # Cancel from inside the first submit so the second job is never submitted.
    fake.on_submit = ctrl.cancel_batch

    ctrl.submit_batch([_batch_item(0), _batch_item(1), _batch_item(2)])

    assert pump_until(lambda: finished)
    assert [r[0] for r in results] == ["job-0"]


def test_run_submit_batch_direct_covers_worker_body(controller) -> None:
    # Invoke the worker method directly (on the test thread) so its logic is
    # exercised deterministically and visibly to coverage; the threaded path is
    # covered by test_submit_batch_* above.
    ctrl, fake = controller
    results: list[tuple] = []
    finished: list[bool] = []
    ctrl.job_submit_finished.connect(lambda jid, rid, err: results.append((jid, rid, err)))
    ctrl.submit_batch_finished.connect(lambda: finished.append(True))

    ctrl._batch_cancelled = False
    ctrl._run_submit_batch([_batch_item(0, delete="/old"), _batch_item(1)])

    assert finished == [True]
    assert [r[0] for r in results] == ["job-0", "job-1"]
    assert ("rmtree", "/old") in fake.calls

    # A per-job submit failure is reported, not raised.
    results.clear()
    finished.clear()
    fake.submit_should_raise = True
    ctrl._batch_cancelled = False
    ctrl._run_submit_batch([_batch_item(9)])
    assert results == [("job-9", None, "boom")]
    assert finished == [True]
    fake.submit_should_raise = False

    # Cancel short-circuits the loop before any job runs.
    results.clear()
    finished.clear()
    ctrl.cancel_batch()
    ctrl._run_submit_batch([_batch_item(2)])
    assert finished == [True]
    assert results == []


def test_run_connect_direct_covers_worker_body(controller) -> None:
    ctrl, fake = controller
    results: list = []
    ctrl.connect_finished.connect(results.append)

    ctrl._run_connect(_FakeConfig(), _FakeCredentials())
    assert results[-1] is None

    jump = _FakeConfig()
    jump.jump_host_enabled = True
    ctrl._run_connect(jump, _FakeCredentials())
    assert results[-1] is None

    fake.connect_should_raise = ConnectionError("x")
    ctrl._run_connect(_FakeConfig(), _FakeCredentials())
    assert isinstance(results[-1], ConnectionError)


def test_run_check_remote_dirs_direct_covers_worker_body(controller) -> None:
    ctrl, fake = controller
    fake.existing_dirs = {"/x"}
    fake.exists_should_raise = {"/e"}
    payloads: list[dict] = []
    ctrl.remote_dirs_checked.connect(payloads.append)

    ctrl._run_check_remote_dirs([("a", "/x"), ("b", "/y"), ("c", "/e")])

    assert payloads[0]["existing"] == ["/x"]
    assert payloads[0]["errors"] == {"c": "stat failed"}


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


def test_ssh_task_runner_run_async_op_reports_success_and_error(qapp) -> None:
    runner = cluster_controller._SshTaskRunner()
    seen: list[tuple] = []
    runner.op_finished.connect(
        lambda rid, kind, result, error: seen.append((rid, kind, result, error))
    )

    runner.run_async_op(1, "k", lambda: 42)
    assert seen[-1] == (1, "k", 42, None)

    def boom() -> None:
        raise ValueError("nope")

    runner.run_async_op(2, "k", boom)
    rid, kind, result, error = seen[-1]
    assert (rid, kind, result) == (2, "k", None)
    assert isinstance(error, ValueError)


def test_bridge_op_falls_back_after_shutdown(controller) -> None:
    ctrl, fake = controller
    ops = _collect_ops(ctrl)
    ctrl.shutdown()
    assert ctrl._thread.isRunning() is False
    # With the worker stopped, the bridge runs the call directly (no dead-thread
    # queue) and emits the completion synchronously so continuations still fire.
    rid = ctrl.clear_tracked_jobs()
    assert ("clear",) in fake.calls
    assert ops and ops[-1] == (rid, ctrl.OP_CLEAR_TRACKED, None, None)


def test_shutdown_stops_worker_thread(qapp, monkeypatch) -> None:
    monkeypatch.setattr(cluster_controller, "ClusterJobRunner", _FakeRunner)
    ctrl = ClusterController(settings=None)
    fake = ctrl._runner
    assert ctrl._thread.isRunning() is True
    ctrl.shutdown()
    assert ctrl._thread.isRunning() is False
    # The disconnect must run synchronously on the worker before quit, otherwise
    # the worker's poll timer is left active and is later killed from the GUI
    # thread ("Timers cannot be stopped from another thread").
    assert ("shutdown",) in fake.calls
    # Idempotent.
    ctrl.shutdown()


def test_real_runner_poll_timer_is_stopped_by_shutdown(qapp) -> None:
    # Use the real ClusterJobRunner so its worker-thread poll QTimer exists, then
    # confirm shutdown leaves it inactive (and on no live thread) so destruction
    # does not stop a timer from the wrong thread.
    from mufinder.app.cluster_controller import ClusterController as RealController

    ctrl = RealController(settings=None)
    # Start the poll timer on the worker thread, as submit/connect/track do.
    ctrl._call_blocking(ctrl._runner._poll_timer.start)
    assert ctrl._call_blocking(ctrl._runner._poll_timer.isActive) is True

    ctrl.shutdown()
    assert ctrl._thread.isRunning() is False
    assert ctrl._runner._poll_timer.isActive() is False


def test_run_submit_batch_deletes_a_finished_runs_directory(controller) -> None:
    ctrl, fake = controller
    results: list[tuple] = []
    ctrl.job_submit_finished.connect(lambda jid, rid, err: results.append((jid, rid, err)))

    ctrl._batch_cancelled = False
    ctrl._run_submit_batch([{**_batch_item(0, delete="/old"), "delete_only_if_finished": "777"}])

    assert ("scheduler_check", "777") in fake.calls
    assert ("rmtree", "/old") in fake.calls
    assert results == [("job-0", "JOB-1", "")]


def test_run_submit_batch_refuses_to_delete_a_running_jobs_directory(controller) -> None:
    ctrl, fake = controller
    fake.active_scheduler_jobs = {"777"}
    results: list[tuple] = []
    ctrl.job_submit_finished.connect(lambda jid, rid, err: results.append((jid, rid, err)))

    ctrl._batch_cancelled = False
    ctrl._run_submit_batch([{**_batch_item(0, delete="/old"), "delete_only_if_finished": "777"}])

    assert ("rmtree", "/old") not in fake.calls
    assert not any(call[0] == "submit" for call in fake.calls)
    assert results == [
        (
            "job-0",
            None,
            "scheduler job 777 is still queued or running, so its directory /old was not deleted",
        )
    ]
