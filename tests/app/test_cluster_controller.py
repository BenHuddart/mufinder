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
from pathlib import Path
from uuid import uuid4

import pytest
from PySide6.QtCore import QObject, QThread, Signal

from mufinder.app import cluster_controller
from mufinder.app.cluster_controller import ClusterController, _Batch
from mufinder.castep_io.geometry_progress import parse_geometry_progress
from mufinder.services.cluster_runner import CheckTarget
from mufinder.services.job_progress import REMOTE_READ_CHUNK_BYTES
from tests.support.cluster_ssh import GrowingOutputSSH
from tests.support.qt import pump_until

#: How long a stalled fake operation waits to be released. Far longer than a
#: close may take, so a close that waits for the worker fails its time limit.
_STALL_SECONDS = 20.0


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

    job_cancelled = Signal(object)
    job_checked = Signal(object, object)
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
        self.check_should_raise = False
        self.cancel_should_raise = False
        # ``stall_in`` names the operation that sticks, as an SSH call can on a
        # real cluster: it waits for ``release`` (at most _STALL_SECONDS).
        # ``abort_releases`` makes ``abort_connection`` free it, as closing the
        # transport frees a blocked paramiko call.
        self.stall_in = ""
        self.stalled = threading.Event()
        self.release = threading.Event()
        self.abort_releases = False
        self.aborted = threading.Event()

    def _stall(self, operation: str) -> None:
        if self.stall_in == operation:
            self.stalled.set()
            self.release.wait(_STALL_SECONDS)

    def abort_connection(self) -> None:
        self.calls.append(("abort",))
        self.aborted.set()
        if self.abort_releases:
            self.release.set()

    def connect_to_cluster(self, config, password=None, *, credentials=None) -> None:
        self.threads["connect"] = QThread.currentThread()
        self.calls.append(("connect", config.name))
        self._stall("connect")
        if self.connect_should_raise is not None:
            raise self.connect_should_raise
        self.connection_status_changed.emit(True)

    def submit(self, **kwargs) -> str:
        self.threads["submit"] = QThread.currentThread()
        if self.submit_should_raise:
            raise ConnectionError("boom")
        self.calls.append(("submit", kwargs.get("seed_name")))
        self._stall("submit")
        if self.aborted.is_set():
            # As paramiko's call fails once its transport is closed.
            raise ConnectionError("the SSH connection was closed")
        if self.on_submit is not None:
            self.on_submit()
        return "JOB-1"

    def remote_directory_exists(self, remote_directory: str) -> bool:
        self.threads["exists"] = QThread.currentThread()
        self.calls.append(("exists", remote_directory))
        self._stall("exists")
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
        self._stall("scheduler_check")
        if self.aborted.is_set():
            raise ConnectionError("the SSH connection was closed")
        return remote_job_id in self.active_scheduler_jobs

    def check_jobs(self, targets, *, refresh: bool = False) -> None:
        self.threads["check"] = QThread.currentThread()
        self.calls.append(("check", tuple(targets)) + (("refresh",) if refresh else ()))
        self._stall("check")
        if self.block_worker is not None:
            self.block_worker.wait(2.0)
        if self.check_should_raise:
            raise RuntimeError("check blew up")
        for target in targets:
            self.job_checked.emit(target, f"evidence-{target}")
        self.queue_checked.emit("snapshot")

    def read_geometry_progress(self, targets) -> None:
        self.threads["progress"] = QThread.currentThread()
        self.calls.append(("progress", tuple(targets)))
        self._stall("progress")
        self.geometry_progress_read.emit({target: f"progress-{target}" for target in targets})

    def forget_progress(self) -> None:
        self.calls.append(("forget_progress",))

    def cancel(self, job_id, remote_job_id="") -> None:
        self.threads["cancel"] = QThread.currentThread()
        self.calls.append(("cancel", job_id, remote_job_id))

    def cancel_scheduler_job(self, remote_job_id: str) -> None:
        self.threads["cancel_scheduler_job"] = QThread.currentThread()
        self.calls.append(("cancel_scheduler_job", remote_job_id))
        if self.cancel_should_raise:
            raise ConnectionError("scancel failed")

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
        # A worker a test left stuck finishes its last task once released, so
        # its thread is never destroyed while it runs.
        fake.release.set()
        ctrl.shutdown()
        assert ctrl._thread.wait(5000)


class _BatchId:
    """The ID ``submit_batch`` returned, for a worker-thread step that needs it.

    The worker can reach the batch's first submit before ``submit_batch``
    returns its ID to the test thread, so the step waits for :meth:`set`.
    """

    def __init__(self) -> None:
        self._value: int | None = None
        self._ready = threading.Event()

    def set(self, value: int) -> None:
        self._value = value
        self._ready.set()

    def get(self) -> int:
        assert self._ready.wait(2.0)
        assert self._value is not None
        return self._value


def _cancel_from_first_submit(ctrl: ClusterController, fake: _FakeRunner) -> _BatchId:
    """Cancel the batch from inside its first submit; ``set`` the ID ``submit_batch`` returns."""

    batch = _BatchId()
    fake.on_submit = lambda: ctrl.cancel_batch(batch.get())
    return batch


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

    # Every call returned a distinct request id immediately.
    assert len({rid_submit, rid_exists, rid_rm}) == 3

    assert pump_until(lambda: len(ops) == 3)
    by_kind = {kind: (result, error) for _rid, kind, result, error in ops}
    assert by_kind[ctrl.OP_SUBMIT] == ("JOB-1", None)
    assert by_kind[ctrl.OP_REMOTE_DIR_EXISTS] == (True, None)
    assert by_kind[ctrl.OP_REMOVE_REMOTE_DIR][1] is None

    for key in ("submit", "exists", "rmtree"):
        assert fake.threads[key] is ctrl._thread
        assert fake.threads[key] is not gui_thread


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
    rid_fast = ctrl.remote_directory_exists("/remote/dir")
    assert time.monotonic() - start < 0.2
    assert rid_fast != rid_slow

    # Release the worker; both completions arrive, each keyed to its request id.
    gate.set()
    assert pump_until(lambda: {rid_slow, rid_fast} <= {op[0] for op in ops})


def test_async_calls_run_on_worker_thread(controller) -> None:
    ctrl, fake = controller
    gui_thread = QThread.currentThread()

    ctrl.check_jobs(["job-1"])
    ctrl.cancel("job-7", "4242")
    ctrl.forget_progress()

    assert pump_until(lambda: ("forget_progress",) in fake.calls)
    assert fake.threads["check"] is ctrl._thread
    assert fake.threads["check"] is not gui_thread
    assert fake.threads["cancel"] is ctrl._thread
    assert ("cancel", "job-7", "4242") in fake.calls


def test_check_reports_evidence_and_the_pass_on_the_gui_thread(controller) -> None:
    ctrl, fake = controller
    gui_thread = QThread.currentThread()
    received: list[tuple] = []
    ctrl.job_checked.connect(
        lambda job_id, evidence: received.append((job_id, evidence, QThread.currentThread()))
    )
    snapshots: list[object] = []
    ctrl.queue_checked.connect(snapshots.append)

    assert ctrl.check_jobs(["job-1", "job-2"]) is True

    assert pump_until(lambda: snapshots == ["snapshot"])
    assert received == [
        ("job-1", "evidence-job-1", gui_thread),
        ("job-2", "evidence-job-2", gui_thread),
    ]
    assert ("check", ("job-1", "job-2")) in fake.calls


def test_a_check_is_skipped_while_another_is_on_the_worker(controller) -> None:
    ctrl, fake = controller
    gate = threading.Event()
    fake.block_worker = gate

    assert ctrl.check_jobs(["job-1"]) is True
    assert pump_until(lambda: "check" in fake.threads)  # the worker is inside the check

    # The once-a-minute check does not pile up behind it; a requested one queues.
    assert ctrl.check_jobs(["job-1"], skip_if_busy=True) is False
    assert ctrl.check_jobs(["job-2"]) is True

    gate.set()
    assert pump_until(lambda: ctrl._checks_in_flight == 0)
    assert [call for call in fake.calls if call[0] == "check"] == [
        ("check", ("job-1",)),
        ("check", ("job-2",)),
    ]
    assert ctrl.check_jobs([], skip_if_busy=True) is True


def test_a_refresh_is_queued_behind_a_running_check(controller) -> None:
    ctrl, fake = controller
    gate = threading.Event()
    fake.block_worker = gate

    assert ctrl.check_jobs(["job-1"], skip_if_busy=True) is True
    assert pump_until(lambda: "check" in fake.threads)

    # A Refresh Queue press waits for the minute check instead of being dropped.
    assert ctrl.check_jobs(["job-1", "job-2"], refresh=True) is True

    gate.set()
    assert pump_until(lambda: ctrl._checks_in_flight == 0)
    assert [call for call in fake.calls if call[0] == "check"] == [
        ("check", ("job-1",)),
        ("check", ("job-1", "job-2"), "refresh"),
    ]


def test_a_check_that_raises_does_not_leave_later_checks_skipped(controller) -> None:
    ctrl, fake = controller
    fake.check_should_raise = True

    ctrl.check_jobs(["job-1"])

    assert pump_until(lambda: ctrl._checks_in_flight == 0 and "check" in fake.threads)
    fake.check_should_raise = False
    assert ctrl.check_jobs([], skip_if_busy=True) is True


def test_a_job_counts_as_being_submitted_until_its_result_arrives(controller) -> None:
    ctrl, fake = controller
    gate = threading.Event()
    fake.on_submit = lambda: gate.wait(2.0)
    seen_at_result: list[bool] = []
    ctrl.job_submit_finished.connect(
        lambda _batch, job_id, _remote_id, _error: seen_at_result.append(ctrl.is_submitting(job_id))
    )
    items = [
        {"job_id": "a", "label": "a", "submit_kwargs": {"seed_name": "a"}},
        {"job_id": "b", "label": "b", "submit_kwargs": {"seed_name": "b"}},
    ]

    assert not ctrl.is_submitting("a")
    ctrl.submit_batch(items)
    assert ctrl.is_submitting("a") and ctrl.is_submitting("b")

    gate.set()
    assert pump_until(lambda: len(seen_at_result) == 2)
    # A listener applying the result already sees the job as submitted.
    assert seen_at_result == [False, False]
    assert not ctrl.is_submitting("a") and not ctrl.is_submitting("b")


def test_a_cancelled_batch_stops_counting_its_unsubmitted_jobs(controller) -> None:
    ctrl, fake = controller
    finished: list[bool] = []
    ctrl.submit_batch_finished.connect(lambda _batch: finished.append(True))
    batch = _cancel_from_first_submit(ctrl, fake)
    items = [
        {"job_id": "a", "label": "a", "submit_kwargs": {"seed_name": "a"}},
        {"job_id": "b", "label": "b", "submit_kwargs": {"seed_name": "b"}},
    ]

    batch.set(ctrl.submit_batch(items))

    assert pump_until(lambda: finished == [True])
    assert not ctrl.is_submitting("b")


def _abandon_while_submitting(ctrl: ClusterController, fake: _FakeRunner) -> threading.Event:
    """Submit a two-job batch and abandon it while the first submit is stuck."""

    gate = threading.Event()
    in_submit = threading.Event()

    def stuck() -> None:
        in_submit.set()
        gate.wait(5.0)

    fake.on_submit = stuck
    batch = ctrl.submit_batch(
        [
            {"job_id": "a", "label": "a", "submit_kwargs": {"seed_name": "a"}},
            {"job_id": "b", "label": "b", "submit_kwargs": {"seed_name": "b"}},
        ]
    )
    assert in_submit.wait(2.0)
    ctrl.abandon_batch(batch)
    return gate


def test_an_abandoned_job_submitted_late_is_cancelled_on_the_scheduler(controller) -> None:
    ctrl, fake = controller
    finished: list[bool] = []
    ctrl.submit_batch_finished.connect(lambda _batch: finished.append(True))
    cancelled: list[object] = []
    ctrl.job_cancelled.connect(cancelled.append)

    gate = _abandon_while_submitting(ctrl, fake)
    gate.set()

    assert pump_until(lambda: ("cancel_scheduler_job", "JOB-1") in fake.calls)
    assert fake.threads["cancel_scheduler_job"] is not QThread.currentThread()
    # Only the job in hand reached the scheduler, and the project, which
    # already shows it cancelled, is not told again.
    assert [call for call in fake.calls if call[0] == "submit"] == [("submit", "a")]
    assert not pump_until(lambda: bool(cancelled), timeout=0.2)
    assert finished == [True]
    assert not ctrl.is_submitting("a")


def test_a_failed_late_cancel_is_reported(controller) -> None:
    ctrl, fake = controller
    fake.cancel_should_raise = True
    failures: list[tuple] = []
    ctrl.abandoned_job_cancel_failed.connect(
        lambda job_id, remote_job_id, error: failures.append((job_id, remote_job_id, error))
    )

    gate = _abandon_while_submitting(ctrl, fake)
    gate.set()

    assert pump_until(lambda: failures == [("a", "JOB-1", "scancel failed")])


def test_a_cancelled_batch_keeps_the_job_in_hand(controller) -> None:
    ctrl, fake = controller
    finished: list[bool] = []
    ctrl.submit_batch_finished.connect(lambda _batch: finished.append(True))
    batch = _cancel_from_first_submit(ctrl, fake)

    batch.set(
        ctrl.submit_batch([{"job_id": "a", "label": "a", "submit_kwargs": {"seed_name": "a"}}])
    )

    assert pump_until(lambda: finished == [True])
    assert not any(call[0] == "cancel_scheduler_job" for call in fake.calls)


def test_a_job_submitted_again_after_an_abandoned_batch_is_not_cancelled(controller) -> None:
    ctrl, fake = controller
    finished: list[bool] = []
    ctrl.submit_batch_finished.connect(lambda _batch: finished.append(True))

    gate = _abandon_while_submitting(ctrl, fake)
    gate.set()
    assert pump_until(lambda: finished == [True])
    fake.on_submit = None
    ctrl.submit_batch([{"job_id": "a", "label": "a", "submit_kwargs": {"seed_name": "a"}}])

    assert pump_until(lambda: finished == [True, True])
    assert [call for call in fake.calls if call[0] == "cancel_scheduler_job"] == [
        ("cancel_scheduler_job", "JOB-1")
    ]


def test_an_abandoned_batch_submits_nothing_more_once_a_second_batch_starts(controller) -> None:
    # Batch 2 used to reset the one shared cancel flag, so batch 1 went on
    # submitting after its stuck job returned, and its finish ended batch 2's wait.
    ctrl, fake = controller
    finished: list[int] = []
    still_submitting_c: list[bool] = []
    results: list[tuple] = []

    def on_finished(batch_id: int) -> None:
        finished.append(batch_id)
        still_submitting_c.append(ctrl.is_submitting("c"))

    ctrl.submit_batch_finished.connect(on_finished)
    ctrl.job_submit_finished.connect(
        lambda batch_id, job_id, remote_id, error: results.append(
            (batch_id, job_id, remote_id, error)
        )
    )
    gate = threading.Event()
    in_submit = threading.Event()

    def stuck() -> None:
        in_submit.set()
        gate.wait(5.0)

    fake.on_submit = stuck
    first = ctrl.submit_batch([_batch_item(0), _batch_item(1)])
    assert in_submit.wait(2.0)
    ctrl.abandon_batch(first)
    fake.on_submit = None
    second = ctrl.submit_batch([{"job_id": "c", "label": "c", "submit_kwargs": {"seed_name": "c"}}])
    assert second != first
    assert ctrl.is_submitting("c")

    gate.set()

    assert pump_until(lambda: finished == [first, second])
    # Batch 1 stopped after the job in hand; batch 2 submitted its own job.
    assert [call[1] for call in fake.calls if call[0] == "submit"] == ["s0", "c"]
    assert [(batch_id, job_id) for batch_id, job_id, _remote, _error in results] == [
        (first, "job-0"),
        (second, "c"),
    ]
    # Batch 1's finish did not stop batch 2's job counting as being submitted.
    assert still_submitting_c == [True, False]
    assert not ctrl.is_submitting("c")


def test_a_job_retried_in_the_next_batch_is_cancelled_late_before_it_is_submitted_again(
    controller,
) -> None:
    # The abandoned batch's late submission of "a" is cancelled on the
    # scheduler before batch 2 submits "a" again into the same folder.
    ctrl, fake = controller
    finished: list[int] = []
    ctrl.submit_batch_finished.connect(finished.append)
    gate = threading.Event()
    in_submit = threading.Event()

    def stuck() -> None:
        in_submit.set()
        gate.wait(5.0)

    fake.on_submit = stuck
    item = {"job_id": "a", "label": "a", "submit_kwargs": {"seed_name": "a"}}
    first = ctrl.submit_batch([item])
    assert in_submit.wait(2.0)
    ctrl.abandon_batch(first)
    fake.on_submit = None
    second = ctrl.submit_batch([item])

    gate.set()

    assert pump_until(lambda: finished == [first, second])
    assert [call[0] for call in fake.calls if call[0] in {"submit", "cancel_scheduler_job"}] == [
        "submit",
        "cancel_scheduler_job",
        "submit",
    ]
    assert ctrl.is_submitting("a") is False


def test_cancelling_one_batch_leaves_a_queued_batch_running(controller) -> None:
    ctrl, fake = controller
    finished: list[int] = []
    ctrl.submit_batch_finished.connect(finished.append)
    gate = threading.Event()
    in_submit = threading.Event()

    def stuck() -> None:
        in_submit.set()
        gate.wait(5.0)

    fake.on_submit = stuck
    first = ctrl.submit_batch([_batch_item(0), _batch_item(1)])
    second = ctrl.submit_batch([_batch_item(2)])
    assert in_submit.wait(2.0)
    ctrl.cancel_batch(first)
    fake.on_submit = None
    gate.set()

    assert pump_until(lambda: finished == [first, second])
    assert [call[1] for call in fake.calls if call[0] == "submit"] == ["s0", "s2"]


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
    ctrl.job_cancelled.connect(received.append)

    fake.job_cancelled.emit("job-1")

    assert pump_until(lambda: received == ["job-1"])


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


def test_check_previous_runs_reports_active_jobs_and_errors(
    controller, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start-again asks about each previous scheduler job before clearing anything."""

    ctrl, fake = controller
    answers = {"111": True, "222": False}

    def scheduler_job_is_active(remote_job_id: str) -> bool:
        if remote_job_id == "333":
            raise RuntimeError("sacct: command not found")
        return answers[remote_job_id]

    monkeypatch.setattr(fake, "scheduler_job_is_active", scheduler_job_is_active, raising=False)
    payloads: list[dict] = []
    ctrl.previous_runs_checked.connect(payloads.append)

    ctrl.check_previous_runs([("a", "111"), ("b", "222"), ("c", "333")])

    assert pump_until(lambda: payloads)
    assert payloads[0] == {"active": ["a"], "errors": {"c": "sacct: command not found"}}


def test_check_previous_runs_always_emits_even_on_unexpected_error(controller) -> None:
    ctrl, _fake = controller
    payloads: list[dict] = []
    ctrl.previous_runs_checked.connect(payloads.append)

    with pytest.raises(ValueError):
        ctrl._run_check_previous_runs([("only-one-element",)])

    assert payloads == [{"active": [], "errors": {}}]


def test_submit_batch_emits_progress_results_and_finished(controller) -> None:
    ctrl, fake = controller
    progress: list[tuple] = []
    results: list[tuple] = []
    finished: list[bool] = []
    ctrl.submit_batch_progress.connect(lambda _b, i, n, label: progress.append((i, n, label)))
    ctrl.job_submit_finished.connect(lambda _b, jid, rid, err: results.append((jid, rid, err)))
    ctrl.submit_batch_finished.connect(lambda _batch: finished.append(True))

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
    ctrl.job_submit_finished.connect(lambda _b, jid, rid, err: results.append((jid, rid, err)))
    ctrl.submit_batch_finished.connect(lambda _batch: finished.append(True))

    ctrl.submit_batch([_batch_item(0)])

    assert pump_until(lambda: finished)
    assert results == [("job-0", None, "boom")]


def test_submit_batch_stops_after_cancel(controller) -> None:
    ctrl, fake = controller
    results: list[tuple] = []
    finished: list[bool] = []
    ctrl.job_submit_finished.connect(lambda _b, jid, rid, err: results.append((jid, rid, err)))
    ctrl.submit_batch_finished.connect(lambda _batch: finished.append(True))
    # Cancel from inside the first submit so the second job is never submitted.
    batch = _cancel_from_first_submit(ctrl, fake)

    batch.set(ctrl.submit_batch([_batch_item(0), _batch_item(1), _batch_item(2)]))

    assert pump_until(lambda: finished)
    assert [r[0] for r in results] == ["job-0"]


def test_run_submit_batch_direct_covers_worker_body(controller) -> None:
    # Invoke the worker method directly (on the test thread) so its logic is
    # exercised deterministically and visibly to coverage; the threaded path is
    # covered by test_submit_batch_* above.
    ctrl, fake = controller
    results: list[tuple] = []
    finished: list[bool] = []
    ctrl.job_submit_finished.connect(lambda _b, jid, rid, err: results.append((jid, rid, err)))
    ctrl.submit_batch_finished.connect(lambda _batch: finished.append(True))

    ctrl._run_submit_batch(
        1, _Batch(submitting=set()), [_batch_item(0, delete="/old"), _batch_item(1)]
    )

    assert finished == [True]
    assert [r[0] for r in results] == ["job-0", "job-1"]
    assert ("rmtree", "/old") in fake.calls

    # A per-job submit failure is reported, not raised.
    results.clear()
    finished.clear()
    fake.submit_should_raise = True
    ctrl._run_submit_batch(2, _Batch(submitting=set()), [_batch_item(9)])
    assert results == [("job-9", None, "boom")]
    assert finished == [True]
    fake.submit_should_raise = False

    # Cancel short-circuits the loop before any job runs.
    results.clear()
    finished.clear()
    cancelled = _Batch(submitting=set())
    cancelled.cancelled.set()
    ctrl._run_submit_batch(3, cancelled, [_batch_item(2)])
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


def test_ssh_task_runner_skips_queued_work_once_closing(qapp) -> None:
    runner = cluster_controller._SshTaskRunner()
    ran: list[str] = []
    finished: list[object] = []
    runner.op_finished.connect(lambda rid, *_rest: finished.append(rid))

    runner.run_task(lambda: ran.append("task"))
    runner.closing.set()
    runner.run_task(lambda: ran.append("queued"))
    runner.run_async_op(1, "k", lambda: ran.append("op"))
    runner.run_final_task(lambda: ran.append("disconnect"))

    assert ran == ["task", "disconnect"]
    assert finished == []


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
    rid = ctrl.remove_remote_directory("/remote/dir")
    assert ("rmtree", "/remote/dir") in fake.calls
    assert ops and ops[-1] == (rid, ctrl.OP_REMOVE_REMOTE_DIR, None, None)


def test_shutdown_stops_worker_thread(qapp, monkeypatch) -> None:
    monkeypatch.setattr(cluster_controller, "ClusterJobRunner", _FakeRunner)
    ctrl = ClusterController(settings=None)
    fake = ctrl._runner
    assert ctrl._thread.isRunning() is True
    ctrl.shutdown()
    assert ctrl._thread.isRunning() is False
    assert not ctrl.results_arrived_while_closing
    # The disconnect must run synchronously on the worker before quit, otherwise
    # the worker's poll timer is left active and is later killed from the GUI
    # thread ("Timers cannot be stopped from another thread").
    assert ("shutdown",) in fake.calls
    # Idempotent.
    ctrl.shutdown()


def _quick_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the close's two waits, so a test of a stuck worker runs quickly."""

    monkeypatch.setattr(cluster_controller, "SHUTDOWN_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(cluster_controller, "SHUTDOWN_ABORT_SECONDS", 0.3)


def _start_stuck(ctrl: ClusterController, fake: _FakeRunner, operation: str) -> None:
    """Start ``operation`` on the worker and wait until it is stuck there."""

    fake.stall_in = operation
    if operation == "connect":
        ctrl.connect_async(_FakeConfig(), credentials=_FakeCredentials("pw"))
    elif operation == "submit":
        ctrl.submit_batch([{"job_id": "a", "label": "a", "submit_kwargs": {"seed_name": "a"}}])
    elif operation == "check":
        ctrl.check_jobs(["job-1"])
    elif operation == "progress":
        ctrl.read_geometry_progress(["job-1"])
    else:
        ctrl.remote_directory_exists("/remote/a")
    assert fake.stalled.wait(2.0)


_STUCK_OPERATIONS = ["connect", "submit", "check", "progress", "exists"]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("operation", _STUCK_OPERATIONS)
def test_shutdown_returns_when_closing_the_connection_frees_a_stuck_worker(
    controller, monkeypatch, operation: str
) -> None:
    ctrl, fake = controller
    _quick_shutdown(monkeypatch)
    fake.abort_releases = True
    _start_stuck(ctrl, fake, operation)

    started = time.monotonic()
    ctrl.shutdown()

    assert time.monotonic() - started < 3.0
    # Freed by the closed connection, the worker disconnected and stopped.
    assert ("abort",) in fake.calls
    assert ("shutdown",) in fake.calls
    assert ctrl._thread.isFinished()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("operation", _STUCK_OPERATIONS)
def test_shutdown_returns_when_nothing_frees_a_stuck_worker(
    controller, monkeypatch, operation: str
) -> None:
    ctrl, fake = controller
    _quick_shutdown(monkeypatch)
    _start_stuck(ctrl, fake, operation)

    started = time.monotonic()
    ctrl.shutdown()

    assert time.monotonic() - started < 3.0
    assert ctrl._thread.isRunning()
    assert ctrl.worker_abandoned
    # Released later, the worker runs nothing more but its disconnect, and stops.
    fake.release.set()
    assert ctrl._thread.wait(5000)
    assert fake.calls[-1] == ("shutdown",)


def _collect_submit_results(ctrl: ClusterController) -> list[tuple]:
    results: list[tuple] = []
    ctrl.job_submit_finished.connect(
        lambda _batch, job_id, remote_job_id, error: results.append((job_id, remote_job_id, error))
    )
    return results


def _items(*job_ids: str) -> list[dict]:
    return [
        {"job_id": job_id, "label": job_id, "submit_kwargs": {"seed_name": job_id}}
        for job_id in job_ids
    ]


_NOT_SUBMITTED = cluster_controller.CLOSED_BEFORE_SUBMITTING


@pytest.mark.timeout(60)
def test_closing_reports_the_jobs_a_batch_never_reached(controller, monkeypatch) -> None:
    ctrl, fake = controller
    _quick_shutdown(monkeypatch)
    results = _collect_submit_results(ctrl)
    fake.stall_in = "submit"
    ctrl.submit_batch(_items("a", "b"))
    ctrl.submit_batch(_items("c"))
    assert fake.stalled.wait(2.0)

    ctrl.shutdown()

    # Reported at once, before the close waits, so the project can record them.
    assert results[:2] == [("b", None, _NOT_SUBMITTED), ("c", None, _NOT_SUBMITTED)]
    assert not ctrl.is_submitting("b") and not ctrl.is_submitting("c")
    assert [call for call in fake.calls if call[0] == "submit"] == [("submit", "a")]


@pytest.mark.timeout(60)
def test_a_submit_answered_while_closing_reports_its_scheduler_job(controller, monkeypatch) -> None:
    ctrl, fake = controller
    results = _collect_submit_results(ctrl)
    fake.stall_in = "submit"
    ctrl.submit_batch(_items("a", "b"))
    assert fake.stalled.wait(2.0)
    # Well inside the default grace, so the connection is not closed under it.
    threading.Timer(0.05, fake.release.set).start()

    ctrl.shutdown()

    assert ("abort",) not in fake.calls
    assert pump_until(lambda: ("a", "JOB-1", "") in results)
    assert ("b", None, _NOT_SUBMITTED) in results
    # So the window saves again what arrived after its save.
    assert ctrl.results_arrived_while_closing


@pytest.mark.timeout(60)
def test_a_submit_ended_by_closing_the_connection_is_not_reported_failed(
    controller, monkeypatch
) -> None:
    ctrl, fake = controller
    _quick_shutdown(monkeypatch)
    results = _collect_submit_results(ctrl)
    finished: list[int] = []
    ctrl.submit_batch_finished.connect(finished.append)
    fake.stall_in = "submit"
    fake.abort_releases = True
    ctrl.submit_batch(_items("a"))
    assert fake.stalled.wait(2.0)

    ctrl.shutdown()

    # The scheduler may have taken it before the connection closed.
    assert pump_until(lambda: bool(finished))
    assert results == []


@pytest.mark.timeout(60)
def test_a_submit_closing_gives_up_on_stays_submitting(controller, monkeypatch) -> None:
    ctrl, fake = controller
    _quick_shutdown(monkeypatch)
    results = _collect_submit_results(ctrl)
    fake.stall_in = "submit"
    ctrl.submit_batch(_items("a"))
    assert fake.stalled.wait(2.0)

    ctrl.shutdown()

    # Whether the scheduler took it is unknown, so it is not reported as failed.
    assert ctrl.worker_abandoned
    assert ctrl.is_submitting("a")
    fake.release.set()
    assert ctrl._thread.wait(5000)
    assert not pump_until(lambda: bool(results), timeout=0.2)


@pytest.mark.timeout(60)
def test_a_batch_step_before_the_submit_cut_off_by_closing_reports_its_error(
    controller, monkeypatch
) -> None:
    # Nothing reached the scheduler for this job, so its failure is certain.
    ctrl, fake = controller
    _quick_shutdown(monkeypatch)
    results = _collect_submit_results(ctrl)
    fake.stall_in = "scheduler_check"
    fake.abort_releases = True
    item = _batch_item(0, delete="/remote/old")
    item["delete_only_if_finished"] = "OLD-1"
    ctrl.submit_batch([item])
    assert fake.stalled.wait(2.0)

    ctrl.shutdown()

    assert pump_until(lambda: (item["job_id"], None, "the SSH connection was closed") in results)
    assert not any(call[0] == "submit" for call in fake.calls)


@pytest.mark.timeout(60)
def test_evidence_gathered_after_closing_closed_the_connection_is_dropped(
    controller, monkeypatch
) -> None:
    ctrl, fake = controller
    _quick_shutdown(monkeypatch)
    checked: list[object] = []
    ctrl.job_checked.connect(lambda job_id, _evidence: checked.append(job_id))
    ctrl.queue_checked.connect(checked.append)
    fake.abort_releases = True
    _start_stuck(ctrl, fake, "check")

    ctrl.shutdown()

    # The check went on after the connection closed: what it saw may be the
    # closed connection's doing.
    assert ("check", ("job-1",)) in fake.calls
    assert not pump_until(lambda: bool(checked), timeout=0.2)


def test_a_late_cancel_failure_arriving_while_closing_is_not_reported(controller) -> None:
    ctrl, fake = controller
    fake.cancel_should_raise = True
    failures: list[tuple] = []
    ctrl.abandoned_job_cancel_failed.connect(lambda *args: failures.append(args))
    worker = threading.Thread(target=lambda: ctrl._run_cancel_abandoned("a", "JOB-1"))
    worker.start()
    worker.join()

    ctrl.shutdown()

    # A warning now would hold the closing window open until it is dismissed.
    assert not pump_until(lambda: bool(failures), timeout=0.2)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("abort_releases", [True, False], ids=["freed", "still-stuck"])
def test_closing_reports_the_submit_it_cut_off_and_only_that_one(
    controller, monkeypatch, abort_releases: bool
) -> None:
    ctrl, fake = controller
    _quick_shutdown(monkeypatch)
    cut_off: list[object] = []
    ctrl.submission_cut_off.connect(cut_off.append)
    fake.stall_in = "submit"
    fake.abort_releases = abort_releases
    ctrl.submit_batch(_items("a", "b"))
    assert fake.stalled.wait(2.0)

    ctrl.shutdown()

    # "b" was never started; it is reported as not submitted instead.
    assert cut_off == ["a"]
    assert ctrl.results_arrived_while_closing


@pytest.mark.timeout(60)
def test_a_submit_answered_while_closing_is_not_reported_cut_off(controller) -> None:
    ctrl, fake = controller
    cut_off: list[object] = []
    ctrl.submission_cut_off.connect(cut_off.append)
    fake.stall_in = "submit"
    ctrl.submit_batch(_items("a"))
    assert fake.stalled.wait(2.0)
    threading.Timer(0.05, fake.release.set).start()

    ctrl.shutdown()

    assert cut_off == []


def test_an_abandoned_worker_keeps_the_process_from_finalising_until_it_stops(
    controller, monkeypatch
) -> None:
    ctrl, fake = controller
    _quick_shutdown(monkeypatch)
    _start_stuck(ctrl, fake, "check")

    ctrl.shutdown()

    assert cluster_controller.worker_threads_abandoned()
    fake.release.set()
    assert ctrl._thread.wait(5000)
    assert not cluster_controller.worker_threads_abandoned()


def test_real_runner_is_disconnected_by_shutdown(qapp) -> None:
    from mufinder.app.cluster_controller import ClusterController as RealController

    ctrl = RealController(settings=None)
    events: list[bool] = []
    ctrl.connection_status_changed.connect(events.append)

    ctrl.shutdown()

    assert ctrl._thread.isRunning() is False
    assert ctrl._runner.is_connected is False


def test_run_submit_batch_deletes_a_finished_runs_directory(controller) -> None:
    ctrl, fake = controller
    results: list[tuple] = []
    ctrl.job_submit_finished.connect(lambda _b, jid, rid, err: results.append((jid, rid, err)))

    ctrl._run_submit_batch(
        1,
        _Batch(submitting=set()),
        [{**_batch_item(0, delete="/old"), "delete_only_if_finished": "777"}],
    )

    assert ("scheduler_check", "777") in fake.calls
    assert ("rmtree", "/old") in fake.calls
    assert results == [("job-0", "JOB-1", "")]


def test_run_submit_batch_refuses_to_delete_a_running_jobs_directory(controller) -> None:
    ctrl, fake = controller
    fake.active_scheduler_jobs = {"777"}
    results: list[tuple] = []
    ctrl.job_submit_finished.connect(lambda _b, jid, rid, err: results.append((jid, rid, err)))

    ctrl._run_submit_batch(
        1,
        _Batch(submitting=set()),
        [{**_batch_item(0, delete="/old"), "delete_only_if_finished": "777"}],
    )

    assert ("rmtree", "/old") not in fake.calls
    assert not any(call[0] == "submit" for call in fake.calls)
    assert results == [
        (
            "job-0",
            None,
            "scheduler job 777 is still queued or running, so its directory /old was not deleted",
        )
    ]


def test_checkpoint_probe_is_not_held_up_by_progress_reads_of_large_logs(qapp, tmp_path) -> None:
    # Several resumed relaxations with large accumulated logs are sent for their
    # first progress read, then a checkpoint probe is queued behind them.
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "geometry" / "Si2_bfgs.castep"
    log = b" filler line of CASTEP output\n" * 200_000 + fixture.read_bytes()
    assert len(log) > 2 * REMOTE_READ_CHUNK_BYTES
    ssh = GrowingOutputSSH()
    ctrl = ClusterController(settings=None)
    try:
        runner = ctrl._runner
        runner._ssh = ssh
        job_ids = [uuid4() for _ in range(3)]
        targets = []
        for index, job_id in enumerate(job_ids):
            ssh.remote_files[f"/remote/job{index}/seed.castep"] = log
            targets.append(
                CheckTarget(
                    job_id=job_id,
                    remote_job_id=str(index),
                    remote_dir=f"/remote/job{index}",
                    seed_name="seed",
                    local_output_dir=tmp_path,
                )
            )
        ssh.remote_files["/remote/job0/seed.check"] = b"checkpoint"
        progress: dict = {}
        ctrl.geometry_progress_read.connect(progress.update)
        probes: list[dict] = []
        ctrl.remote_files_checked.connect(probes.append)

        ctrl.read_geometry_progress(targets)
        ctrl.check_remote_files([(job_ids[0], "/remote/job0/seed.check")])

        assert pump_until(lambda: probes and len(progress) == 3, timeout=20.0)
    finally:
        ctrl.shutdown()

    # The probe waited for one bounded read, not for three whole logs.
    assert ssh.calls.index(("is_file", "/remote/job0/seed.check")) == 1
    assert probes[0]["existing"] == ["/remote/job0/seed.check"]
    assert progress == dict.fromkeys(job_ids, parse_geometry_progress(fixture))
