"""Threaded facade around :class:`ClusterJobRunner`.

All paramiko SSH/SFTP work is blocking, and on the Qt GUI thread it would
freeze the window: during cluster submission and, worst of all, whenever a
check finds finished jobs and downloads their output.

``ClusterController`` moves the runner onto a dedicated worker ``QThread`` so
that *all* SSH activity — connection, job checks, output retrieval, submission,
cancellation — executes off the GUI thread. It forwards the runner's signals,
mirrors its connection state, and is what ``main_window`` holds.

All GUI-initiated work is marshalled onto the worker **asynchronously**: the GUI
thread never blocks waiting on the worker. This matters because the worker also
runs the job checks, each of which serially downloads and parses the output of
every finished job it finds (minutes, in the worst case). A GUI-thread call that
blocked on the worker would freeze the window until that batch drained. Nothing
here uses ``BlockingQueuedConnection``. Even :meth:`ClusterController.shutdown`
waits for the worker for a bounded time only, so a stuck SSH call cannot hold
the window open.

Two asynchronous styles are used:

* **Fire-and-forget with forwarded signals** (``check_jobs``, ``cancel``,
  ``connect_async``/``connect_finished``,
  ``check_remote_dirs``/``remote_dirs_checked``,
  ``check_remote_files``/``remote_files_checked``,
  ``check_previous_runs``/``previous_runs_checked``,
  ``read_geometry_progress``/``geometry_progress_read``, ``submit_batch`` and its
  progress signals): the GUI returns immediately and results arrive later via a
  purpose-built signal. ``submit_batch`` returns a batch ID that its signals
  carry, and each batch has its own cancel state.
* **Request/completion bridge** (the low-level
  ``connect_to_cluster`` / ``submit`` / ``remote_directory_exists`` /
  ``remove_remote_directory`` helpers): the operation is queued on the worker and
  its outcome is delivered back to the GUI thread via
  :attr:`ClusterController.operation_finished`, keyed by a per-call ``request_id``
  and a ``kind`` discriminator so multiple in-flight operations route to their own
  continuation. The completion signal fires on **every** exit path — success,
  error, or an exception escaping the queued task — so a UI element disabled while
  an operation is pending is always re-enabled.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
import threading
from typing import Any
from uuid import UUID

from PySide6.QtCore import (
    QCoreApplication,
    QObject,
    Qt,
    QThread,
    Signal,
    Slot,
)

from mufinder.domain.cluster import ClusterConfig
from mufinder.services.cluster_runner import CheckTarget, ClusterJobRunner
from mufinder.services.ssh_client import SSHCredentials

#: How long :meth:`ClusterController.shutdown` waits for the worker to finish
#: what it holds and disconnect, in seconds.
SHUTDOWN_GRACE_SECONDS = 2.0
#: How long it then waits once the connection is closed under the worker.
SHUTDOWN_ABORT_SECONDS = 3.0
#: The error a batch's job reports when closing stops the batch before it.
CLOSED_BEFORE_SUBMITTING = "MuFinder closed before submitting this job."

#: Worker threads closing gave up on, with the objects that live on them. A
#: ``QThread`` destroyed while its thread runs aborts the process, so they are
#: kept for as long as the process lives.
_abandoned_workers: list[tuple[QThread, QObject, QObject]] = []


def worker_threads_abandoned() -> bool:
    """Return whether closing gave up on a worker thread that is still running.

    The process cannot finalise normally then: Python's clean-up would destroy
    the running thread's ``QThread`` and abort. ``mufinder.main`` ends the
    process without that clean-up instead.
    """

    _abandoned_workers[:] = [entry for entry in _abandoned_workers if entry[0].isRunning()]
    return bool(_abandoned_workers)


@dataclass
class _Batch:
    """One :meth:`ClusterController.submit_batch` call.

    ``cancelled`` is set on the GUI thread and read by the worker between jobs.
    ``abandoned``, ``cancelled_on_worker`` and ``started`` are shared by both
    threads under ``lock``: the worker cancels on the scheduler a job of an
    abandoned batch that it submits after all, and the GUI thread cancels one
    whose result left the worker before the batch was abandoned. The worker
    adds a job to ``started`` before it touches the cluster for it, so a job
    left out once ``cancelled`` is set under the lock is never submitted.
    ``job_ids`` and ``submitting`` are GUI-thread only.
    """

    submitting: set[Any]
    job_ids: tuple[Any, ...] = ()
    cancelled: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    abandoned: bool = False
    cancelled_on_worker: set[Any] = field(default_factory=set)
    started: set[Any] = field(default_factory=set)


class _SshTaskRunner(QObject):
    """Executes callables on the thread it lives on (the SSH worker thread).

    Kept separate from :class:`ClusterJobRunner` so the service layer stays free
    of threading plumbing. Both objects are moved to the same worker thread, so
    a callable invoked here that calls a runner method runs that method on the
    worker thread too.
    """

    #: Emitted on the worker thread when a request/completion op finishes, with
    #: ``(request_id, kind, result, error)`` where exactly one of ``result`` /
    #: ``error`` is meaningful. The controller forwards it to the GUI thread.
    op_finished = Signal(object, str, object, object)

    def __init__(self) -> None:
        super().__init__()
        #: Set on the GUI thread when the application closes: work queued
        #: before then is skipped, except the final disconnect.
        self.closing = threading.Event()

    @Slot(object)
    def run_task(self, task: Callable[[], Any]) -> None:
        """Run a fire-and-forget callable on the worker thread."""

        if not self.closing.is_set():
            task()

    @Slot(object)
    def run_final_task(self, task: Callable[[], Any]) -> None:
        """Run the closing disconnect, which runs even while ``closing`` is set."""

        task()

    @Slot(object, str, object)
    def run_async_op(self, request_id: Any, kind: str, task: Callable[[], Any]) -> None:
        """Run a queued request/completion op, reporting the outcome via a signal.

        Runs on the worker thread. ``op_finished`` is emitted on **every** exit
        path — success, a caught exception, and even a ``BaseException`` — so the
        GUI continuation always fires and a UI element disabled for the duration
        of the op is never stuck disabled. The emit is queued (cross-thread) back
        to the GUI, so the GUI thread never waits here. An op still queued when
        the application closes is skipped: nothing waits for its outcome then.
        """

        if self.closing.is_set():
            return
        result: Any = None
        error: BaseException | None = None
        try:
            result = task()
        except BaseException as exc:  # noqa: BLE001 - delivered to the GUI as data
            error = exc
        self.op_finished.emit(request_id, kind, result, error)


class ClusterController(QObject):
    """GUI-thread facade that runs cluster SSH work on a worker thread."""

    # Re-exposed runner signals (emitted on the GUI thread).
    job_cancelled = Signal(object)
    #: ``(job_id, JobEvidence)``: what a check saw of one job.
    job_checked = Signal(object, object)
    connection_status_changed = Signal(bool)
    queue_checked = Signal(object)
    poll_failed = Signal(str)
    geometry_progress_read = Signal(object)

    # Result of an asynchronous connect: None on success, else the exception.
    connect_finished = Signal(object)

    # Outcome of a request/completion bridge op, delivered on the GUI thread as
    # ``(request_id, kind, result, error)``. ``kind`` is one of the OP_* constants
    # below; ``result`` is the return value on success and ``error`` is the
    # exception on failure (exactly one is meaningful). Fires on every exit path.
    operation_finished = Signal(object, str, object, object)

    # Asynchronous batch-submission progress (all emitted on the GUI thread).
    remote_dirs_checked = Signal(object)  # {"existing": [str], "errors": {job_id: str}}
    remote_files_checked = Signal(object)  # {"existing": [str], "errors": {job_id: str}}
    previous_runs_checked = Signal(object)  # {"active": [job_id], "errors": {job_id: str}}
    # Each batch signal carries the batch ID :meth:`submit_batch` returned, so
    # a listener follows its own batch only.
    submit_batch_progress = Signal(int, int, int, str)  # batch, index, total, job label
    # batch, job_id, remote_job_id|None, error
    job_submit_finished = Signal(int, object, object, str)
    submit_batch_finished = Signal(int)  # batch
    # job_id, remote_job_id, error: a job submitted after its batch was
    # abandoned could not be cancelled on the scheduler.
    abandoned_job_cancel_failed = Signal(object, str, str)
    # job_id: closing cut off this job's submit, or gave up waiting for it, so
    # whether the scheduler took the job is unknown. Emitted by shutdown.
    submission_cut_off = Signal(object)

    # Internal: a check pass left the worker, however it ended.
    _check_finished = Signal()
    # Internal: a late cancel failed on the worker; forwarded unless closing.
    _late_cancel_failed = Signal(object, str, str)
    # Internal: marshals a fire-and-forget callable onto the worker thread.
    _async_task = Signal(object)
    # Internal: marshals the closing disconnect onto the worker thread.
    _final_task = Signal(object)
    # Internal: marshals a request/completion op onto the worker thread as
    # ``(request_id, kind, task)``.
    _async_op = Signal(object, str, object)

    # ``kind`` discriminators for :attr:`operation_finished`.
    OP_CONNECT = "connect"
    OP_SUBMIT = "submit"
    OP_REMOTE_DIR_EXISTS = "remote_directory_exists"
    OP_REMOVE_REMOTE_DIR = "remove_remote_directory"

    def __init__(self, parent: QObject | None = None, *, settings: Any = None) -> None:
        super().__init__(parent)

        self._thread = QThread()
        self._thread.setObjectName("mufinder-cluster-ssh")

        # The runner and the task executor live on the worker thread. They must
        # be parent-less to be moved there.
        self._runner = ClusterJobRunner(settings=settings)
        self._tasks = _SshTaskRunner()
        self._runner.moveToThread(self._thread)
        self._tasks.moveToThread(self._thread)

        # Mirrored connection state, read by the GUI without touching the live
        # paramiko transport (which lives on the worker thread).
        self._connected = False
        self._connected_cluster_name: str | None = None
        self._connected_username: str | None = None
        self._pending_connect: tuple[str, str | None] | None = None
        # Batches handed to the worker that have not finished, by batch ID. A
        # job in one of them has no scheduler job ID yet, by design.
        self._batches: dict[int, _Batch] = {}
        self._batch_ids = count(1)

        # Set once shutdown starts; ``_abandoned`` once it gave up on the worker.
        self._closing = False
        self._abandoned = False
        # Set when shutdown closes the SSH connection under the worker.
        self._connection_closed = False
        # Set when a job result reaches the window after shutdown began.
        self._results_while_closing = False
        # Jobs a batch had in hand when shutdown began, and those of them
        # whose submit answered while closing.
        self._in_hand_at_close: list[Any] = []
        self._answered_while_closing: set[Any] = set()

        # Monotonic request-id source for the request/completion bridge. Only ever
        # incremented on the GUI thread (single-threaded), so no lock is needed.
        self._request_ids = count(1)

        # Check passes queued on the worker or running there, so the
        # once-a-minute check is not queued behind one that has not finished.
        self._checks_in_flight = 0

        # Forward worker signals to the GUI thread. These cross-thread
        # connections are queued automatically.
        self._runner.job_cancelled.connect(self._forward_job_cancelled)
        self._runner.job_checked.connect(self._forward_job_checked)
        self._runner.queue_checked.connect(self._forward_queue_checked)
        self._runner.poll_failed.connect(self._forward_poll_failed)
        self._runner.geometry_progress_read.connect(self.geometry_progress_read)
        self._runner.connection_status_changed.connect(self._on_connection_status_changed)
        self._check_finished.connect(self._on_check_finished)
        self._late_cancel_failed.connect(self._forward_late_cancel_failed)
        # Connected before any outside listener, so a job stops counting as
        # being submitted before its result is applied.
        self.job_submit_finished.connect(self._on_job_submit_finished)
        self.submit_batch_finished.connect(self._on_submit_batch_finished)

        self._async_task.connect(self._tasks.run_task, Qt.ConnectionType.QueuedConnection)
        self._final_task.connect(self._tasks.run_final_task, Qt.ConnectionType.QueuedConnection)
        self._async_op.connect(self._tasks.run_async_op, Qt.ConnectionType.QueuedConnection)
        # Worker-thread completion forwarded to the GUI thread (queued automatically).
        self._tasks.op_finished.connect(self.operation_finished)

        self._thread.start()

    # ------------------------------------------------------------------
    # Connection-state mirror (GUI-thread safe; never touches the transport)
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def connected_cluster_name(self) -> str | None:
        return self._connected_cluster_name if self._connected else None

    @property
    def connected_username(self) -> str | None:
        return self._connected_username if self._connected else None

    # ------------------------------------------------------------------
    # Submissions in flight (GUI thread)
    # ------------------------------------------------------------------
    def is_submitting(self, job_id: UUID) -> bool:
        """Return whether ``job_id`` was handed to the worker and has no answer yet."""

        return any(job_id in batch.submitting for batch in self._batches.values())

    @Slot(int, object, object, str)
    def _on_job_submit_finished(
        self, batch_id: int, job_id: Any, remote_job_id: Any, _error: str
    ) -> None:
        if self._closing:
            self._results_while_closing = True
            self._answered_while_closing.add(job_id)
        batch = self._batches.get(batch_id)
        if batch is None:
            return
        batch.submitting.discard(job_id)
        if not remote_job_id:
            return
        with batch.lock:
            # Submitted before the worker saw the batch abandoned: cancel it here.
            late = batch.abandoned and job_id not in batch.cancelled_on_worker
        if late:
            remote = str(remote_job_id)
            self._async_task.emit(lambda: self._run_cancel_abandoned(job_id, remote))

    def _run_cancel_abandoned(self, job_id: Any, remote_job_id: str) -> None:
        try:
            self._runner.cancel_scheduler_job(remote_job_id)
        except Exception as exc:  # noqa: BLE001 - reported to the GUI
            self._late_cancel_failed.emit(job_id, remote_job_id, str(exc))

    # While closing, a failure is no news and a warning would hold the window
    # open, so the two failure signals are dropped as they arrive. Evidence
    # gathered once the connection was closed under the worker may be the
    # closed connection's doing, so it is dropped too; the next check looks again.
    @Slot(str)
    def _forward_poll_failed(self, message: str) -> None:
        if not self._closing:
            self.poll_failed.emit(message)

    @Slot(object, str, str)
    def _forward_late_cancel_failed(self, job_id: Any, remote_job_id: str, error: str) -> None:
        if not self._closing:
            self.abandoned_job_cancel_failed.emit(job_id, remote_job_id, error)

    @Slot(object, object)
    def _forward_job_checked(self, job_id: Any, evidence: Any) -> None:
        if not self._connection_closed:
            self._results_while_closing |= self._closing
            self.job_checked.emit(job_id, evidence)

    @Slot(object)
    def _forward_job_cancelled(self, job_id: Any) -> None:
        self._results_while_closing |= self._closing
        self.job_cancelled.emit(job_id)

    @Slot(object)
    def _forward_queue_checked(self, snapshot: Any) -> None:
        if not self._connection_closed:
            self.queue_checked.emit(snapshot)

    @Slot(int)
    def _on_submit_batch_finished(self, batch_id: int) -> None:
        self._batches.pop(batch_id, None)

    @Slot()
    def _on_check_finished(self) -> None:
        self._checks_in_flight = max(0, self._checks_in_flight - 1)

    @Slot(bool)
    def _on_connection_status_changed(self, connected: bool) -> None:
        self._connected = bool(connected)
        if connected:
            # Adopt the identity of the connect that is in flight (set on the GUI
            # thread before the worker ran). This carries the cluster name for an
            # async connect, where there is no eager set to read it from. Consume
            # it so a later status change cannot re-adopt a stale identity.
            if self._pending_connect is not None:
                self._connected_cluster_name, self._connected_username = self._pending_connect
                self._pending_connect = None
        else:
            self._connected_cluster_name = None
            self._connected_username = None
        self.connection_status_changed.emit(bool(connected))

    # ------------------------------------------------------------------
    # Pure pass-through helpers (no SSH; safe to call on the GUI thread)
    # ------------------------------------------------------------------
    def remote_job_directory(self, *, working_directory: str, seed_name: str) -> str:
        return ClusterJobRunner.remote_job_directory(
            working_directory=working_directory,
            seed_name=seed_name,
        )

    def invalid_remote_job_labels(self, labels: list[str]) -> list[str]:
        return ClusterJobRunner.invalid_remote_job_labels(labels)

    # ------------------------------------------------------------------
    # Request/completion bridge (queued on the worker; outcome via a signal)
    # ------------------------------------------------------------------
    def connect_to_cluster(
        self,
        config: ClusterConfig,
        password: str | None = None,
        *,
        credentials: SSHCredentials | None = None,
    ) -> object:
        """Connect on the worker thread; outcome via :attr:`operation_finished`.

        Returns immediately with the request id. On success the mirrored
        connection identity is adopted by the queued ``connection_status_changed``
        forwarded from the runner (and by the eager ``_pending_connect`` set here);
        the completion carries ``None`` as its error. See :meth:`connect_async`
        for the higher-level flow ``main_window`` uses.
        """

        self._pending_connect = (config.name, config.username.strip() or None)
        return self._run_async_op(
            self.OP_CONNECT,
            lambda: self._runner.connect_to_cluster(
                config,
                password=password,
                credentials=credentials,
            ),
        )

    def submit(self, **kwargs: Any) -> object:
        """Submit one job on the worker thread; outcome via :attr:`operation_finished`.

        Returns the request id immediately. On success the completion carries the
        remote scheduler id as its ``result``; on failure the raised exception is
        carried as ``error``.
        """

        return self._run_async_op(self.OP_SUBMIT, lambda: self._runner.submit(**kwargs))

    def remote_directory_exists(self, remote_directory: str) -> object:
        """Stat a remote directory on the worker thread; outcome via a signal.

        Returns the request id immediately; the completion carries the boolean
        result (or the exception on failure).
        """

        return self._run_async_op(
            self.OP_REMOTE_DIR_EXISTS,
            lambda: self._runner.remote_directory_exists(remote_directory),
        )

    def remove_remote_directory(self, remote_directory: str) -> object:
        """Remove a remote directory on the worker thread; outcome via a signal."""

        return self._run_async_op(
            self.OP_REMOVE_REMOTE_DIR,
            lambda: self._runner.remove_remote_directory(remote_directory),
        )

    # ------------------------------------------------------------------
    # Asynchronous (fire-and-forget) operations
    # ------------------------------------------------------------------
    def check_jobs(
        self, targets: list[CheckTarget], *, skip_if_busy: bool = False, refresh: bool = False
    ) -> bool:
        """Look at ``targets`` on the worker; evidence arrives via :attr:`job_checked`.

        Returns whether the check was queued. ``skip_if_busy`` drops it while an
        earlier check is still on the worker, so a slow download does not pile
        up one check per minute behind it; otherwise it runs after that check.
        ``refresh`` marks a Refresh Queue pass in its :attr:`queue_checked`
        snapshot.
        """

        if skip_if_busy and self._checks_in_flight:
            return False
        self._checks_in_flight += 1
        batch = list(targets)
        self._async_task.emit(lambda: self._run_check(batch, refresh))
        return True

    def _run_check(self, targets: list[CheckTarget], refresh: bool) -> None:
        # Runs on the worker. The finish is always reported, or one error
        # would leave every later once-a-minute check skipped.
        try:
            self._runner.check_jobs(targets, refresh=refresh)
        finally:
            self._check_finished.emit()

    def forget_progress(self) -> None:
        self._async_task.emit(self._runner.forget_progress)

    def read_geometry_progress(self, targets: list[CheckTarget]) -> None:
        """Read jobs' geometry progress on the worker; results via :attr:`geometry_progress_read`."""

        batch = list(targets)
        self._async_task.emit(lambda: self._runner.read_geometry_progress(batch))

    def cancel(self, job_id: Any, remote_job_id: str = "") -> None:
        self._async_task.emit(lambda: self._runner.cancel(job_id, remote_job_id))

    def connect_async(self, config: ClusterConfig, *, credentials: SSHCredentials) -> None:
        """Connect on the worker thread; result is delivered via connect_finished.

        Keeps the GUI responsive during the (often multi-second) SSH handshake.
        The host-key trust retry is driven by the GUI: on an unknown-host-key
        error it saves the key and calls this again, and the next attempt trusts
        it — so no cross-thread wait is needed here.
        """

        self._pending_connect = (config.name, config.username.strip() or None)
        self._async_task.emit(lambda: self._run_connect(config, credentials))

    def _run_connect(self, config: ClusterConfig, credentials: SSHCredentials) -> None:
        try:
            if config.jump_host_enabled:
                self._runner.connect_to_cluster(config, credentials=credentials)
            else:
                self._runner.connect_to_cluster(config, password=credentials.target_secret)
        except Exception as exc:  # noqa: BLE001 - delivered to the GUI to handle
            self.connect_finished.emit(exc)
            return
        self.connect_finished.emit(None)

    # ------------------------------------------------------------------
    # Asynchronous batch submission (runs SSH off the GUI thread)
    # ------------------------------------------------------------------
    def check_remote_dirs(self, items: list[tuple[Any, str]]) -> None:
        """Stat each ``(job_id, remote_dir)`` on the worker thread.

        Emits :attr:`remote_dirs_checked` with the directories that already exist
        and any per-job errors.
        """

        self._async_task.emit(lambda: self._run_check_remote_dirs(items))

    def _run_check_remote_dirs(self, items: list[tuple[Any, str]]) -> None:
        existing: list[str] = []
        errors: dict[Any, str] = {}
        # The result is always emitted (finally), even if an unexpected error
        # escapes the loop body: the GUI waits for this signal in a nested event
        # loop with no Cancel button, so a missing emit would freeze the app.
        try:
            for job_id, remote_dir in items:
                try:
                    if self._runner.remote_directory_exists(remote_dir):
                        existing.append(remote_dir)
                except Exception as exc:  # noqa: BLE001 - reported back to the GUI
                    errors[job_id] = str(exc)
        finally:
            self.remote_dirs_checked.emit({"existing": existing, "errors": errors})

    def check_remote_files(self, targets: list[tuple[Any, str]]) -> None:
        """Stat each ``(job_id, remote_path)`` on the worker thread.

        Emits :attr:`remote_files_checked` with the paths that exist and any
        per-job errors. Feeds the Continue-relaxation dialog's checkpoint probe
        (``{seed}.check`` existence) without blocking the GUI thread; the
        dialog must await the signal rather than call
        :meth:`ClusterJobRunner.remote_file_exists` directly, mirroring
        :meth:`check_remote_dirs`.
        """

        self._async_task.emit(lambda: self._run_check_remote_files(targets))

    def _run_check_remote_files(self, targets: list[tuple[Any, str]]) -> None:
        existing: list[str] = []
        errors: dict[Any, str] = {}
        # The result is always emitted (finally), even if an unexpected error
        # escapes the loop body: a caller awaiting this signal in a nested event
        # loop (see ``check_remote_dirs``) must never be left hanging.
        try:
            for job_id, remote_path in targets:
                try:
                    if self._runner.remote_file_exists(remote_path):
                        existing.append(remote_path)
                except Exception as exc:  # noqa: BLE001 - reported back to the GUI
                    errors[job_id] = str(exc)
        finally:
            self.remote_files_checked.emit({"existing": existing, "errors": errors})

    def check_previous_runs(self, targets: list[tuple[Any, str]]) -> None:
        """Ask the scheduler about each ``(job_id, remote_job_id)`` on the worker thread.

        Emits :attr:`previous_runs_checked` with the jobs whose previous
        scheduler job is still queued or running
        (:meth:`ClusterJobRunner.scheduler_job_is_active`) and an error per job
        the scheduler could not be asked about. A submission that starts jobs
        again asks this before anything is cleared, so a run still alive on the
        cluster keeps its folders and its record.
        """

        self._async_task.emit(lambda: self._run_check_previous_runs(targets))

    def _run_check_previous_runs(self, targets: list[tuple[Any, str]]) -> None:
        active: list[Any] = []
        errors: dict[Any, str] = {}
        # Always emitted, so a caller awaiting it in a nested event loop is never left hanging.
        try:
            for job_id, remote_job_id in targets:
                try:
                    if self._runner.scheduler_job_is_active(remote_job_id):
                        active.append(job_id)
                except Exception as exc:  # noqa: BLE001 - reported back to the GUI
                    errors[job_id] = str(exc) or type(exc).__name__
        finally:
            self.previous_runs_checked.emit({"active": active, "errors": errors})

    def submit_batch(self, items: list[dict[str, Any]]) -> int:
        """Submit a batch of prepared jobs on the worker thread.

        Each item is ``{"job_id", "label", "submit_kwargs", "remote_dir_to_delete"}``,
        optionally with ``"delete_only_if_finished"``: the scheduler id of the
        run that used ``remote_dir_to_delete``. The directory is then deleted
        only once the scheduler confirms that run is no longer queued or
        running; otherwise the item fails without deleting anything.
        Returns the batch's ID. Batches run one after another on the worker;
        each has its own cancel state, and each of its signals carries its ID.
        Emits :attr:`submit_batch_progress` before each job, :attr:`job_submit_finished`
        with the result of each, and :attr:`submit_batch_finished` once (even when
        cancelled or on error).
        """

        batch_id = next(self._batch_ids)
        job_ids = tuple(item["job_id"] for item in items)
        batch = _Batch(submitting=set(job_ids), job_ids=job_ids)
        self._batches[batch_id] = batch
        self._async_task.emit(lambda: self._run_submit_batch(batch_id, batch, items))
        return batch_id

    def cancel_batch(self, batch_id: int) -> None:
        """Stop batch ``batch_id`` before its next job; other batches go on.

        The job the worker has in hand finishes submitting and its result is
        reported as usual.
        """

        batch = self._batches.get(batch_id)
        if batch is not None:
            batch.cancelled.set()

    def abandon_batch(self, batch_id: int) -> None:
        """Stop batch ``batch_id`` for a caller that no longer waits for its results.

        Cancels like :meth:`cancel_batch`, and a job of the batch that is
        submitted after all is cancelled on the scheduler, since the caller has
        already recorded it as cancelled. The worker cancels it before it moves
        on to another batch, which may submit the same job again.
        """

        batch = self._batches.get(batch_id)
        if batch is None:
            return
        with batch.lock:
            batch.abandoned = True
        batch.cancelled.set()

    def _run_submit_batch(self, batch_id: int, batch: _Batch, items: list[dict[str, Any]]) -> None:
        total = len(items)
        try:
            for index, item in enumerate(items, start=1):
                job_id = item["job_id"]
                with batch.lock:
                    if batch.cancelled.is_set():
                        break
                    batch.started.add(job_id)
                self.submit_batch_progress.emit(batch_id, index, total, item.get("label", ""))
                submitting = False
                try:
                    delete_dir = item.get("remote_dir_to_delete")
                    if delete_dir:
                        previous_run = item.get("delete_only_if_finished")
                        if previous_run and self._runner.scheduler_job_is_active(previous_run):
                            raise RuntimeError(
                                f"scheduler job {previous_run} is still queued or running, "
                                f"so its directory {delete_dir} was not deleted"
                            )
                        self._runner.remove_remote_directory(delete_dir)
                    submitting = True
                    remote_job_id = self._runner.submit(**item["submit_kwargs"])
                except Exception as exc:  # noqa: BLE001 - reported per job to the GUI
                    if submitting and self._connection_closed:
                        # Closing cut the submission off: whether the scheduler
                        # took the job is unknown, so it is not reported failed.
                        break
                    self.job_submit_finished.emit(batch_id, job_id, None, str(exc))
                    continue
                with batch.lock:
                    cancel_now = batch.abandoned and bool(remote_job_id)
                    if cancel_now:
                        batch.cancelled_on_worker.add(job_id)
                if cancel_now:
                    self._run_cancel_abandoned(job_id, str(remote_job_id))
                self.job_submit_finished.emit(batch_id, job_id, remote_job_id, "")
        finally:
            self.submit_batch_finished.emit(batch_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def results_arrived_while_closing(self) -> bool:
        """Whether a job's result reached the window after :meth:`shutdown` began.

        The window saved the project before closing the connection, so it
        saves again when this holds.
        """

        return self._results_while_closing

    @property
    def worker_abandoned(self) -> bool:
        """Whether :meth:`shutdown` gave up on a worker that did not stop."""

        return self._abandoned

    def shutdown(self, *, wait: Callable[[threading.Event, float], bool] | None = None) -> None:
        """Disconnect and stop the worker thread, waiting a bounded time at most.

        Safe to call more than once. Queued work that has not started is
        skipped, and every submission batch stops before its next job; a job
        it never reached reports :data:`CLOSED_BEFORE_SUBMITTING` at once, so
        the project records it. The disconnect then runs *on the worker
        thread*, which owns the connection, and the same task quits the
        thread's event loop, so the loop cannot end before the disconnect.

        ``wait(event, seconds)`` waits for that task and returns whether it
        ran; the window passes one that shows a busy dialog. When the worker
        is still busy after :data:`SHUTDOWN_GRACE_SECONDS`, the SSH connection
        is closed under it from this thread, which ends any SSH call it is
        stuck in; a submission that ends so reports no result, since whether
        the scheduler took the job is unknown, but :attr:`submission_cut_off`
        instead, as does one still running when shutdown gives up. If it has still not stopped after
        :data:`SHUTDOWN_ABORT_SECONDS`, it is given up on: this controller's
        signals are blocked so nothing it does later reaches the window, and
        :func:`worker_threads_abandoned` reports it.
        """

        if self._closing or not self._thread.isRunning():
            return
        self._closing = True
        self._tasks.closing.set()
        self._stop_batches_for_close()
        stopped = threading.Event()
        self._final_task.emit(lambda: self._run_final_disconnect(stopped))
        wait_for = wait or (lambda event, seconds: event.wait(seconds))
        if not wait_for(stopped, SHUTDOWN_GRACE_SECONDS):
            self._connection_closed = True
            self._runner.abort_connection()
            if not wait_for(stopped, SHUTDOWN_ABORT_SECONDS):
                self._report_cut_off_submissions()
                self._abandon_worker()
                return
        if not self._thread.wait(int(SHUTDOWN_ABORT_SECONDS * 1000)):
            self._report_cut_off_submissions()
            self._abandon_worker()
            return
        # Apply what the worker delivered before it stopped, which a wait that
        # ended as soon as the disconnect ran may have left queued.
        QCoreApplication.sendPostedEvents()
        self._report_cut_off_submissions()

    def _report_cut_off_submissions(self) -> None:
        """Emit :attr:`submission_cut_off` for each job in hand whose submit never answered."""

        for job_id in self._in_hand_at_close:
            if job_id not in self._answered_while_closing:
                self._results_while_closing = True
                self.submission_cut_off.emit(job_id)

    def _stop_batches_for_close(self) -> None:
        """Stop every batch before its next job, and report the jobs none will reach."""

        for batch_id, batch in list(self._batches.items()):
            with batch.lock:
                batch.cancelled.set()
                never_started = [job_id for job_id in batch.job_ids if job_id not in batch.started]
                self._in_hand_at_close.extend(
                    job_id
                    for job_id in batch.job_ids
                    if job_id in batch.started and job_id in batch.submitting
                )
            for job_id in never_started:
                if job_id in batch.submitting:
                    self.job_submit_finished.emit(batch_id, job_id, None, CLOSED_BEFORE_SUBMITTING)

    def _run_final_disconnect(self, stopped: threading.Event) -> None:
        # Runs on the worker, after whatever it held when closing began.
        try:
            self._runner.shutdown()
        except Exception:  # noqa: BLE001 - best-effort during teardown
            pass
        finally:
            stopped.set()
            self._thread.quit()

    def _abandon_worker(self) -> None:
        self._abandoned = True
        self.blockSignals(True)
        self._runner.blockSignals(True)
        self._tasks.blockSignals(True)
        _abandoned_workers.append((self._thread, self._runner, self._tasks))

    # ------------------------------------------------------------------
    # Internal invocation machinery
    # ------------------------------------------------------------------
    def _run_async_op(self, kind: str, fn: Callable[[], Any]) -> object:
        """Queue ``fn`` on the worker and route its outcome via a signal.

        Mints a fresh request id (GUI-thread only, so no lock needed), queues the
        op onto the worker, and returns immediately. The worker runs ``fn`` and
        emits :attr:`operation_finished` ``(request_id, kind, result, error)`` on
        every exit path — success, caught exception, or an escaping
        ``BaseException`` — so a continuation registered for this request id always
        fires and any pending-state UI is always restored.

        During the startup/teardown window (worker not running) the call is run
        directly and the completion is emitted synchronously, preserving the
        contract for callers that only key off :attr:`operation_finished`.
        """

        request_id = next(self._request_ids)
        if not self._thread.isRunning():
            result: Any = None
            error: BaseException | None = None
            try:
                result = fn()
            except BaseException as exc:  # noqa: BLE001 - delivered to the GUI as data
                error = exc
            self.operation_finished.emit(request_id, kind, result, error)
            return request_id
        self._async_op.emit(request_id, kind, fn)
        return request_id
