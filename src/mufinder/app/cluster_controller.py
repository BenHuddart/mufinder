"""Threaded facade around :class:`ClusterJobRunner`.

All paramiko SSH/SFTP work is blocking. Running it on the Qt GUI thread froze
the window during cluster submission and, worst of all, whenever the 60-second
queue poll discovered finished jobs and downloaded their output inline.

``ClusterController`` moves the runner onto a dedicated worker ``QThread`` so
that *all* SSH activity — connection, polling, result retrieval, submission,
cancellation — executes off the GUI thread. The controller exposes the same
public surface as :class:`ClusterJobRunner` (identical signal names, the same
methods and connection-state properties) so it is a drop-in replacement for
the ``main_window`` call sites.

Two interaction styles are used to marshal work onto the worker:

* **Asynchronous** (``refresh_queue``, ``cancel``, ``track_existing_job``):
  fire-and-forget. The GUI returns immediately; results arrive later via the
  forwarded signals. This is what keeps the UI responsive.
* **Blocking bridge** (``connect_to_cluster``, ``submit``,
  ``remote_directory_exists``, ``remove_remote_directory``): the operation runs
  on the worker thread but the calling GUI thread waits for its result (or
  exception). These callers still expect synchronous semantics today; the work
  is at least confined to the worker thread so the single paramiko connection is
  only ever touched from one thread. Converting these to a fully asynchronous
  progress/cancel flow is the follow-up phase.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from PySide6.QtCore import (
    QMetaObject,
    QObject,
    Qt,
    QThread,
    Signal,
    Slot,
)

from mufinder.cluster_setup import ClusterConfig
from mufinder.domain.enums import ACTIVE_JOB_STATUSES, JobStatus
from mufinder.services.cluster_runner import ClusterJobRunner
from mufinder.services.ssh_client import SSHCredentials

# Status values that mean a tracked cluster job is still active (mirror of the
# domain ACTIVE_JOB_STATUSES as their serialised strings, which is what the
# runner emits via ``status_changed``).
_ACTIVE_STATUS_VALUES = frozenset(status.value for status in ACTIVE_JOB_STATUSES)


class _SshTaskRunner(QObject):
    """Executes callables on the thread it lives on (the SSH worker thread).

    Kept separate from :class:`ClusterJobRunner` so the service layer stays free
    of threading plumbing. Both objects are moved to the same worker thread, so
    a callable invoked here that calls a runner method runs that method on the
    worker thread too.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pending_task: Callable[[], Any] | None = None
        self._pending_result: Any = None
        self._pending_error: BaseException | None = None

    @Slot(object)
    def run_task(self, task: Callable[[], Any]) -> None:
        """Run a fire-and-forget callable on the worker thread."""

        task()

    @Slot()
    def run_pending(self) -> None:
        """Run the stashed callable, capturing its result or exception.

        Paired with a blocking cross-thread invocation so the calling thread can
        recover the return value or re-raise the failure.
        """

        task = self._pending_task
        self._pending_result = None
        self._pending_error = None
        if task is None:
            return
        try:
            self._pending_result = task()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            self._pending_error = exc


class ClusterController(QObject):
    """GUI-thread facade that runs cluster SSH work on a worker thread."""

    # Re-exposed runner signals (emitted on the GUI thread).
    job_started = Signal(object)
    job_finished = Signal(object, object)
    job_failed = Signal(object, str)
    job_cancelled = Signal(object)
    status_changed = Signal(object, object)
    connection_status_changed = Signal(bool)
    queue_checked = Signal(object)
    poll_failed = Signal(str)

    # Result of an asynchronous connect: None on success, else the exception.
    connect_finished = Signal(object)

    # Asynchronous batch-submission progress (all emitted on the GUI thread).
    remote_dirs_checked = Signal(object)  # {"existing": [str], "errors": {job_id: str}}
    submit_batch_progress = Signal(int, int, str)  # index, total, job label
    job_submit_finished = Signal(object, object, str)  # job_id, remote_job_id|None, error
    submit_batch_finished = Signal()

    # Internal: marshals a fire-and-forget callable onto the worker thread.
    _async_task = Signal(object)

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
        self._batch_cancelled = False

        # GUI-thread mirror of the runner's active tracked-job ids, maintained
        # from the signals/calls this controller already sees. The runner's own
        # ``_tracked_jobs`` live on the worker thread and must not be read from
        # the GUI thread; this set lets the job watchdog ask "is this cluster job
        # still being tracked?" without crossing threads. It is deliberately NOT
        # cleared on disconnect: the runner keeps its tracked jobs across a
        # disconnect (only ``clear_tracked_jobs`` drops them, on project switch),
        # so clearing here would make the watchdog stop live jobs after a
        # reconnect. The watchdog never flags cluster jobs while disconnected, so
        # retaining the mirror across a disconnect is safe.
        self._tracked_active_job_ids: set[UUID] = set()

        # Forward worker signals to the GUI thread. These cross-thread
        # connections are queued automatically.
        self._runner.job_started.connect(self.job_started)
        self._runner.job_finished.connect(self.job_finished)
        self._runner.job_failed.connect(self.job_failed)
        self._runner.job_cancelled.connect(self.job_cancelled)
        self._runner.status_changed.connect(self.status_changed)
        # Maintain the GUI-thread tracked-id mirror from the same status stream.
        # Queued automatically (cross-thread) and runs on the GUI thread.
        self._runner.status_changed.connect(self._mirror_status_changed)
        self._runner.queue_checked.connect(self.queue_checked)
        self._runner.poll_failed.connect(self.poll_failed)
        self._runner.connection_status_changed.connect(self._on_connection_status_changed)

        self._async_task.connect(self._tasks.run_task, Qt.ConnectionType.QueuedConnection)

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
    # Tracked-job mirror (GUI-thread safe; never touches the runner's dicts)
    # ------------------------------------------------------------------
    def tracked_active_job_ids(self) -> frozenset[UUID]:
        """Return the ids of cluster jobs the runner is actively tracking.

        A GUI-thread mirror of the runner's non-terminal ``_tracked_jobs``. Used
        by the job watchdog to detect active cluster jobs the runner has silently
        stopped tracking. Conservative by construction: a job is only dropped
        when a terminal ``status_changed`` is observed for it, so an in-flight
        job is never briefly absent.
        """

        return frozenset(self._tracked_active_job_ids)

    @Slot(object, object)
    def _mirror_status_changed(self, job_id: Any, status_value: Any) -> None:
        try:
            normalized = job_id if isinstance(job_id, UUID) else UUID(str(job_id))
        except (TypeError, ValueError, AttributeError):
            return
        value = status_value.value if isinstance(status_value, JobStatus) else str(status_value)
        if value in _ACTIVE_STATUS_VALUES:
            self._tracked_active_job_ids.add(normalized)
        else:
            self._tracked_active_job_ids.discard(normalized)

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
    # Blocking bridge (runs on the worker thread; caller waits for the result)
    # ------------------------------------------------------------------
    def connect_to_cluster(
        self,
        config: ClusterConfig,
        password: str | None = None,
        *,
        credentials: SSHCredentials | None = None,
    ) -> None:
        self._pending_connect = (config.name, config.username.strip() or None)
        self._call_blocking(
            lambda: self._runner.connect_to_cluster(
                config,
                password=password,
                credentials=credentials,
            )
        )
        # The connect succeeded if no exception propagated. Adopt the target
        # identity eagerly so the GUI can read it immediately, even before the
        # queued connection_status_changed signal is delivered.
        self._connected = True
        self._connected_cluster_name = config.name
        self._connected_username = config.username.strip() or None

    def submit(self, **kwargs: Any) -> str:
        return self._call_blocking(lambda: self._runner.submit(**kwargs))

    def remote_directory_exists(self, remote_directory: str) -> bool:
        return self._call_blocking(
            lambda: self._runner.remote_directory_exists(remote_directory)
        )

    def remove_remote_directory(self, remote_directory: str) -> None:
        self._call_blocking(
            lambda: self._runner.remove_remote_directory(remote_directory)
        )

    def clear_tracked_jobs(self) -> None:
        # Fast, but must run on the worker thread because it mutates state the
        # poll loop also touches. Bridged (not async) so switching projects has
        # synchronous semantics for the caller. Clear the GUI mirror to match:
        # this is the only place the runner drops all tracked jobs.
        self._tracked_active_job_ids.clear()
        self._call_blocking(self._runner.clear_tracked_jobs)

    # ------------------------------------------------------------------
    # Asynchronous (fire-and-forget) operations
    # ------------------------------------------------------------------
    def refresh_queue(self) -> None:
        self._async_task.emit(self._runner.refresh_queue)

    def cancel(self, job_id: Any) -> None:
        self._async_task.emit(lambda: self._runner.cancel(job_id))

    def track_existing_job(self, **kwargs: Any) -> None:
        # The runner emits no signal when re-tracking a persisted job, so mirror
        # the addition here on the GUI thread. Only active jobs are tracked; a
        # terminal status means the runner will mark it terminal and never poll,
        # so it must not enter the active mirror.
        status = kwargs.get("status")
        job_id = kwargs.get("job_id")
        if status in ACTIVE_JOB_STATUSES and isinstance(job_id, UUID):
            self._tracked_active_job_ids.add(job_id)
        self._async_task.emit(lambda: self._runner.track_existing_job(**kwargs))

    def retrieve_outputs_for(self, job_id: Any, **kwargs: Any) -> None:
        """Re-attempt output retrieval for one job on the worker thread.

        Fire-and-forget, mirroring :meth:`refresh_queue`: the blocking SSH
        download runs off the GUI thread and results arrive via the forwarded
        ``job_finished`` / ``job_failed`` / ``status_changed`` signals.
        """

        self._async_task.emit(lambda: self._runner.retrieve_outputs_for(job_id, **kwargs))

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

    def submit_batch(self, items: list[dict[str, Any]]) -> None:
        """Submit a batch of prepared jobs on the worker thread.

        Each item is ``{"job_id", "label", "submit_kwargs", "remote_dir_to_delete"}``.
        Emits :attr:`submit_batch_progress` before each job, :attr:`job_submit_finished`
        with the result of each, and :attr:`submit_batch_finished` once (even when
        cancelled or on error).
        """

        self._batch_cancelled = False
        self._async_task.emit(lambda: self._run_submit_batch(items))

    def cancel_batch(self) -> None:
        """Request cancellation; checked between jobs by the worker."""

        self._batch_cancelled = True

    def _run_submit_batch(self, items: list[dict[str, Any]]) -> None:
        total = len(items)
        try:
            for index, item in enumerate(items, start=1):
                if self._batch_cancelled:
                    break
                job_id = item["job_id"]
                self.submit_batch_progress.emit(index, total, item.get("label", ""))
                try:
                    delete_dir = item.get("remote_dir_to_delete")
                    if delete_dir:
                        self._runner.remove_remote_directory(delete_dir)
                    remote_job_id = self._runner.submit(**item["submit_kwargs"])
                except Exception as exc:  # noqa: BLE001 - reported per job to the GUI
                    self.job_submit_finished.emit(job_id, None, str(exc))
                    continue
                self.job_submit_finished.emit(job_id, remote_job_id, "")
        finally:
            self.submit_batch_finished.emit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Disconnect and stop the worker thread. Safe to call more than once.

        The disconnect runs synchronously *on the worker thread* (blocking bridge)
        so the poll ``QTimer`` — which lives on that thread — is stopped before the
        event loop quits. Doing it asynchronously and then calling ``quit()`` races:
        the loop can exit before the disconnect runs, leaving the timer active and
        later killed from the GUI thread ("Timers cannot be stopped from another
        thread"). The cost is that closing while a large download is in flight
        waits for it, which matches how the app's other worker threads shut down.
        """

        if not self._thread.isRunning():
            return
        try:
            self._call_blocking(self._runner.shutdown)
        except Exception:  # noqa: BLE001 - best-effort during teardown
            pass
        self._thread.quit()
        self._thread.wait(5000)

    # ------------------------------------------------------------------
    # Internal blocking-invocation machinery
    # ------------------------------------------------------------------
    def _call_blocking(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on the worker thread and return its result (or re-raise).

        Only ever called from the GUI thread, which is single-threaded, so the
        stash on ``_tasks`` cannot be clobbered by a concurrent caller. The
        BlockingQueuedConnection provides the memory barrier that makes the
        stashed callable and its result visible across threads.
        """

        if not self._thread.isRunning():
            # Fall back to a direct call during startup/teardown windows.
            return fn()
        self._tasks._pending_task = fn
        QMetaObject.invokeMethod(
            self._tasks,
            "run_pending",
            Qt.ConnectionType.BlockingQueuedConnection,
        )
        error = self._tasks._pending_error
        result = self._tasks._pending_result
        self._tasks._pending_task = None
        self._tasks._pending_result = None
        self._tasks._pending_error = None
        if error is not None:
            raise error
        return result
