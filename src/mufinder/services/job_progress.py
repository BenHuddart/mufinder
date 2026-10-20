"""Geometry-optimisation progress of running jobs, read from their growing output.

A geometry optimisation can run for hours or days, and CASTEP appends a
convergence table to its ``.castep`` after every step. :class:`GeometryProgressReader`
follows that file while the job runs: it remembers how far it has read and keeps
only the lines the progress parser needs
(:func:`~mufinder.castep_io.geometry_progress.is_geometry_progress_line`), so
each refresh fetches only what CASTEP appended since the last one. It reads a
local file from disk and a cluster file over SFTP from the same cursor logic.

:class:`JobProgressMonitor` decides when to read. It follows the running
geometry-optimisation jobs it is given, refreshes them at a slow fixed rate
while refreshing is enabled (the dashboard is visible), and refreshes chosen
jobs on demand, rate-limited per job. Local files are read on a background
thread; cluster files are read by the cluster runner on its SSH worker thread,
which the monitor asks through a callable and which reports back through
:meth:`JobProgressMonitor.accept_remote_progress`. The monitor never touches
the UI; it emits :attr:`JobProgressMonitor.progress_changed`.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import posixpath
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from PySide6.QtCore import QObject, QTimer, Signal

from mufinder.castep_io.geometry_progress import (
    GeometryProgress,
    is_geometry_progress_line,
    parse_geometry_progress_lines,
)
from mufinder.castep_io.gui_metadata import WORKFLOW_TASK_MAP
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import Job
from mufinder.services.job_monitoring import is_cluster_job

if TYPE_CHECKING:
    from mufinder.services.ssh_client import SSHConnection

#: How often the monitor refreshes every followed job while refreshing is enabled.
PROGRESS_REFRESH_INTERVAL_MS = 60_000
#: Shortest gap between two on-demand reads of the same job's output.
MIN_ON_DEMAND_REFRESH_SECONDS = 10.0


def job_reports_geometry_progress(job: Job) -> bool:
    """Return whether ``job`` runs a CASTEP geometry optimisation."""

    return WORKFLOW_TASK_MAP.get(job.workflow_stage.value) == "GEOMETRYOPTIMIZATION"


def castep_output_names(seed_name: str, farm_count: int) -> tuple[str, ...]:
    """Return the ``.castep`` names a job may write, most specific first.

    Task farming writes ``<seed>_farm001.castep`` alongside the master
    ``<seed>.castep``; farm 001 holds the calculation's own output.
    """

    names = (f"{seed_name}.castep",)
    if farm_count > 1:
        names = (f"{seed_name}_farm001.castep", *names)
    return names


@dataclass(slots=True)
class _Cursor:
    offset: int = 0
    pending: bytes = b""
    lines: list[str] = field(default_factory=list)
    progress: GeometryProgress | None = None


class GeometryProgressReader:
    """Follow growing ``.castep`` files and parse their geometry progress.

    Each key (normally a job id) has its own cursor. A file that shrinks was
    replaced, so its cursor starts again from the beginning. A trailing partial
    line is held back until CASTEP finishes writing it. Safe to call from
    several threads.
    """

    def __init__(self) -> None:
        self._cursors: dict[Hashable, _Cursor] = {}
        self._lock = threading.Lock()

    def read(
        self,
        key: Hashable,
        *,
        size: int | None,
        read_from: Callable[[int], bytes],
    ) -> GeometryProgress | None:
        """Read what was appended since the last read and return the progress.

        ``size`` is the file's current size, ``None`` when it does not exist;
        ``read_from(offset)`` returns the file's bytes from ``offset`` on.
        """

        with self._lock:
            if size is None:
                self._cursors.pop(key, None)
                return None
            cursor = self._cursors.get(key)
            if cursor is None or size < cursor.offset:
                cursor = _Cursor()
                self._cursors[key] = cursor
            if size > cursor.offset:
                data = read_from(cursor.offset)
                cursor.offset += len(data)
                complete, newline, rest = (cursor.pending + data).rpartition(b"\n")
                cursor.pending = rest
                if newline:
                    text = complete.decode("utf-8", errors="replace")
                    cursor.lines.extend(line for line in text.splitlines() if is_geometry_progress_line(line))
                    cursor.progress = parse_geometry_progress_lines(cursor.lines)
            return cursor.progress

    def read_local(self, key: Hashable, candidates: Sequence[Path]) -> GeometryProgress | None:
        """Read the first of ``candidates`` that exists on this machine."""

        for path in candidates:
            try:
                size = path.stat().st_size
            except OSError:
                continue

            def read_from(offset: int, path: Path = path) -> bytes:
                with path.open("rb") as handle:
                    handle.seek(offset)
                    return handle.read()

            return self.read(key, size=size, read_from=read_from)
        return self.read(key, size=None, read_from=lambda _offset: b"")

    def read_remote(self, key: Hashable, ssh: SSHConnection, candidates: Sequence[str]) -> GeometryProgress | None:
        """Read the first of ``candidates`` that exists on the cluster, over SFTP."""

        for remote_path in candidates:
            attributes = ssh.stat(remote_path)
            size = getattr(attributes, "st_size", None) if attributes is not None else None
            if size is None:
                continue
            return self.read(
                key,
                size=int(size),
                read_from=lambda offset, remote_path=remote_path: ssh.read_bytes(remote_path, offset),
            )
        return self.read(key, size=None, read_from=lambda _offset: b"")

    def forget(self, key: Hashable) -> None:
        with self._lock:
            self._cursors.pop(key, None)

    def forget_all_except(self, keys: Iterable[Hashable]) -> None:
        keep = set(keys)
        with self._lock:
            for key in [key for key in self._cursors if key not in keep]:
                del self._cursors[key]


def remote_castep_candidates(remote_dir: str, seed_name: str, farm_count: int) -> tuple[str, ...]:
    """Return the remote ``.castep`` paths a job may write; see :func:`castep_output_names`."""

    return tuple(posixpath.join(remote_dir, name) for name in castep_output_names(seed_name, farm_count))


def _start_daemon_thread(task: Callable[[], None]) -> None:
    threading.Thread(target=task, name="mufinder-geometry-progress", daemon=True).start()


class JobProgressMonitor(QObject):
    """Keep the geometry progress of running jobs current, at a slow rate."""

    #: ``(job_id, GeometryProgress | None)`` whenever a job's progress changes.
    progress_changed = Signal(object, object)
    _local_progress_read = Signal(object)

    def __init__(
        self,
        *,
        local_output_paths: Callable[[UUID], Sequence[Path]],
        request_remote_progress: Callable[[list[UUID]], None],
        run_in_background: Callable[[Callable[[], None]], None] = _start_daemon_thread,
        clock: Callable[[], float] = time.monotonic,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._local_output_paths = local_output_paths
        self._request_remote_progress = request_remote_progress
        self._run_in_background = run_in_background
        self._clock = clock
        self._reader = GeometryProgressReader()
        self._jobs: dict[UUID, bool] = {}  # job id -> runs on a cluster
        self._progress: dict[UUID, GeometryProgress | None] = {}
        self._last_requested: dict[UUID, float] = {}
        self._local_in_flight: set[UUID] = set()
        self._remote_in_flight: set[UUID] = set()
        self._local_progress_read.connect(self._on_local_progress_read)
        self._timer = QTimer(self)
        self._timer.setInterval(PROGRESS_REFRESH_INTERVAL_MS)
        self._timer.timeout.connect(lambda: self.refresh(force=True))
        self._enabled = False

    def progress_for(self, job_id: UUID) -> GeometryProgress | None:
        return self._progress.get(job_id)

    def followed_job_ids(self) -> frozenset[UUID]:
        return frozenset(self._jobs)

    def follow(self, jobs: Iterable[Job]) -> None:
        """Follow the running geometry optimisations among ``jobs`` and drop the rest.

        A job seen for the first time is read at once, so its progress appears
        without waiting for the next periodic refresh.
        """

        followed = {
            job.id: is_cluster_job(job)
            for job in jobs
            if job.status == JobStatus.RUNNING and job_reports_geometry_progress(job)
        }
        new_ids = [job_id for job_id in followed if job_id not in self._jobs]
        for job_id in [job_id for job_id in self._jobs if job_id not in followed]:
            self._last_requested.pop(job_id, None)
            if self._progress.pop(job_id, None) is not None:
                self.progress_changed.emit(job_id, None)
        self._jobs = followed
        self._reader.forget_all_except(followed)
        self._sync_timer()
        if new_ids:
            self.refresh(new_ids)

    def set_enabled(self, enabled: bool) -> None:
        """Turn the periodic refresh on or off (on while the dashboard is visible)."""

        self._enabled = bool(enabled)
        self._sync_timer()

    def refresh(self, job_ids: Iterable[UUID] | None = None, *, force: bool = False) -> None:
        """Read the output of ``job_ids`` (default: every followed job).

        Without ``force`` a job read less than :data:`MIN_ON_DEMAND_REFRESH_SECONDS`
        ago is skipped. A job whose previous read has not come back is skipped.
        """

        now = self._clock()
        local_ids: list[UUID] = []
        remote_ids: list[UUID] = []
        for job_id in self._jobs if job_ids is None else job_ids:
            if job_id not in self._jobs:
                continue
            last = self._last_requested.get(job_id)
            if not force and last is not None and now - last < MIN_ON_DEMAND_REFRESH_SECONDS:
                continue
            on_cluster = self._jobs[job_id]
            if job_id in (self._remote_in_flight if on_cluster else self._local_in_flight):
                continue
            self._last_requested[job_id] = now
            (remote_ids if on_cluster else local_ids).append(job_id)
        if local_ids:
            self._read_local(local_ids)
        if remote_ids:
            self._remote_in_flight.update(remote_ids)
            self._request_remote_progress(remote_ids)

    def accept_remote_progress(self, results: Mapping[UUID, GeometryProgress | None]) -> None:
        """Take the progress the cluster runner read for some jobs."""

        self._remote_in_flight.difference_update(results)
        self._store(results)

    def _read_local(self, job_ids: list[UUID]) -> None:
        targets = {job_id: tuple(self._local_output_paths(job_id)) for job_id in job_ids}
        self._local_in_flight.update(targets)
        reader = self._reader

        def task() -> None:
            results: dict[UUID, GeometryProgress | None] = {}
            for job_id, candidates in targets.items():
                try:
                    results[job_id] = reader.read_local(job_id, candidates)
                except OSError:
                    results[job_id] = None
            try:
                self._local_progress_read.emit(results)
            except RuntimeError:
                # The monitor was deleted (the window closed) while this read ran.
                pass

        self._run_in_background(task)

    def _on_local_progress_read(self, results: Mapping[UUID, GeometryProgress | None]) -> None:
        self._local_in_flight.difference_update(results)
        self._store(results)

    def _store(self, results: Mapping[UUID, GeometryProgress | None]) -> None:
        for job_id, progress in results.items():
            if job_id not in self._jobs:
                continue
            if progress is None and job_id in self._progress:
                # A failed read keeps the last progress shown rather than blanking it.
                continue
            if self._progress.get(job_id) != progress or job_id not in self._progress:
                self._progress[job_id] = progress
                self.progress_changed.emit(job_id, progress)

    def _sync_timer(self) -> None:
        if self._enabled and self._jobs:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()
