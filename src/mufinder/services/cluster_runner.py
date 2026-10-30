"""Remote cluster execution service using SSH and scheduler polling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
import posixpath
import tempfile
from typing import Any
from uuid import UUID

from PySide6.QtCore import QObject, QTimer, Signal

from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import utc_now
from mufinder.services.cluster_retrieval import (
    RetrievalOutcome,
    queue_observation,
    retrieve_error_files,
    retrieve_job_outputs,
)
from mufinder.services.cluster_security import invalid_remote_job_labels
from mufinder.services.failure_diagnosis import (
    MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE,
    RETRIEVAL_FAILURE_PREFIX,
    RETRY_RETRIEVAL_NOT_CONNECTED_MESSAGE,
)
from mufinder.services.job_artifacts import RetrievalFlags
from mufinder.services.job_progress import GeometryProgressReader, remote_castep_candidates
from mufinder.services.scheduler import (
    Scheduler,
    SchedulerJobInfo,
    SchedulerSubmitRequest,
    create_scheduler,
)
from mufinder.services.script_template import render_job_script
from mufinder.services.ssh_client import SSHConnection, SSHCredentials

MISSING_ENTRY_RETRIEVAL_ATTEMPTS = 3
"""Polls a scheduler-missing job may fail output retrieval before it is failed.

Only a retry-eligible failure counts against this limit; see
:class:`~mufinder.services.cluster_retrieval.RetrievalFailure`.
"""


_ACTIVE_TRACKED_STATUSES = frozenset(
    {JobStatus.SUBMITTING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING}
)


def _same_credential_target(previous: ClusterConfig, current: ClusterConfig) -> bool:
    """Return whether cached credentials for ``previous`` may be offered to ``current``.

    Credentials are reused only when the profile still names the same account on
    the same target and gateway endpoints, so editing a profile's host or port
    while keeping its name never sends a cached secret to the new host. Turning
    the jump host on or off also invalidates the cache, because a direct
    connection caches its target password as the jump secret too.
    """

    def _host(value: str) -> str:
        return value.strip().casefold()

    return (
        _host(previous.name) == _host(current.name)
        and _host(previous.hostname) == _host(current.hostname)
        and previous.port == current.port
        and previous.username.strip() == current.username.strip()
        and previous.jump_host_enabled == current.jump_host_enabled
        and _host(previous.jump_hostname) == _host(current.jump_hostname)
        and previous.jump_port == current.jump_port
        and previous.jump_username.strip() == current.jump_username.strip()
    )


@dataclass(slots=True)
class _TrackedClusterJob:
    job_id: UUID
    remote_job_id: str
    remote_dir: str
    seed_name: str
    local_output_dir: Path
    cluster_name: str = ""
    status: JobStatus = JobStatus.QUEUED
    terminal: bool = False
    retrieval: RetrievalFlags = RetrievalFlags()
    #: Consecutive polls where this job was missing from the scheduler queue
    #: and its output could not be downloaded. Reset when the job reappears.
    missing_retrieval_attempts: int = 0
    #: The job was failed because neither the queue nor accounting listed it.
    #: It stays terminal but is still polled, so ``job_reappeared`` can report
    #: it if the scheduler lists it again.
    watch_for_recovery: bool = False


@dataclass(frozen=True, slots=True)
class QueueCheckSnapshot:
    """Summary emitted after one queue polling pass."""

    checked_at: datetime
    elapsed_by_job_id: dict[UUID, str]
    #: Short human-readable statement of what the scheduler reported for each
    #: polled job this cycle (e.g. "scheduler reports RUNNING"). Consumed by
    #: ``JobExecutionController.mark_queue_checked`` to stamp per-job monitor
    #: metadata on the document.
    observations_by_job_id: dict[UUID, str] = field(default_factory=dict)
    #: Jobs this poll checked only as reappearance watches (see
    #: ``ClusterJobRunner.job_reappeared``), so the GUI can end a watch whose
    #: job no longer awaits reappearance.
    watched_job_ids: frozenset[UUID] = frozenset()


class CheckpointProbeStatus(StrEnum):
    """Outcome vocabulary for a ``{seed}.check`` checkpoint existence probe.

    Feeds the informational "checkpoint found / not found / cannot verify"
    line in the Continue-relaxation dialog. The same three-way result covers
    both local jobs (a plain :class:`pathlib.Path` check, which can only ever
    report ``FOUND``/``ABSENT``) and cluster jobs (an SSH stat that can also
    fail to connect, time out, or hit a transport error, reported as
    ``UNAVAILABLE`` rather than blocking the dialog).
    """

    FOUND = "found"
    ABSENT = "absent"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CheckpointProbeResult:
    """Result of probing whether a job's checkpoint file exists.

    Construct via :meth:`found`, :meth:`absent`, or :meth:`unavailable` rather
    than the constructor directly, so ``reason`` is only ever populated
    alongside :attr:`CheckpointProbeStatus.UNAVAILABLE`. The dialog that
    displays this result performs no I/O itself; the probe runs in the
    controller (or, for local jobs, a plain path check) before the dialog is
    constructed.
    """

    status: CheckpointProbeStatus
    reason: str = ""

    @classmethod
    def found(cls) -> "CheckpointProbeResult":
        """Return a result reporting the checkpoint file exists."""

        return cls(CheckpointProbeStatus.FOUND)

    @classmethod
    def absent(cls) -> "CheckpointProbeResult":
        """Return a result reporting the checkpoint file does not exist."""

        return cls(CheckpointProbeStatus.ABSENT)

    @classmethod
    def unavailable(cls, reason: str) -> "CheckpointProbeResult":
        """Return a result reporting the probe could not be completed."""

        return cls(CheckpointProbeStatus.UNAVAILABLE, reason)


class ClusterJobRunner(QObject):
    """Run CASTEP jobs on a remote scheduler over SSH."""

    job_started = Signal(object)
    #: ``(job_id, CastepResult, ParsedJobArtifacts)``. The third payload carries
    #: the secondary outputs (relaxed cell, phonon, transition state) parsed on
    #: this worker thread so the GUI handler never re-parses them.
    job_finished = Signal(object, object, object)
    job_failed = Signal(object, str)
    job_cancelled = Signal(object)
    status_changed = Signal(object, object)
    #: ``(job_id, status_value)``: a job failed for a missing scheduler entry is
    #: listed again as queued, running or completed. The runner has stopped
    #: watching it; the GUI recovers the job and re-tracks it.
    job_reappeared = Signal(object, object)

    connection_status_changed = Signal(bool)
    queue_checked = Signal(object)
    poll_failed = Signal(str)
    #: ``{job_id: GeometryProgress | None}`` from :meth:`read_geometry_progress`.
    geometry_progress_read = Signal(object)

    def __init__(self, parent: QObject | None = None, *, settings=None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._config: ClusterConfig | None = None
        self._ssh: SSHConnection | None = None
        self._scheduler: Scheduler | None = None
        self._tracked_jobs: dict[UUID, _TrackedClusterJob] = {}
        self._credential_cache: SSHCredentials | None = None
        self._progress_reader = GeometryProgressReader()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(60_000)
        self._poll_timer.timeout.connect(self._poll_queue)

    @property
    def is_connected(self) -> bool:
        return self._ssh is not None and self._ssh.is_connected

    @property
    def connected_cluster_name(self) -> str | None:
        if not self.is_connected or self._config is None:
            return None
        return self._config.name

    @property
    def connected_username(self) -> str | None:
        if not self.is_connected or self._config is None:
            return None
        username = self._config.username.strip()
        return username or None

    def connect_to_cluster(
        self,
        config: ClusterConfig,
        password: str | None = None,
        *,
        credentials: SSHCredentials | None = None,
    ) -> None:
        """Connect to cluster and scheduler backend."""

        if self.is_connected and self._config == config:
            return

        resolved_credentials = credentials or SSHCredentials.from_legacy_password(password)
        if (
            resolved_credentials.target_secret is None
            and resolved_credentials.jump_secret is None
            and self._config is not None
            and _same_credential_target(self._config, config)
            and self._credential_cache is not None
        ):
            resolved_credentials = self._credential_cache

        self.disconnect()
        ssh = SSHConnection(config, settings=self._settings)
        if config.jump_host_enabled:
            ssh.connect(credentials=resolved_credentials)
        else:
            ssh.connect(password=resolved_credentials.target_secret)

        self._config = config
        self._ssh = ssh
        self._scheduler = create_scheduler(
            config.queue_system,
            submit_command_path=config.submit_command_path,
            queue_check_command_path=config.queue_check_command_path,
            cancel_command_path=config.cancel_command_path,
        )
        self._credential_cache = resolved_credentials
        self.connection_status_changed.emit(True)
        if self._has_active_jobs_for_connected_cluster() and not self._poll_timer.isActive():
            self._poll_timer.start()

    def disconnect(self) -> None:
        """Disconnect from cluster and stop polling."""

        self._poll_timer.stop()
        if self._ssh is not None:
            self._ssh.disconnect()
        self._ssh = None
        self._scheduler = None
        self._config = None
        self._credential_cache = None
        self.connection_status_changed.emit(False)

    def submit(
        self,
        *,
        job_id: UUID,
        local_job_dir: Path,
        seed_name: str,
        num_processes: int,
        walltime: str,
        working_directory: str,
        template_script_path: str,
        cores_per_node: int,
        memory_gb: float | None = None,
        mem_per_cpu_gb: float | None = None,
        queue_name: str = "",
        comment: str = "",
        castep_executable: str = "",
        remote_directory: str | None = None,
        reuse_existing_files: bool = False,
        allow_existing_remote_directory: bool = False,
        **retrieval: Any,
    ) -> str:
        """Upload files, submit scheduler script, and start tracking the remote job.

        ``retrieval`` holds the :class:`~mufinder.services.job_artifacts.RetrievalFlags`
        fields as keywords (``retrieve_den_fmt``, ``task_farming_enabled``, ...),
        as in :meth:`track_existing_job` and :meth:`retrieve_outputs_for`.

        ``reuse_existing_files`` skips only the ``.cell``/``.param`` upload, so a
        resubmission into an existing remote directory preserves the remote
        inputs, the outputs already produced there, and any ``.check``
        continuation state. It does **not** skip the submit script: ``run.sh`` is
        regenerated from the current template and uploaded on every submission,
        because ``sbatch`` is invoked with no command-line resource flags and the
        script is therefore the sole carrier of walltime, task count, memory
        and partition. Reusing a stale script would silently ignore the resources the current
        submission asked for.

        ``allow_existing_remote_directory`` is deliberately unread here. It is a
        caller-side marker: the main window uses it to skip the remote-directory
        existence check and the overwrite prompt for a directory it means to
        submit into. This method already creates the directory with
        ``exist_ok=True``, so it needs no separate opt-in.
        """

        ssh = self._require_ssh()
        scheduler = self._require_scheduler()

        local_dir = Path(local_job_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            raise ConnectionError(f"Local job directory does not exist: {local_dir}")

        remote_root = working_directory.strip()
        normalized_remote_directory = (remote_directory or "").strip()
        if not remote_root and not normalized_remote_directory:
            raise ConnectionError("Remote working directory is required.")
        self._validate_seed_name(seed_name)

        remote_dir = normalized_remote_directory or posixpath.join(remote_root, seed_name)
        ssh.mkdir(remote_dir, exist_ok=True)

        if not reuse_existing_files:
            for suffix in (".cell", ".param"):
                source = local_dir / f"{seed_name}{suffix}"
                if not source.exists():
                    raise ConnectionError(f"Missing local input file: {source.name}")
                ssh.put(source, posixpath.join(remote_dir, source.name))

        cluster_name = self._config.name if self._config is not None else ""
        queue_system = self._config.queue_system if self._config is not None else ""
        resolved_castep_executable = castep_executable.strip()
        if not resolved_castep_executable and self._config is not None:
            resolved_castep_executable = self._config.castep_executable.strip()
        if not resolved_castep_executable:
            resolved_castep_executable = "castep.mpi"

        remote_script = ""
        queue = queue_system.strip().lower()
        if queue != "addqueue":
            remote_script = posixpath.join(remote_dir, "run.sh")
            script_content = render_job_script(
                template_script_path=template_script_path,
                cluster_name=cluster_name,
                queue_system=queue_system,
                seed_name=seed_name,
                num_processes=num_processes,
                cores_per_node=cores_per_node,
                walltime=walltime,
                remote_dir=remote_dir,
                castep_executable=resolved_castep_executable,
                mem_per_cpu_gb=mem_per_cpu_gb,
                queue_name=queue_name,
            )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                suffix=".sh",
                delete=False,
            ) as handle:
                handle.write(script_content)
                temp_path = Path(handle.name)
            try:
                ssh.put(temp_path, remote_script)
            finally:
                temp_path.unlink(missing_ok=True)

            ssh.chmod(remote_script, 0o755)
        request = SchedulerSubmitRequest(
            working_dir=remote_dir,
            seed_name=seed_name,
            num_processes=max(1, int(num_processes)),
            cores_per_node=max(1, int(cores_per_node)),
            walltime=walltime,
            castep_executable=resolved_castep_executable,
            script_path=remote_script,
            memory_gb=memory_gb,
            queue_name=queue_name,
            comment=comment,
        )
        remote_job_id = scheduler.submit(ssh, request)

        self._track(
            job_id,
            remote_job_id=remote_job_id,
            remote_dir=remote_dir,
            seed_name=seed_name,
            local_output_dir=local_dir,
            cluster_name=cluster_name,
            status=JobStatus.QUEUED,
            retrieval=retrieval,
        )

        self.status_changed.emit(job_id, JobStatus.QUEUED.value)

        if not self._poll_timer.isActive():
            self._poll_timer.start()

        return remote_job_id

    def cancel(self, job_id: UUID) -> None:
        """Cancel a tracked remote job."""

        ssh = self._require_ssh()
        scheduler = self._require_scheduler()

        tracked = self._tracked_jobs.get(job_id)
        if tracked is None or tracked.terminal:
            return

        scheduler.cancel(ssh, tracked.remote_job_id)
        tracked.terminal = True
        tracked.status = JobStatus.CANCELLED
        self.job_cancelled.emit(job_id)
        self.status_changed.emit(job_id, JobStatus.CANCELLED.value)

    def refresh_queue(self) -> None:
        """Manually trigger one queue poll."""

        self._poll_queue()

    def clear_tracked_jobs(self) -> None:
        """Drop all in-memory tracked jobs.

        This is used when switching projects so stale runtime state does not leak
        across documents.
        """

        self._tracked_jobs.clear()
        self._progress_reader.forget_all_except(())

    def read_geometry_progress(self, job_ids: list[UUID]) -> None:
        """Read tracked jobs' geometry-optimisation progress from their remote output.

        Fetches only what CASTEP appended since the previous read of each job.
        Emits :attr:`geometry_progress_read` with an entry for every requested
        job: ``None`` when the job is not tracked on the connected cluster, the
        cluster is disconnected, or the output cannot be read.
        """

        results: dict[UUID, Any] = {}
        ssh = self._ssh if self.is_connected else None
        for job_id in job_ids:
            entry = self._tracked_jobs.get(job_id)
            progress = None
            if (
                ssh is not None
                and entry is not None
                and self._tracked_job_matches_connected_cluster(entry)
            ):
                candidates = remote_castep_candidates(
                    entry.remote_dir, entry.seed_name, entry.retrieval.farm_count
                )
                try:
                    progress = self._progress_reader.read_remote(job_id, ssh, candidates)
                except Exception:  # noqa: BLE001 - a failed read keeps the last progress shown
                    progress = None
            results[job_id] = progress
        self.geometry_progress_read.emit(results)

    def stop_watching(self, job_ids: list[UUID]) -> None:
        """End the reappearance watch of each of ``job_ids``; other entries are untouched.

        The GUI calls this for a watched job the user has since reset,
        dismissed, or otherwise moved on from, so it is no longer polled.
        """

        for job_id in job_ids:
            tracked = self._tracked_jobs.get(job_id)
            if tracked is not None:
                tracked.watch_for_recovery = False

    def track_existing_job(
        self,
        *,
        job_id: UUID,
        remote_job_id: str,
        remote_dir: str,
        seed_name: str,
        local_output_dir: Path,
        cluster_name: str,
        status: JobStatus,
        watch_for_recovery: bool = False,
        **retrieval: Any,
    ) -> None:
        """Register a persisted cluster job for queue polling and retrieval.

        ``watch_for_recovery`` registers a job failed for a missing scheduler
        entry: it is polled only to emit ``job_reappeared`` if the scheduler
        lists it again. ``retrieval`` holds the
        :class:`~mufinder.services.job_artifacts.RetrievalFlags` fields as
        keywords.
        """

        if not remote_job_id.strip() or not remote_dir.strip():
            return

        tracked = self._track(
            job_id,
            remote_job_id=remote_job_id,
            remote_dir=remote_dir,
            seed_name=seed_name,
            local_output_dir=local_output_dir,
            cluster_name=cluster_name,
            status=status,
            retrieval=retrieval,
        )
        tracked.watch_for_recovery = tracked.terminal and watch_for_recovery
        if self.is_connected and self._is_polled(tracked) and not self._poll_timer.isActive():
            self._poll_timer.start()

    def retrieve_outputs_for(
        self,
        job_id: UUID,
        *,
        remote_job_id: str,
        remote_directory: str,
        seed_name: str,
        local_output_dir: Path,
        cluster_name: str = "",
        **retrieval: Any,
    ) -> None:
        """Re-attempt output retrieval for a scheduler-completed job on demand.

        Backs the dashboard "Retry retrieval" action: it (re-)registers a tracked
        entry from the persisted remote metadata and performs one retrieval pass
        through the same :meth:`_retrieve_results` used by the poll loop, so the
        normal signal flow (``status_changed`` -> retrieving, then
        ``job_finished`` + completed / ``job_failed`` + failed / stopped) reaches
        the GUI unchanged.

        If the cluster is not connected (or does not match this job's cluster) a
        retrieval pass would strand the job in RETRIEVING, so that case, and any
        error raised before retrieval reports a result, is resolved to
        ``failed`` via ``job_failed`` + ``status_changed`` instead.

        ``retrieval`` holds the :class:`~mufinder.services.job_artifacts.RetrievalFlags`
        fields as keywords.
        """

        tracked = self._track(
            job_id,
            remote_job_id=remote_job_id,
            remote_dir=remote_directory,
            seed_name=seed_name,
            local_output_dir=local_output_dir,
            cluster_name=cluster_name,
            status=JobStatus.RETRIEVING,
            retrieval=retrieval,
        )

        if not self.is_connected or not self._tracked_job_matches_connected_cluster(tracked):
            self._fail(tracked, RETRY_RETRIEVAL_NOT_CONNECTED_MESSAGE)
            return

        try:
            self._retrieve_results(tracked)
        except Exception as exc:  # noqa: BLE001 - resolve rather than strand in RETRIEVING
            self._fail(tracked, f"{RETRIEVAL_FAILURE_PREFIX}: {exc}")

    def shutdown(self) -> None:
        """Best-effort shutdown during application close."""

        self.disconnect()

    def _track(
        self,
        job_id: UUID,
        *,
        remote_job_id: str,
        remote_dir: str,
        seed_name: str,
        local_output_dir: Path,
        cluster_name: str,
        status: JobStatus,
        retrieval: dict[str, Any],
    ) -> _TrackedClusterJob:
        """Register (or replace) the tracked entry for ``job_id`` and return it.

        ``retrieval`` is the caller's :class:`RetrievalFlags` keywords. An entry
        whose status is not one of the four active ones is tracked as terminal.
        """

        tracked = _TrackedClusterJob(
            job_id=job_id,
            remote_job_id=remote_job_id.strip(),
            remote_dir=remote_dir.strip(),
            seed_name=seed_name,
            local_output_dir=Path(local_output_dir),
            cluster_name=cluster_name.strip(),
            status=status,
            terminal=status not in _ACTIVE_TRACKED_STATUSES,
            retrieval=RetrievalFlags(**retrieval),
        )
        self._tracked_jobs[job_id] = tracked
        return tracked

    def _fail(self, entry: _TrackedClusterJob, message: str) -> None:
        """Mark ``entry`` terminal and report it: ``job_failed`` then ``status_changed``."""

        entry.terminal = True
        self.job_failed.emit(entry.job_id, message)
        self.status_changed.emit(entry.job_id, JobStatus.FAILED.value)

    def _tracked_job_matches_connected_cluster(self, entry: _TrackedClusterJob) -> bool:
        connected_cluster_key = (
            self._config.name.strip().casefold()
            if self._config is not None and self._config.name.strip()
            else ""
        )
        return (
            not entry.cluster_name.strip()
            or not connected_cluster_key
            or entry.cluster_name.strip().casefold() == connected_cluster_key
        )

    def _has_active_jobs_for_connected_cluster(self) -> bool:
        return any(self._is_polled(entry) for entry in self._tracked_jobs.values())

    def _is_polled(self, entry: _TrackedClusterJob) -> bool:
        """Return whether queue polls check ``entry``: active or watched, on this cluster."""

        return (
            not entry.terminal or entry.watch_for_recovery
        ) and self._tracked_job_matches_connected_cluster(entry)

    @staticmethod
    def remote_job_directory(*, working_directory: str, seed_name: str) -> str:
        """Build deterministic remote path for one submitted job."""

        ClusterJobRunner._validate_seed_name(seed_name)
        return posixpath.join(working_directory.strip(), seed_name.strip())

    @staticmethod
    def invalid_remote_job_labels(labels: list[str]) -> list[str]:
        """Return labels that are unsafe to use for remote execution."""

        return invalid_remote_job_labels(labels)

    def remote_directory_exists(self, remote_directory: str) -> bool:
        """Return true when remote directory exists."""

        ssh = self._require_ssh()
        return ssh.is_dir(remote_directory)

    def remote_file_exists(self, remote_path: str) -> bool:
        """Return true when remote file exists.

        Used to probe for a job's ``{seed}.check`` checkpoint file ahead of a
        relaxation continuation (see :class:`CheckpointProbeResult`).
        """

        ssh = self._require_ssh()
        return ssh.is_file(remote_path)

    def remove_remote_directory(self, remote_directory: str) -> None:
        """Remove one remote directory recursively."""

        ssh = self._require_ssh()
        ssh.rmtree(remote_directory)

    def scheduler_job_is_active(self, remote_job_id: str) -> bool:
        """Return whether the scheduler still lists ``remote_job_id`` as unfinished.

        A job missing from the queue is looked up in scheduler accounting, so a
        pending job the queue command filters out still counts as unfinished. A
        job neither lists, or one in a terminal state, is finished. Errors
        propagate: a caller about to delete the job's directory must refuse
        rather than guess.
        """

        ssh = self._require_ssh()
        scheduler = self._require_scheduler()
        info = scheduler.status(ssh, [remote_job_id]).get(remote_job_id)
        if info is None:
            info = scheduler.history(ssh, [remote_job_id]).get(remote_job_id)
        return info is not None and not scheduler.is_terminal(info.state)

    def _poll_queue(self) -> None:
        if not self.is_connected:
            self._poll_timer.stop()
            if self._ssh is not None:
                # The transport died under an open connection. Close it so the
                # connection state stops reporting a live link.
                cluster_name = self._config.name if self._config is not None else "the cluster"
                has_polled_jobs = self._has_active_jobs_for_connected_cluster()
                self.disconnect()
                if has_polled_jobs:
                    self.poll_failed.emit(
                        f"Queue polling failed for {cluster_name}: the connection was lost."
                    )
            return

        ssh = self._require_ssh()
        scheduler = self._require_scheduler()

        polled = [entry for entry in self._tracked_jobs.values() if self._is_polled(entry)]
        if not polled:
            self.queue_checked.emit(QueueCheckSnapshot(checked_at=utc_now(), elapsed_by_job_id={}))
            return

        try:
            statuses = scheduler.status(ssh, [entry.remote_job_id for entry in polled])
        except Exception as exc:  # noqa: BLE001
            cluster_name = self._config.name if self._config is not None else "the cluster"
            self.disconnect()
            self.poll_failed.emit(f"Queue polling failed for {cluster_name}: {exc}")
            return

        history = self._scheduler_history(
            ssh,
            scheduler,
            [entry.remote_job_id for entry in polled if entry.remote_job_id not in statuses],
        )

        observations_by_job_id: dict[UUID, str] = {}
        elapsed_by_job_id: dict[UUID, str] = {}
        watched_job_ids = frozenset(entry.job_id for entry in polled if entry.watch_for_recovery)
        for entry in polled:
            info = statuses.get(entry.remote_job_id)
            missing_from_queue = info is None
            if info is None:
                info = history.get(entry.remote_job_id)

            if entry.watch_for_recovery:
                observations_by_job_id[entry.job_id] = self._check_reappearance(
                    entry, info, scheduler
                )
                continue

            if info is not None and info.elapsed_runtime:
                elapsed_by_job_id[entry.job_id] = info.elapsed_runtime

            if info is None:
                # Neither the queue nor accounting lists the job: usually
                # completion with expired history, so retrieval is attempted,
                # but output-unavailable errors are retried across a short
                # grace window instead of failing the job immediately.
                observations_by_job_id[entry.job_id] = self._retrieve_results(
                    entry, missing_from_queue=True
                )
                continue

            observations_by_job_id[entry.job_id] = self._apply_scheduler_state(
                entry, info, scheduler, missing_from_queue=missing_from_queue
            )

        self.queue_checked.emit(
            QueueCheckSnapshot(
                checked_at=utc_now(),
                elapsed_by_job_id=elapsed_by_job_id,
                observations_by_job_id=observations_by_job_id,
                watched_job_ids=watched_job_ids,
            )
        )

    @staticmethod
    def _scheduler_history(
        ssh, scheduler: Scheduler, job_ids: list[str]
    ) -> dict[str, SchedulerJobInfo]:
        """Return accounting records for ``job_ids``, or none if the lookup fails.

        The queue poll already succeeded, so a failed accounting lookup leaves
        the jobs unknown rather than disconnecting.
        """

        if not job_ids:
            return {}
        try:
            return scheduler.history(ssh, job_ids)
        except Exception:  # noqa: BLE001 - accounting is a best-effort fallback
            return {}

    def _apply_scheduler_state(
        self,
        entry: _TrackedClusterJob,
        info: SchedulerJobInfo,
        scheduler: Scheduler,
        *,
        missing_from_queue: bool,
    ) -> str:
        """Act on the scheduler state ``info`` reports for ``entry`` and return the observation.

        ``missing_from_queue`` marks a state read from accounting because the
        queue did not list the job.
        """

        mapped = scheduler.map_to_job_status(info.state)

        if mapped == JobStatus.COMPLETED:
            entry.status = mapped
            return self._retrieve_results(
                entry, missing_from_queue=missing_from_queue, accounted=missing_from_queue
            )

        # The job is listed as unfinished, so any earlier grace window restarts.
        entry.missing_retrieval_attempts = 0
        source = "scheduler accounting" if missing_from_queue else "scheduler"
        observation = queue_observation(
            f"{source} reports {mapped.value.upper()}", missing_from_queue=missing_from_queue
        )
        if mapped == entry.status:
            return observation
        if mapped == JobStatus.QUEUED and entry.status == JobStatus.RUNNING:
            # No transition leads back to queued: a running job SLURM suspends
            # or requeues stays running, and starting again is not a new start.
            return observation

        previous = entry.status
        entry.status = mapped
        if mapped in (JobStatus.QUEUED, JobStatus.RUNNING):
            self.status_changed.emit(entry.job_id, mapped.value)
            if mapped == JobStatus.RUNNING and previous in (JobStatus.SUBMITTING, JobStatus.QUEUED):
                self.job_started.emit(entry.job_id)
            return observation
        if mapped == JobStatus.CANCELLED:
            entry.terminal = True
            self.job_cancelled.emit(entry.job_id)
            self.status_changed.emit(entry.job_id, JobStatus.CANCELLED.value)
            return observation
        if mapped in (JobStatus.STOPPED, JobStatus.FAILED):
            entry.terminal = True
            # Download the error files and scheduler log, so a stop can be
            # diagnosed and inspected like a failure.
            retrieve_error_files(
                self._require_ssh(),
                remote_job_id=entry.remote_job_id,
                remote_dir=entry.remote_dir,
                seed_name=entry.seed_name,
                local_output_dir=entry.local_output_dir,
            )
            if mapped == JobStatus.STOPPED:
                self.status_changed.emit(entry.job_id, JobStatus.STOPPED.value)
            else:
                self._fail(
                    entry, f"Scheduler reported the job as {scheduler.state_name(info.state)}."
                )
        return observation

    def _check_reappearance(
        self,
        entry: _TrackedClusterJob,
        info: SchedulerJobInfo | None,
        scheduler: Scheduler,
    ) -> str:
        """Check a watched job against this poll and return the observation.

        Any scheduler record ends the watch. A queued, running or completed job
        is reported through ``job_reappeared``; any other state leaves the job
        failed.
        """

        if info is None:
            return queue_observation(
                "not in scheduler accounting; watching for the job to reappear",
                missing_from_queue=True,
            )

        entry.watch_for_recovery = False
        mapped = scheduler.map_to_job_status(info.state)
        if mapped in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED):
            self.job_reappeared.emit(entry.job_id, mapped.value)
            return f"scheduler reports {mapped.value.upper()} again; tracking resumed"
        return f"scheduler reports {scheduler.state_name(info.state)}"

    def _retrieve_results(
        self,
        entry: _TrackedClusterJob,
        *,
        missing_from_queue: bool = False,
        accounted: bool = False,
    ) -> str:
        """Run one retrieval pass for ``entry``, emit its outcome, and return the observation.

        :func:`~mufinder.services.cluster_retrieval.retrieve_job_outputs` decides
        the outcome. A retry-eligible failure is retried for up to
        ``MISSING_ENTRY_RETRIEVAL_ATTEMPTS`` polls before the job is failed.
        ``accounted`` marks a job that scheduler accounting reports as completed:
        it fails with the retrieval error. Otherwise the scheduler listed the
        job nowhere, so it fails as a missing scheduler entry and stays watched
        for the scheduler listing it again.
        """

        ssh = self._require_ssh()
        self.status_changed.emit(entry.job_id, JobStatus.RETRIEVING.value)
        result = retrieve_job_outputs(
            ssh,
            remote_job_id=entry.remote_job_id,
            remote_dir=entry.remote_dir,
            seed_name=entry.seed_name,
            local_output_dir=entry.local_output_dir,
            retrieval=entry.retrieval,
            missing_from_queue=missing_from_queue,
        )
        failure = result.failure
        if failure is not None and failure.retry_eligible:
            entry.missing_retrieval_attempts += 1
            attempts = entry.missing_retrieval_attempts
            if attempts < MISSING_ENTRY_RETRIEVAL_ATTEMPTS:
                detail = (
                    f"waiting for output (attempt {attempts} of {MISSING_ENTRY_RETRIEVAL_ATTEMPTS})"
                )
                return queue_observation(detail, missing_from_queue=True)
            if accounted:
                self._fail(entry, failure.message)
            else:
                self._fail(entry, MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE)
                entry.watch_for_recovery = True
            detail = (
                f"output could not be retrieved after {MISSING_ENTRY_RETRIEVAL_ATTEMPTS} checks"
            )
            return queue_observation(detail, missing_from_queue=True)
        if failure is not None:
            self._fail(entry, failure.message)
            return result.observation

        entry.terminal = True
        if result.outcome is RetrievalOutcome.COMPLETED:
            self.job_finished.emit(entry.job_id, result.parsed, result.artifacts)
            self.status_changed.emit(entry.job_id, JobStatus.COMPLETED.value)
        else:
            self.status_changed.emit(entry.job_id, JobStatus.STOPPED.value)
        return result.observation

    def _require_ssh(self) -> SSHConnection:
        if self._ssh is None or not self._ssh.is_connected:
            raise ConnectionError("Cluster connection is not active.")
        return self._ssh

    def _require_scheduler(self) -> Scheduler:
        if self._scheduler is None:
            raise ConnectionError("Cluster scheduler is not configured.")
        return self._scheduler

    @staticmethod
    def _validate_seed_name(seed_name: str) -> None:
        invalid = invalid_remote_job_labels([seed_name])
        if not invalid:
            return
        raise ConnectionError(
            "Remote job labels may contain only letters, numbers, dot, underscore, and hyphen. "
            f"Rename the job before cluster submission: {invalid[0]}"
        )
