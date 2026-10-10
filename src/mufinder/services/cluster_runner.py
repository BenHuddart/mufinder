"""Remote cluster execution service using SSH and scheduler polling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
import posixpath
import re
import tempfile
from typing import Any
from uuid import UUID

from PySide6.QtCore import QObject, QTimer, Signal

from mufinder.cluster_setup import ClusterConfig
from mufinder.castep_io.output_parser import CastepResult, parse_castep_output
from mufinder.domain.enums import FailureReason, JobStatus
from mufinder.domain.models import utc_now
from mufinder.services.cluster_security import invalid_remote_job_labels
from mufinder.services.failure_diagnosis import (
    MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE,
    RETRIEVAL_FAILURE_PREFIX,
    RETRY_RETRIEVAL_NOT_CONNECTED_MESSAGE,
    diagnose_failure,
    find_failure_log_files,
)
from mufinder.services.job_artifacts import RetrievalFlags
from mufinder.services.scheduler import Scheduler, SchedulerSubmitRequest, create_scheduler
from mufinder.services.script_template import render_job_script
from mufinder.services.ssh_client import SSHConnection, SSHCredentials

MISSING_ENTRY_RETRIEVAL_ATTEMPTS = 3
"""Polls a scheduler-missing job may fail output retrieval before it is failed.

A job absent from the scheduler queue has usually just completed, so retrieval
is attempted immediately. But the absence can also mean expired scheduler
history, a filtered queue command, or the wrong cluster — and the output may
simply not be flushed yet. Output-unavailable retrieval errors are therefore
retried for this many polls before the job is declared failed; parse failures
of successfully downloaded output still fail immediately.
"""


REMOTE_STALE_OUTPUT_SLACK_SECONDS = 5.0
"""Slack allowed when ordering two remote mtimes against each other.

Both sides of the comparison live on the cluster's own filesystem, so no
allowance for the offset between this machine and the cluster is needed — only
for remote mtime granularity (1 s on ext3/ext4, 2 s on some network stores) and
the sub-second NTP drift between a login node writing the uploaded input and a
compute node writing the output. The staleness being detected is a whole run
old, so a few seconds of slack costs nothing.
"""


_ACTIVE_TRACKED_STATUSES = frozenset(
    {JobStatus.SUBMITTING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING}
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

    connection_status_changed = Signal(bool)
    queue_checked = Signal(object)
    poll_failed = Signal(str)

    def __init__(self, parent: QObject | None = None, *, settings=None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._config: ClusterConfig | None = None
        self._ssh: SSHConnection | None = None
        self._scheduler: Scheduler | None = None
        self._tracked_jobs: dict[UUID, _TrackedClusterJob] = {}
        self._credential_cache: SSHCredentials | None = None

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
            and self._config.name.strip().casefold() == config.name.strip().casefold()
            and self._config.username.strip() == config.username.strip()
            and self._config.jump_username.strip() == config.jump_username.strip()
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
        script is therefore the sole carrier of walltime, task count, and memory.
        Reusing a stale script would silently ignore the resources the current
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
        **retrieval: Any,
    ) -> None:
        """Register a persisted cluster job for queue polling and retrieval.

        ``retrieval`` holds the :class:`~mufinder.services.job_artifacts.RetrievalFlags`
        fields as keywords.
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
        if (
            self.is_connected
            and not tracked.terminal
            and self._tracked_job_matches_connected_cluster(tracked)
            and not self._poll_timer.isActive()
        ):
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
        retrieval pass would strand the job in RETRIEVING, so the not-connected
        case is resolved to ``failed`` via ``job_failed`` + ``status_changed``
        instead. ``_retrieve_results`` already converts any SSH/parse error to
        ``failed`` internally, and the surrounding guard here catches the
        pre-``try`` ``_require_ssh`` raise for the same reason.

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
        return any(
            not entry.terminal and self._tracked_job_matches_connected_cluster(entry)
            for entry in self._tracked_jobs.values()
        )

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

        A job the scheduler no longer lists, or lists in a terminal state, is
        finished. Errors propagate: a caller about to delete the job's directory
        must refuse rather than guess.
        """

        ssh = self._require_ssh()
        scheduler = self._require_scheduler()
        info = scheduler.status(ssh, [remote_job_id]).get(remote_job_id)
        return info is not None and not scheduler.is_terminal(info.state)

    def _poll_queue(self) -> None:
        if not self.is_connected:
            self._poll_timer.stop()
            return

        ssh = self._require_ssh()
        scheduler = self._require_scheduler()

        connected_cluster_key = (
            self._config.name.strip().casefold()
            if self._config is not None and self._config.name.strip()
            else ""
        )
        active = [
            entry
            for entry in self._tracked_jobs.values()
            if not entry.terminal
            and (
                not entry.cluster_name.strip()
                or not connected_cluster_key
                or entry.cluster_name.strip().casefold() == connected_cluster_key
            )
        ]
        if not active:
            self.queue_checked.emit(
                QueueCheckSnapshot(
                    checked_at=utc_now(),
                    elapsed_by_job_id={},
                    observations_by_job_id={},
                )
            )
            return

        try:
            statuses = scheduler.status(ssh, [entry.remote_job_id for entry in active])
        except Exception as exc:  # noqa: BLE001
            cluster_name = self._config.name if self._config is not None else "the cluster"
            self.disconnect()
            self.poll_failed.emit(f"Queue polling failed for {cluster_name}: {exc}")
            return

        observations_by_job_id: dict[UUID, str] = {}
        for entry in active:
            info = statuses.get(entry.remote_job_id)

            if info is None:
                # Missing scheduler entry: usually completion, so retrieval is
                # attempted, but output-unavailable errors are retried across a
                # short grace window instead of failing the job immediately.
                observations_by_job_id[entry.job_id] = self._retrieve_results(
                    entry, missing_from_queue=True
                )
                continue

            entry.missing_retrieval_attempts = 0
            mapped = scheduler.map_to_job_status(info.state)

            if mapped == JobStatus.COMPLETED:
                entry.status = mapped
                observations_by_job_id[entry.job_id] = self._retrieve_results(entry)
                continue

            observations_by_job_id[entry.job_id] = f"scheduler reports {mapped.value.upper()}"
            if mapped == entry.status:
                continue

            entry.status = mapped
            if mapped == JobStatus.QUEUED:
                self.status_changed.emit(entry.job_id, JobStatus.QUEUED.value)
                continue
            if mapped == JobStatus.RUNNING:
                self.status_changed.emit(entry.job_id, JobStatus.RUNNING.value)
                continue
            if mapped == JobStatus.CANCELLED:
                entry.terminal = True
                self.job_cancelled.emit(entry.job_id)
                self.status_changed.emit(entry.job_id, JobStatus.CANCELLED.value)
                continue
            if mapped == JobStatus.STOPPED:
                entry.terminal = True
                # Download the scheduler log too, so the stop can be diagnosed
                # and inspected like a failure.
                self._retrieve_error_files(entry)
                self.status_changed.emit(entry.job_id, JobStatus.STOPPED.value)
                continue
            if mapped == JobStatus.FAILED:
                entry.terminal = True
                self._retrieve_error_files(entry)
                self._fail(entry, f"Scheduler reported the job as {scheduler.state_name(info.state)}.")
                continue

        elapsed_by_job_id = {
            entry.job_id: info.elapsed_runtime
            for entry in active
            if (info := statuses.get(entry.remote_job_id)) is not None and info.elapsed_runtime
        }
        self.queue_checked.emit(
            QueueCheckSnapshot(
                checked_at=utc_now(),
                elapsed_by_job_id=elapsed_by_job_id,
                observations_by_job_id=observations_by_job_id,
            )
        )

    def _retrieve_results(self, entry: _TrackedClusterJob, *, missing_from_queue: bool = False) -> str:
        """Retrieve and parse remote output; return a short observation.

        ``missing_from_queue`` distinguishes retrieval triggered because the
        scheduler no longer lists the job (usually completion, but possibly
        expired history or a filtered queue command) from retrieval for a job
        the scheduler explicitly reported as completed. In the missing case,
        output-unavailable download errors are retried for up to
        ``MISSING_ENTRY_RETRIEVAL_ATTEMPTS`` polls before the job is failed;
        parse failures of successfully downloaded output fail immediately in
        both cases.

        A continuation run reuses its remote directory, so the *previous* run's
        ``.castep`` and ``-out.cell`` are still sitting there and neither one's
        existence proves this run wrote it. Retrieval therefore orders both
        against the ``{seed}.param`` uploaded at submission — see
        :meth:`_submission_marker_mtime` and
        :meth:`_remote_output_predates_submission`. A stale primary ``.castep``
        aborts retrieval (the alternative is completing the job on the previous
        run's success banner); a stale ``-out.cell`` is simply not downloaded,
        which leaves the local copy carrying its previous-retrieval mtime so
        ``MainWindow``'s local stale-output guard rejects it. Both comparisons
        fail open when the marker is unavailable.
        """

        ssh = self._require_ssh()
        prefix = "missing from scheduler queue; " if missing_from_queue else ""

        self.status_changed.emit(entry.job_id, JobStatus.RETRIEVING.value)
        remote_param = posixpath.join(entry.remote_dir, f"{entry.seed_name}.param")
        remote_castep = posixpath.join(entry.remote_dir, f"{entry.seed_name}.castep")
        remote_farm_castep = posixpath.join(entry.remote_dir, f"{entry.seed_name}_farm001.castep")
        remote_den_fmt = posixpath.join(entry.remote_dir, f"{entry.seed_name}.den_fmt")
        remote_out_cell = posixpath.join(entry.remote_dir, f"{entry.seed_name}-out.cell")
        remote_phonon = posixpath.join(entry.remote_dir, f"{entry.seed_name}.phonon")
        # Task farming renames every output to <seed>_farmNNN.<ext>, so a farmed
        # NEB run writes no <seed>.ts at all. Every farm writes an identical,
        # complete band; farm 001 is the canonical copy. Download it under the
        # same name locally so collect_job_artifacts finds it where it looks.
        flags = entry.retrieval
        farm_count = flags.farm_count
        ts_basename = f"{entry.seed_name}_farm001.ts" if farm_count > 1 else f"{entry.seed_name}.ts"
        remote_ts = posixpath.join(entry.remote_dir, ts_basename)
        local_castep = entry.local_output_dir / f"{entry.seed_name}.castep"
        local_farm_castep = entry.local_output_dir / f"{entry.seed_name}_farm001.castep"
        local_den_fmt = entry.local_output_dir / f"{entry.seed_name}.den_fmt"
        local_out_cell = entry.local_output_dir / f"{entry.seed_name}-out.cell"
        local_phonon = entry.local_output_dir / f"{entry.seed_name}.phonon"
        local_ts = entry.local_output_dir / ts_basename

        try:
            submitted_at = self._submission_marker_mtime(ssh, remote_param)
            # Logs are (re)downloaded just below, so any local log older than
            # this moment is a previous run's leftover in a reused directory.
            logs_fetched_from = datetime.now(UTC)
            self._retrieve_error_files(entry)
            use_farm = farm_count > 1 and ssh.exists(remote_farm_castep)
            primary_remote = remote_farm_castep if use_farm else remote_castep
            if self._remote_output_predates_submission(ssh, primary_remote, submitted_at):
                # The run wrote nothing at all into the reused directory (it
                # failed to start, or died before CASTEP opened its log). The
                # surviving .castep is the PREVIOUS run's — including its
                # success banner — so parsing it would complete this job on the
                # old geometry under parameters that were never applied. Treat
                # it exactly like an unavailable output: the caller's grace
                # window applies when the trigger was a missing queue entry.
                raise ConnectionError(
                    f"{posixpath.basename(primary_remote)} predates this submission: "
                    "the run produced no new output in the reused remote directory."
                )
            if use_farm:
                if ssh.exists(remote_castep):
                    ssh.get(remote_castep, local_castep)
                ssh.get(remote_farm_castep, local_farm_castep)
                primary_output = local_farm_castep
            else:
                ssh.get(remote_castep, local_castep)
                primary_output = local_castep
            if flags.retrieve_den_fmt and ssh.exists(remote_den_fmt):
                ssh.get(remote_den_fmt, local_den_fmt)
            if (
                flags.retrieve_out_cell
                and ssh.exists(remote_out_cell)
                and not self._remote_output_predates_submission(ssh, remote_out_cell, submitted_at)
            ):
                ssh.get(remote_out_cell, local_out_cell)
            if flags.retrieve_phonon and ssh.exists(remote_phonon):
                ssh.get(remote_phonon, local_phonon)
            if flags.retrieve_ts and ssh.exists(remote_ts):
                ssh.get(remote_ts, local_ts)
        except Exception as exc:  # noqa: BLE001
            # Output-unavailable failure (SFTP download error, missing remote
            # .castep, connection hiccup). For a scheduler-missing job the
            # output may simply not exist yet, so defer the failure across a
            # short grace window and retry on subsequent polls.
            if missing_from_queue:
                entry.missing_retrieval_attempts += 1
                if entry.missing_retrieval_attempts < MISSING_ENTRY_RETRIEVAL_ATTEMPTS:
                    return (
                        f"{prefix}waiting for output "
                        f"(attempt {entry.missing_retrieval_attempts} of "
                        f"{MISSING_ENTRY_RETRIEVAL_ATTEMPTS})"
                    )
                self._fail(entry, MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE)
                return (
                    f"{prefix}output could not be retrieved after "
                    f"{MISSING_ENTRY_RETRIEVAL_ATTEMPTS} checks"
                )
            self._fail(entry, f"{RETRIEVAL_FAILURE_PREFIX}: {exc}")
            return f"{prefix}output retrieval failed"

        try:
            parsed = parse_castep_output(primary_output)
        except Exception as exc:  # noqa: BLE001
            # The output was downloaded but is unreadable: a real failure in
            # both trigger modes, so no grace window applies.
            self._fail(entry, f"{RETRIEVAL_FAILURE_PREFIX}: {exc}")
            return f"{prefix}output parsing failed"

        if not parsed.completed:
            entry.terminal = True
            if self._stopped_at_walltime(entry, parsed, logs_fetched_from=logs_fetched_from):
                self.status_changed.emit(entry.job_id, JobStatus.STOPPED.value)
                return f"{prefix}output incomplete; stopped (walltime)"
            self._fail(entry, parsed.error_message or "CASTEP did not report successful completion.")
            return f"{prefix}output incomplete; job failed"

        entry.terminal = True
        # Parse the secondary outputs (relaxed cell / phonon / transition state)
        # here on the worker thread, exactly once, driven by the retrieval flags
        # this job tracked. The GUI handler decides failure from the carrier and
        # the owning controller consumes the pre-parsed data — neither re-reads
        # disk. Errors are captured on the carrier, never raised.
        artifacts = flags.collect(entry.local_output_dir, entry.seed_name)
        self.job_finished.emit(entry.job_id, parsed, artifacts)
        self.status_changed.emit(entry.job_id, JobStatus.COMPLETED.value)
        return f"{prefix}output retrieved and parsed"

    @staticmethod
    def _submission_marker_mtime(ssh: SSHConnection, remote_param: str) -> float | None:
        """Return the remote mtime that marks when this submission happened.

        ``{seed}.param`` is uploaded fresh on every submission that does not
        set ``reuse_existing_files`` — including every continuation, and
        regardless of queue system (unlike ``run.sh``, which ``addqueue`` never
        receives). Its remote mtime is therefore a same-clock marker for "this
        run's inputs landed", which any output the run wrote must postdate.

        ``None`` means no ordering is available and every comparison against it
        fails open: the file is missing (a ``reuse_existing_files``
        resubmission never re-uploaded it), the SFTP server reported no mtime,
        or the connection object predates the ``mtime`` primitive.
        """

        probe = getattr(ssh, "mtime", None)
        if probe is None:
            return None
        try:
            return probe(remote_param)
        except Exception:  # noqa: BLE001 - a stat failure must not fail retrieval
            return None

    @staticmethod
    def _remote_output_predates_submission(
        ssh: SSHConnection,
        remote_path: str,
        submitted_at: float | None,
    ) -> bool:
        """Return whether ``remote_path`` was last written before this submission.

        Both mtimes come from the cluster's own filesystem, so they are
        directly comparable up to ``REMOTE_STALE_OUTPUT_SLACK_SECONDS``. An
        unknown marker or an unreadable output mtime returns ``False`` — an
        output is only ever rejected on positive evidence that it is stale.
        """

        if submitted_at is None:
            return False
        probe = getattr(ssh, "mtime", None)
        if probe is None:
            return False
        try:
            modified = probe(remote_path)
        except Exception:  # noqa: BLE001 - see _submission_marker_mtime
            return False
        if modified is None:
            return False
        return (submitted_at - modified) > REMOTE_STALE_OUTPUT_SLACK_SECONDS

    def _retrieve_error_files(self, entry: _TrackedClusterJob) -> None:
        ssh = self._require_ssh()
        try:
            entry.local_output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

        for name in self._remote_error_file_names(entry):
            remote_path = posixpath.join(entry.remote_dir, name)
            local_path = entry.local_output_dir / name
            try:
                if ssh.exists(remote_path):
                    ssh.get(remote_path, local_path)
            except Exception:  # noqa: BLE001
                continue

    def _remote_error_file_names(self, entry: _TrackedClusterJob) -> list[str]:
        names: list[str] = [f"{entry.seed_name}.err"]
        remote_job_id = entry.remote_job_id.strip()
        if remote_job_id and re.fullmatch(r"[A-Za-z0-9_.-]+", remote_job_id):
            names.append(f"slurm-{remote_job_id}.out")
        if not hasattr(self._require_ssh(), "listdir"):
            return names

        try:
            entries = self._require_ssh().listdir(entry.remote_dir)
        except Exception:  # noqa: BLE001
            return names

        # CASTEP embeds the MPI rank in every error filename (io_parallel_filename),
        # and task farming additionally rewrites the seedname to <seed>_farmNNN
        # before stderr is opened — so a farmed run writes <seed>_farm002.0003.err
        # and never <seed>.0003.err. Accept the optional farm infix so failure
        # diagnostics come back for farmed NEB and phonon jobs too.
        numbered = sorted(
            str(name)
            for name in entries
            if re.fullmatch(rf"{re.escape(entry.seed_name)}(_farm\d+)?\.\d+\.err", str(name))
        )
        for name in numbered:
            if name not in names:
                names.append(name)
        if remote_job_id and re.fullmatch(r"[A-Za-z0-9_.-]+", remote_job_id):
            scheduler_outputs = sorted(
                str(name)
                for name in entries
                if re.fullmatch(rf".*-{re.escape(remote_job_id)}\.out", str(name))
            )
            for name in scheduler_outputs:
                if name not in names:
                    names.append(name)
        return names

    @staticmethod
    def _stopped_at_walltime(
        entry: _TrackedClusterJob,
        parsed: CastepResult,
        *,
        logs_fetched_from: datetime | None = None,
    ) -> bool:
        """Return whether an incomplete run was killed at the walltime limit.

        A walltime kill leaves no trace in the ``.castep`` file: the scheduler
        records it in its own log, which ``_retrieve_error_files`` has already
        downloaded next to the output. Local logs older than
        ``logs_fetched_from`` were not part of that download — a continuation
        reuses its directory, so they can be an earlier run's — and are ignored.
        """

        log_files = find_failure_log_files(
            entry.local_output_dir, [entry.seed_name], entry.remote_job_id
        )
        diagnosis = diagnose_failure(parsed.error_message, log_files, not_before=logs_fetched_from)
        return diagnosis.reason == FailureReason.WALLTIME

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
