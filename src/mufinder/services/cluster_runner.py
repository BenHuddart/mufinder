"""Remote cluster execution service using SSH and scheduler polling."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from mufinder.services.cluster_retrieval import retrieve_job_outputs
from mufinder.services.cluster_queues import ProcessPlacement
from mufinder.services.cluster_security import invalid_remote_job_labels
from mufinder.services.job_artifacts import RetrievalFlags
from mufinder.services.job_progress import GeometryProgressReader, remote_castep_candidates
from mufinder.services.job_reconcile import JobEvidence, OutputState, SchedulerState, wants_output
from mufinder.services.scheduler import (
    Scheduler,
    SchedulerJobInfo,
    SchedulerSubmitRequest,
    create_scheduler,
)
from mufinder.services.script_template import render_job_script
from mufinder.services.ssh_client import SSHConnection, SSHCredentials

#: The scheduler evidence each mapped scheduler status stands for.
_SCHEDULER_STATE_BY_STATUS = {
    JobStatus.QUEUED: SchedulerState.PENDING,
    JobStatus.RUNNING: SchedulerState.RUNNING,
    JobStatus.COMPLETED: SchedulerState.COMPLETED,
    JobStatus.CANCELLED: SchedulerState.CANCELLED,
    JobStatus.STOPPED: SchedulerState.TIMED_OUT,
    JobStatus.FAILED: SchedulerState.FAILED,
}

_REPORTED_FINISHED = frozenset(
    {SchedulerState.COMPLETED, SchedulerState.TIMED_OUT, SchedulerState.FAILED}
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


@dataclass(frozen=True, slots=True)
class CheckTarget:
    """One cluster job to look at, with where its scheduler entry and files are.

    ``remote_job_id`` is empty for a job that never recorded a scheduler job
    ID; its folder can still be looked at. ``on_request`` marks a check the
    user asked for, which looks at the output even of a cancelled job.
    """

    job_id: UUID
    remote_job_id: str
    remote_dir: str
    seed_name: str
    local_output_dir: Path
    cluster_name: str = ""
    retrieval: RetrievalFlags = RetrievalFlags()
    on_request: bool = False


@dataclass(frozen=True, slots=True)
class QueueCheckSnapshot:
    """Summary emitted after one pass over a list of check targets."""

    checked_at: datetime
    elapsed_by_job_id: dict[UUID, str]
    #: The jobs this pass looked at.
    job_ids: tuple[UUID, ...] = ()


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

    job_cancelled = Signal(object)
    #: ``(job_id, JobEvidence)``: what one check saw of a job. A job whose
    #: output is about to be downloaded is reported twice, first with the
    #: scheduler's answer alone.
    job_checked = Signal(object, object)

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
        self._credential_cache: SSHCredentials | None = None
        self._progress_reader = GeometryProgressReader()
        # Bumped by forget_progress, so a read continuing in turns can tell
        # that it belongs to a project that is no longer open.
        self._progress_generation = 0

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
            accounting_command_path=config.accounting_command_path,
        )
        self._credential_cache = resolved_credentials
        self.connection_status_changed.emit(True)

    def disconnect(self) -> None:
        """Disconnect from the cluster."""

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
        processes_per_node: int | None = None,
        castep_executable: str = "",
        remote_directory: str | None = None,
        reuse_existing_files: bool = False,
        allow_existing_remote_directory: bool = False,
        **retrieval: Any,
    ) -> str:
        """Upload files, submit the scheduler script, and return the scheduler job ID.

        ``retrieval`` holds the :class:`~mufinder.services.job_artifacts.RetrievalFlags`
        fields as keywords (``retrieve_den_fmt``, ``task_farming_enabled``, ...).
        They are accepted so one set of submission keywords serves both runners;
        the cluster runner keeps nothing about a submitted job.

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
        if processes_per_node is not None:
            # Refuse an impossible split before anything reaches the cluster.
            ProcessPlacement.split(max(1, int(num_processes)), int(processes_per_node))

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
            processes_per_node=processes_per_node,
        )
        return scheduler.submit(ssh, request)

    def cancel(self, job_id: UUID, remote_job_id: str = "") -> None:
        """Cancel scheduler job ``remote_job_id`` and report ``job_id`` cancelled.

        A job with no scheduler job ID has nothing to cancel on the cluster,
        so nothing is reported for it.
        """

        ssh = self._require_ssh()
        scheduler = self._require_scheduler()
        if not remote_job_id.strip():
            return
        scheduler.cancel(ssh, remote_job_id.strip())
        self.job_cancelled.emit(job_id)

    def read_geometry_progress(
        self, targets: list[CheckTarget], _generation: int | None = None
    ) -> None:
        """Read jobs' geometry-optimisation progress from their remote output.

        Fetches only what CASTEP appended since the previous read of each job.
        Emits :attr:`geometry_progress_read` once per target, when its output
        is read to the end: ``None`` when the job is not on the connected
        cluster, the cluster is disconnected, the output cannot be read, or
        :meth:`forget_progress` was called while it was being read.

        Each call does one bounded read of the first target and queues the rest
        of the work behind whatever else is waiting for this thread, so reading
        large accumulated outputs never holds up a queue check, a submission or
        a checkpoint probe for longer than one read.
        """

        remaining = list(targets)
        if not remaining:
            return
        generation = self._progress_generation if _generation is None else _generation
        target = remaining[0]
        ssh = self._ssh if self.is_connected else None
        progress = None
        more_to_read = False
        if (
            ssh is not None
            and generation == self._progress_generation
            and self._on_connected_cluster(target)
        ):
            candidates = remote_castep_candidates(
                target.remote_dir, target.seed_name, target.retrieval.farm_count
            )
            try:
                progress = self._progress_reader.read_remote(target.job_id, ssh, candidates)
                more_to_read = self._progress_reader.has_backlog(target.job_id)
            except Exception:  # noqa: BLE001 - a failed read keeps the last progress shown
                progress = None
        if not more_to_read:
            del remaining[0]
            self.geometry_progress_read.emit({target.job_id: progress})
        if remaining:
            QTimer.singleShot(0, self, lambda: self.read_geometry_progress(remaining, generation))

    def forget_progress(self) -> None:
        """Drop the progress read positions and end reads in flight.

        Called when another project is opened: a large output still being read
        in turns belongs to the project that was closed.
        """

        self._progress_generation += 1
        self._progress_reader.forget_all_except(())

    def check_jobs(self, targets: list[CheckTarget]) -> None:
        """Look at each of ``targets`` on the connected cluster and report what is seen.

        One queue command covers every target, and one accounting command
        those the queue does not list. Each job is reported through
        :attr:`job_checked`; a job the scheduler no longer runs has its output
        folder looked at first. Targets of another cluster are skipped. The
        pass ends with :attr:`queue_checked`, or with :attr:`poll_failed` when
        the connection is gone or the queue command fails.
        """

        if not self.is_connected:
            if self._ssh is not None:
                # The transport died under an open connection. Close it so the
                # connection state stops reporting a live link.
                cluster_name = self._config.name if self._config is not None else "the cluster"
                self.disconnect()
                if targets:
                    self.poll_failed.emit(
                        f"Queue polling failed for {cluster_name}: the connection was lost."
                    )
            return

        ssh = self._require_ssh()
        scheduler = self._require_scheduler()
        checked = [target for target in targets if self._on_connected_cluster(target)]
        listed_ids = [target.remote_job_id for target in checked if target.remote_job_id]
        checked_at = utc_now()
        try:
            statuses = scheduler.status(ssh, listed_ids) if listed_ids else {}
        except Exception as exc:  # noqa: BLE001
            cluster_name = self._config.name if self._config is not None else "the cluster"
            for target in checked:
                self.job_checked.emit(
                    target.job_id,
                    self._evidence(target, checked_at, SchedulerState.UNAVAILABLE, detail=str(exc)),
                )
            self.disconnect()
            self.poll_failed.emit(f"Queue polling failed for {cluster_name}: {exc}")
            return

        history = self._scheduler_history(
            ssh, scheduler, [job_id for job_id in listed_ids if job_id not in statuses]
        )

        elapsed_by_job_id: dict[UUID, str] = {}
        for target in checked:
            info = statuses.get(target.remote_job_id)
            from_accounting = info is None
            if info is None:
                info = history.get(target.remote_job_id)
            if info is not None and info.elapsed_runtime:
                elapsed_by_job_id[target.job_id] = info.elapsed_runtime

            if not target.remote_job_id:
                state, detail = SchedulerState.NO_JOB_ID, ""
            elif info is None:
                state, detail = SchedulerState.NOT_LISTED, ""
            else:
                state = _SCHEDULER_STATE_BY_STATUS[scheduler.map_to_job_status(info.state)]
                detail = scheduler.state_name(info.state)
            evidence = self._evidence(
                target,
                checked_at,
                state,
                detail=detail,
                from_accounting=from_accounting and info is not None,
            )
            if wants_output(state, on_request=target.on_request) and target.remote_dir:
                if state in _REPORTED_FINISHED:
                    # Reported before the download, so the job reads as
                    # retrieving while its output is fetched.
                    self.job_checked.emit(target.job_id, evidence)
                evidence = self._with_output(ssh, target, evidence)
            self.job_checked.emit(target.job_id, evidence)

        self.queue_checked.emit(
            QueueCheckSnapshot(
                checked_at=utc_now(),
                elapsed_by_job_id=elapsed_by_job_id,
                job_ids=tuple(target.job_id for target in checked),
            )
        )

    @staticmethod
    def _evidence(
        target: CheckTarget,
        checked_at: datetime,
        state: SchedulerState,
        *,
        detail: str = "",
        from_accounting: bool = False,
    ) -> JobEvidence:
        return JobEvidence(
            checked_at=checked_at,
            scheduler=state,
            scheduler_detail=detail,
            from_accounting=from_accounting,
            remote_job_id=target.remote_job_id,
            remote_directory=target.remote_dir,
            on_request=target.on_request,
        )

    @staticmethod
    def _with_output(ssh: SSHConnection, target: CheckTarget, evidence: JobEvidence) -> JobEvidence:
        """Return ``evidence`` with what the job's remote folder holds."""

        try:
            result = retrieve_job_outputs(
                ssh,
                remote_job_id=target.remote_job_id,
                remote_dir=target.remote_dir,
                seed_name=target.seed_name,
                local_output_dir=target.local_output_dir,
                retrieval=target.retrieval,
            )
        except Exception as exc:  # noqa: BLE001 - one job's folder must not end the pass
            return replace(evidence, output=OutputState.UNREACHABLE, output_detail=str(exc))
        return replace(
            evidence,
            output=result.output,
            output_detail=result.detail,
            walltime_seen=result.walltime_seen,
            parsed=result.parsed,
            artifacts=result.artifacts,
        )

    def shutdown(self) -> None:
        """Best-effort shutdown during application close."""

        self.disconnect()

    def _on_connected_cluster(self, target: CheckTarget) -> bool:
        connected_cluster_key = (
            self._config.name.strip().casefold()
            if self._config is not None and self._config.name.strip()
            else ""
        )
        return (
            not target.cluster_name.strip()
            or not connected_cluster_key
            or target.cluster_name.strip().casefold() == connected_cluster_key
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

    @staticmethod
    def _scheduler_history(
        ssh, scheduler: Scheduler, job_ids: list[str]
    ) -> dict[str, SchedulerJobInfo]:
        """Return accounting records for ``job_ids``, or none if the lookup fails.

        The queue command already succeeded, so a failed accounting lookup
        leaves the jobs unlisted rather than disconnecting.
        """

        if not job_ids:
            return {}
        try:
            return scheduler.history(ssh, job_ids)
        except Exception:  # noqa: BLE001 - accounting is a best-effort fallback
            return {}

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
