"""Remote cluster execution service using SSH and scheduler polling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
import posixpath
import shlex
import tempfile
from uuid import UUID

from PySide6.QtCore import QObject, QTimer, Signal

from mufinder.cluster_setup import ClusterConfig
from mufinder.castep_io.output_parser import CastepResult, parse_castep_output
from mufinder.domain.enums import JobStatus
from mufinder.services.cluster_security import invalid_remote_job_labels
from mufinder.services.scheduler import Scheduler, SchedulerSubmitRequest, create_scheduler
from mufinder.services.script_template import load_template_for_cluster, populate_template
from mufinder.services.ssh_client import SSHConnection


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


@dataclass(frozen=True, slots=True)
class QueueCheckSnapshot:
    """Summary emitted after one queue polling pass."""

    checked_at: datetime
    elapsed_by_job_id: dict[UUID, str]


class ClusterJobRunner(QObject):
    """Run CASTEP jobs on a remote scheduler over SSH."""

    job_started = Signal(object)
    job_finished = Signal(object, object)
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
        self._password_cache: str | None = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(120_000)
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

    def connect_to_cluster(self, config: ClusterConfig, password: str | None = None) -> None:
        """Connect to cluster and scheduler backend."""

        if self.is_connected and self._config == config:
            return

        resolved_password = password
        if (
            resolved_password is None
            and config.auth_method == "password"
            and self._config is not None
            and self._config.name.strip().casefold() == config.name.strip().casefold()
            and self._config.username.strip() == config.username.strip()
            and self._password_cache
        ):
            resolved_password = self._password_cache

        self.disconnect()
        ssh = SSHConnection(config, settings=self._settings)
        ssh.connect(password=resolved_password)

        self._config = config
        self._ssh = ssh
        self._scheduler = create_scheduler(
            config.queue_system,
            submit_command_path=config.submit_command_path,
            queue_check_command_path=config.queue_check_command_path,
            cancel_command_path=config.cancel_command_path,
        )
        self._password_cache = resolved_password if config.auth_method == "password" else None
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
        self._password_cache = None
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
        queue_name: str = "",
        comment: str = "",
        castep_executable: str = "",
    ) -> str:
        """Upload files, submit scheduler script, and start tracking the remote job."""

        ssh = self._require_ssh()
        scheduler = self._require_scheduler()

        local_dir = Path(local_job_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            raise ConnectionError(f"Local job directory does not exist: {local_dir}")

        remote_root = working_directory.strip()
        if not remote_root:
            raise ConnectionError("Remote working directory is required.")
        self._validate_seed_name(seed_name)

        remote_dir = posixpath.join(remote_root, seed_name)
        ssh.mkdir(remote_dir, exist_ok=True)

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
            template = load_template_for_cluster(
                template_script_path,
                cluster_name=cluster_name,
                queue_system=queue_system,
            )
            nodes = max(1, int(ceil(max(1, int(num_processes)) / max(1, int(cores_per_node)))))
            if queue == "slurm":
                command = (
                    f"srun -n {max(1, int(num_processes))} "
                    f"{shlex.quote(resolved_castep_executable)} {seed_name}"
                )
            else:
                command = (
                    f"mpirun -np {max(1, int(num_processes))} "
                    f"{shlex.quote(resolved_castep_executable)} {seed_name}"
                )
            script_content = populate_template(
                template,
                {
                    "job_name": seed_name,
                    "num_processes": str(max(1, int(num_processes))),
                    "num_nodes": str(nodes),
                    "cores_per_node": str(max(1, int(cores_per_node))),
                    "walltime": walltime,
                    "working_directory": remote_dir,
                    "seed_name": seed_name,
                    "castep_command": command,
                },
            )

            remote_script = posixpath.join(remote_dir, "run.sh")
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".sh", delete=False) as handle:
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

        self._tracked_jobs[job_id] = _TrackedClusterJob(
            job_id=job_id,
            remote_job_id=remote_job_id,
            remote_dir=remote_dir,
            seed_name=seed_name,
            local_output_dir=local_dir,
            cluster_name=self._config.name if self._config is not None else "",
            status=JobStatus.QUEUED,
            terminal=False,
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
    ) -> None:
        """Register a persisted cluster job for queue polling and retrieval."""

        normalized_remote_job_id = remote_job_id.strip()
        normalized_remote_dir = remote_dir.strip()
        if not normalized_remote_job_id or not normalized_remote_dir:
            return

        tracked = _TrackedClusterJob(
            job_id=job_id,
            remote_job_id=normalized_remote_job_id,
            remote_dir=normalized_remote_dir,
            seed_name=seed_name,
            local_output_dir=Path(local_output_dir),
            cluster_name=cluster_name.strip(),
            status=status,
            terminal=status not in {JobStatus.SUBMITTING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING},
        )
        self._tracked_jobs[job_id] = tracked
        if (
            self.is_connected
            and not tracked.terminal
            and self._tracked_job_matches_connected_cluster(tracked)
            and not self._poll_timer.isActive()
        ):
            self._poll_timer.start()

    def shutdown(self) -> None:
        """Best-effort shutdown during application close."""

        self.disconnect()

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

    def remove_remote_directory(self, remote_directory: str) -> None:
        """Remove one remote directory recursively."""

        ssh = self._require_ssh()
        ssh.rmtree(remote_directory)

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
            self.queue_checked.emit(QueueCheckSnapshot(checked_at=datetime.now(UTC), elapsed_by_job_id={}))
            return

        try:
            statuses = scheduler.status(ssh, [entry.remote_job_id for entry in active])
        except Exception as exc:  # noqa: BLE001
            cluster_name = self._config.name if self._config is not None else "the cluster"
            self.disconnect()
            self.poll_failed.emit(f"Queue polling failed for {cluster_name}: {exc}")
            return

        for entry in active:
            info = statuses.get(entry.remote_job_id)
            mapped = JobStatus.COMPLETED if info is None else scheduler.map_to_job_status(info.state)

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
                self.status_changed.emit(entry.job_id, JobStatus.STOPPED.value)
                continue
            if mapped == JobStatus.FAILED:
                entry.terminal = True
                self.job_failed.emit(entry.job_id, "Scheduler reported job failure.")
                self.status_changed.emit(entry.job_id, JobStatus.FAILED.value)
                continue
            if mapped == JobStatus.COMPLETED:
                self._retrieve_results(entry)

        elapsed_by_job_id = {
            entry.job_id: info.elapsed_runtime
            for entry in active
            if (info := statuses.get(entry.remote_job_id)) is not None and info.elapsed_runtime
        }
        self.queue_checked.emit(
            QueueCheckSnapshot(
                checked_at=datetime.now(UTC),
                elapsed_by_job_id=elapsed_by_job_id,
            )
        )

    def _retrieve_results(self, entry: _TrackedClusterJob) -> None:
        ssh = self._require_ssh()

        self.status_changed.emit(entry.job_id, JobStatus.RETRIEVING.value)
        remote_castep = posixpath.join(entry.remote_dir, f"{entry.seed_name}.castep")
        remote_out_cell = posixpath.join(entry.remote_dir, f"{entry.seed_name}-out.cell")
        local_castep = entry.local_output_dir / f"{entry.seed_name}.castep"
        local_out_cell = entry.local_output_dir / f"{entry.seed_name}-out.cell"

        try:
            ssh.get(remote_castep, local_castep)
            if ssh.exists(remote_out_cell):
                ssh.get(remote_out_cell, local_out_cell)
            parsed = parse_castep_output(local_castep)
        except Exception as exc:  # noqa: BLE001
            entry.terminal = True
            self.job_failed.emit(entry.job_id, f"Failed to retrieve or parse remote output: {exc}")
            self.status_changed.emit(entry.job_id, JobStatus.FAILED.value)
            return

        if not parsed.completed:
            entry.terminal = True
            if self._looks_like_walltime(parsed):
                self.status_changed.emit(entry.job_id, JobStatus.STOPPED.value)
            else:
                self.job_failed.emit(
                    entry.job_id,
                    parsed.error_message or "CASTEP did not report successful completion.",
                )
                self.status_changed.emit(entry.job_id, JobStatus.FAILED.value)
            return

        entry.terminal = True
        self.job_finished.emit(entry.job_id, parsed)
        self.status_changed.emit(entry.job_id, JobStatus.COMPLETED.value)

    @staticmethod
    def _looks_like_walltime(parsed: CastepResult) -> bool:
        if not parsed.error_message:
            return False
        text = parsed.error_message.lower()
        return "timeout" in text or "time limit" in text or "wall" in text

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
