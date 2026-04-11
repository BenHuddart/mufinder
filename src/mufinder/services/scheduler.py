"""Scheduler abstraction and SLURM implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
import shlex

from mufinder.domain.enums import JobStatus


@dataclass(frozen=True, slots=True)
class SchedulerJobInfo:
    """Scheduler-level status payload for one remote job."""

    job_id: str
    state: str
    elapsed_runtime: str = ""
    name: str = ""


@dataclass(frozen=True, slots=True)
class SchedulerSubmitRequest:
    """Scheduler submission payload passed from cluster runner."""

    working_dir: str
    seed_name: str
    num_processes: int
    cores_per_node: int
    walltime: str
    castep_executable: str
    script_path: str = ""
    memory_gb: float | None = None
    queue_name: str = ""
    comment: str = ""


class Scheduler(ABC):
    """Abstract scheduler contract used by cluster runner backends."""

    @abstractmethod
    def submit(self, ssh, request: SchedulerSubmitRequest) -> str:
        """Submit one script and return the scheduler job id."""

    @abstractmethod
    def cancel(self, ssh, job_id: str) -> None:
        """Cancel one scheduler job id."""

    @abstractmethod
    def status(self, ssh, job_ids: list[str]) -> dict[str, SchedulerJobInfo]:
        """Return current status for requested scheduler job ids."""

    @abstractmethod
    def is_terminal(self, state: str) -> bool:
        """Return true if scheduler state is terminal."""

    @abstractmethod
    def map_to_job_status(self, state: str) -> JobStatus:
        """Map scheduler state to domain job status."""


class SlurmScheduler(Scheduler):
    """SLURM scheduler implementation via shell commands over SSH."""

    _STATE_ALIAS: dict[str, str] = {
        "PENDING": "PD",
        "RUNNING": "R",
        "COMPLETING": "CG",
        "COMPLETED": "CD",
        "FAILED": "F",
        "CANCELLED": "CA",
        "TIMEOUT": "TO",
        "NODE_FAIL": "NF",
        "OUT_OF_MEMORY": "OOM",
    }

    def __init__(
        self,
        *,
        submit_command_path: str = "",
        queue_check_command_path: str = "",
        cancel_command_path: str = "",
    ) -> None:
        self._submit_command_path = submit_command_path.strip()
        self._queue_check_command_path = queue_check_command_path.strip()
        self._cancel_command_path = cancel_command_path.strip()
        self._resolved_commands: dict[str, str] = {}

    def submit(self, ssh, request: SchedulerSubmitRequest) -> str:
        script_path = request.script_path.strip()
        if not script_path:
            raise RuntimeError("SLURM submission requires a script path.")

        working_dir = request.working_dir.strip()
        submit_command = self._resolve_command(ssh, "submit", "sbatch", self._submit_command_path)
        command = (
            f"cd {shlex.quote(working_dir)} && "
            f"{shlex.quote(submit_command)} {shlex.quote(script_path)}"
        )
        exit_code, stdout, stderr = ssh.exec_command(command, timeout=60)
        if exit_code != 0:
            details = stderr.strip() or stdout.strip() or "sbatch failed"
            raise RuntimeError(details)

        return _parse_submitted_job_id(stdout.strip(), "sbatch")

    def cancel(self, ssh, job_id: str) -> None:
        cancel_command = self._resolve_command(ssh, "cancel", "scancel", self._cancel_command_path)
        command = f"{shlex.quote(cancel_command)} {shlex.quote(job_id)}"
        exit_code, _stdout, stderr = ssh.exec_command(command, timeout=30)
        if exit_code != 0:
            details = stderr.strip() or "scancel failed"
            raise RuntimeError(details)

    def status(self, ssh, job_ids: list[str]) -> dict[str, SchedulerJobInfo]:
        if not job_ids:
            return {}

        ids_arg = ",".join(job_ids)
        queue_command = self._resolve_command(ssh, "status", "squeue", self._queue_check_command_path)
        command = (
            f"{shlex.quote(queue_command)} "
            f"--jobs={shlex.quote(ids_arg)} "
            '--noheader --format="%i|%t|%M|%j"'
        )
        exit_code, stdout, stderr = ssh.exec_command(command, timeout=60)
        if exit_code != 0:
            details = stderr.strip() or "squeue failed"
            if self._is_invalid_job_id_error(details):
                return self._status_with_invalid_id_fallback(ssh, queue_command, job_ids)
            raise RuntimeError(details)

        return self._parse_status_output(stdout)

    def _status_with_invalid_id_fallback(
        self,
        ssh,
        queue_command: str,
        job_ids: list[str],
    ) -> dict[str, SchedulerJobInfo]:
        """Fallback for mixed valid/invalid job-id queries.

        SLURM can fail a combined ``--jobs=id1,id2`` request if any id is stale.
        Querying per-id lets us keep valid statuses while treating invalid ids as missing.
        """
        results: dict[str, SchedulerJobInfo] = {}
        for job_id in job_ids:
            command = (
                f"{shlex.quote(queue_command)} "
                f"--jobs={shlex.quote(job_id)} "
                '--noheader --format="%i|%t|%M|%j"'
            )
            exit_code, stdout, stderr = ssh.exec_command(command, timeout=60)
            if exit_code != 0:
                details = stderr.strip() or stdout.strip() or "squeue failed"
                if self._is_invalid_job_id_error(details):
                    continue
                raise RuntimeError(details)
            results.update(self._parse_status_output(stdout))
        return results

    @staticmethod
    def _is_invalid_job_id_error(details: str) -> bool:
        normalized = details.casefold()
        return "invalid job id" in normalized or "invalid jobid" in normalized

    @staticmethod
    def _parse_status_output(stdout: str) -> dict[str, SchedulerJobInfo]:
        results: dict[str, SchedulerJobInfo] = {}
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|", maxsplit=3)
            if len(parts) < 2:
                continue
            if len(parts) == 2:
                job_id, state = parts
                elapsed_runtime = ""
                name = ""
            elif len(parts) == 3:
                job_id, state, name = parts
                elapsed_runtime = ""
            else:
                job_id, state, elapsed_runtime, name = parts
            results[job_id] = SchedulerJobInfo(
                job_id=job_id,
                state=state,
                elapsed_runtime=elapsed_runtime,
                name=name,
            )
        return results

    def is_terminal(self, state: str) -> bool:
        normalized = self._normalise_state(state)
        return normalized in {"CD", "F", "CA", "TO", "NF", "OOM"}

    def map_to_job_status(self, state: str) -> JobStatus:
        normalized = self._normalise_state(state)
        if normalized in {"PD"}:
            return JobStatus.QUEUED
        if normalized in {"R", "CG"}:
            return JobStatus.RUNNING
        if normalized in {"CD"}:
            return JobStatus.COMPLETED
        if normalized in {"CA"}:
            return JobStatus.CANCELLED
        if normalized in {"TO"}:
            return JobStatus.STOPPED
        if normalized in {"F", "NF", "OOM"}:
            return JobStatus.FAILED
        return JobStatus.RUNNING

    def _normalise_state(self, state: str) -> str:
        key = state.strip().upper()
        if not key:
            return ""
        return self._STATE_ALIAS.get(key, key)

    def _resolve_command(self, ssh, cache_key: str, command_name: str, configured: str) -> str:
        if cache_key in self._resolved_commands:
            return self._resolved_commands[cache_key]

        if hasattr(ssh, "resolve_command_path"):
            resolved = ssh.resolve_command_path(command_name, configured)
            self._resolved_commands[cache_key] = resolved
            return resolved

        if configured:
            self._resolved_commands[cache_key] = configured
            return configured

        probe = f"command -v {shlex.quote(command_name)}"
        exit_code, stdout, _stderr = ssh.exec_command(probe, timeout=10)
        if exit_code == 0:
            candidate = stdout.strip().splitlines()[0].strip() if stdout.strip() else ""
            if candidate:
                self._resolved_commands[cache_key] = candidate
                return candidate

        self._resolved_commands[cache_key] = command_name
        return command_name


class AddqueueScheduler(SlurmScheduler):
    """addqueue submitter with SLURM-compatible polling/cancel."""

    def submit(self, ssh, request: SchedulerSubmitRequest) -> str:
        submit_command = self._resolve_command(ssh, "submit", "addqueue", self._submit_command_path)
        args: list[str] = [submit_command, "--sbatch", "-n", str(max(1, int(request.num_processes)))]

        queue_name = request.queue_name.strip()
        if queue_name:
            args.extend(["-q", queue_name])

        if request.memory_gb is not None:
            args.extend(["-m", format(float(request.memory_gb), "g")])

        comment = request.comment.strip()
        if comment:
            args.extend(["-c", comment])

        walltime = request.walltime.strip()
        if walltime:
            args.extend(["-t", walltime])

        args.extend([request.castep_executable.strip() or "castep.mpi", request.seed_name.strip()])
        command = " ".join(shlex.quote(part) for part in args)
        wrapped = f"cd {shlex.quote(request.working_dir.strip())} && {command}"

        exit_code, stdout, stderr = ssh.exec_command(wrapped, timeout=60)
        if exit_code != 0:
            details = stderr.strip() or stdout.strip() or "addqueue failed"
            raise RuntimeError(details)

        return _parse_submitted_job_id(stdout.strip(), "addqueue")


def _parse_submitted_job_id(output: str, source: str) -> str:
    for line in output.splitlines()[::-1]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("submitted batch job"):
            parts = stripped.split()
            if parts and parts[-1].isdigit():
                return parts[-1]

    for line in output.splitlines()[::-1]:
        match = re.search(r"\b(\d{2,})\b", line)
        if match:
            return match.group(1)

    raise RuntimeError(f"Could not parse job id from {source} output: {output}")


def create_scheduler(
    queue_system: str,
    *,
    submit_command_path: str = "",
    queue_check_command_path: str = "",
    cancel_command_path: str = "",
) -> Scheduler:
    """Create a scheduler backend for a configured queue system."""

    normalized = queue_system.strip().lower()
    if normalized == "slurm":
        return SlurmScheduler(
            submit_command_path=submit_command_path,
            queue_check_command_path=queue_check_command_path,
            cancel_command_path=cancel_command_path,
        )
    if normalized == "addqueue":
        return AddqueueScheduler(
            submit_command_path=submit_command_path,
            queue_check_command_path=queue_check_command_path,
            cancel_command_path=cancel_command_path,
        )
    raise ValueError(f"Unsupported queue system: {queue_system}")
