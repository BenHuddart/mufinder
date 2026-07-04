"""Local CASTEP execution service built around QProcess."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque
from uuid import UUID

from PySide6.QtCore import QObject, QProcess, Signal

from mufinder.castep_io.output_parser import parse_castep_output
from mufinder.domain.enums import JobStatus
from mufinder.services.job_artifacts import collect_job_artifacts


@dataclass(slots=True)
class _PendingJob:
    job_id: UUID
    job_dir: Path
    seed_name: str
    castep_executable: str
    mpirun_path: str
    num_cores: int
    task_farming_enabled: bool = False
    num_farms: int | None = None
    # Which secondary outputs this job produces, mirroring the cluster runner's
    # retrieval flags. Drive off-thread artifact collection on completion.
    retrieve_out_cell: bool = False
    retrieve_phonon: bool = False
    retrieve_ts: bool = False


@dataclass(slots=True)
class _ActiveJob:
    request: _PendingJob
    process: QProcess
    process_error: str | None = None


class LocalJobRunner(QObject):
    """Run local CASTEP calculations without blocking the GUI thread."""

    job_started = Signal(object)
    #: ``(job_id, CastepResult, ParsedJobArtifacts)``. The third payload carries
    #: the secondary outputs (relaxed cell, phonon, transition state) parsed on
    #: the completion callback so the GUI handler never re-parses them.
    job_finished = Signal(object, object, object)
    job_failed = Signal(object, str)
    job_cancelled = Signal(object)
    status_changed = Signal(object, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._max_concurrent = 1
        self._pending: Deque[_PendingJob] = deque()
        self._active: dict[UUID, _ActiveJob] = {}
        self._cancel_requested: set[UUID] = set()

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @max_concurrent.setter
    def max_concurrent(self, value: int) -> None:
        self._max_concurrent = max(1, int(value))
        self._start_queued_jobs()

    def running_count(self) -> int:
        return len(self._active)

    def queued_count(self) -> int:
        return len(self._pending)

    def is_running(self, job_id: UUID) -> bool:
        return job_id in self._active

    def active_job_ids(self) -> frozenset[UUID]:
        """Return the ids of jobs with a live ``QProcess`` (running)."""

        return frozenset(self._active)

    def pending_job_ids(self) -> frozenset[UUID]:
        """Return the ids of jobs queued but not yet started."""

        return frozenset(request.job_id for request in self._pending)

    def submit(
        self,
        *,
        job_id: UUID,
        job_dir: Path,
        seed_name: str,
        castep_executable: str,
        mpirun_path: str,
        num_cores: int,
        task_farming_enabled: bool = False,
        num_farms: int | None = None,
        retrieve_out_cell: bool = False,
        retrieve_phonon: bool = False,
        retrieve_ts: bool = False,
    ) -> None:
        """Submit one local CASTEP job for execution."""

        request = _PendingJob(
            job_id=job_id,
            job_dir=Path(job_dir),
            seed_name=seed_name,
            castep_executable=str(castep_executable).strip(),
            mpirun_path=str(mpirun_path).strip(),
            num_cores=max(1, int(num_cores)),
            task_farming_enabled=bool(task_farming_enabled),
            num_farms=(max(1, int(num_farms)) if num_farms is not None else None),
            retrieve_out_cell=bool(retrieve_out_cell),
            retrieve_phonon=bool(retrieve_phonon),
            retrieve_ts=bool(retrieve_ts),
        )

        validation_error = self._validate_request(request)
        if validation_error is not None:
            self.job_failed.emit(job_id, validation_error)
            self.status_changed.emit(job_id, JobStatus.FAILED.value)
            return

        self._pending.append(request)
        if self.running_count() >= self._max_concurrent:
            self.status_changed.emit(job_id, JobStatus.QUEUED.value)
        self._start_queued_jobs()

    def cancel(self, job_id: UUID) -> None:
        """Cancel a queued or running job."""

        pending = self._remove_pending_job(job_id)
        if pending is not None:
            self.job_cancelled.emit(job_id)
            self.status_changed.emit(job_id, JobStatus.CANCELLED.value)
            return

        active = self._active.get(job_id)
        if active is None:
            return

        self._cancel_requested.add(job_id)
        active.process.kill()

    def cancel_all(self) -> None:
        """Cancel all queued and active jobs."""

        while self._pending:
            request = self._pending.popleft()
            self.job_cancelled.emit(request.job_id)
            self.status_changed.emit(request.job_id, JobStatus.CANCELLED.value)

        for job_id in list(self._active):
            self.cancel(job_id)

    def shutdown(self) -> None:
        """Best-effort shutdown used during application teardown."""

        self.cancel_all()
        for active in list(self._active.values()):
            active.process.kill()
        self._active.clear()
        self._cancel_requested.clear()

    def _remove_pending_job(self, job_id: UUID) -> _PendingJob | None:
        for request in list(self._pending):
            if request.job_id != job_id:
                continue
            self._pending.remove(request)
            return request
        return None

    def _validate_request(self, request: _PendingJob) -> str | None:
        if not request.castep_executable:
            return "No CASTEP executable is configured."
        if request.num_cores > 1 and not request.mpirun_path:
            return "Configure mpirun in Setup -> CASTEP before requesting multiple cores."
        if not request.job_dir.exists() or not request.job_dir.is_dir():
            return f"Job directory does not exist: {request.job_dir}"
        return None

    def _start_queued_jobs(self) -> None:
        while self._pending and self.running_count() < self._max_concurrent:
            request = self._pending.popleft()
            self._launch_process(request)

    def _launch_process(self, request: _PendingJob) -> None:
        process = QProcess(self)
        process.setWorkingDirectory(str(request.job_dir))

        if request.num_cores > 1:
            process.setProgram(request.mpirun_path)
            process.setArguments(
                ["-np", str(request.num_cores), request.castep_executable, request.seed_name]
            )
        else:
            process.setProgram(request.castep_executable)
            process.setArguments([request.seed_name])

        process.started.connect(lambda request=request: self._on_process_started(request.job_id))
        process.errorOccurred.connect(
            lambda error, request=request: self._on_process_error(request.job_id, error)
        )
        process.finished.connect(
            lambda exit_code, _exit_status, request=request: self._on_process_finished(
                request.job_id,
                exit_code,
            )
        )

        self._active[request.job_id] = _ActiveJob(request=request, process=process)
        process.start()

    def _on_process_started(self, job_id: UUID) -> None:
        self.job_started.emit(job_id)
        self.status_changed.emit(job_id, JobStatus.RUNNING.value)

    def _on_process_error(self, job_id: UUID, process_error: QProcess.ProcessError) -> None:
        active = self._active.get(job_id)
        if active is None:
            return
        active.process_error = str(process_error)

        # FailedToStart is fatal and Qt may never deliver a subsequent finished
        # event, which would strand the job in ``submitting`` forever. Report the
        # failure immediately and clean up the active entry exactly as the finish
        # path does, so a late finished event for this job is a no-op (its entry
        # is already gone). Other ProcessError values keep the store-and-wait
        # behaviour and are reported by the finish handler.
        if process_error != QProcess.ProcessError.FailedToStart:
            return

        self._active.pop(job_id, None)
        self._cancel_requested.discard(job_id)
        detail = active.process.errorString().strip() or process_error.name
        active.process.deleteLater()
        message = (
            f"The CASTEP process failed to start: {detail}. "
            "Check the CASTEP executable path and file permissions."
        )
        self.job_failed.emit(job_id, message)
        self.status_changed.emit(job_id, JobStatus.FAILED.value)
        self._start_queued_jobs()

    def _on_process_finished(self, job_id: UUID, exit_code: int) -> None:
        active = self._active.pop(job_id, None)
        if active is None:
            self._start_queued_jobs()
            return

        request = active.request
        process = active.process
        process.deleteLater()

        if job_id in self._cancel_requested:
            self._cancel_requested.remove(job_id)
            self.job_cancelled.emit(job_id)
            self.status_changed.emit(job_id, JobStatus.CANCELLED.value)
            self._start_queued_jobs()
            return

        if exit_code != 0:
            stderr = bytes(process.readAllStandardError()).decode("utf-8", errors="replace").strip()
            details = stderr or active.process_error or f"Process exited with code {exit_code}."
            self.job_failed.emit(job_id, details)
            self.status_changed.emit(job_id, JobStatus.FAILED.value)
            self._start_queued_jobs()
            return

        output_path = self._primary_output_path(request)
        if not output_path.exists():
            self.job_failed.emit(job_id, f"Expected CASTEP output was not found: {output_path.name}")
            self.status_changed.emit(job_id, JobStatus.FAILED.value)
            self._start_queued_jobs()
            return

        try:
            result = parse_castep_output(output_path)
        except Exception as exc:  # noqa: BLE001
            self.job_failed.emit(job_id, f"Failed to parse CASTEP output: {exc}")
            self.status_changed.emit(job_id, JobStatus.FAILED.value)
            self._start_queued_jobs()
            return

        if not result.completed:
            self.job_failed.emit(
                job_id,
                result.error_message or "CASTEP did not report successful completion.",
            )
            self.status_changed.emit(job_id, JobStatus.FAILED.value)
            self._start_queued_jobs()
            return

        # Parse the secondary outputs (relaxed cell / phonon / transition state)
        # here on the completion callback, exactly once, driven by the retrieval
        # flags this job carries. The GUI handler decides failure from the
        # carrier and the owning controller consumes the pre-parsed data —
        # neither re-reads disk. Errors are captured on the carrier, never raised.
        artifacts = collect_job_artifacts(
            request.job_dir,
            request.seed_name,
            want_out_cell=request.retrieve_out_cell,
            want_phonon=request.retrieve_phonon,
            want_ts=request.retrieve_ts,
        )
        self.job_finished.emit(job_id, result, artifacts)
        self.status_changed.emit(job_id, JobStatus.COMPLETED.value)
        self._start_queued_jobs()

    @staticmethod
    def _primary_output_path(request: _PendingJob) -> Path:
        if request.task_farming_enabled and (request.num_farms or 1) > 1:
            farmed = request.job_dir / f"{request.seed_name}_farm001.castep"
            if farmed.exists():
                return farmed
        return request.job_dir / f"{request.seed_name}.castep"
