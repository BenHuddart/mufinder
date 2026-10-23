"""Tests for local CASTEP execution runner service."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QProcess

import mufinder.services.local_runner as local_runner_module
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus
from mufinder.services.job_artifacts import RetrievalFlags
from mufinder.services.local_runner import LocalJobRunner


class _DummyProcess:
    def __init__(self, error_string: str = "") -> None:
        self.killed = False
        self.stderr = b""
        self.deleted = False
        self._error_string = error_string

    def kill(self) -> None:
        self.killed = True

    def deleteLater(self) -> None:
        self.deleted = True

    def readAllStandardError(self) -> bytes:
        return self.stderr

    def errorString(self) -> str:
        return self._error_string


class _FakeSignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs) -> None:
        for callback in self._callbacks:
            callback(*args, **kwargs)


class _FakeQProcess:
    def __init__(self, _parent=None) -> None:
        self.working_dir = ""
        self.program = ""
        self.arguments: list[str] = []
        self.started = _FakeSignal()
        self.errorOccurred = _FakeSignal()
        self.finished = _FakeSignal()
        self.killed = False
        self.stderr = b""

    def setWorkingDirectory(self, path: str) -> None:
        self.working_dir = path

    def setProgram(self, program: str) -> None:
        self.program = program

    def setArguments(self, arguments: list[str]) -> None:
        self.arguments = list(arguments)

    def start(self) -> None:
        return

    def kill(self) -> None:
        self.killed = True

    def deleteLater(self) -> None:
        return

    def readAllStandardError(self) -> bytes:
        return self.stderr


def test_local_runner_rejects_missing_executable(qapp, tmp_path: Path) -> None:
    runner = LocalJobRunner()
    failures: list[tuple[object, str]] = []
    statuses: list[tuple[object, str]] = []

    runner.job_failed.connect(lambda job_id, message: failures.append((job_id, message)))
    runner.status_changed.connect(lambda job_id, status: statuses.append((job_id, status)))

    job_id = uuid4()
    runner.submit(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name="job_a",
        castep_executable="",
        mpirun_path="",
        num_cores=1,
    )

    assert len(failures) == 1
    assert failures[0][0] == job_id
    assert "executable" in failures[0][1].lower()
    assert statuses == [(job_id, JobStatus.FAILED.value)]


def test_local_runner_queues_and_cancels_pending_jobs(qapp, monkeypatch, tmp_path: Path) -> None:
    runner = LocalJobRunner()
    runner.max_concurrent = 1

    started: list[object] = []

    def fake_launch(request):
        started.append(request.job_id)
        runner._active[request.job_id] = local_runner_module._ActiveJob(
            request=request,
            process=_DummyProcess(),
        )

    monkeypatch.setattr(runner, "_launch_process", fake_launch)

    status_events: list[tuple[object, str]] = []
    cancel_events: list[object] = []
    runner.status_changed.connect(lambda job_id, status: status_events.append((job_id, status)))
    runner.job_cancelled.connect(lambda job_id: cancel_events.append(job_id))

    first_job = uuid4()
    second_job = uuid4()

    runner.submit(
        job_id=first_job,
        job_dir=tmp_path,
        seed_name="job_1",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    runner.submit(
        job_id=second_job,
        job_dir=tmp_path,
        seed_name="job_2",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )

    assert started == [first_job]
    assert runner.running_count() == 1
    assert len(runner.pending_job_ids()) == 1
    assert (second_job, JobStatus.QUEUED.value) in status_events

    runner.cancel(second_job)

    assert len(runner.pending_job_ids()) == 0
    assert second_job in cancel_events
    assert (second_job, JobStatus.CANCELLED.value) in status_events


def test_local_runner_launch_uses_mpirun_for_multicore(qapp, monkeypatch, tmp_path: Path) -> None:
    runner = LocalJobRunner()
    fake_process = _FakeQProcess()
    monkeypatch.setattr(local_runner_module, "QProcess", lambda _parent: fake_process)

    job_id = uuid4()
    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name="job_multi",
        castep_executable="/usr/bin/castep",
        mpirun_path="/usr/bin/mpirun",
        num_cores=4,
    )

    statuses: list[tuple[object, object]] = []
    runner.status_changed.connect(lambda jid, status: statuses.append((jid, status)))

    runner._launch_process(request)
    fake_process.started.emit()

    assert fake_process.working_dir == str(tmp_path)
    assert fake_process.program == "/usr/bin/mpirun"
    assert fake_process.arguments == ["-np", "4", "/usr/bin/castep", "job_multi"]
    assert statuses[-1] == (job_id, JobStatus.RUNNING.value)


def test_local_runner_launch_uses_executable_for_single_core(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    runner = LocalJobRunner()
    fake_process = _FakeQProcess()
    monkeypatch.setattr(local_runner_module, "QProcess", lambda _parent: fake_process)

    job_id = uuid4()
    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name="job_single",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )

    runner._launch_process(request)

    assert fake_process.program == "/usr/bin/castep"
    assert fake_process.arguments == ["job_single"]


def test_local_runner_cancel_running_process(qapp, tmp_path: Path) -> None:
    runner = LocalJobRunner()
    job_id = uuid4()
    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name="job_running",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    process = _DummyProcess()
    runner._active[job_id] = local_runner_module._ActiveJob(request=request, process=process)

    cancelled: list[object] = []
    statuses: list[tuple[object, object]] = []
    runner.job_cancelled.connect(lambda jid: cancelled.append(jid))
    runner.status_changed.connect(lambda jid, status: statuses.append((jid, status)))

    runner.cancel(job_id)
    runner._on_process_finished(job_id, 0)

    assert process.killed is True
    assert cancelled == [job_id]
    assert statuses[-1] == (job_id, JobStatus.CANCELLED.value)


def test_local_runner_process_finished_nonzero_exit_reports_stderr(qapp, tmp_path: Path) -> None:
    runner = LocalJobRunner()
    job_id = uuid4()
    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name="job_fail",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    process = _DummyProcess()
    process.stderr = b"fatal error"
    runner._active[job_id] = local_runner_module._ActiveJob(request=request, process=process)

    failures: list[str] = []
    statuses: list[tuple[object, object]] = []
    runner.job_failed.connect(lambda _jid, msg: failures.append(msg))
    runner.status_changed.connect(lambda jid, status: statuses.append((jid, status)))

    runner._on_process_finished(job_id, 2)

    assert failures == ["fatal error"]
    assert statuses[-1] == (job_id, JobStatus.FAILED.value)


def test_local_runner_process_finished_handles_missing_output(qapp, tmp_path: Path) -> None:
    runner = LocalJobRunner()
    job_id = uuid4()
    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name="no_output",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    process = _DummyProcess()
    runner._active[job_id] = local_runner_module._ActiveJob(request=request, process=process)

    failures: list[str] = []
    runner.job_failed.connect(lambda _jid, msg: failures.append(msg))

    runner._on_process_finished(job_id, 0)

    assert "Expected CASTEP output was not found" in failures[0]


def test_local_runner_process_finished_handles_parse_error(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    runner = LocalJobRunner()
    job_id = uuid4()
    seed = "parse_error"
    output_file = tmp_path / f"{seed}.castep"
    output_file.write_text("bad", encoding="utf-8")

    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name=seed,
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    runner._active[job_id] = local_runner_module._ActiveJob(
        request=request, process=_DummyProcess()
    )

    monkeypatch.setattr(
        local_runner_module,
        "parse_castep_output",
        lambda _path: (_ for _ in ()).throw(ValueError("boom")),
    )

    failures: list[str] = []
    runner.job_failed.connect(lambda _jid, msg: failures.append(msg))
    runner._on_process_finished(job_id, 0)

    assert "Failed to parse CASTEP output" in failures[0]


def test_local_runner_process_finished_handles_incomplete_result(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    runner = LocalJobRunner()
    job_id = uuid4()
    seed = "incomplete"
    output_file = tmp_path / f"{seed}.castep"
    output_file.write_text("text", encoding="utf-8")

    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name=seed,
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    runner._active[job_id] = local_runner_module._ActiveJob(
        request=request, process=_DummyProcess()
    )

    monkeypatch.setattr(
        local_runner_module,
        "parse_castep_output",
        lambda _path: CastepResult(completed=False, error_message="did not converge"),
    )

    failures: list[str] = []
    statuses: list[tuple[object, object]] = []
    runner.job_failed.connect(lambda _jid, msg: failures.append(msg))
    runner.status_changed.connect(lambda jid, status: statuses.append((jid, status)))

    runner._on_process_finished(job_id, 0)

    assert failures == ["did not converge"]
    assert statuses[-1] == (job_id, JobStatus.FAILED.value)


def test_local_runner_process_finished_emits_success(qapp, monkeypatch, tmp_path: Path) -> None:
    runner = LocalJobRunner()
    job_id = uuid4()
    seed = "success"
    output_file = tmp_path / f"{seed}.castep"
    output_file.write_text("ok", encoding="utf-8")

    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name=seed,
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    runner._active[job_id] = local_runner_module._ActiveJob(
        request=request, process=_DummyProcess()
    )

    parsed = CastepResult(completed=True, total_energy=-1.23)
    monkeypatch.setattr(local_runner_module, "parse_castep_output", lambda _path: parsed)

    finished: list[tuple[object, object]] = []
    statuses: list[tuple[object, object]] = []
    runner.job_finished.connect(lambda jid, result: finished.append((jid, result)))
    runner.status_changed.connect(lambda jid, status: statuses.append((jid, status)))

    runner._on_process_finished(job_id, 0)

    assert finished == [(job_id, parsed)]
    assert statuses[-1] == (job_id, JobStatus.COMPLETED.value)


def test_local_runner_process_finished_uses_task_farm_primary_output(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    runner = LocalJobRunner()
    job_id = uuid4()
    seed = "farm_success"
    (tmp_path / f"{seed}_farm001.castep").write_text("ok", encoding="utf-8")

    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name=seed,
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
        retrieval=RetrievalFlags(task_farming_enabled=True, num_farms=4),
    )
    runner._active[job_id] = local_runner_module._ActiveJob(
        request=request, process=_DummyProcess()
    )

    seen_paths: list[Path] = []
    parsed = CastepResult(completed=True, total_energy=-1.23)
    monkeypatch.setattr(
        local_runner_module,
        "parse_castep_output",
        lambda path: (seen_paths.append(Path(path)), parsed)[1],
    )

    runner._on_process_finished(job_id, 0)

    assert seen_paths == [tmp_path / f"{seed}_farm001.castep"]


def test_local_runner_cancel_all_cancels_pending_and_running(qapp, tmp_path: Path) -> None:
    runner = LocalJobRunner()

    pending_id = uuid4()
    running_id = uuid4()

    runner._pending.append(
        local_runner_module._PendingJob(
            job_id=pending_id,
            job_dir=tmp_path,
            seed_name="pending",
            castep_executable="/usr/bin/castep",
            mpirun_path="",
            num_cores=1,
        )
    )
    process = _DummyProcess()
    runner._active[running_id] = local_runner_module._ActiveJob(
        request=local_runner_module._PendingJob(
            job_id=running_id,
            job_dir=tmp_path,
            seed_name="running",
            castep_executable="/usr/bin/castep",
            mpirun_path="",
            num_cores=1,
        ),
        process=process,
    )

    cancelled: list[object] = []
    runner.job_cancelled.connect(lambda jid: cancelled.append(jid))

    runner.cancel_all()

    assert pending_id in cancelled
    assert process.killed is True


def test_local_runner_misc_small_branches(qapp, monkeypatch, tmp_path: Path) -> None:
    runner = LocalJobRunner()

    # Property and helper accessors.
    runner.max_concurrent = 3
    assert runner.max_concurrent == 3

    unknown_id = uuid4()
    assert unknown_id not in runner.active_job_ids()

    # Cancel is a no-op for unknown IDs.
    runner.cancel(unknown_id)

    # Validation branches.
    missing_dir_request = local_runner_module._PendingJob(
        job_id=uuid4(),
        job_dir=tmp_path / "missing",
        seed_name="seed",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    assert "does not exist" in (runner._validate_request(missing_dir_request) or "")

    multicore_request = local_runner_module._PendingJob(
        job_id=uuid4(),
        job_dir=tmp_path,
        seed_name="seed",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=2,
    )
    assert "mpirun" in (runner._validate_request(multicore_request) or "")

    # _remove_pending_job loop continue branch.
    first = local_runner_module._PendingJob(
        job_id=uuid4(),
        job_dir=tmp_path,
        seed_name="first",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    second = local_runner_module._PendingJob(
        job_id=uuid4(),
        job_dir=tmp_path,
        seed_name="second",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    runner._pending.append(first)
    runner._pending.append(second)
    removed = runner._remove_pending_job(second.job_id)
    assert removed is second

    # Process-error and active-none finish branches.
    launched: list[object] = []
    monkeypatch.setattr(runner, "_launch_process", lambda request: launched.append(request))
    runner._on_process_error(uuid4(), "NoProcess")
    runner._on_process_finished(uuid4(), 0)

    assert len(launched) == 1


def test_local_runner_process_error_updates_active_entry(qapp, tmp_path: Path) -> None:
    runner = LocalJobRunner()
    job_id = uuid4()
    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name="seed",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    active = local_runner_module._ActiveJob(request=request, process=_DummyProcess())
    runner._active[job_id] = active

    runner._on_process_error(job_id, "FailedToStart")

    assert active.process_error == "FailedToStart"


def test_local_runner_shutdown_clears_state(qapp, tmp_path: Path) -> None:
    runner = LocalJobRunner()
    job_id = uuid4()
    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name="seed",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    process = _DummyProcess()
    runner._active[job_id] = local_runner_module._ActiveJob(request=request, process=process)
    runner._pending.append(request)

    runner.shutdown()

    assert runner.running_count() == 0
    assert len(runner.pending_job_ids()) == 0
    assert process.killed is True


def _active_failed_to_start_job(runner: LocalJobRunner, tmp_path: Path, *, error_string: str):
    job_id = uuid4()
    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name="seed",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    process = _DummyProcess(error_string=error_string)
    runner._active[job_id] = local_runner_module._ActiveJob(request=request, process=process)
    return job_id, process


def test_local_runner_failed_to_start_fails_fast(qapp, monkeypatch, tmp_path: Path) -> None:
    # REC #6: FailedToStart reports failure immediately (no finished event),
    # cleans up the active entry, and starts the next pending job.
    runner = LocalJobRunner()
    job_id, process = _active_failed_to_start_job(
        runner, tmp_path, error_string="No such file or directory"
    )

    pending_id = uuid4()
    runner._pending.append(
        local_runner_module._PendingJob(
            job_id=pending_id,
            job_dir=tmp_path,
            seed_name="next",
            castep_executable="/usr/bin/castep",
            mpirun_path="",
            num_cores=1,
        )
    )
    launched: list[object] = []
    monkeypatch.setattr(runner, "_launch_process", lambda request: launched.append(request.job_id))

    failures: list[tuple[object, str]] = []
    statuses: list[tuple[object, str]] = []
    runner.job_failed.connect(lambda jid, msg: failures.append((jid, msg)))
    runner.status_changed.connect(lambda jid, status: statuses.append((jid, status)))

    runner._on_process_error(job_id, QProcess.ProcessError.FailedToStart)

    assert failures == [
        (
            job_id,
            "The CASTEP process failed to start: No such file or directory. "
            "Check the CASTEP executable path and file permissions.",
        )
    ]
    assert statuses == [(job_id, JobStatus.FAILED.value)]
    assert job_id not in runner._active
    assert process.deleted is True
    assert launched == [pending_id]


def test_local_runner_failed_to_start_uses_enum_name_when_no_detail(qapp, tmp_path: Path) -> None:
    # REC #6: when errorString() is empty the message falls back to the enum name.
    runner = LocalJobRunner()
    job_id, _process = _active_failed_to_start_job(runner, tmp_path, error_string="")

    failures: list[tuple[object, str]] = []
    runner.job_failed.connect(lambda jid, msg: failures.append((jid, msg)))

    runner._on_process_error(job_id, QProcess.ProcessError.FailedToStart)

    assert failures[0][1] == (
        "The CASTEP process failed to start: FailedToStart. "
        "Check the CASTEP executable path and file permissions."
    )


def test_local_runner_failed_to_start_then_finished_is_noop(qapp, tmp_path: Path) -> None:
    # REC #6: a finished event arriving after the fail-fast report must not
    # re-emit failure — the active entry is already gone.
    runner = LocalJobRunner()
    job_id, _process = _active_failed_to_start_job(runner, tmp_path, error_string="boom")

    failures: list[object] = []
    statuses: list[tuple[object, str]] = []
    runner.job_failed.connect(lambda jid, _msg: failures.append(jid))
    runner.status_changed.connect(lambda jid, status: statuses.append((jid, status)))

    runner._on_process_error(job_id, QProcess.ProcessError.FailedToStart)
    runner._on_process_finished(job_id, 1)

    assert failures == [job_id]
    assert statuses == [(job_id, JobStatus.FAILED.value)]


def test_local_runner_non_fatal_process_error_stores_and_waits(qapp, tmp_path: Path) -> None:
    # REC #6: only FailedToStart is fail-fast; other errors keep the
    # store-and-wait behaviour so the finish handler reports the outcome.
    runner = LocalJobRunner()
    job_id = uuid4()
    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name="seed",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    active = local_runner_module._ActiveJob(request=request, process=_DummyProcess())
    runner._active[job_id] = active

    failures: list[object] = []
    runner.job_failed.connect(lambda jid, _msg: failures.append(jid))

    runner._on_process_error(job_id, QProcess.ProcessError.Crashed)

    # No fail-fast: entry retained, error stored, nothing emitted yet.
    assert failures == []
    assert job_id in runner._active
    assert active.process_error == str(QProcess.ProcessError.Crashed)


def test_local_runner_active_and_pending_id_accessors(qapp, tmp_path: Path) -> None:
    # REC #7: watchdog introspection accessors reflect live runner state.
    runner = LocalJobRunner()
    assert runner.active_job_ids() == frozenset()
    assert runner.pending_job_ids() == frozenset()

    active_id = uuid4()
    pending_id = uuid4()
    request = local_runner_module._PendingJob(
        job_id=active_id,
        job_dir=tmp_path,
        seed_name="active",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
    )
    runner._active[active_id] = local_runner_module._ActiveJob(
        request=request, process=_DummyProcess()
    )
    runner._pending.append(
        local_runner_module._PendingJob(
            job_id=pending_id,
            job_dir=tmp_path,
            seed_name="pending",
            castep_executable="/usr/bin/castep",
            mpirun_path="",
            num_cores=1,
        )
    )

    assert runner.active_job_ids() == frozenset({active_id})
    assert runner.pending_job_ids() == frozenset({pending_id})


def test_local_runner_output_candidates_name_a_running_jobs_castep(qapp, tmp_path: Path) -> None:
    runner = LocalJobRunner()
    job_id = uuid4()
    request = local_runner_module._PendingJob(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name="seed",
        castep_executable="/usr/bin/castep",
        mpirun_path="",
        num_cores=1,
        retrieval=RetrievalFlags(task_farming_enabled=True, num_farms=3),
    )
    runner._active[job_id] = local_runner_module._ActiveJob(
        request=request, process=_DummyProcess()
    )

    assert runner.output_candidates(job_id) == (
        tmp_path / "seed_farm001.castep",
        tmp_path / "seed.castep",
    )
    assert runner.output_candidates(uuid4()) == ()
