"""Tests for the retry-retrieval action (job-lifecycle hardening, rec #4).

Covers three layers:

* ``ProjectService.retry_job_retrieval`` primitive behaviour.
* ``ClusterJobRunner.retrieve_outputs_for`` registering a tracked entry and
  emitting the poll-loop signal sequence via ``_retrieve_results``.
* ``JobExecutionController.prepare_retry_retrieval`` run bookkeeping.

The cluster-runner fakes mirror ``tests/services/test_cluster_runner.py`` (copied
here rather than imported, since that file is owned by another stream this round).
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

import mufinder.services.cluster_runner as cluster_runner_module
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import Job
from mufinder.services.cluster_runner import ClusterJobRunner
from mufinder.services.job_execution_controller import JobExecutionController
from mufinder.services.project_service import ProjectDocument, ProjectService
from tests.support.jobs import make_job


# ---------------------------------------------------------------------------
# Cluster-runner fakes (copied from test_cluster_runner.py)
# ---------------------------------------------------------------------------
class _FakeSSH:
    def __init__(self) -> None:
        self.is_connected = True
        self.commands: list[tuple[str, object]] = []
        self.remote_files: dict[str, bytes] = {}

    def mkdir(self, path: str, exist_ok: bool = True) -> None:
        self.commands.append(("mkdir", (path, exist_ok)))

    def put(self, local_path: Path, remote_path: str) -> None:
        self.commands.append(("put", remote_path))
        self.remote_files[remote_path] = Path(local_path).read_bytes()

    def chmod(self, remote_path: str, mode: int) -> None:
        self.commands.append(("chmod", (remote_path, mode)))

    def get(self, remote_path: str, local_path: Path) -> None:
        self.commands.append(("get", remote_path))
        local_path.write_bytes(self.remote_files.get(remote_path, b"ok"))

    def exists(self, remote_path: str) -> bool:
        return remote_path in self.remote_files

    def listdir(self, remote_path: str) -> list[str]:
        prefix = remote_path.rstrip("/") + "/"
        entries: list[str] = []
        for path in self.remote_files:
            if path.startswith(prefix):
                entries.append(path[len(prefix):].split("/", 1)[0])
        return sorted(set(entries))

    def disconnect(self) -> None:
        self.is_connected = False


class _FakeScheduler:
    def submit(self, _ssh, request) -> str:
        return "12345"

    def cancel(self, _ssh, _job_id: str) -> None:
        return


# ---------------------------------------------------------------------------
# ProjectService.retry_job_retrieval
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("source", [JobStatus.FAILED, JobStatus.STOPPED])
def test_retry_job_retrieval_moves_terminal_to_retrieving(source: JobStatus) -> None:
    service = ProjectService()
    job = make_job(status=source, remote_job_id="99", remote_directory="/remote/job")
    job.error_message = "retrieval blew up"

    service.retry_job_retrieval(job)

    assert job.status == JobStatus.RETRIEVING
    assert job.error_message is None
    # Remote metadata is preserved so retrieval targets the same directory.
    assert job.remote_job_id == "99"
    assert job.remote_directory == "/remote/job"


@pytest.mark.parametrize(
    "source",
    [JobStatus.DEFINED, JobStatus.COMPLETED, JobStatus.CANCELLED],
)
def test_retry_job_retrieval_rejects_illegal_source(source: JobStatus) -> None:
    service = ProjectService()
    job = make_job(status=source)
    with pytest.raises(ValueError, match="Invalid job transition"):
        service.retry_job_retrieval(job)
    assert job.status == source


# ---------------------------------------------------------------------------
# ClusterJobRunner.retrieve_outputs_for
# ---------------------------------------------------------------------------
def test_retrieve_outputs_for_registers_entry_and_completes(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    ssh = _FakeSSH()
    runner._ssh = ssh
    runner._scheduler = _FakeScheduler()
    ssh.remote_files["/remote/job/seed.castep"] = b"castep output"

    monkeypatch.setattr(
        cluster_runner_module,
        "parse_castep_output",
        lambda _path: CastepResult(completed=True, total_energy=-1.0),
    )

    job_id = uuid4()
    finished: list[object] = []
    statuses: list[str] = []
    runner.job_finished.connect(lambda _jid, parsed: finished.append(parsed))
    runner.status_changed.connect(lambda _jid, status: statuses.append(status))

    runner.retrieve_outputs_for(
        job_id,
        remote_job_id="120",
        remote_directory="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )

    # The tracked entry is (re-)registered from the persisted metadata.
    assert job_id in runner._tracked_jobs
    # Signal sequence matches the poll-loop success path: retrieving -> completed.
    assert statuses[0] == JobStatus.RETRIEVING.value
    assert statuses[-1] == JobStatus.COMPLETED.value
    assert len(finished) == 1


def test_retrieve_outputs_for_incomplete_marks_failed(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
    runner._scheduler = _FakeScheduler()
    runner._ssh.remote_files["/remote/job/seed.castep"] = b"castep output"

    monkeypatch.setattr(
        cluster_runner_module,
        "parse_castep_output",
        lambda _path: CastepResult(completed=False, error_message="not converged"),
    )

    failures: list[str] = []
    statuses: list[str] = []
    runner.job_failed.connect(lambda _jid, message: failures.append(message))
    runner.status_changed.connect(lambda _jid, status: statuses.append(status))

    runner.retrieve_outputs_for(
        uuid4(),
        remote_job_id="120",
        remote_directory="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )

    assert failures == ["not converged"]
    assert statuses[0] == JobStatus.RETRIEVING.value
    assert statuses[-1] == JobStatus.FAILED.value


def test_retrieve_outputs_for_not_connected_resolves_to_failed(
    qapp, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    # No _ssh set -> is_connected is False; a naive retrieval would strand the
    # job in RETRIEVING, so it must resolve to FAILED instead.
    job_id = uuid4()
    failures: list[str] = []
    statuses: list[str] = []
    runner.job_failed.connect(lambda _jid, message: failures.append(message))
    runner.status_changed.connect(lambda _jid, status: statuses.append(status))

    runner.retrieve_outputs_for(
        job_id,
        remote_job_id="120",
        remote_directory="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )

    assert len(failures) == 1
    assert "not connected" in failures[0].lower()
    assert statuses == [JobStatus.FAILED.value]


# ---------------------------------------------------------------------------
# JobExecutionController.prepare_retry_retrieval
# ---------------------------------------------------------------------------
def _controller_with_job(
    tmp_path: Path, job: Job
) -> tuple[JobExecutionController, ProjectDocument]:
    service = ProjectService()
    document = service.create_project(tmp_path / "retry.mufinder", "Retry")
    document.jobs[job.id] = job
    controller = JobExecutionController(
        document_getter=lambda: document,
        project_service=service,
        local_runner=object(),
        cluster_runner=object(),
        job_output_directory=lambda _job: None,
        job_seed_name=lambda _job: "seed",
    )
    return controller, document


def test_prepare_retry_retrieval_creates_fresh_retrieving_run(tmp_path: Path) -> None:
    job = make_job(
        status=JobStatus.FAILED,
        execution_target="scarf",
        remote_job_id="99",
        remote_directory="/remote/job",
    )
    controller, document = _controller_with_job(tmp_path, job)

    result = controller.prepare_retry_retrieval(job.id)

    assert result is job
    assert job.status == JobStatus.RETRIEVING
    assert job.last_status_source == "user"
    assert job.last_status_message == "Retry retrieval requested"
    assert job.last_checked_at is not None

    # A fresh active run was registered, in RETRIEVING, distinct from any prior run.
    active_run_id = controller.active_run_by_job.get(job.id)
    assert active_run_id is not None
    run = document.runs[active_run_id]
    assert run.status == JobStatus.RETRIEVING
    assert run.job_id == job.id
    assert run.run_index == 1


def test_prepare_retry_retrieval_increments_run_index_over_prior_run(tmp_path: Path) -> None:
    job = make_job(
        status=JobStatus.FAILED,
        execution_target="scarf",
        remote_job_id="99",
        remote_directory="/remote/job",
    )
    controller, document = _controller_with_job(tmp_path, job)
    # Simulate a prior terminal run from the original submission.
    from mufinder.domain.models import CalculationRun

    prior = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.FAILED)
    document.runs[prior.id] = prior

    controller.prepare_retry_retrieval(job.id)

    active_run_id = controller.active_run_by_job[job.id]
    assert document.runs[active_run_id].run_index == 2
    # The prior terminal run is untouched (not overwritten to RETRIEVING).
    assert prior.status == JobStatus.FAILED


def test_retry_completion_lands_on_fresh_run_not_the_prior_terminal_run(
    tmp_path: Path,
) -> None:
    """The whole point of the fresh RETRIEVING run: a successful re-retrieval must
    complete the NEW run and leave the original FAILED run untouched, rather than
    overwriting a terminal run to COMPLETED."""

    from mufinder.domain.models import CalculationRun, Result

    job = make_job(
        status=JobStatus.FAILED,
        execution_target="scarf",
        remote_job_id="99",
        remote_directory="/remote/job",
    )
    controller, document = _controller_with_job(tmp_path, job)
    prior = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.FAILED)
    document.runs[prior.id] = prior

    controller.prepare_retry_retrieval(job.id)
    # Capture the active (fresh) run before complete_job_with_result pops it.
    fresh_run_id = controller.active_run_by_job[job.id]

    result = Result(result_type="castep_cutoff", data={"total_energy": -1.0})
    document.results[result.id] = result
    assert (
        controller.complete_job_with_result(
            job, result=result, warnings=[], convergence_achieved=None
        )
        is True
    )

    # The job completes from the RETRIEVING state the retry established.
    assert job.status == JobStatus.COMPLETED
    # The completion landed on the fresh run, not the prior terminal run.
    assert document.runs[fresh_run_id].status == JobStatus.COMPLETED
    assert document.runs[fresh_run_id].result_id == result.id
    assert prior.status == JobStatus.FAILED
    assert prior.result_id is None


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda j: setattr(j, "status", JobStatus.COMPLETED), id="not-terminal"),
        pytest.param(lambda j: setattr(j, "execution_target", "local"), id="local-target"),
        pytest.param(lambda j: setattr(j, "remote_job_id", None), id="missing-remote-id"),
        pytest.param(lambda j: setattr(j, "remote_directory", None), id="missing-remote-dir"),
    ],
)
def test_prepare_retry_retrieval_returns_none_when_ineligible(mutate, tmp_path: Path) -> None:
    job = make_job(
        status=JobStatus.FAILED,
        execution_target="scarf",
        remote_job_id="99",
        remote_directory="/remote/job",
    )
    mutate(job)
    original_status = job.status
    controller, _document = _controller_with_job(tmp_path, job)

    assert controller.prepare_retry_retrieval(job.id) is None
    # Ineligible jobs are left untouched.
    assert job.status == original_status
    assert controller.active_run_by_job.get(job.id) is None
