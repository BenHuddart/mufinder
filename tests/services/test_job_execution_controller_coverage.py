"""Additional branch/error-path coverage for ``JobExecutionController``.

These tests complement ``test_job_execution_controller.py`` by exercising the
guard clauses, no-op branches, and error-handling paths that the primary
suite leaves uncovered. All runner/remote dependencies are faked; no network,
GUI, real files, sleeps, or randomness are involved.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Result
from mufinder.services.job_execution_controller import (
    ClusterJobSubmission,
    JobExecutionController,
)
from mufinder.services.project_service import ProjectDocument, ProjectService


class FakeLocalRunner:
    def __init__(self, *, raise_on_submit: Exception | None = None) -> None:
        self.max_concurrent = 1
        self.submitted: list[dict[str, object]] = []
        self._raise_on_submit = raise_on_submit

    def submit(self, **kwargs: object) -> None:
        self.submitted.append(kwargs)
        if self._raise_on_submit is not None:
            raise self._raise_on_submit


class FakeClusterRunner:
    def submit(self, **kwargs: object) -> str:  # pragma: no cover - guard only
        return "remote-id"


def _document(tmp_path: Path) -> tuple[ProjectService, ProjectDocument]:
    service = ProjectService()
    return service, service.create_project(tmp_path / "execution.mufinder", "Execution")


def _controller(
    *,
    document: ProjectDocument | None,
    service: ProjectService,
    local_runner: FakeLocalRunner | None = None,
    cluster_runner: FakeClusterRunner | None = None,
    output_dir: Path | None,
) -> JobExecutionController:
    return JobExecutionController(
        document_getter=lambda: document,
        project_service=service,
        local_runner=local_runner or FakeLocalRunner(),
        cluster_runner=cluster_runner or FakeClusterRunner(),
        job_output_directory=lambda _job: output_dir,
        job_seed_name=lambda job: job.label,
    )


def _add_job(service: ProjectService, document: ProjectDocument, label: str = "job") -> UUID:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
    ).id


# --------------------------------------------------------------------------- #
# Document-None guard clauses
# --------------------------------------------------------------------------- #


def test_next_run_index_returns_one_when_document_missing(tmp_path: Path) -> None:
    service = ProjectService()
    controller = _controller(
        document=None, service=service, output_dir=tmp_path / "x"
    )
    assert controller.next_run_index(uuid4()) == 1


def test_run_for_job_returns_none_when_document_missing(tmp_path: Path) -> None:
    service = ProjectService()
    controller = _controller(
        document=None, service=service, output_dir=tmp_path / "x"
    )
    assert controller.run_for_job(uuid4()) is None


def test_ensure_run_for_job_returns_none_when_document_missing(tmp_path: Path) -> None:
    service = ProjectService()
    controller = _controller(
        document=None, service=service, output_dir=tmp_path / "x"
    )
    job = service.add_job(
        ProjectService().create_project(tmp_path / "throwaway.mufinder", "T"),
        label="orphan",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
    )
    assert controller.ensure_run_for_job(job) is None


def test_submit_local_jobs_returns_empty_outcome_when_document_missing(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    controller = _controller(
        document=None, service=service, output_dir=tmp_path / "x"
    )
    outcome = controller.submit_local_jobs(
        [],
        num_cores=1,
        max_concurrent=1,
        castep_executable="/bin/castep",
        mpirun_path="",
    )
    assert outcome.submitted_count == 0
    assert outcome.journal_changed is False


def test_submit_cluster_job_returns_empty_outcome_when_document_missing(
    tmp_path: Path,
) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "c")
    job = document.jobs[job_id]
    controller = _controller(
        document=None, service=service, output_dir=tmp_path / "x"
    )
    outcome = controller.submit_cluster_job(
        ClusterJobSubmission(
            job=job,
            output_dir=tmp_path / "x",
            seed_name="c",
            remote_target=None,
            execution_target="cluster",
            num_processes=1,
            walltime="01:00:00",
            working_directory="/remote",
            template_script_path="",
            cores_per_node=1,
        )
    )
    assert outcome.submitted_job is None
    assert outcome.journal_changed is False


def test_record_submission_false_when_document_missing(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "rec")
    controller = _controller(
        document=None, service=service, output_dir=tmp_path / "x"
    )
    assert controller.record_submission([document.jobs[job_id]], "local") is False


def test_record_submission_false_when_no_jobs(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    assert controller.record_submission([], "local") is False


def test_record_submission_true_records_event(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "rec-ok")
    job = document.jobs[job_id]
    service.submit_job(job)  # DEFINED -> SUBMITTING so the milestone is recorded
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    assert controller.record_submission([job], "local") is True
    assert document.provenance[-1].action == "submit_jobs"


# --------------------------------------------------------------------------- #
# run_for_job / ensure_run_for_job persisted fallbacks
# --------------------------------------------------------------------------- #


def test_run_for_job_falls_back_to_latest_persisted_run(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "persisted")
    older = CalculationRun(job_id=job_id, run_index=1, status=JobStatus.FAILED)
    newer = CalculationRun(job_id=job_id, run_index=2, status=JobStatus.COMPLETED)
    document.runs[older.id] = older
    document.runs[newer.id] = newer
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    # No active run mapped -> picks highest run_index.
    assert controller.run_for_job(job_id) is newer


def test_run_for_job_returns_none_when_no_runs(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "norun")
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    assert controller.run_for_job(job_id) is None


def test_run_for_job_uses_active_mapping(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "active")
    run = CalculationRun(job_id=job_id, run_index=1, status=JobStatus.RUNNING)
    document.runs[run.id] = run
    controller = JobExecutionController(
        document_getter=lambda: document,
        project_service=service,
        local_runner=FakeLocalRunner(),
        cluster_runner=FakeClusterRunner(),
        job_output_directory=lambda _job: tmp_path / "x",
        job_seed_name=lambda job: job.label,
        active_run_by_job={job_id: run.id},
    )
    assert controller.active_run_by_job == {job_id: run.id}
    assert controller.run_for_job(job_id) is run


def test_ensure_run_for_job_returns_existing_active_run(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "ensure-existing")
    job = document.jobs[job_id]
    run = CalculationRun(job_id=job_id, run_index=1, status=JobStatus.RUNNING)
    document.runs[run.id] = run
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    controller.active_run_by_job[job_id] = run.id
    assert controller.ensure_run_for_job(job) is run


def test_ensure_run_for_job_creates_fallback_with_started_at(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "ensure-create")
    job = document.jobs[job_id]
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    run = controller.ensure_run_for_job(job, status=JobStatus.RUNNING, started=True)
    assert run is not None
    assert run.status == JobStatus.RUNNING
    assert run.started_at is not None
    assert controller.active_run_by_job[job_id] == run.id


# --------------------------------------------------------------------------- #
# submit_local_jobs branch coverage
# --------------------------------------------------------------------------- #


def test_submit_local_jobs_skips_ineligible_status(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "ineligible")
    job = document.jobs[job_id]
    job.status = JobStatus.COMPLETED
    runner = FakeLocalRunner()
    controller = _controller(
        document=document,
        service=service,
        local_runner=runner,
        output_dir=tmp_path / "ineligible",
    )
    outcome = controller.submit_local_jobs(
        [job],
        num_cores=2,
        max_concurrent=1,
        castep_executable="/bin/castep",
        mpirun_path="",
    )
    assert outcome.submitted_count == 0
    assert runner.submitted == []


def test_submit_local_jobs_reports_missing_output_dir(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "missing-dir")
    job = document.jobs[job_id]
    controller = JobExecutionController(
        document_getter=lambda: document,
        project_service=service,
        local_runner=FakeLocalRunner(),
        cluster_runner=FakeClusterRunner(),
        job_output_directory=lambda _job: None,
        job_seed_name=lambda j: j.label,
    )
    outcome = controller.submit_local_jobs(
        [job],
        num_cores=1,
        max_concurrent=1,
        castep_executable="/bin/castep",
        mpirun_path="",
    )
    assert outcome.missing_paths == (job.label,)
    assert outcome.submitted_count == 0


def test_submit_local_jobs_skips_when_submit_job_raises_value_error(
    tmp_path: Path,
) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "submit-raises")
    job = document.jobs[job_id]
    # COMPLETED is allowed-through the eligibility filter here but rejected by
    # ProjectService.submit_job, exercising the ValueError continue branch.
    job.status = JobStatus.COMPLETED
    runner = FakeLocalRunner()
    controller = _controller(
        document=document,
        service=service,
        local_runner=runner,
        output_dir=tmp_path / "submit-raises",
    )
    outcome = controller.submit_local_jobs(
        [job],
        num_cores=1,
        max_concurrent=1,
        castep_executable="/bin/castep",
        mpirun_path="",
        eligible_statuses={JobStatus.COMPLETED},
    )
    assert outcome.submitted_count == 0
    assert runner.submitted == []
    assert job.status == JobStatus.COMPLETED


def test_submit_local_jobs_handles_runner_exception_and_fails_job(
    tmp_path: Path,
) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "runner-error")
    job = document.jobs[job_id]
    runner = FakeLocalRunner(raise_on_submit=RuntimeError("runner exploded"))
    controller = _controller(
        document=document,
        service=service,
        local_runner=runner,
        output_dir=tmp_path / "runner-error",
    )
    outcome = controller.submit_local_jobs(
        [job],
        num_cores=2,
        max_concurrent=1,
        castep_executable="/bin/castep",
        mpirun_path="",
    )
    assert outcome.submitted_count == 0
    assert outcome.failed_messages == ("- runner-error: runner exploded",)
    assert outcome.journal_changed is True
    assert document.jobs[job_id].status == JobStatus.FAILED
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_failed"


def test_submit_local_jobs_invokes_task_farming_callback(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "task-farm")
    job = document.jobs[job_id]
    runner = FakeLocalRunner()
    controller = _controller(
        document=document,
        service=service,
        local_runner=runner,
        output_dir=tmp_path / "task-farm",
    )
    outcome = controller.submit_local_jobs(
        [job],
        num_cores=4,
        max_concurrent=1,
        castep_executable="/bin/castep",
        mpirun_path="",
        task_farming_for_job=lambda _j: (True, 3),
    )
    assert outcome.submitted_count == 1
    assert runner.submitted[0]["task_farming_enabled"] is True
    assert runner.submitted[0]["num_farms"] == 3
    assert runner.max_concurrent == 1


# --------------------------------------------------------------------------- #
# submit_cluster_job guard for ineligible status & submit_job ValueError
# --------------------------------------------------------------------------- #


def test_submit_cluster_job_rejects_ineligible_status(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "cluster-ineligible")
    job = document.jobs[job_id]
    job.status = JobStatus.RUNNING
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    outcome = controller.submit_cluster_job(
        ClusterJobSubmission(
            job=job,
            output_dir=tmp_path / "x",
            seed_name="cluster-ineligible",
            remote_target=None,
            execution_target="cluster",
            num_processes=1,
            walltime="01:00:00",
            working_directory="/remote",
            template_script_path="",
            cores_per_node=1,
        )
    )
    assert outcome.submitted_job is None


# --------------------------------------------------------------------------- #
# Lifecycle notifications: job-not-found and no-op branches
# --------------------------------------------------------------------------- #


def test_mark_local_status_changed_unknown_job(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    update = controller.mark_local_status_changed(uuid4(), JobStatus.QUEUED.value)
    assert update.job is None
    assert update.changed is False


def test_mark_local_status_changed_ignores_non_queued_value(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "nonqueued")
    job = document.jobs[job_id]
    job.status = JobStatus.SUBMITTING
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    update = controller.mark_local_status_changed(job_id, JobStatus.RUNNING.value)
    assert update.job is job
    assert update.changed is False


def test_mark_local_status_changed_value_error_returns_unchanged(
    tmp_path: Path,
) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "queued-error")
    job = document.jobs[job_id]
    # job.status == SUBMITTING is required to pass the gate, but the underlying
    # job state is DEFINED so mark_job_queued raises ValueError. We force the
    # gate condition by setting status to SUBMITTING while the project-service
    # transition still rejects it.
    job.status = JobStatus.SUBMITTING
    captured: dict[str, object] = {}

    def _raise(_job: object, *, remote_job_id: object) -> None:
        captured["called"] = True
        raise ValueError("invalid transition")

    service.mark_job_queued = _raise  # type: ignore[method-assign]
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    update = controller.mark_local_status_changed(job_id, JobStatus.QUEUED.value)
    assert captured.get("called") is True
    assert update.job is job
    assert update.changed is False


def test_mark_started_unknown_job(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    update = controller.mark_started(uuid4())
    assert update.job is None
    assert update.changed is False


def test_mark_started_creates_started_at_when_missing(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "start")
    job = document.jobs[job_id]
    service.submit_job(job)  # DEFINED -> SUBMITTING
    run = CalculationRun(
        job_id=job_id, run_index=1, status=JobStatus.SUBMITTING, started_at=None
    )
    document.runs[run.id] = run
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    controller.active_run_by_job[job_id] = run.id
    update = controller.mark_started(job_id)
    assert update.changed is True
    assert run.status == JobStatus.RUNNING
    assert run.started_at is not None
    assert job.status == JobStatus.RUNNING


def test_complete_job_with_result_returns_false_when_document_missing(
    tmp_path: Path,
) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "complete-no-doc")
    job = document.jobs[job_id]
    controller = _controller(
        document=None, service=service, output_dir=tmp_path / "x"
    )
    result = Result(result_type="castep_cutoff", data={"e": 1.0})
    assert (
        controller.complete_job_with_result(
            job, result=result, warnings=[], convergence_achieved=None
        )
        is False
    )


def test_complete_job_with_result_creates_run_when_none_exists(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "complete-create-run")
    job = document.jobs[job_id]
    service.submit_job(job)
    service.start_job(job)  # now RUNNING
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    result = Result(result_type="castep_cutoff", data={"e": -2.0})
    document.results[result.id] = result
    assert (
        controller.complete_job_with_result(
            job,
            result=result,
            warnings=["w"],
            convergence_achieved=False,
        )
        is True
    )
    run = controller.run_for_job(job_id)
    assert run is not None
    assert run.status == JobStatus.COMPLETED
    assert run.result_id == result.id
    assert job.status == JobStatus.COMPLETED


# --------------------------------------------------------------------------- #
# fail_job / cancel_job
# --------------------------------------------------------------------------- #


def test_fail_job_unknown_job(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    update = controller.fail_job(uuid4(), "boom")
    assert update.job is None
    assert update.changed is False


def test_fail_job_with_active_run_sets_started_and_completed(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "fail")
    job = document.jobs[job_id]
    service.submit_job(job)
    service.start_job(job)  # RUNNING
    run = CalculationRun(
        job_id=job_id, run_index=1, status=JobStatus.RUNNING, started_at=None
    )
    document.runs[run.id] = run
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    controller.active_run_by_job[job_id] = run.id
    update = controller.fail_job(job_id, "kaboom")
    assert update.changed is True
    assert update.journal_changed is True
    assert job.status == JobStatus.FAILED
    assert run.status == JobStatus.FAILED
    assert run.started_at is not None
    assert run.completed_at is not None
    assert run.warnings == ["kaboom"]
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_failed"


def test_fail_job_ignores_terminal_status_transition(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "fail-terminal")
    job = document.jobs[job_id]
    job.status = JobStatus.COMPLETED  # not in the failable set
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    update = controller.fail_job(job_id, "late failure")
    # Still records a terminal event and returns changed, but job stays COMPLETED.
    assert update.job is job
    assert job.status == JobStatus.COMPLETED
    assert document.provenance[-1].action == "job_failed"


def test_cancel_job_unknown_job(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    update = controller.cancel_job(uuid4())
    assert update.job is None
    assert update.changed is False


def test_cancel_job_with_active_run(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "cancel")
    job = document.jobs[job_id]  # DEFINED is cancellable
    run = CalculationRun(job_id=job_id, run_index=1, status=JobStatus.SUBMITTING)
    document.runs[run.id] = run
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    controller.active_run_by_job[job_id] = run.id
    update = controller.cancel_job(job_id)
    assert update.changed is True
    assert update.journal_changed is True
    assert job.status == JobStatus.CANCELLED
    assert run.status == JobStatus.CANCELLED
    assert run.completed_at is not None
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_cancelled"


# --------------------------------------------------------------------------- #
# mark_cluster_status_changed branches
# --------------------------------------------------------------------------- #


def test_mark_cluster_status_changed_unknown_job(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    update = controller.mark_cluster_status_changed(uuid4(), JobStatus.RUNNING.value)
    assert update.job is None
    assert update.changed is False


def test_mark_cluster_status_changed_unmapped_status(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "cluster-unmapped")
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    update = controller.mark_cluster_status_changed(job_id, "definitely-not-a-status")
    assert update.job is document.jobs[job_id]
    assert update.changed is False


def test_mark_cluster_status_changed_noop_when_status_not_applicable(
    tmp_path: Path,
) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "cluster-noop")
    job = document.jobs[job_id]
    # RETRIEVING is a valid mapped status, but job is DEFINED so the transition
    # conditions all fail and we hit the else -> unchanged branch.
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    update = controller.mark_cluster_status_changed(job_id, JobStatus.RETRIEVING.value)
    assert update.job is job
    assert update.changed is False
    assert job.status == JobStatus.DEFINED


def test_mark_cluster_status_changed_value_error_returns_unchanged(
    tmp_path: Path,
) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "cluster-valueerror")
    job = document.jobs[job_id]
    job.status = JobStatus.SUBMITTING

    def _raise(_job: object, *, remote_job_id: object) -> None:
        raise ValueError("nope")

    service.mark_job_queued = _raise  # type: ignore[method-assign]
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    update = controller.mark_cluster_status_changed(job_id, JobStatus.QUEUED.value)
    assert update.job is job
    assert update.changed is False


def test_mark_cluster_status_changed_retrieving_transition(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "cluster-retrieving")
    job = document.jobs[job_id]
    service.submit_job(job)
    service.mark_job_queued(job, remote_job_id="r1")
    service.start_job(job)  # RUNNING
    run = CalculationRun(
        job_id=job_id, run_index=1, status=JobStatus.RUNNING, started_at=None
    )
    document.runs[run.id] = run
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    controller.active_run_by_job[job_id] = run.id
    update = controller.mark_cluster_status_changed(
        job_id, JobStatus.RETRIEVING.value
    )
    assert update.changed is True
    assert update.journal_changed is False
    assert job.status == JobStatus.RETRIEVING
    assert run.status == JobStatus.RETRIEVING


def test_mark_cluster_status_changed_running_sets_started_at(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _add_job(service, document, "cluster-running")
    job = document.jobs[job_id]
    service.submit_job(job)
    service.mark_job_queued(job, remote_job_id="r1")  # QUEUED
    run = CalculationRun(
        job_id=job_id, run_index=1, status=JobStatus.QUEUED, started_at=None
    )
    document.runs[run.id] = run
    controller = _controller(
        document=document, service=service, output_dir=tmp_path / "x"
    )
    controller.active_run_by_job[job_id] = run.id
    update = controller.mark_cluster_status_changed(job_id, JobStatus.RUNNING.value)
    assert update.changed is True
    assert job.status == JobStatus.RUNNING
    assert run.status == JobStatus.RUNNING
    assert run.started_at is not None
