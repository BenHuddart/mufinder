from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.models import CalculationRun
from mufinder.domain.enums import FailureReason, JobStatus
from mufinder.domain.models import Result, utc_now
from mufinder.services.job_artifacts import RetrievalFlags
from mufinder.services.job_execution_controller import ClusterJobSubmission, JobExecutionController
from mufinder.services.project_service import ProjectDocument, ProjectService
from tests.support.jobs import make_job


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
    def __init__(self, *, fail_message: str | None = None) -> None:
        self.fail_message = fail_message
        self.submitted: list[dict[str, object]] = []

    def submit(self, **kwargs: object) -> str:
        self.submitted.append(kwargs)
        if self.fail_message is not None:
            raise ConnectionError(self.fail_message)
        return "12345"


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


def test_local_submission_creates_run_and_records_submission(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("local-job", service=service, document=document).id
    runner = FakeLocalRunner()
    controller = _controller(
        document=document,
        service=service,
        local_runner=runner,
        output_dir=tmp_path / "local-job",
    )

    outcome = controller.submit_local_jobs(
        [document.jobs[job_id]],
        num_cores=4,
        max_concurrent=2,
        castep_executable="/bin/castep",
        mpirun_path="/bin/mpirun",
    )

    job = document.jobs[job_id]
    run = controller.run_for_job(job_id)
    assert outcome.submitted_count == 1
    assert outcome.journal_changed is True
    assert runner.max_concurrent == 2
    assert runner.submitted[0]["job_id"] == job_id
    assert job.status == JobStatus.SUBMITTING
    assert job.execution_target == "local"
    assert job.num_processes == 4
    assert run is not None
    assert run.status == JobStatus.SUBMITTING
    assert document.provenance[-1].action == "submit_jobs"


def test_local_resubmission_accepts_failed_job_and_creates_next_run(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("local-retry", service=service, document=document).id
    job = document.jobs[job_id]
    job.status = JobStatus.FAILED
    job.error_message = "previous failure"
    runner = FakeLocalRunner()
    controller = _controller(
        document=document,
        service=service,
        local_runner=runner,
        output_dir=tmp_path / "local-retry",
    )

    outcome = controller.submit_local_jobs(
        [job],
        num_cores=2,
        max_concurrent=1,
        castep_executable="/bin/castep",
        mpirun_path="",
        eligible_statuses={JobStatus.FAILED},
    )

    run = controller.run_for_job(job_id)
    assert outcome.submitted_count == 1
    assert runner.submitted[0]["job_id"] == job_id
    assert job.status == JobStatus.SUBMITTING
    assert job.error_message is None
    assert run is not None
    assert run.run_index == 1
    assert run.status == JobStatus.SUBMITTING


def test_local_resubmission_accepts_cancelled_job(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("local-cancelled-retry", service=service, document=document).id
    job = document.jobs[job_id]
    job.status = JobStatus.CANCELLED
    runner = FakeLocalRunner()
    controller = _controller(
        document=document,
        service=service,
        local_runner=runner,
        output_dir=tmp_path / "local-cancelled-retry",
    )

    outcome = controller.submit_local_jobs(
        [job],
        num_cores=2,
        max_concurrent=1,
        castep_executable="/bin/castep",
        mpirun_path="",
        eligible_statuses={JobStatus.FAILED, JobStatus.CANCELLED},
    )

    assert outcome.submitted_count == 1
    assert runner.submitted[0]["job_id"] == job_id
    assert job.status == JobStatus.SUBMITTING


def _submit_local(controller: JobExecutionController, jobs, **options):
    return controller.submit_local_jobs(
        jobs,
        num_cores=options.pop("num_cores", 1),
        max_concurrent=1,
        castep_executable="/bin/castep",
        mpirun_path="",
        **options,
    )


def test_local_submission_skips_jobs_outside_the_eligible_statuses(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("ineligible", service=service, document=document)
    job.status = JobStatus.COMPLETED
    runner = FakeLocalRunner()
    controller = _controller(document=document, service=service, local_runner=runner, output_dir=tmp_path)

    outcome = _submit_local(controller, [job])

    assert outcome.submitted_count == 0
    assert runner.submitted == []


def test_local_submission_skips_jobs_the_project_cannot_submit(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("submit-raises", service=service, document=document)
    # COMPLETED passes the caller's eligibility filter but ProjectService.submit_job
    # rejects the transition, so the job is skipped untouched.
    job.status = JobStatus.COMPLETED
    runner = FakeLocalRunner()
    controller = _controller(document=document, service=service, local_runner=runner, output_dir=tmp_path)

    outcome = _submit_local(controller, [job], eligible_statuses={JobStatus.COMPLETED})

    assert outcome.submitted_count == 0
    assert runner.submitted == []
    assert job.status == JobStatus.COMPLETED


def test_local_submission_reports_jobs_without_an_output_directory(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("missing-dir", service=service, document=document)
    controller = _controller(document=document, service=service, output_dir=None)

    outcome = _submit_local(controller, [job])

    assert outcome.missing_paths == (job.label,)
    assert outcome.submitted_count == 0


def test_local_submission_fails_the_job_when_the_runner_raises(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("runner-error", service=service, document=document)
    runner = FakeLocalRunner(raise_on_submit=RuntimeError("runner exploded"))
    controller = _controller(document=document, service=service, local_runner=runner, output_dir=tmp_path)

    outcome = _submit_local(controller, [job], num_cores=2)

    assert outcome.submitted_count == 0
    assert outcome.failed_messages == ("- runner-error: runner exploded",)
    assert outcome.journal_changed is True
    assert job.status == JobStatus.FAILED
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_failed"


def test_local_submission_forwards_task_farming_to_the_runner(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("task-farm", service=service, document=document)
    runner = FakeLocalRunner()
    controller = _controller(document=document, service=service, local_runner=runner, output_dir=tmp_path)

    outcome = _submit_local(controller, [job], num_cores=4, task_farming_for_job=lambda _job: (True, 3))

    assert outcome.submitted_count == 1
    assert runner.submitted[0]["task_farming_enabled"] is True
    assert runner.submitted[0]["num_farms"] == 3


def _cluster_submission(job, output_dir: Path) -> ClusterJobSubmission:
    return ClusterJobSubmission(
        job=job,
        output_dir=output_dir,
        seed_name=job.label,
        remote_target=None,
        execution_target="cluster",
        num_processes=1,
        walltime="01:00:00",
        working_directory="/remote",
        template_script_path="",
        cores_per_node=1,
    )


def test_controller_is_inert_without_an_open_document(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("orphan", service=service, document=document)
    controller = _controller(document=None, service=service, output_dir=tmp_path)

    assert controller.next_run_index(job.id) == 1
    assert controller.run_for_job(job.id) is None
    assert controller.ensure_run_for_job(job) is None
    local = _submit_local(controller, [job])
    assert (local.submitted_count, local.journal_changed) == (0, False)
    cluster = controller.submit_cluster_job(_cluster_submission(job, tmp_path))
    assert (cluster.submitted_job, cluster.journal_changed) == (None, False)
    assert controller.record_submission([job], "local") is False
    result = Result(result_type="castep_cutoff", data={"e": 1.0})
    assert controller.complete_job_with_result(job, result=result, warnings=[], convergence_achieved=None) is False
    assert job.status == JobStatus.DEFINED


def test_record_submission_journals_only_when_there_are_jobs(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("rec-ok", service=service, document=document)
    service.submit_job(job)
    controller = _controller(document=document, service=service, output_dir=tmp_path)

    assert controller.record_submission([], "local") is False
    assert controller.record_submission([job], "local") is True
    assert document.provenance[-1].action == "submit_jobs"


def test_run_for_job_falls_back_to_the_latest_persisted_run(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("persisted", service=service, document=document).id
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    assert controller.run_for_job(job_id) is None

    older = CalculationRun(job_id=job_id, run_index=1, status=JobStatus.FAILED)
    newer = CalculationRun(job_id=job_id, run_index=2, status=JobStatus.COMPLETED)
    document.runs[older.id] = older
    document.runs[newer.id] = newer

    assert controller.run_for_job(job_id) is newer


def test_run_for_job_prefers_the_active_run_passed_to_the_controller(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("active", service=service, document=document)
    active = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.RUNNING)
    newer = CalculationRun(job_id=job.id, run_index=2, status=JobStatus.FAILED)
    document.runs[active.id] = active
    document.runs[newer.id] = newer
    controller = JobExecutionController(
        document_getter=lambda: document,
        project_service=service,
        local_runner=FakeLocalRunner(),
        cluster_runner=FakeClusterRunner(),
        job_output_directory=lambda _job: tmp_path,
        job_seed_name=lambda job: job.label,
        active_run_by_job={job.id: active.id},
    )

    assert controller.active_run_by_job == {job.id: active.id}
    assert controller.run_for_job(job.id) is active
    assert controller.ensure_run_for_job(job) is active


def test_ensure_run_for_job_creates_a_started_run_when_none_is_active(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("ensure-create", service=service, document=document)
    controller = _controller(document=document, service=service, output_dir=tmp_path)

    run = controller.ensure_run_for_job(job, status=JobStatus.RUNNING, started=True)

    assert run is not None
    assert run.status == JobStatus.RUNNING
    assert run.started_at is not None
    assert controller.active_run_by_job[job.id] == run.id


def test_lifecycle_events_update_job_run_and_terminal_journal(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("lifecycle-job", service=service, document=document).id
    controller = _controller(
        document=document,
        service=service,
        output_dir=tmp_path / "lifecycle-job",
    )
    controller.submit_local_jobs(
        [document.jobs[job_id]],
        num_cores=1,
        max_concurrent=1,
        castep_executable="/bin/castep",
        mpirun_path="",
    )

    queued = controller.mark_local_status_changed(job_id, JobStatus.QUEUED.value)
    started = controller.mark_started(job_id)
    result = Result(result_type="castep_cutoff", data={"total_energy": -1.0})
    document.results[result.id] = result
    controller.complete_job_with_result(
        document.jobs[job_id],
        result=result,
        warnings=["minor warning"],
        convergence_achieved=True,
    )

    job = document.jobs[job_id]
    run = controller.run_for_job(job_id)
    assert queued.changed is True
    assert started.changed is True
    assert job.status == JobStatus.COMPLETED
    assert run is not None
    assert run.status == JobStatus.COMPLETED
    assert run.result_id == result.id
    assert run.warnings == ["minor warning"]
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_completed"


def test_cluster_submission_success_marks_remote_job_queued(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("cluster-job", service=service, document=document).id
    runner = FakeClusterRunner()
    controller = _controller(
        document=document,
        service=service,
        cluster_runner=runner,
        output_dir=tmp_path / "cluster-job",
    )

    outcome = controller.submit_cluster_job(
        ClusterJobSubmission(
            job=document.jobs[job_id],
            output_dir=tmp_path / "cluster-job",
            seed_name="cluster-job",
            remote_target="/remote/cluster-job",
            execution_target="test-cluster",
            num_processes=8,
            walltime="01:00:00",
            working_directory="/remote",
            template_script_path="",
            cores_per_node=4,
        )
    )

    job = document.jobs[job_id]
    run = controller.run_for_job(job_id)
    assert outcome.submitted_job is job
    assert runner.submitted[0]["job_id"] == job_id
    assert job.status == JobStatus.QUEUED
    assert job.execution_target == "test-cluster"
    assert job.remote_job_id == "12345"
    assert job.remote_directory == "/remote/cluster-job"
    assert run is not None
    assert run.status == JobStatus.QUEUED


def test_cluster_resubmission_reuses_remote_directory_and_updates_remote_job_id(
    tmp_path: Path,
) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("cluster-retry", service=service, document=document).id
    job = document.jobs[job_id]
    job.status = JobStatus.FAILED
    job.remote_job_id = "old123"
    job.remote_directory = "/remote/existing/cluster-retry"
    job.error_message = "scheduler failed"
    runner = FakeClusterRunner()
    controller = _controller(
        document=document,
        service=service,
        cluster_runner=runner,
        output_dir=tmp_path / "cluster-retry",
    )

    outcome = controller.submit_cluster_job(
        ClusterJobSubmission(
            job=job,
            output_dir=tmp_path / "cluster-retry",
            seed_name="cluster-retry",
            remote_target="/remote/existing/cluster-retry",
            execution_target="test-cluster",
            num_processes=8,
            walltime="01:00:00",
            working_directory="/remote",
            template_script_path="",
            cores_per_node=4,
            reuse_existing_files=True,
        )
    )

    run = controller.run_for_job(job_id)
    assert outcome.submitted_job is job
    assert runner.submitted[0]["remote_directory"] == "/remote/existing/cluster-retry"
    assert runner.submitted[0]["reuse_existing_files"] is True
    assert job.status == JobStatus.QUEUED
    assert job.remote_job_id == "12345"
    assert job.remote_directory == "/remote/existing/cluster-retry"
    assert job.error_message is None
    assert run is not None
    assert run.status == JobStatus.QUEUED


def test_cluster_resubmission_failure_keeps_existing_remote_context(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("cluster-retry-fail", service=service, document=document).id
    job = document.jobs[job_id]
    job.status = JobStatus.FAILED
    job.remote_job_id = "old123"
    job.remote_directory = "/remote/existing/cluster-retry-fail"
    runner = FakeClusterRunner(fail_message="scheduler rejected retry")
    controller = _controller(
        document=document,
        service=service,
        cluster_runner=runner,
        output_dir=tmp_path / "cluster-retry-fail",
    )

    outcome = controller.submit_cluster_job(
        ClusterJobSubmission(
            job=job,
            output_dir=tmp_path / "cluster-retry-fail",
            seed_name="cluster-retry-fail",
            remote_target="/remote/existing/cluster-retry-fail",
            execution_target="test-cluster",
            num_processes=8,
            walltime="01:00:00",
            working_directory="/remote",
            template_script_path="",
            cores_per_node=4,
            reuse_existing_files=True,
        )
    )

    assert outcome.submitted_job is None
    assert outcome.failed_message == "- cluster-retry-fail: scheduler rejected retry"
    assert runner.submitted[0]["reuse_existing_files"] is True
    assert job.status == JobStatus.FAILED
    assert job.remote_job_id == "old123"
    assert job.remote_directory == "/remote/existing/cluster-retry-fail"


def test_cluster_resubmission_accepts_cancelled_job(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("cluster-cancelled-retry", service=service, document=document).id
    job = document.jobs[job_id]
    job.status = JobStatus.CANCELLED
    job.remote_job_id = "old123"
    job.remote_directory = "/remote/existing/cluster-cancelled-retry"
    runner = FakeClusterRunner()
    controller = _controller(
        document=document,
        service=service,
        cluster_runner=runner,
        output_dir=tmp_path / "cluster-cancelled-retry",
    )

    outcome = controller.submit_cluster_job(
        ClusterJobSubmission(
            job=job,
            output_dir=tmp_path / "cluster-cancelled-retry",
            seed_name="cluster-cancelled-retry",
            remote_target="/remote/existing/cluster-cancelled-retry",
            execution_target="test-cluster",
            num_processes=8,
            walltime="01:00:00",
            working_directory="/remote",
            template_script_path="",
            cores_per_node=4,
            reuse_existing_files=True,
        )
    )

    assert outcome.submitted_job is job
    assert runner.submitted[0]["reuse_existing_files"] is True
    assert job.status == JobStatus.QUEUED
    assert job.remote_job_id == "12345"
    assert job.remote_directory == "/remote/existing/cluster-cancelled-retry"


def test_cluster_submission_failure_marks_job_failed_and_records_terminal_event(tmp_path: Path) -> None:
    # JobRunner design decision 1: ClusterJobRunner.submit may raise; this
    # controller caller converts the exception into a failed outcome.
    service, document = _document(tmp_path)
    job_id = make_job("bad-cluster-job", service=service, document=document).id
    controller = _controller(
        document=document,
        service=service,
        cluster_runner=FakeClusterRunner(fail_message="scheduler rejected job"),
        output_dir=tmp_path / "bad-cluster-job",
    )

    outcome = controller.submit_cluster_job(
        ClusterJobSubmission(
            job=document.jobs[job_id],
            output_dir=tmp_path / "bad-cluster-job",
            seed_name="bad-cluster-job",
            remote_target="/remote/bad-cluster-job",
            execution_target="test-cluster",
            num_processes=8,
            walltime="01:00:00",
            working_directory="/remote",
            template_script_path="",
            cores_per_node=4,
        )
    )

    job = document.jobs[job_id]
    run = controller.run_for_job(job_id)
    assert outcome.submitted_job is None
    assert outcome.failed_message == "- bad-cluster-job: scheduler rejected job"
    assert outcome.journal_changed is True
    assert job.status == JobStatus.FAILED
    assert job.error_message == "scheduler rejected job"
    assert job.failure_reason == FailureReason.SUBMISSION
    assert run is not None
    assert run.status == JobStatus.FAILED
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_failed"


def test_cluster_submission_rejects_jobs_outside_the_eligible_statuses(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("cluster-ineligible", service=service, document=document)
    job.status = JobStatus.RUNNING
    runner = FakeClusterRunner()
    controller = _controller(document=document, service=service, cluster_runner=runner, output_dir=tmp_path)

    outcome = controller.submit_cluster_job(_cluster_submission(job, tmp_path))

    assert outcome.submitted_job is None
    assert runner.submitted == []
    assert job.status == JobStatus.RUNNING


def test_cluster_stopped_status_is_terminal_and_clears_active_run(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("stopped-cluster-job", service=service, document=document).id
    controller = _controller(
        document=document,
        service=service,
        cluster_runner=FakeClusterRunner(),
        output_dir=tmp_path / "stopped-cluster-job",
    )
    controller.submit_cluster_job(
        ClusterJobSubmission(
            job=document.jobs[job_id],
            output_dir=tmp_path / "stopped-cluster-job",
            seed_name="stopped-cluster-job",
            remote_target="/remote/stopped-cluster-job",
            execution_target="test-cluster",
            num_processes=8,
            walltime="01:00:00",
            working_directory="/remote",
            template_script_path="",
            cores_per_node=4,
        )
    )
    controller.mark_cluster_status_changed(job_id, JobStatus.RUNNING.value)

    update = controller.mark_cluster_status_changed(job_id, JobStatus.STOPPED.value)

    job = document.jobs[job_id]
    run = controller.run_for_job(job_id)
    assert update.changed is True
    assert update.journal_changed is True
    assert job.status == JobStatus.STOPPED
    # The runner reports ``stopped`` only for walltime stops, so the cause is
    # recorded even when no scheduler log could be downloaded.
    assert job.failure_reason == FailureReason.WALLTIME
    assert job.error_message == "The scheduler stopped the job when it reached its walltime limit."
    assert run is not None
    assert run.status == JobStatus.STOPPED
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_stopped"


def _tracked_run(controller: JobExecutionController, document: ProjectDocument, job, status: JobStatus):
    """Register an active run for ``job`` without a start time."""

    run = CalculationRun(job_id=job.id, run_index=1, status=status, started_at=None)
    document.runs[run.id] = run
    controller.active_run_by_job[job.id] = run.id
    return run


@pytest.mark.parametrize(
    "notify",
    [
        lambda controller, job_id: controller.mark_local_status_changed(job_id, JobStatus.QUEUED.value),
        lambda controller, job_id: controller.mark_started(job_id),
        lambda controller, job_id: controller.mark_cluster_status_changed(job_id, JobStatus.RUNNING.value),
        lambda controller, job_id: controller.fail_job(job_id, "boom"),
        lambda controller, job_id: controller.cancel_job(job_id),
    ],
    ids=["local-status", "started", "cluster-status", "fail", "cancel"],
)
def test_lifecycle_notifications_ignore_unknown_jobs(tmp_path: Path, notify) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)

    update = notify(controller, uuid4())

    assert update.job is None
    assert update.changed is False


def test_local_status_change_only_acts_on_queued(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("nonqueued", service=service, document=document)
    service.submit_job(job)
    controller = _controller(document=document, service=service, output_dir=tmp_path)

    update = controller.mark_local_status_changed(job.id, JobStatus.RUNNING.value)

    assert update.job is job
    assert update.changed is False
    assert job.status == JobStatus.SUBMITTING


@pytest.mark.parametrize(
    "notify",
    [
        lambda controller, job_id: controller.mark_local_status_changed(job_id, JobStatus.QUEUED.value),
        lambda controller, job_id: controller.mark_cluster_status_changed(job_id, JobStatus.QUEUED.value),
    ],
    ids=["local", "cluster"],
)
def test_queued_notification_is_unchanged_when_the_project_rejects_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notify
) -> None:
    service, document = _document(tmp_path)
    job = make_job("queued-error", service=service, document=document)
    service.submit_job(job)
    calls: list[object] = []

    def _reject(rejected_job: object, *, remote_job_id: object) -> None:
        calls.append(rejected_job)
        raise ValueError("invalid transition")

    monkeypatch.setattr(service, "mark_job_queued", _reject)
    controller = _controller(document=document, service=service, output_dir=tmp_path)

    update = notify(controller, job.id)

    assert calls == [job]
    assert update.job is job
    assert update.changed is False


def test_mark_started_stamps_a_missing_run_start_time(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("start", service=service, document=document)
    service.submit_job(job)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    run = _tracked_run(controller, document, job, JobStatus.SUBMITTING)

    update = controller.mark_started(job.id)

    assert update.changed is True
    assert job.status == JobStatus.RUNNING
    assert run.status == JobStatus.RUNNING
    assert run.started_at is not None


def test_complete_job_with_result_creates_a_run_when_none_exists(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("complete-create-run", service=service, document=document)
    service.submit_job(job)
    service.start_job(job)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    result = Result(result_type="castep_cutoff", data={"e": -2.0})
    document.results[result.id] = result

    completed = controller.complete_job_with_result(
        job, result=result, warnings=["w"], convergence_achieved=False
    )

    run = controller.run_for_job(job.id)
    assert completed is True
    assert job.status == JobStatus.COMPLETED
    assert run is not None
    assert run.status == JobStatus.COMPLETED
    assert run.result_id == result.id


def test_fail_job_closes_the_active_run(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("fail", service=service, document=document)
    service.submit_job(job)
    service.start_job(job)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    run = _tracked_run(controller, document, job, JobStatus.RUNNING)

    update = controller.fail_job(job.id, "kaboom")

    assert update.changed is True
    assert update.journal_changed is True
    assert job.status == JobStatus.FAILED
    assert run.status == JobStatus.FAILED
    assert run.started_at is not None
    assert run.completed_at is not None
    assert run.warnings == ["kaboom"]
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_failed"


def test_fail_job_journals_a_late_failure_without_reopening_a_completed_job(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("fail-terminal", service=service, document=document)
    job.status = JobStatus.COMPLETED
    controller = _controller(document=document, service=service, output_dir=tmp_path)

    update = controller.fail_job(job.id, "late failure")

    assert update.job is job
    assert job.status == JobStatus.COMPLETED
    assert document.provenance[-1].action == "job_failed"


def test_cancel_job_closes_the_active_run(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("cancel", service=service, document=document)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    run = _tracked_run(controller, document, job, JobStatus.SUBMITTING)

    update = controller.cancel_job(job.id)

    assert update.changed is True
    assert update.journal_changed is True
    assert job.status == JobStatus.CANCELLED
    assert run.status == JobStatus.CANCELLED
    assert run.completed_at is not None
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_cancelled"


@pytest.mark.parametrize(
    "status",
    ["definitely-not-a-status", JobStatus.RETRIEVING.value],
    ids=["unmapped", "not-applicable"],
)
def test_cluster_status_change_ignores_statuses_that_do_not_apply(tmp_path: Path, status: str) -> None:
    service, document = _document(tmp_path)
    job = make_job("cluster-noop", service=service, document=document)
    controller = _controller(document=document, service=service, output_dir=tmp_path)

    update = controller.mark_cluster_status_changed(job.id, status)

    assert update.job is job
    assert update.changed is False
    assert job.status == JobStatus.DEFINED


def test_cluster_status_change_moves_a_running_job_to_retrieving(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("cluster-retrieving", service=service, document=document)
    service.submit_job(job)
    service.mark_job_queued(job, remote_job_id="r1")
    service.start_job(job)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    run = _tracked_run(controller, document, job, JobStatus.RUNNING)

    update = controller.mark_cluster_status_changed(job.id, JobStatus.RETRIEVING.value)

    assert update.changed is True
    assert update.journal_changed is False
    assert job.status == JobStatus.RETRIEVING
    assert run.status == JobStatus.RETRIEVING


def test_cluster_status_change_to_running_stamps_the_run_start_time(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("cluster-running", service=service, document=document)
    service.submit_job(job)
    service.mark_job_queued(job, remote_job_id="r1")
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    run = _tracked_run(controller, document, job, JobStatus.QUEUED)

    update = controller.mark_cluster_status_changed(job.id, JobStatus.RUNNING.value)

    assert update.changed is True
    assert job.status == JobStatus.RUNNING
    assert run.status == JobStatus.RUNNING
    assert run.started_at is not None


def test_fail_job_diagnoses_castep_error_files(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _local_running_job(service, document, "castep-err")
    (tmp_path / "castep-err.0001.err").write_text("Error in parameters: bad keyword\n", encoding="utf-8")

    controller.fail_job(job.id, "CASTEP did not report successful completion.")

    assert job.failure_reason == FailureReason.CASTEP_ERROR
    assert job.error_message == "CASTEP did not report successful completion."


def test_fail_job_ignores_logs_from_before_the_submission(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _local_running_job(service, document, "continued")
    job.submitted_at = utc_now()
    stale = tmp_path / "continued.err"
    stale.write_text("Error from the previous run\n", encoding="utf-8")
    old = (job.submitted_at - timedelta(hours=1)).timestamp()
    os.utime(stale, (old, old))

    controller.fail_job(job.id, "process exited with code 1")

    assert job.failure_reason is None


def test_fail_job_keeps_an_explicit_reason(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _cluster_running_job(service, document, "explicit")

    controller.fail_job(job.id, "Scheduler reported the job as TIMEOUT.", reason=FailureReason.SUBMISSION)

    assert job.failure_reason == FailureReason.SUBMISSION


# ---------------------------------------------------------------------------
# Monitor-metadata stamping (job-lifecycle hardening, rec #2).
# ---------------------------------------------------------------------------


def test_mark_queue_checked_stamps_observed_jobs(tmp_path: Path) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace
    from uuid import uuid4

    service, document = _document(tmp_path)
    observed_id = make_job("observed", service=service, document=document).id
    unobserved_id = make_job("unobserved", service=service, document=document).id
    controller = _controller(document=document, service=service, output_dir=tmp_path)

    checked_at = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)
    snapshot = SimpleNamespace(
        checked_at=checked_at,
        observations_by_job_id={
            observed_id: "scheduler reports RUNNING",
            uuid4(): "not in this document",
        },
    )

    assert controller.mark_queue_checked(snapshot) is True

    observed = document.jobs[observed_id]
    assert observed.last_checked_at == checked_at
    assert observed.last_status_source == "scheduler_poll"
    assert observed.last_status_message == "scheduler reports RUNNING"
    # No status transition and no run mutation happened.
    assert observed.status == JobStatus.DEFINED
    assert controller.run_for_job(observed_id) is None

    unobserved = document.jobs[unobserved_id]
    assert unobserved.last_checked_at is None
    assert unobserved.last_status_source is None
    assert unobserved.last_status_message is None


def test_mark_queue_checked_tolerates_malformed_snapshots(tmp_path: Path) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace

    service, document = _document(tmp_path)
    job_id = make_job("untouched", service=service, document=document).id
    controller = _controller(document=document, service=service, output_dir=tmp_path)

    # Legacy emissions may pass a bare datetime instead of a snapshot object.
    assert controller.mark_queue_checked(datetime(2026, 7, 4, tzinfo=UTC)) is False
    # A snapshot without a datetime stamp is ignored too.
    assert (
        controller.mark_queue_checked(
            SimpleNamespace(checked_at=None, observations_by_job_id={job_id: "x"})
        )
        is False
    )
    assert document.jobs[job_id].last_checked_at is None
    assert document.jobs[job_id].last_status_source is None


def _cluster_running_job(service, document, label: str):
    job = make_job(label, service=service, document=document)
    job.execution_target = "scarf"
    job.status = JobStatus.RUNNING
    return job


def _local_running_job(service, document, label: str):
    job = make_job(label, service=service, document=document)
    job.execution_target = "local"
    job.status = JobStatus.RUNNING
    return job


def test_complete_job_with_result_stamps_retrieval_for_cluster_jobs(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _cluster_running_job(service, document, "cluster-done")
    result = Result(result_type="castep_cutoff", data={"total_energy": -1.0})

    controller.complete_job_with_result(
        job, result=result, warnings=[], convergence_achieved=True
    )

    assert job.status == JobStatus.COMPLETED
    assert job.last_status_source == "retrieval"
    assert job.last_status_message == "output retrieved and parsed"
    assert job.last_checked_at is not None


def test_complete_job_with_result_stamps_local_runner_for_local_jobs(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _local_running_job(service, document, "local-done")
    result = Result(result_type="castep_cutoff", data={"total_energy": -1.0})

    controller.complete_job_with_result(
        job, result=result, warnings=[], convergence_achieved=True
    )

    assert job.status == JobStatus.COMPLETED
    assert job.last_status_source == "local_runner"
    assert job.last_status_message == "process completed"
    assert job.last_checked_at is not None


def test_fail_job_stamps_retrieval_for_cluster_jobs(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _cluster_running_job(service, document, "cluster-broken")

    controller.fail_job(job.id, "Failed to retrieve or parse remote output: boom")

    assert job.status == JobStatus.FAILED
    # The full diagnostic stays on error_message; the stamp is the short
    # observation, not a duplicate of the error text.
    assert job.error_message == "Failed to retrieve or parse remote output: boom"
    assert job.last_status_source == "retrieval"
    assert job.last_status_message == "output retrieval failed"
    assert job.last_checked_at is not None


def test_fail_job_stamps_local_runner_for_local_jobs(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _local_running_job(service, document, "local-broken")

    controller.fail_job(job.id, "process exited with code 1")

    assert job.status == JobStatus.FAILED
    assert job.error_message == "process exited with code 1"
    assert job.last_status_source == "local_runner"
    assert job.last_status_message == "process failed"
    assert job.last_checked_at is not None


def test_fail_job_stamp_accepts_explicit_source_override(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _cluster_running_job(service, document, "cluster-override")

    controller.fail_job(
        job.id,
        "watchdog detected an impossible state",
        source="watchdog",
        status_message="impossible state detected",
    )

    assert job.last_status_source == "watchdog"
    assert job.last_status_message == "impossible state detected"


def test_mark_started_stamps_local_jobs_only(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)

    local_job = make_job("local-start", service=service, document=document)
    local_job.execution_target = "local"
    local_job.status = JobStatus.QUEUED
    controller.mark_started(local_job.id)
    assert local_job.status == JobStatus.RUNNING
    assert local_job.last_status_source == "local_runner"
    assert local_job.last_status_message == "process started"
    assert local_job.last_checked_at is not None

    # Cluster jobs are stamped from queue polls, not from mark_started.
    cluster_job = make_job("cluster-start", service=service, document=document)
    cluster_job.execution_target = "scarf"
    cluster_job.status = JobStatus.QUEUED
    controller.mark_started(cluster_job.id)
    assert cluster_job.status == JobStatus.RUNNING
    assert cluster_job.last_checked_at is None
    assert cluster_job.last_status_source is None


def test_mark_local_status_changed_stamps_queued_locally(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = make_job("local-queued", service=service, document=document)
    job.execution_target = "local"
    job.status = JobStatus.SUBMITTING

    update = controller.mark_local_status_changed(job.id, JobStatus.QUEUED.value)

    assert update.changed is True
    assert job.status == JobStatus.QUEUED
    assert job.last_status_source == "local_runner"
    assert job.last_status_message == "queued locally"
    assert job.last_checked_at is not None

    # An inert notification (wrong current status) does not stamp.
    other = make_job("local-inert", service=service, document=document)
    other.execution_target = "local"
    update = controller.mark_local_status_changed(other.id, JobStatus.QUEUED.value)
    assert update.changed is False
    assert other.last_checked_at is None


def test_prepare_cluster_submission_forwards_mem_per_cpu_to_the_runner(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("mem-job", service=service, document=document).id
    runner = FakeClusterRunner()
    controller = _controller(
        document=document,
        service=service,
        cluster_runner=runner,
        output_dir=tmp_path / "mem-job",
    )

    prepared = controller.prepare_cluster_submission(
        ClusterJobSubmission(
            job=document.jobs[job_id],
            output_dir=tmp_path / "mem-job",
            seed_name="mem-job",
            remote_target="/remote/mem-job",
            execution_target="test-cluster",
            num_processes=144,
            walltime="12:00:00",
            working_directory="/remote",
            template_script_path="",
            cores_per_node=24,
            mem_per_cpu_gb=6.0,
        )
    )

    assert prepared is not None
    assert prepared.submit_kwargs["mem_per_cpu_gb"] == 6.0


def test_prepare_cluster_submission_accepts_every_resubmittable_status(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    runner = FakeClusterRunner()

    for status in (JobStatus.DEFINED, JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED):
        label = f"status-{status.value}"
        job = make_job(label, service=service, document=document)
        job.status = status
        controller = _controller(
            document=document,
            service=service,
            cluster_runner=runner,
            output_dir=tmp_path / label,
        )

        prepared = controller.prepare_cluster_submission(
            ClusterJobSubmission(
                job=job,
                output_dir=tmp_path / label,
                seed_name=label,
                remote_target=f"/remote/{label}",
                execution_target="test-cluster",
                num_processes=8,
                walltime="01:00:00",
                working_directory="/remote",
                template_script_path="",
                cores_per_node=4,
            )
        )

        assert prepared is not None, f"{status.value} should be submittable to a cluster"
        assert job.status == JobStatus.SUBMITTING

    # Statuses with an in-flight or finished run are still refused.
    running = make_job("status-running", service=service, document=document)
    running.status = JobStatus.RUNNING
    controller = _controller(
        document=document,
        service=service,
        cluster_runner=runner,
        output_dir=tmp_path / "status-running",
    )
    assert (
        controller.prepare_cluster_submission(
            ClusterJobSubmission(
                job=running,
                output_dir=tmp_path / "status-running",
                seed_name="status-running",
                remote_target="/remote/status-running",
                execution_target="test-cluster",
                num_processes=8,
                walltime="01:00:00",
                working_directory="/remote",
                template_script_path="",
                cores_per_node=4,
            )
        )
        is None
    )
    assert running.status == JobStatus.RUNNING


def test_prepare_cluster_submission_defaults_mem_per_cpu_to_unset(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("no-mem-job", service=service, document=document).id
    runner = FakeClusterRunner()
    controller = _controller(
        document=document,
        service=service,
        cluster_runner=runner,
        output_dir=tmp_path / "no-mem-job",
    )

    prepared = controller.prepare_cluster_submission(
        ClusterJobSubmission(
            job=document.jobs[job_id],
            output_dir=tmp_path / "no-mem-job",
            seed_name="no-mem-job",
            remote_target="/remote/no-mem-job",
            execution_target="test-cluster",
            num_processes=8,
            walltime="01:00:00",
            working_directory="/remote",
            template_script_path="",
            cores_per_node=4,
        )
    )

    assert prepared is not None
    assert prepared.submit_kwargs["mem_per_cpu_gb"] is None


def test_prepare_cluster_submission_expands_retrieval_flags_into_runner_keywords(
    tmp_path: Path,
) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("neb-job", service=service, document=document).id
    controller = _controller(document=document, service=service, output_dir=tmp_path / "neb-job")
    flags = RetrievalFlags(retrieve_ts=True, task_farming_enabled=True, num_farms=3)

    prepared = controller.prepare_cluster_submission(
        ClusterJobSubmission(
            job=document.jobs[job_id],
            output_dir=tmp_path / "neb-job",
            seed_name="neb-job",
            remote_target="/remote/neb-job",
            execution_target="test-cluster",
            num_processes=8,
            walltime="01:00:00",
            working_directory="/remote",
            template_script_path="run.sh",
            cores_per_node=4,
            reuse_existing_files=True,
            retrieval=flags,
        )
    )

    assert prepared is not None
    assert prepared.submit_kwargs == {
        "job_id": job_id,
        "local_job_dir": tmp_path / "neb-job",
        "seed_name": "neb-job",
        "remote_directory": "/remote/neb-job",
        "num_processes": 8,
        "walltime": "01:00:00",
        "working_directory": "/remote",
        "template_script_path": "run.sh",
        "cores_per_node": 4,
        "memory_gb": None,
        "mem_per_cpu_gb": None,
        "queue_name": "",
        "comment": "",
        "castep_executable": "",
        "reuse_existing_files": True,
        "allow_existing_remote_directory": False,
        **flags.as_kwargs(),
    }


def test_local_submission_forwards_retrieval_flags_to_the_runner(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("phonon-job", service=service, document=document).id
    runner = FakeLocalRunner()
    controller = _controller(
        document=document,
        service=service,
        local_runner=runner,
        output_dir=tmp_path / "phonon-job",
    )
    flags = RetrievalFlags(retrieve_phonon=True, task_farming_enabled=True, num_farms=2)

    outcome = controller.submit_local_jobs(
        [document.jobs[job_id]],
        num_cores=2,
        max_concurrent=1,
        castep_executable="/bin/castep",
        mpirun_path="/bin/mpirun",
        retrieval_for_job=lambda _job: flags,
    )

    assert outcome.submitted_count == 1
    assert RetrievalFlags(
        **{name: runner.submitted[0][name] for name in flags.as_kwargs()}
    ) == flags
