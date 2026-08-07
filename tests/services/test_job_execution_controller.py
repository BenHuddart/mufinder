from __future__ import annotations

from pathlib import Path
from uuid import UUID

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Result
from mufinder.services.job_execution_controller import ClusterJobSubmission, JobExecutionController
from mufinder.services.project_service import ProjectDocument, ProjectService


class FakeLocalRunner:
    def __init__(self) -> None:
        self.max_concurrent = 1
        self.submitted: list[dict[str, object]] = []

    def submit(self, **kwargs: object) -> None:
        self.submitted.append(kwargs)


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
    document: ProjectDocument,
    service: ProjectService,
    local_runner: FakeLocalRunner | None = None,
    cluster_runner: FakeClusterRunner | None = None,
    output_dir: Path,
) -> JobExecutionController:
    return JobExecutionController(
        document_getter=lambda: document,
        project_service=service,
        local_runner=local_runner or FakeLocalRunner(),
        cluster_runner=cluster_runner or FakeClusterRunner(),
        job_output_directory=lambda _job: output_dir,
        job_seed_name=lambda job: job.label,
    )


def _job(service: ProjectService, document: ProjectDocument, label: str = "job") -> UUID:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
    ).id


def test_local_submission_creates_run_and_records_submission(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _job(service, document, "local-job")
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
    job_id = _job(service, document, "local-retry")
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
    job_id = _job(service, document, "local-cancelled-retry")
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


def test_lifecycle_events_update_job_run_and_terminal_journal(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _job(service, document, "lifecycle-job")
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
    job_id = _job(service, document, "cluster-job")
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
    job_id = _job(service, document, "cluster-retry")
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
    job_id = _job(service, document, "cluster-retry-fail")
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
    job_id = _job(service, document, "cluster-cancelled-retry")
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
    job_id = _job(service, document, "bad-cluster-job")
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
    assert run is not None
    assert run.status == JobStatus.FAILED
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_failed"


def test_cluster_stopped_status_is_terminal_and_clears_active_run(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _job(service, document, "stopped-cluster-job")
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
    assert run is not None
    assert run.status == JobStatus.STOPPED
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_stopped"


# ---------------------------------------------------------------------------
# Monitor-metadata stamping (job-lifecycle hardening, rec #2).
# ---------------------------------------------------------------------------


def test_mark_queue_checked_stamps_observed_jobs(tmp_path: Path) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace
    from uuid import uuid4

    service, document = _document(tmp_path)
    observed_id = _job(service, document, "observed")
    unobserved_id = _job(service, document, "unobserved")
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
    job_id = _job(service, document, "untouched")
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
    job = document.jobs[_job(service, document, label)]
    job.execution_target = "scarf"
    job.status = JobStatus.RUNNING
    return job


def _local_running_job(service, document, label: str):
    job = document.jobs[_job(service, document, label)]
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

    local_job = document.jobs[_job(service, document, "local-start")]
    local_job.execution_target = "local"
    local_job.status = JobStatus.QUEUED
    controller.mark_started(local_job.id)
    assert local_job.status == JobStatus.RUNNING
    assert local_job.last_status_source == "local_runner"
    assert local_job.last_status_message == "process started"
    assert local_job.last_checked_at is not None

    # Cluster jobs are stamped from queue polls, not from mark_started.
    cluster_job = document.jobs[_job(service, document, "cluster-start")]
    cluster_job.execution_target = "scarf"
    cluster_job.status = JobStatus.QUEUED
    controller.mark_started(cluster_job.id)
    assert cluster_job.status == JobStatus.RUNNING
    assert cluster_job.last_checked_at is None
    assert cluster_job.last_status_source is None


def test_mark_local_status_changed_stamps_queued_locally(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = document.jobs[_job(service, document, "local-queued")]
    job.execution_target = "local"
    job.status = JobStatus.SUBMITTING

    update = controller.mark_local_status_changed(job.id, JobStatus.QUEUED.value)

    assert update.changed is True
    assert job.status == JobStatus.QUEUED
    assert job.last_status_source == "local_runner"
    assert job.last_status_message == "queued locally"
    assert job.last_checked_at is not None

    # An inert notification (wrong current status) does not stamp.
    other = document.jobs[_job(service, document, "local-inert")]
    other.execution_target = "local"
    update = controller.mark_local_status_changed(other.id, JobStatus.QUEUED.value)
    assert update.changed is False
    assert other.last_checked_at is None


def test_prepare_cluster_submission_forwards_mem_per_cpu_to_the_runner(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = _job(service, document, "mem-job")
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
        job = document.jobs[_job(service, document, label)]
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
    running = document.jobs[_job(service, document, "status-running")]
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
    job_id = _job(service, document, "no-mem-job")
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
