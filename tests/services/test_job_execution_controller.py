from __future__ import annotations

from dataclasses import replace

from mufinder.services.job_monitoring import (
    CANCEL_NOT_SENT_MESSAGE,
    CANCEL_NOT_SENT_SOURCE,
    awaits_cluster_answer,
    cancel_not_sent_to_cluster,
    job_status_word,
)
import os
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.cluster import ClusterSubmissionConfig
from mufinder.domain.models import CalculationRun
from mufinder.domain.enums import EndReason, JobActivity, JobStatus
from mufinder.domain.models import Result, utc_now
from mufinder.services.job_artifacts import ParsedJobArtifacts, RetrievalFlags
from mufinder.services.job_execution_controller import (
    LAST_CLUSTER_SUBMISSION_KEY,
    ClusterJobSubmission,
    JobExecutionController,
    LifecycleUpdate,
    LOCAL_SUBMISSION_CORES_KEY,
    remember_cluster_submission,
    remember_local_cores,
    remembered_cluster_submission,
    remembered_local_cores,
)
from mufinder.castep_io.output_parser import CastepResult
from mufinder.services.cluster_runner import CheckTarget
from mufinder.services.failure_diagnosis import failure_reason_summary
from mufinder.services.job_reconcile import (
    NO_LOCAL_PROCESS_MESSAGE,
    JobEvidence,
    OutputState,
    SchedulerState,
)
from mufinder.services.job_file_cleanup import JobFileCleanup
from mufinder.services.project_service import ProjectDocument, ProjectService
from tests.support.jobs import set_job_state
from tests.support.jobs import make_job


class FakeLocalRunner:
    #: Jobs the fake holds a queue entry or a process for.
    pending: frozenset = frozenset()
    active: frozenset = frozenset()

    def pending_job_ids(self) -> frozenset:
        return self.pending

    def active_job_ids(self) -> frozenset:
        return self.active

    def __init__(self, *, raise_on_submit: Exception | None = None) -> None:
        self.max_concurrent = 1
        self.submitted: list[dict[str, object]] = []
        self._raise_on_submit = raise_on_submit

    def submit(self, **kwargs: object) -> None:
        self.submitted.append(kwargs)
        if self._raise_on_submit is not None:
            raise self._raise_on_submit


class FakeClusterRunner:
    #: The cluster the fake is connected to; ``None`` when disconnected.
    connected_cluster_name: str | None = None

    @property
    def is_connected(self) -> bool:
        return self.connected_cluster_name is not None

    def is_submitting(self, job_id: object) -> bool:
        return job_id in self.submitting

    def read_geometry_progress(self, targets: list) -> None:
        self.progress_requests.append(list(targets))

    def check_jobs(
        self, targets: list, *, skip_if_busy: bool = False, refresh: bool = False
    ) -> bool:
        self.checks.append((list(targets), skip_if_busy))
        self.refreshes.append(refresh)
        return True

    def __init__(self, *, fail_message: str | None = None) -> None:
        self.submitting: set = set()
        self.checks: list[tuple[list, bool]] = []
        #: Whether each check was a Refresh Queue pass.
        self.refreshes: list[bool] = []
        self.progress_requests: list[list] = []
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
    assert job_status_word(job) == "submitting"
    assert job.execution_target == "local"
    assert job.num_processes == 4
    assert run is not None
    assert run.status == JobStatus.ACTIVE
    assert document.provenance[-1].action == "submit_jobs"


def test_local_resubmission_accepts_failed_job_and_creates_next_run(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("local-retry", service=service, document=document).id
    job = document.jobs[job_id]
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
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
        eligible_statuses={JobStatus.ENDED},
    )

    run = controller.run_for_job(job_id)
    assert outcome.submitted_count == 1
    assert runner.submitted[0]["job_id"] == job_id
    assert job_status_word(job) == "submitting"
    assert job.error_message is None
    assert run is not None
    assert run.run_index == 1
    assert run.status == JobStatus.ACTIVE


def test_local_resubmission_accepts_cancelled_job(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("local-cancelled-retry", service=service, document=document).id
    job = document.jobs[job_id]
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.CANCELLED
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
        eligible_statuses={JobStatus.ENDED},
    )

    assert outcome.submitted_count == 1
    assert runner.submitted[0]["job_id"] == job_id
    assert job_status_word(job) == "submitting"


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
    controller = _controller(
        document=document, service=service, local_runner=runner, output_dir=tmp_path
    )

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
    controller = _controller(
        document=document, service=service, local_runner=runner, output_dir=tmp_path
    )

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
    controller = _controller(
        document=document, service=service, local_runner=runner, output_dir=tmp_path
    )

    outcome = _submit_local(controller, [job], num_cores=2)

    assert outcome.submitted_count == 0
    assert outcome.failed_messages == ("- runner-error: runner exploded",)
    assert outcome.journal_changed is True
    assert job_status_word(job) == "failed"
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_failed"


def test_local_submission_forwards_task_farming_to_the_runner(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("task-farm", service=service, document=document)
    runner = FakeLocalRunner()
    controller = _controller(
        document=document, service=service, local_runner=runner, output_dir=tmp_path
    )

    outcome = _submit_local(
        controller, [job], num_cores=4, task_farming_for_job=lambda _job: (True, 3)
    )

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
    assert (
        controller.complete_job_with_result(
            job, result=result, warnings=[], convergence_achieved=None
        )
        is False
    )
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

    older = CalculationRun(job_id=job_id, run_index=1, status=JobStatus.ENDED)
    newer = CalculationRun(job_id=job_id, run_index=2, status=JobStatus.COMPLETED)
    document.runs[older.id] = older
    document.runs[newer.id] = newer

    assert controller.run_for_job(job_id) is newer


def test_run_for_job_prefers_the_active_run_passed_to_the_controller(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("active", service=service, document=document)
    active = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.ACTIVE)
    newer = CalculationRun(job_id=job.id, run_index=2, status=JobStatus.ENDED)
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

    run = controller.ensure_run_for_job(job, status=JobStatus.ACTIVE, started=True)

    assert run is not None
    assert run.status == JobStatus.ACTIVE
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

    queued = controller.mark_local_status_changed(job_id, "queued")
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
    assert job_status_word(job) == "queued"
    assert job.execution_target == "test-cluster"
    assert job.remote_job_id == "12345"
    assert job.remote_directory == "/remote/cluster-job"
    assert run is not None
    assert run.status == JobStatus.ACTIVE


def test_cluster_resubmission_reuses_remote_directory_and_updates_remote_job_id(
    tmp_path: Path,
) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("cluster-retry", service=service, document=document).id
    job = document.jobs[job_id]
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
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
        )
    )

    run = controller.run_for_job(job_id)
    assert outcome.submitted_job is job
    assert runner.submitted[0]["remote_directory"] == "/remote/existing/cluster-retry"
    assert job_status_word(job) == "queued"
    assert job.remote_job_id == "12345"
    assert job.remote_directory == "/remote/existing/cluster-retry"
    assert job.error_message is None
    assert run is not None
    assert run.status == JobStatus.ACTIVE


def test_cluster_resubmission_failure_keeps_existing_remote_context(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("cluster-retry-fail", service=service, document=document).id
    job = document.jobs[job_id]
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
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
        )
    )

    assert outcome.submitted_job is None
    assert outcome.failed_message == "- cluster-retry-fail: scheduler rejected retry"
    assert job_status_word(job) == "failed"
    assert job.remote_job_id == "old123"
    assert job.remote_directory == "/remote/existing/cluster-retry-fail"


def test_cluster_resubmission_accepts_cancelled_job(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job_id = make_job("cluster-cancelled-retry", service=service, document=document).id
    job = document.jobs[job_id]
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.CANCELLED
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
        )
    )

    assert outcome.submitted_job is job
    assert job_status_word(job) == "queued"
    assert job.remote_job_id == "12345"
    assert job.remote_directory == "/remote/existing/cluster-cancelled-retry"


def test_cluster_submission_failure_marks_job_failed_and_records_terminal_event(
    tmp_path: Path,
) -> None:
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
    assert job_status_word(job) == "failed"
    assert job.error_message == "scheduler rejected job"
    assert job.end_reason == EndReason.SUBMISSION
    assert run is not None
    assert run.status == JobStatus.ENDED
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_failed"


def test_cluster_submission_rejects_jobs_outside_the_eligible_statuses(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("cluster-ineligible", service=service, document=document)
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    runner = FakeClusterRunner()
    controller = _controller(
        document=document, service=service, cluster_runner=runner, output_dir=tmp_path
    )

    outcome = controller.submit_cluster_job(_cluster_submission(job, tmp_path))

    assert outcome.submitted_job is None
    assert runner.submitted == []
    assert job_status_word(job) == "running"


def _tracked_run(
    controller: JobExecutionController, document: ProjectDocument, job, status: JobStatus
):
    """Register an active run for ``job`` without a start time."""

    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.ACTIVE, started_at=None)
    document.runs[run.id] = run
    controller.active_run_by_job[job.id] = run.id
    return run


@pytest.mark.parametrize(
    "notify",
    [
        lambda controller, job_id: controller.mark_local_status_changed(job_id, "queued"),
        lambda controller, job_id: controller.mark_started(job_id),
        lambda controller, job_id: controller.apply_evidence(
            job_id, JobEvidence(checked_at=utc_now(), scheduler=SchedulerState.RUNNING)
        ),
        lambda controller, job_id: controller.fail_job(job_id, "boom"),
        lambda controller, job_id: controller.cancel_job(job_id),
    ],
    ids=["local-status", "started", "evidence", "fail", "cancel"],
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

    update = controller.mark_local_status_changed(job.id, "running")

    assert update.job is job
    assert update.changed is False
    assert job_status_word(job) == "submitting"


def test_mark_started_stamps_a_missing_run_start_time(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("start", service=service, document=document)
    service.submit_job(job)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    run = _tracked_run(controller, document, job, "submitting")

    update = controller.mark_started(job.id)

    assert update.changed is True
    assert job_status_word(job) == "running"
    assert run.status == JobStatus.ACTIVE
    assert run.started_at is not None


def test_complete_job_with_result_creates_a_run_when_none_exists(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    job = make_job("complete-create-run", service=service, document=document)
    service.submit_job(job)
    service.mark_job_active(job, JobActivity.RUNNING)
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
    service.mark_job_active(job, JobActivity.RUNNING)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    run = _tracked_run(controller, document, job, "running")

    update = controller.fail_job(job.id, "kaboom")

    assert update.changed is True
    assert update.journal_changed is True
    assert job_status_word(job) == "failed"
    assert run.status == JobStatus.ENDED
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
    run = _tracked_run(controller, document, job, "submitting")

    update = controller.cancel_job(job.id)

    assert update.changed is True
    assert update.journal_changed is True
    assert job_status_word(job) == "cancelled"
    assert run.status == JobStatus.ENDED
    assert run.completed_at is not None
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_cancelled"


def test_fail_job_diagnoses_castep_error_files(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _local_running_job(service, document, "castep-err")
    (tmp_path / "castep-err.0001.err").write_text(
        "Error in parameters: bad keyword\n", encoding="utf-8"
    )

    controller.fail_job(job.id, "CASTEP did not report successful completion.")

    assert job.end_reason == EndReason.CASTEP_ERROR
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

    assert job.end_reason == EndReason.UNKNOWN


def test_fail_job_keeps_an_explicit_reason(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _cluster_running_job(service, document, "explicit")

    controller.fail_job(
        job.id, "Scheduler reported the job as TIMEOUT.", reason=EndReason.SUBMISSION
    )

    assert job.end_reason == EndReason.SUBMISSION


# ---------------------------------------------------------------------------
# Monitor-metadata stamping (job-lifecycle hardening, rec #2).
# ---------------------------------------------------------------------------


def _cluster_running_job(service, document, label: str):
    job = make_job(label, service=service, document=document)
    job.execution_target = "scarf"
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    return job


def test_a_failure_the_logs_show_as_a_walltime_kill_is_journalled_as_stopped(
    tmp_path: Path,
) -> None:
    # It reads "stopped" on the dashboard, so the journal says so too.
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _local_running_job(service, document, "timed-out")
    (tmp_path / "slurm-1.out").write_text(
        "slurmstepd: error: *** JOB 1 CANCELLED DUE TO TIME LIMIT ***\n", encoding="utf-8"
    )
    job.remote_job_id = "1"

    controller.fail_job(job.id, "CASTEP did not report successful completion.")

    assert job.end_reason == EndReason.WALLTIME
    assert job_status_word(job) == "stopped"
    assert document.provenance[-1].action == "job_stopped"


def test_a_failure_with_no_cause_in_the_logs_is_journalled_as_failed(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _local_running_job(service, document, "plain")

    controller.fail_job(job.id, "process exited with code 1")

    assert job.end_reason == EndReason.UNKNOWN
    assert document.provenance[-1].action == "job_failed"


def _local_running_job(service, document, label: str):
    job = make_job(label, service=service, document=document)
    job.execution_target = "local"
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    return job


def test_complete_job_with_result_stamps_retrieval_for_cluster_jobs(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _cluster_running_job(service, document, "cluster-done")
    result = Result(result_type="castep_cutoff", data={"total_energy": -1.0})

    controller.complete_job_with_result(job, result=result, warnings=[], convergence_achieved=True)

    assert job.status == JobStatus.COMPLETED
    assert job.last_status_source == "retrieval"
    assert job.last_status_message == "output retrieved and parsed"
    assert job.last_checked_at is not None


def test_complete_job_with_result_stamps_local_runner_for_local_jobs(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _local_running_job(service, document, "local-done")
    result = Result(result_type="castep_cutoff", data={"total_energy": -1.0})

    controller.complete_job_with_result(job, result=result, warnings=[], convergence_achieved=True)

    assert job.status == JobStatus.COMPLETED
    assert job.last_status_source == "local_runner"
    assert job.last_status_message == "process completed"
    assert job.last_checked_at is not None


def test_fail_job_stamps_retrieval_for_cluster_jobs(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = _cluster_running_job(service, document, "cluster-broken")

    controller.fail_job(job.id, "Failed to retrieve or parse remote output: boom")

    assert job_status_word(job) == "failed"
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

    assert job_status_word(job) == "failed"
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
        "detected an impossible state",
        source="user",
        status_message="impossible state detected",
    )

    assert job.last_status_source == "user"
    assert job.last_status_message == "impossible state detected"


def test_mark_started_stamps_local_jobs_only(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)

    local_job = make_job("local-start", service=service, document=document)
    local_job.execution_target = "local"
    local_job.status = JobStatus.ACTIVE
    local_job.activity = JobActivity.QUEUED
    controller.mark_started(local_job.id)
    assert job_status_word(local_job) == "running"
    assert local_job.last_status_source == "local_runner"
    assert local_job.last_status_message == "process started"
    assert local_job.last_checked_at is not None

    # Cluster jobs are stamped from queue polls, not from mark_started.
    cluster_job = make_job("cluster-start", service=service, document=document)
    cluster_job.execution_target = "scarf"
    cluster_job.status = JobStatus.ACTIVE
    cluster_job.activity = JobActivity.QUEUED
    controller.mark_started(cluster_job.id)
    assert job_status_word(cluster_job) == "running"
    assert cluster_job.last_checked_at is None
    assert cluster_job.last_status_source is None


def test_mark_local_status_changed_stamps_queued_locally(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = make_job("local-queued", service=service, document=document)
    job.execution_target = "local"
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.SUBMITTING

    update = controller.mark_local_status_changed(job.id, "queued")

    assert update.changed is True
    assert job_status_word(job) == "queued"
    assert job.last_status_source == "local_runner"
    assert job.last_status_message == "queued locally"
    assert job.last_checked_at is not None

    # An inert notification (wrong current status) does not stamp.
    other = make_job("local-inert", service=service, document=document)
    other.execution_target = "local"
    update = controller.mark_local_status_changed(other.id, "queued")
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

    for status in (JobStatus.DEFINED, "failed", "stopped", "cancelled"):
        label = f"status-{status}"
        job = make_job(label, service=service, document=document)
        set_job_state(job, status)
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

        assert prepared is not None, f"{status} should be submittable to a cluster"
        assert job_status_word(job) == "submitting"

    # Statuses with an in-flight or finished run are still refused.
    running = make_job("status-running", service=service, document=document)
    running.status = JobStatus.ACTIVE
    running.activity = JobActivity.RUNNING
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
    assert job_status_word(running) == "running"


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
        "processes_per_node": None,
        "castep_executable": "",
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
    assert (
        RetrievalFlags(**{name: runner.submitted[0][name] for name in flags.as_kwargs()}) == flags
    )


def test_local_cores_are_remembered_in_the_project_ui_state() -> None:
    ui_state: dict[str, object] = {}
    assert remembered_local_cores(ui_state) is None

    remember_local_cores(ui_state, 6)

    assert ui_state[LOCAL_SUBMISSION_CORES_KEY] == 6
    assert remembered_local_cores(ui_state) == 6


@pytest.mark.parametrize("saved", [0, -2, "8", 2.5, True, None])
def test_remembered_local_cores_ignores_a_saved_value_that_is_not_a_core_count(saved) -> None:
    assert remembered_local_cores({LOCAL_SUBMISSION_CORES_KEY: saved}) is None


def _submission(**overrides: object) -> ClusterSubmissionConfig:
    values: dict[str, object] = {
        "cluster_name": "SCARF",
        "working_directory": "/work4/isis/alice/project_z",
        "username": "alice",
        "num_processes": 144,
        "walltime": "12:00:00",
    }
    values.update(overrides)
    return ClusterSubmissionConfig(**values)  # type: ignore[arg-type]


def test_the_last_cluster_submission_is_remembered_in_the_project_ui_state() -> None:
    ui_state: dict[str, object] = {}
    assert remembered_cluster_submission(ui_state) is None
    config = _submission(
        queue_name="scarf",
        memory_gb=64.0,
        mem_per_cpu_gb=6.0,
        comment="first pass",
        processes_per_node=32,
    )

    remember_cluster_submission(ui_state, config, workflow_type="muon_relaxation/positivemuon")

    saved = ui_state[LAST_CLUSTER_SUBMISSION_KEY]
    assert isinstance(saved, dict)
    assert saved["workflow_type"] == "muon_relaxation/positivemuon"
    assert saved["mem_per_cpu_gb"] == 6.0
    assert saved["processes_per_node"] == 32
    assert (
        remembered_cluster_submission(ui_state, workflow_type="muon_relaxation/positivemuon")
        == config
    )


def test_a_remembered_submission_keeps_its_folder_only_for_the_same_workflow() -> None:
    ui_state: dict[str, object] = {}
    remember_cluster_submission(ui_state, _submission(), workflow_type="convergence")

    other = remembered_cluster_submission(ui_state, workflow_type="muon_relaxation")
    unnamed = remembered_cluster_submission(ui_state)

    assert other is not None and other.working_directory == ""
    assert other.cluster_name == "SCARF" and other.num_processes == 144
    assert unnamed is not None and unnamed.working_directory == "/work4/isis/alice/project_z"


@pytest.mark.parametrize("saved", ["not a mapping", {}, {"cluster_name": "  "}])
def test_no_submission_is_remembered_without_a_cluster_name(saved: object) -> None:
    assert remembered_cluster_submission({LAST_CLUSTER_SUBMISSION_KEY: saved}) is None


@pytest.mark.parametrize("stored", [None, "", "bad-memory", 0, -2.0])
def test_a_remembered_mem_per_cpu_that_cannot_be_used_is_unset(stored: object) -> None:
    ui_state = {
        LAST_CLUSTER_SUBMISSION_KEY: {
            "cluster_name": "SCARF",
            "num_processes": 144,
            "mem_per_cpu_gb": stored,
        }
    }

    restored = remembered_cluster_submission(ui_state)

    assert restored is not None
    assert restored.mem_per_cpu_gb is None


@pytest.mark.parametrize("stored", [None, "", "two", 0, -4])
def test_a_remembered_processes_per_node_that_cannot_be_used_is_unset(stored: object) -> None:
    payload: dict[str, object] = {"cluster_name": "Redwood", "num_processes": 128}
    if stored is not None:
        payload["processes_per_node"] = stored

    restored = remembered_cluster_submission({LAST_CLUSTER_SUBMISSION_KEY: payload})

    assert restored is not None
    assert restored.processes_per_node is None


def test_unreadable_remembered_numbers_fall_back_to_their_defaults() -> None:
    restored = remembered_cluster_submission(
        {
            LAST_CLUSTER_SUBMISSION_KEY: {
                "cluster_name": "SCARF",
                "num_processes": "many",
                "memory_gb": "lots",
            }
        }
    )

    assert restored is not None
    assert restored.num_processes == 1
    assert restored.memory_gb is None


# --- Resetting ended jobs --------------------------------------------------------


def _reset_setup(
    tmp_path: Path, labels: tuple[str, ...]
) -> tuple[ProjectService, ProjectDocument, JobExecutionController, JobFileCleanup, dict]:
    """A project with one failed job per label, each with a folder holding output."""

    service, document = _document(tmp_path)
    folders: dict[str, Path] = {}
    for label in labels:
        job = make_job(label, service=service, document=document)
        set_job_state(job, "failed")
        folder = document.project_root / "files" / "convergence_cutoff" / label
        folder.mkdir(parents=True)
        (folder / f"{label}.castep").write_text("old output", encoding="utf-8")
        folders[label] = folder
    controller = JobExecutionController(
        document_getter=lambda: document,
        project_service=service,
        local_runner=FakeLocalRunner(),
        cluster_runner=FakeClusterRunner(),
        job_output_directory=lambda job: folders.get(job.label),
        job_seed_name=lambda job: job.label,
    )
    cleanup = JobFileCleanup(
        document_getter=lambda: document,
        job_output_directory=lambda job: folders.get(job.label),
        job_seed_name=lambda job: job.label,
    )
    return service, document, controller, cleanup, folders


def _jobs_by_label(document: ProjectDocument) -> dict[str, object]:
    return {job.label: job for job in document.jobs.values()}


def test_resetting_ended_jobs_records_each_reset_and_touches_no_file(tmp_path: Path) -> None:
    _service, document, controller, _cleanup, folders = _reset_setup(tmp_path, ("a", "b"))
    jobs = _jobs_by_label(document)

    reset = controller.reset_ended_jobs(list(jobs.values()), cleaned_job_ids={jobs["a"].id})

    assert set(reset) == {job.id for job in jobs.values()}
    assert all(job.status is JobStatus.DEFINED for job in jobs.values())
    # The caller clears the folders of jobs that start again; this only changes state.
    assert all((folder / f"{label}.castep").is_file() for label, folder in folders.items())
    cleaned = {
        record.parameters["job_label"]: record.parameters["cleaned_output_directory"]
        for record in document.provenance
        if record.action == "job_reset"
    }
    assert cleaned == {"a": True, "b": False}


def test_continuing_ended_jobs_keeps_their_folders_and_records_each_continuation(
    tmp_path: Path,
) -> None:
    """Continue returns ended jobs to defined with their files and remote folder kept.

    The kept ``remote_directory`` is what sends the next cluster submission into
    the same folder, where the checkpoint is; the imported job and the active
    one are left alone.
    """

    from mufinder.domain.constants import IMPORTED_JOB_TAG
    from mufinder.services.cluster_submission_planner import plan_remote_targets

    service, document, controller, _cleanup, folders = _reset_setup(
        tmp_path, ("a", "imported", "active")
    )
    jobs = _jobs_by_label(document)
    for label, job in jobs.items():
        job.execution_target = "alpha"
        job.remote_directory = f"/remote/work/{label}"
        job.remote_job_id = "4242"
    jobs["imported"].tags.append(IMPORTED_JOB_TAG)
    set_job_state(jobs["active"], "running")

    continued = controller.continue_ended_jobs(list(jobs.values()))

    assert continued == (jobs["a"].id,)
    job = jobs["a"]
    assert job.status is JobStatus.DEFINED
    assert (job.remote_directory, job.remote_job_id, job.end_reason) == (
        "/remote/work/a",
        None,
        None,
    )
    assert (folders["a"] / "a.castep").is_file()
    assert job_status_word(jobs["imported"]) == "failed"
    assert job_status_word(jobs["active"]) == "running"
    assert [record.action for record in document.provenance].count("job_continuation") == 1
    # On its own cluster the next submission writes the rebuilt inputs into the kept folder.
    (target,) = plan_remote_targets(
        [job],
        cluster_name="alpha",
        working_directory="/remote/work",
        seed_name_by_job={job.id: "a"},
        remote_job_directory=lambda **_kwargs: pytest.fail("the kept folder is reused"),
    ).targets
    assert (target.remote_target, target.allow_existing_remote_directory) == (
        "/remote/work/a",
        True,
    )


def test_resetting_a_completed_job_removes_its_results_and_records_it(tmp_path: Path) -> None:
    """An edited completed row's job runs again without its result; others are left alone."""

    from mufinder.domain.models import CalculationRun, Result
    from mufinder.services.journal import build_journal

    _service, document, controller, _cleanup, folders = _reset_setup(
        tmp_path, ("done", "failed", "imported")
    )
    jobs = _jobs_by_label(document)
    for label in ("done", "imported"):
        set_job_state(jobs[label], "completed")
    jobs["imported"].tags.append("imported")
    run = CalculationRun(job_id=jobs["done"].id, status=JobStatus.COMPLETED)
    result = Result(calculation_run_id=run.id, data={"muon_zpe_total_eV": 0.5})
    orphan = Result(calculation_run_id=run.id)
    run.result_id = result.id
    document.runs[run.id] = run
    document.results[result.id] = result
    document.results[orphan.id] = orphan

    reset = controller.reset_completed_jobs(list(jobs.values()), cleaned_job_ids={jobs["done"].id})

    assert reset == (jobs["done"].id,)
    assert jobs["done"].status is JobStatus.DEFINED
    assert jobs["done"].completed_at is None
    assert result.id not in document.results and orphan.id not in document.results
    assert run.result_id is None and run.id in document.runs
    assert job_status_word(jobs["failed"]) == "failed"
    assert job_status_word(jobs["imported"]) == "completed"
    # The caller clears the folder; this only changes state and records.
    assert (folders["done"] / "done.castep").is_file()
    (record,) = [record for record in document.provenance if record.action == "job_reset"]
    assert record.parameters["removed_result"] is True
    assert record.parameters["cleaned_output_directory"] is True
    (entry,) = build_journal("Rerun", [record]).days[0].entries
    assert entry.body.endswith(
        "and cleaned its directory. Its previous result was removed, to be replaced by the new run."
    )


def test_an_imported_job_is_never_reset(tmp_path: Path) -> None:
    _service, document, controller, cleanup, folders = _reset_setup(tmp_path, ("a",))
    job = _jobs_by_label(document)["a"]
    job.tags = [*job.tags, "imported"]

    reset = controller.reset_ended_jobs([job])

    assert reset == ()
    assert job_status_word(job) == "failed"
    assert (folders["a"] / "a.castep").is_file()


# --- Local jobs checked against the runner and their output folder -------------

_COMPLETE_OUTPUT = Path(__file__).parents[1] / "fixtures" / "uep" / "Si2.castep"


def _active_local_job(service, document, label: str, status: JobStatus):
    job = make_job(label, service=service, document=document, execution_target="local")
    service.submit_job(job)
    if status in {"queued", "running"}:
        service.mark_job_active(job, JobActivity.QUEUED)
    if status == "running":
        service.mark_job_active(job, JobActivity.RUNNING)
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.ACTIVE)
    document.runs[run.id] = run
    return job, run


@pytest.mark.parametrize("status", ["submitting", "queued", "running"])
def test_local_job_with_no_process_and_no_output_is_stopped(tmp_path: Path, status) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job, run = _active_local_job(service, document, "orphan", status)
    controller.active_run_by_job[job.id] = run.id

    updates = controller.check_local_jobs()

    assert [update.job for update in updates] == [job]
    assert updates[0].changed and updates[0].journal_changed
    assert job_status_word(job) == "stopped"
    assert job.end_reason == EndReason.INTERRUPTED
    assert job.error_message == NO_LOCAL_PROCESS_MESSAGE
    assert job.last_status_source == "reconciliation"
    assert job.last_checked_at is not None
    assert run.status == JobStatus.ENDED and run.completed_at is not None
    assert job.id not in controller.active_run_by_job
    assert [record.action for record in document.provenance][-1] == "job_stopped"


def test_local_job_the_runner_holds_is_left_alone(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    runner = FakeLocalRunner()
    controller = _controller(
        document=document, service=service, local_runner=runner, output_dir=tmp_path
    )
    running, _ = _active_local_job(service, document, "running", "running")
    queued, _ = _active_local_job(service, document, "queued", "queued")
    runner.active = frozenset({running.id})
    runner.pending = frozenset({queued.id})

    assert controller.check_local_jobs() == []
    assert (job_status_word(running), job_status_word(queued)) == ("running", "queued")


def test_local_job_with_no_process_and_complete_output_is_reported_finished(
    tmp_path: Path,
) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job, _ = _active_local_job(service, document, "finished", "running")
    (tmp_path / "finished.castep").write_bytes(_COMPLETE_OUTPUT.read_bytes())

    updates = controller.check_local_jobs()

    assert len(updates) == 1
    assert updates[0].finished is not None and updates[0].finished.parsed.completed
    # The caller records the result and completes the job.
    assert not updates[0].changed
    assert job_status_word(job) == "running"


def test_local_check_leaves_cluster_and_inactive_jobs_alone(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    cluster = _cluster_running_job(service, document, "on-cluster")
    done = make_job("done", service=service, document=document, execution_target="local")
    done.status = JobStatus.COMPLETED
    failed = make_job("failed", service=service, document=document, execution_target="local")
    failed.status = JobStatus.ENDED
    failed.end_reason = EndReason.UNKNOWN

    assert controller.check_local_jobs() == []
    assert [job_status_word(job) for job in (cluster, done, failed)] == [
        "running",
        "completed",
        "failed",
    ]


@pytest.mark.parametrize("target", ["", "  ", "LOCAL"])
def test_local_check_treats_a_blank_or_any_case_local_target_as_local(
    tmp_path: Path, target: str
) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job, _ = _active_local_job(service, document, "orphan", "running")
    job.execution_target = target

    controller.check_local_jobs()

    assert job_status_word(job) == "stopped"


def test_stopping_a_local_job_does_not_reopen_a_run_that_already_ended(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job, run = _active_local_job(service, document, "orphan", "running")
    run.status = JobStatus.COMPLETED

    controller.check_local_jobs()

    assert job_status_word(job) == "stopped"
    assert run.status == JobStatus.COMPLETED and run.completed_at is None


def test_local_check_without_a_document_does_nothing(tmp_path: Path) -> None:
    controller = _controller(document=None, service=ProjectService(), output_dir=tmp_path)

    assert controller.check_local_jobs() == []


def _interrupted_local_job(service, document, label: str, *, submitted_ago: float = 60.0):
    """Return a local job MuFinder stopped as ``interrupted``, submitted ``submitted_ago`` s ago."""

    job, run = _active_local_job(service, document, label, "running")
    job.submitted_at = utc_now() - timedelta(seconds=submitted_ago)
    service.end_job(job, EndReason.INTERRUPTED, NO_LOCAL_PROCESS_MESSAGE)
    run.status = JobStatus.ENDED
    return job, run


def test_project_open_check_completes_an_interrupted_local_job_from_its_own_output(
    tmp_path: Path,
) -> None:
    # A CASTEP process that outlived MuFinder finished after the job was stopped.
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job, ended_run = _interrupted_local_job(service, document, "outlived")
    job.dismissed_at = utc_now()
    (tmp_path / "outlived.castep").write_bytes(_COMPLETE_OUTPUT.read_bytes())

    assert controller.check_local_jobs() == []
    assert job_status_word(job) == "stopped"

    updates = controller.check_local_jobs(include_interrupted=True)

    assert [update.job for update in updates] == [job]
    assert updates[0].finished is not None and updates[0].finished.parsed.completed
    # Made active again under a fresh run, so the caller can record the result.
    assert job_status_word(job) == "retrieving" and job.dismissed_at is None
    fresh_run = controller.run_for_job(job.id)
    assert fresh_run is not None and fresh_run is not ended_run


def test_project_open_check_leaves_an_interrupted_local_job_with_stale_output(
    tmp_path: Path,
) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job, _run = _interrupted_local_job(service, document, "stale", submitted_ago=0.0)
    output = tmp_path / "stale.castep"
    output.write_bytes(_COMPLETE_OUTPUT.read_bytes())
    # Written by an earlier run in the same folder, an hour before this submission.
    earlier = (utc_now() - timedelta(hours=1)).timestamp()
    os.utime(output, (earlier, earlier))

    assert controller.check_local_jobs(include_interrupted=True) == []
    assert job_status_word(job) == "stopped"
    assert job.end_reason == EndReason.INTERRUPTED


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_project_open_check_reads_only_interrupted_ended_local_jobs(
    tmp_path: Path, status: str
) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = make_job("other", service=service, document=document, execution_target="local")
    set_job_state(job, status)
    (tmp_path / "other.castep").write_bytes(_COMPLETE_OUTPUT.read_bytes())

    assert controller.check_local_jobs(include_interrupted=True) == []
    assert job_status_word(job) == status


# --- Cluster jobs: applying what a check saw ----------------------------------

_REMOTE_ID = "4242"
_REMOTE_DIR = "/remote/work/job"
_COMPLETE = CastepResult(completed=True, total_energy=-1.0)


def _cluster_setup(tmp_path: Path, status: JobStatus, *, connected: str | None = "scarf"):
    """Return ``(controller, runner, document, job, run)`` for one cluster job in ``status``."""

    service, document = _document(tmp_path)
    runner = FakeClusterRunner()
    runner.connected_cluster_name = connected
    controller = _controller(
        document=document, service=service, cluster_runner=runner, output_dir=tmp_path
    )
    job = make_job("on-cluster", service=service, document=document, execution_target="scarf")
    set_job_state(job, status)
    job.remote_job_id = _REMOTE_ID
    job.remote_directory = _REMOTE_DIR
    run = _tracked_run(controller, document, job, status)
    return controller, runner, document, job, run


def _seen(scheduler: SchedulerState, output: OutputState = OutputState.NOT_LOOKED, **fields):
    fields.setdefault("remote_job_id", _REMOTE_ID)
    fields.setdefault("remote_directory", _REMOTE_DIR)
    return JobEvidence(checked_at=utc_now(), scheduler=scheduler, output=output, **fields)


def _cancelled_unsent(tmp_path: Path, *, remote_job_id: str | None = _REMOTE_ID):
    controller, runner, document, job, run = _cluster_setup(tmp_path, "running", connected=None)
    job.remote_job_id = remote_job_id
    controller.cancel_job(job.id)
    assert controller.mark_cancel_not_sent(job.id) is True
    return controller, runner, document, job, run


def test_a_cancel_in_the_project_only_is_stamped_as_not_sent(tmp_path: Path) -> None:
    controller, _runner, _document_, job, _run = _cancelled_unsent(tmp_path)

    assert job_status_word(job) == "cancelled"
    assert (job.last_status_source, job.last_status_message) == (
        CANCEL_NOT_SENT_SOURCE,
        CANCEL_NOT_SENT_MESSAGE,
    )
    # Nothing was checked, so the time of the last check is left alone.
    assert job.last_checked_at is None
    assert cancel_not_sent_to_cluster(job)


def test_only_a_cancelled_cluster_job_is_stamped_as_not_sent(tmp_path: Path) -> None:
    controller, _runner, _document_, job, _run = _cluster_setup(tmp_path, "running")

    assert controller.mark_cancel_not_sent(job.id) is False
    assert controller.mark_cancel_not_sent(uuid4()) is False
    assert job.last_status_source is None


def test_a_late_submission_whose_cancel_failed_records_its_scheduler_job_id(
    tmp_path: Path,
) -> None:
    controller, _runner, _document_, job, _run = _cancelled_unsent(tmp_path, remote_job_id=None)

    assert controller.mark_cancel_not_sent(job.id, " 9001 ") is True
    assert job.remote_job_id == "9001"
    # A recorded ID is never replaced.
    controller.mark_cancel_not_sent(job.id, "9002")
    assert job.remote_job_id == "9001"


def _awaiting_answer(tmp_path: Path, kind: str, remote_job_id: str | None):
    """Return ``(controller, job)`` for an ended job awaiting a check's answer."""

    if kind == "unsent-cancel":
        controller, _runner, _document_, job, _run = _cancelled_unsent(
            tmp_path, remote_job_id=remote_job_id
        )
        return controller, job
    controller, _runner, _document_, job, _run = _cluster_setup(tmp_path, "submitting")
    job.remote_job_id = remote_job_id
    controller.cancel_job(job.id)
    controller.note_submission_cut_off(job.id)
    return controller, job


_AWAITING_KINDS = pytest.mark.parametrize("kind", ["unsent-cancel", "cut-off"])


@_AWAITING_KINDS
@pytest.mark.parametrize(
    ("remote_job_id", "evidence"),
    [
        (_REMOTE_ID, _seen(SchedulerState.UNAVAILABLE, scheduler_detail="timeout")),
        (None, _seen(SchedulerState.NO_JOB_ID, OutputState.UNREACHABLE, remote_job_id="")),
        (None, _seen(SchedulerState.NO_JOB_ID, remote_job_id="")),
        # A folder without complete output does not show the job is not queued.
        (None, _seen(SchedulerState.NO_JOB_ID, OutputState.ABSENT, remote_job_id="")),
        (None, _seen(SchedulerState.NO_JOB_ID, OutputState.INCOMPLETE, remote_job_id="")),
    ],
    ids=[
        "scheduler-unavailable",
        "no-id-folder-unreachable",
        "no-id-folder-not-read",
        "no-id-folder-empty",
        "no-id-output-incomplete",
    ],
)
def test_a_check_with_no_answer_keeps_the_question_open(
    tmp_path: Path, kind, remote_job_id, evidence
) -> None:
    controller, job = _awaiting_answer(tmp_path, kind, remote_job_id)
    stamp = (job.last_status_source, job.last_status_message)

    controller.apply_evidence(job.id, replace(evidence, on_request=True))

    assert job_status_word(job) == "cancelled"
    assert awaits_cluster_answer(job)
    assert (job.last_status_source, job.last_status_message) == stamp


@_AWAITING_KINDS
@pytest.mark.parametrize(
    "evidence",
    [
        _seen(SchedulerState.NOT_LISTED, OutputState.ABSENT),
        _seen(SchedulerState.CANCELLED, OutputState.ABSENT),
        _seen(SchedulerState.COMPLETED, OutputState.UNREACHABLE),
        # The scheduler answered though the folder was not read.
        _seen(SchedulerState.NOT_LISTED),
    ],
    ids=["not-listed", "cancelled", "finished-folder-unreachable", "not-listed-folder-not-read"],
)
def test_a_check_the_scheduler_answers_settles_the_question(tmp_path: Path, kind, evidence) -> None:
    controller, job = _awaiting_answer(tmp_path, kind, _REMOTE_ID)

    controller.apply_evidence(job.id, replace(evidence, on_request=True))

    assert job_status_word(job) == "cancelled"
    assert not awaits_cluster_answer(job)
    assert job.last_status_source in {"scheduler_poll", "retrieval"}


def test_evidence_about_an_unknown_job_changes_nothing(tmp_path: Path) -> None:
    controller, _runner, _document_, _job, _run = _cluster_setup(tmp_path, "queued")

    assert controller.apply_evidence(uuid4(), _seen(SchedulerState.RUNNING)) == LifecycleUpdate()


def test_scheduler_running_starts_a_queued_cluster_job(tmp_path: Path) -> None:
    controller, _runner, _document_, job, run = _cluster_setup(tmp_path, "queued")
    evidence = _seen(SchedulerState.RUNNING)

    update = controller.apply_evidence(job.id, evidence)

    assert update.changed
    assert controller.take_check_summary(1) == "Cluster job started running: on-cluster"
    assert job_status_word(job) == "running" and job.started_at is not None
    assert run.status == JobStatus.ACTIVE and run.started_at is not None
    assert job.last_checked_at == evidence.checked_at
    assert (job.last_status_source, job.last_status_message) == (
        "scheduler_poll",
        "scheduler reports RUNNING",
    )


def test_an_unchanged_job_still_records_that_it_was_checked(tmp_path: Path) -> None:
    controller, _runner, _document_, job, _run = _cluster_setup(tmp_path, "running")
    evidence = _seen(SchedulerState.RUNNING)

    update = controller.apply_evidence(job.id, evidence)

    assert (update.changed, update.stamped) == (False, True)
    assert controller.take_check_summary(1) == ""
    assert job.last_checked_at == evidence.checked_at


def test_scheduler_pending_queues_a_submitting_cluster_job(tmp_path: Path) -> None:
    controller, _runner, _document_, job, run = _cluster_setup(tmp_path, "submitting")

    update = controller.apply_evidence(job.id, _seen(SchedulerState.PENDING))

    assert update.changed
    assert (job_status_word(job), run.status) == ("queued", JobStatus.ACTIVE)
    assert job.remote_job_id == _REMOTE_ID


@pytest.mark.parametrize(
    ("status", "expected"),
    [("running", "retrieving"), ("queued", "retrieving")],
)
def test_a_finished_job_reads_as_retrieving_while_its_output_is_fetched(
    tmp_path: Path, status: str, expected: str
) -> None:
    # Any active job the scheduler reports finished reads as retrieving.
    controller, _runner, _document_, job, run = _cluster_setup(tmp_path, status)

    update = controller.apply_evidence(job.id, _seen(SchedulerState.COMPLETED))

    assert update.changed is (expected != status)
    assert (job_status_word(job), run.status) == (expected, JobStatus.ACTIVE)


@pytest.mark.parametrize("status", ["queued", "running", "retrieving"])
def test_complete_output_is_handed_back_for_the_result_to_be_recorded(
    tmp_path: Path, status: JobStatus
) -> None:
    controller, _runner, _document_, job, _run = _cluster_setup(tmp_path, status)
    artifacts = ParsedJobArtifacts()
    evidence = _seen(
        SchedulerState.NOT_LISTED, OutputState.COMPLETE, parsed=_COMPLETE, artifacts=artifacts
    )

    update = controller.apply_evidence(job.id, evidence)

    assert update.finished is not None
    assert update.finished.parsed is _COMPLETE and update.finished.artifacts is artifacts
    assert not update.changed
    assert job_status_word(job) == status


@pytest.mark.parametrize("status", ["submitting", "queued", "running"])
def test_a_walltime_stop_ends_the_job_from_any_active_status(
    tmp_path: Path, status: JobStatus
) -> None:
    # The job may have run and hit its walltime between two checks.
    controller, _runner, document, job, run = _cluster_setup(tmp_path, status)

    update = controller.apply_evidence(
        job.id, _seen(SchedulerState.TIMED_OUT, OutputState.INCOMPLETE)
    )

    assert update.changed and update.journal_changed
    assert controller.take_check_summary(1) == (
        "Cluster job stopped at its walltime limit: on-cluster"
    )
    assert job_status_word(job) == "stopped"
    assert job.end_reason == EndReason.WALLTIME
    assert job.error_message == failure_reason_summary(EndReason.WALLTIME)
    assert run.status == JobStatus.ENDED and run.completed_at is not None
    assert controller.active_run_by_job == {}
    assert document.provenance[-1].action == "job_stopped"


def test_a_scheduler_failure_fails_the_job_with_the_scheduler_state(tmp_path: Path) -> None:
    controller, _runner, document, job, run = _cluster_setup(tmp_path, "running")

    update = controller.apply_evidence(
        job.id,
        _seen(SchedulerState.FAILED, OutputState.ABSENT, scheduler_detail="NODE_FAIL"),
    )

    assert update.changed
    assert controller.take_check_summary(1) == "Cluster job failed (node fail): on-cluster"
    assert job_status_word(job) == "failed"
    assert job.error_message == "Scheduler reported the job as NODE_FAIL."
    assert job.end_reason == EndReason.NODE_FAILURE
    assert job.last_status_source == "retrieval"
    assert run.status == JobStatus.ENDED
    assert document.provenance[-1].action == "job_failed"


def test_output_that_cannot_be_parsed_fails_the_job_as_output_unavailable(tmp_path: Path) -> None:
    controller, _runner, _document_, job, _run = _cluster_setup(tmp_path, "running")

    controller.apply_evidence(
        job.id, _seen(SchedulerState.COMPLETED, OutputState.UNREADABLE, output_detail="boom")
    )

    assert job_status_word(job) == "failed"
    assert job.end_reason == EndReason.OUTPUT_UNAVAILABLE
    assert job.error_message == "Failed to retrieve or parse remote output: boom"


def test_scheduler_cancelled_cancels_the_job(tmp_path: Path) -> None:
    controller, _runner, document, job, run = _cluster_setup(tmp_path, "running")

    update = controller.apply_evidence(job.id, _seen(SchedulerState.CANCELLED))

    assert update.changed
    assert controller.take_check_summary(1) == "Cluster job cancelled: on-cluster"
    assert (job_status_word(job), run.status) == ("cancelled", JobStatus.ENDED)
    assert document.provenance[-1].action == "job_cancelled"


@pytest.mark.parametrize("status", ["submitting", "queued", "running"])
def test_a_job_the_scheduler_lost_stays_active_and_carries_a_marker(
    tmp_path: Path, status: JobStatus
) -> None:
    controller, _runner, document, job, run = _cluster_setup(tmp_path, status)
    records = len(document.provenance)

    update = controller.apply_evidence(job.id, _seen(SchedulerState.NOT_LISTED, OutputState.ABSENT))

    assert (update.changed, update.stamped) == (False, True)
    assert (job_status_word(job), run.status) == (status, JobStatus.ACTIVE)
    assert job.end_reason is None and job.error_message is None
    assert len(document.provenance) == records
    marker = controller.staleness(job, now=utc_now())
    assert marker is not None and marker.startswith("lost track: the scheduler does not list")

    # The scheduler lists it again: the marker goes with the next check.
    controller.apply_evidence(job.id, _seen(SchedulerState.RUNNING))
    assert controller.staleness(job, now=utc_now()) is None


def test_a_job_still_being_submitted_is_left_alone_and_is_not_stale(tmp_path: Path) -> None:
    controller, runner, _document_, job, _run = _cluster_setup(tmp_path, "submitting")
    job.remote_job_id = None
    runner.submitting.add(job.id)

    update = controller.apply_evidence(
        job.id, _seen(SchedulerState.FAILED, OutputState.ABSENT, remote_job_id="")
    )

    assert update == LifecycleUpdate(job=job)
    assert job_status_word(job) == "submitting"
    assert controller.staleness(job, now=utc_now()) is None

    runner.submitting.clear()
    assert controller.staleness(job, now=utc_now()) == "no scheduler job ID was recorded"


def test_evidence_about_an_earlier_submission_is_dropped(tmp_path: Path) -> None:
    controller, _runner, _document_, job, _run = _cluster_setup(tmp_path, "queued")

    update = controller.apply_evidence(
        job.id, _seen(SchedulerState.FAILED, OutputState.ABSENT, remote_job_id="1111")
    )

    assert update == LifecycleUpdate(job=job)
    assert job_status_word(job) == "queued" and job.last_checked_at is None


def test_staleness_reads_disconnected_when_another_cluster_is_connected(tmp_path: Path) -> None:
    controller, runner, _document_, job, _run = _cluster_setup(tmp_path, "running")
    runner.connected_cluster_name = "other"

    assert controller.staleness(job, now=utc_now()) == "cluster disconnected"


# --- Ended cluster jobs the user asks to have checked --------------------------


def _ended_cluster_setup(tmp_path: Path, status: str = "failed"):
    controller, runner, document, job, run = _cluster_setup(tmp_path, "running")
    if status == "cancelled":
        controller.cancel_job(job.id)
    else:
        controller.fail_job(job.id, "Failed to retrieve or parse remote output: boom")
        set_job_state(job, status)
    return controller, runner, document, job, run


@pytest.mark.parametrize("status", ["failed", "stopped", "cancelled"])
def test_complete_output_of_an_ended_job_is_recorded_on_a_fresh_run(
    tmp_path: Path, status: JobStatus
) -> None:
    controller, _runner, document, job, ended_run = _ended_cluster_setup(tmp_path, status)
    ended_status = ended_run.status
    evidence = _seen(
        SchedulerState.COMPLETED, OutputState.COMPLETE, parsed=_COMPLETE, on_request=True
    )

    update = controller.apply_evidence(job.id, evidence)

    assert update.changed and update.finished is not None
    assert job_status_word(job) == "retrieving"
    assert job.error_message is None and job.end_reason is None
    fresh_run = controller.run_for_job(job.id)
    assert fresh_run is not None and fresh_run is not ended_run
    assert fresh_run.run_index == ended_run.run_index + 1

    result = Result(calculation_run_id=fresh_run.id, result_type=job.calc_type, data={})
    document.results[result.id] = result
    controller.complete_job_with_result(job, result=result, warnings=[], convergence_achieved=None)

    assert job.status == JobStatus.COMPLETED
    assert fresh_run.status == JobStatus.COMPLETED and fresh_run.result_id == result.id
    # The ended run keeps its outcome.
    assert ended_run.status == ended_status and ended_run.result_id is None


@pytest.mark.parametrize(
    ("scheduler", "status"),
    [(SchedulerState.PENDING, "queued"), (SchedulerState.RUNNING, "running")],
)
def test_an_ended_job_the_scheduler_lists_again_resumes_on_a_fresh_run(
    tmp_path: Path, scheduler: SchedulerState, status: JobStatus
) -> None:
    controller, _runner, _document_, job, ended_run = _ended_cluster_setup(tmp_path)

    update = controller.apply_evidence(job.id, _seen(scheduler, on_request=True))

    assert update.changed
    assert job_status_word(job) == status
    assert job.error_message is None and job.completed_at is None
    assert (job.remote_job_id, job.remote_directory) == (_REMOTE_ID, _REMOTE_DIR)
    fresh_run = controller.run_for_job(job.id)
    assert fresh_run is not ended_run and fresh_run.status == JobStatus.ACTIVE
    assert (fresh_run.started_at is not None) is (status == "running")
    assert ended_run.status == JobStatus.ENDED


def test_an_ended_job_keeps_its_failure_when_the_check_finds_nothing_new(tmp_path: Path) -> None:
    controller, _runner, _document_, job, _run = _ended_cluster_setup(tmp_path)
    message = job.error_message

    update = controller.apply_evidence(
        job.id, _seen(SchedulerState.NOT_LISTED, OutputState.INCOMPLETE, on_request=True)
    )

    assert (update.changed, update.stamped) == (False, True)
    assert (job_status_word(job), job.error_message) == ("failed", message)
    assert job.last_status_message == "checked; no complete output found"


def test_an_ended_job_ignores_evidence_the_user_did_not_ask_for(tmp_path: Path) -> None:
    controller, _runner, _document_, job, _run = _ended_cluster_setup(tmp_path)

    update = controller.apply_evidence(
        job.id, _seen(SchedulerState.COMPLETED, OutputState.COMPLETE, parsed=_COMPLETE)
    )

    assert update.finished is None and not update.changed
    assert job_status_word(job) == "failed"


# --- Asking the cluster --------------------------------------------------------


def test_check_asks_about_every_active_job_on_the_connected_cluster(tmp_path: Path) -> None:
    controller, runner, document, job, _run = _cluster_setup(tmp_path, "running")
    service = controller._project_service
    elsewhere = make_job("elsewhere", service=service, document=document, execution_target="b")
    elsewhere.status = JobStatus.ACTIVE
    elsewhere.activity = JobActivity.RUNNING
    elsewhere.remote_job_id, elsewhere.remote_directory = "9", "/remote/elsewhere"
    ended = make_job("ended", service=service, document=document, execution_target="SCARF")
    ended.status = JobStatus.ENDED
    ended.end_reason = EndReason.UNKNOWN
    ended.remote_job_id, ended.remote_directory = "8", "/remote/ended"
    submitting = make_job("sub", service=service, document=document, execution_target="scarf")
    submitting.status = JobStatus.ACTIVE
    submitting.activity = JobActivity.SUBMITTING
    submitting.remote_directory = "/remote/sub"
    runner.submitting.add(submitting.id)
    local = make_job("local", service=service, document=document, execution_target="local")
    local.status = JobStatus.ACTIVE
    local.activity = JobActivity.RUNNING

    no_id = make_job("no-id", service=service, document=document, execution_target="scarf")
    no_id.status = JobStatus.ACTIVE
    no_id.activity = JobActivity.SUBMITTING
    no_id.remote_directory = "/remote/no-id"

    assert controller.check_cluster_jobs(skip_if_busy=True) == 1

    targets, skip_if_busy = runner.checks[0]
    assert skip_if_busy is True
    assert targets == [
        CheckTarget(
            job_id=job.id,
            remote_job_id=_REMOTE_ID,
            remote_dir=_REMOTE_DIR,
            seed_name="on-cluster",
            local_output_dir=tmp_path,
            cluster_name="scarf",
        )
    ]


def test_check_on_request_includes_the_ended_jobs_named(tmp_path: Path) -> None:
    controller, runner, _document_, job, _run = _ended_cluster_setup(tmp_path)

    assert controller.check_cluster_jobs([job.id]) == 0
    assert runner.checks == []

    assert controller.check_cluster_jobs([job.id], on_request=True) == 1
    targets, skip_if_busy = runner.checks[0]
    assert [(target.job_id, target.on_request) for target in targets] == [(job.id, True)]
    assert skip_if_busy is False


def _ended_on_cluster(
    controller: JobExecutionController,
    document: ProjectDocument,
    label: str,
    reason: EndReason,
    *,
    source: str | None = "retrieval",
    remote_job_id: str | None = "9",
    remote_directory: str | None = None,
):
    job = make_job(
        label,
        service=controller._project_service,
        document=document,
        execution_target="scarf",
    )
    job.status = JobStatus.ENDED
    job.end_reason = reason
    job.last_status_source = source
    job.remote_job_id = remote_job_id
    job.remote_directory = f"/remote/{label}" if remote_directory is None else remote_directory
    return job


def test_refresh_asks_about_every_active_and_ended_job_on_the_cluster(tmp_path: Path) -> None:
    controller, runner, document, active, _run = _cluster_setup(tmp_path, "running")
    unavailable = _ended_on_cluster(
        controller, document, "unavailable", EndReason.OUTPUT_UNAVAILABLE
    )
    lost = _ended_on_cluster(controller, document, "lost", EndReason.INTERRUPTED)
    cancelled = _ended_on_cluster(
        controller, document, "cancelled", EndReason.CANCELLED, source=None
    )
    cancelled.dismissed_at = utc_now()
    judged = _ended_on_cluster(controller, document, "judged", EndReason.CASTEP_ERROR)
    walltime = _ended_on_cluster(controller, document, "walltime", EndReason.WALLTIME)
    never_read = _ended_on_cluster(
        controller, document, "never-read", EndReason.UNKNOWN, source="scheduler_poll"
    )
    no_id_ended = _ended_on_cluster(
        controller, document, "no-id-ended", EndReason.CANCELLED, remote_job_id=None
    )
    no_id_active = make_job("no-id-active", service=controller._project_service, document=document)
    no_id_active.execution_target = "scarf"
    no_id_active.status = JobStatus.ACTIVE
    no_id_active.activity = JobActivity.RUNNING
    no_id_active.remote_directory = "/remote/no-id-active"

    assert controller.refresh_cluster_jobs() == 9

    targets, skip_if_busy = runner.checks[0]
    # Queued behind a running check, never dropped, and marked as a refresh.
    assert (skip_if_busy, runner.refreshes) == (False, [True])
    assert all(target.on_request for target in targets)
    read = {target.job_id: target.read_output for target in targets}
    assert read == {
        active.id: True,
        unavailable.id: True,
        lost.id: True,
        cancelled.id: True,
        judged.id: False,
        walltime.id: False,
        never_read.id: True,
        no_id_ended.id: True,
        no_id_active.id: True,
    }


def test_refresh_leaves_out_jobs_it_has_no_business_with(tmp_path: Path) -> None:
    controller, runner, document, job, _run = _cluster_setup(tmp_path, "running")
    service = controller._project_service
    submitting = make_job("sub", service=service, document=document, execution_target="scarf")
    submitting.status = JobStatus.ACTIVE
    submitting.activity = JobActivity.SUBMITTING
    submitting.remote_directory = "/remote/sub"
    runner.submitting.add(submitting.id)
    # The last submission never reached the scheduler: whatever the ID or the
    # folder holds belongs to an earlier run.
    _ended_on_cluster(controller, document, "rejected", EndReason.SUBMISSION, source=None)
    # Nothing to ask and no folder to read.
    _ended_on_cluster(
        controller, document, "judged-no-id", EndReason.CASTEP_ERROR, remote_job_id=None
    )
    _ended_on_cluster(controller, document, "no-folder", EndReason.CANCELLED, remote_directory="")
    elsewhere = _ended_on_cluster(controller, document, "elsewhere", EndReason.CANCELLED)
    elsewhere.execution_target = "other"
    done = _ended_on_cluster(controller, document, "done", EndReason.CANCELLED)
    done.status, done.end_reason = JobStatus.COMPLETED, None
    local = make_job("local", service=service, document=document, execution_target="local")
    set_job_state(local, "failed")

    assert controller.refresh_cluster_jobs() == 1
    assert [target.job_id for target in runner.checks[0][0]] == [job.id]


def test_refresh_with_nothing_to_ask_still_reports_a_pass(tmp_path: Path) -> None:
    controller, runner, _document_, job, _run = _cluster_setup(tmp_path, "running")
    job.status = JobStatus.COMPLETED

    assert controller.refresh_cluster_jobs() == 0
    assert (runner.checks, runner.refreshes) == ([([], False)], [True])


def test_refresh_asks_nothing_while_no_cluster_is_connected(tmp_path: Path) -> None:
    controller, runner, _document_, _job, _run = _ended_cluster_setup(tmp_path)
    runner.connected_cluster_name = None

    assert controller.refresh_cluster_jobs() == 0
    assert runner.checks == []


def test_the_automatic_check_still_leaves_ended_jobs_out(tmp_path: Path) -> None:
    controller, runner, document, active, _run = _cluster_setup(tmp_path, "running")
    _ended_on_cluster(controller, document, "unavailable", EndReason.OUTPUT_UNAVAILABLE)

    assert controller.check_cluster_jobs(skip_if_busy=True) == 1
    targets, skip_if_busy = runner.checks[0]
    assert [(target.job_id, target.on_request) for target in targets] == [(active.id, False)]
    assert (skip_if_busy, runner.refreshes) == (True, [False])


@pytest.mark.parametrize("other", [{"remote_job_id": "1"}, {"remote_directory": "/remote/older"}])
def test_refresh_evidence_about_another_submission_changes_nothing(tmp_path: Path, other) -> None:
    controller, _runner, _document_, job, _run = _ended_cluster_setup(tmp_path, "cancelled")
    evidence = _seen(
        SchedulerState.COMPLETED, OutputState.COMPLETE, parsed=_COMPLETE, on_request=True, **other
    )

    assert controller.apply_evidence(job.id, evidence) == LifecycleUpdate(job=job)
    assert job_status_word(job) == "cancelled"


def test_check_now_reads_the_folder_of_a_job_already_read_to_a_verdict(tmp_path: Path) -> None:
    controller, runner, document, _job, _run = _cluster_setup(tmp_path, "completed")
    judged = _ended_on_cluster(controller, document, "judged", EndReason.CASTEP_ERROR)

    _updates, requested = controller.check_now([judged.id])

    assert requested == 1
    assert [(t.job_id, t.read_output) for t in runner.checks[0][0]] == [(judged.id, True)]


def test_check_asks_nothing_while_no_cluster_is_connected(tmp_path: Path) -> None:
    controller, runner, _document_, _job, _run = _cluster_setup(tmp_path, "running", connected=None)

    assert controller.check_cluster_jobs() == 0
    assert runner.checks == []


def test_check_with_nothing_to_ask_still_reports_a_pass(tmp_path: Path) -> None:
    controller, runner, _document_, job, _run = _cluster_setup(tmp_path, "running")
    job.status = JobStatus.COMPLETED

    assert controller.check_cluster_jobs() == 0
    assert runner.checks == [([], False)]


def test_check_target_uses_the_job_s_retrieval_flags_and_inferred_folder(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    flags = RetrievalFlags(retrieve_out_cell=True)
    controller = JobExecutionController(
        document_getter=lambda: document,
        project_service=service,
        local_runner=FakeLocalRunner(),
        cluster_runner=FakeClusterRunner(),
        job_output_directory=lambda _job: tmp_path,
        job_seed_name=lambda job: job.label,
        retrieval_for_job=lambda _job: flags,
        remote_directory_for_job=lambda job: f"/work/{job.label}",
    )
    job = make_job("legacy", service=service, document=document, execution_target="scarf")
    job.remote_job_id = "77"

    target = controller.check_target(job)

    assert target is not None
    assert (target.remote_dir, target.retrieval) == ("/work/legacy", flags)
    assert job.remote_directory == "/work/legacy"
    assert controller.check_targets_for([job.id, uuid4()]) == [target]


def test_check_target_needs_somewhere_to_look(tmp_path: Path) -> None:
    controller, _runner, document, job, _run = _cluster_setup(tmp_path, "submitting")

    job.remote_job_id = None
    with_folder = controller.check_target(job)
    assert with_folder is not None and with_folder.remote_job_id == ""

    job.remote_directory = None
    assert controller.check_target(job) is None

    local = make_job(
        "local", service=controller._project_service, document=document, execution_target="local"
    )
    assert controller.check_target(local) is None


def test_check_target_needs_a_local_output_folder(tmp_path: Path) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=None)
    job = _cluster_running_job(service, document, "nowhere")
    job.remote_job_id, job.remote_directory = "1", "/remote/nowhere"

    assert controller.check_target(job) is None


@pytest.mark.parametrize(
    ("status", "folder", "connected", "submitting", "expected"),
    [
        ("running", _REMOTE_DIR, "scarf", False, True),
        ("failed", _REMOTE_DIR, "SCARF", False, True),
        ("cancelled", _REMOTE_DIR, "scarf", False, True),
        (JobStatus.COMPLETED, _REMOTE_DIR, "scarf", False, False),
        (JobStatus.DEFINED, _REMOTE_DIR, "scarf", False, False),
        ("failed", None, "scarf", False, False),
        ("failed", _REMOTE_DIR, "other", False, False),
        ("failed", _REMOTE_DIR, None, False, False),
        ("submitting", _REMOTE_DIR, "scarf", True, False),
    ],
)
def test_check_now_is_offered_for_a_cluster_job_with_somewhere_to_look(
    tmp_path: Path, status, folder, connected, submitting, expected
) -> None:
    controller, runner, _document_, job, _run = _cluster_setup(
        tmp_path, status, connected=connected
    )
    job.remote_directory = folder
    if submitting:
        runner.submitting.add(job.id)

    assert controller.can_check_now(job) is expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("failed", True),
        ("stopped", True),
        ("cancelled", True),
        ("running", False),
        (JobStatus.COMPLETED, False),
    ],
)
def test_check_now_is_offered_for_a_local_job_that_ended(tmp_path: Path, status, expected) -> None:
    service, document = _document(tmp_path)
    controller = _controller(document=document, service=service, output_dir=tmp_path)
    job = make_job("local", service=service, document=document, execution_target="local")
    set_job_state(job, status)

    assert controller.can_check_now(job) is expected


def test_check_now_decides_a_local_job_at_once_and_asks_about_cluster_jobs(
    tmp_path: Path,
) -> None:
    controller, runner, document, cluster_job, _run = _ended_cluster_setup(tmp_path)
    service = controller._project_service
    finished = make_job("finished", service=service, document=document, execution_target="local")
    finished.status = JobStatus.ENDED
    finished.end_reason = EndReason.WALLTIME
    (tmp_path / "finished.castep").write_bytes(_COMPLETE_OUTPUT.read_bytes())
    unfinished = make_job("unfinished", service=service, document=document)
    unfinished.execution_target = "local"
    unfinished.status = JobStatus.ENDED
    unfinished.end_reason = EndReason.UNKNOWN
    running = make_job("running", service=service, document=document, execution_target="local")
    running.status = JobStatus.ACTIVE
    running.activity = JobActivity.RUNNING

    updates, requested = controller.check_now(
        [finished.id, unfinished.id, running.id, cluster_job.id, finished.id, uuid4()]
    )

    assert requested == 1
    assert [target.job_id for target in runner.checks[0][0]] == [cluster_job.id]
    assert [update.job for update in updates] == [finished, unfinished]
    assert updates[0].finished is not None and job_status_word(finished) == "retrieving"
    assert updates[1].finished is None and job_status_word(unfinished) == "failed"


@pytest.mark.parametrize(
    ("remote_id", "connected", "submitting", "expected"),
    [
        (_REMOTE_ID, "scarf", False, True),
        (None, "scarf", False, False),
        (_REMOTE_ID, None, False, False),
        (_REMOTE_ID, "other", False, False),
        (None, "scarf", True, True),
    ],
)
def test_a_cancel_reaches_the_cluster_only_for_a_job_it_can_name(
    tmp_path: Path, remote_id, connected, submitting, expected
) -> None:
    controller, runner, _document_, job, _run = _cluster_setup(
        tmp_path, "running", connected=connected
    )
    job.remote_job_id = remote_id
    if submitting:
        runner.submitting.add(job.id)

    assert controller.can_cancel_on_cluster(job) is expected


# --- One status-bar line per check ---------------------------------------------


def test_check_summary_counts_what_several_jobs_came_to(tmp_path: Path) -> None:
    controller, _runner, document, stopped, _run = _cluster_setup(tmp_path, "queued")
    service = controller._project_service

    def another(label: str, status: JobStatus):
        job = make_job(label, service=service, document=document, execution_target="scarf")
        set_job_state(job, status)
        job.remote_job_id, job.remote_directory = _REMOTE_ID, _REMOTE_DIR
        return job

    started = another("started", "queued")
    also_started = another("also-started", "queued")
    finished = another("finished", "running")
    unchanged = another("unchanged", "running")
    controller.apply_evidence(stopped.id, _seen(SchedulerState.TIMED_OUT, OutputState.ABSENT))
    controller.apply_evidence(started.id, _seen(SchedulerState.RUNNING))
    controller.apply_evidence(also_started.id, _seen(SchedulerState.RUNNING))
    controller.apply_evidence(unchanged.id, _seen(SchedulerState.RUNNING))
    update = controller.apply_evidence(
        finished.id, _seen(SchedulerState.COMPLETED, OutputState.COMPLETE, parsed=_COMPLETE)
    )
    # The caller records the result, which completes the job.
    assert update.finished is not None
    service.complete_job(finished)

    assert controller.take_check_summary(5) == (
        "Checked 5 cluster jobs: 1 stopped at its walltime limit, 2 started running, 1 completed"
    )
    # Taking the summary starts afresh.
    assert controller.take_check_summary(5) == ""


def test_check_summary_names_a_single_completed_job(tmp_path: Path) -> None:
    controller, _runner, _document_, job, _run = _cluster_setup(tmp_path, "running")
    controller.apply_evidence(
        job.id, _seen(SchedulerState.COMPLETED, OutputState.COMPLETE, parsed=_COMPLETE)
    )
    controller._project_service.complete_job(job)

    assert controller.take_check_summary(1) == "Completed cluster job on-cluster"


def test_check_summary_leaves_out_a_job_still_being_retrieved(tmp_path: Path) -> None:
    controller, _runner, _document_, job, _run = _cluster_setup(tmp_path, "running")
    controller.apply_evidence(job.id, _seen(SchedulerState.COMPLETED))

    assert job_status_word(job) == "retrieving"
    assert controller.take_check_summary(1) == ""


def test_check_summary_leaves_out_local_jobs_and_jobs_since_removed(tmp_path: Path) -> None:
    controller, _runner, document, job, _run = _cluster_setup(tmp_path, "queued")
    controller.apply_evidence(job.id, _seen(SchedulerState.RUNNING))
    del document.jobs[job.id]
    local, _ = _active_local_job(controller._project_service, document, "local", "running")
    controller.check_local_jobs()

    assert job_status_word(local) == "stopped"
    assert controller.take_check_summary(1) == ""


def test_a_job_with_no_scheduler_id_is_asked_about_only_on_request(tmp_path: Path) -> None:
    controller, runner, _document_, job, _run = _cluster_setup(tmp_path, "submitting")
    job.remote_job_id = None

    assert controller.check_cluster_jobs() == 0
    assert runner.checks == [([], False)]

    assert controller.check_cluster_jobs([job.id], on_request=True) == 1
    assert [target.remote_job_id for target in runner.checks[-1][0]] == [""]

    # Complete output found by that check completes the job.
    update = controller.apply_evidence(
        job.id,
        _seen(
            SchedulerState.NO_JOB_ID,
            OutputState.COMPLETE,
            parsed=_COMPLETE,
            remote_job_id="",
            on_request=True,
        ),
    )
    assert update.finished is not None


def test_progress_is_requested_for_jobs_with_somewhere_to_look(tmp_path: Path) -> None:
    controller, runner, _document_, job, _run = _cluster_setup(tmp_path, "running")
    nowhere = make_job(
        "nowhere",
        service=controller._project_service,
        document=_document_,
        execution_target="scarf",
    )
    nowhere.status = JobStatus.ACTIVE
    nowhere.activity = JobActivity.RUNNING
    gone = uuid4()

    unasked = controller.read_cluster_progress([job.id, nowhere.id, gone])

    # The caller is told which jobs will get no answer from the worker.
    assert unasked == [nowhere.id, gone]
    assert [[target.job_id for target in batch] for batch in runner.progress_requests] == [[job.id]]
    assert controller.read_cluster_progress([nowhere.id]) == [nowhere.id]
    assert len(runner.progress_requests) == 1
