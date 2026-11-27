"""Tests for what the status bar says about the project's clusters."""

from __future__ import annotations

import pytest

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job
from mufinder.services.cluster_status import (
    NO_CLUSTER_JOBS_TOOLTIP,
    NO_REMOTE_IDS_TOOLTIP,
    REFRESH_TOOLTIP,
    ClusterConnectionTone,
    cluster_status,
    monitored_cluster_names,
    refreshable_cluster_names,
)
from mufinder.domain.enums import EndReason
from mufinder.domain.models import utc_now
from mufinder.services.job_monitoring import CANCEL_NOT_SENT_SOURCE, RESULT_NOT_RECORDED_PREFIX
from tests.support.jobs import state_fields


def _job(
    target: str = "SCARF",
    status: str = "running",
    remote_job_id: str | None = "12345",
) -> Job:
    return Job(
        label="job",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
        **state_fields(status),
        execution_target=target,
        remote_job_id=remote_job_id,
    )


def test_hidden_without_jobs_the_poll_follows() -> None:
    for jobs in (
        [],
        [_job(target="local")],
        [_job(target="")],
        [_job(status=JobStatus.COMPLETED)],
        [_job(status="failed")],
    ):
        status = cluster_status(jobs, connected_cluster="SCARF")
        assert status.visible is False
        assert status.can_poll is False
        assert status.refresh_tooltip == NO_CLUSTER_JOBS_TOOLTIP


def test_shown_for_every_active_status_including_retrieving() -> None:
    for job_status in (
        "submitting",
        "queued",
        "running",
        "retrieving",
    ):
        assert cluster_status([_job(status=job_status)], connected_cluster=None).visible is True


def test_hidden_for_a_job_that_ended_whatever_its_message() -> None:
    assert cluster_status([_job(status="failed")], connected_cluster=None).visible is False


def test_idle_reads_neutral_with_the_cluster_name() -> None:
    status = cluster_status([_job()], connected_cluster=None)
    assert status.connection_text == "Not connected to SCARF"
    assert status.tone is ClusterConnectionTone.NEUTRAL
    assert status.can_poll is True
    assert status.refresh_tooltip == REFRESH_TOOLTIP


def test_a_connection_to_another_cluster_does_not_count() -> None:
    status = cluster_status([_job()], connected_cluster="PSI")
    assert status.connection_text == "Not connected to SCARF"
    assert status.tone is ClusterConnectionTone.NEUTRAL


def test_connected_matches_the_cluster_name_case_insensitively() -> None:
    status = cluster_status([_job()], connected_cluster="scarf")
    assert status.connection_text == "Connected to SCARF"
    assert status.tone is ClusterConnectionTone.CONNECTED


def test_lost_only_while_that_cluster_has_jobs_the_poll_follows() -> None:
    status = cluster_status([_job()], connected_cluster=None, lost_clusters={"scarf"})
    assert status.connection_text == "Connection to SCARF lost"
    assert status.tone is ClusterConnectionTone.LOST

    status = cluster_status([_job(target="PSI")], connected_cluster=None, lost_clusters={"scarf"})
    assert status.connection_text == "Not connected to PSI"
    assert status.tone is ClusterConnectionTone.NEUTRAL


def test_several_clusters_are_each_described_and_a_loss_takes_the_colour() -> None:
    jobs = [_job(target="SCARF"), _job(target="PSI"), _job(target="ARCHER"), _job(target="scarf")]

    status = cluster_status(jobs, connected_cluster="SCARF", lost_clusters={"psi"})
    assert status.connection_text == (
        "Connected to SCARF; connection to PSI lost; not connected to ARCHER"
    )
    assert status.tone is ClusterConnectionTone.LOST

    status = cluster_status(jobs, connected_cluster="SCARF")
    assert status.connection_text == "Connected to SCARF; not connected to PSI, ARCHER"
    assert status.tone is ClusterConnectionTone.CONNECTED


def test_nothing_to_check_until_a_job_has_a_remote_id_or_folder() -> None:
    submitting = _job(status="submitting", remote_job_id=None)
    submitting.remote_directory = None
    status = cluster_status([submitting], connected_cluster=None)
    assert status.visible is True
    assert status.can_poll is False
    assert status.refresh_tooltip == NO_REMOTE_IDS_TOOLTIP

    submitting.remote_directory = "/remote/job"
    assert cluster_status([submitting], connected_cluster=None).can_poll is True

    submitting.remote_directory = None
    status = cluster_status([submitting, _job()], connected_cluster=None)
    assert status.can_poll is True


def test_monitored_cluster_names_keep_first_seen_order_and_spelling() -> None:
    jobs = [_job(target="PSI"), _job(target="local"), _job(target=" SCARF "), _job(target="psi")]
    assert monitored_cluster_names(jobs) == ["PSI", "SCARF"]


def _ended(
    target: str = "SCARF",
    reason: EndReason = EndReason.OUTPUT_UNAVAILABLE,
    *,
    remote_directory: str | None = "/remote/job",
) -> Job:
    job = _job(target=target, status="failed")
    job.end_reason = reason
    job.remote_directory = remote_directory
    return job


@pytest.mark.parametrize("reason", [EndReason.OUTPUT_UNAVAILABLE, EndReason.INTERRUPTED])
@pytest.mark.parametrize("dismissed", [False, True])
def test_shown_for_an_ended_job_whose_output_may_still_be_on_the_cluster(
    reason: EndReason, dismissed: bool
) -> None:
    job = _ended(reason=reason)
    if dismissed:
        job.dismissed_at = utc_now()

    status = cluster_status([job], connected_cluster=None)

    assert (status.visible, status.can_poll) == (True, True)
    assert status.connection_text == "Not connected to SCARF"
    assert status.refresh_tooltip == REFRESH_TOOLTIP


@pytest.mark.parametrize(
    "job",
    [
        _ended(reason=EndReason.CASTEP_ERROR),
        _ended(reason=EndReason.WALLTIME),
        _ended(reason=EndReason.CANCELLED),
        _ended(remote_directory=None),
        _job(status=JobStatus.COMPLETED),
    ],
    ids=["castep-error", "walltime", "cancelled", "no-folder", "completed"],
)
def test_hidden_when_no_ended_job_has_output_left_to_fetch(job: Job) -> None:
    assert cluster_status([job], connected_cluster=None).visible is False


def test_shown_for_a_cancel_the_scheduler_never_received() -> None:
    job = _ended(reason=EndReason.CANCELLED)
    job.last_status_source = CANCEL_NOT_SENT_SOURCE

    status = cluster_status([job], connected_cluster=None)

    assert (status.visible, status.can_poll) == (True, True)
    assert status.connection_text == "Not connected to SCARF"
    assert refreshable_cluster_names([job]) == ["SCARF"]


def test_shown_for_a_job_whose_result_was_not_recorded() -> None:
    job = _ended(reason=EndReason.UNKNOWN)
    job.last_status_source = "retrieval"
    job.error_message = f"{RESULT_NOT_RECORDED_PREFIX}: unexpected output layout"

    assert cluster_status([job], connected_cluster=None).visible is True


def test_a_cluster_with_only_ended_jobs_never_reads_as_lost() -> None:
    status = cluster_status([_ended()], connected_cluster=None, lost_clusters={"scarf"})

    assert status.tone is ClusterConnectionTone.NEUTRAL
    assert status.connection_text == "Not connected to SCARF"

    # Its active job makes the dropped connection matter again.
    status = cluster_status([_ended(), _job()], connected_cluster=None, lost_clusters={"scarf"})
    assert status.tone is ClusterConnectionTone.LOST


def test_refreshable_cluster_names_add_clusters_of_ended_jobs_with_output_left() -> None:
    jobs = [_job(target="PSI"), _ended(target="SCARF"), _ended("ARCHER", EndReason.CASTEP_ERROR)]

    assert monitored_cluster_names(jobs) == ["PSI"]
    assert refreshable_cluster_names(jobs) == ["PSI", "SCARF"]
