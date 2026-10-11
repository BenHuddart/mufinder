"""Characterisation tests for the cluster submission planner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest

from mufinder.cluster_setup import ClusterConfig
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import Job
from mufinder.services.cluster_security import invalid_remote_job_labels
from mufinder.services.cluster_submission_planner import (
    ClusterTargetPlan,
    OverwriteDecisions,
    build_cluster_submissions,
    can_reuse_cluster_connection,
    plan_cluster_submission,
    plan_remote_targets,
    remote_check_failures,
    resolve_overwrite_decisions,
    split_stranded_quantum_jobs,
    stranded_quantum_jobs_message,
    unsafe_job_labels_message,
)
from mufinder.services.job_artifacts import RetrievalFlags
from mufinder.services.job_execution_controller import PreparedClusterSubmission
from tests.support.jobs import make_job

CLUSTER = ClusterConfig(
    name="Alpha",
    hostname="alpha.example",
    username="user",
    cores_per_node=32,
    template_script_path="/templates/run.sh",
    castep_executable="castep.mpi",
)


@dataclass
class _Settings:
    working_directory: str = "/remote/work"
    num_processes: int = 16
    walltime: str = "02:00:00"
    queue_name: str = "short"
    memory_gb: float | None = 64.0
    mem_per_cpu_gb: float | None = None
    comment: str = "note"


def _remote_job_directory(*, working_directory: str, seed_name: str) -> str:
    return f"{working_directory}/{seed_name}"


def _no_quantum(_job: Job, *, cluster_name: str) -> str:
    return ""


def _plan(
    jobs: list[Job],
    *,
    quantum: dict[UUID, str] | None = None,
    fresh_start_job_ids: tuple[UUID, ...] = (),
    reuse_existing_remote_files: bool = False,
) -> ClusterTargetPlan:
    quantum = quantum or {}
    return plan_remote_targets(
        jobs,
        cluster_name="alpha",
        working_directory="/remote/work",
        seed_name_by_job={job.id: f"seed_{job.label}" for job in jobs},
        remote_job_directory=_remote_job_directory,
        quantum_continuation_target=lambda job, *, cluster_name: quantum.get(job.id, ""),
        fresh_start_job_ids=fresh_start_job_ids,
        reuse_existing_remote_files=reuse_existing_remote_files,
    )


def _submissions(plan: ClusterTargetPlan, decisions: OverwriteDecisions, **kwargs):
    kwargs.setdefault("output_directory", lambda job: Path("/local") / job.label)
    return build_cluster_submissions(
        plan,
        decisions,
        cluster=CLUSTER,
        submission=_Settings(),
        retrieval_flags=lambda _job: RetrievalFlags(retrieve_den_fmt=True),
        **kwargs,
    )


def _never_asked(*_args: object) -> str:
    raise AssertionError("no overwrite prompt expected")


def test_fresh_job_gets_a_target_under_the_working_directory_and_is_checked() -> None:
    job = make_job("a")

    plan = _plan([job])

    (target,) = plan.targets
    assert target.remote_target == "/remote/work/seed_a"
    assert not target.reuse_existing_files and not target.allow_existing_remote_directory
    assert plan.targets_to_check() == [(job.id, "/remote/work/seed_a")]
    assert plan.protected_remote_targets() == set()


def test_continuation_job_keeps_its_preserved_remote_directory_on_the_same_cluster() -> None:
    job = make_job("a", execution_target=" ALPHA ", remote_directory=" /remote/old/a ")

    plan = _plan([job])

    (target,) = plan.targets
    assert target.remote_target == "/remote/old/a"
    assert target.allow_existing_remote_directory and not target.reuse_existing_files
    assert plan.targets_to_check() == []
    assert plan.protected_remote_targets() == {"/remote/old/a"}
    # Even when the directory exists, a continuation is never an overwrite candidate.
    decisions = resolve_overwrite_decisions(
        plan, ["/remote/old/a"], ask=_never_asked, confirm_bulk_delete=_never_asked
    )
    assert decisions.decision_by_job == {}
    (item,) = _submissions(plan, decisions).items
    assert item.submission.remote_target == "/remote/old/a"
    assert item.submission.allow_existing_remote_directory is True
    assert item.remote_dir_to_delete is None


def test_continuation_on_another_cluster_falls_through_to_a_fresh_target() -> None:
    job = make_job("a", execution_target="beta", remote_directory="/remote/old/a")

    (target,) = _plan([job]).targets

    assert target.remote_target == "/remote/work/seed_a"
    assert not target.allow_existing_remote_directory


def test_resubmission_reuses_its_previous_directory_without_uploading() -> None:
    job = make_job("a", status=JobStatus.FAILED, execution_target="alpha", remote_directory="/remote/old/a")

    plan = _plan([job], reuse_existing_remote_files=True)

    (target,) = plan.targets
    assert target.remote_target == "/remote/old/a"
    assert target.reuse_existing_files and not target.allow_existing_remote_directory
    assert plan.targets_to_check() == []


def test_quantum_phonon_job_submits_into_its_stage_one_directory() -> None:
    job = make_job("phonon")

    plan = _plan([job], quantum={job.id: "/remote/work/singlepoint"})

    (target,) = plan.targets
    assert target.remote_target == "/remote/work/singlepoint"
    assert target.allow_existing_remote_directory
    assert plan.protected_remote_targets() == {"/remote/work/singlepoint"}


def test_start_again_job_deletes_its_own_directory_without_a_prompt() -> None:
    job = make_job(
        "a",
        status=JobStatus.STOPPED,
        execution_target="alpha",
        remote_directory="/remote/old/a",
        remote_job_id=" 4242 ",
    )

    plan = _plan([job], fresh_start_job_ids=(job.id,))
    decisions = resolve_overwrite_decisions(
        plan, ["/remote/old/a"], ask=_never_asked, confirm_bulk_delete=_never_asked
    )

    (target,) = plan.targets
    assert target.fresh_delete and target.previous_remote_job_id == "4242"
    assert plan.targets_to_check() == []
    assert decisions.decision_by_job == {job.id: "delete"}
    assert plan.fresh_start_jobs(decisions) == [job]
    (item,) = _submissions(plan, decisions).items
    assert item.remote_dir_to_delete == "/remote/old/a"
    assert item.delete_only_if_finished == "4242"
    assert item.submission.reuse_existing_files is False
    assert item.submission.allow_existing_remote_directory is False


def test_start_again_job_on_another_cluster_is_a_fresh_job_still_cleared_locally() -> None:
    job = make_job("a", status=JobStatus.STOPPED, execution_target="beta", remote_directory="/remote/old/a")

    plan = _plan([job], fresh_start_job_ids=(job.id,), reuse_existing_remote_files=True)

    (target,) = plan.targets
    assert target.remote_target == "/remote/work/seed_a"
    assert not target.fresh_delete and not target.reuse_existing_files
    assert plan.fresh_start_jobs(OverwriteDecisions()) == [job]


@pytest.mark.parametrize(
    ("choice", "confirmed", "expected"),
    [
        ("yes", None, "delete"),
        ("no", None, "skip"),
        ("yes_to_all", True, "delete"),
        ("yes_to_all", False, "skip"),
        ("no_to_all", None, "skip"),
    ],
)
def test_existing_target_follows_each_overwrite_choice(choice: str, confirmed: bool | None, expected: str) -> None:
    job = make_job("a")
    plan = _plan([job])
    bulk_calls: list[list[str]] = []

    def confirm(paths: list[str]) -> bool:
        bulk_calls.append(paths)
        assert confirmed is not None
        return confirmed

    decisions = resolve_overwrite_decisions(
        plan, ["/remote/work/seed_a"], ask=lambda label, path: choice, confirm_bulk_delete=confirm
    )

    assert decisions.decision_by_job == {job.id: expected}
    assert bulk_calls == ([] if confirmed is None else [["/remote/work/seed_a"]])
    if expected == "skip":
        assert decisions.skipped == [
            "- a: Submission skipped because remote directory already exists (/remote/work/seed_a)"
        ]
        assert _submissions(plan, decisions).items == []
    else:
        assert decisions.skipped == []
        (item,) = _submissions(plan, decisions).items
        assert item.remote_dir_to_delete == "/remote/work/seed_a"


@pytest.mark.parametrize(("first", "expected"), [("yes_to_all", "delete"), ("no_to_all", "skip")])
def test_to_all_answers_the_remaining_prompts(first: str, expected: str) -> None:
    jobs = [make_job("a"), make_job("b"), make_job("c")]
    plan = _plan(jobs)
    asked: list[tuple[str, str]] = []

    decisions = resolve_overwrite_decisions(
        plan,
        ["/remote/work/seed_a", "/remote/work/seed_c"],
        ask=lambda label, path: asked.append((label, path)) or first,
        confirm_bulk_delete=lambda _paths: True,
    )

    assert asked == [("a", "/remote/work/seed_a")]
    assert decisions.decision_by_job == {jobs[0].id: expected, jobs[2].id: expected}
    assert jobs[1].id not in decisions.decision_by_job


def test_protected_directory_is_never_deleted_for_a_colliding_job() -> None:
    continuing = make_job("a", execution_target="alpha", remote_directory="/remote/work/seed_b")
    colliding = make_job("b")

    plan = _plan([continuing, colliding])
    asked: list[tuple[str, str]] = []
    decisions = resolve_overwrite_decisions(
        plan,
        ["/remote/work/seed_b"],
        ask=lambda label, path: asked.append((label, path)) or "yes",
        confirm_bulk_delete=_never_asked,
    )
    submissions = _submissions(plan, decisions)

    assert asked == [("b", "/remote/work/seed_b")]
    assert decisions.decision_by_job == {colliding.id: "delete"}
    assert [item.remote_dir_to_delete for item in submissions.items] == [None, None]


def test_start_again_job_in_a_protected_directory_fails() -> None:
    continuing = make_job("a", execution_target="alpha", remote_directory="/remote/shared")
    restarting = make_job("b", status=JobStatus.STOPPED, execution_target="alpha", remote_directory="/remote/shared")

    plan = _plan([continuing, restarting], fresh_start_job_ids=(restarting.id,))
    decisions = resolve_overwrite_decisions(plan, [], ask=_never_asked, confirm_bulk_delete=_never_asked)
    submissions = _submissions(plan, decisions)

    assert [item.submission.job for item in submissions.items] == [continuing]
    assert submissions.failed == [
        "- b: another job in this submission is continuing in its directory (/remote/shared), "
        "so it was not started again"
    ]


def test_job_without_an_output_directory_is_missing() -> None:
    present, absent = make_job("a"), make_job("b")
    plan = _plan([present, absent])

    submissions = _submissions(
        plan,
        OverwriteDecisions(),
        output_directory=lambda job: None if job is absent else Path("/local/a"),
    )

    assert submissions.missing == ["b"]
    assert [item.submission.job for item in submissions.items] == [present]


def test_uncleaned_start_again_job_is_left_out() -> None:
    job = make_job("a", status=JobStatus.STOPPED, execution_target="alpha", remote_directory="/remote/old/a")
    plan = _plan([job], fresh_start_job_ids=(job.id,))
    decisions = resolve_overwrite_decisions(plan, [], ask=_never_asked, confirm_bulk_delete=_never_asked)

    submissions = _submissions(plan, decisions, not_cleaned_job_ids={job.id})

    assert submissions.items == [] and submissions.failed == [] and submissions.missing == []


def test_submission_carries_the_cluster_and_dialog_settings() -> None:
    job = make_job("a")
    plan = _plan([job])

    (item,) = _submissions(plan, OverwriteDecisions()).items

    submission = item.submission
    assert submission.output_dir == Path("/local/a")
    assert submission.seed_name == "seed_a"
    assert submission.remote_target == "/remote/work/seed_a"
    assert submission.execution_target == "Alpha"
    assert (submission.num_processes, submission.walltime, submission.working_directory) == (
        16,
        "02:00:00",
        "/remote/work",
    )
    assert (submission.template_script_path, submission.cores_per_node, submission.castep_executable) == (
        "/templates/run.sh",
        32,
        "castep.mpi",
    )
    assert (submission.memory_gb, submission.mem_per_cpu_gb, submission.queue_name, submission.comment) == (
        64.0,
        None,
        "short",
        "note",
    )
    assert submission.retrieval == RetrievalFlags(retrieve_den_fmt=True)
    assert item.remote_dir_to_delete is None and item.delete_only_if_finished is None


def test_prepare_batch_skips_declined_jobs() -> None:
    accepted, declined = make_job("a"), make_job("b")
    submissions = _submissions(_plan([accepted, declined]), OverwriteDecisions())

    def prepare(request):
        if request.job is declined:
            return None
        return PreparedClusterSubmission(job=request.job, run=None, submit_kwargs={"seed_name": request.seed_name})

    submit_items, prepared_by_job = submissions.prepare_batch(prepare)

    assert list(prepared_by_job) == [accepted.id]
    assert submit_items == [
        {
            "job_id": accepted.id,
            "label": "a",
            "submit_kwargs": {"seed_name": "seed_a"},
            "remote_dir_to_delete": None,
            "delete_only_if_finished": None,
        }
    ]


def test_remote_check_failures_name_each_failed_job() -> None:
    jobs = [make_job("a"), make_job("b")]
    plan = _plan(jobs)

    assert remote_check_failures(plan, {jobs[1].id: "timeout", jobs[0].id: ""}) == [
        "- b: Failed to check remote directory (timeout)"
    ]


def test_stranded_quantum_jobs_are_split_out_and_explained() -> None:
    ordinary, stranded_local, stranded_remote, continuing = (make_job(label) for label in "abcd")
    stage_one = {stranded_local.id: "local", stranded_remote.id: "beta", continuing.id: "alpha"}

    kept, stranded = split_stranded_quantum_jobs(
        [ordinary, stranded_local, stranded_remote, continuing],
        cluster_name="alpha",
        stage_one_target=lambda job: stage_one.get(job.id),
        continuation_target=lambda job, *, cluster_name: "/remote/sp" if job is continuing else "",
    )

    assert kept == [ordinary, continuing]
    assert stranded == [(stranded_local, "local"), (stranded_remote, "beta")]
    assert stranded_quantum_jobs_message("alpha", stranded) == (
        "These phonon jobs were not submitted to alpha because their stage-1 checkpoint (.check) "
        "is not in a directory there:\n\n"
        "  - b (stage 1 ran on this computer)\n"
        "  - c (stage 1 ran on beta)"
    )


def test_unsafe_labels_are_detected_and_explained() -> None:
    invalid = invalid_remote_job_labels(["ok_1.a-b", "bad label", "bad/label"])

    assert invalid == ["bad label", "bad/label"]
    assert unsafe_job_labels_message(invalid) == (
        "Cluster job labels may contain only letters, numbers, dot, underscore, and hyphen.\n\n"
        "Rename the following jobs before cluster submission:\n"
        "- bad label\n"
        "- bad/label"
    )


@pytest.mark.parametrize(
    ("is_connected", "name", "user", "expected"),
    [
        (True, " alpha ", "user", True),
        (True, "alpha", "", True),
        (True, "alpha", "other", False),
        (True, "beta", "user", False),
        (False, "alpha", "user", False),
    ],
)
def test_connection_reuse_requires_the_same_cluster_and_user(
    is_connected: bool, name: str, user: str, expected: bool
) -> None:
    assert (
        bool(
            can_reuse_cluster_connection(
                CLUSTER, is_connected=is_connected, connected_cluster_name=name, connected_username=user
            )
        )
        is expected
    )


def _drive(jobs: list[Job], **overrides):
    calls: dict[str, list[object]] = {"check": [], "ask": [], "clear": []}
    kwargs = dict(
        cluster=CLUSTER,
        submission=_Settings(),
        seed_name_by_job={job.id: f"seed_{job.label}" for job in jobs},
        remote_job_directory=_remote_job_directory,
        quantum_continuation_target=_no_quantum,
        output_directory=lambda job: Path("/local") / job.label,
        retrieval_flags=lambda _job: RetrievalFlags(),
        check_remote_directories=lambda targets: calls["check"].append(targets) or {"existing": [], "errors": {}},
        ask_overwrite=lambda label, path: calls["ask"].append((label, path)) or "no",
        confirm_bulk_delete=_never_asked,
        clear_fresh_outputs=lambda fresh: calls["clear"].append(fresh) or (list(fresh), []),
    )
    kwargs.update(overrides)
    return plan_cluster_submission(jobs, **kwargs), calls


def test_driver_skips_the_callbacks_it_does_not_need() -> None:
    job = make_job("a", execution_target="alpha", remote_directory="/remote/old/a")

    plan, calls = _drive([job])

    assert calls == {"check": [], "ask": [], "clear": []}
    assert [item.submission.remote_target for item in plan.items] == ["/remote/old/a"]


def test_driver_reports_failures_in_stage_order_and_skips() -> None:
    errored = make_job("errored")
    existing = make_job("existing")
    uncleared = make_job("uncleared", status=JobStatus.STOPPED, execution_target="alpha", remote_directory="/remote/x")
    blocked = make_job("blocked", status=JobStatus.STOPPED, execution_target="alpha", remote_directory="/remote/c")
    continuing = make_job("continuing", execution_target="alpha", remote_directory="/remote/c")
    jobs = [errored, existing, uncleared, blocked, continuing]

    plan, calls = _drive(
        jobs,
        fresh_start_job_ids=(uncleared.id, blocked.id),
        check_remote_directories=lambda targets: {
            "existing": ["/remote/work/seed_existing"],
            "errors": {errored.id: "boom"},
        },
        clear_fresh_outputs=lambda fresh: (
            [job for job in fresh if job is not uncleared],
            ["- uncleared: could not clear its previous output (denied)"],
        ),
    )

    assert calls["ask"] == [("existing", "/remote/work/seed_existing")]
    assert plan.failed == [
        "- errored: Failed to check remote directory (boom)",
        "- uncleared: could not clear its previous output (denied)",
        "- blocked: another job in this submission is continuing in its directory (/remote/c), "
        "so it was not started again",
    ]
    assert plan.skipped == [
        "- existing: Submission skipped because remote directory already exists (/remote/work/seed_existing)"
    ]
    # A job whose directory check errored is still planned for submission.
    assert [item.submission.job for item in plan.items] == [errored, continuing]
