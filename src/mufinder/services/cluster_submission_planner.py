"""Plan a cluster submission: remote targets, overwrite decisions and submit items.

Submitting a batch of jobs to a cluster interleaves decisions about remote
directories with questions only the user can answer. This module holds the
decisions; the window keeps the dialog, the prompts and the message boxes.
Before connecting, :func:`can_reuse_cluster_connection` decides whether the
open connection serves. :func:`plan_cluster_submission` then runs the planning stages in
order, calling back into the window for the remote existence check, the
prompts and the local clean-up:

1. :func:`plan_remote_targets` resolves each job's remote directory: a
   start-again job's own directory (deleted and re-uploaded), a continuation's
   preserved ``remote_directory`` (submitted into as-is), a resubmitted job's
   previous directory (its files reused), or a fresh directory under the
   working directory.
   :meth:`ClusterTargetPlan.targets_to_check` lists the fresh directories whose
   existence the worker checks. A job whose check raises is not submitted:
   :func:`remote_check_failures` reports it and
   :meth:`ClusterTargetPlan.without_jobs` drops it from the plan, since its
   directory may hold a previous run the user was never asked to overwrite.
2. :meth:`ClusterTargetPlan.overwrite_prompts` lists, as data, the "remote
   directory exists" questions to ask, and :func:`resolve_overwrite_decisions`
   walks them with the window's prompt callbacks, honouring *Yes to All* and
   *No to All*.
3. :meth:`ClusterTargetPlan.fresh_start_jobs` names the start-again jobs whose
   local output is cleared before submission.
4. :func:`build_cluster_submissions` builds the :class:`ClusterJobSubmission`
   for every job still going ahead and separates the jobs that fail (a
   start-again job whose directory another job continues in) or have no local
   output directory.

:meth:`ClusterSubmissionPlan.prepare_batch` then prepares each submission and
returns the worker batch.

A directory any job in the batch submits into as-is (a continuation) is
*protected*: it is never deleted, whatever the
overwrite decision of another job whose target collides with it.

Layering: this lives in ``mufinder.services`` and imports ``domain``,
``cluster_setup`` and sibling services only; it holds no widgets.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

from mufinder.cluster_setup import ClusterConfig
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import Job
from mufinder.services.job_artifacts import RetrievalFlags
from mufinder.services.job_execution_controller import (
    ClusterJobSubmission,
    PreparedClusterSubmission,
)

#: The answers to the "remote directory exists" prompt.
OverwriteChoice = Literal["yes", "yes_to_all", "no", "no_to_all"]
#: What happens to an existing remote directory: deleted before submission, or
#: the job is skipped.
OverwriteDecision = Literal["delete", "skip"]


class ClusterSubmissionSettings(Protocol):
    """The submission-dialog settings the planner reads."""

    working_directory: str
    num_processes: int
    walltime: str


def _same_cluster(target: str | None, cluster_name: str) -> bool:
    return (target or "").strip().casefold() == cluster_name.strip().casefold()


def can_reuse_cluster_connection(
    cluster: ClusterConfig,
    *,
    is_connected: bool,
    connected_cluster_name: str | None,
    connected_username: str | None,
) -> bool:
    """Return whether the open connection already reaches ``cluster`` as its user.

    A connection opened without a recorded username matches any user.
    """

    connected_name = (connected_cluster_name or "").strip().casefold()
    connected_user = (connected_username or "").strip()
    return (
        is_connected
        and connected_name == cluster.name.strip().casefold()
        and (not connected_user or connected_user == cluster.username.strip())
    )


def unsafe_job_labels_message(invalid_labels: Iterable[str]) -> str:
    """Return the message asking the user to rename jobs with unsafe remote labels."""

    return (
        "Cluster job labels may contain only letters, numbers, dot, underscore, and hyphen.\n\n"
        "Rename the following jobs before cluster submission:\n"
        + "\n".join(f"- {label}" for label in invalid_labels)
    )


@dataclass(frozen=True, slots=True)
class RemoteTarget:
    """Where one job is submitted on the cluster, and how."""

    job: Job
    remote_target: str
    #: Submit without uploading inputs: a resubmission into its previous directory.
    reuse_existing_files: bool = False
    #: Submit into a directory that already exists, as-is: a continuation.
    allow_existing_remote_directory: bool = False
    #: Start again: delete the job's own directory and upload its inputs afresh.
    fresh_delete: bool = False
    #: The previous run's scheduler id; a start-again deletion waits until it
    #: has finished.
    previous_remote_job_id: str = ""


@dataclass(frozen=True, slots=True)
class OverwritePrompt:
    """One "remote directory exists" question the user must answer."""

    job_id: UUID
    job_label: str
    remote_target: str


@dataclass(frozen=True, slots=True)
class OverwriteDecisions:
    """The resolved overwrite decisions and the messages for skipped jobs."""

    decision_by_job: dict[UUID, OverwriteDecision] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ClusterTargetPlan:
    """Each job's remote target, in submission order."""

    targets: list[RemoteTarget]
    seed_name_by_job: dict[UUID, str]
    fresh_start_job_ids: frozenset[UUID] = frozenset()

    def targets_to_check(self) -> list[tuple[UUID, str]]:
        """Return ``(job_id, remote_target)`` for fresh targets that may already exist."""

        return [
            (target.job.id, target.remote_target)
            for target in self.targets
            if not target.fresh_delete
            and not target.reuse_existing_files
            and not target.allow_existing_remote_directory
        ]

    def protected_remote_targets(self) -> set[str]:
        """Return the directories a job submits into as-is; they are never deleted."""

        return {
            target.remote_target
            for target in self.targets
            if target.allow_existing_remote_directory and target.remote_target
        }

    def overwrite_prompts(self, existing_remote_targets: Collection[str]) -> list[OverwritePrompt]:
        """Return the overwrite questions for targets found in ``existing_remote_targets``.

        A start-again job is deleted without asking, and a directory submitted
        into as-is is never a candidate.
        """

        return [
            OverwritePrompt(target.job.id, target.job.label, target.remote_target)
            for target in self.targets
            if not target.fresh_delete
            and not target.allow_existing_remote_directory
            and target.remote_target
            and target.remote_target in existing_remote_targets
        ]

    def without_jobs(self, job_ids: Collection[UUID]) -> ClusterTargetPlan:
        """Return this plan without the targets of ``job_ids``."""

        return replace(
            self, targets=[target for target in self.targets if target.job.id not in job_ids]
        )

    def fresh_start_jobs(self, decisions: OverwriteDecisions) -> list[Job]:
        """Return the start-again jobs going ahead, whose local output is cleared first."""

        return [
            target.job
            for target in self.targets
            if target.job.id in self.fresh_start_job_ids
            and decisions.decision_by_job.get(target.job.id) != "skip"
        ]


def plan_remote_targets(
    jobs: Sequence[Job],
    *,
    cluster_name: str,
    working_directory: str,
    seed_name_by_job: dict[UUID, str],
    remote_job_directory: Callable[..., str],
    fresh_start_job_ids: Collection[UUID] = (),
    reuse_existing_remote_files: bool = False,
) -> ClusterTargetPlan:
    """Resolve the remote directory each of ``jobs`` is submitted into on ``cluster_name``.

    In priority order:

    - a job in ``fresh_start_job_ids`` whose ``execution_target`` is
      ``cluster_name`` starts again in its own ``remote_directory``, which is
      deleted first once its previous run (``remote_job_id``) has finished;
    - a ``defined`` job with a ``remote_directory`` on ``cluster_name`` is a
      continuation and is submitted into that directory as-is, so CASTEP
      resumes from the ``.check`` file there; on any other cluster it falls
      through to the rules below;
    - with ``reuse_existing_remote_files`` a job resubmits into its previous
      ``remote_directory`` without uploading inputs;
    - otherwise ``remote_job_directory(working_directory=...,
      seed_name=...)`` gives a fresh directory, checked for existence before
      submission.
    """

    fresh_ids = frozenset(fresh_start_job_ids)
    targets: list[RemoteTarget] = []
    for job in jobs:
        own_directory = (job.remote_directory or "").strip()
        on_cluster = _same_cluster(job.execution_target, cluster_name)
        if job.id in fresh_ids and on_cluster and own_directory:
            targets.append(
                RemoteTarget(
                    job=job,
                    remote_target=own_directory,
                    fresh_delete=True,
                    previous_remote_job_id=(job.remote_job_id or "").strip(),
                )
            )
            continue
        # Only ProjectService.prepare_job_continuation leaves a DEFINED job with a
        # remote directory, preserved so CASTEP resumes from its ``.check`` file.
        # On a different cluster that path does not exist, so the job falls
        # through to a recomputed target and the normal existence check.
        if job.status == JobStatus.DEFINED and on_cluster and own_directory:
            targets.append(
                RemoteTarget(
                    job=job, remote_target=own_directory, allow_existing_remote_directory=True
                )
            )
            continue
        if reuse_existing_remote_files and job.id not in fresh_ids and own_directory:
            targets.append(
                RemoteTarget(job=job, remote_target=own_directory, reuse_existing_files=True)
            )
            continue
        targets.append(
            RemoteTarget(
                job=job,
                remote_target=remote_job_directory(
                    working_directory=working_directory,
                    seed_name=seed_name_by_job.get(job.id, job.label),
                ),
            )
        )
    return ClusterTargetPlan(
        targets=targets, seed_name_by_job=dict(seed_name_by_job), fresh_start_job_ids=fresh_ids
    )


def remote_check_failures(plan: ClusterTargetPlan, errors: Mapping[UUID, str]) -> list[str]:
    """Return a failure message for each job in ``errors``, whose directory check raised.

    Such a job is not submitted. A job is in ``errors`` even when the exception
    carried no message.
    """

    return [
        f"- {target.job.label}: not submitted because its remote directory could not be checked "
        f"({errors[target.job.id] or 'no error message'})"
        for target in plan.targets
        if target.job.id in errors
    ]


def resolve_overwrite_decisions(
    plan: ClusterTargetPlan,
    existing_remote_targets: Sequence[str],
    *,
    ask: Callable[[str, str], str],
    confirm_bulk_delete: Callable[[list[str]], bool],
) -> OverwriteDecisions:
    """Decide, for every job whose target exists, whether to delete it or skip the job.

    Start-again jobs are always ``"delete"``. Every other prompt from
    :meth:`ClusterTargetPlan.overwrite_prompts` is put to ``ask(job_label,
    remote_target)`` until the user answers *Yes to All* (confirmed through
    ``confirm_bulk_delete(existing_remote_targets)``; declining it means *No to
    All*) or *No to All*, which then answers the remaining prompts.
    """

    decision_by_job: dict[UUID, OverwriteDecision] = {
        target.job.id: "delete" for target in plan.targets if target.fresh_delete
    }
    skipped: list[str] = []
    overwrite_mode: str | None = None
    for prompt in plan.overwrite_prompts(existing_remote_targets):
        if overwrite_mode is None:
            choice = ask(prompt.job_label, prompt.remote_target)
            if choice == "yes_to_all":
                confirmed = confirm_bulk_delete(list(existing_remote_targets))
                overwrite_mode = "yes_to_all" if confirmed else "no_to_all"
                choice = overwrite_mode
            elif choice == "no_to_all":
                overwrite_mode = "no_to_all"
        else:
            choice = overwrite_mode
        if choice in {"yes", "yes_to_all"}:
            decision_by_job[prompt.job_id] = "delete"
        else:
            decision_by_job[prompt.job_id] = "skip"
            skipped.append(
                f"- {prompt.job_label}: Submission skipped because remote directory already exists "
                f"({prompt.remote_target})"
            )
    return OverwriteDecisions(decision_by_job=decision_by_job, skipped=skipped)


@dataclass(frozen=True, slots=True)
class PlannedClusterSubmission:
    """One job ready to prepare and hand to the cluster worker."""

    submission: ClusterJobSubmission
    remote_dir_to_delete: str | None = None
    delete_only_if_finished: str | None = None

    def batch_item(self, prepared: PreparedClusterSubmission) -> dict[str, object]:
        """Return the worker batch entry for this job once it is prepared."""

        job = self.submission.job
        return {
            "job_id": job.id,
            "label": job.label,
            "submit_kwargs": prepared.submit_kwargs,
            "remote_dir_to_delete": self.remote_dir_to_delete,
            "delete_only_if_finished": self.delete_only_if_finished,
        }


@dataclass(frozen=True, slots=True)
class ClusterSubmissionPlan:
    """The jobs to submit, and those that fail, are skipped or have no local output directory.

    ``failed`` and ``skipped`` hold one ``"- label: reason"`` line per job;
    ``missing`` holds the labels of jobs without a local output directory.
    """

    items: list[PlannedClusterSubmission] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def prepare_batch(
        self,
        prepare: Callable[[ClusterJobSubmission], PreparedClusterSubmission | None],
    ) -> tuple[list[dict[str, object]], dict[UUID, PreparedClusterSubmission]]:
        """Prepare each item and return the worker batch and the prepared state by job.

        ``prepare`` is ``JobExecutionController.prepare_cluster_submission``; a
        job it declines (``None``) is left out of the batch.
        """

        submit_items: list[dict[str, object]] = []
        prepared_by_job: dict[UUID, PreparedClusterSubmission] = {}
        for item in self.items:
            prepared = prepare(item.submission)
            if prepared is None:
                continue
            prepared_by_job[item.submission.job.id] = prepared
            submit_items.append(item.batch_item(prepared))
        return submit_items, prepared_by_job


def build_cluster_submissions(
    plan: ClusterTargetPlan,
    decisions: OverwriteDecisions,
    *,
    cluster: ClusterConfig,
    submission: ClusterSubmissionSettings,
    output_directory: Callable[[Job], Path | None],
    retrieval_flags: Callable[[Job], RetrievalFlags],
    not_cleaned_job_ids: Collection[UUID] = (),
) -> ClusterSubmissionPlan:
    """Build the submission for every job going ahead on ``cluster``.

    A job is left out when its overwrite decision is ``"skip"`` or its local
    output could not be cleared for a fresh start (``not_cleaned_job_ids``). A
    start-again job whose own directory is protected fails, since it would
    resume from the checkpoint the user chose to discard; a job with no local
    output directory is listed in ``missing``. A ``"delete"`` decision deletes
    the target before submission unless the target is protected.
    """

    protected_remote_targets = plan.protected_remote_targets()
    queue_name = getattr(submission, "queue_name", "")
    # The chosen queue's overrides (template, cores per node, executable)
    # apply to every job in this submission.
    queue_cluster = cluster.for_queue(queue_name)
    result = ClusterSubmissionPlan()
    for target in plan.targets:
        job = target.job
        decision = decisions.decision_by_job.get(job.id)
        if decision == "skip" or job.id in not_cleaned_job_ids:
            continue
        if target.fresh_delete and target.remote_target in protected_remote_targets:
            result.failed.append(
                f"- {job.label}: another job in this submission is continuing in its "
                f"directory ({target.remote_target}), so it was not started again"
            )
            continue
        output_dir = output_directory(job)
        if output_dir is None:
            result.missing.append(job.label)
            continue
        result.items.append(
            PlannedClusterSubmission(
                submission=ClusterJobSubmission(
                    job=job,
                    output_dir=output_dir,
                    seed_name=plan.seed_name_by_job.get(job.id, job.label),
                    remote_target=target.remote_target,
                    execution_target=cluster.name,
                    num_processes=submission.num_processes,
                    walltime=submission.walltime,
                    working_directory=submission.working_directory,
                    template_script_path=queue_cluster.template_script_path,
                    cores_per_node=queue_cluster.cores_per_node,
                    memory_gb=getattr(submission, "memory_gb", None),
                    mem_per_cpu_gb=getattr(submission, "mem_per_cpu_gb", None),
                    queue_name=queue_name,
                    comment=getattr(submission, "comment", ""),
                    castep_executable=queue_cluster.castep_executable,
                    reuse_existing_files=target.reuse_existing_files,
                    allow_existing_remote_directory=target.allow_existing_remote_directory,
                    retrieval=retrieval_flags(job),
                ),
                remote_dir_to_delete=(
                    target.remote_target
                    if decision == "delete" and target.remote_target not in protected_remote_targets
                    else None
                ),
                delete_only_if_finished=target.previous_remote_job_id or None,
            )
        )
    return result


def plan_cluster_submission(
    jobs: Sequence[Job],
    *,
    cluster: ClusterConfig,
    submission: ClusterSubmissionSettings,
    seed_name_by_job: dict[UUID, str],
    remote_job_directory: Callable[..., str],
    output_directory: Callable[[Job], Path | None],
    retrieval_flags: Callable[[Job], RetrievalFlags],
    check_remote_directories: Callable[[list[tuple[UUID, str]]], Mapping[str, object]],
    ask_overwrite: Callable[[str, str], str],
    confirm_bulk_delete: Callable[[list[str]], bool],
    clear_fresh_outputs: Callable[[list[Job]], tuple[list[Job], list[str]]],
    fresh_start_job_ids: Collection[UUID] = (),
    reuse_existing_remote_files: bool = False,
) -> ClusterSubmissionPlan:
    """Run every planning stage for ``jobs`` on ``cluster`` and return the plan.

    The callbacks are the steps only the caller can perform:
    ``check_remote_directories(targets)`` returns ``{"existing": [...],
    "errors": {job_id: message}}`` for the fresh targets (called only when
    there are any); a job in ``errors`` is not submitted. ``ask_overwrite`` and ``confirm_bulk_delete`` answer the
    overwrite prompts (see :func:`resolve_overwrite_decisions`), and
    ``clear_fresh_outputs(jobs)`` clears the local output of the start-again
    jobs going ahead, returning the jobs it cleared and an error line for each
    it could not (called only when there are any).

    ``failed`` lists directory-check errors, then local clean-up errors, then
    start-again jobs in a protected directory.
    """

    target_plan = plan_remote_targets(
        jobs,
        cluster_name=cluster.name,
        working_directory=submission.working_directory,
        seed_name_by_job=seed_name_by_job,
        remote_job_directory=remote_job_directory,
        fresh_start_job_ids=fresh_start_job_ids,
        reuse_existing_remote_files=reuse_existing_remote_files,
    )
    failed: list[str] = []
    existing_remote_targets: list[str] = []
    if targets_to_check := target_plan.targets_to_check():
        check_result = check_remote_directories(targets_to_check)
        existing_remote_targets = list(check_result.get("existing", []))
        check_errors: Mapping[UUID, str] = check_result.get("errors", {})
        failed.extend(remote_check_failures(target_plan, check_errors))
        target_plan = target_plan.without_jobs(check_errors)
    decisions = resolve_overwrite_decisions(
        target_plan,
        existing_remote_targets,
        ask=ask_overwrite,
        confirm_bulk_delete=confirm_bulk_delete,
    )
    not_cleaned_job_ids: set[UUID] = set()
    if fresh_jobs := target_plan.fresh_start_jobs(decisions):
        cleaned, cleanup_errors = clear_fresh_outputs(fresh_jobs)
        failed.extend(cleanup_errors)
        not_cleaned_job_ids = {job.id for job in fresh_jobs} - {job.id for job in cleaned}
    planned = build_cluster_submissions(
        target_plan,
        decisions,
        cluster=cluster,
        submission=submission,
        output_directory=output_directory,
        retrieval_flags=retrieval_flags,
        not_cleaned_job_ids=not_cleaned_job_ids,
    )
    return ClusterSubmissionPlan(
        items=planned.items,
        failed=failed + planned.failed,
        skipped=decisions.skipped,
        missing=planned.missing,
    )
