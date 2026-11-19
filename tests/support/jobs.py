"""Job-record factory shared by the app and services tests."""

from __future__ import annotations

from typing import Any

from mufinder.domain.enums import EndReason, JobActivity, JobStatus, WorkflowStage
from mufinder.domain.models import Job
from mufinder.services.project_service import ProjectDocument, ProjectService

#: The stored state each status word the job tables show stands for. A stopped
#: job stopped at its walltime and a failed one for no identified reason.
STATE_FIELDS: dict[str, dict[str, Any]] = {
    "defined": {"status": JobStatus.DEFINED},
    **{
        activity.value: {"status": JobStatus.ACTIVE, "activity": activity}
        for activity in JobActivity
    },
    "completed": {"status": JobStatus.COMPLETED},
    "failed": {"status": JobStatus.ENDED, "end_reason": EndReason.UNKNOWN},
    "stopped": {"status": JobStatus.ENDED, "end_reason": EndReason.WALLTIME},
    "cancelled": {"status": JobStatus.ENDED, "end_reason": EndReason.CANCELLED},
}


def state_fields(word: str) -> dict[str, Any]:
    """Return the ``Job`` fields that put a job in the state ``word`` names.

    Every field is given, so the result also clears what another state set. A
    ``JobStatus`` that is not a word (``ACTIVE``, ``ENDED``) sets the status alone.
    """

    state = STATE_FIELDS[str(word)] if str(word) in STATE_FIELDS else {"status": JobStatus(word)}
    return {"activity": None, "end_reason": None, **state}


def set_job_state(job: Job, word: str) -> Job:
    """Put ``job`` in the state ``word`` names (``"running"``, ``"failed"``, ...)."""

    for name, value in state_fields(word).items():
        setattr(job, name, value)
    return job


def make_job(
    label: str = "job",
    *,
    status: JobStatus | None = None,
    workflow_stage: WorkflowStage = WorkflowStage.CONVERGENCE_CUTOFF,
    calc_type: str = "castep_cutoff",
    service: ProjectService | None = None,
    document: ProjectDocument | None = None,
    **fields: Any,
) -> Job:
    """Build a job, either standalone or registered in a project.

    Without ``service`` and ``document`` this returns a bare ``Job`` (status
    ``DEFINED`` unless ``status`` is given) and ``fields`` sets any other
    ``Job`` field. ``status`` may also be a status word (``"running"``,
    ``"stopped"``), which sets the activity or end reason with it. With
    ``service`` and ``document``, the job is created through
    ``ProjectService.add_job`` so it lands in ``document.jobs`` with the
    service's tags and creation order; ``fields`` then accepts only what
    ``add_job`` does (``execution_target``), and the status is always
    ``DEFINED``.
    """

    if service is None and document is None:
        state = {
            name: value
            for name, value in state_fields(status or JobStatus.DEFINED).items()
            if value is not None
        }
        return Job(
            label=label,
            workflow_stage=workflow_stage,
            calc_type=calc_type,
            **{**state, **fields},
        )
    if service is None or document is None:
        raise TypeError("make_job needs both service and document to register a job")
    if status is not None:
        raise TypeError("ProjectService.add_job always creates DEFINED jobs; set status afterwards")
    return service.add_job(
        document,
        label=label,
        workflow_stage=workflow_stage,
        calc_type=calc_type,
        **fields,
    )
