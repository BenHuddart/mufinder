"""Job-record factory shared by the app and services tests."""

from __future__ import annotations

from typing import Any

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job
from mufinder.services.project_service import ProjectDocument, ProjectService


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
    ``Job`` field. With both, the job is created through
    ``ProjectService.add_job`` so it lands in ``document.jobs`` with the
    service's tags and creation order; ``fields`` then accepts only what
    ``add_job`` does (``execution_target``), and the status is always
    ``DEFINED``.
    """

    if service is None and document is None:
        return Job(
            label=label,
            workflow_stage=workflow_stage,
            calc_type=calc_type,
            status=JobStatus.DEFINED if status is None else status,
            **fields,
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
