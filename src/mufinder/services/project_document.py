"""The in-memory project aggregate and the read-only queries over it.

Every project service module works on a :class:`ProjectDocument`;
:mod:`mufinder.services.project_service` re-exports the names here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    ComparisonTable,
    Job,
    MuonSite,
    Project,
    ProvenanceRecord,
    Result,
    Structure,
)


@dataclass(slots=True)
class ProjectDocument:
    """In-memory aggregate for all project content currently loaded."""

    project_root: Path
    project: Project
    structures: dict[UUID, Structure] = field(default_factory=dict)
    sites: dict[UUID, MuonSite] = field(default_factory=dict)
    calculations: dict[UUID, CalculationDefinition] = field(default_factory=dict)
    runs: dict[UUID, CalculationRun] = field(default_factory=dict)
    results: dict[UUID, Result] = field(default_factory=dict)
    comparison_tables: dict[UUID, ComparisonTable] = field(default_factory=dict)
    jobs: dict[UUID, Job] = field(default_factory=dict)
    provenance: list[ProvenanceRecord] = field(default_factory=list)
    ui_state: dict[str, Any] = field(default_factory=dict)


def active_host_structure(document: ProjectDocument) -> Structure | None:
    """Return the currently selected host structure if available."""
    host_id = document.project.host_structure_id
    if host_id is None:
        return None
    return document.structures.get(host_id)


def definitions_for_stage(
    document: ProjectDocument,
    stage: WorkflowStage,
    *,
    calc_type: str | None = None,
) -> list[CalculationDefinition]:
    """Return ``document``'s definitions for ``stage``, newest first.

    ``calc_type`` narrows the match to definitions of that calc type; ``None``
    accepts any calc type at ``stage``. Definitions created at the same instant
    keep their insertion order.
    """

    definitions = [
        definition
        for definition in document.calculations.values()
        if definition.workflow_stage == stage
        and (calc_type is None or definition.calc_type == calc_type)
    ]
    definitions.sort(key=lambda definition: definition.created_at, reverse=True)
    return definitions


def active_definition(
    document: ProjectDocument,
    stage: WorkflowStage,
    *,
    calc_type: str | None = None,
    active_key: str | None = None,
) -> CalculationDefinition | None:
    """Return the definition a workflow currently works on.

    When ``active_key`` names a ``ui_state`` entry holding the id of a
    definition that matches ``stage`` (and ``calc_type``, when given), that
    definition wins. Otherwise, including when ``active_key`` is ``None``, the
    newest matching definition is returned, or ``None`` when there is none.
    """

    if active_key is not None:
        raw_definition_id = document.ui_state.get(active_key)
        if isinstance(raw_definition_id, str):
            try:
                definition_id = UUID(raw_definition_id)
            except ValueError:
                definition_id = None
            definition = document.calculations.get(definition_id) if definition_id else None
            if (
                definition is not None
                and definition.workflow_stage == stage
                and (calc_type is None or definition.calc_type == calc_type)
            ):
                return definition

    definitions = definitions_for_stage(document, stage, calc_type=calc_type)
    return definitions[0] if definitions else None


def job_for_mapping_key(
    document: ProjectDocument,
    definition: CalculationDefinition | None,
    key: str,
    *,
    mapping_name: str = "job_ids_by_label",
    calc_type: str | None = None,
) -> Job | None:
    """Return the job ``definition.parameters[mapping_name][key]`` points at.

    Returns ``None`` when the definition is missing, the mapping is absent or
    not a mapping, the key is unmapped, the stored id is not a UUID string, the
    job no longer exists, or ``calc_type`` is given and the job's differs.
    """

    if definition is None:
        return None
    mapping = definition.parameters.get(mapping_name)
    if not isinstance(mapping, dict):
        return None
    raw_job_id = mapping.get(key)
    if not isinstance(raw_job_id, str):
        return None
    try:
        job_id = UUID(raw_job_id)
    except ValueError:
        return None
    job = document.jobs.get(job_id)
    if job is None or (calc_type is not None and job.calc_type != calc_type):
        return None
    return job
