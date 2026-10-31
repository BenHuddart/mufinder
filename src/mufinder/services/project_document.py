"""The in-memory project aggregate and the read-only queries over it.

Every project service module works on a :class:`ProjectDocument`;
:mod:`mufinder.services.project_service` re-exports the names here.
:func:`append_provenance` is the one write here: the stream is append-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from mufinder.domain.charge_states import MUON_RELAXATION_JOB_PREFIXES
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    ComparisonTable,
    GenerationRun,
    Job,
    MuonSite,
    Project,
    ProvenanceRecord,
    Result,
    Structure,
)
from mufinder.persistence.codec import to_jsonable
from mufinder.persistence.json_store import ProjectStore


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
    generation_runs: dict[UUID, GenerationRun] = field(default_factory=dict)
    jobs: dict[UUID, Job] = field(default_factory=dict)
    provenance: list[ProvenanceRecord] = field(default_factory=list)
    ui_state: dict[str, Any] = field(default_factory=dict)


def active_host_structure(document: ProjectDocument) -> Structure | None:
    """Return the currently selected host structure if available."""
    host_id = document.project.host_structure_id
    if host_id is None:
        return None
    return document.structures.get(host_id)


def append_provenance(document: ProjectDocument, record: ProvenanceRecord) -> None:
    """Append a provenance record to the project's stream on disk, then in memory.

    In that order, so a record that could not be written is not shown as if it had been.
    """

    ProjectStore(document.project_root).append_provenance(to_jsonable(record))
    document.provenance.append(record)


def definitions_for_stage(
    document: ProjectDocument,
    stage: WorkflowStage,
    *,
    calc_type: str | None = None,
) -> list[CalculationDefinition]:
    """Return ``document``'s definitions for ``stage``, newest first.

    ``calc_type`` narrows the match to definitions of that calc type; ``None``
    accepts any calc type at ``stage``. Registration gives every definition a
    strictly later ``created_at`` than the last, so this is creation order;
    definitions that share a timestamp are ordered by id, so the order never
    depends on the order the document loaded them in.
    """

    definitions = [
        definition
        for definition in document.calculations.values()
        if definition.workflow_stage == stage
        and (calc_type is None or definition.calc_type == calc_type)
    ]
    definitions.sort(
        key=lambda definition: (definition.created_at, str(definition.id)), reverse=True
    )
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


#: Workflows whose definitions bind jobs to named rows through ``job_ids_by_label``.
#: Quantum Corrections keys its rows by the relaxed site's id (and, for a second
#: charge state, ``"{site id}:{prefix}"``), so its row is named by the site.
_LABELLED_ROW_STAGES = frozenset(
    {
        WorkflowStage.PRISTINE_RELAXATION,
        WorkflowStage.UEP,
        WorkflowStage.MUON_RELAXATION,
        WorkflowStage.QUANTUM_CORRECTIONS,
    }
)


def job_row_label(document: ProjectDocument, job: Job) -> str | None:
    """Return the label of the workflow row that owns ``job``, as the workflow's table shows it.

    Read from the ``job_ids_by_label`` mapping (``phonon_job_ids_by_label``
    for Quantum Corrections) of the job's workflow definitions: a muon-relaxation key's charge-state prefix is dropped, and a
    Quantum Corrections key names the relaxed site, whose label is returned
    (with ``_<prefix>`` for a source in the site's second charge state).
    ``None`` when no definition binds the job, or for a workflow whose rows are
    not named (the convergence sweeps, Energy Barriers).
    """

    if job.workflow_stage not in _LABELLED_ROW_STAGES:
        return None
    job_key = str(job.id)
    for definition in definitions_for_stage(document, job.workflow_stage):
        for key, raw_job_id in definition_row_jobs(definition):
            if raw_job_id == job_key:
                return _row_label_from_key(document, key)
    return None


def definition_row_jobs(definition: CalculationDefinition) -> list[tuple[str, str]]:
    """Return the ``(row key, job id text)`` pairs ``definition`` binds."""

    return [
        (str(key), str(raw_job_id))
        for mapping_name in _ROW_JOB_MAPPINGS
        if isinstance(mapping := definition.parameters.get(mapping_name), dict)
        for key, raw_job_id in mapping.items()
    ]


#: The mappings that bind a definition's rows to jobs: Quantum Corrections keeps
#: its phonon jobs in ``phonon_job_ids_by_label`` and may mirror them in
#: ``job_ids_by_label``.
_ROW_JOB_MAPPINGS = ("phonon_job_ids_by_label", "job_ids_by_label")


_CHARGE_STATE_PREFIXES = frozenset(MUON_RELAXATION_JOB_PREFIXES.values())


def _row_label_from_key(document: ProjectDocument, key: str) -> str | None:
    head, separator, tail = key.partition(":")
    suffix = ""
    if separator and head in _CHARGE_STATE_PREFIXES:
        key = tail
    elif separator and tail in _CHARGE_STATE_PREFIXES:
        # A Quantum Corrections source relaxed in the site's second charge
        # state; its table names it ``"{site label}_{prefix}"``.
        key, suffix = head, f"_{tail}"
    try:
        site_id = UUID(key)
    except ValueError:
        return key.strip() or None
    site = document.sites.get(site_id)
    return f"{site.label}{suffix}" if site is not None and site.label else None
