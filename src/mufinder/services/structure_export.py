"""Find a relaxation's relaxed structure and describe it for export.

A completed pristine or muon relaxation stores its relaxed ``-out.cell`` as a
derived :class:`~mufinder.domain.models.Structure` (imported legacy relaxations
are stored the same way). :func:`relaxed_structure_export_for_job` and
:func:`relaxed_structure_export_for_structure` resolve that structure together
with a default file stem and the provenance comment lines an exported file
starts with, or say why there is nothing to export. Writing the file is
:func:`mufinder.science.structure_io.export_structure`; asking where to write it
is the caller's.

Layering: this lives in ``mufinder.services`` and reads the project document
only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from uuid import UUID

from mufinder.domain.charge_states import muon_charge_state_label, muon_charge_state_prefix
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job, Structure, utc_now
from mufinder.services.job_results import find_job_derived_structure, total_energy_for_job
from mufinder.services.project_document import ProjectDocument
from mufinder.services.source_candidates import charge_state_for_job

#: Derived-structure origin of each relaxation stage that has a relaxed structure.
_RELAXATION_ORIGINS = {
    WorkflowStage.PRISTINE_RELAXATION: "pristine_relaxation",
    WorkflowStage.MUON_RELAXATION: "muon_relaxation",
}
_STAGE_TITLES = {
    WorkflowStage.PRISTINE_RELAXATION: "Pristine relaxation",
    WorkflowStage.MUON_RELAXATION: "Muon relaxation",
}

NOT_COMPLETED_REASON = "Available once the relaxation has completed."
NO_STRUCTURE_REASON = (
    "This relaxation has no relaxed structure: its -out.cell was missing or could not be parsed."
)
NOT_A_RELAXATION_REASON = "Only pristine and muon relaxations have a relaxed structure."


@dataclass(frozen=True, slots=True)
class RelaxedStructureExport:
    """A relaxed structure ready to export, or the reason there is none.

    ``structure`` is ``None`` exactly when ``unavailable_reason`` is set.
    ``default_stem`` is a file name without extension (``"mu1_muonium_relaxed"``),
    ``header_lines`` the comment lines to write at the top of the file and
    ``default_directory`` the folder a save dialog starts in (the folder that
    holds the project).
    """

    structure: Structure | None
    unavailable_reason: str | None = None
    default_stem: str = ""
    header_lines: tuple[str, ...] = ()
    default_directory: Path | None = None


def relaxed_structure_export_for_job(
    document: ProjectDocument | None, job: Job | None, *, now: datetime | None = None
) -> RelaxedStructureExport:
    """Return ``job``'s relaxed structure for export, or why it cannot be exported.

    Only a completed pristine or muon relaxation qualifies, and only when its
    completion stored a relaxed structure (see
    :func:`~mufinder.services.job_results.find_job_derived_structure`).
    """

    if document is None or job is None:
        return RelaxedStructureExport(None, NOT_COMPLETED_REASON)
    origin = _RELAXATION_ORIGINS.get(job.workflow_stage)
    if origin is None:
        return RelaxedStructureExport(None, NOT_A_RELAXATION_REASON)
    if job.status != JobStatus.COMPLETED:
        return RelaxedStructureExport(None, NOT_COMPLETED_REASON)
    structure = find_job_derived_structure(document, job, origin=origin)
    if structure is None:
        return RelaxedStructureExport(None, NO_STRUCTURE_REASON)
    return _export(document, structure, job, now=now)


def relaxed_structure_export_for_job_id(
    document: ProjectDocument | None, job_id: UUID, *, now: datetime | None = None
) -> RelaxedStructureExport:
    """Return :func:`relaxed_structure_export_for_job` for the job with id ``job_id``."""

    job = document.jobs.get(job_id) if document is not None else None
    return relaxed_structure_export_for_job(document, job, now=now)


def relaxed_structure_export_for_structure(
    document: ProjectDocument | None, structure: Structure | None, *, now: datetime | None = None
) -> RelaxedStructureExport:
    """Return a relaxed structure already in hand for export, with its job's provenance.

    The job is found through the structure's ``source_job_id`` property; a
    structure without one still exports, with a header that names the
    structure only.
    """

    if document is None or structure is None:
        return RelaxedStructureExport(None, NO_STRUCTURE_REASON)
    job = _source_job(document, structure)
    return _export(document, structure, job, now=now)


def _source_job(document: ProjectDocument, structure: Structure) -> Job | None:
    raw = structure.properties.get("source_job_id")
    try:
        return document.jobs.get(UUID(str(raw))) if raw else None
    except ValueError:
        return None


def _export(
    document: ProjectDocument, structure: Structure, job: Job | None, *, now: datetime | None
) -> RelaxedStructureExport:
    charge_state = charge_state_for_job(job)
    if job is not None:
        stem_parts = [job.label]
        if charge_state is not None:
            stem_parts.append(muon_charge_state_prefix(charge_state))
        stem_parts.append("relaxed")
    else:
        stem_parts = [structure.label]
    default_stem = _file_stem("_".join(part for part in stem_parts if part)) or "relaxed_structure"

    timestamp = (now or utc_now()).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    lines = [
        "Relaxed structure exported by MuFinder",
        f"Project: {document.project.name}",
        f"Structure: {structure.label}",
    ]
    if job is not None:
        stage = _STAGE_TITLES.get(job.workflow_stage, job.workflow_stage.value)
        lines.append(f"Source job: {job.label} ({stage}, id {job.id})")
        if charge_state is not None:
            lines.append(f"Charge state: {muon_charge_state_label(charge_state)}")
        energy = total_energy_for_job(document, job.id)
        if energy is not None:
            lines.append(f"Final energy: {energy:.6f} eV")
    if structure.properties.get("import_source"):
        lines.append("Imported from a legacy MuFinder project")
    lines.append(f"Exported: {timestamp}")
    return RelaxedStructureExport(
        structure=structure,
        default_stem=default_stem,
        header_lines=tuple(lines),
        default_directory=document.project_root.parent,
    )


def _file_stem(text: str) -> str:
    """Lower-case ``text`` and keep letters, digits, ``-`` and ``.``, joining the rest with ``_``."""

    return re.sub(r"[^a-z0-9.\-]+", "_", text.strip().lower()).strip("_.")


__all__ = [
    "NOT_A_RELAXATION_REASON",
    "NOT_COMPLETED_REASON",
    "NO_STRUCTURE_REASON",
    "RelaxedStructureExport",
    "relaxed_structure_export_for_job",
    "relaxed_structure_export_for_job_id",
    "relaxed_structure_export_for_structure",
]
