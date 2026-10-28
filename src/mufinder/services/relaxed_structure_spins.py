"""The ``SPIN=`` values a relaxation leaves on its relaxed structure.

A pristine or muon relaxation run in MuFinder derives a "<job> relaxed"
structure from its ``-out.cell``. CASTEP echoes the run's ``SPIN=`` values into
that file, and :func:`apply_relaxed_cell_spins` stores them on the structure as
``properties["atomic_spins"]``. Everything started from the relaxed structure
then inherits them: ``workflow.bridge.structure_to_spec`` prefers that
property, so the ``.cell`` export, quantum corrections, energy barriers and any
workflow that takes a relaxed pristine structure as its host render the run's
spins.

A project stored below schema :data:`SPINS_KEPT_SCHEMA_VERSION` has relaxed
structures without spins. ``ProjectService.load_project`` calls
:func:`backfill_relaxed_structure_spins` for such a project, which reads them
from the source job's ``-out.cell`` or, failing that, its input ``.cell``.

Layering: this lives in ``mufinder.services`` and imports ``castep_io``,
``domain`` and ``services.job_paths``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from mufinder.castep_io.cell_parser import CellParseError, ParsedCell, parse_cell_file
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.magnetic_properties import has_nonzero_spins
from mufinder.domain.models import Job, Structure
from mufinder.services.job_paths import job_seed_name

if TYPE_CHECKING:
    from mufinder.services.project_service import ProjectDocument

#: Workflow stages whose completion derives a relaxed structure from ``-out.cell``.
RELAXATION_STAGES = frozenset({WorkflowStage.PRISTINE_RELAXATION, WorkflowStage.MUON_RELAXATION})

#: The first saved-project schema whose relaxed structures keep their spins.
SPINS_KEPT_SCHEMA_VERSION = 6


def relaxed_cell_spins(cell: ParsedCell) -> list[float] | None:
    """Return ``cell``'s per-atom ``SPIN=`` values, or ``None`` when it has none.

    A cell whose spins are all zero, or whose spin count does not match its
    atom count, counts as having none.
    """

    spins = cell.atomic_spins
    if spins is None or len(spins) != len(cell.species) or not has_nonzero_spins(spins):
        return None
    return [float(value) for value in spins]


def apply_relaxed_cell_spins(structure: Structure, cell: ParsedCell) -> None:
    """Set ``structure``'s ``atomic_spins`` to ``cell``'s, removing stale ones.

    Called with the cell ``structure``'s geometry came from, so the spins always
    describe that geometry: a continuation whose ``-out.cell`` has no spins
    leaves the structure with none.
    """

    spins = relaxed_cell_spins(cell)
    if spins is None:
        structure.properties.pop("atomic_spins", None)
    else:
        structure.properties["atomic_spins"] = spins


def relaxation_run_spins(output_dir: Path, seed_name: str, atom_count: int) -> list[float] | None:
    """Return the ``SPIN=`` values of the relaxation run in ``output_dir``.

    Reads ``{seed_name}-out.cell``, then the input ``{seed_name}.cell``. A file
    that is missing, cannot be parsed, has a different atom count or carries no
    nonzero spin is skipped.
    """

    for name in (f"{seed_name}-out.cell", f"{seed_name}.cell"):
        path = output_dir / name
        if not path.is_file():
            continue
        try:
            cell = parse_cell_file(path)
        except (CellParseError, OSError, ValueError):
            continue
        if len(cell.species) != atom_count:
            continue
        spins = relaxed_cell_spins(cell)
        if spins is not None:
            return spins
    return None


def backfill_relaxed_structure_spins(
    document: ProjectDocument,
    output_directory: Callable[[Job], Path | None],
) -> list[UUID]:
    """Give relaxed structures saved without spins their run's ``SPIN=`` values.

    A structure is considered when it has no ``atomic_spins`` key and a
    relaxation job derived it: the job its ``source_job_id`` names, else the job
    whose run's result lists it in ``derived_structure_ids`` (as
    ``job_results.find_job_derived_structure`` resolves it). A structure that
    already has the key, even with no nonzero value, is left as it is. Returns
    the ids of the structures that gained spins.
    """

    changed: list[UUID] = []
    jobs_by_id = {str(job.id): job for job in document.jobs.values()}
    jobs_by_structure = _jobs_by_derived_structure(document)
    for structure in document.structures.values():
        properties = structure.properties
        if "atomic_spins" in properties:
            continue
        job = jobs_by_id.get(str(properties.get("source_job_id", ""))) or jobs_by_structure.get(
            structure.id
        )
        if job is None or job.workflow_stage not in RELAXATION_STAGES:
            continue
        output_dir = output_directory(job)
        if output_dir is None:
            continue
        seed_name = job_seed_name(job, lambda _job: output_dir)
        spins = relaxation_run_spins(output_dir, seed_name, len(structure.species))
        if spins is None:
            continue
        properties["atomic_spins"] = spins
        changed.append(structure.id)
    return changed


def _jobs_by_derived_structure(document: ProjectDocument) -> dict[UUID, Job]:
    """Map each derived structure id to the job whose run's result lists it.

    Runs are taken in ``run_index`` order and the first job found wins.
    """

    jobs: dict[UUID, Job] = {}
    for run in sorted(document.runs.values(), key=lambda run: run.run_index):
        result = document.results.get(run.result_id) if run.result_id else None
        job = document.jobs.get(run.job_id)
        if result is None or job is None:
            continue
        for structure_id in result.derived_structure_ids:
            jobs.setdefault(structure_id, job)
    return jobs


__all__ = [
    "RELAXATION_STAGES",
    "SPINS_KEPT_SCHEMA_VERSION",
    "apply_relaxed_cell_spins",
    "backfill_relaxed_structure_spins",
    "relaxation_run_spins",
    "relaxed_cell_spins",
]
