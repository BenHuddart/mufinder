"""Where a UEP search's positions sit: its source structure and its lattice.

The searches themselves are stored results (:mod:`mufinder.services.uep_results`).
These helpers place a search's positions in the host cell for the readers of its
groups: Generation, Magnetic Fields and the project survey.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from mufinder.domain.models import Structure
from mufinder.domain.enums import WorkflowStage
from mufinder.persistence.codec import (
    uuid_or_none,
    float_triplet,
    float_triplet_or_none,
)
from mufinder.science.lattice import (
    MUON_FRACTIONAL_SNAP_TOLERANCE,
    LatticeError,
    cell_volume,
    lattice_cart_to_frac,
    wrap_fractional,
)
from mufinder.services.project_document import ProjectDocument


def uep_source_structure_id(document: ProjectDocument, job_id: UUID) -> UUID | None:
    """Return the structure a UEP source job was configured on, ``None`` when it has none."""

    for run in document.runs.values():
        if run.job_id != job_id or run.definition_id is None:
            continue
        definition = document.calculations.get(run.definition_id)
        if definition is not None and definition.input_structure_ids:
            return definition.input_structure_ids[0]
    for definition in document.calculations.values():
        if definition.workflow_stage != WorkflowStage.UEP:
            continue
        mapping = definition.parameters.get("job_ids_by_label")
        if (
            isinstance(mapping, dict)
            and str(job_id) in {str(value) for value in mapping.values()}
            and definition.input_structure_ids
        ):
            return definition.input_structure_ids[0]
    return None


def uep_search_lattice(
    document: ProjectDocument, host_structure: Structure, raw_job_id: object
) -> list[list[float]]:
    """Return the lattice that turns a UEP search's Cartesian positions into host fractionals.

    That is the structure the search was configured on, so a relaxed copy of the host
    keeps the fractional position its groups use. A source that is not one host cell,
    such as a relaxed supercell, or a source that is gone, leaves the host's lattice.
    """

    job_id = uuid_or_none(raw_job_id)
    source_id = None if job_id is None else uep_source_structure_id(document, job_id)
    source = document.structures.get(source_id) if source_id is not None else None
    host_volume = cell_volume(host_structure.lattice_vectors)
    if source is None or abs(cell_volume(source.lattice_vectors) - host_volume) > host_volume / 2:
        return host_structure.lattice_vectors
    return source.lattice_vectors


def uep_result_host_position(
    result: Mapping[str, object], lattice: Sequence[Sequence[float]]
) -> tuple[float, float, float] | None:
    """Return a UEP result's position in the host cell, ``None`` when it cannot be worked out.

    A result's stored fractional position is in the cell the job ran on, which may be a
    supercell, so the position comes from its Cartesian one, wrapped into the cell.
    """

    cartesian = float_triplet_or_none(result.get("optimized_position_ang"))
    if cartesian is None:
        return None
    try:
        fractional = lattice_cart_to_frac(cartesian, lattice)
    except LatticeError:
        return None
    return float_triplet(wrap_fractional(fractional, snap_tolerance=MUON_FRACTIONAL_SNAP_TOLERANCE))


__all__ = [
    "uep_source_structure_id",
    "uep_search_lattice",
    "uep_result_host_position",
]
