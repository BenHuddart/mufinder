"""Muon-site workflow expansion into resolved JobSpec objects."""

from __future__ import annotations

from typing import Sequence

from mufinder.castep_io.types import AtomicPosition, PositionBlock
from mufinder.domain.enums import WorkflowStage

from ._common import require_jobs
from .merge import resolve_job
from .naming import job_name
from .types import InputTemplate, JobPatch, JobSpec, StructureSpec

#: Calc type of the definition that holds a host's muon-relaxation rows.
MUON_RELAXATION_DEFINITION_CALC_TYPE = "castep_muon_relaxation"
#: ``ui_state`` key naming the muon-relaxation definition the workflow works on.
ACTIVE_MUON_RELAXATION_DEFINITION_KEY = "active_muon_relaxation_definition_id"


def _append_muon(
    structure: StructureSpec,
    position: tuple[float, float, float],
    *,
    muon_species_label: str,
    index: int,
) -> StructureSpec:
    wrapped = tuple(component % 1.0 for component in position)
    muon_position = AtomicPosition(species=muon_species_label, coordinates=wrapped)  # type: ignore[arg-type]
    positions = tuple(structure.positions_frac.positions) + (muon_position,)
    species = tuple(dict.fromkeys((*structure.species, muon_species_label)))
    atomic_spins = (
        None if structure.atomic_spins is None else tuple(structure.atomic_spins) + (0.0,)
    )
    return StructureSpec(
        lattice_cart=structure.lattice_cart,
        positions_frac=PositionBlock(positions=positions),
        species=species,
        atomic_spins=atomic_spins,
        label=f"{structure.label}_muon_{index:03d}" if structure.label else f"muon_{index:03d}",
        source_structure_id=structure.source_structure_id,
    )


def expand_muon_site_jobs(
    *,
    host_structure: StructureSpec,
    muon_positions_frac: Sequence[tuple[float, float, float] | list[float]],
    muon_species_label: str = "H:mu",
    name_prefix: str = "muon_site",
    workflow_stage: WorkflowStage = WorkflowStage.MUON_RELAXATION,
    global_defaults: InputTemplate | None = None,
    workflow_defaults: InputTemplate | None = None,
    structure_defaults: InputTemplate | None = None,
    user_overrides: InputTemplate | None = None,
    tags: tuple[str, ...] = (),
) -> list[JobSpec]:
    """Expand muon initial positions into resolved JobSpec objects.

    Appends one muon per fractional position to a copy of ``host_structure``,
    layering the supplied ``InputTemplate`` arguments in
    :data:`~mufinder.workflow.merge.MERGE_PRECEDENCE_ORDER`.
    """

    require_jobs(muon_positions_frac, "muon position")

    jobs: list[JobSpec] = []
    for index, raw_position in enumerate(muon_positions_frac, start=1):
        if len(raw_position) != 3:
            raise ValueError(f"Muon position {index} must have three components.")

        position = (float(raw_position[0]), float(raw_position[1]), float(raw_position[2]))
        structure = _append_muon(
            host_structure,
            position,
            muon_species_label=muon_species_label,
            index=index,
        )
        patch = JobPatch(label_suffix=f"site_{index:03d}", structure=structure)
        name = job_name(name_prefix, patch.label_suffix)

        jobs.append(
            resolve_job(
                name=name,
                base_structure=host_structure,
                workflow_stage=workflow_stage,
                global_defaults=global_defaults,
                workflow_defaults=workflow_defaults,
                structure_defaults=structure_defaults,
                patch=patch,
                user_overrides=user_overrides,
                tags=tags,
                provenance={
                    "workflow": "muon_site",
                    "muon_species_label": muon_species_label,
                    "muon_position_frac": position,
                    "site_index": index,
                },
            )
        )

    return jobs
