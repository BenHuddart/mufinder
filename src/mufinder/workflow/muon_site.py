"""Muon-site workflow expansion into resolved JobSpec objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mufinder.castep_io.types import AtomicPosition, PositionBlock
from mufinder.domain.enums import WorkflowStage

from .merge import resolve_job
from .naming import job_name
from .types import InputTemplate, JobPatch, JobSpec, StructureSpec


@dataclass(slots=True)
class MuonSiteWorkflow:
    """Expand muon initial positions into resolved jobs."""

    host_structure: StructureSpec
    muon_positions_frac: Sequence[tuple[float, float, float] | list[float]]
    muon_species_label: str = "H:mu"
    name_prefix: str = "muon_site"
    workflow_stage: WorkflowStage = WorkflowStage.MUON_RELAXATION
    global_defaults: InputTemplate | None = None
    workflow_defaults: InputTemplate | None = None
    structure_defaults: InputTemplate | None = None
    user_overrides: InputTemplate | None = None
    tags: tuple[str, ...] = ()

    def expand(self) -> list[JobSpec]:
        """Expand muon positions into resolved JobSpec objects."""

        if not self.muon_positions_frac:
            raise ValueError("Muon-site workflow requires at least one position.")

        jobs: list[JobSpec] = []
        for index, raw_position in enumerate(self.muon_positions_frac, start=1):
            if len(raw_position) != 3:
                raise ValueError(f"Muon position {index} must have three components.")

            position = (float(raw_position[0]), float(raw_position[1]), float(raw_position[2]))
            structure = self._append_muon(self.host_structure, position, index=index)
            patch = JobPatch(label_suffix=f"site_{index:03d}", structure=structure)
            name = job_name(self.name_prefix, patch.label_suffix)

            jobs.append(
                resolve_job(
                    name=name,
                    base_structure=self.host_structure,
                    workflow_stage=self.workflow_stage,
                    global_defaults=self.global_defaults,
                    workflow_defaults=self.workflow_defaults,
                    structure_defaults=self.structure_defaults,
                    patch=patch,
                    user_overrides=self.user_overrides,
                    tags=self.tags,
                    provenance={
                        "workflow": "muon_site",
                        "muon_species_label": self.muon_species_label,
                        "muon_position_frac": position,
                        "site_index": index,
                    },
                )
            )

        return jobs

    def _append_muon(
        self,
        structure: StructureSpec,
        position: tuple[float, float, float],
        *,
        index: int,
    ) -> StructureSpec:
        wrapped = tuple(component % 1.0 for component in position)
        muon_position = AtomicPosition(species=self.muon_species_label, coordinates=wrapped)  # type: ignore[arg-type]
        positions = tuple(structure.positions_frac.positions) + (muon_position,)
        species = tuple(dict.fromkeys((*structure.species, self.muon_species_label)))
        atomic_spins = (
            None
            if structure.atomic_spins is None
            else tuple(structure.atomic_spins) + (0.0,)
        )
        return StructureSpec(
            lattice_cart=structure.lattice_cart,
            positions_frac=PositionBlock(positions=positions),
            species=species,
            atomic_spins=atomic_spins,
            label=f"{structure.label}_muon_{index:03d}" if structure.label else f"muon_{index:03d}",
            source_structure_id=structure.source_structure_id,
        )
