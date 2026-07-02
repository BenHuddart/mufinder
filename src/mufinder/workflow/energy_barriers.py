"""Transition-state-search workflow expansion for Energy Barriers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from mufinder.castep_io.types import AtomicPosition, BlockSection, KeywordAssignment, PositionBlock
from mufinder.domain.enums import WorkflowStage

from .merge import compose_patch, resolve_job
from .naming import job_name
from .types import CellTemplate, InputTemplate, JobPatch, JobSpec, ParamTemplate, StructureSpec


TRANSITION_STATE_SEARCH_CALC_TYPE = "castep_transition_state_search"
TRANSITION_STATE_SEARCH_SET_CALC_TYPE = "castep_transition_state_search_set"
NUDGED_ELASTIC_BAND_CALC_TYPE = "castep_nudged_elastic_band"
NUDGED_ELASTIC_BAND_SET_CALC_TYPE = "castep_nudged_elastic_band_set"


@dataclass(frozen=True, slots=True)
class NudgedElasticBandSource:
    """One diffusion-path endpoint pair expanded into a NEB job."""

    label: str
    reactant_structure: StructureSpec
    # Fractional positions: CASTEP's POSITIONS_FRAC_PRODUCT/_INTERMEDIATE reader accepts tagged
    # species (e.g. the muon "H:mu"); the POSITIONS_ABS_* reader does not and aborts with
    # "Unknown element in cell file".
    product_positions_frac: PositionBlock
    # CASTEP's NEB driver requires an INTERMEDIATE structure (it seeds the central image and
    # interpolates linearly to each endpoint); it does NOT build the band from the two endpoints
    # alone. Callers should supply the minimum-image midpoint so the initial band is a straight
    # reactant->product line. None is only for renderer tests; a real NEB job without it aborts.
    intermediate_positions_frac: PositionBlock | None = None
    product_atomic_spins: tuple[float, ...] | None = None
    intermediate_atomic_spins: tuple[float, ...] | None = None
    source_site_ids: tuple[UUID, UUID] = field(default_factory=tuple)
    source_relaxation_run_ids: tuple[UUID, UUID] = field(default_factory=tuple)
    source_relaxation_job_ids: tuple[UUID | None, UUID | None] = field(default_factory=tuple)
    path_length_ang: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NudgedElasticBandJobConfig:
    """Configuration shared by one generated NEB job."""

    source: NudgedElasticBandSource
    method: str = "NEB"
    cell_overrides: CellTemplate | None = None
    param_overrides: ParamTemplate | None = None


def expand_nudged_elastic_band_jobs(
    *,
    jobs: tuple[NudgedElasticBandJobConfig, ...] | list[NudgedElasticBandJobConfig],
    name_prefix: str,
    global_defaults: InputTemplate | None = None,
    workflow_defaults: InputTemplate | None = None,
    structure_defaults: InputTemplate | None = None,
    user_overrides: InputTemplate | None = None,
    tags: tuple[str, ...] = (),
) -> list[JobSpec]:
    """Expand diffusion-path endpoint pairs into CASTEP NEB jobs."""

    if not jobs:
        raise ValueError("At least one NEB job must be configured.")

    resolved: list[JobSpec] = []
    for config in jobs:
        label = config.source.label.strip()
        if not label:
            raise ValueError("NEB path labels must not be empty.")
        patch_template = compose_patch(
            InputTemplate(cell=_neb_cell_base_template(config), param=_neb_param_base_template(config)),
            InputTemplate(cell=config.cell_overrides, param=config.param_overrides),
        )
        patch = JobPatch(
            label_suffix=label,
            structure=config.source.reactant_structure,
            cell=patch_template.cell,
            param=patch_template.param,
        )
        resolved.append(
            resolve_job(
                name=job_name(name_prefix, label),
                base_structure=config.source.reactant_structure,
                workflow_stage=WorkflowStage.ENERGY_BARRIERS,
                global_defaults=global_defaults,
                workflow_defaults=workflow_defaults,
                structure_defaults=structure_defaults,
                patch=patch,
                user_overrides=user_overrides,
                tags=("energy_barriers", "nudged_elastic_band", *tags),
                provenance={
                    "workflow": "energy_barriers",
                    "calc_type": NUDGED_ELASTIC_BAND_CALC_TYPE,
                    "label": label,
                    "tssearch_method": config.method,
                    "source_site_ids": [str(value) for value in config.source.source_site_ids],
                    "source_relaxation_run_ids": [str(value) for value in config.source.source_relaxation_run_ids],
                    "source_relaxation_job_ids": [
                        str(value) if value is not None else None for value in config.source.source_relaxation_job_ids
                    ],
                    "path_length_ang": config.source.path_length_ang,
                    **dict(config.source.provenance),
                },
            )
        )
    return resolved


def _neb_cell_base_template(config: NudgedElasticBandJobConfig) -> CellTemplate:
    blocks = [
        BlockSection(
            name="positions_frac_product",
            rows=tuple(_rows_from_positions(config.source.product_positions_frac, atomic_spins=config.source.product_atomic_spins)),
            unit=None,
        )
    ]
    if config.source.intermediate_positions_frac is not None:
        blocks.append(
            BlockSection(
                name="positions_frac_intermediate",
                rows=tuple(
                    _rows_from_positions(
                        config.source.intermediate_positions_frac,
                        atomic_spins=config.source.intermediate_atomic_spins,
                    )
                ),
                unit=None,
            )
        )
    return CellTemplate(
        extra_scalars=(KeywordAssignment("fix_com", False),),
        extra_blocks=tuple(blocks),
    )


def _neb_param_base_template(config: NudgedElasticBandJobConfig) -> ParamTemplate:
    return _nudged_elastic_band_param_template(
        method=config.method,
        atomic_spins=config.source.reactant_structure.atomic_spins,
    )


def _nudged_elastic_band_param_template(*, method: str, atomic_spins: tuple[float, ...] | None) -> ParamTemplate:
    # Reuse the shared transition-state template (task + method + spin handling) and
    # add the climbing-image flag so the central bead lands on the saddle (the barrier).
    # The image count (`tssearch_max_path_points`) and tolerances come from the
    # workflow/UI configuration; CASTEP requires an odd image count when climbing.
    base = _transition_state_param_template(method=method, atomic_spins=atomic_spins)
    return ParamTemplate(
        keywords=(
            *base.keywords,
            KeywordAssignment("tssearch_neb_climbing", True),
            # Odd default so the climbing image sits on the saddle without CASTEP silently
            # bumping the count; 11 gives a smoother, better-resolved band than the minimum.
            # User/UI config may override this.
            KeywordAssignment("tssearch_max_path_points", 11),
        )
    )


def _transition_state_param_template(*, method: str, atomic_spins: tuple[float, ...] | None) -> ParamTemplate:
    keywords = [
        KeywordAssignment("task", "TRANSITIONSTATESEARCH"),
        KeywordAssignment("tssearch_method", method),
    ]
    if atomic_spins is not None and any(abs(float(value)) > 1e-12 for value in atomic_spins):
        keywords.append(KeywordAssignment("spin_polarized", True))
    return ParamTemplate(keywords=tuple(keywords))


def _rows_from_positions(
    block: PositionBlock,
    *,
    atomic_spins: tuple[float, ...] | None = None,
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for index, position in enumerate(block.positions):
        values: tuple[object, ...] = (position.species, *position.coordinates)
        if atomic_spins is not None and index < len(atomic_spins) and abs(float(atomic_spins[index])) > 1e-12:
            values = (*values, f"SPIN={float(atomic_spins[index]):.12g}")
        rows.append(values)
    return rows


def position_block_from_structure(structure: StructureSpec) -> PositionBlock:
    """Return a positions block copy from a workflow structure."""

    return PositionBlock(
        positions=tuple(
            AtomicPosition(species=row.species, coordinates=row.coordinates)
            for row in structure.positions_frac.positions
        ),
        unit=structure.positions_frac.unit,
    )