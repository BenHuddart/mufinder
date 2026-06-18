"""Transition-state-search workflow expansion for Energy Barriers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from mufinder.castep_io.types import AtomicPosition, BlockSection, KeywordAssignment, PositionBlock
from mufinder.domain.enums import WorkflowStage

from .merge import merge_templates, resolve_job
from .naming import job_name
from .types import CellTemplate, InputTemplate, JobPatch, JobSpec, ParamTemplate, StructureSpec


TRANSITION_STATE_SEARCH_CALC_TYPE = "castep_transition_state_search"
TRANSITION_STATE_SEARCH_SET_CALC_TYPE = "castep_transition_state_search_set"
NUDGED_ELASTIC_BAND_CALC_TYPE = "castep_nudged_elastic_band"
NUDGED_ELASTIC_BAND_SET_CALC_TYPE = "castep_nudged_elastic_band_set"


@dataclass(frozen=True, slots=True)
class TransitionStateSearchSource:
    """One resolved ordered path source for a transition-state search."""

    label: str
    reactant_structure: StructureSpec
    product_positions_frac: PositionBlock
    product_atomic_spins: tuple[float, ...] | None = None
    source_site_ids: tuple[UUID, UUID] = field(default_factory=tuple)
    source_relaxation_run_ids: tuple[UUID, UUID] = field(default_factory=tuple)
    source_relaxation_job_ids: tuple[UUID | None, UUID | None] = field(default_factory=tuple)
    path_start_fractional: tuple[float, float, float] = (0.0, 0.0, 0.0)
    path_end_fractional: tuple[float, float, float] = (0.0, 0.0, 0.0)
    path_length_ang: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransitionStateSearchJobConfig:
    """Configuration shared by one generated transition-state job."""

    source: TransitionStateSearchSource
    method: str = "LSTQST"
    cell_overrides: CellTemplate | None = None
    param_overrides: ParamTemplate | None = None


@dataclass(frozen=True, slots=True)
class NudgedElasticBandSource:
    """One completed TSS source expanded into a NEB job."""

    label: str
    reactant_structure: StructureSpec
    product_positions_abs: PositionBlock
    intermediate_positions_abs: PositionBlock | None = None
    product_atomic_spins: tuple[float, ...] | None = None
    intermediate_atomic_spins: tuple[float, ...] | None = None
    source_site_ids: tuple[UUID, UUID] = field(default_factory=tuple)
    source_relaxation_run_ids: tuple[UUID, UUID] = field(default_factory=tuple)
    source_relaxation_job_ids: tuple[UUID | None, UUID | None] = field(default_factory=tuple)
    source_transition_state_run_id: UUID | None = None
    source_transition_state_job_id: UUID | None = None
    path_length_ang: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NudgedElasticBandJobConfig:
    """Configuration shared by one generated NEB job."""

    source: NudgedElasticBandSource
    method: str = "NEB"
    cell_overrides: CellTemplate | None = None
    param_overrides: ParamTemplate | None = None


def expand_transition_state_search_jobs(
    *,
    jobs: tuple[TransitionStateSearchJobConfig, ...] | list[TransitionStateSearchJobConfig],
    name_prefix: str,
    global_defaults: InputTemplate | None = None,
    workflow_defaults: InputTemplate | None = None,
    user_overrides: InputTemplate | None = None,
    tags: tuple[str, ...] = (),
) -> list[JobSpec]:
    """Expand configured transition-state sources into CASTEP jobs."""

    if not jobs:
        raise ValueError("At least one transition-state search job must be configured.")

    resolved: list[JobSpec] = []
    for config in jobs:
        label = config.source.label.strip()
        if not label:
            raise ValueError("Transition-state path labels must not be empty.")
        if config.source.product_atomic_spins is not None and (
            len(config.source.product_atomic_spins) != len(config.source.product_positions_frac.positions)
        ):
            raise ValueError("Product atomic spins must match the product position count.")
        patch = JobPatch(
            label_suffix=label,
            structure=config.source.reactant_structure,
            cell=_cell_template(config),
            param=_param_template(config),
        )
        resolved.append(
            resolve_job(
                name=job_name(name_prefix, label),
                base_structure=config.source.reactant_structure,
                workflow_stage=WorkflowStage.ENERGY_BARRIERS,
                global_defaults=global_defaults,
                workflow_defaults=workflow_defaults,
                patch=patch,
                user_overrides=user_overrides,
                tags=("energy_barriers", "transition_state_search", *tags),
                provenance={
                    "workflow": "energy_barriers",
                    "calc_type": TRANSITION_STATE_SEARCH_CALC_TYPE,
                    "label": label,
                    "tssearch_method": config.method,
                    "source_site_ids": [str(value) for value in config.source.source_site_ids],
                    "source_relaxation_run_ids": [str(value) for value in config.source.source_relaxation_run_ids],
                    "source_relaxation_job_ids": [
                        str(value) if value is not None else None for value in config.source.source_relaxation_job_ids
                    ],
                    "path_start_fractional": list(config.source.path_start_fractional),
                    "path_end_fractional": list(config.source.path_end_fractional),
                    "path_length_ang": config.source.path_length_ang,
                    **dict(config.source.provenance),
                },
            )
        )
    return resolved


def expand_nudged_elastic_band_jobs(
    *,
    jobs: tuple[NudgedElasticBandJobConfig, ...] | list[NudgedElasticBandJobConfig],
    name_prefix: str,
    global_defaults: InputTemplate | None = None,
    workflow_defaults: InputTemplate | None = None,
    user_overrides: InputTemplate | None = None,
    tags: tuple[str, ...] = (),
) -> list[JobSpec]:
    """Expand completed transition-state sources into CASTEP NEB jobs."""

    if not jobs:
        raise ValueError("At least one NEB job must be configured.")

    resolved: list[JobSpec] = []
    for config in jobs:
        label = config.source.label.strip()
        if not label:
            raise ValueError("NEB path labels must not be empty.")
        patch = JobPatch(
            label_suffix=label,
            structure=config.source.reactant_structure,
            cell=_neb_cell_template(config),
            param=_neb_param_template(config),
        )
        resolved.append(
            resolve_job(
                name=job_name(name_prefix, label),
                base_structure=config.source.reactant_structure,
                workflow_stage=WorkflowStage.ENERGY_BARRIERS,
                global_defaults=global_defaults,
                workflow_defaults=workflow_defaults,
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
                    "source_transition_state_run_id": (
                        str(config.source.source_transition_state_run_id)
                        if config.source.source_transition_state_run_id is not None
                        else None
                    ),
                    "source_transition_state_job_id": (
                        str(config.source.source_transition_state_job_id)
                        if config.source.source_transition_state_job_id is not None
                        else None
                    ),
                    "path_length_ang": config.source.path_length_ang,
                    **dict(config.source.provenance),
                },
            )
        )
    return resolved


def _cell_template(config: TransitionStateSearchJobConfig) -> CellTemplate:
    rows = _rows_from_positions(config.source.product_positions_frac, atomic_spins=config.source.product_atomic_spins)
    base = CellTemplate(
        extra_scalars=(KeywordAssignment("fix_com", False),),
        extra_blocks=(
            BlockSection(
                name="positions_frac_product",
                rows=tuple(rows),
            ),
        ),
    )
    if config.cell_overrides is None:
        return base
    merged = merge_templates(InputTemplate(cell=base), InputTemplate(cell=config.cell_overrides))
    return merged.cell or base


def _neb_cell_template(config: NudgedElasticBandJobConfig) -> CellTemplate:
    blocks = [
        BlockSection(
            name="positions_abs_product",
            rows=tuple(_rows_from_positions(config.source.product_positions_abs, atomic_spins=config.source.product_atomic_spins)),
            unit=config.source.product_positions_abs.unit,
        )
    ]
    if config.source.intermediate_positions_abs is not None:
        blocks.append(
            BlockSection(
                name="positions_abs_intermediate",
                rows=tuple(
                    _rows_from_positions(
                        config.source.intermediate_positions_abs,
                        atomic_spins=config.source.intermediate_atomic_spins,
                    )
                ),
                unit=config.source.intermediate_positions_abs.unit,
            )
        )
    base = CellTemplate(
        extra_scalars=(KeywordAssignment("fix_com", False),),
        extra_blocks=tuple(blocks),
    )
    if config.cell_overrides is None:
        return base
    merged = merge_templates(InputTemplate(cell=base), InputTemplate(cell=config.cell_overrides))
    return merged.cell or base


def _param_template(config: TransitionStateSearchJobConfig) -> ParamTemplate:
    base = _transition_state_param_template(
        method=config.method,
        atomic_spins=config.source.reactant_structure.atomic_spins,
    )
    if config.param_overrides is None:
        return base
    merged = merge_templates(InputTemplate(param=base), InputTemplate(param=config.param_overrides))
    return merged.param or base


def _neb_param_template(config: NudgedElasticBandJobConfig) -> ParamTemplate:
    base = _nudged_elastic_band_param_template(
        method=config.method,
        atomic_spins=config.source.reactant_structure.atomic_spins,
    )
    if config.param_overrides is None:
        return base
    merged = merge_templates(InputTemplate(param=base), InputTemplate(param=config.param_overrides))
    return merged.param or base


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
            # Odd default so the climbing image sits on the saddle without CASTEP
            # silently bumping the count; user config may override this.
            KeywordAssignment("tssearch_max_path_points", 7),
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