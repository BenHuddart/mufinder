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
class NudgedElasticBandSource:
    """One diffusion-path endpoint pair expanded into a NEB job."""

    label: str
    reactant_structure: StructureSpec
    product_positions_abs: PositionBlock
    intermediate_positions_abs: PositionBlock | None = None
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
                    "path_length_ang": config.source.path_length_ang,
                    **dict(config.source.provenance),
                },
            )
        )
    return resolved


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