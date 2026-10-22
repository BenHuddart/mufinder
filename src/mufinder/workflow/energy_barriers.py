"""Nudged-elastic-band (NEB) workflow expansion for Energy Barriers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from mufinder.castep_io.types import AtomicPosition, BlockSection, KeywordAssignment, PositionBlock, ValueWithUnit
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.magnetic_properties import has_nonzero_spins

from ._common import require_jobs
from .merge import compose_patch, resolve_job
from .naming import job_name
from .types import CellTemplate, InputTemplate, JobPatch, JobSpec, ParamTemplate, StructureSpec


# Read-only calc types of the retired LST/QST transition-state search. Nothing
# creates them; they remain so the jobs, results and path definitions an older
# project stored still load. See the ledger in docs/compatibility_policy.rst.
TRANSITION_STATE_SEARCH_CALC_TYPE ="castep_transition_state_search"
TRANSITION_STATE_SEARCH_SET_CALC_TYPE = "castep_transition_state_search_set"
NUDGED_ELASTIC_BAND_CALC_TYPE = "castep_nudged_elastic_band"
NUDGED_ELASTIC_BAND_SET_CALC_TYPE = "castep_nudged_elastic_band_set"


def merge_serialized_sources(
    existing_sources: object,
    new_sources: list[dict[str, object]],
    *,
    key_field: str,
) -> list[dict[str, object]]:
    """Merge serialized path sources by ``key_field``, keeping first-seen order.

    ``existing_sources`` may be any stored payload; only a list of dicts is read
    and entries without a key are dropped. A source in ``new_sources`` replaces an
    existing source with the same key in place, and new keys are appended.
    """

    merged_by_key: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for source in tuple(existing_sources) if isinstance(existing_sources, list) else ():
        if not isinstance(source, dict):
            continue
        key = str(source.get(key_field, "")).strip()
        if not key:
            continue
        if key not in merged_by_key:
            order.append(key)
        merged_by_key[key] = dict(source)
    for source in new_sources:
        key = str(source.get(key_field, "")).strip()
        if not key:
            continue
        if key not in merged_by_key:
            order.append(key)
        merged_by_key[key] = dict(source)
    return [merged_by_key[key] for key in order]


def replace_serialized_sources_for_pair(
    existing_sources: object,
    *,
    reactant_site_id: str,
    product_site_id: str,
    new_sources: list[dict[str, object]],
    charge_state: str | None = None,
) -> list[dict[str, object]]:
    """Replace every stored path source of one reactant/product pair.

    With ``charge_state`` only the pair's paths in that charge state are
    replaced (a stored path without one counts as matching), so a pair relaxed
    in both charge states keeps the other charge state's paths. Sources of
    other endpoint pairs are kept in order; ``new_sources`` are then merged in
    by ``key``.
    """

    filtered: list[dict[str, object]] = []
    for source in tuple(existing_sources) if isinstance(existing_sources, list) else ():
        if not isinstance(source, dict):
            continue
        stored_charge_state = source.get("charge_state")
        if (
            str(source.get("reactant_site_id", "")).strip() == reactant_site_id
            and str(source.get("product_site_id", "")).strip() == product_site_id
            and (charge_state is None or stored_charge_state in (None, "", charge_state))
        ):
            continue
        filtered.append(dict(source))
    return merge_serialized_sources(filtered, new_sources, key_field="key")


_LEGACY_ENERGY_BARRIER_OBSOLETE_PARAMS: frozenset[str] = frozenset({"tssearch_lst_max_iter"})
_LEGACY_ENERGY_BARRIER_ZERO_DEFAULT_PARAMS: frozenset[str] = frozenset({
    "fine_cut_off_energy",
    "tssearch_force_tol",
    "tssearch_disp_tol",
    "tssearch_energy_tol",
})


def strip_legacy_energy_barrier_params(params: dict[str, object]) -> dict[str, object]:
    """Drop stored Energy Barriers params that CASTEP must not receive.

    ``tssearch_lst_max_iter`` is obsolete. ``tssearch_force_tol``,
    ``tssearch_disp_tol`` and ``tssearch_energy_tol`` default to their ``GEOM_*``
    counterparts in CASTEP, and ``fine_cut_off_energy`` has its own CASTEP
    default, so a stored value of zero or less for any of them is a placeholder
    and is omitted. Every other entry is returned unchanged.
    """

    result: dict[str, object] = {}
    for name, value in params.items():
        if name in _LEGACY_ENERGY_BARRIER_OBSOLETE_PARAMS:
            continue
        if name in _LEGACY_ENERGY_BARRIER_ZERO_DEFAULT_PARAMS:
            numeric = value.value if isinstance(value, ValueWithUnit) else value
            try:
                if float(numeric) <= 0:
                    continue
            except (TypeError, ValueError):
                pass
        result[name] = value
    return result


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
    # CASTEP task farms for the band. Each farm runs the SCF for a contiguous block of
    # images, so the sweep cost falls as ceil(images / farms). 1 disables farming.
    num_farms: int = 1
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

    require_jobs(jobs, "NEB job")

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
                    "num_farms": max(1, int(config.num_farms)),
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
        num_farms=config.num_farms,
    )


def _nudged_elastic_band_param_template(
    *,
    method: str,
    atomic_spins: tuple[float, ...] | None,
    num_farms: int = 1,
) -> ParamTemplate:
    # The climbing-image flag makes the central bead land on the saddle (the barrier).
    # The image count (`tssearch_max_path_points`) and tolerances come from the
    # workflow/UI configuration; CASTEP requires an odd image count when climbing.
    keywords = [
        KeywordAssignment("task", "TRANSITIONSTATESEARCH"),
        KeywordAssignment("tssearch_method", method),
    ]
    if has_nonzero_spins(atomic_spins):
        keywords.append(KeywordAssignment("spin_polarized", True))
    keywords += [
        KeywordAssignment("tssearch_neb_climbing", True),
        # Odd default so the climbing image sits on the saddle without CASTEP adding
        # one image to an even count; 11 gives a smoother, better-resolved band than the minimum.
        # User/UI config may override this.
        KeywordAssignment("tssearch_max_path_points", 11),
    ]
    # Only NEB is farm-aware: CASTEP's parameters_validate resets NUM_FARMS to 1 (with a
    # warning) for any other TSSEARCH_METHOD, so emitting it there would be noise.
    if int(num_farms) > 1:
        keywords.append(KeywordAssignment("num_farms", max(1, int(num_farms))))
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