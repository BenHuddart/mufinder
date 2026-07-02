"""Adapter from convergence GUI state into workflow-layer job expansion."""

from __future__ import annotations

from collections.abc import Mapping

from mufinder.app.convergence_types import ConvergenceConfig, KPointGrid, SweepValue
from mufinder.castep_io.gui_metadata import make_block_section, make_keyword_assignment
from mufinder.castep_io.schema import canonical_keyword_name, get_keyword_spec
from mufinder.castep_io.types import (
    AtomicPosition,
    BlockSection,
    KeywordAssignment,
    KPoint,
    LatticeABC,
    LatticeCart,
    PositionBlock,
    SpeciesPotential,
    ValueWithUnit,
)
from mufinder.domain.enums import WorkflowStage
from mufinder.workflow._common import format_value_token
from mufinder.workflow.convergence import ConvergenceWorkflow, expand_convergence_sweep_jobs
from mufinder.workflow.merge import merge_templates, resolve_job
from mufinder.workflow.naming import job_name
from mufinder.workflow.types import (
    CellTemplate,
    InputTemplate,
    JobPatch,
    JobSpec,
    ParamTemplate,
    StructureSpec,
)

_STRUCTURE_CONTROLLED_CELL_KEYWORDS = {
    "lattice_cart",
    "lattice_abc",
    "positions_frac",
    "positions_abs",
}


def _as_kpoint_grid(value: object) -> KPointGrid:
    triplet = tuple(int(component) for component in value)  # type: ignore[arg-type]
    if len(triplet) != 3:
        raise ValueError("kpoint_mp_grid sweep values must be integer triplets.")
    return (triplet[0], triplet[1], triplet[2])


def _sweep_label_suffix(sweep_parameter: str, value: SweepValue) -> str:
    if sweep_parameter == "cut_off_energy":
        return f"ecut_{format_value_token(float(value))}"
    if sweep_parameter == "kpoint_mp_grid":
        grid = _as_kpoint_grid(value)
        return f"kgrid_{grid[0]}x{grid[1]}x{grid[2]}"
    raise ValueError(
        "Convergence sweep_parameter must be 'cut_off_energy' or 'kpoint_mp_grid'."
    )


def convergence_job_name(
    name_prefix: str,
    value: SweepValue,
    *,
    sweep_parameter: str = "cut_off_energy",
) -> str:
    """Return the deterministic job label for one convergence sweep point."""

    return job_name(name_prefix, _sweep_label_suffix(sweep_parameter, value))


def _stage_and_tags_for_sweep(sweep_parameter: str) -> tuple[WorkflowStage, tuple[str, ...]]:
    if sweep_parameter == "cut_off_energy":
        return WorkflowStage.CONVERGENCE_CUTOFF, ("convergence_cutoff",)
    if sweep_parameter == "kpoint_mp_grid":
        return WorkflowStage.CONVERGENCE_KPOINT, ("convergence_kpoint",)
    raise ValueError(
        "Convergence sweep_parameter must be 'cut_off_energy' or 'kpoint_mp_grid'."
    )


def _normalize_sweep_values(config: ConvergenceConfig) -> tuple[SweepValue, ...]:
    if config.sweep_parameter == "cut_off_energy":
        values = tuple(float(value) for value in config.sweep_values)
    elif config.sweep_parameter == "kpoint_mp_grid":
        values = tuple(_as_kpoint_grid(value) for value in config.sweep_values)
    else:
        raise ValueError(
            "Convergence sweep_parameter must be 'cut_off_energy' or 'kpoint_mp_grid'."
        )

    unique = tuple(dict.fromkeys(values))
    if len(unique) != len(values):
        descriptor = "Cutoff energy list" if config.sweep_parameter == "cut_off_energy" else "K-point grid list"
        raise ValueError(f"{descriptor} must not contain duplicates.")
    return unique


def _find_keyword_spec(name: str):
    for file_kind in ("param", "cell"):
        spec = get_keyword_spec(file_kind, name)
        if spec is not None:
            return spec
    return None


def _lattice_cart_from_block(block: BlockSection) -> LatticeCart:
    return LatticeCart(
        vectors=tuple(tuple(float(component) for component in row) for row in block.rows),  # type: ignore[arg-type]
        unit=block.unit,
    )


def _lattice_abc_from_block(block: BlockSection) -> LatticeABC:
    return LatticeABC(
        lengths=tuple(float(component) for component in block.rows[0]),  # type: ignore[arg-type]
        angles=tuple(float(component) for component in block.rows[1]),  # type: ignore[arg-type]
        unit=block.unit,
    )


def _positions_from_block(block: BlockSection) -> PositionBlock:
    return PositionBlock(
        positions=tuple(
            AtomicPosition(
                species=str(row[0]),
                coordinates=tuple(float(component) for component in row[1:4]),  # type: ignore[arg-type]
            )
            for row in block.rows
        ),
        unit=block.unit,
    )


def _species_pot_from_block(block: BlockSection) -> tuple[SpeciesPotential, ...]:
    return tuple(SpeciesPotential(species=str(row[0]), potential=str(row[1])) for row in block.rows)


def _kpoints_list_from_block(block: BlockSection) -> tuple[KPoint, ...]:
    return tuple(
        KPoint(
            coordinates=tuple(float(component) for component in row[0:3]),  # type: ignore[arg-type]
            weight=float(row[3]),
        )
        for row in block.rows
    )


def _template_from_values(
    values: Mapping[str, object],
    *,
    sweep_parameter: str = "cut_off_energy",
) -> InputTemplate:
    lattice_cart = None
    lattice_abc = None
    positions_frac = None
    positions_abs = None
    cell_scalars: list[KeywordAssignment] = []
    cell_blocks: list[BlockSection] = []
    param_keywords: list[KeywordAssignment] = []
    param_blocks: list[BlockSection] = []

    species_pot = None
    kpoint_mp_grid = None
    kpoint_mp_spacing = None
    kpoints_list = None
    symmetry_generate = None

    for raw_name, raw_value in values.items():
        spec = _find_keyword_spec(raw_name)
        if spec is None:
            raise ValueError(f"Unsupported CASTEP keyword '{raw_name}' in convergence settings.")

        canonical_name = canonical_keyword_name(spec.file, spec.name)
        if sweep_parameter == "cut_off_energy" and canonical_name == "cut_off_energy":
            raise ValueError("cut_off_energy is controlled by the sweep list and cannot be overridden here.")
        if sweep_parameter == "kpoint_mp_grid" and canonical_name == "kpoint_mp_grid":
            raise ValueError("kpoint_mp_grid is controlled by the sweep list and cannot be overridden here.")
        if spec.file == "cell" and canonical_name in _STRUCTURE_CONTROLLED_CELL_KEYWORDS:
            raise ValueError(
                f"{canonical_name} is controlled by the selected structure and cannot be overridden here."
            )

        if spec.kind == "block":
            block = make_block_section(spec, raw_value)
            if block is None:
                continue
            if spec.file == "cell":
                if canonical_name == "lattice_cart":
                    lattice_cart = _lattice_cart_from_block(block)
                    lattice_abc = None
                    continue
                if canonical_name == "lattice_abc":
                    lattice_abc = _lattice_abc_from_block(block)
                    lattice_cart = None
                    continue
                if canonical_name == "positions_frac":
                    positions_frac = _positions_from_block(block)
                    positions_abs = None
                    continue
                if canonical_name == "positions_abs":
                    positions_abs = _positions_from_block(block)
                    positions_frac = None
                    continue
                if canonical_name == "species_pot":
                    species_pot = _species_pot_from_block(block)
                    continue
                if canonical_name == "kpoints_list":
                    kpoints_list = _kpoints_list_from_block(block)
                    kpoint_mp_grid = None
                    kpoint_mp_spacing = None
                    continue
                cell_blocks.append(block)
                continue
            param_blocks.append(block)
            continue

        if spec.file == "cell":
            if canonical_name == "symmetry_generate":
                symmetry_generate = bool(raw_value)
                continue
            if canonical_name == "kpoint_mp_grid":
                assignment = make_keyword_assignment(spec, raw_value)
                if assignment is None:
                    continue
                triplet = tuple(int(component) for component in assignment.value)  # type: ignore[arg-type]
                if len(triplet) != 3:
                    raise ValueError("kpoint_mp_grid requires three integer values.")
                kpoint_mp_grid = (triplet[0], triplet[1], triplet[2])
                kpoint_mp_spacing = None
                kpoints_list = ()
                continue
            if canonical_name == "kpoint_mp_spacing":
                assignment = make_keyword_assignment(spec, raw_value)
                if assignment is None:
                    continue
                kpoint_mp_spacing = assignment.value  # type: ignore[assignment]
                kpoint_mp_grid = None
                kpoints_list = ()
                continue

        assignment = make_keyword_assignment(spec, raw_value)
        if assignment is None:
            continue

        if spec.file == "param":
            param_keywords.append(assignment)
        else:
            cell_scalars.append(assignment)

    cell_template = None
    if (
        lattice_cart is not None
        or lattice_abc is not None
        or positions_frac is not None
        or positions_abs is not None
        or species_pot is not None
        or kpoint_mp_grid is not None
        or kpoint_mp_spacing is not None
        or kpoints_list is not None
        or symmetry_generate is not None
        or cell_scalars
        or cell_blocks
    ):
        cell_template = CellTemplate(
            lattice_cart=lattice_cart,
            lattice_abc=lattice_abc,
            positions_frac=positions_frac,
            positions_abs=positions_abs,
            species_pot=species_pot,
            kpoint_mp_grid=kpoint_mp_grid,
            kpoint_mp_spacing=kpoint_mp_spacing,
            kpoints_list=kpoints_list,
            symmetry_generate=symmetry_generate,
            extra_scalars=tuple(cell_scalars),
            extra_blocks=tuple(cell_blocks),
        )

    param_template = (
        ParamTemplate(keywords=tuple(param_keywords), extra_blocks=tuple(param_blocks))
        if param_keywords or param_blocks
        else None
    )
    return InputTemplate(cell=cell_template, param=param_template)


def build_convergence_workflow(
    *,
    structure: StructureSpec,
    config: ConvergenceConfig,
) -> ConvergenceWorkflow:
    """Build a ConvergenceWorkflow for shared-parameter-only convergence sweeps.

    .. deprecated::
        This helper predates the function-based expansion path. The GUI now
        calls :func:`expand_convergence_jobs` (which delegates to
        :func:`~mufinder.workflow.convergence.expand_convergence_sweep_jobs`)
        directly, so ``build_convergence_workflow`` is orphaned on its happy
        path and only remains as a public compatibility helper returning the
        deprecated :class:`~mufinder.workflow.convergence.ConvergenceWorkflow`
        shim. Prefer :func:`expand_convergence_jobs` in new code. See the
        compatibility ledger in ``docs/compatibility_policy.rst``; this helper
        retires together with ``ConvergenceWorkflow``.
    """

    if config.overrides:
        raise ValueError(
            "Per-cutoff overrides (or other per-sweep-point overrides) require expand_convergence_jobs()."
        )

    stage, tags = _stage_and_tags_for_sweep(config.sweep_parameter)
    workflow_defaults = _template_from_values(config.shared_params, sweep_parameter=config.sweep_parameter)
    return ConvergenceWorkflow(
        structure=structure,
        sweep_parameter=config.sweep_parameter,
        sweep_values=_normalize_sweep_values(config),
        name_prefix=config.name_prefix,
        workflow_stage=stage,
        workflow_defaults=workflow_defaults,
        tags=tags,
    )


def expand_convergence_jobs(
    *,
    structure: StructureSpec,
    config: ConvergenceConfig,
) -> list[JobSpec]:
    """Expand convergence config into resolved JobSpec objects."""

    if not config.sweep_values:
        descriptor = "cutoff energy" if config.sweep_parameter == "cut_off_energy" else "k-point grid"
        raise ValueError(f"At least one {descriptor} must be provided.")

    unique_values = _normalize_sweep_values(config)
    stage, tags = _stage_and_tags_for_sweep(config.sweep_parameter)

    if not config.overrides:
        workflow_defaults = _template_from_values(
            config.shared_params, sweep_parameter=config.sweep_parameter
        )
        return expand_convergence_sweep_jobs(
            structure=structure,
            sweep_parameter=config.sweep_parameter,
            sweep_values=unique_values,
            name_prefix=config.name_prefix,
            workflow_stage=stage,
            workflow_defaults=workflow_defaults,
            tags=tags,
        )

    workflow_defaults = _template_from_values(config.shared_params, sweep_parameter=config.sweep_parameter)

    per_value_user_overrides: dict[SweepValue, InputTemplate] = {}
    for value_set, override_values in config.overrides.items():
        override_template = _template_from_values(
            override_values,
            sweep_parameter=config.sweep_parameter,
        )
        for raw_value in value_set:
            value: SweepValue
            if config.sweep_parameter == "cut_off_energy":
                value = float(raw_value)
            else:
                value = _as_kpoint_grid(raw_value)

            if value not in unique_values:
                raise ValueError(
                    f"Override references sweep value {value}, but it is not present in the sweep list."
                )
            existing = per_value_user_overrides.get(value)
            if existing is None:
                per_value_user_overrides[value] = override_template
            else:
                per_value_user_overrides[value] = merge_templates(
                    existing,
                    override_template,
                )

    jobs: list[JobSpec] = []
    for value in unique_values:
        if config.sweep_parameter == "cut_off_energy":
            cutoff = float(value)
            patch = JobPatch(
                label_suffix=_sweep_label_suffix(config.sweep_parameter, cutoff),
                param=ParamTemplate(
                    keywords=(KeywordAssignment("cut_off_energy", ValueWithUnit(cutoff, "eV")),)
                ),
            )
        else:
            grid = _as_kpoint_grid(value)
            patch = JobPatch(
                label_suffix=_sweep_label_suffix(config.sweep_parameter, grid),
                cell=CellTemplate(kpoint_mp_grid=grid),
            )

        jobs.append(
            resolve_job(
                name=convergence_job_name(
                    config.name_prefix,
                    value,
                    sweep_parameter=config.sweep_parameter,
                ),
                base_structure=structure,
                workflow_stage=stage,
                workflow_defaults=workflow_defaults,
                patch=patch,
                user_overrides=per_value_user_overrides.get(value),
                tags=tags,
                provenance={
                    "workflow": "convergence",
                    "sweep_parameter": config.sweep_parameter,
                    "sweep_value": value,
                },
            )
        )

    return jobs
