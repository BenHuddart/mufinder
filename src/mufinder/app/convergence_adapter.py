"""Adapter from convergence GUI state into workflow-layer job expansion."""

from __future__ import annotations

from collections.abc import Mapping

from mufinder.app.adapter_common import (
    STRUCTURE_CONTROLLED_CELL_KEYWORDS,
    AdapterExpansion,
    build_input_template,
)
from mufinder.app.convergence_types import ConvergenceConfig, KPointGrid, SweepValue
from mufinder.castep_io.types import (
    KeywordAssignment,
    ValueWithUnit,
)
from mufinder.domain.enums import WorkflowStage
from mufinder.workflow._common import format_value_token
from mufinder.workflow.convergence import expand_convergence_sweep_jobs
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
    raise ValueError("Convergence sweep_parameter must be 'cut_off_energy' or 'kpoint_mp_grid'.")


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
    raise ValueError("Convergence sweep_parameter must be 'cut_off_energy' or 'kpoint_mp_grid'.")


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
        descriptor = (
            "Cutoff energy list"
            if config.sweep_parameter == "cut_off_energy"
            else "K-point grid list"
        )
        raise ValueError(f"{descriptor} must not contain duplicates.")
    return unique


def _template_from_values(
    values: Mapping[str, object],
    *,
    sweep_parameter: str = "cut_off_energy",
) -> InputTemplate:
    forbidden: dict[str, str] = {
        name: f"{name} is controlled by the selected structure and cannot be overridden here."
        for name in STRUCTURE_CONTROLLED_CELL_KEYWORDS
    }
    if sweep_parameter == "cut_off_energy":
        forbidden["cut_off_energy"] = (
            "cut_off_energy is controlled by the sweep list and cannot be overridden here."
        )
    elif sweep_parameter == "kpoint_mp_grid":
        forbidden["kpoint_mp_grid"] = (
            "kpoint_mp_grid is controlled by the sweep list and cannot be overridden here."
        )

    return build_input_template(
        values,
        context_label="convergence settings",
        forbidden=forbidden,
        kpoints_list_reset=(),
    )


def expand_convergence_jobs(
    *,
    structure: StructureSpec,
    config: ConvergenceConfig,
) -> AdapterExpansion:
    """Expand convergence config into resolved JobSpec objects.

    Convergence sweeps a single fixed structure, so no supercell k-point
    rescaling occurs and ``AdapterExpansion.kpoint_adjustments`` is always empty. The
    result shape matches the other adapters for a uniform call surface.
    """

    if not config.sweep_values:
        descriptor = (
            "cutoff energy" if config.sweep_parameter == "cut_off_energy" else "k-point grid"
        )
        raise ValueError(f"At least one {descriptor} must be provided.")

    unique_values = _normalize_sweep_values(config)
    stage, tags = _stage_and_tags_for_sweep(config.sweep_parameter)

    if not config.overrides:
        workflow_defaults = _template_from_values(
            config.shared_params, sweep_parameter=config.sweep_parameter
        )
        return AdapterExpansion(
            jobs=tuple(
                expand_convergence_sweep_jobs(
                    structure=structure,
                    sweep_parameter=config.sweep_parameter,
                    sweep_values=unique_values,
                    name_prefix=config.name_prefix,
                    workflow_stage=stage,
                    workflow_defaults=workflow_defaults,
                    tags=tags,
                )
            )
        )

    workflow_defaults = _template_from_values(
        config.shared_params, sweep_parameter=config.sweep_parameter
    )

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

    return AdapterExpansion(jobs=tuple(jobs))
