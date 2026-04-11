"""Pristine-relaxation workflow expansion into resolved JobSpec objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from mufinder.castep_io.types import KeywordAssignment
from mufinder.domain.enums import WorkflowStage

from .merge import merge_templates, resolve_job
from .naming import job_name
from .types import CellTemplate, InputTemplate, JobPatch, JobSpec, ParamTemplate, StructureSpec


@dataclass(frozen=True, slots=True)
class RelaxationJobConfig:
    """Configuration for one pristine-relaxation job."""

    label: str
    structure: StructureSpec
    fix_all_cell: bool = False
    kpoint_grid: tuple[int, int, int] | None = None
    cell_overrides: CellTemplate | None = None
    param_overrides: ParamTemplate | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


def _task_param_template() -> ParamTemplate:
    return ParamTemplate(keywords=(KeywordAssignment("task", "GEOMETRYOPTIMIZATION"),))


def _cell_override_template(config: RelaxationJobConfig) -> CellTemplate:
    scalars = [KeywordAssignment("fix_all_cell", bool(config.fix_all_cell))]
    base = CellTemplate(
        kpoint_mp_grid=(tuple(int(value) for value in config.kpoint_grid) if config.kpoint_grid is not None else None),
        extra_scalars=tuple(scalars),
    )
    if config.cell_overrides is None:
        return base

    merged = merge_templates(
        InputTemplate(cell=base),
        InputTemplate(cell=config.cell_overrides),
    )
    return merged.cell or base


def _param_override_template(config: RelaxationJobConfig) -> ParamTemplate:
    merged = merge_templates(
        InputTemplate(param=_task_param_template()),
        InputTemplate(param=config.param_overrides),
    )
    return merged.param or _task_param_template()


def expand_relaxation_jobs(
    *,
    jobs: Sequence[RelaxationJobConfig],
    name_prefix: str,
    global_defaults: InputTemplate | None = None,
    workflow_defaults: InputTemplate | None = None,
    user_overrides: InputTemplate | None = None,
    tags: tuple[str, ...] = (),
) -> list[JobSpec]:
    """Expand pristine-relaxation job configurations into resolved jobs."""

    if not jobs:
        raise ValueError("At least one pristine-relaxation job must be configured.")

    resolved: list[JobSpec] = []
    for config in jobs:
        suffix = config.label.strip()
        if not suffix:
            raise ValueError("Pristine-relaxation job labels must not be empty.")

        patch = JobPatch(
            label_suffix=suffix,
            structure=config.structure,
            cell=_cell_override_template(config),
            param=_param_override_template(config),
        )

        resolved.append(
            resolve_job(
                name=job_name(name_prefix, suffix),
                base_structure=config.structure,
                workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
                global_defaults=global_defaults,
                workflow_defaults=workflow_defaults,
                patch=patch,
                user_overrides=user_overrides,
                tags=tags,
                provenance={
                    "workflow": "pristine_relaxation",
                    "label": suffix,
                    "fix_all_cell": bool(config.fix_all_cell),
                    "kpoint_mp_grid": config.kpoint_grid,
                    **dict(config.provenance),
                },
            )
        )

    return resolved
