"""Convergence workflow expansion into resolved JobSpec objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mufinder.castep_io.types import KeywordAssignment, ValueWithUnit
from mufinder.domain.enums import WorkflowStage

from .merge import resolve_job
from .naming import job_name
from .types import CellTemplate, InputTemplate, JobPatch, JobSpec, ParamTemplate, StructureSpec


def _format_value_token(value: float) -> str:
    token = f"{value:.12g}"
    token = token.replace(".", "p")
    token = token.replace("-", "m")
    return token


def _patch_for_value(
    sweep_parameter: str, value: float | tuple[int, int, int]
) -> JobPatch:
    if sweep_parameter == "cut_off_energy":
        numeric_value = float(value)  # type: ignore[arg-type]
        suffix = f"ecut_{_format_value_token(numeric_value)}"
        return JobPatch(
            label_suffix=suffix,
            param=ParamTemplate(
                keywords=(
                    KeywordAssignment("cut_off_energy", ValueWithUnit(numeric_value, "eV")),
                )
            ),
        )

    triplet = tuple(int(component) for component in value)  # type: ignore[arg-type]
    if len(triplet) != 3:
        raise ValueError("kpoint_mp_grid sweep values must be integer triplets.")
    suffix = f"kgrid_{triplet[0]}x{triplet[1]}x{triplet[2]}"
    return JobPatch(
        label_suffix=suffix,
        cell=CellTemplate(kpoint_mp_grid=(triplet[0], triplet[1], triplet[2])),
    )


def expand_convergence_sweep_jobs(
    *,
    structure: StructureSpec,
    sweep_parameter: str,
    sweep_values: Sequence[float | tuple[int, int, int]],
    name_prefix: str = "convergence",
    global_defaults: InputTemplate | None = None,
    workflow_defaults: InputTemplate | None = None,
    structure_defaults: InputTemplate | None = None,
    user_overrides: InputTemplate | None = None,
    workflow_stage: WorkflowStage | None = None,
    tags: tuple[str, ...] = (),
) -> list[JobSpec]:
    """Expand a convergence sweep into resolved JobSpec objects.

    Iterates ``sweep_values`` along a single ``sweep_parameter`` axis
    (``"cut_off_energy"`` or ``"kpoint_mp_grid"``), layering the supplied
    ``InputTemplate`` arguments in
    :data:`~mufinder.workflow.merge.MERGE_PRECEDENCE_ORDER`.
    """

    if not sweep_values:
        raise ValueError("Convergence workflow requires at least one sweep value.")

    if sweep_parameter not in {"cut_off_energy", "kpoint_mp_grid"}:
        raise ValueError(
            "Convergence sweep_parameter must be 'cut_off_energy' or 'kpoint_mp_grid'."
        )

    stage = workflow_stage
    if stage is None:
        stage = (
            WorkflowStage.CONVERGENCE_CUTOFF
            if sweep_parameter == "cut_off_energy"
            else WorkflowStage.CONVERGENCE_KPOINT
        )

    jobs: list[JobSpec] = []
    for value in sweep_values:
        patch = _patch_for_value(sweep_parameter, value)
        name = job_name(name_prefix, patch.label_suffix)
        jobs.append(
            resolve_job(
                name=name,
                base_structure=structure,
                workflow_stage=stage,
                global_defaults=global_defaults,
                workflow_defaults=workflow_defaults,
                structure_defaults=structure_defaults,
                patch=patch,
                user_overrides=user_overrides,
                tags=tags,
                provenance={
                    "workflow": "convergence",
                    "sweep_parameter": sweep_parameter,
                    "sweep_value": value,
                },
            )
        )

    return jobs


@dataclass(slots=True)
class ConvergenceWorkflow:
    """Deprecated class wrapper around :func:`expand_convergence_sweep_jobs`.

    Retained as a public compatibility alias for older imports; see the
    compatibility ledger in ``docs/compatibility_policy.rst``. New code should
    call :func:`expand_convergence_sweep_jobs` directly.
    """

    structure: StructureSpec
    sweep_parameter: str
    sweep_values: Sequence[float | tuple[int, int, int]]
    name_prefix: str = "convergence"
    workflow_stage: WorkflowStage | None = None
    global_defaults: InputTemplate | None = None
    workflow_defaults: InputTemplate | None = None
    structure_defaults: InputTemplate | None = None
    user_overrides: InputTemplate | None = None
    tags: tuple[str, ...] = ()

    def expand(self) -> list[JobSpec]:
        """Expand sweep values into resolved JobSpec objects."""

        return expand_convergence_sweep_jobs(
            structure=self.structure,
            sweep_parameter=self.sweep_parameter,
            sweep_values=self.sweep_values,
            name_prefix=self.name_prefix,
            global_defaults=self.global_defaults,
            workflow_defaults=self.workflow_defaults,
            structure_defaults=self.structure_defaults,
            user_overrides=self.user_overrides,
            workflow_stage=self.workflow_stage,
            tags=self.tags,
        )
