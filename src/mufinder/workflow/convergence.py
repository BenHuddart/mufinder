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


@dataclass(slots=True)
class ConvergenceWorkflow:
    """Expand convergence sweeps into resolved jobs."""

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

        if not self.sweep_values:
            raise ValueError("Convergence workflow requires at least one sweep value.")

        if self.sweep_parameter not in {"cut_off_energy", "kpoint_mp_grid"}:
            raise ValueError(
                "Convergence sweep_parameter must be 'cut_off_energy' or 'kpoint_mp_grid'."
            )

        stage = self.workflow_stage
        if stage is None:
            stage = (
                WorkflowStage.CONVERGENCE_CUTOFF
                if self.sweep_parameter == "cut_off_energy"
                else WorkflowStage.CONVERGENCE_KPOINT
            )

        jobs: list[JobSpec] = []
        for value in self.sweep_values:
            patch = self._patch_for_value(value)
            name = job_name(self.name_prefix, patch.label_suffix)
            jobs.append(
                resolve_job(
                    name=name,
                    base_structure=self.structure,
                    workflow_stage=stage,
                    global_defaults=self.global_defaults,
                    workflow_defaults=self.workflow_defaults,
                    structure_defaults=self.structure_defaults,
                    patch=patch,
                    user_overrides=self.user_overrides,
                    tags=self.tags,
                    provenance={
                        "workflow": "convergence",
                        "sweep_parameter": self.sweep_parameter,
                        "sweep_value": value,
                    },
                )
            )

        return jobs

    def _patch_for_value(self, value: float | tuple[int, int, int]) -> JobPatch:
        if self.sweep_parameter == "cut_off_energy":
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
