"""Adapter from UEP UI state into CASTEP single-point source-job expansion."""

from __future__ import annotations

from dataclasses import dataclass

from mufinder.app.relaxation_adapter import (
    _is_identity_supercell,
    _structure_for_row,
    _template_from_values,
)
from mufinder.app.uep_types import UepConfig
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Structure
from mufinder.science.kpoint_grids import equivalent_grid_for_supercell
from mufinder.workflow import JobPatch, StructureBridgeError, job_name, resolve_job, structure_to_spec
from mufinder.workflow.types import InputTemplate, JobSpec


@dataclass(frozen=True, slots=True)
class UepExpansion:
    """Expanded UEP source jobs plus any k-point-density adjustment notes."""

    jobs: tuple[JobSpec, ...]
    kpoint_messages: tuple[str, ...] = ()


def expand_uep_jobs(
    *,
    host_structure: Structure,
    config: UepConfig,
    project_cutoff_energy: float | None,
    project_kpoint_grid: tuple[int, int, int] | None,
    project_kpoint_reference_lattice: list[list[float]] | None,
) -> UepExpansion:
    """Expand one UEP source-job configuration into resolved CASTEP job specs."""

    if not config.rows:
        raise ValueError("At least one UEP source row must be configured.")

    workflow_defaults_values: dict[str, object] = {
        "task": "SINGLEPOINT",
        "symmetry_generate": True,
        "write_formatted_density": True,
    }
    if project_cutoff_energy is not None:
        workflow_defaults_values["cut_off_energy"] = float(project_cutoff_energy)
    workflow_defaults = _template_from_values(workflow_defaults_values)

    reference_lattice = (
        project_kpoint_reference_lattice
        if isinstance(project_kpoint_reference_lattice, list) and len(project_kpoint_reference_lattice) == 3
        else host_structure.lattice_vectors
    )

    seen_labels: set[str] = set()
    messages: list[str] = []
    jobs: list[JobSpec] = []
    for row in config.rows:
        label = row.label.strip()
        if not label:
            raise ValueError("UEP source row labels must not be empty.")
        if label in seen_labels:
            raise ValueError(f"Duplicate UEP source row label '{label}'.")
        seen_labels.add(label)

        row_structure = _structure_for_row(host_structure, row)  # type: ignore[arg-type]
        try:
            structure_spec = structure_to_spec(row_structure)
        except StructureBridgeError as exc:
            raise ValueError(f"Could not convert row '{label}' structure: {exc}") from exc

        row_values = dict(row.parameter_values)
        row_values["task"] = "SINGLEPOINT"
        row_values["write_formatted_density"] = True
        row_template = _template_from_values(row_values)

        explicit_cell = row_template.cell
        explicit_kpoint_settings = (
            explicit_cell is not None
            and (
                explicit_cell.kpoint_mp_grid is not None
                or explicit_cell.kpoint_mp_spacing is not None
                or explicit_cell.kpoints_list is not None
            )
        )

        cell = row_structure.lattice_vectors
        kpoint_grid = None
        if project_kpoint_grid is not None and not explicit_kpoint_settings:
            kpoint_grid = (
                project_kpoint_grid
                if _is_identity_supercell(row.supercell_matrix)
                else equivalent_grid_for_supercell(reference_lattice, project_kpoint_grid, cell)
            )
            if row.supercell_matrix is not None and kpoint_grid != project_kpoint_grid:
                messages.append(
                    "K-point grid adjusted from "
                    f"{project_kpoint_grid[0]}x{project_kpoint_grid[1]}x{project_kpoint_grid[2]} to "
                    f"{kpoint_grid[0]}x{kpoint_grid[1]}x{kpoint_grid[2]} for row '{label}' "
                    "(equivalent reciprocal-space density)."
                )

        patch_cell = row_template.cell
        if kpoint_grid is not None:
            from mufinder.workflow import merge_templates
            from mufinder.workflow.types import CellTemplate

            patch_cell = merge_templates(
                InputTemplate(cell=CellTemplate(kpoint_mp_grid=kpoint_grid)),
                InputTemplate(cell=row_template.cell),
            ).cell

        jobs.append(
            resolve_job(
                name=job_name(config.name_prefix, label),
                base_structure=structure_spec,
                workflow_stage=WorkflowStage.UEP,
                workflow_defaults=workflow_defaults,
                patch=JobPatch(
                    label_suffix=label,
                    structure=structure_spec,
                    cell=patch_cell,
                    param=row_template.param,
                ),
                tags=("uep", "uep_source"),
                provenance={
                    "workflow": "uep",
                    "label": label,
                    "supercell_matrix": [list(line) for line in row.supercell_matrix]
                    if row.supercell_matrix is not None
                    else None,
                    "kpoint_mp_grid": kpoint_grid,
                },
            )
        )

    return UepExpansion(jobs=tuple(jobs), kpoint_messages=tuple(messages))
