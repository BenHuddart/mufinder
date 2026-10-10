"""Adapter from UEP UI state into CASTEP single-point source-job expansion."""

from __future__ import annotations

from mufinder.app.adapter_common import (
    AdapterExpansion,
    claim_unique_name,
    kpoint_reference_lattice,
    prepare_row,
    require_row_label,
    seed_workflow_defaults,
)
from mufinder.app.relaxation_adapter import (
    _structure_for_row,
    _template_from_values,
)
from mufinder.app.uep_types import UepConfig
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Structure
from mufinder.workflow import JobPatch, job_name, resolve_job
from mufinder.workflow.types import InputTemplate, JobSpec

# ``UepExpansion`` is a thin backward-compatible alias for the generic
# :class:`~mufinder.app.adapter_common.AdapterExpansion`. The former UEP-specific
# ``kpoint_messages`` field is exposed as a property on ``AdapterExpansion``.
UepExpansion = AdapterExpansion


def expand_uep_jobs(
    *,
    host_structure: Structure,
    config: UepConfig,
    project_cutoff_energy: float | None,
    project_kpoint_grid: tuple[int, int, int] | None,
    project_kpoint_reference_lattice: list[list[float]] | None,
    project_xc_functional: str | None = None,
) -> AdapterExpansion:
    """Expand one UEP source-job configuration into resolved CASTEP job specs."""

    if not config.rows:
        raise ValueError("At least one UEP source row must be configured.")

    workflow_defaults = _template_from_values(
        seed_workflow_defaults(
            {
                "task": "SINGLEPOINT",
                "symmetry_generate": True,
                "write_formatted_density": True,
            },
            project_xc_functional=project_xc_functional,
            project_cutoff_energy=project_cutoff_energy,
        )
    )
    reference_lattice = kpoint_reference_lattice(project_kpoint_reference_lattice, host_structure)

    seen_labels: set[str] = set()
    messages: list[str] = []
    jobs: list[JobSpec] = []
    for row in config.rows:
        label = require_row_label(row.label, workflow="UEP source")
        claim_unique_name(label, seen_labels, workflow="UEP source")

        prepared = prepare_row(
            label=label,
            structure=_structure_for_row(host_structure, row),  # type: ignore[arg-type]
            parameter_values=row.parameter_values,
            template_from_values=_template_from_values,
            supercell_matrix=row.supercell_matrix,
            project_kpoint_grid=project_kpoint_grid,
            reference_lattice=reference_lattice,
            target_description=f"row '{label}' (equivalent reciprocal-space density).",
            force={"task": "SINGLEPOINT", "write_formatted_density": True},
        )
        if prepared.message is not None:
            messages.append(prepared.message)
        kpoint_grid = prepared.kpoint_grid
        row_template = prepared.template
        structure_spec = prepared.structure

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

    return AdapterExpansion(jobs=tuple(jobs), messages=tuple(messages))
