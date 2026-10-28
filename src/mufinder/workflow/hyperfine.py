"""Hyperfine Coupling CASTEP workflow expansion.

One ``TASK : MAGRES`` / ``MAGRES_TASK : HYPERFINE`` job per relaxed muon site.
The job is spin polarised (CASTEP stops a hyperfine run that is not) and writes
its formatted density, whose spin column gives the pseudo spin density at the
muon alongside CASTEP's reconstructed tensor. The sources are the relaxed sites
Quantum Corrections also starts from, so this module reuses its source record
and eligibility check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from mufinder.castep_io.types import KeywordAssignment
from mufinder.domain.charge_states import is_muonium
from mufinder.domain.constants import MUON_GAMMA_RAD_S_T
from mufinder.domain.enums import WorkflowStage

from ._common import require_jobs
from .merge import compose_patch, resolve_job
from .naming import job_name
from .quantum import QuantumPhononSource, muon_species_blocks, seed_muonium_spin
from .types import CellTemplate, InputTemplate, JobPatch, JobSpec, ParamTemplate

HYPERFINE_CALC_TYPE = "castep_hyperfine"
HYPERFINE_SET_CALC_TYPE = "castep_hyperfine_set"

#: Parameters the workflow sets on every job; a row's settings may not override them.
HYPERFINE_CONTROLLED_PARAM_KEYWORDS = (
    "task",
    "magres_task",
    "spin_polarized",
    "spin_polarised",
    "write_formatted_density",
)


@dataclass(frozen=True, slots=True)
class HyperfineJobConfig:
    """Configuration of one generated Hyperfine Coupling job."""

    source: QuantumPhononSource
    muon_species: str = "H:mu"
    cell_overrides: CellTemplate | None = None
    param_overrides: ParamTemplate | None = None


def expand_hyperfine_jobs(
    *,
    jobs: Sequence[HyperfineJobConfig],
    name_prefix: str,
    global_defaults: InputTemplate | None = None,
    workflow_defaults: InputTemplate | None = None,
    structure_defaults: InputTemplate | None = None,
    user_overrides: InputTemplate | None = None,
    tags: tuple[str, ...] = (),
) -> list[JobSpec]:
    """Expand Hyperfine Coupling configurations into one CASTEP magres job each.

    Each job carries the muon ``SPECIES_MASS`` and ``SPECIES_GAMMA`` blocks, so
    CASTEP scales the muon's tensor by the muon gyromagnetic ratio. A muonium
    source has +1 seeded on the muon so the run stays open-shell. Settings that
    would override a workflow-controlled parameter are refused.
    """

    require_jobs(jobs, "hyperfine job")
    _refuse_controlled_overrides(user_overrides)

    resolved: list[JobSpec] = []
    for config in jobs:
        label = config.source.label.strip()
        if not label:
            raise ValueError("Hyperfine source labels must not be empty.")
        overrides = InputTemplate(cell=config.cell_overrides, param=config.param_overrides)
        _refuse_controlled_overrides(overrides)
        structure = config.source.structure
        muonium = is_muonium(config.source.charge_state)
        if muonium:
            structure = seed_muonium_spin(structure, config.muon_species)
        keywords = [
            KeywordAssignment("task", "MAGRES"),
            KeywordAssignment("magres_task", "HYPERFINE"),
            KeywordAssignment("spin_polarized", True),
            KeywordAssignment("write_formatted_density", True),
        ]
        if config.source.charge_state:
            keywords.append(KeywordAssignment("charge", 0 if muonium else 1))
        base = InputTemplate(
            cell=CellTemplate(extra_blocks=muon_species_blocks(config.muon_species)),
            param=ParamTemplate(keywords=tuple(keywords)),
        )
        patch_template = compose_patch(base, overrides)
        patch = JobPatch(
            label_suffix=label,
            structure=structure,
            cell=patch_template.cell,
            param=patch_template.param,
        )
        provenance: dict[str, Any] = {
            "workflow": "hyperfine_coupling",
            "calc_type": HYPERFINE_CALC_TYPE,
            "label": label,
            "muon_species": config.muon_species,
            "muon_gamma_rad_s_T": MUON_GAMMA_RAD_S_T,
            "source_site_id": str(config.source.source_site_id),
            "source_relaxation_run_id": str(config.source.source_relaxation_run_id),
            "source_relaxation_job_id": (
                str(config.source.source_relaxation_job_id)
                if config.source.source_relaxation_job_id is not None
                else None
            ),
            "charge_state": config.source.charge_state,
        }
        provenance.update(config.source.provenance)
        resolved.append(
            resolve_job(
                name=job_name(name_prefix, label),
                base_structure=structure,
                workflow_stage=WorkflowStage.HYPERFINE_COUPLING,
                global_defaults=global_defaults,
                workflow_defaults=workflow_defaults,
                structure_defaults=structure_defaults,
                patch=patch,
                user_overrides=user_overrides,
                tags=("hyperfine_coupling", "magres", *tags),
                provenance=provenance,
            )
        )
    return resolved


def _refuse_controlled_overrides(template: InputTemplate | None) -> None:
    if template is None or template.param is None:
        return
    for assignment in template.param.keywords:
        if assignment.name.strip().lower() in HYPERFINE_CONTROLLED_PARAM_KEYWORDS:
            raise ValueError(
                f"{assignment.name.upper()} is set by the Hyperfine Coupling workflow and "
                "cannot be overridden."
            )
