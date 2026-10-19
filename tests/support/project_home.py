"""A small muon-site project shared by the project-home summary, survey and widget tests.

``muon_project`` builds a perovskite host with two candidate sites and a
muon-relaxation definition: ``mu1`` relaxed and converged, ``mu2`` running.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mufinder.domain.charge_states import (
    POSITIVE_MUON_CHARGE_STATE,
    muon_charge_state_prefix,
    muon_relaxation_calc_type,
)
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, MuonSite, Result, Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.workflow.muon_site import MUON_RELAXATION_DEFINITION_CALC_TYPE
from tests.support.structures import host_structure

PEROVSKITE_POSITIONS = (
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
)


@dataclass
class MuonProject:
    document: ProjectDocument
    host: Structure
    relaxed_site: MuonSite
    running_site: MuonSite
    relaxed_job: Job
    running_job: Job
    definition: CalculationDefinition


def perovskite_host() -> Structure:
    """SrTiO3 in a 4 Å cube."""

    return host_structure(
        label="SrTiO3",
        lattice=4.0,
        species=("Sr", "Ti", "O", "O", "O"),
        positions=PEROVSKITE_POSITIONS,
    )


def muon_project(tmp_path: Path) -> MuonProject:
    service = ProjectService()
    document = service.create_project(tmp_path / "home.mufinder", "Home")
    host = perovskite_host()
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    relaxed_site = MuonSite(
        label="mu1",
        status="relaxed",
        initial_fractional_position=[0.5, 0.2, 0.2],
        generation_method="near_element",
        source_structure_id=host.id,
    )
    running_site = MuonSite(
        label="mu2",
        status="staged",
        initial_fractional_position=[0.2, 0.5, 0.75],
        generation_method="uep_cluster",
        source_structure_id=host.id,
    )
    for site in (relaxed_site, running_site):
        document.sites[site.id] = site

    calc_type = muon_relaxation_calc_type(POSITIVE_MUON_CHARGE_STATE)
    relaxed_job = Job(
        label="mu1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=calc_type,
        status=JobStatus.COMPLETED,
    )
    running_job = Job(
        label="mu2",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=calc_type,
        status=JobStatus.RUNNING,
    )
    for job in (relaxed_job, running_job):
        document.jobs[job.id] = job

    relaxed_cell = host_structure(
        label="mu1 relaxed",
        lattice=4.0,
        species=("Sr", "Ti", "O", "O", "O", "H:mu"),
        positions=(*PEROVSKITE_POSITIONS, (0.5, 0.25, 0.25)),
        origin="muon_relaxation",
        parent_structure_id=host.id,
    )
    document.structures[relaxed_cell.id] = relaxed_cell
    result = Result(
        result_type=calc_type,
        data={"total_energy": -1000.0, "muon_final_fractional_position_host": [0.5, 0.25, 0.25]},
        derived_structure_ids=[relaxed_cell.id],
    )
    run = CalculationRun(
        job_id=relaxed_job.id,
        status=JobStatus.COMPLETED,
        result_id=result.id,
        convergence_achieved=True,
    )
    result.calculation_run_id = run.id
    document.runs[run.id] = run
    document.results[result.id] = result
    relaxed_site.record_relaxation(run.id, total_energy_eV=-1000.0, fractional_position=[0.5, 0.25, 0.25])

    prefix = muon_charge_state_prefix(POSITIVE_MUON_CHARGE_STATE)
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=MUON_RELAXATION_DEFINITION_CALC_TYPE,
        input_structure_ids=[host.id],
        parameters={
            "configuration": {
                "rows": [
                    {"label": "mu1", "charge_state": POSITIVE_MUON_CHARGE_STATE, "site_id": str(relaxed_site.id)},
                    {"label": "mu2", "charge_state": POSITIVE_MUON_CHARGE_STATE, "site_id": str(running_site.id)},
                ]
            },
            "job_ids_by_label": {f"{prefix}:mu1": str(relaxed_job.id), f"{prefix}:mu2": str(running_job.id)},
        },
    )
    document.calculations[definition.id] = definition
    return MuonProject(document, host, relaxed_site, running_site, relaxed_job, running_job, definition)
