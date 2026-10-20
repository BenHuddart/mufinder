"""Factories for relaxed muon sites and the clustered analyses pinned from them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

from mufinder.domain.charge_states import POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Result, Structure
from mufinder.services.project_document import ProjectDocument
from mufinder.services.project_summary import RELAXATION_ANALYSIS_WORKSPACES_KEY

POSITIVE_RELAXATION_CALC_TYPE = "castep_positivemuon_relaxation"
MUONIUM_RELAXATION_CALC_TYPE = "castep_muonium_relaxation"


def add_relaxed_muon_site(
    doc: ProjectDocument,
    *,
    label: str,
    charge_calc_type: str = POSITIVE_RELAXATION_CALC_TYPE,
    species: Sequence[str] = ("Si", "H:mu"),
    total_energy: float = -100.0,
) -> MuonSite:
    """Add one eligible relaxed muon source whose site stores the absolute total energy."""

    positions = [[0.1 * index, 0.0, 0.0] for index in range(len(species) - 1)] + [[0.25, 0.25, 0.25]]
    relaxed = Structure(
        label=f"{label} relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=list(species),
        fractional_positions=positions,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed.id] = relaxed
    job = Job(
        label=f"relax-{label}",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=charge_calc_type,
        status=JobStatus.COMPLETED,
    )
    doc.jobs[job.id] = job
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result = Result(
        calculation_run_id=run.id,
        result_type=job.calc_type,
        data={"total_energy": total_energy},
        derived_structure_ids=[relaxed.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label=label, status="relaxed", relaxation_run_id=run.id, total_energy_eV=total_energy)
    doc.sites[site.id] = site
    return site


def relaxation_job_id(doc: ProjectDocument, site: MuonSite) -> UUID:
    """Return the id of the muon-relaxation job that produced ``site``."""

    run = doc.runs[site.relaxation_run_id]
    assert run.job_id is not None
    return run.job_id


@dataclass(frozen=True)
class ClusterSpec:
    """One cluster of a clustered analysis workspace."""

    cluster_index: int
    representative_job_id: UUID
    fractional_position: tuple[float, float, float] = (0.25, 0.25, 0.25)
    relative_energy_ev: float | None = 0.0
    wyckoff_label: str = ""


def clustered_workspace(
    host: Structure,
    clusters: Sequence[ClusterSpec],
    *,
    charge_state: str = POSITIVE_MUON_CHARGE_STATE,
    name: str = "Analysis",
) -> dict[str, object]:
    """Return a relaxation-analysis workspace dict with a clustering result."""

    return {
        "id": str(uuid4()),
        "name": name,
        "charge_state": charge_state,
        "host_structure_id": str(host.id),
        "job_ids": sorted({str(cluster.representative_job_id) for cluster in clusters}),
        "clustering": {
            "result": {
                "clusters": [
                    {
                        "cluster_index": cluster.cluster_index,
                        "size": 1,
                        "representative_job_id": str(cluster.representative_job_id),
                        "representative_fractional_position": list(cluster.fractional_position),
                        "minimum_relative_energy_ev": cluster.relative_energy_ev,
                    }
                    for cluster in clusters
                ],
                "wyckoff_assignments": [
                    {"cluster_index": cluster.cluster_index, "match": {"label": cluster.wyckoff_label}}
                    for cluster in clusters
                    if cluster.wyckoff_label
                ],
            }
        },
    }


def store_workspaces(doc: ProjectDocument, *workspaces: dict[str, object]) -> None:
    """Store ``workspaces`` as the project's relaxation-analysis workspaces."""

    doc.ui_state[RELAXATION_ANALYSIS_WORKSPACES_KEY] = list(workspaces)
