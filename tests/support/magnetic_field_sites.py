"""Projects holding the muon sites the Magnetic Fields page offers.

The host is a 3 Å FeO cube with spins ``(1, -0.5)``. ``uep_cluster_document``
holds one UEP batch cluster, ``relaxation_cluster_document`` one
relaxation-analysis cluster whose representative relaxation produced a given
relaxed structure.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from uuid import uuid4

from mufinder.domain.models import CalculationRun, Job, Result, Structure
from mufinder.services.project_document import ProjectDocument
from mufinder.services.project_service import ProjectService
from mufinder.services.project_summary import RELAXATION_ANALYSIS_WORKSPACES_KEY
from mufinder.services.project_survey import UEP_ANALYSIS_STATES_KEY
from tests.support.structures import host_structure, relaxed_muon_supercell

#: The FeO host every document here holds.
feo_host_structure = partial(
    host_structure,
    label="FeO",
    lattice=3.0,
    species=("Fe", "O"),
    properties={"atomic_spins": [1.0, -0.5], "propagation_vector": [0.0, 0.0, 0.0]},
)

#: Where the relaxation cluster's representative sits, in host fractional coordinates.
RELAXED_SITE = (0.5, 0.0, 0.0)
#: The relaxation-analysis workspace ``relaxation_cluster_document`` writes.
RELAXATION_WORKSPACE_NAME = "Positive muon relaxations"


def add_uep_cluster(document: ProjectDocument) -> None:
    """Give ``document`` one UEP batch cluster (index 2) at (0.1, 0.2, 0.3), from job "UEP batch"."""

    job_id = uuid4()
    document.jobs[job_id] = Job(id=job_id, label="UEP batch")
    document.ui_state[UEP_ANALYSIS_STATES_KEY] = {
        str(job_id): {
            "batch_results": [
                {
                    "optimized_fractional_position": [0.1, 0.2, 0.3],
                    "optimized_position_ang": [0.3, 0.6, 0.9],
                }
            ],
            "batch_summary_data": {
                "clusters": [
                    {
                        "cluster_index": 2,
                        "size": 4,
                        "representative_index": 0,
                    }
                ]
            },
        }
    }


def uep_cluster_document(tmp_path: Path) -> ProjectDocument:
    """A project with the FeO host and one UEP batch cluster (:func:`add_uep_cluster`)."""

    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-uep-sites.mufinder", "Magnetic UEP Sites"
    )
    service.add_structure(document, feo_host_structure())
    add_uep_cluster(document)
    return document


def relaxation_cluster_document(tmp_path: Path, relaxed_structure: Structure) -> ProjectDocument:
    """A project whose relaxation-analysis workspace has one cluster represented by ``relaxed_structure``."""

    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-relax-sites.mufinder", "Magnetic Relax Sites"
    )
    service.add_structure(document, feo_host_structure())
    document.structures[relaxed_structure.id] = relaxed_structure
    job_id = uuid4()
    result = Result(derived_structure_ids=[relaxed_structure.id])
    run = CalculationRun(job_id=job_id, result_id=result.id)
    document.results[result.id] = result
    document.runs[run.id] = run
    document.ui_state[RELAXATION_ANALYSIS_WORKSPACES_KEY] = [
        {
            "name": RELAXATION_WORKSPACE_NAME,
            "clustering": {
                "result": {
                    "clusters": [
                        {
                            "cluster_index": 1,
                            "size": 3,
                            "representative_job_id": str(job_id),
                            "representative_fractional_position": list(RELAXED_SITE),
                            "representative_cartesian_position_ang": [1.5, 0.0, 0.0],
                        }
                    ]
                }
            },
        }
    ]
    return document


def displaced_iron_supercell() -> Structure:
    """The FeO host relaxed around the muon, with the Fe one cell along +x pushed 0.1 Å away."""

    return relaxed_muon_supercell(
        feo_host_structure(), (2, 2, 2), RELAXED_SITE, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)}
    )
