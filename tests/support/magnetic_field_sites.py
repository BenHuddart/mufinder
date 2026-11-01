"""Projects holding the muon sites the Magnetic Fields page offers.

The host is a 3 Å FeO cube with spins ``(1, -0.5)``. ``uep_cluster_document``
holds one UEP batch cluster, ``relaxation_cluster_document`` one
relaxation-analysis cluster whose representative relaxation produced a given
relaxed structure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from functools import partial
from pathlib import Path
from uuid import uuid4

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationRun, Job, Result, Structure
from mufinder.persistence.codec import to_jsonable
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    build_collinear_component,
)
from mufinder.science.uep_clustering import group_uep_minima
from mufinder.services.project_document import ProjectDocument
from mufinder.services.project_service import ProjectService
from mufinder.services.project_summary import RELAXATION_ANALYSIS_WORKSPACES_KEY
from mufinder.services.project_survey import UEP_ANALYSIS_STATES_KEY
from tests.support.structures import host_structure, relaxed_muon_supercell
from tests.support.uep_results import fake_optimization

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


def cof2_host() -> Structure:
    """Rutile CoF2, a host whose images of a site can see different fields."""

    a, c, x = 4.695, 3.180, 0.306
    return Structure(
        label="CoF2",
        lattice_vectors=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, c]],
        species=["Co", "Co", "F", "F", "F", "F"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [x, x, 0.0],
            [1 - x, 1 - x, 0.0],
            [0.5 + x, 0.5 - x, 0.5],
            [0.5 - x, 0.5 + x, 0.5],
        ],
    )


def cof2_antiferromagnet() -> MagneticStructureModel:
    """CoF2's antiferromagnet, moments along c."""

    return MagneticStructureModel(
        components=(
            build_collinear_component(
                label="Corner",
                selector=AtomSelector.indices([0]),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, 2.64),
                component_id="corner",
            ),
            build_collinear_component(
                label="Centre",
                selector=AtomSelector.indices([1]),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, -2.64),
                component_id="centre",
            ),
        ),
    )


def cof2_ferromagnet() -> MagneticStructureModel:
    """A hypothetical CoF2 ferromagnet, moments along a."""

    return MagneticStructureModel(
        components=(
            build_collinear_component(
                label="Ferromagnet",
                selector=AtomSelector.species("Co"),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(2.64, 0.0, 0.0),
                component_id="ferromagnet",
            ),
        ),
    )


def uep_search_document(
    tmp_path: Path,
    host: Structure,
    job_position: Sequence[float],
    *,
    supercell: Sequence[int] = (1, 1, 1),
    source: Structure | None = None,
) -> ProjectDocument:
    """A project with ``host`` and one UEP search that found one minimum.

    The search ran on ``supercell`` repeats of ``source`` (``host`` by default), and
    ``job_position`` is the minimum's fractional position in that job cell, which is how
    the search stores it. The state is grouped and saved as the UEP window saves it.
    """

    service = ProjectService()
    document = service.create_project(tmp_path / "uep-search.mufinder", "UEP search")
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    cell = source if source is not None else host
    if source is not None:
        document.structures[source.id] = source
    definition = service.register_uep_definition(
        document,
        input_structure_id=cell.id,
        parameters={"rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}]},
        output_directory=document.project_root / "files" / "uep",
    )
    job = service.add_job(
        document, label="uep_source", workflow_stage=WorkflowStage.UEP, calc_type="uep"
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    job_lattice = tuple(
        tuple(float(repeat) * float(value) for value in vector)
        for repeat, vector in zip(supercell, cell.lattice_vectors)
    )
    result = fake_optimization(
        (float(job_position[0]), float(job_position[1]), float(job_position[2])),
        classical=-16.0,
        lattice_vectors=job_lattice,
    )
    document.ui_state[UEP_ANALYSIS_STATES_KEY] = {
        str(job.id): to_jsonable(
            {
                "batch_results": [asdict(result)],
                "batch_summary_data": asdict(group_uep_minima([result], source_structure=cell)),
            }
        )
    }
    return document


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
