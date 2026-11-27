"""An Energy Barriers project with generated paths and a finished band, for the controller and window tests."""

from __future__ import annotations

from pathlib import Path

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition as _CalculationDefinition
from mufinder.domain.models import CalculationRun, Job
from mufinder.domain.models import MuonSite as _MuonSite
from mufinder.domain.models import Result as _Result
from mufinder.domain.models import Structure as _Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.workflow.energy_barriers import NUDGED_ELASTIC_BAND_CALC_TYPE


def seed_energy_barriers_document(
    service: ProjectService, tmp_path: Path, *, row_settings=None, paths: int = 1
):
    """Seed a document with a host, two relaxed endpoints, generated paths in the
    Energy Barriers form state, and a NEB definition holding them.

    The first path is keyed ``source-key``; with ``paths`` > 1 the others are
    keyed ``source-key-2`` and on, between the same endpoints. ``row_settings``
    is stored as the first path's configuration. Returns the document, the
    definition, the first path's source and the output folder.
    """

    document = service.create_project(tmp_path / "eb_mat.mufinder", "EBMat")
    host = _Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    document.structures[host.id] = host
    document.project.host_structure_id = host.id

    ids: dict[str, dict[str, str]] = {}
    for side, label, muon in (("reactant", "siteA", 0.25), ("product", "siteB", 0.75)):
        relaxed = _Structure(
            label=f"{label} relaxed",
            origin="muon_relaxation",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si", "Si", "H:mu"],
            fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [muon, 0.25, 0.25]],
            parent_structure_id=host.id,
            properties={
                "atomic_spins": [1.0, -1.0, 0.0],
                "source_workflow_stage": "muon_relaxation",
            },
        )
        document.structures[relaxed.id] = relaxed
        job = service.add_job(
            document,
            label=f"mu-{label}",
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type="castep_positivemuon_relaxation",
        )
        job.status = JobStatus.COMPLETED
        run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
        result = _Result(calculation_run_id=run.id, derived_structure_ids=[relaxed.id])
        run.result_id = result.id
        document.runs[run.id] = run
        document.results[result.id] = result
        site = _MuonSite(
            label=label, status="relaxed", relaxation_run_id=run.id, source_structure_id=host.id
        )
        document.sites[site.id] = site
        ids[side] = {
            "site": str(site.id),
            "run": str(run.id),
            "job": str(job.id),
            "structure": str(relaxed.id),
        }

    source = {
        "key": "source-key",
        "label": "siteA_to_siteB_path1",
        "path_index": 1,
        **{
            f"{side}_{field}_id": value
            for side, fields in ids.items()
            for field, value in fields.items()
        },
        "path_start_fractional": [0.25, 0.25, 0.25],
        "path_end_fractional": [0.75, 0.25, 0.25],
        "path_length_ang": 2.5,
        "charge_state": "positive_muon",
    }
    sources = [source] + [
        {
            **source,
            "key": f"source-key-{index}",
            "label": f"siteA_to_siteB_path{index}",
            "path_index": index,
        }
        for index in range(2, paths + 1)
    ]
    document.ui_state["energy_barriers_form"] = {
        "generated_sources": [dict(entry) for entry in sources]
    }
    output_root = document.project_root / "files" / "energy_barriers_neb"
    configuration: dict[str, object] = {"method": "NEB", "sources": sources}
    if row_settings is not None:
        configuration["row_settings"] = {"source-key": row_settings}
    definition = _CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type="castep_nudged_elastic_band_set",
        parameters={
            "output_directory": "files/energy_barriers_neb",
            "job_ids_by_label": {},
            "configuration": configuration,
        },
    )
    document.calculations[definition.id] = definition
    return document, definition, source, output_root


#: A finished seven-image band, as CASTEP writes it to a ``.ts`` file.
NEB_TS_FIXTURE = Path(__file__).parent.parent / "fixtures" / "neb_path.ts"


def neb_job(service: ProjectService, document: ProjectDocument, label: str, calc_type: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=calc_type,
    )


def add_band(
    service: ProjectService, document: ProjectDocument, ts_path: Path | str = "/tmp/barrier.ts"
) -> tuple[Job, _Result]:
    """Add a completed NEB job whose result reads its band from ``ts_path``."""

    job = neb_job(service, document, "barrier_neb", NUDGED_ELASTIC_BAND_CALC_TYPE)
    job.status = JobStatus.COMPLETED
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    result = _Result(
        calculation_run_id=run.id,
        result_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
        data={"transition_state_file": str(ts_path), "image_energies_eV": [-1.0, -0.8, -0.9]},
    )
    run.result_id = result.id
    document.runs[run.id] = run
    document.results[result.id] = result
    return job, result
