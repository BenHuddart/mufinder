"""A Quantum Corrections project with relaxed sources and a phonon fixture, for the tests.

``seed_quantum_document`` builds a host with two relaxed muon sites sharing one
phonon definition; ``bind_phonon`` and ``store_phonon_result`` give a row a
phonon job and its result. ``QUANTUM_PHONON_TEXT`` is a minimal ``.phonon`` file
with an ``H:mu`` muon.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import UUID

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Result, Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE
from tests.support.jobs import set_job_state


# A minimal, valid CASTEP ``.phonon`` file with one Gamma q-point and an H:mu
# atom (mirrors the fixture used by tests/app/test_main_window.py so build_result
# exercises the same real science path validate_artifacts accepts).
QUANTUM_PHONON_TEXT = """ BEGIN header
 Number of ions         2
 Number of branches     6
 Number of wavevectors  1
 Frequencies in         cm-1
 Unit cell vectors (A)
    1.000000    0.000000    0.000000
    0.000000    1.000000    0.000000
    0.000000    0.000000    1.000000
 Fractional Co-ordinates
     1     0.000000    0.000000    0.000000   N        14.006740
     2     0.500000    0.500000    0.500000   H:mu      0.113429
 END header
     q-pt=    1    0.000000  0.000000  0.000000      1.0000000000
       1     100.000000
       2     400.000000
       3     900.000000
       4    1600.000000
       5    2500.000000
       6    3600.000000
                        Phonon Eigenvectors
Mode Ion                X                                   Y                                   Z
   1   1  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   1   2  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   2   1 -0.336650000000  0.000000000000      0.019803000000  0.000000000000      0.138621000000  0.000000000000
   2   2 -0.693103000000  0.000000000000      0.376256000000  0.000000000000      0.495074000000  0.000000000000
   3   1 -0.080432200000  0.000000000000     -0.350117000000  0.000000000000      0.230257000000  0.000000000000
   3   2  0.189252000000  0.000000000000     -0.619801000000  0.000000000000      0.407510000000  0.000000000000
   4   1  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   4   2  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   5   1 -0.336650000000  0.000000000000      0.019803000000  0.000000000000      0.138621000000  0.000000000000
   5   2 -0.693103000000  0.000000000000      0.376256000000  0.000000000000      0.495074000000  0.000000000000
   6   1 -0.080432200000  0.000000000000     -0.350117000000  0.000000000000      0.230257000000  0.000000000000
   6   2  0.189252000000  0.000000000000     -0.619801000000  0.000000000000      0.407510000000  0.000000000000
"""


def quantum_job(
    service: ProjectService, document: ProjectDocument, label: str, calc_type: str
) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=calc_type,
    )


def store_phonon_result(
    document: ProjectDocument, job: Job, data: dict, created_at: datetime | None = None
) -> Result:
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    document.runs[run.id] = run
    result = Result(result_type=QUANTUM_PHONON_CALC_TYPE, calculation_run_id=run.id, data=data)
    if created_at is not None:
        result.created_at = created_at
    document.results[result.id] = result
    return result


def seed_quantum_document(service: ProjectService, tmp_path: Path):
    """Seed a document with two eligible relaxed sources sharing one definition.

    Returns ``(document, definition, candidates, output_root)``; ``candidates``
    maps each source key (its site id) to its real
    :class:`~mufinder.services.source_candidates.QuantumSourceCandidate`, for
    ``siteA`` then ``siteB``.
    """

    from mufinder.services.source_candidates import eligible_quantum_source_candidates

    document = service.create_project(tmp_path / "q2.mufinder", "Q2")
    output_root = document.project_root / "files" / "quantum_corrections"
    host = Structure(
        label="host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    document.structures[host.id] = host
    document.project.host_structure_id = host.id

    site_ids = []
    structure_ids = []
    for label in ("siteA", "siteB"):
        relaxed = Structure(
            label=f"{label} relaxed",
            origin="muon_relaxation",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si", "H:mu"],
            fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            parent_structure_id=host.id,
            properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
        )
        document.structures[relaxed.id] = relaxed
        source_job = service.add_job(
            document,
            label=f"mu-{label}",
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type="castep_positivemuon_relaxation",
        )
        source_job.status = JobStatus.COMPLETED
        run = CalculationRun(
            job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True
        )
        result = Result(calculation_run_id=run.id, derived_structure_ids=[relaxed.id])
        run.result_id = result.id
        document.runs[run.id] = run
        document.results[result.id] = result
        site = MuonSite(
            label=label,
            status="relaxed",
            relaxation_run_id=run.id,
            total_energy_eV=-1.0,
            source_structure_id=host.id,
        )
        document.sites[site.id] = site
        site_ids.append(site.id)
        structure_ids.append(relaxed.id)

    definition = service.register_quantum_phonon_definition(
        document,
        input_structure_ids=structure_ids,
        input_site_ids=site_ids,
        parameters={
            "phonon_job_ids_by_label": {},
            "job_seed_names_by_label": {},
        },
        output_directory=output_root,
    )
    candidates = {
        candidate.key: candidate for candidate in eligible_quantum_source_candidates(document)
    }
    assert [candidate.site.label for candidate in candidates.values()] == ["siteA", "siteB"]
    return document, definition, candidates, output_root


def bind_phonon(
    service: ProjectService,
    document: ProjectDocument,
    definition,
    key: str,
    *,
    status: JobStatus = JobStatus.DEFINED,
    recorded_source: bool = True,
) -> Job:
    """Bind a phonon job in ``status`` to the row ``key``, read from the row's current run."""

    phonon = quantum_job(service, document, f"quantum_phonon_{key}", QUANTUM_PHONON_CALC_TYPE)
    set_job_state(phonon, status)
    definition.parameters["phonon_job_ids_by_label"][key] = str(phonon.id)
    if recorded_source:
        source_run = document.sites[UUID(key)].relaxation_run_id
        definition.parameters.setdefault("phonon_source_run_ids_by_label", {})[key] = str(
            source_run
        )
    return phonon
