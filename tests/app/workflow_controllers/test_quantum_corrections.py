"""Tests for :class:`QuantumCorrectionsController`.

Covers the two-stage (singlepoint, then phonon) submission, ``build_result`` on
a real ``.phonon`` file, the stage-two species-mass and execution-target guards,
the phonon-method checks and ``validate_artifacts``.
"""

from __future__ import annotations

import re
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, Structure
from mufinder.services.job_artifacts import ParsedJobArtifacts, collect_job_artifacts
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_SINGLEPOINT_CALC_TYPE

from mufinder.app.workflow_controllers.quantum_corrections import (
    QUANTUM_STAGE_PHONON,
    QUANTUM_STAGE_SINGLEPOINT,
    QuantumCorrectionsController,
    QuantumCorrectionsDeps,
)

from tests.support.controllers import (
    configure_result,
    RecordingBackend,
    completed_castep_result,
    make_context,
    no_artifacts,
)


# --------------------------------------------------------------------------- #
# QuantumCorrectionsController
# --------------------------------------------------------------------------- #
#
# The two-stage (singlepoint → phonon) migration. Its ``build_result`` carries the
# richest result logic — parsing a real ``.phonon`` file into the muon ZPE,
# eigenvalues, mode participations, and the ``w1`` entanglement witness — for a
# quantum-*phonon* job, while a quantum-*singlepoint* job contributes an empty
# result (but is still ``handles_job``-owned so the two-stage pair is coherent).
# ``pre_submit_check`` runs the shared host guard; ``materialize`` dispatches each
# selected source to its current stage. Driven below with narrow fakes (no
# MainWindow) and a REAL ``.phonon`` fixture containing an ``H:mu`` muon.


# A minimal, valid CASTEP ``.phonon`` file with one Gamma q-point and an H:mu
# atom (mirrors the fixture used by tests/app/test_main_window.py so build_result
# exercises the same real science path validate_artifacts accepts).
_QUANTUM_PHONON_TEXT = """ BEGIN header
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


class _FakeQuantumView:
    """Minimal QuantumCorrectionsWidget stand-in for controller tests."""

    def __init__(self) -> None:
        self.sources: list[list] = []
        self.results: list[list] = []

    def set_sources(self, rows) -> None:
        self.sources.append(list(rows))

    def set_results(self, rows) -> None:
        self.results.append(list(rows))


def _stub_quantum_deps(**overrides) -> QuantumCorrectionsDeps:
    base = dict(
        handle_structure_host_mismatch=lambda _mismatch: True,
        handle_site_host_mismatch=lambda _mismatch: True,
        confirm=lambda _title, _text: True,
        warn=lambda _title, _text: None,
        reset_jobs_keeping_files=lambda jobs: (len(jobs), []),
        refresh_after_reset=lambda: None,
    )
    base.update(overrides)
    return QuantumCorrectionsDeps(**base)


def _quantum_job(service: ProjectService, document: ProjectDocument, label: str, calc_type: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=calc_type,
    )


def test_quantum_controller_handles_both_stage_calc_types(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    controller = QuantumCorrectionsController(
        make_context(document=document), _stub_quantum_deps(), _FakeQuantumView()
    )
    singlepoint = _quantum_job(service, document, "quantum_singlepoint_s", QUANTUM_SINGLEPOINT_CALC_TYPE)
    phonon = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    other = service.add_job(
        document,
        label="u",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_uep_source",
    )
    # Both stage calc types are owned so the two-stage pair is coherent.
    assert controller.handles_job(singlepoint) is True
    assert controller.handles_job(phonon) is True
    assert controller.handles_job(other) is False


def test_quantum_controller_refresh_view_without_document_clears_tables() -> None:
    view = _FakeQuantumView()
    controller = QuantumCorrectionsController(
        make_context(document=None), _stub_quantum_deps(), view
    )
    controller.refresh_view()
    assert view.sources == [[]]
    assert view.results == [[]]


def test_quantum_controller_refresh_view_populates_result_row_status(tmp_path: Path) -> None:
    """A completed phonon result becomes a result row whose ``status`` mirrors the
    producing job's status, so the results table can colour it with the shared palette."""

    from mufinder.domain.models import Result

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    job.status = JobStatus.COMPLETED
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    document.runs[run.id] = run
    result = Result(
        result_type=QUANTUM_PHONON_CALC_TYPE,
        calculation_run_id=run.id,
        data={"muon_zpe_total_eV": 0.1},
    )
    document.results[result.id] = result

    view = _FakeQuantumView()
    controller = QuantumCorrectionsController(
        make_context(document=document, project_service=service), _stub_quantum_deps(), view
    )
    controller.refresh_view()

    assert len(view.results[-1]) == 1
    result_row = view.results[-1][0]
    assert result_row.label == "quantum_phonon_s"
    assert result_row.status == "completed"


def test_quantum_controller_pre_submit_check_blocks_submission(tmp_path: Path) -> None:
    """An unresolved host mismatch silently refuses the submission: the host guard
    returns False, so no submission plan is built and nothing is dispatched."""

    service = ProjectService()
    document, _definition, candidates, _output_root = _seed_quantum_two_stage_document(service, tmp_path)
    other_host = Structure(
        label="other host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    document.structures[other_host.id] = other_host
    document.project.host_structure_id = other_host.id
    mismatches: list[object] = []
    deps = _stub_quantum_deps(
        handle_structure_host_mismatch=lambda mismatch: mismatches.append(mismatch) or False,
    )
    local = RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = make_context(
        document=document, project_service=service, local_backend=local, infos=infos
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())
    materialize_calls: list[list] = []
    # Prove materialize is never reached by making it record if called.
    controller.materialize = lambda keys: materialize_calls.append(list(keys)) or []  # type: ignore[assignment]

    controller.submit(list(candidates), target="local")

    assert len(mismatches) == 1  # the relaxed structures descend from the other host
    assert local.calls == []  # nothing dispatched
    assert infos == []  # the refusal is silent (no "no ready jobs" dialog)
    assert materialize_calls == []  # no submission plan was even built


def test_quantum_controller_pre_submit_check_passes_when_host_matches(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, _output_root = _seed_quantum_two_stage_document(service, tmp_path)
    mismatches: list[object] = []
    deps = _stub_quantum_deps(
        handle_structure_host_mismatch=lambda mismatch: mismatches.append(mismatch) or False,
        handle_site_host_mismatch=lambda mismatch: mismatches.append(mismatch) or False,
    )
    controller = QuantumCorrectionsController(
        make_context(document=document, project_service=service), deps, _FakeQuantumView()
    )
    assert controller.pre_submit_check(list(candidates)) is True
    assert mismatches == []


def test_quantum_controller_build_result_singlepoint_is_empty_but_handled(tmp_path: Path) -> None:
    """A quantum-singlepoint job contributes an empty ControllerResult (base
    payload only), yet is still owned by the controller so the two-stage pair is
    coherent for result dispatch."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_singlepoint_s", QUANTUM_SINGLEPOINT_CALC_TYPE)
    run = CalculationRun(job_id=job.id)
    controller = QuantumCorrectionsController(
        make_context(document=document, project_service=service), _stub_quantum_deps(), _FakeQuantumView()
    )

    assert controller.handles_job(job) is True
    result = controller.build_result(job, completed_castep_result(), run, no_artifacts())
    assert result.payload == {}
    assert result.output_files == []
    assert result.derived_structure_ids == []
    assert result.warnings == []


def test_quantum_controller_build_result_phonon_parses_zpe_and_witness(tmp_path: Path) -> None:
    """A quantum-phonon job's build_result parses the real ``.phonon`` file and
    contributes the ZPE / eigenvalue / mode / witness payload — matching the
    fields the former inline result block assigned."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=job.id)

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    phonon_path = output_dir / "quantum_s.phonon"
    phonon_path.write_text(_QUANTUM_PHONON_TEXT, encoding="utf-8")

    deps = _stub_quantum_deps()
    controller = QuantumCorrectionsController(
        make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: output_dir,
            job_seed_name=lambda _job: "quantum_s",
        ),
        deps,
        _FakeQuantumView(),
    )

    artifacts = collect_job_artifacts(output_dir, "quantum_s", want_phonon=True)
    result = controller.build_result(job, completed_castep_result(), run, artifacts)

    payload = result.payload
    assert payload["phonon_file"] == str(phonon_path)
    assert payload["gamma_only"] is True
    assert payload["phonon_mode_eigenvalues_cm"] == [100.0, 400.0, 900.0, 1600.0, 2500.0, 3600.0]
    assert isinstance(payload["muon_zpe_total_eV"], float)
    assert payload["muon_zpe_total_eV"] > 0.0
    assert len(payload["muon_zpe_modes"]) == 3
    assert all("mode_index" in mode and "frequency_cm" in mode for mode in payload["muon_zpe_modes"])
    assert isinstance(payload["w1_entanglement_witness"], float)
    # No output_files or derived structures contributed by build_result.
    assert result.output_files == []
    assert result.derived_structure_ids == []
    # The muon mass CASTEP reported is the muon mass, so no warning is raised.
    assert result.warnings == []


def test_quantum_controller_build_result_warns_on_a_hydrogen_mass_muon(tmp_path: Path) -> None:
    """A phonon run CASTEP performed with the default hydrogen mass for H:mu is
    reported as a run warning (dashboard-visible), not only in the payload."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=job.id)

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / "quantum_s.phonon").write_text(
        _QUANTUM_PHONON_TEXT.replace("H:mu      0.113429", "H:mu      1.007940"),
        encoding="utf-8",
    )

    controller = QuantumCorrectionsController(
        make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: output_dir,
            job_seed_name=lambda _job: "quantum_s",
        ),
        _stub_quantum_deps(),
        _FakeQuantumView(),
    )

    artifacts = collect_job_artifacts(output_dir, "quantum_s", want_phonon=True)
    result = controller.build_result(job, completed_castep_result(), run, artifacts)

    assert len(result.warnings) == 1
    assert "not the muon mass" in result.warnings[0]
    assert any(
        "not the muon mass" in warning for warning in result.payload["phonon_warnings"]
    )


def test_quantum_controller_build_result_phonon_missing_dir_is_empty(tmp_path: Path) -> None:
    """If the job's output directory can't be resolved, build_result yields an
    empty result rather than crashing (validate_artifacts fails such jobs first)."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=job.id)
    deps = _stub_quantum_deps()
    controller = QuantumCorrectionsController(
        make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: None,
        ),
        deps,
        _FakeQuantumView(),
    )
    assert controller.build_result(job, completed_castep_result(), run, ParsedJobArtifacts()).payload == {}


def _seed_quantum_two_stage_document(service: ProjectService, tmp_path: Path):
    """Seed a document with two eligible relaxed sources sharing one definition.

    Returns ``(document, definition, candidates, output_root)``; ``candidates``
    maps each source key (its site id) to its real
    :class:`~mufinder.services.source_candidates.QuantumSourceCandidate`, for
    ``siteA`` then ``siteB``.
    """

    from mufinder.domain.models import MuonSite, Result
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
        run = CalculationRun(job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
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
            "method": "finite_displacement",
            "pseudopotential_family": "unknown",
            "singlepoint_job_ids_by_label": {},
            "phonon_job_ids_by_label": {},
            "job_seed_names_by_label": {},
        },
        output_directory=output_root,
    )
    candidates = {candidate.key: candidate for candidate in eligible_quantum_source_candidates(document)}
    assert [candidate.site.label for candidate in candidates.values()] == ["siteA", "siteB"]
    return document, definition, candidates, output_root


def _complete_stage_one(
    service: ProjectService,
    document: ProjectDocument,
    definition,
    key: str,
    *,
    execution_target: str = "local",
) -> Job:
    """Bind a completed stage-1 singlepoint to ``key`` so the row is on the phonon stage."""

    singlepoint = _quantum_job(service, document, f"quantum_singlepoint_{key}", QUANTUM_SINGLEPOINT_CALC_TYPE)
    singlepoint.status = JobStatus.COMPLETED
    singlepoint.execution_target = execution_target
    definition.parameters["singlepoint_job_ids_by_label"][key] = str(singlepoint.id)
    return singlepoint


def _quantum_document_controller(service: ProjectService, document, output_root, *, deps=None, **context_kwargs):
    context = make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _definition: output_root,
        **context_kwargs,
    )
    return QuantumCorrectionsController(context, deps or _stub_quantum_deps(), _FakeQuantumView())


def test_quantum_controller_materialize_two_stage_mixed_keys(tmp_path: Path) -> None:
    """Mixed selection where one source is at the singlepoint stage and another at
    the phonon stage: materialize writes each key into the correct per-stage
    mapping with the correct calc_type, in one pass."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(
        service, tmp_path
    )
    singlepoint_key, phonon_key = list(candidates)

    # siteA is fresh (singlepoint stage); siteB already has a COMPLETED singlepoint
    # job, so its stage has advanced to phonon.
    completed_singlepoint = _complete_stage_one(service, document, definition, phonon_key)
    controller = _quantum_document_controller(service, document, output_root)

    jobs = controller.materialize([singlepoint_key, phonon_key])

    assert len(jobs) == 2
    calc_types = {job.calc_type for job in jobs}
    assert calc_types == {QUANTUM_SINGLEPOINT_CALC_TYPE, QUANTUM_PHONON_CALC_TYPE}

    singlepoint_mapping = definition.parameters["singlepoint_job_ids_by_label"]
    phonon_mapping = definition.parameters["phonon_job_ids_by_label"]
    # siteA got a fresh singlepoint job; siteB got a fresh phonon continuation.
    assert singlepoint_key in singlepoint_mapping
    new_singlepoint_id = singlepoint_mapping[singlepoint_key]
    assert document.jobs[UUID(new_singlepoint_id)].calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE
    assert phonon_key in phonon_mapping
    new_phonon_id = phonon_mapping[phonon_key]
    assert document.jobs[UUID(new_phonon_id)].calc_type == QUANTUM_PHONON_CALC_TYPE
    # The pre-existing completed singlepoint for siteB is untouched (not re-created).
    assert singlepoint_mapping.get(phonon_key) == str(completed_singlepoint.id)
    # Both seed names were recorded, and both stages write into the seed directory.
    seed_mapping = definition.parameters["job_seed_names_by_label"]
    assert seed_mapping == {singlepoint_key: "quantum_siteA", phonon_key: "quantum_siteB"}
    assert (output_root / "quantum_siteA" / "quantum_siteA.param").is_file()
    assert (output_root / "quantum_siteB" / "quantum_siteB.cell").is_file()
    # The phonon mapping stays aliased into job_ids_by_label (load-bearing fallback).
    assert definition.parameters["job_ids_by_label"] is phonon_mapping


def _write_stage_one_cell(output_root: Path, seed_name: str, *, with_species_mass: bool) -> Path:
    """Write a stage-1 singlepoint ``.cell`` into the shared seed directory."""

    job_dir = output_root / seed_name
    job_dir.mkdir(parents=True, exist_ok=True)
    cell_path = job_dir / f"{seed_name}.cell"
    species_mass = (
        "\n%BLOCK SPECIES_MASS\nAMU\nH:mu 0.11342892566\n%ENDBLOCK SPECIES_MASS\n"
        if with_species_mass
        else ""
    )
    cell_path.write_text(
        "%BLOCK LATTICE_CART\nang\n5 0 0\n0 5 0\n0 0 5\n%ENDBLOCK LATTICE_CART\n"
        "\n%BLOCK POSITIONS_FRAC\nSi 0 0 0\nH:mu 0.25 0.25 0.25\n%ENDBLOCK POSITIONS_FRAC\n"
        f"{species_mass}",
        encoding="utf-8",
    )
    return cell_path


def _quantum_stage_two_guard_setup(service: ProjectService, tmp_path: Path, *, stage_one_done: bool = True):
    """Return ``(controller, phonon_key, seed_name, output_root, infos)`` for the guard.

    ``siteA`` is on the phonon stage (its singlepoint completed) unless
    ``stage_one_done`` is false.
    """

    document, definition, candidates, output_root = _seed_quantum_two_stage_document(
        service, tmp_path
    )
    phonon_key = list(candidates)[0]
    if stage_one_done:
        _complete_stage_one(service, document, definition, phonon_key)
    infos: list[tuple[str, str]] = []
    controller = _quantum_document_controller(service, document, output_root, infos=infos)
    return controller, phonon_key, controller.seed_name_for_key(definition, phonon_key), output_root, infos


def test_quantum_controller_refuses_stage_two_when_stage_one_omits_species_mass(
    tmp_path: Path,
) -> None:
    """A phonon continuation whose stage-1 cell has no SPECIES_MASS is refused.

    CASTEP restores species masses from the stage-1 checkpoint, so continuing from
    a singlepoint that ran without the muon mass would silently produce phonons at
    the default hydrogen mass. The refusal explains itself rather than being silent.
    """

    controller, phonon_key, seed_name, output_root, infos = _quantum_stage_two_guard_setup(
        ProjectService(), tmp_path
    )
    _write_stage_one_cell(output_root, seed_name, with_species_mass=False)

    assert controller.pre_submit_check([phonon_key]) is False
    assert len(infos) == 1
    title, message = infos[0]
    assert title == "Stage-1 muon mass missing"
    assert "SPECIES_MASS" in message
    assert "siteA" in message


def test_quantum_controller_allows_stage_two_when_stage_one_declares_species_mass(
    tmp_path: Path,
) -> None:
    controller, phonon_key, seed_name, output_root, infos = _quantum_stage_two_guard_setup(
        ProjectService(), tmp_path
    )
    _write_stage_one_cell(output_root, seed_name, with_species_mass=True)

    assert controller.pre_submit_check([phonon_key]) is True
    assert infos == []


def test_quantum_controller_stage_two_guard_fails_open_without_a_stage_one_cell(
    tmp_path: Path,
) -> None:
    """No readable stage-1 cell is no evidence: the guard must not block."""

    controller, phonon_key, seed_name, output_root, infos = _quantum_stage_two_guard_setup(
        ProjectService(), tmp_path
    )
    unreadable = output_root / seed_name / f"{seed_name}.cell"
    unreadable.parent.mkdir(parents=True, exist_ok=True)
    unreadable.write_text("this is not a cell file", encoding="utf-8")

    # Unparseable cell, and (below) no cell at all: both allow submission.
    assert controller.pre_submit_check([phonon_key]) is True
    unreadable.unlink()
    assert controller.pre_submit_check([phonon_key]) is True
    assert infos == []


def test_quantum_controller_stage_two_guard_ignores_stage_one_rows(tmp_path: Path) -> None:
    """A row still on stage 1 is never blocked: it has no checkpoint to inherit."""

    controller, phonon_key, seed_name, output_root, infos = _quantum_stage_two_guard_setup(
        ProjectService(), tmp_path, stage_one_done=False
    )
    _write_stage_one_cell(output_root, seed_name, with_species_mass=False)

    assert controller.pre_submit_check([phonon_key]) is True
    assert infos == []


def test_quantum_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    """End-to-end submit wiring: build_states → is_ready (base default) →
    jobs_for_keys → materialize → local backend, dispatched as a single group with
    no extra kwargs (task farming lives in the shared backend, not the controller)."""

    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_two_stage_document(
        service, tmp_path
    )
    key = next(iter(candidates))
    local = RecordingBackend()
    cluster = RecordingBackend()
    controller = _quantum_document_controller(
        service, document, output_root, local_backend=local, cluster_backend=cluster
    )

    controller.submit([key], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted, extras = local.calls[0]
    assert extras == {}
    assert len(submitted) == 1
    assert submitted[0].calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE


def test_quantum_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_two_stage_document(
        service, tmp_path
    )
    key = next(iter(candidates))
    local = RecordingBackend()
    cluster = RecordingBackend()
    controller = _quantum_document_controller(
        service, document, output_root, local_backend=local, cluster_backend=cluster
    )

    controller.submit([key], target="cluster")

    assert local.calls == []
    assert len(cluster.calls) == 1
    assert cluster.calls[0][0][0].calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE



def test_quantum_controller_configure_request_opens_editable_stage_one_rows(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    controller = _quantum_document_controller(service, document, output_root)

    request = controller.configure_request([*candidates, "not-a-source"])

    assert request is not None
    assert request.keys == tuple(candidates)
    assert request.title == "Configure Quantum Corrections Parameters"
    assert (request.read_only, request.fully_editable, request.farms_only) == (False, True, False)
    assert request.pseudopotential_family == "unknown"
    assert request.num_farms == 1
    assert request.species_options == ("Si", "H:mu")
    assert "Selected sources: 2" in request.intro_text
    assert "Current stage: mixed/selected" in request.intro_text
    assert "Displacements required for the phonon stage: 6" in request.intro_text
    # Every row inherits the tight stage default and the legacy method.
    assert all(values["phonon_method"] == "FINITEDISPLACEMENT" for values in request.row_values)
    assert controller.configure_request(["not-a-source"]) is None
    assert QuantumCorrectionsController(
        make_context(document=None), _stub_quantum_deps(), _FakeQuantumView()
    ).configure_request(list(candidates)) is None


def test_quantum_controller_configure_request_follows_each_rows_stage(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    stage_one_key, stage_two_key = list(candidates)
    _complete_stage_one(service, document, definition, stage_two_key)
    controller = _quantum_document_controller(service, document, output_root)

    farms_only = controller.configure_request([stage_two_key])
    assert farms_only is not None
    assert farms_only.title == "Configure Quantum Corrections Stage 2 Farms"
    assert (farms_only.read_only, farms_only.fully_editable, farms_only.farms_only) == (False, False, True)
    assert "Current stage: phonon" in farms_only.intro_text

    assert "only Stage 2 farms can change" in farms_only.locked_reason
    assert farms_only.locked_reason in farms_only.intro_text

    mixed = controller.configure_request([stage_one_key, stage_two_key])
    assert mixed is not None
    assert mixed.title == "Inspect Quantum Corrections Parameters"
    assert mixed.read_only is True
    assert mixed.locked_reason == "The selection mixes stage-1 and stage-2 rows. Configure them separately."


def test_quantum_controller_configure_request_edits_a_defined_singlepoint(tmp_path: Path) -> None:
    """A staged-but-unsubmitted or dashboard-reset singlepoint is rewritten on submit, so it stays editable."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    singlepoint = _quantum_job(service, document, "quantum_singlepoint_siteA", QUANTUM_SINGLEPOINT_CALC_TYPE)
    definition.parameters["singlepoint_job_ids_by_label"][key] = str(singlepoint.id)
    controller = _quantum_document_controller(service, document, output_root)

    request = controller.configure_request([key])

    assert singlepoint.status == JobStatus.DEFINED
    assert request is not None
    assert (request.read_only, request.fully_editable, request.farms_only) == (False, True, False)
    assert request.title == "Configure Quantum Corrections Parameters"
    assert request.locked_reason == ""


@pytest.mark.parametrize("dismissed", [False, True])
def test_quantum_controller_configure_request_edits_farms_of_a_reopened_phonon_job(
    tmp_path: Path, dismissed: bool
) -> None:
    """After a phonon failure is reset to ``defined`` or dismissed, its farms can change again."""

    from datetime import datetime, timezone

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    _complete_stage_one(service, document, definition, key)
    phonon = _quantum_job(service, document, "quantum_phonon_siteA", QUANTUM_PHONON_CALC_TYPE)
    if dismissed:
        phonon.status = JobStatus.FAILED
        phonon.dismissed_at = datetime.now(timezone.utc)
    definition.parameters["phonon_job_ids_by_label"][key] = str(phonon.id)
    controller = _quantum_document_controller(service, document, output_root)

    request = controller.configure_request([key])

    assert request is not None
    assert (request.read_only, request.fully_editable, request.farms_only) == (False, False, True)
    assert request.title == "Configure Quantum Corrections Stage 2 Farms"


@pytest.mark.parametrize(
    ("singlepoint_status", "phonon_status", "expected"),
    [
        (JobStatus.FAILED, None, "stage-1 singlepoint is failed. Use Reset to Stage 1"),
        (JobStatus.RUNNING, None, "job is running. Cancel the job first, then use Reset to Stage 1"),
        (JobStatus.COMPLETED, JobStatus.QUEUED, "job is queued. Cancel the job first"),
        (JobStatus.COMPLETED, JobStatus.FAILED, "phonon job is failed. Use Reset to Stage 2"),
        (JobStatus.COMPLETED, JobStatus.CANCELLED, "phonon job is cancelled. Use Reset to Stage 2"),
        (JobStatus.COMPLETED, JobStatus.COMPLETED, "completed both stages. Use Reset to Stage 1"),
    ],
)
def test_quantum_controller_read_only_configure_names_the_unlocking_reset(
    tmp_path: Path, singlepoint_status: JobStatus, phonon_status: JobStatus | None, expected: str
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    singlepoint = _complete_stage_one(service, document, definition, key)
    singlepoint.status = singlepoint_status
    if phonon_status is not None:
        phonon = _quantum_job(service, document, "quantum_phonon_siteA", QUANTUM_PHONON_CALC_TYPE)
        phonon.status = phonon_status
        definition.parameters["phonon_job_ids_by_label"][key] = str(phonon.id)
    controller = _quantum_document_controller(service, document, output_root)

    request = controller.configure_request([key])

    assert request is not None
    assert request.read_only is True
    assert request.title == "Inspect Quantum Corrections Parameters"
    assert expected in request.locked_reason
    assert request.intro_text.endswith(request.locked_reason)


def test_quantum_controller_apply_configuration_stores_every_row(tmp_path: Path) -> None:
    import dataclasses

    from mufinder.app.workflow_config_codecs import (
        store_workflow_row_configuration,
        workflow_row_configuration,
    )

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    first, second = list(candidates)
    store_workflow_row_configuration(definition, first, {"parameter_values": {"max_scf_cycles": 80}})
    persisted: list[str] = []
    view = _FakeQuantumView()
    context = make_context(document=document, project_service=service)
    controller = QuantumCorrectionsController(
        dataclasses.replace(
            context, persist_ui_state=lambda **kwargs: persisted.append(kwargs["failure_message"])
        ),
        _stub_quantum_deps(),
        view,
    )

    controller.apply_configuration(
        [first, second],
        configure_result({"task": "SINGLEPOINT", "xc_functional": "PBE"}, untouched={"max_scf_cycles"}),
        {"pseudopotential_family": "norm_conserving", "num_farms": 2},
    )

    first_params, _advanced, first_extras = workflow_row_configuration(definition, first)
    second_params, _advanced, second_extras = workflow_row_configuration(definition, second)
    assert first_params == {"xc_functional": "PBE", "max_scf_cycles": 80}
    assert second_params == {"xc_functional": "PBE"}
    assert first_extras == second_extras == {"pseudopotential_family": "norm_conserving", "num_farms": 2}
    assert definition.parameters["configuration"]["sources"]  # the definition now records its sources
    assert persisted == ["Quantum Corrections configuration was updated, but autosave failed."]
    assert view.sources  # refreshed


def test_quantum_controller_renames_a_source_site(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key, candidate = next(iter(candidates.items()))
    controller = _quantum_document_controller(service, document, output_root)

    assert controller.rename_source(key, "  ") is False
    assert controller.rename_source(key, "siteA") is False
    assert controller.rename_source("missing", "renamed") is False
    assert candidate.site.label == "siteA"
    assert controller.rename_source(key, " renamed ") is True
    assert document.sites[candidate.site.id].label == "renamed"


def test_quantum_controller_lists_stage_one_resets(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    finished_key, running_key = list(candidates)
    singlepoint = _complete_stage_one(service, document, definition, finished_key)
    phonon = _quantum_job(service, document, "quantum_phonon_siteA", QUANTUM_PHONON_CALC_TYPE)
    phonon.status = JobStatus.FAILED
    definition.parameters["phonon_job_ids_by_label"][finished_key] = str(phonon.id)
    running = _complete_stage_one(service, document, definition, running_key)
    running.status = JobStatus.RUNNING
    definition.parameters["job_seed_names_by_label"][finished_key] = "quantum_siteA"
    controller = _quantum_document_controller(service, document, output_root)

    resets = controller.stage_one_resets([finished_key, running_key, "missing"])

    assert resets is not None
    assert [(reset.key, reset.label, reset.jobs) for reset in resets] == [
        (finished_key, "siteA", (phonon, singlepoint))
    ]
    controller.forget_seed_name(finished_key)
    assert finished_key not in definition.parameters["job_seed_names_by_label"]

    empty = service.create_project(tmp_path / "empty.mufinder", "Empty")
    assert _quantum_document_controller(service, empty, output_root).stage_one_resets([finished_key]) is None


@pytest.mark.parametrize(
    ("status", "resettable"),
    [
        (JobStatus.DEFINED, True),
        (JobStatus.COMPLETED, True),
        (JobStatus.FAILED, True),
        (JobStatus.STOPPED, True),
        (JobStatus.CANCELLED, True),
        (JobStatus.SUBMITTING, False),
        (JobStatus.QUEUED, False),
        (JobStatus.RUNNING, False),
        (JobStatus.RETRIEVING, False),
    ],
)
def test_quantum_controller_resets_to_stage_one_from_any_inactive_phonon_job(
    tmp_path: Path, status: JobStatus, resettable: bool
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    _complete_stage_one(service, document, definition, key)
    phonon = _quantum_job(service, document, "quantum_phonon_siteA", QUANTUM_PHONON_CALC_TYPE)
    phonon.status = status
    definition.parameters["phonon_job_ids_by_label"][key] = str(phonon.id)
    controller = _quantum_document_controller(service, document, output_root)

    assert controller.can_reset_to_stage_one(definition, candidates[key]) is resettable


def _reset_recording_controller(service, document, output_root, *, confirm: bool, refusal: str | None = None):
    """Return a controller whose reset collaborators record what they were asked to do.

    File and record clean-up runs through the context's real ``JobFileCleanup``,
    wrapped to record the paths and jobs it deletes; ``refusal`` stands in for
    another job using the row's directory.
    """

    import dataclasses

    calls: dict[str, list] = {
        "confirm": [],
        "deleted_paths": [],
        "deleted_jobs": [],
        "reset_jobs": [],
        "refreshes": [],
        "warnings": [],
        "infos": [],
        "status": [],
        "persisted": [],
    }

    def reset_jobs_keeping_files(jobs) -> tuple[int, list[str]]:
        calls["reset_jobs"].append(list(jobs))
        for job in jobs:
            service.reset_failed_job(job)
        return len(jobs), []

    deps = _stub_quantum_deps(
        confirm=lambda title, text: calls["confirm"].append((title, text)) or confirm,
        warn=lambda title, text: calls["warnings"].append(text),
        reset_jobs_keeping_files=reset_jobs_keeping_files,
        refresh_after_reset=lambda: calls["refreshes"].append(True),
    )
    context = make_context(
        document=document,
        project_service=service,
        infos=calls["infos"],
        definition_output_directory=lambda _definition: output_root,
        job_output_directory=lambda job: (
            output_root / "quantum_siteA" if job.workflow_stage == WorkflowStage.QUANTUM_CORRECTIONS else None
        ),
        job_seed_name=lambda _job: "quantum_siteA",
    )
    context = dataclasses.replace(
        context,
        status_message=calls["status"].append,
        persist_ui_state=lambda **kwargs: calls["persisted"].append(kwargs["failure_message"]),
    )
    cleanup = context.job_file_cleanup
    real_delete_path, real_delete_job_records = cleanup.delete_path, cleanup.delete_job_records

    def delete_path(path) -> None:
        calls["deleted_paths"].append(path)
        real_delete_path(path)

    def delete_job_records(job_id) -> None:
        calls["deleted_jobs"].append(job_id)
        real_delete_job_records(job_id)

    cleanup.delete_path = delete_path  # type: ignore[method-assign]
    cleanup.delete_job_records = delete_job_records  # type: ignore[method-assign]
    if refusal is not None:
        cleanup.directory_ownership_refusals = (  # type: ignore[method-assign]
            lambda jobs, **_kwargs: {jobs[0].id: refusal}
        )
    return QuantumCorrectionsController(context, deps, _FakeQuantumView()), calls


def _row_with_failed_phonon(service, document, definition, key):
    singlepoint = _complete_stage_one(service, document, definition, key)
    phonon = _quantum_job(service, document, "quantum_phonon_siteA", QUANTUM_PHONON_CALC_TYPE)
    phonon.status = JobStatus.FAILED
    definition.parameters["phonon_job_ids_by_label"][key] = str(phonon.id)
    definition.parameters["job_seed_names_by_label"][key] = "quantum_siteA"
    return singlepoint, phonon


def test_quantum_controller_reset_to_stage_one_deletes_the_row_after_confirmation(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    singlepoint, phonon = _row_with_failed_phonon(service, document, definition, key)
    (output_root / "quantum_siteA").mkdir(parents=True)
    controller, calls = _reset_recording_controller(service, document, output_root, confirm=True)

    controller.reset_to_stage_one([key])

    assert calls["confirm"][0][0] == "Reset Quantum Corrections"
    assert calls["deleted_paths"] == [output_root / "quantum_siteA"]
    assert not (output_root / "quantum_siteA").exists()
    assert calls["deleted_jobs"] == [phonon.id, singlepoint.id]
    assert phonon.id not in document.jobs and singlepoint.id not in document.jobs
    assert key not in definition.parameters["job_seed_names_by_label"]
    assert calls["refreshes"] == [True]
    assert calls["status"] == ["Reset 1 Quantum Corrections row(s) to stage 1"]
    assert calls["persisted"] == ["Quantum Corrections stage reset was recorded, but autosave failed."]


def test_quantum_controller_reset_to_stage_one_does_nothing_when_declined(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    _row_with_failed_phonon(service, document, definition, key)
    controller, calls = _reset_recording_controller(service, document, output_root, confirm=False)

    controller.reset_to_stage_one([key])

    assert len(calls["confirm"]) == 1
    assert calls["deleted_paths"] == calls["deleted_jobs"] == calls["refreshes"] == calls["status"] == []
    assert definition.parameters["job_seed_names_by_label"] == {key: "quantum_siteA"}


def test_quantum_controller_reset_to_stage_one_keeps_a_shared_directory(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    _row_with_failed_phonon(service, document, definition, key)
    controller, calls = _reset_recording_controller(
        service, document, output_root, confirm=True, refusal="- quantum_siteA: another job uses the same output folder"
    )

    controller.reset_to_stage_one([key])

    assert calls["deleted_paths"] == calls["deleted_jobs"] == calls["status"] == []
    assert calls["warnings"] == [
        "Some Quantum Corrections rows could not be reset cleanly:\n"
        "- quantum_siteA: another job uses the same output folder; the row was not reset"
    ]


def test_quantum_controller_reset_to_stage_one_explains_an_active_row(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    _complete_stage_one(service, document, definition, key).status = JobStatus.QUEUED
    controller, calls = _reset_recording_controller(service, document, output_root, confirm=True)

    controller.reset_to_stage_one([key])

    assert calls["confirm"] == []
    assert calls["infos"][0][0] == "No resettable rows"
    assert "Cancel an active job first." in calls["infos"][0][1]


@pytest.mark.parametrize(
    ("singlepoint_status", "phonon_status", "resettable"),
    [
        (JobStatus.COMPLETED, JobStatus.FAILED, True),
        (JobStatus.COMPLETED, JobStatus.STOPPED, True),
        (JobStatus.COMPLETED, JobStatus.CANCELLED, True),
        (JobStatus.COMPLETED, JobStatus.COMPLETED, False),
        (JobStatus.COMPLETED, JobStatus.DEFINED, False),
        (JobStatus.COMPLETED, JobStatus.RUNNING, False),
        (JobStatus.COMPLETED, None, False),
        (JobStatus.FAILED, None, False),
    ],
)
def test_quantum_controller_reset_to_stage_two_gate(
    tmp_path: Path, singlepoint_status: JobStatus, phonon_status: JobStatus | None, resettable: bool
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    _complete_stage_one(service, document, definition, key).status = singlepoint_status
    if phonon_status is not None:
        phonon = _quantum_job(service, document, "quantum_phonon_siteA", QUANTUM_PHONON_CALC_TYPE)
        phonon.status = phonon_status
        definition.parameters["phonon_job_ids_by_label"][key] = str(phonon.id)
    controller = _quantum_document_controller(service, document, output_root)

    assert controller.can_reset_to_stage_two(definition, candidates[key]) is resettable
    assert bool(controller.stage_two_resets([key])) is resettable


def test_quantum_controller_reset_to_stage_two_reopens_only_the_phonon_job(tmp_path: Path) -> None:
    from mufinder.app.workflow_config_codecs import store_workflow_row_configuration

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    singlepoint, phonon = _row_with_failed_phonon(service, document, definition, key)
    controller, calls = _reset_recording_controller(service, document, output_root, confirm=True)

    controller.reset_to_stage_two([key])

    assert calls["confirm"][0][0] == "Reset Quantum Corrections to stage 2"
    assert calls["reset_jobs"] == [[phonon]]
    assert calls["deleted_paths"] == calls["deleted_jobs"] == []
    assert phonon.status == JobStatus.DEFINED
    assert singlepoint.status == JobStatus.COMPLETED
    assert calls["status"] == ["Reset 1 Quantum Corrections row(s) to stage 2"]

    # The reopened phonon stage takes a new farm count and is rewritten on submit.
    request = controller.configure_request([key])
    assert request is not None and request.farms_only is True
    store_workflow_row_configuration(definition, key, {"num_farms": 3})
    assert controller.materialize([key]) == [phonon]
    param_text = (output_root / "quantum_siteA" / "quantum_siteA.param").read_text(encoding="utf-8")
    assert re.search(r"^NUM_FARMS\s*:\s*3\s*$", param_text, re.MULTILINE)
    assert re.search(r"^TASK\s*:\s*PHONON\s*$", param_text, re.MULTILINE)


def test_quantum_controller_reset_to_stage_two_does_nothing_when_declined(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    _singlepoint, phonon = _row_with_failed_phonon(service, document, definition, key)
    controller, calls = _reset_recording_controller(service, document, output_root, confirm=False)

    controller.reset_to_stage_two([key])

    assert len(calls["confirm"]) == 1
    assert calls["reset_jobs"] == calls["status"] == []
    assert phonon.status == JobStatus.FAILED


def test_quantum_controller_reset_to_stage_two_explains_when_nothing_qualifies(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    _complete_stage_one(service, document, definition, key)
    controller, calls = _reset_recording_controller(service, document, output_root, confirm=True)

    controller.reset_to_stage_two([key])

    assert calls["confirm"] == []
    assert calls["infos"][0][0] == "No rows to reset to stage 2"


def test_quantum_controller_resolves_where_a_phonon_job_continues(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    key = next(iter(candidates))
    singlepoint = _complete_stage_one(service, document, definition, key, execution_target="archer2")
    singlepoint.remote_directory = "/work/quantum_siteA"
    phonon = _quantum_job(service, document, "quantum_phonon_siteA", QUANTUM_PHONON_CALC_TYPE)
    definition.parameters["phonon_job_ids_by_label"][key] = str(phonon.id)
    controller = _quantum_document_controller(service, document, output_root)

    assert controller.definition_and_key_for_job(phonon) == (definition, key)
    assert controller.stage_one_target(phonon) == "archer2"
    assert controller.continuation_remote_target(phonon, cluster_name="ARCHER2 ") == "/work/quantum_siteA"
    assert controller.continuation_remote_target(phonon, cluster_name="young") == ""
    # A singlepoint job has no stage-1 target of its own.
    assert controller.stage_one_target(singlepoint) is None
    assert controller.continuation_remote_target(singlepoint, cluster_name="archer2") == ""
    singlepoint.status = JobStatus.FAILED
    assert controller.stage_one_target(phonon) is None


def test_quantum_controller_resolves_the_phonon_analysis_input(tmp_path: Path) -> None:
    from mufinder.domain.models import Result

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_siteA", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    document.runs[run.id] = run
    result = Result(calculation_run_id=run.id, result_type=QUANTUM_PHONON_CALC_TYPE, data={})
    document.results[result.id] = result
    controller = QuantumCorrectionsController(
        make_context(document=document, project_service=service), _stub_quantum_deps(), _FakeQuantumView()
    )

    assert controller.phonon_result("not-a-uuid") is None
    assert controller.phonon_result(str(uuid4())) is None
    assert controller.phonon_result(str(result.id)) is result
    with pytest.raises(ValueError, match="does not reference a phonon file"):
        controller.phonon_analysis_input(result)
    result.data["phonon_file"] = "files/site.phonon"
    with pytest.raises(ValueError, match="Could not find phonon file"):
        controller.phonon_analysis_input(result)
    phonon_path = document.project_root / "files" / "site.phonon"
    phonon_path.parent.mkdir(parents=True, exist_ok=True)
    phonon_path.write_text(_QUANTUM_PHONON_TEXT, encoding="utf-8")
    assert controller.phonon_analysis_input(result) == (phonon_path, "quantum_phonon_siteA")


def _stage_two_target_controller(
    service: ProjectService,
    tmp_path: Path,
    *,
    stage_one_targets: list[str],
    infos: list[tuple[str, str]],
    local: RecordingBackend | None = None,
    cluster: RecordingBackend | None = None,
):
    """Build a controller whose rows are all ready for stage 2.

    Row ``i`` (sites ``siteA``, ``siteB``) has a completed stage-1 singlepoint
    that ran on ``stage_one_targets[i]``. Materialize is stubbed to hand back one
    defined phonon job per row, so an allowed submission reaches the backend.
    """

    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    keys = list(candidates)[: len(stage_one_targets)]
    for key, stage_one in zip(keys, stage_one_targets, strict=True):
        _complete_stage_one(service, document, definition, key, execution_target=stage_one)
    controller = _quantum_document_controller(
        service, document, output_root, local_backend=local, cluster_backend=cluster, infos=infos
    )
    phonon_jobs = {
        key: _quantum_job(service, document, f"quantum_phonon_{key}", QUANTUM_PHONON_CALC_TYPE) for key in keys
    }
    controller.materialize = lambda values: [phonon_jobs[str(value)] for value in values]  # type: ignore[assignment]
    return controller, keys


def test_quantum_controller_refuses_local_stage_two_after_a_cluster_singlepoint(tmp_path: Path) -> None:
    """The stage-1 checkpoint is never downloaded, so a local continuation would fail."""

    infos: list[tuple[str, str]] = []
    local = RecordingBackend()
    controller, keys = _stage_two_target_controller(
        ProjectService(), tmp_path, stage_one_targets=["archer2"], infos=infos, local=local
    )

    controller.submit(keys, target="local")

    assert local.calls == []
    assert len(infos) == 1
    title, message = infos[0]
    assert title == "Phonon stage must run where stage 1 ran"
    assert "siteA (stage 1 ran on archer2)" in message


def test_quantum_controller_refuses_cluster_stage_two_after_a_local_singlepoint(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    cluster = RecordingBackend()
    controller, keys = _stage_two_target_controller(
        ProjectService(), tmp_path, stage_one_targets=["local"], infos=infos, cluster=cluster
    )

    controller.submit(keys, target="cluster")

    assert cluster.calls == []
    assert len(infos) == 1
    assert "ran on this computer" in infos[0][1]
    assert "siteA" in infos[0][1]


def test_quantum_controller_refuses_stage_two_rows_from_different_clusters(tmp_path: Path) -> None:
    """One cluster submission cannot reach two checkpoints on two clusters."""

    infos: list[tuple[str, str]] = []
    cluster = RecordingBackend()
    controller, keys = _stage_two_target_controller(
        ProjectService(), tmp_path, stage_one_targets=["archer2", "young"], infos=infos, cluster=cluster
    )

    controller.submit(keys, target="cluster")

    assert cluster.calls == []
    assert infos[0][0] == "Submit each cluster's rows separately"
    assert "archer2: siteA" in infos[0][1]
    assert "young: siteB" in infos[0][1]


def test_quantum_controller_preselects_the_stage_one_cluster_for_stage_two(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    cluster = RecordingBackend()
    controller, keys = _stage_two_target_controller(
        ProjectService(), tmp_path, stage_one_targets=["archer2", "archer2"], infos=infos, cluster=cluster
    )

    controller.submit(keys, target="cluster")

    assert infos == []
    assert len(cluster.calls) == 1
    jobs, kwargs = cluster.calls[0]
    assert len(jobs) == 2
    assert kwargs["default_cluster_name"] == "archer2"
    # The preselection does not leak into the next submission.
    assert controller.submission_groups(jobs) == [(jobs, {})]


def test_quantum_controller_runs_local_stage_two_after_a_local_singlepoint(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    local = RecordingBackend()
    controller, keys = _stage_two_target_controller(
        ProjectService(), tmp_path, stage_one_targets=["local"], infos=infos, local=local
    )

    controller.submit(keys, target="local")

    assert infos == []
    assert len(local.calls) == 1
    assert local.calls[0][1] == {}


def _phonon_method_controller(
    service: ProjectService,
    tmp_path: Path,
    *,
    rows: dict[str, tuple[str, str, str]],
    infos: list[tuple[str, str]],
    questions: list[tuple[str, str]],
    answer: bool,
    local: RecordingBackend,
):
    """Build a controller whose rows ``siteA``/``siteB`` are ready to submit.

    ``rows`` maps a site label to ``(stage, GUI phonon method, pseudopotential
    family)``, stored as the row's configuration; materialize hands back one
    defined job per row so an allowed submission reaches the local backend.
    """

    from mufinder.app.workflow_config_codecs import store_workflow_row_configuration

    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    by_label = {candidate.site.label: key for key, candidate in candidates.items()}
    keys = []
    for label, (stage, method, family) in rows.items():
        key = by_label[label]
        keys.append(key)
        if stage == QUANTUM_STAGE_PHONON:
            _complete_stage_one(service, document, definition, key)
        store_workflow_row_configuration(
            definition,
            key,
            {"parameter_values": {"phonon_method": method}, "pseudopotential_family": family},
        )
    controller = _quantum_document_controller(
        service,
        document,
        output_root,
        deps=_stub_quantum_deps(confirm=lambda title, text: questions.append((title, text)) or answer),
        local_backend=local,
        infos=infos,
    )
    jobs = {key: _quantum_job(service, document, f"quantum_{key}", QUANTUM_SINGLEPOINT_CALC_TYPE) for key in keys}
    controller.materialize = lambda values: [jobs[str(value)] for value in values]  # type: ignore[assignment]
    return controller, keys


def test_quantum_controller_refuses_dfpt_with_ultrasoft_pseudopotentials_by_name(tmp_path: Path) -> None:
    """The refusal names the rows instead of ending in a generic "No ready jobs"."""

    infos: list[tuple[str, str]] = []
    questions: list[tuple[str, str]] = []
    local = RecordingBackend()
    controller, keys = _phonon_method_controller(
        ProjectService(),
        tmp_path,
        rows={
            "siteA": (QUANTUM_STAGE_SINGLEPOINT, "DFPT", "ultrasoft"),
            "siteB": (QUANTUM_STAGE_PHONON, "LINEARRESPONSE", "ultrasoft"),
        },
        infos=infos,
        questions=questions,
        answer=True,
        local=local,
    )

    controller.submit(keys, target="local")

    assert local.calls == []
    assert questions == []
    assert len(infos) == 1
    title, message = infos[0]
    assert title == "Phonon method needs norm-conserving pseudopotentials"
    assert "siteA (DFPT, stage 1)" in message
    assert "siteB (LINEARRESPONSE, stage 2)" in message
    assert "No ready jobs" not in title


def test_quantum_controller_asks_before_dfpt_with_an_unknown_pseudopotential_family(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    questions: list[tuple[str, str]] = []
    local = RecordingBackend()
    controller, keys = _phonon_method_controller(
        ProjectService(),
        tmp_path,
        rows={"siteA": (QUANTUM_STAGE_SINGLEPOINT, "DFPT", "unknown")},
        infos=infos,
        questions=questions,
        answer=False,
        local=local,
    )

    controller.submit(keys, target="local")

    assert local.calls == []
    assert infos == []
    assert len(questions) == 1
    title, message = questions[0]
    assert title == "Check pseudopotentials for DFPT"
    assert "siteA (DFPT, stage 1)" in message
    assert "norm-conserving" in message


def test_quantum_controller_submits_dfpt_with_an_unknown_family_once_confirmed(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    questions: list[tuple[str, str]] = []
    local = RecordingBackend()
    controller, keys = _phonon_method_controller(
        ProjectService(),
        tmp_path,
        rows={"siteA": (QUANTUM_STAGE_SINGLEPOINT, "DFPT", "unknown")},
        infos=infos,
        questions=questions,
        answer=True,
        local=local,
    )

    controller.submit(keys, target="local")

    assert len(questions) == 1
    assert len(local.calls) == 1


def test_quantum_controller_submits_finite_displacement_without_asking(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    questions: list[tuple[str, str]] = []
    local = RecordingBackend()
    controller, keys = _phonon_method_controller(
        ProjectService(),
        tmp_path,
        rows={
            "siteA": (QUANTUM_STAGE_SINGLEPOINT, "FINITEDISPLACEMENT", "ultrasoft"),
            "siteB": (QUANTUM_STAGE_SINGLEPOINT, "DFPT", "norm_conserving"),
        },
        infos=infos,
        questions=questions,
        answer=False,
        local=local,
    )

    controller.submit(keys, target="local")

    assert infos == []
    assert questions == []
    assert len(local.calls) == 1


def test_quantum_controller_validate_artifacts_checks_phonon_jobs_only(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    controller = QuantumCorrectionsController(
        make_context(document=document), _stub_quantum_deps(), _FakeQuantumView()
    )
    singlepoint = _quantum_job(service, document, "quantum_singlepoint_s", QUANTUM_SINGLEPOINT_CALC_TYPE)
    phonon = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=phonon.id)
    parsed = CastepResult(completed=True, total_energy=-1.0)

    assert controller.validate_artifacts(
        singlepoint, parsed, run, ParsedJobArtifacts(phonon_missing=True)
    ) is None
    assert controller.validate_artifacts(
        phonon, parsed, run, ParsedJobArtifacts(phonon_missing=True)
    ) == "Expected CASTEP phonon output was not found: quantum_phonon_s.phonon"
    assert controller.validate_artifacts(
        phonon, parsed, run, ParsedJobArtifacts(phonon_error="no Gamma point")
    ) == "Failed to parse CASTEP phonon output: no Gamma point"
    assert controller.validate_artifacts(phonon, parsed, run, no_artifacts()) is None
