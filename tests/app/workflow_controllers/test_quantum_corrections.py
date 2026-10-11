"""Tests for :class:`QuantumCorrectionsController`.

Covers the two-stage (singlepoint, then phonon) submission, ``build_result`` on
a real ``.phonon`` file, the stage-two species-mass and execution-target guards,
the phonon-method checks and ``validate_artifacts``.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from pathlib import Path
from uuid import UUID

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job
from mufinder.services.job_artifacts import ParsedJobArtifacts, collect_job_artifacts
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_SINGLEPOINT_CALC_TYPE
from mufinder.workflow.types import InputTemplate

from mufinder.app.workflow_controllers.quantum_corrections import (
    QUANTUM_STAGE_PHONON,
    QUANTUM_STAGE_SINGLEPOINT,
    QuantumCorrectionsController,
    QuantumCorrectionsDeps,
)

from tests.support.controllers import (
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


@_dataclass
class _FakeQuantumStageState:
    stage: str
    status: str
    current_job: Job | None = None


@_dataclass
class _FakeQuantumCandidate:
    key: str
    site: object
    run: object
    job: object
    structure: object


def _stub_quantum_deps(**overrides) -> QuantumCorrectionsDeps:
    base = dict(
        ensure_quantum_corrections_definition=lambda _keys: None,
        eligible_quantum_source_candidates=lambda: [],
        quantum_source_relative_energies=lambda: {},
        quantum_source_candidate_by_key=lambda _key: None,
        quantum_stage_state=lambda _definition, _candidate: _FakeQuantumStageState(
            stage=QUANTUM_STAGE_SINGLEPOINT, status="ready"
        ),
        build_quantum_corrections_states=lambda: {},
        jobs_for_quantum_source_keys=lambda _keys: [],
        singlepoint_job_for_quantum_source_key=lambda _definition, _key: None,
        phonon_job_for_quantum_source_key=lambda _definition, _key: None,
        quantum_seed_name_for_key=lambda _definition, key: f"quantum_{key}",
        quantum_runtime_configuration=lambda _definition, _key: ({}, "unknown"),
        workflow_row_configuration=lambda _definition, _key: ({}, (), {}),
        workflow_row_configuration_exists=lambda _definition, _key: False,
        quantum_num_farms_from_extras=lambda _extras: 1,
        quantum_source_label=lambda candidate: candidate.site.label,
        quantum_method_from_gui_value=lambda _value: "finite_displacement",
        charge_state_for_job=lambda _job: None,
        charge_state_label_for_job=lambda _job: "",
        quantum_can_reset_to_stage_one=lambda _definition, _candidate: False,
        input_template_from_keyword_values=lambda _values: InputTemplate(),
        write_quantum_job_directory=lambda _spec, _root, *, seed_name: Path("/tmp") / seed_name,
        dedupe_jobs=lambda jobs: jobs,
        check_definition_host=lambda _definition: True,
        format_energy=_format_energy_stub,
        confirm=lambda _title, _text: True,
    )
    base.update(overrides)
    return QuantumCorrectionsDeps(**base)


def _format_energy_stub(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g} eV"
    except (TypeError, ValueError):
        return ""


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
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    materialize_calls: list[list] = []
    deps = _stub_quantum_deps(
        check_definition_host=lambda _definition: False,
    )
    local = RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = make_context(
        document=document, project_service=service, local_backend=local, infos=infos
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())
    # Prove materialize is never reached by making it record + explode if called.
    controller.materialize = lambda keys: materialize_calls.append(list(keys)) or []  # type: ignore[assignment]

    controller.submit(["site-key"], target="local")

    assert local.calls == []  # nothing dispatched
    assert infos == []  # the refusal is silent (no "no ready jobs" dialog)
    assert materialize_calls == []  # no submission plan was even built


def test_quantum_controller_pre_submit_check_passes_when_host_matches(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    deps = _stub_quantum_deps(check_definition_host=lambda _definition: True)
    controller = QuantumCorrectionsController(
        make_context(document=document, project_service=service), deps, _FakeQuantumView()
    )
    assert controller.pre_submit_check(["site-key"]) is True


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
    """Seed a document with two eligible relaxed sources sharing one definition."""

    from mufinder.domain.models import MuonSite, Structure

    document = service.create_project(tmp_path / "q2.mufinder", "Q2")
    output_root = document.project_root / "files" / "quantum_corrections"

    candidates = {}
    site_ids = []
    structure_ids = []
    for label in ("siteA", "siteB"):
        relaxed = Structure(
            label=f"{label} relaxed",
            origin="muon_relaxation",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si", "H:mu"],
            fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
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
        document.runs[run.id] = run
        site = MuonSite(label=label, status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
        document.sites[site.id] = site
        site_ids.append(site.id)
        structure_ids.append(relaxed.id)
        candidates[str(site.id)] = _FakeQuantumCandidate(
            key=str(site.id), site=site, run=run, job=source_job, structure=relaxed
        )

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
    return document, definition, candidates, output_root


def test_quantum_controller_materialize_two_stage_mixed_keys(tmp_path: Path) -> None:
    """Mixed selection where one source is at the singlepoint stage and another at
    the phonon stage: materialize writes each key into the correct per-stage
    mapping with the correct calc_type, in one pass."""

    from mufinder.app.relaxation_types import SupercellMatrix  # noqa: F401  (import parity)

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(
        service, tmp_path
    )
    keys = list(candidates.keys())
    singlepoint_key, phonon_key = keys[0], keys[1]

    # siteA is fresh (singlepoint stage); siteB already has a COMPLETED singlepoint
    # job, so its stage has advanced to phonon.
    completed_singlepoint = _quantum_job(
        service, document, "quantum_singlepoint_siteB", QUANTUM_SINGLEPOINT_CALC_TYPE
    )
    completed_singlepoint.status = JobStatus.COMPLETED
    definition.parameters["singlepoint_job_ids_by_label"][phonon_key] = str(completed_singlepoint.id)

    stage_by_key = {
        singlepoint_key: _FakeQuantumStageState(stage=QUANTUM_STAGE_SINGLEPOINT, status="ready"),
        phonon_key: _FakeQuantumStageState(stage=QUANTUM_STAGE_PHONON, status="ready"),
    }

    def write_dir(spec, root, *, seed_name):
        job_dir = root / seed_name
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def singlepoint_job_for_key(_definition, key):
        job_id = _definition.parameters.get("singlepoint_job_ids_by_label", {}).get(key)
        return document.jobs.get(UUID(job_id)) if isinstance(job_id, str) else None

    def phonon_job_for_key(_definition, key):
        job_id = _definition.parameters.get("phonon_job_ids_by_label", {}).get(key)
        return document.jobs.get(UUID(job_id)) if isinstance(job_id, str) else None

    deps = _stub_quantum_deps(
        ensure_quantum_corrections_definition=lambda _keys: definition,
        quantum_source_candidate_by_key=lambda key: candidates.get(key),
        quantum_stage_state=lambda _definition, candidate: stage_by_key[candidate.key],
        singlepoint_job_for_quantum_source_key=singlepoint_job_for_key,
        phonon_job_for_quantum_source_key=phonon_job_for_key,
        write_quantum_job_directory=write_dir,
    )
    context = make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _definition: output_root,
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())

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
    # Both seed names were recorded.
    seed_mapping = definition.parameters["job_seed_names_by_label"]
    assert singlepoint_key in seed_mapping and phonon_key in seed_mapping
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


def _quantum_stage_two_guard_setup(service: ProjectService, tmp_path: Path):
    """Return (deps-kwargs, context-kwargs, phonon_key, output_root) for the guard."""

    document, definition, candidates, output_root = _seed_quantum_two_stage_document(
        service, tmp_path
    )
    phonon_key = list(candidates)[0]
    deps_kwargs = dict(
        quantum_source_candidate_by_key=lambda key: candidates.get(key),
        quantum_stage_state=lambda _definition, _candidate: _FakeQuantumStageState(
            stage=QUANTUM_STAGE_PHONON, status="ready"
        ),
    )
    context_kwargs = dict(
        document=document,
        project_service=service,
        definition_output_directory=lambda _definition: output_root,
    )
    return deps_kwargs, context_kwargs, phonon_key, output_root


def test_quantum_controller_refuses_stage_two_when_stage_one_omits_species_mass(
    tmp_path: Path,
) -> None:
    """A phonon continuation whose stage-1 cell has no SPECIES_MASS is refused.

    CASTEP restores species masses from the stage-1 checkpoint, so continuing from
    a singlepoint that ran without the muon mass would silently produce phonons at
    the default hydrogen mass. The refusal explains itself rather than being silent.
    """

    service = ProjectService()
    deps_kwargs, context_kwargs, phonon_key, output_root = _quantum_stage_two_guard_setup(
        service, tmp_path
    )
    _write_stage_one_cell(output_root, f"quantum_{phonon_key}", with_species_mass=False)
    infos: list[tuple[str, str]] = []
    controller = QuantumCorrectionsController(
        make_context(infos=infos, **context_kwargs),
        _stub_quantum_deps(**deps_kwargs),
        _FakeQuantumView(),
    )

    assert controller.pre_submit_check([phonon_key]) is False
    assert len(infos) == 1
    title, message = infos[0]
    assert title == "Stage-1 muon mass missing"
    assert "SPECIES_MASS" in message
    assert "siteA" in message


def test_quantum_controller_allows_stage_two_when_stage_one_declares_species_mass(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    deps_kwargs, context_kwargs, phonon_key, output_root = _quantum_stage_two_guard_setup(
        service, tmp_path
    )
    _write_stage_one_cell(output_root, f"quantum_{phonon_key}", with_species_mass=True)
    infos: list[tuple[str, str]] = []
    controller = QuantumCorrectionsController(
        make_context(infos=infos, **context_kwargs),
        _stub_quantum_deps(**deps_kwargs),
        _FakeQuantumView(),
    )

    assert controller.pre_submit_check([phonon_key]) is True
    assert infos == []


def test_quantum_controller_stage_two_guard_fails_open_without_a_stage_one_cell(
    tmp_path: Path,
) -> None:
    """No readable stage-1 cell is no evidence: the guard must not block."""

    service = ProjectService()
    deps_kwargs, context_kwargs, phonon_key, output_root = _quantum_stage_two_guard_setup(
        service, tmp_path
    )
    unreadable = output_root / f"quantum_{phonon_key}" / f"quantum_{phonon_key}.cell"
    unreadable.parent.mkdir(parents=True, exist_ok=True)
    unreadable.write_text("this is not a cell file", encoding="utf-8")
    infos: list[tuple[str, str]] = []
    controller = QuantumCorrectionsController(
        make_context(infos=infos, **context_kwargs),
        _stub_quantum_deps(**deps_kwargs),
        _FakeQuantumView(),
    )

    # Unparseable cell, and (below) no cell at all: both allow submission.
    assert controller.pre_submit_check([phonon_key]) is True
    unreadable.unlink()
    assert controller.pre_submit_check([phonon_key]) is True
    assert infos == []


def test_quantum_controller_stage_two_guard_ignores_stage_one_rows(tmp_path: Path) -> None:
    """A row still on stage 1 is never blocked: it has no checkpoint to inherit."""

    service = ProjectService()
    deps_kwargs, context_kwargs, phonon_key, output_root = _quantum_stage_two_guard_setup(
        service, tmp_path
    )
    _write_stage_one_cell(output_root, f"quantum_{phonon_key}", with_species_mass=False)
    deps_kwargs["quantum_stage_state"] = lambda _definition, _candidate: _FakeQuantumStageState(
        stage=QUANTUM_STAGE_SINGLEPOINT, status="ready"
    )
    infos: list[tuple[str, str]] = []
    controller = QuantumCorrectionsController(
        make_context(infos=infos, **context_kwargs),
        _stub_quantum_deps(**deps_kwargs),
        _FakeQuantumView(),
    )

    assert controller.pre_submit_check([phonon_key]) is True
    assert infos == []


def test_quantum_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    """End-to-end submit wiring: build_states → is_ready (base default) →
    jobs_for_keys → materialize → local backend, dispatched as a single group with
    no extra kwargs (task farming lives in the shared backend, not the controller)."""

    from mufinder.app.relaxation_types import RelaxationRunState

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(
        service, tmp_path
    )
    key = next(iter(candidates))

    stage_state = _FakeQuantumStageState(stage=QUANTUM_STAGE_SINGLEPOINT, status="ready")

    def write_dir(spec, root, *, seed_name):
        job_dir = root / seed_name
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    deps = _stub_quantum_deps(
        ensure_quantum_corrections_definition=lambda _keys: definition,
        eligible_quantum_source_candidates=lambda: list(candidates.values()),
        quantum_source_candidate_by_key=lambda k: candidates.get(k),
        quantum_stage_state=lambda _definition, _candidate: stage_state,
        build_quantum_corrections_states=lambda: {key: RelaxationRunState(status="ready")},
        jobs_for_quantum_source_keys=lambda _keys: [],
        singlepoint_job_for_quantum_source_key=lambda _definition, _key: None,
        phonon_job_for_quantum_source_key=lambda _definition, _key: None,
        write_quantum_job_directory=write_dir,
    )
    local = RecordingBackend()
    cluster = RecordingBackend()
    context = make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _definition: output_root,
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())

    controller.submit([key], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted, extras = local.calls[0]
    assert extras == {}
    assert len(submitted) == 1
    assert submitted[0].calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE


def test_quantum_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    from mufinder.app.relaxation_types import RelaxationRunState

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(
        service, tmp_path
    )
    key = next(iter(candidates))
    stage_state = _FakeQuantumStageState(stage=QUANTUM_STAGE_SINGLEPOINT, status="ready")

    def write_dir(spec, root, *, seed_name):
        job_dir = root / seed_name
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    deps = _stub_quantum_deps(
        ensure_quantum_corrections_definition=lambda _keys: definition,
        eligible_quantum_source_candidates=lambda: list(candidates.values()),
        quantum_source_candidate_by_key=lambda k: candidates.get(k),
        quantum_stage_state=lambda _definition, _candidate: stage_state,
        build_quantum_corrections_states=lambda: {key: RelaxationRunState(status="ready")},
        jobs_for_quantum_source_keys=lambda _keys: [],
        singlepoint_job_for_quantum_source_key=lambda _definition, _key: None,
        phonon_job_for_quantum_source_key=lambda _definition, _key: None,
        write_quantum_job_directory=write_dir,
    )
    local = RecordingBackend()
    cluster = RecordingBackend()
    context = make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _definition: output_root,
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())

    controller.submit([key], target="cluster")

    assert local.calls == []
    assert len(cluster.calls) == 1
    assert cluster.calls[0][0][0].calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE


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
    singlepoints: dict[str, Job] = {}
    for key, stage_one in zip(keys, stage_one_targets, strict=True):
        singlepoint = _quantum_job(service, document, f"quantum_singlepoint_{key}", QUANTUM_SINGLEPOINT_CALC_TYPE)
        singlepoint.status = JobStatus.COMPLETED
        singlepoint.execution_target = stage_one
        singlepoints[key] = singlepoint
    ready = _FakeQuantumStageState(stage=QUANTUM_STAGE_PHONON, status="ready")
    deps = _stub_quantum_deps(
        quantum_source_candidate_by_key=lambda key: candidates.get(key),
        quantum_stage_state=lambda _definition, _candidate: ready,
        build_quantum_corrections_states=lambda: {key: ready for key in keys},
        singlepoint_job_for_quantum_source_key=lambda _definition, key: singlepoints.get(key),
    )
    context = make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _definition: output_root,
        local_backend=local,
        cluster_backend=cluster,
        infos=infos,
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())
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


_GUI_TO_METHOD = {
    "FINITEDISPLACEMENT": "finite_displacement",
    "DFPT": "dfpt",
    "LINEARRESPONSE": "linear_response",
}


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
    family)``; materialize hands back one defined job per row so an allowed
    submission reaches the local backend.
    """

    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    by_label = {candidate.site.label: key for key, candidate in candidates.items()}
    config = {by_label[label]: spec for label, spec in rows.items()}
    keys = list(config)
    deps = _stub_quantum_deps(
        quantum_source_candidate_by_key=lambda key: candidates.get(key),
        quantum_stage_state=lambda _definition, candidate: _FakeQuantumStageState(
            stage=config[candidate.key][0], status="ready"
        ),
        build_quantum_corrections_states=lambda: {
            key: _FakeQuantumStageState(stage=config[key][0], status="ready") for key in keys
        },
        quantum_runtime_configuration=lambda _definition, key: (
            {"phonon_method": config[key][1]},
            config[key][2],
        ),
        quantum_method_from_gui_value=lambda value: _GUI_TO_METHOD[str(value).upper()],
        confirm=lambda title, text: questions.append((title, text)) or answer,
    )
    context = make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _definition: output_root,
        local_backend=local,
        infos=infos,
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())
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
