"""Tests for :class:`QuantumCorrectionsController`.

Covers the one-phonon-job-per-row submission and row lifecycle, ``build_result``
on a real ``.phonon`` file, the Configure dialog's editable and read-only
states, the phonon-method checks, ``validate_artifacts`` and how rows of a
project that ran the legacy singlepoint job first are read.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, Result, Structure
from mufinder.science.quantum_phonons import mode_width_angstrom
from mufinder.services.job_artifacts import ParsedJobArtifacts, collect_job_artifacts
from mufinder.services.job_paths import job_output_directory
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.source_candidates import quantum_seed_name_for_key
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_SINGLEPOINT_CALC_TYPE

from mufinder.app.workflow_config_codecs import store_workflow_row_configuration
from mufinder.app.workflow_controllers.quantum_corrections import (
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
# Each row runs one CASTEP phonon job. ``build_result`` carries the richest
# result logic — parsing a real ``.phonon`` file into the muon ZPE, eigenvalues,
# mode participations, and the ``w1`` entanglement witness — while a legacy
# quantum-*singlepoint* job an older project still holds contributes an empty
# result. Driven below with narrow fakes (no MainWindow) and a REAL ``.phonon``
# fixture containing an ``H:mu`` muon.


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
        self.site_scopes: list[dict] = []

    def set_site_scope(self, **kwargs) -> None:
        self.site_scopes.append(kwargs)

    def set_sources(self, rows) -> None:
        self.sources.append(list(rows))

    def set_results(self, rows) -> None:
        self.results.append(list(rows))


def _stub_quantum_deps(**overrides) -> QuantumCorrectionsDeps:
    base = dict(
        handle_structure_host_mismatch=lambda _mismatch: True,
        handle_site_host_mismatch=lambda _mismatch: True,
        confirm=lambda _title, _text: True,
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


def test_quantum_controller_handles_phonon_and_legacy_singlepoint_calc_types(tmp_path: Path) -> None:
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
    # The legacy singlepoint type stays owned so an older project's job still collects.
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


def _displayed_witness(tmp_path: Path, data: dict) -> str:
    """Return the witness text ``refresh_view`` shows for one stored phonon payload."""

    return _displayed_result_row(tmp_path, data).witness


def _displayed_result_row(tmp_path: Path, data: dict):
    """Return the result row ``refresh_view`` shows for one stored phonon payload."""

    from mufinder.domain.models import Result

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    document.runs[run.id] = run
    result = Result(result_type=QUANTUM_PHONON_CALC_TYPE, calculation_run_id=run.id, data=data)
    document.results[result.id] = result
    view = _FakeQuantumView()
    controller = QuantumCorrectionsController(
        make_context(document=document, project_service=service), _stub_quantum_deps(), view
    )
    controller.refresh_view()
    return view.results[-1][0]


_STORED_MODES = [
    {"mode_index": 4, "frequency_cm": 1600.0, "participation": 0.9, "zpe_eV": 0.099},
    {"mode_index": 5, "frequency_cm": 2500.0, "participation": 0.8, "zpe_eV": 0.155},
    {"mode_index": 6, "frequency_cm": 3600.0, "participation": 0.7, "zpe_eV": 0.223},
]


def test_quantum_controller_refresh_view_shows_a_current_witness_as_stored(tmp_path: Path) -> None:
    data = {
        "muon_zpe_modes": _STORED_MODES,
        "w1_entanglement_witness": -0.25,
        "w1_definition": "participation_minus_3",
    }

    assert _displayed_witness(tmp_path, data) == "-0.25"


def test_quantum_controller_refresh_view_recomputes_a_witness_stored_before_its_definition(
    tmp_path: Path,
) -> None:
    """A payload without ``w1_definition`` holds the earlier ``-sqrt(3 - P)``
    form; the row shows ``P - 3`` recomputed from the stored participations."""

    old_form = -math.sqrt(3.0 - 2.4)
    data = {"muon_zpe_modes": _STORED_MODES, "w1_entanglement_witness": old_form}

    assert _displayed_witness(tmp_path, data) == f"{2.4 - 3.0:.4g}"


def test_quantum_controller_refresh_view_shows_an_old_witness_without_modes_as_stored(
    tmp_path: Path,
) -> None:
    assert _displayed_witness(tmp_path, {"w1_entanglement_witness": -0.5}) == "-0.5"
    assert _displayed_witness(tmp_path, {"muon_zpe_total_eV": 0.1}) == ""


def test_quantum_controller_refresh_view_shows_muon_modes_and_stored_widths(
    tmp_path: Path,
) -> None:
    """The row lists the three muon modes and their stored ``width_A`` in
    frequency order, and the ZPE as a bare number (the header carries eV)."""

    data = {
        "muon_zpe_total_eV": 0.5786,
        "muon_zpe_modes": [
            {"mode_index": 9, "frequency_cm": 3226.0, "participation": 0.9, "width_A": 0.2149},
            {"mode_index": 7, "frequency_cm": 2742.0, "participation": 0.95, "width_A": 0.2331},
            {"mode_index": 8, "frequency_cm": 3226.0, "participation": 0.9, "width_A": 0.2149},
        ],
        "w1_entanglement_witness": -0.14,
        "w1_definition": "participation_minus_3",
        "phonon_mode_eigenvalues_cm": [100.0, 400.0, 900.0, 2742.0, 3226.0, 3226.0],
    }

    row = _displayed_result_row(tmp_path, data)

    assert row.zpe == "0.579"
    assert row.muon_modes == "2742.0, 3226.0, 3226.0"
    assert row.widths == "0.23, 0.21, 0.21"
    assert row.witness == "-0.14"


def test_quantum_controller_refresh_view_computes_widths_for_a_payload_without_width(
    tmp_path: Path,
) -> None:
    """A payload stored without ``width_A`` shows widths computed from each
    mode's frequency with the muon mass."""

    row = _displayed_result_row(tmp_path, {"muon_zpe_modes": _STORED_MODES})

    assert row.muon_modes == "1600.0, 2500.0, 3600.0"
    assert row.widths == ", ".join(
        f"{mode_width_angstrom(mode['frequency_cm']):.2f}" for mode in _STORED_MODES
    )


def test_quantum_controller_refresh_view_leaves_mode_cells_blank_without_modes(
    tmp_path: Path,
) -> None:
    row = _displayed_result_row(tmp_path, {"muon_zpe_total_eV": "not a number"})

    assert (row.zpe, row.muon_modes, row.widths) == ("", "", "")


def test_quantum_controller_pre_submit_check_blocks_submission(tmp_path: Path) -> None:
    """An unresolved host mismatch silently refuses the submission: the host guard
    returns False, so no submission plan is built and nothing is dispatched."""

    service = ProjectService()
    document, _definition, candidates, _output_root = _seed_quantum_document(service, tmp_path)
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
    document, _definition, candidates, _output_root = _seed_quantum_document(service, tmp_path)
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
    """A legacy quantum-singlepoint job contributes an empty ControllerResult (base
    payload only), yet is still owned by the controller so it collects cleanly."""

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
    # Each selected mode carries its muon zero-point width, from its frequency.
    for mode in payload["muon_zpe_modes"]:
        assert mode["width_A"] == pytest.approx(mode_width_angstrom(mode["frequency_cm"]))
        assert mode["width_A"] > 0.0
    # The witness is the paper's ``P - 3`` and the payload names that definition.
    assert payload["w1_definition"] == "participation_minus_3"
    assert payload["w1_entanglement_witness"] == pytest.approx(
        sum(mode["participation"] for mode in payload["muon_zpe_modes"]) - 3.0
    )
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


def _seed_quantum_document(service: ProjectService, tmp_path: Path):
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
            "phonon_job_ids_by_label": {},
            "job_seed_names_by_label": {},
        },
        output_directory=output_root,
    )
    candidates = {candidate.key: candidate for candidate in eligible_quantum_source_candidates(document)}
    assert [candidate.site.label for candidate in candidates.values()] == ["siteA", "siteB"]
    return document, definition, candidates, output_root


def _quantum_document_controller(service: ProjectService, document, output_root, *, deps=None, **context_kwargs):
    context = make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _definition: output_root,
        **context_kwargs,
    )
    return QuantumCorrectionsController(context, deps or _stub_quantum_deps(), _FakeQuantumView())


def _bind_phonon(
    service: ProjectService,
    document: ProjectDocument,
    definition,
    key: str,
    *,
    status: JobStatus = JobStatus.DEFINED,
) -> Job:
    """Bind a phonon job in ``status`` to the row ``key``."""

    phonon = _quantum_job(service, document, f"quantum_phonon_{key}", QUANTUM_PHONON_CALC_TYPE)
    phonon.status = status
    definition.parameters["phonon_job_ids_by_label"][key] = str(phonon.id)
    return phonon


def _bind_legacy_singlepoint(
    service: ProjectService,
    document: ProjectDocument,
    definition,
    key: str,
    *,
    status: JobStatus = JobStatus.COMPLETED,
) -> Job:
    """Bind the singlepoint an older MuFinder version ran first (``singlepoint_job_ids_by_label``)."""

    singlepoint = _quantum_job(service, document, f"quantum_singlepoint_{key}", QUANTUM_SINGLEPOINT_CALC_TYPE)
    singlepoint.status = status
    definition.parameters.setdefault("singlepoint_job_ids_by_label", {})[key] = str(singlepoint.id)
    return singlepoint


def _write_legacy_seed_directory(output_root: Path, seed_name: str) -> Path:
    """Leave the files a legacy singlepoint wrote in the row's seed directory."""

    job_dir = output_root / seed_name
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{seed_name}.param").write_text("TASK : SINGLEPOINT\n", encoding="utf-8")
    (job_dir / f"{seed_name}.check").write_bytes(b"legacy checkpoint")
    return job_dir


def test_quantum_controller_materialize_writes_one_phonon_job_per_row(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    store_workflow_row_configuration(definition, second, {"num_farms": 3})
    controller = _quantum_document_controller(service, document, output_root)

    jobs = controller.materialize([first, second])

    assert [job.calc_type for job in jobs] == [QUANTUM_PHONON_CALC_TYPE, QUANTUM_PHONON_CALC_TYPE]
    assert [job.label for job in jobs] == ["quantum_phonon_siteA", "quantum_phonon_siteB"]
    assert all(job.status == JobStatus.DEFINED for job in jobs)
    phonon_mapping = definition.parameters["phonon_job_ids_by_label"]
    assert phonon_mapping == {first: str(jobs[0].id), second: str(jobs[1].id)}
    # The phonon mapping stays aliased into job_ids_by_label (load-bearing fallback).
    assert definition.parameters["job_ids_by_label"] is phonon_mapping
    assert "singlepoint_job_ids_by_label" not in definition.parameters
    assert definition.parameters["job_seed_names_by_label"] == {first: "quantum_siteA", second: "quantum_siteB"}

    param_text = (output_root / "quantum_siteA" / "quantum_siteA.param").read_text(encoding="utf-8")
    cell_text = (output_root / "quantum_siteA" / "quantum_siteA.cell").read_text(encoding="utf-8")
    assert "TASK : PHONON" in param_text
    assert "PHONON_METHOD : FINITEDISPLACEMENT" in param_text
    assert "CONTINUATION : DEFAULT" in param_text
    assert "CHARGE : 1" in param_text
    assert "NUM_FARMS" not in param_text
    assert "SINGLEPOINT" not in param_text
    assert "%BLOCK SPECIES_MASS" in cell_text
    assert "%BLOCK SPECIES_GAMMA" in cell_text
    assert "PHONON_KPOINT_MP_GRID 1 1 1" in cell_text
    farmed = (output_root / "quantum_siteB" / "quantum_siteB.param").read_text(encoding="utf-8")
    assert "NUM_FARMS : 3" in farmed


def test_quantum_controller_materialize_runs_one_site_in_both_charge_states(tmp_path: Path) -> None:
    from mufinder.services.source_candidates import eligible_quantum_source_candidates

    from tests.support.muon_sites import (
        MUONIUM_RELAXATION_CALC_TYPE,
        add_relaxed_muon_site,
        bind_muon_relaxation_row,
        relaxation_job_id,
    )

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    site_a = next(candidate.site for candidate in candidates.values() if candidate.site.label == "siteA")
    scratch = add_relaxed_muon_site(
        document, label="siteA", charge_calc_type=MUONIUM_RELAXATION_CALC_TYPE, total_energy=-2.0
    )
    muonium_job = document.jobs[relaxation_job_id(document, scratch)]
    del document.sites[scratch.id]
    bind_muon_relaxation_row(document, site_a, muonium_job, "muonium")
    keys = [str(site_a.id), f"{site_a.id}:muonium"]
    assert set(keys) <= {candidate.key for candidate in eligible_quantum_source_candidates(document)}
    controller = _quantum_document_controller(service, document, output_root)

    jobs = controller.materialize(keys)

    assert [job.label for job in jobs] == ["quantum_phonon_siteA", "quantum_phonon_siteA_muonium"]
    assert definition.parameters["phonon_job_ids_by_label"] == {
        keys[0]: str(jobs[0].id),
        keys[1]: str(jobs[1].id),
    }
    assert definition.parameters["job_seed_names_by_label"] == {
        keys[0]: "quantum_siteA",
        keys[1]: "quantum_siteA_muonium",
    }
    muonium_param = (output_root / "quantum_siteA_muonium" / "quantum_siteA_muonium.param").read_text(
        encoding="utf-8"
    )
    assert "CHARGE : 0" in muonium_param
    stored = definition.parameters["configuration"]["sources"]
    assert [(entry["key"], entry["charge_state"]) for entry in stored if entry["site_id"] == str(site_a.id)] == [
        (keys[0], "positive_muon"),
        (keys[1], "muonium"),
    ]


def test_quantum_controller_keeps_an_unselected_charge_state_out_of_the_sources(tmp_path: Path) -> None:
    from tests.support.muon_sites import (
        MUONIUM_RELAXATION_CALC_TYPE,
        add_relaxed_muon_site,
        bind_muon_relaxation_row,
        relaxation_job_id,
    )

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    site_a = next(candidate.site for candidate in candidates.values() if candidate.site.label == "siteA")
    scratch = add_relaxed_muon_site(
        document, label="siteA", charge_calc_type=MUONIUM_RELAXATION_CALC_TYPE, total_energy=-2.0
    )
    muonium_job = document.jobs[relaxation_job_id(document, scratch)]
    del document.sites[scratch.id]
    bind_muon_relaxation_row(document, site_a, muonium_job, "muonium")
    # The definition so far holds only siteB; select siteA's muonium relaxation alone.
    site_b = next(candidate.site for candidate in candidates.values() if candidate.site.label == "siteB")
    definition.input_site_ids = [site_b.id]
    definition.parameters["configuration"] = {"sources": [{"site_id": str(site_b.id)}]}
    controller = _quantum_document_controller(service, document, output_root)
    muonium_key = f"{site_a.id}:muonium"

    controller.ensure_definition([muonium_key])
    controller.ensure_definition([muonium_key])

    stored = definition.parameters["configuration"]["sources"]
    assert [entry["key"] for entry in stored] == [str(site_b.id), muonium_key]


def test_quantum_controller_materialize_reuses_a_rows_job(tmp_path: Path) -> None:
    """A defined job is rewritten from the row's configuration; a finished one is left alone."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    defined = _bind_phonon(service, document, definition, first)
    failed = _bind_phonon(service, document, definition, second, status=JobStatus.FAILED)
    store_workflow_row_configuration(definition, first, {"num_farms": 2})
    controller = _quantum_document_controller(service, document, output_root)

    jobs = controller.materialize([first, second])

    # Only the ready row is materialized; the failed row is not ready.
    assert jobs == [defined]
    assert len(document.jobs) == 2 + 2  # two source relaxations, two phonon jobs
    assert "NUM_FARMS : 2" in (output_root / "quantum_siteA" / "quantum_siteA.param").read_text(encoding="utf-8")
    assert not (output_root / "quantum_siteB").exists()
    assert failed.status == JobStatus.FAILED


def test_quantum_controller_row_lifecycle_from_ready_to_a_completed_result(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    local = RecordingBackend()
    view = _FakeQuantumView()
    context = make_context(
        document=document,
        project_service=service,
        local_backend=local,
        definition_output_directory=lambda _definition: output_root,
    )
    controller = QuantumCorrectionsController(context, _stub_quantum_deps(), view)

    assert controller.build_states()[key].status == "ready"
    assert controller.jobs_for_keys([key]) == []

    controller.submit([key], target="local")

    ((submitted, extras),) = local.calls
    assert extras == {}
    (job,) = submitted
    assert job.calc_type == QUANTUM_PHONON_CALC_TYPE
    assert controller.build_states()[key].status == "defined"
    assert controller.jobs_for_keys([key]) == [job]

    job.status = JobStatus.RUNNING
    assert controller.build_states()[key].status == "running"
    # Submitting again leaves a running row alone: nothing new is created.
    controller.submit([key], target="local")
    assert controller.jobs_for_keys([key]) == [job]
    assert sum(1 for other in document.jobs.values() if other.calc_type == QUANTUM_PHONON_CALC_TYPE) == 1

    job.status = JobStatus.COMPLETED
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    document.runs[run.id] = run
    result = Result(result_type=QUANTUM_PHONON_CALC_TYPE, calculation_run_id=run.id, data={"muon_zpe_total_eV": 0.2})
    document.results[result.id] = result
    controller.refresh_view()

    source_row = next(row for row in view.sources[-1] if row.key == key)
    assert source_row.status == "completed"
    assert [row.label for row in view.results[-1]] == ["quantum_phonon_siteA"]


def test_quantum_controller_refresh_view_always_shows_the_farm_count(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    store_workflow_row_configuration(definition, second, {"num_farms": 4})
    view = _FakeQuantumView()
    controller = QuantumCorrectionsController(
        make_context(document=document, project_service=service), _stub_quantum_deps(), view
    )
    controller.set_show_every_site(True)

    rows = {row.key: row for row in view.sources[-1]}
    assert (rows[first].num_farms_display, rows[first].status) == ("1", "ready")
    assert (rows[second].num_farms_display, rows[second].configured) == ("4", True)


def test_quantum_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    local = RecordingBackend()
    cluster = RecordingBackend()
    controller = _quantum_document_controller(
        service, document, output_root, local_backend=local, cluster_backend=cluster
    )

    controller.submit([key], target="cluster")

    assert local.calls == []
    ((submitted, extras),) = cluster.calls
    # No preselected cluster: a phonon job can run on any target.
    assert extras == {}
    assert [job.calc_type for job in submitted] == [QUANTUM_PHONON_CALC_TYPE]


def test_quantum_controller_dismissed_failed_job_resubmits_without_a_new_job(tmp_path: Path) -> None:
    """A dismissed failed row reads ready; the shared submit cycle resets and rewrites its job."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    phonon = _bind_phonon(service, document, definition, key, status=JobStatus.FAILED)
    phonon.dismissed_at = datetime.now(timezone.utc)
    reset: list[list[Job]] = []

    def _reset(jobs) -> bool:
        reset.append(list(jobs))
        for job in jobs:
            service.reset_failed_job(job)
        return True

    local = RecordingBackend()
    controller = _quantum_document_controller(
        service, document, output_root, local_backend=local, prepare_jobs_for_resubmission=_reset
    )

    assert controller.build_states()[key].status == "ready"
    controller.submit([key], target="local")

    assert reset == [[phonon]]
    assert local.calls[0][0] == [phonon]
    assert phonon.status == JobStatus.DEFINED
    assert (output_root / "quantum_siteA" / "quantum_siteA.param").is_file()


def test_quantum_controller_configure_request_opens_editable_ready_rows(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    controller = _quantum_document_controller(service, document, output_root)

    request = controller.configure_request([*candidates, "not-a-source"])

    assert request is not None
    assert request.keys == tuple(candidates)
    assert request.title == "Configure Quantum Corrections Parameters"
    assert request.read_only is False
    assert request.locked_reason == ""
    assert request.pseudopotential_family == "unknown"
    assert request.num_farms == 1
    assert request.species_options == ("Si", "H:mu")
    assert "Selected sources: 2" in request.intro_text
    assert "Displacements required for the phonon run: 6" in request.intro_text
    assert "stage" not in request.intro_text.lower()
    # Every row inherits the tight default and the legacy method.
    assert all(values["phonon_method"] == "FINITEDISPLACEMENT" for values in request.row_values)
    assert controller.configure_request(["not-a-source"]) is None
    assert QuantumCorrectionsController(
        make_context(document=None), _stub_quantum_deps(), _FakeQuantumView()
    ).configure_request(list(candidates)) is None


@pytest.mark.parametrize("dismissed", [False, True])
def test_quantum_controller_configure_request_edits_a_defined_or_dismissed_job(tmp_path: Path, dismissed: bool) -> None:
    """A job still defined (or dismissed, so its row reads ready) is rewritten on submit."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    phonon = _bind_phonon(service, document, definition, key)
    if dismissed:
        phonon.status = JobStatus.FAILED
        phonon.dismissed_at = datetime.now(timezone.utc)
    controller = _quantum_document_controller(service, document, output_root)

    request = controller.configure_request([key])

    assert request is not None
    assert request.read_only is False
    assert request.title == "Configure Quantum Corrections Parameters"
    assert request.locked_reason == ""


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (JobStatus.RUNNING, "phonon job is running; cancel it, then use Reset Failed Jobs..."),
        (JobStatus.QUEUED, "phonon job is queued; cancel it"),
        (JobStatus.FAILED, "phonon job is failed; use Reset Failed Jobs... to edit"),
        (JobStatus.CANCELLED, "phonon job is cancelled; use Reset Failed Jobs..."),
        (JobStatus.COMPLETED, "phonon job has completed; delete it from the Job Dashboard"),
    ],
)
def test_quantum_controller_read_only_configure_gives_one_reason(
    tmp_path: Path, status: JobStatus, expected: str
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    _bind_phonon(service, document, definition, first, status=status)
    controller = _quantum_document_controller(service, document, output_root)

    # One locked row makes the whole selection read-only.
    request = controller.configure_request([second, first])

    assert request is not None
    assert request.read_only is True
    assert request.title == "Inspect Quantum Corrections Parameters"
    assert expected in request.locked_reason
    assert "\n" not in request.locked_reason
    assert request.intro_text.endswith(request.locked_reason)


def _phonon_method_controller(
    service: ProjectService,
    tmp_path: Path,
    *,
    rows: dict[str, tuple[str, str]],
    infos: list[tuple[str, str]],
    questions: list[tuple[str, str]],
    answer: bool,
    local: RecordingBackend,
):
    """Build a controller whose rows ``siteA``/``siteB`` are ready to submit.

    ``rows`` maps a site label to ``(GUI phonon method, pseudopotential family)``,
    stored as the row's configuration; materialize hands back one defined job per
    row so an allowed submission reaches the local backend.
    """

    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    by_label = {candidate.site.label: key for key, candidate in candidates.items()}
    keys = []
    for label, (method, family) in rows.items():
        key = by_label[label]
        keys.append(key)
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
    jobs = {key: _quantum_job(service, document, f"quantum_{key}", QUANTUM_PHONON_CALC_TYPE) for key in keys}
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
        rows={"siteA": ("DFPT", "ultrasoft"), "siteB": ("LINEARRESPONSE", "ultrasoft")},
        infos=infos,
        questions=questions,
        answer=True,
        local=local,
    )

    controller.submit(keys, target="local")

    assert local.calls == []
    assert questions == []
    ((title, message),) = infos
    assert title == "Phonon method needs norm-conserving pseudopotentials"
    assert "siteA (DFPT)" in message
    assert "siteB (LINEARRESPONSE)" in message
    assert "stage" not in message.lower()


def test_quantum_controller_asks_before_dfpt_with_an_unknown_pseudopotential_family(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    questions: list[tuple[str, str]] = []
    local = RecordingBackend()
    controller, keys = _phonon_method_controller(
        ProjectService(),
        tmp_path,
        rows={"siteA": ("DFPT", "unknown")},
        infos=infos,
        questions=questions,
        answer=False,
        local=local,
    )

    controller.submit(keys, target="local")

    assert local.calls == []
    assert infos == []
    ((title, message),) = questions
    assert title == "Check pseudopotentials for DFPT"
    assert "siteA (DFPT)" in message
    assert "norm-conserving" in message


def test_quantum_controller_submits_dfpt_with_an_unknown_family_once_confirmed(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    questions: list[tuple[str, str]] = []
    local = RecordingBackend()
    controller, keys = _phonon_method_controller(
        ProjectService(),
        tmp_path,
        rows={"siteA": ("DFPT", "unknown")},
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
        rows={"siteA": ("FINITEDISPLACEMENT", "ultrasoft"), "siteB": ("DFPT", "norm_conserving")},
        infos=infos,
        questions=questions,
        answer=False,
        local=local,
    )

    controller.submit(keys, target="local")

    assert infos == []
    assert questions == []
    assert len(local.calls) == 1


# --- projects that ran the legacy singlepoint job first ----------------------


def test_quantum_controller_keeps_a_completed_legacy_pairs_phonon_result(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    _bind_legacy_singlepoint(service, document, definition, key)
    phonon = _bind_phonon(service, document, definition, key, status=JobStatus.COMPLETED)
    run = CalculationRun(job_id=phonon.id, status=JobStatus.COMPLETED)
    document.runs[run.id] = run
    result = Result(result_type=QUANTUM_PHONON_CALC_TYPE, calculation_run_id=run.id, data={"muon_zpe_total_eV": 0.3})
    document.results[result.id] = result
    view = _FakeQuantumView()
    context = make_context(
        document=document, project_service=service, definition_output_directory=lambda _d: output_root
    )
    controller = QuantumCorrectionsController(context, _stub_quantum_deps(), view)

    assert controller.build_states()[key].status == "completed"
    assert controller.jobs_for_keys([key]) == [phonon]
    assert controller.materialize([key]) == []
    controller.refresh_view()
    assert [row.key for row in view.results[-1]] == [str(result.id)]
    assert controller.phonon_result(str(result.id)) is result


@pytest.mark.parametrize("legacy_status", [JobStatus.COMPLETED, JobStatus.FAILED])
def test_quantum_controller_singlepoint_only_row_is_ready_with_a_fresh_directory(
    tmp_path: Path, legacy_status: JobStatus
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    singlepoint = _bind_legacy_singlepoint(service, document, definition, key, status=legacy_status)
    definition.parameters["job_seed_names_by_label"][key] = "quantum_siteA"
    legacy_dir = _write_legacy_seed_directory(output_root, "quantum_siteA")
    controller = _quantum_document_controller(service, document, output_root)

    assert controller.build_states()[key].status == "ready"
    assert controller.jobs_for_keys([key]) == []

    (job,) = controller.materialize([key])

    assert job.calc_type == QUANTUM_PHONON_CALC_TYPE
    assert definition.parameters["job_seed_names_by_label"][key] == "quantum_siteA_phonon"
    assert definition.parameters["singlepoint_seed_names_by_label"] == {key: "quantum_siteA"}
    assert (output_root / "quantum_siteA_phonon" / "quantum_siteA_phonon.param").is_file()
    assert not (output_root / "quantum_siteA_phonon" / "quantum_siteA_phonon.check").exists()
    # The legacy singlepoint keeps its directory and its record.
    assert (legacy_dir / "quantum_siteA.check").read_bytes() == b"legacy checkpoint"
    assert document.jobs[singlepoint.id] is singlepoint
    assert job_output_directory(
        document,
        singlepoint,
        quantum_seed_name=lambda definition, row: quantum_seed_name_for_key(document, definition, row),
    ) == output_root / "quantum_siteA"
    assert job_output_directory(
        document,
        job,
        quantum_seed_name=lambda definition, row: quantum_seed_name_for_key(document, definition, row),
    ) == output_root / "quantum_siteA_phonon"


def test_quantum_controller_fresh_seed_skips_names_already_taken(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    definition.parameters["job_seed_names_by_label"].update({first: "quantum_siteA", second: "quantum_siteA_phonon"})
    _write_legacy_seed_directory(output_root, "quantum_siteA")
    (output_root / "quantum_siteA_phonon_2").mkdir()
    controller = _quantum_document_controller(service, document, output_root)

    controller.materialize([first])

    assert definition.parameters["job_seed_names_by_label"][first] == "quantum_siteA_phonon_3"
    # No legacy singlepoint is bound, so none is recorded.
    assert "singlepoint_seed_names_by_label" not in definition.parameters


@pytest.mark.parametrize(
    "status", [JobStatus.SUBMITTING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING]
)
def test_quantum_controller_refuses_a_row_whose_legacy_singlepoint_is_active(
    tmp_path: Path, status: JobStatus
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    singlepoint = _bind_legacy_singlepoint(service, document, definition, key, status=status)
    infos: list[tuple[str, str]] = []
    local = RecordingBackend()
    controller = _quantum_document_controller(service, document, output_root, local_backend=local, infos=infos)

    # The row shows the singlepoint, so it can be cancelled from the table.
    assert controller.build_states()[key].status == status.value
    assert controller.jobs_for_keys([key]) == [singlepoint]
    request = controller.configure_request([key])
    assert request is not None and request.read_only is True
    assert "older MuFinder version" in request.locked_reason

    controller.submit([key], target="local")

    assert local.calls == []
    ((title, message),) = infos
    assert title == "Singlepoint still in progress"
    assert f"siteA ({status.value})" in message
    assert "phonon_job_ids_by_label" in definition.parameters
    assert definition.parameters["phonon_job_ids_by_label"] == {}


def test_quantum_controller_directory_companions_name_the_legacy_singlepoint(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    definition.parameters["job_seed_names_by_label"].update({first: "quantum_siteA", second: "quantum_siteB"})
    singlepoint = _bind_legacy_singlepoint(service, document, definition, first)
    phonon = _bind_phonon(service, document, definition, first, status=JobStatus.FAILED)
    lone_phonon = _bind_phonon(service, document, definition, second, status=JobStatus.FAILED)

    def resolve(job: Job) -> Path | None:
        return job_output_directory(
            document,
            job,
            quantum_seed_name=lambda definition, row: quantum_seed_name_for_key(document, definition, row),
        )

    controller = _quantum_document_controller(service, document, output_root, job_output_directory=resolve)

    assert controller.definition_and_key_for_job(phonon) == (definition, first)
    assert controller.directory_companions(phonon) == [singlepoint]
    assert controller.directory_companions(singlepoint) == []
    assert controller.directory_companions(lone_phonon) == []
    singlepoint.status = JobStatus.RUNNING
    assert controller.directory_companions(phonon) == []
    singlepoint.status = JobStatus.COMPLETED
    # Once the singlepoint records its own seed, it no longer shares the directory.
    definition.parameters["singlepoint_seed_names_by_label"] = {first: "quantum_siteA_legacy"}
    assert controller.directory_companions(phonon) == []


def test_quantum_controller_apply_configuration_stores_every_row(tmp_path: Path) -> None:
    import dataclasses

    from mufinder.app.workflow_config_codecs import workflow_row_configuration

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
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
        configure_result({"task": "PHONON", "xc_functional": "PBE"}, untouched={"max_scf_cycles"}),
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
    document, _definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key, candidate = next(iter(candidates.items()))
    controller = _quantum_document_controller(service, document, output_root)

    assert controller.rename_source(key, "  ") is False
    assert controller.rename_source(key, "siteA") is False
    assert controller.rename_source("missing", "renamed") is False
    assert candidate.site.label == "siteA"
    assert controller.rename_source(key, " renamed ") is True
    assert document.sites[candidate.site.id].label == "renamed"


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


def test_quantum_controller_fresh_seed_and_legacy_binding_survive_a_reload(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    singlepoint = _bind_legacy_singlepoint(service, document, definition, key)
    definition.parameters["job_seed_names_by_label"][key] = "quantum_siteA"
    _write_legacy_seed_directory(output_root, "quantum_siteA")
    (job,) = _quantum_document_controller(service, document, output_root).materialize([key])
    service.save_project(document)

    loaded = service.load_project(document.project_root)
    controller = _quantum_document_controller(service, loaded, output_root)
    loaded_definition = controller.active_definition()

    assert loaded_definition is not None
    assert controller.phonon_job_for_key(loaded_definition, key).id == job.id
    assert controller.legacy_singlepoint_job_for_key(loaded_definition, key).id == singlepoint.id
    assert loaded_definition.parameters["job_seed_names_by_label"][key] == "quantum_siteA_phonon"
    assert loaded_definition.parameters["singlepoint_seed_names_by_label"] == {key: "quantum_siteA"}
    assert controller.build_states()[key].status == "defined"

    def seed(owner, row: str) -> str:
        return quantum_seed_name_for_key(loaded, owner, row)

    assert job_output_directory(loaded, loaded.jobs[singlepoint.id], quantum_seed_name=seed) == (
        output_root / "quantum_siteA"
    )
