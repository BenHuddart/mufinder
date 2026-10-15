"""Quantum Corrections actions in the main window: staging rows and resetting them to stage 1 or 2."""

from __future__ import annotations

from pathlib import Path

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Result, Structure
from mufinder.services.project_service import ProjectService
from mufinder.workflow.quantum import QUANTUM_SINGLEPOINT_CALC_TYPE
from tests.support.main_window import stage_quantum_row_with_singlepoint


def _answer_yes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Answer every yes/no question with Yes and return the questions' titles."""

    titles: list[str] = []

    def question(_parent, title, _text, _buttons, default):
        assert default == main_window_module.QMessageBox.StandardButton.No
        titles.append(title)
        return main_window_module.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(main_window_module.QMessageBox, "question", question)
    return titles


def test_main_window_reset_quantum_corrections_to_stage_one_clears_shared_job_state(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_stage_reset.mufinder", "QuantumStageReset")
    window._document = doc

    relaxed_structure = Structure(
        label="site relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_structure.id] = relaxed_structure

    source_job = Job(
        label="mu-site",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[source_job.id] = source_job
    run = CalculationRun(job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
    doc.sites[site.id] = site

    definition = window._quantum_corrections_controller.ensure_definition([str(site.id)])
    assert definition is not None
    definition.parameters["job_seed_names_by_label"] = {str(site.id): "quantum_siteA"}
    config = definition.parameters.setdefault("configuration", {})
    assert isinstance(config, dict)
    row_settings = config.setdefault("row_settings", {})
    assert isinstance(row_settings, dict)
    row_settings[str(site.id)] = {
        "parameter_values": {"cut_off_energy": {"value": 650.0, "unit": "eV"}},
        "added_advanced": [],
        "pseudopotential_family": "norm_conserving",
        "num_farms": 4,
    }

    singlepoint_job = service.add_job(
        doc,
        label="quantum_singlepoint_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_SINGLEPOINT_CALC_TYPE,
    )
    singlepoint_job.status = JobStatus.COMPLETED
    definition.parameters["singlepoint_job_ids_by_label"] = {str(site.id): str(singlepoint_job.id)}

    output_dir = doc.project_root / "files" / "quantum_corrections" / "quantum_siteA"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quantum_siteA.check").write_text("stale", encoding="utf-8")
    (output_dir / "quantum_siteA.param").write_text("stale", encoding="utf-8")

    _answer_yes(monkeypatch)

    window._quantum_corrections_view.reset_stage_requested.emit([str(site.id)])

    assert singlepoint_job.id not in doc.jobs
    assert definition.parameters["singlepoint_job_ids_by_label"] == {}
    assert definition.parameters["job_seed_names_by_label"] == {}
    assert row_settings[str(site.id)]["num_farms"] == 4
    assert output_dir.exists() is False

    candidate = window._quantum_corrections_controller.candidate_by_key(str(site.id))
    assert candidate is not None
    stage_state = window._quantum_corrections_controller.stage_state(definition, candidate)
    assert stage_state.stage == "singlepoint"
    assert stage_state.status == "ready"

    window.close()


@pytest.mark.parametrize(
    "singlepoint_status",
    [JobStatus.DEFINED, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED],
)
def test_quantum_can_reset_to_stage_one_for_inactive_bare_singlepoint(
    main_window, tmp_path: Path, singlepoint_status: JobStatus
) -> None:
    # Every inactive stage 1 can be rewound, including one still "defined"
    # (staged but never submitted, or returned there by the dashboard's Reset).
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / f"quantum_gate_{singlepoint_status.value}.mufinder", "QuantumGate"
    )
    window._document = doc
    definition, candidate, _ = stage_quantum_row_with_singlepoint(
        window, service, doc, singlepoint_status=singlepoint_status
    )

    assert window._quantum_corrections_controller.can_reset_to_stage_one(definition, candidate) is True
    window.close()


@pytest.mark.parametrize(
    "singlepoint_status",
    [JobStatus.SUBMITTING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING],
)
def test_quantum_cannot_reset_to_stage_one_for_active_bare_singlepoint(
    main_window, tmp_path: Path, singlepoint_status: JobStatus
) -> None:
    # Deleting a live job's records would strand the run: cancel it first.
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / f"quantum_gate_{singlepoint_status.value}.mufinder", "QuantumGate"
    )
    window._document = doc
    definition, candidate, _ = stage_quantum_row_with_singlepoint(
        window, service, doc, singlepoint_status=singlepoint_status
    )

    assert window._quantum_corrections_controller.can_reset_to_stage_one(definition, candidate) is False
    window.close()


def test_reset_quantum_corrections_to_stage_one_recovers_a_failed_singlepoint(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_failed_stage1.mufinder", "QuantumFailedStage1")
    window._document = doc
    definition, candidate, singlepoint_job = stage_quantum_row_with_singlepoint(
        window, service, doc, singlepoint_status=JobStatus.FAILED
    )
    definition.parameters["job_seed_names_by_label"] = {candidate.key: "quantum_siteA"}

    output_dir = doc.project_root / "files" / "quantum_corrections" / "quantum_siteA"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quantum_siteA.castep").write_text("failed run", encoding="utf-8")

    _answer_yes(monkeypatch)

    window._quantum_corrections_view.reset_stage_requested.emit([candidate.key])

    assert singlepoint_job.id not in doc.jobs
    assert definition.parameters["singlepoint_job_ids_by_label"] == {}
    assert output_dir.exists() is False

    refreshed_candidate = window._quantum_corrections_controller.candidate_by_key(candidate.key)
    assert refreshed_candidate is not None
    stage_state = window._quantum_corrections_controller.stage_state(definition, refreshed_candidate)
    assert stage_state.stage == "singlepoint"
    assert stage_state.status == "ready"
    window.close()


class _FakeConfigDialog:
    """Records how a source Configure dialog was opened and accepts it."""

    def __init__(self, opened: list[dict[str, object]], *, num_farms: int, **kwargs) -> None:
        self._opened = opened
        self._num_farms = num_farms
        opened.append({"num_farms": num_farms, **kwargs})

    def setWindowTitle(self, title: str) -> None:  # noqa: N802 - Qt API naming.
        self._opened[-1]["title"] = title

    def set_values(self, values, *, preferred_exclusive=None) -> None:
        self._opened[-1]["values"] = dict(values)

    def set_advanced_names(self, names) -> None:
        return None

    def set_mixed_names(self, names) -> None:
        return None

    def exec(self):
        from PySide6.QtWidgets import QDialog

        return QDialog.DialogCode.Accepted

    def collect_values(self) -> dict[str, object]:
        return {}

    def advanced_names(self) -> tuple[str, ...]:
        return ()

    def untouched_mixed_names(self) -> set[str]:
        return set()

    def pseudopotential_family(self) -> str:
        return "unknown"

    def num_farms(self) -> int:
        return 4


def test_configuring_a_stage_two_row_edits_only_its_farm_count(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.workflow_config_codecs import workflow_row_configuration

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_farms.mufinder", "QuantumFarms")
    window._document = doc
    definition, candidate, _singlepoint = stage_quantum_row_with_singlepoint(
        window, service, doc, singlepoint_status=JobStatus.COMPLETED
    )
    opened: list[dict[str, object]] = []
    monkeypatch.setattr(
        main_window_module,
        "QuantumCorrectionsConfigDialog",
        lambda **kwargs: _FakeConfigDialog(opened, **kwargs),
    )

    window._quantum_corrections_view.configure_requested.emit([candidate.key])

    assert len(opened) == 1
    assert opened[0]["title"] == "Configure Quantum Corrections Stage 2 Farms"
    assert opened[0]["editable_param_names"] == ()
    assert opened[0]["num_farms_editable"] is True
    assert opened[0]["pseudopotential_editable"] is False
    assert opened[0]["read_only"] is False
    assert workflow_row_configuration(definition, candidate.key)[2] == {
        "pseudopotential_family": "unknown",
        "num_farms": 4,
    }


def test_reset_to_stage_one_after_a_dashboard_reset_failed(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed singlepoint the dashboard returned to "defined" is editable and resettable."""

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_dashboard_reset.mufinder", "QuantumDashboardReset")
    window._document = doc
    definition, candidate, singlepoint_job = stage_quantum_row_with_singlepoint(
        window, service, doc, singlepoint_status=JobStatus.FAILED
    )
    controller = window._quantum_corrections_controller

    window._reset_failed_jobs([singlepoint_job], cleanup_choice=False)

    assert singlepoint_job.status == JobStatus.DEFINED
    request = controller.configure_request([candidate.key])
    assert request is not None
    assert (request.read_only, request.fully_editable) == (False, True)
    assert controller.can_reset_to_stage_one(definition, candidate) is True

    titles = _answer_yes(monkeypatch)
    window._quantum_corrections_view.reset_stage_requested.emit([candidate.key])

    assert titles == ["Reset Quantum Corrections"]
    assert singlepoint_job.id not in doc.jobs
    assert controller.stage_state(definition, candidate).status == "ready"
    window.close()


def test_changing_stage_two_farms_after_a_phonon_failure(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reset to Stage 2 reopens a failed phonon job without redoing stage 1."""

    from mufinder.app.workflow_config_codecs import workflow_row_configuration
    from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_phonon_failure.mufinder", "QuantumPhononFailure")
    window._document = doc
    definition, candidate, singlepoint_job = stage_quantum_row_with_singlepoint(
        window, service, doc, singlepoint_status=JobStatus.COMPLETED
    )
    definition.parameters["job_seed_names_by_label"] = {candidate.key: "quantum_siteA"}
    phonon_job = service.add_job(
        doc,
        label="quantum_phonon_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
    )
    phonon_job.status = JobStatus.FAILED
    definition.parameters["phonon_job_ids_by_label"] = {candidate.key: str(phonon_job.id)}
    output_dir = doc.project_root / "files" / "quantum_corrections" / "quantum_siteA"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quantum_siteA.check").write_text("stage-1 checkpoint", encoding="utf-8")
    opened: list[dict[str, object]] = []
    monkeypatch.setattr(
        main_window_module,
        "QuantumCorrectionsConfigDialog",
        lambda **kwargs: _FakeConfigDialog(opened, **kwargs),
    )

    # Locked while the failure stands, and the dialog says what unlocks it.
    window._quantum_corrections_view.configure_requested.emit([candidate.key])
    assert opened[-1]["read_only"] is True
    assert "Use Reset to Stage 2" in str(opened[-1]["intro_text"])

    titles = _answer_yes(monkeypatch)
    window._quantum_corrections_view.reset_stage_two_requested.emit([candidate.key])

    assert titles == ["Reset Quantum Corrections to stage 2"]
    assert phonon_job.status == JobStatus.DEFINED
    assert singlepoint_job.status == JobStatus.COMPLETED
    assert (output_dir / "quantum_siteA.check").read_text(encoding="utf-8") == "stage-1 checkpoint"

    window._quantum_corrections_view.configure_requested.emit([candidate.key])
    assert opened[-1]["read_only"] is False
    assert opened[-1]["num_farms_editable"] is True
    assert workflow_row_configuration(definition, candidate.key)[2]["num_farms"] == 4
    window.close()


def test_reset_to_stage_two_is_not_applied_when_declined(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_stage_two_declined.mufinder", "QuantumDeclined")
    window._document = doc
    definition, candidate, _singlepoint = stage_quantum_row_with_singlepoint(
        window, service, doc, singlepoint_status=JobStatus.COMPLETED
    )
    phonon_job = service.add_job(
        doc,
        label="quantum_phonon_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
    )
    phonon_job.status = JobStatus.FAILED
    definition.parameters["phonon_job_ids_by_label"] = {candidate.key: str(phonon_job.id)}
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *_args: main_window_module.QMessageBox.StandardButton.No,
    )

    window._quantum_corrections_view.reset_stage_two_requested.emit([candidate.key])

    assert phonon_job.status == JobStatus.FAILED
    window.close()
