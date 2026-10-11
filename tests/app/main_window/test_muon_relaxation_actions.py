"""Muon relaxation actions in the main window."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

import mufinder.app.main_window as main_window_module
import mufinder.services.job_artifacts as job_artifacts_module
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.charge_states import MUON_RELAXATION_CALC_TYPES
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, MuonSite, Result, Structure
from mufinder.services.project_service import ProjectService
from mufinder.app.workflow_config_codecs import (
    deserialize_muon_relaxation_config,
    serialize_muon_relaxation_config,
)


def test_main_window_muon_job_uses_row_label_and_charge_seed_directory(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_seed_dir.mufinder", "MuonSeedDir")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    job = service.add_job(
        doc,
        label="Random Site 1",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
    )
    definition = CalculationDefinition(
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={
            "output_directory": "files/muon_relaxation",
            "job_ids_by_label": {"positivemuon:Random Site 1": str(job.id)},
            "job_seed_names_by_label": {"positivemuon:Random Site 1": "positivemuon_Random_Site_1"},
        },
        input_structure_ids=[host.id],
    )
    doc.calculations[definition.id] = definition
    window._document = doc

    output_dir = doc.project_root / "files" / "muon_relaxation" / "positivemuon" / "positivemuon_Random_Site_1"
    output_dir.mkdir(parents=True, exist_ok=True)

    assert job.label == "Random Site 1"
    assert window._job_output_directory(job) == output_dir
    assert window._job_seed_name(job) == "positivemuon_Random_Site_1"
    assert window._cluster_workflow_directory_fragment([job]) == "muon_relaxation/positivemuon"
    window.close()


def test_main_window_muon_cluster_job_recovers_seed_from_remote_directory(
    main_window,
    tmp_path: Path,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_remote_seed_dir.mufinder", "MuonRemoteSeedDir")
    job = service.add_job(
        doc,
        label="Random Site 20",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
    )
    job.remote_job_id = "703535"
    job.remote_directory = "/mnt/redwoodfs/bmh/CoF2/muon_relaxation/positivemuon/positivemuon_Random_Site_20"
    definition = CalculationDefinition(
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={
            "output_directory": "files/muon_relaxation",
            "job_ids_by_label": {"positivemuon:Random Site 20": str(job.id)},
            "job_seed_names_by_label": {},
        },
    )
    doc.calculations[definition.id] = definition
    window._document = doc

    output_dir = (
        doc.project_root
        / "files"
        / "muon_relaxation"
        / "positivemuon"
        / "positivemuon_Random_Site_20"
    )
    stale_label_dir = doc.project_root / "files" / "muon_relaxation" / "positivemuon" / "Random Site 20"
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_label_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "positivemuon_Random_Site_20.0001.err").write_text("rank error\n", encoding="utf-8")
    (output_dir / "castep.mpi-703535.out").write_text("scheduler error\n", encoding="utf-8")

    candidates = window._error_file_candidates(job, output_dir)

    assert window._job_output_directory(job) == output_dir
    assert window._job_seed_name(job) == "positivemuon_Random_Site_20"
    assert [path.name for path in candidates] == [
        "castep.mpi-703535.out",
        "positivemuon_Random_Site_20.0001.err",
    ]
    window.close()


def test_main_window_muon_job_recovers_legacy_sanitized_seed_directory(
    main_window,
    tmp_path: Path,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_legacy_seed_dir.mufinder", "MuonLegacySeedDir")
    job = service.add_job(
        doc,
        label="Random Site 16",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=MUON_RELAXATION_CALC_TYPES["positive_muon"],
    )
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={
            "output_directory": "files/muon_relaxation",
            "job_ids_by_label": {"positivemuon:Random Site 16": str(job.id)},
            "job_seed_names_by_label": {},
        },
    )
    doc.calculations[definition.id] = definition
    window._document = doc

    output_dir = (
        doc.project_root
        / "files"
        / "muon_relaxation"
        / "positivemuon"
        / "positivemuon_Random_Site_16"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    assert window._job_output_directory(job) == output_dir
    assert window._job_seed_name(job) == "positivemuon_Random_Site_16"
    assert window._cluster_runner.invalid_remote_job_labels([window._job_seed_name(job)]) == []
    window.close()


def test_materialize_muon_relaxation_filters_charge_qualified_selection(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_charge_selection.mufinder", "MuonChargeSelection")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    site_id = uuid4()
    definition = CalculationDefinition(
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={
            "configuration": {
                "rows": [
                    {
                        "site_id": str(site_id),
                        "label": "Shared Site",
                        "initial_fractional_position": [0.1, 0.2, 0.3],
                        "charge_state": "positive_muon",
                    },
                    {
                        "site_id": str(site_id),
                        "label": "Shared Site",
                        "initial_fractional_position": [0.1, 0.2, 0.3],
                        "charge_state": "muonium",
                    },
                ]
            },
            "output_directory": "files/muon_relaxation",
            "job_ids_by_label": {},
        },
        input_structure_ids=[host.id],
    )
    doc.calculations[definition.id] = definition
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    window._document = doc
    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    jobs = window._muon_relaxation_controller.materialize(["muonium:Shared Site"])

    assert len(jobs) == 1
    assert save_calls == ["save"]
    assert jobs[0].calc_type == "castep_muonium_relaxation"
    assert definition.parameters["job_ids_by_label"] == {"muonium:Shared Site": str(jobs[0].id)}
    assert "positivemuon:Shared Site" not in definition.parameters["job_ids_by_label"]
    assert (
        doc.project_root / "files" / "muon_relaxation" / "muonium" / "muonium_Shared_Site" / "muonium_Shared_Site.cell"
    ).exists()
    assert not (doc.project_root / "files" / "muon_relaxation" / "positivemuon").exists()
    window.close()


def test_muon_relaxation_staging_updates_memory_without_autosave(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_stage_autosave.mufinder", "MuonStageAutosave")
    site = MuonSite(
        label="Random Site 1",
        status="generated",
        fractional_position=[0.1, 0.2, 0.3],
        generation_method="random",
    )
    doc.sites[site.id] = site
    service.save_project(doc)
    window._document = doc
    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    class _AcceptedChargeDialog:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def exec(self):
            return main_window_module.QDialog.DialogCode.Accepted

        def selected_charge_states(self):
            return ("positive_muon",)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(main_window_module, "MuonChargeStateDialog", _AcceptedChargeDialog)

    window._on_create_muon_relaxation_jobs_requested(str(site.id))

    assert save_calls == []
    assert site.status == "staged"
    reloaded = service.load_project(doc.project_root)
    assert reloaded.sites[site.id].status == "generated"
    assert "muon_relaxation_form" not in reloaded.ui_state
    monkeypatch.undo()
    window.close()


def test_muon_relaxation_staging_rejected_charge_dialog_leaves_site_generated(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_stage_rejected.mufinder", "MuonStageRejected")
    site = MuonSite(
        label="Random Site 1",
        status="generated",
        fractional_position=[0.1, 0.2, 0.3],
        generation_method="random",
    )
    doc.sites[site.id] = site
    window._document = doc

    class _RejectedChargeDialog:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def exec(self):
            return main_window_module.QDialog.DialogCode.Rejected

    patch = pytest.MonkeyPatch()
    patch.setattr(main_window_module, "MuonChargeStateDialog", _RejectedChargeDialog)

    window._on_create_muon_relaxation_jobs_requested(str(site.id))

    assert site.status == "generated"
    assert "muon_relaxation_form" not in doc.ui_state
    patch.undo()
    window.close()


def test_muon_relaxation_row_removal_returns_staged_site_to_generated_without_autosave(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_remove_unsaved_stage.mufinder", "MuonRemoveUnsavedStage")
    site = MuonSite(
        label="Random Site 1",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        generation_method="random",
    )
    doc.sites[site.id] = site
    doc.ui_state["muon_relaxation_form"] = {
        "rows": [
            {
                "site_id": str(site.id),
                "label": "Random Site 1",
                "fractional_position": [0.1, 0.2, 0.3],
                "charge_state": "positive_muon",
            }
        ]
    }
    window._document = doc
    save_calls: list[str] = []

    def _save_project(document) -> None:
        save_calls.append("save")

    window._project_service.save_project = _save_project

    window._on_remove_muon_relaxation_rows(["positivemuon:Random Site 1"])

    assert save_calls == []
    assert site.status == "generated"
    assert doc.ui_state["muon_relaxation_form"]["rows"] == []
    window.close()


def test_muon_relaxation_row_removal_autosaves_persisted_staged_site_revert(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_remove_saved_stage.mufinder", "MuonRemoveSavedStage")
    site = MuonSite(
        label="Random Site 1",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        generation_method="random",
    )
    doc.sites[site.id] = site
    doc.ui_state["muon_relaxation_form"] = {
        "rows": [
            {
                "site_id": str(site.id),
                "label": "Random Site 1",
                "fractional_position": [0.1, 0.2, 0.3],
                "charge_state": "positive_muon",
            }
        ]
    }
    service.save_project(doc)
    window._document = doc
    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    window._on_remove_muon_relaxation_rows(["positivemuon:Random Site 1"])

    assert save_calls == ["save"]
    assert site.status == "generated"
    reloaded = service.load_project(doc.project_root)
    assert reloaded.sites[site.id].status == "generated"
    assert reloaded.ui_state["muon_relaxation_form"]["rows"] == []
    window.close()


def test_muon_relaxation_row_removal_keeps_site_staged_when_other_jobs_share_site(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_remove_shared_site.mufinder", "MuonRemoveSharedSite")
    site = MuonSite(
        label="Random Site 1",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        generation_method="random",
    )
    doc.sites[site.id] = site
    doc.ui_state["muon_relaxation_form"] = {
        "rows": [
            {
                "site_id": str(site.id),
                "label": "Random Site 1",
                "fractional_position": [0.1, 0.2, 0.3],
                "charge_state": "positive_muon",
            },
            {
                "site_id": str(site.id),
                "label": "Random Site 1",
                "fractional_position": [0.1, 0.2, 0.3],
                "charge_state": "muonium",
            },
        ]
    }
    window._document = doc

    window._on_remove_muon_relaxation_rows(["positivemuon:Random Site 1"])

    assert site.status == "staged"
    rows = doc.ui_state["muon_relaxation_form"]["rows"]
    assert len(rows) == 1
    assert rows[0]["charge_state"] == "muonium"
    assert rows[0]["site_id"] == str(site.id)
    window.close()


def test_muon_relaxation_staging_can_create_both_charge_state_tabs(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_stage_both.mufinder", "MuonStageBoth")
    site = MuonSite(
        label="Random Site 1",
        status="generated",
        fractional_position=[0.1, 0.2, 0.3],
        generation_method="random",
    )
    doc.sites[site.id] = site
    window._document = doc

    class _BothChargeDialog:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def exec(self):
            return main_window_module.QDialog.DialogCode.Accepted

        def selected_charge_states(self):
            return ("positive_muon", "muonium")

    patch = pytest.MonkeyPatch()
    patch.setattr(main_window_module, "MuonChargeStateDialog", _BothChargeDialog)

    window._on_create_muon_relaxation_jobs_requested(str(site.id))

    rows = window._document.ui_state["muon_relaxation_form"]["rows"]
    assert [row["charge_state"] for row in rows] == ["positive_muon", "muonium"]
    assert rows[0]["label"] == "Random Site 1"
    assert rows[1]["label"] == "Random Site 1"
    assert window._muon_relaxation_view._tables_by_charge_state["positive_muon"].rowCount() == 1
    assert window._muon_relaxation_view._tables_by_charge_state["muonium"].rowCount() == 1
    patch.undo()
    window.close()


def test_muon_relaxation_configuration_updates_memory_without_autosave(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_config_autosave.mufinder", "MuonConfigAutosave")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc
    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    window._on_muon_relaxation_configured(
        main_window_module.MuonRelaxationConfig(
            rows=(
                main_window_module.MuonRelaxationRow(
                    site_id=uuid4(),
                    label="Random Site 1",
                    initial_fractional_position=(0.1, 0.2, 0.3),
                    charge_state="positive_muon",
                ),
            )
        )
    )

    assert save_calls == []
    definition_id = UUID(doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY])
    assert definition_id in doc.calculations
    reloaded = service.load_project(doc.project_root)
    assert definition_id not in reloaded.calculations
    assert window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY not in reloaded.ui_state
    window.close()


def test_muon_relaxation_form_state_change_remains_manual_save_state(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_form_manual_save.mufinder", "MuonFormManualSave")
    window._document = doc
    save_calls: list[str] = []

    def _save_project(document) -> None:
        save_calls.append("save")

    window._project_service.save_project = _save_project

    window._on_muon_relaxation_form_state_changed({"rows": [{"label": "Unsaved UI row"}]})

    assert save_calls == []
    assert doc.ui_state["muon_relaxation_form"]["rows"][0]["label"] == "Unsaved UI row"
    reloaded = service.load_project(doc.project_root)
    assert "muon_relaxation_form" not in reloaded.ui_state
    window.close()


def test_main_window_jobs_table_refresh_syncs_muon_relaxation_status(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_status_sync.mufinder", "MuonStatusSync")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    site = MuonSite(
        label="Shared Site",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    doc.sites[site.id] = site
    job = service.add_job(
        doc,
        label="Shared Site",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
    )
    job.status = JobStatus.QUEUED
    definition = CalculationDefinition(
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={
            "configuration": {
                "rows": [
                    {
                        "site_id": str(site.id),
                        "label": "Shared Site",
                        "initial_fractional_position": [0.1, 0.2, 0.3],
                        "charge_state": "positive_muon",
                    }
                ]
            },
            "output_directory": "files/muon_relaxation",
            "job_ids_by_label": {"positivemuon:Shared Site": str(job.id)},
        },
        input_structure_ids=[host.id],
    )
    doc.calculations[definition.id] = definition
    doc.ui_state[main_window_module.MainWindow._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    window._document = doc
    window._muon_relaxation_view.set_form_state(
        {
            "rows": [
                {
                    "site_id": str(site.id),
                    "label": "Shared Site",
                    "fractional_position": [0.1, 0.2, 0.3],
                    "charge_state": "positive_muon",
                }
            ]
        }
    )

    window._refresh_jobs_table()

    status_item = window._muon_relaxation_view._table.item(
        0,
        window._muon_relaxation_view._STATUS_COLUMN,
    )
    assert status_item.text() == "queued"
    window.close()


def test_main_window_local_finish_muon_relaxation_records_folded_final_muon_position(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_final_position.mufinder", "MuonFinalPosition")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    site = MuonSite(
        label="Shared Site",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    doc.sites[site.id] = site
    job = service.add_job(
        doc,
        label="Shared Site",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muonium_relaxation",
    )
    job.status = JobStatus.RUNNING
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.RUNNING)
    doc.runs[run.id] = run
    definition = CalculationDefinition(
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={
            "configuration": {
                "rows": [
                    {
                        "site_id": str(site.id),
                        "label": "Shared Site",
                        "initial_fractional_position": [0.1, 0.2, 0.3],
                        "charge_state": "muonium",
                        "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                    }
                ]
            },
            "output_directory": "files/muon_relaxation",
            "job_ids_by_label": {"muonium:Shared Site": str(job.id)},
            "job_seed_names_by_label": {"muonium:Shared Site": "muonium_Shared_Site"},
        },
        input_structure_ids=[host.id],
    )
    doc.calculations[definition.id] = definition
    newer_definition = CalculationDefinition(
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={
            "configuration": {
                "rows": [
                    {
                        "site_id": str(site.id),
                        "label": "Newer Site",
                        "initial_fractional_position": [0.1, 0.2, 0.3],
                        "charge_state": "muonium",
                    }
                ]
            },
            "output_directory": "files/muon_relaxation",
            "job_ids_by_label": {},
        },
        input_structure_ids=[host.id],
    )
    doc.calculations[newer_definition.id] = newer_definition
    doc.ui_state[main_window_module.MainWindow._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(newer_definition.id)
    window._document = doc
    window._active_run_by_job[job.id] = run.id
    window._muon_relaxation_view.set_form_state(
        {
            "rows": [
                {
                    "site_id": str(site.id),
                    "label": "Shared Site",
                    "fractional_position": [0.1, 0.2, 0.3],
                    "charge_state": "muonium",
                    "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                }
            ]
        }
    )

    output_dir = doc.project_root / "files" / "muon_relaxation" / "muonium" / "muonium_Shared_Site"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "muonium_Shared_Site-out.cell").write_text("dummy", encoding="utf-8")
    # The relaxed -out.cell is parsed once by the off-thread collector
    # (mufinder.services.job_artifacts); patch its helpers, not main_window's.
    monkeypatch.setattr(
        job_artifacts_module,
        "parse_cell_file",
        lambda _path: SimpleNamespace(
            lattice_vectors=((8.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 4.0)),
            species=("Si", "H:mu"),
            fractional_positions=((0.0, 0.0, 0.0), (0.3, 0.4, 0.5)),
        ),
    )
    monkeypatch.setattr(
        job_artifacts_module,
        "lattice_parameters",
        lambda _vectors: (8.0, 8.0, 4.0, 90.0, 90.0, 90.0),
    )

    parsed = CastepResult(
        completed=True,
        total_energy=-12.0,
        warnings=(),
        geom_converged=True,
    )
    assert window._muon_relaxation_controller.active_definition().id == newer_definition.id

    window._on_local_job_finished(job.id, parsed)

    assert run.result_id is not None
    result = doc.results[run.result_id]
    assert result.data["muon_final_fractional_position_supercell"] == pytest.approx([0.3, 0.4, 0.5])
    assert result.data["muon_final_fractional_position_host"] == pytest.approx([0.6, 0.8, 0.5])
    doc.ui_state[main_window_module.MainWindow._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    states = window._muon_relaxation_controller.build_states()
    assert states["muonium:Shared Site"].final_fractional_position == pytest.approx((0.6, 0.8, 0.5))
    window._refresh_muon_relaxation_rows()
    muonium_table = window._muon_relaxation_view._tables_by_charge_state["muonium"]
    assert muonium_table.item(
        0,
        window._muon_relaxation_view._FINAL_FX_COLUMN,
    ).text() == "0.600"
    assert muonium_table.item(
        0,
        window._muon_relaxation_view._TOTAL_ENERGY_COLUMN,
    ).text() == "-12"
    window.close()


def test_main_window_form_muon_relaxation_shows_ready_before_submission(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_form_ready.mufinder", "MuonFormReady")
    site = MuonSite(label="Site 1", fractional_position=[0.25, 0.25, 0.25])
    doc.sites[site.id] = site
    doc.ui_state["muon_relaxation_form"] = {
        "rows": [
            {
                "site_id": str(site.id),
                "label": "Site 1",
                "initial_fractional_position": [0.25, 0.25, 0.25],
                "charge_state": "positive_muon",
                "parameter_values": {},
                "added_advanced": [],
            },
        ],
    }

    window._document = doc
    window._refresh_ui_after_project_change()

    table = window._muon_relaxation_view._tables_by_charge_state["positive_muon"]
    assert window._jobs_table.rowCount() == 0
    assert table.item(0, window._muon_relaxation_view._STATUS_COLUMN).text() == "ready"
    window.close()


def test_main_window_muon_relaxation_existing_defined_job_shows_defined(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_defined_status.mufinder", "MuonDefinedStatus")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    site = MuonSite(label="Site 1", fractional_position=[0.25, 0.25, 0.25])
    doc.sites[site.id] = site
    window._document = doc

    row = main_window_module.MuonRelaxationRow(
        site_id=site.id,
        label="Site 1",
        initial_fractional_position=(0.0, 0.0, 0.0),
        charge_state="positive_muon",
    )
    definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialize_muon_relaxation_config(main_window_module.MuonRelaxationConfig(rows=(row,))),
        output_directory=doc.project_root / "files" / "muon_relaxation",
    )
    job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.muon_relaxation_calc_type("positive_muon"),
    )
    assert job.status == JobStatus.DEFINED
    definition.parameters["job_ids_by_label"] = {"positivemuon:Site 1": str(job.id)}
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["muon_relaxation_form"] = serialize_muon_relaxation_config(
        main_window_module.MuonRelaxationConfig(rows=(row,))
    )

    window._refresh_ui_after_project_change()

    table = window._muon_relaxation_view._tables_by_charge_state["positive_muon"]
    assert table.item(0, window._muon_relaxation_view._STATUS_COLUMN).text() == "defined"
    assert window._muon_relaxation_controller.build_states()["positivemuon:Site 1"].status == "defined"
    window.close()


def test_main_window_muon_relaxation_sync_prefers_defined_job_over_cancelled_history(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_defined_beats_cancelled.mufinder", "MuonDefinedBeatsCancelled")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    site = MuonSite(label="Site 1", fractional_position=[0.25, 0.25, 0.25])
    doc.sites[site.id] = site
    window._document = doc

    row = main_window_module.MuonRelaxationRow(
        site_id=site.id,
        label="Site 1",
        initial_fractional_position=(0.0, 0.0, 0.0),
        charge_state="positive_muon",
    )
    config = main_window_module.MuonRelaxationConfig(rows=(row,))
    serialized = serialize_muon_relaxation_config(config)
    output_directory = doc.project_root / "files" / "muon_relaxation"
    historical_definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialized,
        output_directory=output_directory,
    )
    active_definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialized,
        output_directory=output_directory,
    )
    cancelled_job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.muon_relaxation_calc_type("positive_muon"),
    )
    service.cancel_job(cancelled_job)
    defined_job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.muon_relaxation_calc_type("positive_muon"),
    )
    key = "positivemuon:Site 1"
    historical_definition.parameters["job_ids_by_label"] = {key: str(cancelled_job.id)}
    active_definition.parameters["job_ids_by_label"] = {key: str(defined_job.id)}
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(active_definition.id)
    doc.ui_state["muon_relaxation_form"] = serialized

    window._refresh_ui_after_project_change()

    table = window._muon_relaxation_view._tables_by_charge_state["positive_muon"]
    assert active_definition.parameters["job_ids_by_label"][key] == str(defined_job.id)
    assert table.item(0, window._muon_relaxation_view._STATUS_COLUMN).text() == "defined"
    assert window._muon_relaxation_controller.jobs_for_keys([key]) == [defined_job]
    window.close()


def test_main_window_muon_relaxation_sync_ignores_historical_defined_jobs(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_defined_history.mufinder", "MuonDefinedHistory")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    site = MuonSite(label="Site 1", fractional_position=[0.25, 0.25, 0.25])
    doc.sites[site.id] = site
    window._document = doc

    row = main_window_module.MuonRelaxationRow(
        site_id=site.id,
        label="Site 1",
        initial_fractional_position=(0.25, 0.25, 0.25),
        charge_state="positive_muon",
    )
    config = main_window_module.MuonRelaxationConfig(rows=(row,))
    serialized = serialize_muon_relaxation_config(config)
    output_directory = doc.project_root / "files" / "muon_relaxation"
    historical_definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialized,
        output_directory=output_directory,
    )
    active_definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialized,
        output_directory=output_directory,
    )
    defined_job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.muon_relaxation_calc_type("positive_muon"),
    )
    key = "positivemuon:Site 1"
    historical_definition.parameters["job_ids_by_label"] = {key: str(defined_job.id)}
    historical_definition.parameters["job_seed_names_by_label"] = {key: "positivemuon_Site_1"}
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(active_definition.id)
    doc.ui_state["muon_relaxation_form"] = serialized

    window._refresh_ui_after_project_change()

    table = window._muon_relaxation_view._tables_by_charge_state["positive_muon"]
    assert active_definition.parameters.get("job_ids_by_label", {}) == {}
    assert table.item(0, window._muon_relaxation_view._STATUS_COLUMN).text() == "ready"
    window.close()


def test_main_window_muon_relaxation_recovers_active_dashboard_job_mapping(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_dashboard_recover.mufinder", "MuonDashboardRecover")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    site = MuonSite(label="Site 1", fractional_position=[0.25, 0.25, 0.25])
    doc.sites[site.id] = site
    window._document = doc
    # Post-Milestone-6.3, the only active jobs still present on project open are
    # cluster jobs (active local jobs are reconciled to STOPPED). Use a cluster
    # target so this test keeps exercising the active-mapping recovery path.
    monkeypatch.setattr(window._cluster_runner, "track_existing_job", lambda **kwargs: None)
    monkeypatch.setattr(window, "_job_output_directory", lambda job: tmp_path / "out" / job.label)
    monkeypatch.setattr(window, "_job_seed_name", lambda job: job.label)

    row = main_window_module.MuonRelaxationRow(
        site_id=site.id,
        label="Site 1",
        initial_fractional_position=(0.25, 0.25, 0.25),
        charge_state="positive_muon",
    )
    serialized = serialize_muon_relaxation_config(
        main_window_module.MuonRelaxationConfig(rows=(row,))
    )
    definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialized,
        output_directory=doc.project_root / "files" / "muon_relaxation",
    )
    queued_job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.muon_relaxation_calc_type("positive_muon"),
        execution_target="SCARF",
    )
    queued_job.status = JobStatus.QUEUED
    queued_job.submitted_at = datetime(2026, 5, 1, tzinfo=UTC)
    queued_job.remote_job_id = "12345"
    queued_job.remote_directory = "/remote/work/site1"
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["muon_relaxation_form"] = serialized

    window._refresh_ui_after_project_change()

    key = "positivemuon:Site 1"
    table = window._muon_relaxation_view._tables_by_charge_state["positive_muon"]
    assert definition.parameters["job_ids_by_label"][key] == str(queued_job.id)
    assert table.item(0, window._muon_relaxation_view._STATUS_COLUMN).text() == "queued"
    window.close()


def test_main_window_muon_relaxation_added_form_rows_preserve_existing_job_states(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_add_rows_preserve_status.mufinder", "MuonAddRowsPreserveStatus")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    first_site = MuonSite(label="Site 1", fractional_position=[0.25, 0.25, 0.25])
    second_site = MuonSite(label="Site 2", fractional_position=[0.5, 0.25, 0.25])
    doc.sites[first_site.id] = first_site
    doc.sites[second_site.id] = second_site
    window._document = doc

    existing_row = main_window_module.MuonRelaxationRow(
        site_id=first_site.id,
        label="Site 1",
        initial_fractional_position=(0.0, 0.0, 0.0),
        charge_state="positive_muon",
    )
    new_row = main_window_module.MuonRelaxationRow(
        site_id=second_site.id,
        label="Site 2",
        initial_fractional_position=(0.5, 0.25, 0.25),
        charge_state="positive_muon",
    )
    definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialize_muon_relaxation_config(
            main_window_module.MuonRelaxationConfig(rows=(existing_row,))
        ),
        output_directory=doc.project_root / "files" / "muon_relaxation",
    )
    # Cluster target so the running job survives orphaned-local-job
    # reconciliation on refresh (Milestone 6.3); this test is about existing
    # row-status preservation when a new row is added, not local reconciliation.
    monkeypatch.setattr(window._cluster_runner, "track_existing_job", lambda **kwargs: None)
    monkeypatch.setattr(window, "_job_output_directory", lambda job: tmp_path / "out" / job.label)
    monkeypatch.setattr(window, "_job_seed_name", lambda job: job.label)
    job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.muon_relaxation_calc_type("positive_muon"),
        execution_target="SCARF",
    )
    job.status = JobStatus.RUNNING
    job.remote_job_id = "12345"
    job.remote_directory = "/remote/work/site1"
    definition.parameters["job_ids_by_label"] = {"positivemuon:Site 1": str(job.id)}
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["muon_relaxation_form"] = serialize_muon_relaxation_config(
        main_window_module.MuonRelaxationConfig(rows=(existing_row, new_row))
    )

    window._refresh_ui_after_project_change()

    table = window._muon_relaxation_view._tables_by_charge_state["positive_muon"]
    assert table.item(0, window._muon_relaxation_view._STATUS_COLUMN).text() == "running"
    assert table.item(1, window._muon_relaxation_view._STATUS_COLUMN).text() == "ready"
    states = window._muon_relaxation_controller.build_states()
    assert states["positivemuon:Site 1"].status == "running"
    assert states["positivemuon:Site 2"].status == "ready"
    window.close()


def test_muon_relaxation_preserves_running_row_across_reconfiguration(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_running.mufinder", "MuonRunning")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    site = MuonSite(label="Site 1", fractional_position=[0.25, 0.25, 0.25])
    doc.sites[site.id] = site
    window._document = doc

    row = main_window_module.MuonRelaxationRow(
        site_id=site.id,
        label="Site 1",
        initial_fractional_position=(0.25, 0.25, 0.25),
        charge_state="positive_muon",
    )
    old_definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialize_muon_relaxation_config(main_window_module.MuonRelaxationConfig(rows=(row,))),
        output_directory=doc.project_root / "files" / "muon_relaxation",
    )
    running_job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.muon_relaxation_calc_type("positive_muon"),
    )
    running_job.status = JobStatus.RUNNING
    key = "positivemuon:Site 1"
    old_definition.parameters["job_ids_by_label"] = {key: str(running_job.id)}
    old_definition.parameters["job_seed_names_by_label"] = {key: "positivemuon_Site_1"}

    new_definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialize_muon_relaxation_config(main_window_module.MuonRelaxationConfig(rows=(row,))),
        output_directory=doc.project_root / "files" / "muon_relaxation",
    )
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(new_definition.id)

    states = window._muon_relaxation_controller.build_states()

    assert states["positivemuon:Site 1"].status == JobStatus.RUNNING.value
    assert states["positivemuon:Site 1"].job_id == running_job.id
    assert new_definition.parameters["job_ids_by_label"][key] == str(running_job.id)
    assert new_definition.parameters["job_seed_names_by_label"][key] == "positivemuon_Site_1"
    window.close()


def test_muon_relaxation_row_parameter_edit_touches_only_its_charge_state(
    main_window, tmp_path: Path
) -> None:
    """Two rows can share a label, so the edit is keyed by the charge-qualified
    row key and must leave the other charge state's row alone."""

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_param_edit.mufinder", "MuonParamEdit")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    site = MuonSite(label="Site 1", fractional_position=[0.25, 0.25, 0.25])
    doc.sites[site.id] = site
    window._document = doc

    rows = (
        main_window_module.MuonRelaxationRow(
            site_id=site.id,
            label="Site 1",
            initial_fractional_position=(0.25, 0.25, 0.25),
            charge_state="positive_muon",
            parameter_values={"geom_max_iter": 120},
        ),
        main_window_module.MuonRelaxationRow(
            site_id=site.id,
            label="Site 1",
            initial_fractional_position=(0.25, 0.25, 0.25),
            charge_state="muonium",
            parameter_values={"geom_max_iter": 400},
        ),
    )
    definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialize_muon_relaxation_config(
            main_window_module.MuonRelaxationConfig(rows=rows)
        ),
        output_directory=doc.project_root / "files" / "muon_relaxation",
    )
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.muon_relaxation_calc_type("positive_muon"),
    )
    job.status = JobStatus.FAILED
    definition.parameters["job_ids_by_label"] = {"positivemuon:Site 1": str(job.id)}
    window._restore_muon_relaxation_form_from_definition()
    assert (
        window._muon_relaxation_controller.build_states()["positivemuon:Site 1"].job_id
        == job.id
    )

    updated = window._muon_relaxation_controller.update_row_parameters(
        "positivemuon:Site 1", {"geom_max_iter": 500}
    )

    assert updated is True
    assert [
        candidate.id
        for candidate in doc.calculations.values()
        if candidate.workflow_stage == WorkflowStage.MUON_RELAXATION
    ] == [definition.id]
    assert definition.parameters["job_ids_by_label"] == {"positivemuon:Site 1": str(job.id)}
    states = window._muon_relaxation_controller.build_states()
    assert states["positivemuon:Site 1"].job_id == job.id
    assert states["positivemuon:Site 1"].status == JobStatus.FAILED.value

    config = deserialize_muon_relaxation_config(
        definition.parameters["configuration"]
    )
    by_charge = {row.charge_state: row for row in config.rows}
    assert by_charge["positive_muon"].parameter_values["geom_max_iter"] == 500
    assert by_charge["muonium"].parameter_values["geom_max_iter"] == 400
    form_config = window._muon_relaxation_form_config()
    form_by_charge = {row.charge_state: row for row in form_config.rows}
    for charge_state in ("positive_muon", "muonium"):
        assert window._muon_relaxation_row_signature(
            form_by_charge[charge_state]
        ) == window._muon_relaxation_row_signature(by_charge[charge_state])
    window.close()


def test_muon_relaxation_deleted_row_does_not_reinherit_stale_status(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_deleted_stale.mufinder", "MuonDeletedStale")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    site = MuonSite(label="Site 1", fractional_position=[0.25, 0.25, 0.25])
    doc.sites[site.id] = site
    window._document = doc

    row = main_window_module.MuonRelaxationRow(
        site_id=site.id,
        label="Site 1",
        initial_fractional_position=(0.25, 0.25, 0.25),
        charge_state="positive_muon",
    )
    old_definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialize_muon_relaxation_config(main_window_module.MuonRelaxationConfig(rows=(row,))),
        output_directory=doc.project_root / "files" / "muon_relaxation",
    )
    running_job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.muon_relaxation_calc_type("positive_muon"),
    )
    running_job.status = JobStatus.RUNNING
    old_definition.parameters["job_ids_by_label"] = {"positivemuon:Site 1": str(running_job.id)}

    window._on_remove_muon_relaxation_rows(["positivemuon:Site 1"])

    new_definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialize_muon_relaxation_config(main_window_module.MuonRelaxationConfig(rows=(row,))),
        output_directory=doc.project_root / "files" / "muon_relaxation",
    )
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(new_definition.id)

    states = window._muon_relaxation_controller.build_states()

    assert states["positivemuon:Site 1"].status == "ready"
    assert new_definition.parameters.get("job_ids_by_label", {}) == {}
    assert old_definition.id not in doc.calculations
    window.close()


def test_muon_relaxation_remove_row_archives_associated_files_and_job(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "muon_remove_archive.mufinder", "MuonRemoveArchive")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    site = MuonSite(label="Site 1", fractional_position=[0.25, 0.25, 0.25])
    doc.sites[site.id] = site
    window._document = doc

    row = main_window_module.MuonRelaxationRow(
        site_id=site.id,
        label="Site 1",
        initial_fractional_position=(0.25, 0.25, 0.25),
        charge_state="positive_muon",
    )
    definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialize_muon_relaxation_config(main_window_module.MuonRelaxationConfig(rows=(row,))),
        output_directory=doc.project_root / "files" / "muon_relaxation",
    )
    job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.muon_relaxation_calc_type("positive_muon"),
    )
    definition.parameters["job_ids_by_label"] = {"positivemuon:Site 1": str(job.id)}
    definition.parameters["job_seed_names_by_label"] = {"positivemuon:Site 1": "positivemuon_Site_1"}
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED, run_index=1)
    result = Result(calculation_run_id=run.id, result_type=job.calc_type, data={"total_energy": -1.0})
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site.status = "relaxed"
    site.relaxation_run_id = run.id
    site.total_energy_eV = -1.0
    job_dir = doc.project_root / "files" / "muon_relaxation" / "positivemuon" / "positivemuon_Site_1"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "positivemuon_Site_1.cell").write_text("cell", encoding="utf-8")
    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project
    monkeypatch.setattr(window, "_prompt_job_file_disposition", lambda **_kwargs: "archive")

    window._on_remove_muon_relaxation_rows(["positivemuon:Site 1"])

    assert save_calls == ["save"]
    archived = doc.project_root / "files" / "muon_relaxation" / "positivemuon" / "removed_from_workflow" / "positivemuon_Site_1"
    assert archived.exists()
    assert not job_dir.exists()
    assert job.id not in doc.jobs
    assert run.id not in doc.runs
    assert result.id not in doc.results
    assert site.relaxation_run_id is None
    assert site.total_energy_eV is None
    assert site.status == "generated"
    assert definition.id not in doc.calculations
    assert service.validate_document(doc) == []
    reloaded = service.load_project(doc.project_root)
    assert job.id not in reloaded.jobs
    assert run.id not in reloaded.runs
    assert reloaded.sites[site.id].relaxation_run_id is None
    assert reloaded.sites[site.id].status == "generated"
    window.close()
