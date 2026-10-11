"""Cutoff-energy convergence actions in the main window."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.app.convergence_types import ConvergenceConfig, SweepPointRunState
from mufinder.app.main_window import MainWindow
from mufinder.castep_io.types import BlockSection, ValueWithUnit
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, Result, Structure
from mufinder.services.job_paths import definition_output_directory
from mufinder.services.project_service import ProjectService
from tests.support.main_window import (
    select_route,
)


def test_convergence_configured_registers_snapshot(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "conv_ui.mufinder", "ConvUI")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    window._project_service.add_structure(window._document, host)
    window._document.project.host_structure_id = host.id

    config = ConvergenceConfig(
        cutoff_energies=(300.0, 360.0),
        shared_params={"task": "SINGLEPOINT", "xc_functional": "PBE"},
        overrides={},
        name_prefix="conv",
    )

    window._on_convergence_configured(config)

    assert len(window._document.jobs) == 0
    assert len(window._document.calculations) == 1
    definition = next(iter(window._document.calculations.values()))
    assert definition.parameters["configuration"]["cutoff_energies"] == [300.0, 360.0]
    assert definition.parameters["job_ids_by_cutoff"] == {}
    window.close()


def test_refresh_convergence_simulation_cell_without_document_disables_selector(main_window) -> None:
    window = main_window

    window._refresh_convergence_simulation_cell()

    assert not window._convergence_view._simulation_cell_combo.isEnabled()
    assert "Load a host structure" in window._convergence_view._simulation_cell_summary.text()
    window.close()


def test_convergence_simulation_cell_create_requested_handles_missing_context(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    infos: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._on_convergence_simulation_cell_create_requested()

    service = ProjectService()
    window._document = service.create_project(tmp_path / "conv_supercell_missing_host.mufinder", "ConvNoHost")
    window._on_convergence_simulation_cell_create_requested()

    assert infos == ["No project"]
    assert warnings == ["No host structure"]
    window.close()


def test_convergence_simulation_cell_use_unit_cell_requested_resets_selection(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_supercell_unit_reset.mufinder", "ConvSupercellUnitReset")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    service.use_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        asset_structure_id=asset.id,
    )
    window._document = doc

    window._on_convergence_simulation_cell_use_unit_cell_requested()

    assert doc.ui_state["workflow_supercells"]["convergence_cutoff"]["mode"] == "unit_cell"
    window.close()


def test_convergence_supercell_initial_matrix_prefers_selected_asset(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_supercell_initial_matrix.mufinder", "ConvSupercellInitial")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[3, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    service.use_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        asset_structure_id=asset.id,
    )
    window._document = doc

    assert window._convergence_supercell_initial_matrix() == [[3, 0, 0], [0, 1, 0], [0, 0, 1]]
    window.close()


def test_convergence_manage_supercells_selects_returned_asset(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_supercell_editor_select.mufinder", "ConvEditorSelect")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        label="Shared 2x2x1",
        reuse_existing=False,
    )
    window._document = doc

    class _FakeDialog:
        def selected_structure_id(self):
            return asset.id

    monkeypatch.setattr(window, "_open_supercell_editor", lambda **_kwargs: _FakeDialog())

    window._on_convergence_simulation_cell_create_requested()

    assert doc.ui_state["workflow_supercells"]["convergence_cutoff"]["structure_id"] == str(asset.id)
    window.close()


def test_convergence_simulation_cell_selection_clones_other_workflow_asset(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_supercell_clone_ui.mufinder", "ConvSupercellClone")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    source_asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    window._document = doc

    window._refresh_convergence_simulation_cell()
    window._on_convergence_simulation_cell_selection_requested(str(source_asset.id))

    selection = doc.ui_state["workflow_supercells"]["convergence_cutoff"]
    cloned = doc.structures[UUID(selection["structure_id"])]

    assert selection["mode"] == "saved_supercell"
    assert cloned.id != source_asset.id
    assert cloned.properties["supercell_asset"]["owner_workflow_stage"] == "convergence_cutoff"
    assert cloned.properties["supercell_asset"]["source_asset_structure_id"] == str(source_asset.id)
    window.close()


def test_convergence_configured_uses_selected_saved_supercell(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_saved_supercell.mufinder", "ConvSavedSupercell")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    service.use_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        asset_structure_id=asset.id,
    )
    window._document = doc

    config = ConvergenceConfig(
        cutoff_energies=(300.0,),
        shared_params={"task": "SINGLEPOINT"},
        overrides={},
        name_prefix="conv",
    )

    window._on_convergence_configured(config)

    definition = next(iter(doc.calculations.values()))
    assert definition.input_structure_ids == [asset.id]
    assert definition.parameters["simulation_cell_mode"] == "saved_supercell"
    assert definition.parameters["supercell_source_workflow"] == "convergence_cutoff"
    window.close()


def test_convergence_configured_reuses_matching_definition(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_reuse_definition.mufinder", "ConvReuseDefinition")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    config = ConvergenceConfig(
        cutoff_energies=(300.0, 400.0),
        shared_params={"task": "SINGLEPOINT"},
        overrides={},
        name_prefix="conv",
    )

    window._on_convergence_configured(config)
    definition = next(iter(doc.calculations.values()))
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    definition.parameters["job_ids_by_cutoff"]["300"] = str(job.id)

    window._on_convergence_configured(config)

    definitions = [
        candidate
        for candidate in doc.calculations.values()
        if candidate.workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF
    ]
    assert definitions == [definition]
    assert doc.ui_state[window._ACTIVE_CONVERGENCE_DEFINITION_KEY] == str(definition.id)
    assert definition.parameters["job_ids_by_cutoff"] == {"300": str(job.id)}
    window.close()


def test_materialize_convergence_jobs_writes_saved_supercell_structure(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_materialize_supercell.mufinder", "ConvMaterializeSupercell")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    definition = service.register_convergence_cutoff_definition(
        doc,
        input_structure_id=asset.id,
        parameters={
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
            "name_prefix": "conv",
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        simulation_cell_mode="saved_supercell",
        supercell_source_workflow="convergence_cutoff",
    )
    doc.ui_state[window._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(definition.id)
    window._document = doc

    jobs = window._convergence_controller.materialize([300.0])

    assert len(jobs) == 1
    cell_path = doc.project_root / "files" / "convergence_cutoff" / "conv_ecut_300" / "conv_ecut_300.cell"
    cell_text = cell_path.read_text(encoding="utf-8")
    assert "10 0 0" in cell_text
    assert cell_text.count("Si ") >= 2
    window.close()


def test_submit_convergence_ignores_stale_same_label_jobs_for_ready_cutoffs(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_stale_same_label.mufinder", "ConvStaleSameLabel")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    definition = service.register_convergence_cutoff_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "cutoff_energies": [300.0, 1300.0, 1400.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
            "name_prefix": "cutoff_convergence",
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )
    doc.ui_state[window._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(definition.id)
    window._document = doc

    stale_300 = service.add_job(
        doc,
        label="cutoff_convergence_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    stale_1400 = service.add_job(
        doc,
        label="cutoff_convergence_ecut_1400",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )

    submitted: list[list[Job]] = []
    monkeypatch.setattr(window, "_submit_jobs_on_cluster", lambda jobs: submitted.append(jobs))

    window._convergence_controller.submit([300.0, 1300.0, 1400.0], target="cluster")

    assert len(submitted) == 1
    assert [job.label for job in submitted[0]] == [
        "cutoff_convergence_ecut_300",
        "cutoff_convergence_ecut_1300",
        "cutoff_convergence_ecut_1400",
    ]
    assert stale_300 not in submitted[0]
    assert stale_1400 not in submitted[0]
    assert len(doc.jobs) == 5
    window.close()


def test_convergence_states_merge_equivalent_definition_mappings(
    main_window,
    tmp_path: Path,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_merge_equivalent.mufinder", "ConvMergeEquivalent")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    parameters = {
        "cutoff_energies": [300.0, 400.0, 1400.0],
        "shared_params": {"task": "SINGLEPOINT"},
        "overrides": [],
        "name_prefix": "cutoff_convergence",
    }
    output_directory = doc.project_root / "files" / "convergence_cutoff"
    older = service.register_convergence_cutoff_definition(
        doc,
        input_structure_id=host.id,
        parameters=parameters,
        output_directory=output_directory,
    )
    active = service.register_convergence_cutoff_definition(
        doc,
        input_structure_id=host.id,
        parameters=parameters,
        output_directory=output_directory,
    )

    running_300 = service.add_job(
        doc,
        label="cutoff_convergence_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    running_300.status = JobStatus.RUNNING
    completed_400 = service.add_job(
        doc,
        label="cutoff_convergence_ecut_400",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    completed_400.status = JobStatus.COMPLETED
    failed_1400 = service.add_job(
        doc,
        label="cutoff_convergence_ecut_1400",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    failed_1400.status = JobStatus.FAILED
    stale_300 = service.add_job(
        doc,
        label="cutoff_convergence_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )

    older.parameters["job_ids_by_cutoff"] = {
        "300": str(running_300.id),
        "400": str(completed_400.id),
        "1400": str(failed_1400.id),
    }
    active.parameters["job_ids_by_cutoff"] = {"300": str(stale_300.id)}
    doc.ui_state[window._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(active.id)
    window._document = doc

    states = window._convergence_controller.build_states()

    assert states[300.0].status == "running"
    assert states[300.0].job_id == running_300.id
    assert states[400.0].status == "completed"
    assert states[1400.0].status == "failed"
    assert active.parameters["job_ids_by_cutoff"]["300"] == str(running_300.id)
    assert active.parameters["job_ids_by_cutoff"]["400"] == str(completed_400.id)
    assert active.parameters["job_ids_by_cutoff"]["1400"] == str(failed_1400.id)
    window.close()


def test_materialize_convergence_jobs_repairs_legacy_zero_spin_supercell_asset(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "conv_materialize_legacy_spin_repair.mufinder",
        "ConvMaterializeLegacySpinRepair",
    )
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        properties={
            "atomic_spins": [2.0],
            "magnetic_structure": {
                "moments": [{"label": "Fe1", "vector": [0.0, 0.0, 2.0], "coordinate_system": "cartesian"}],
                "propagation_vector": [0.0, 0.0, 0.0],
            },
        },
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    asset.properties["atomic_spins"] = [0.0] * len(asset.species)
    definition = service.register_convergence_cutoff_definition(
        doc,
        input_structure_id=asset.id,
        parameters={
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT", "spin_polarized": True},
            "overrides": [],
            "name_prefix": "conv",
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        simulation_cell_mode="saved_supercell",
        supercell_source_workflow="convergence_cutoff",
    )
    service.use_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        asset_structure_id=asset.id,
    )
    doc.ui_state[window._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(definition.id)
    window._document = doc

    jobs = window._convergence_controller.materialize([300.0])

    assert len(jobs) == 1
    cell_path = doc.project_root / "files" / "convergence_cutoff" / "conv_ecut_300" / "conv_ecut_300.cell"
    cell_text = cell_path.read_text(encoding="utf-8")
    assert "SPIN=2" in cell_text
    window.close()


def test_main_window_syncs_convergence_row_status_and_energy(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_runtime.mufinder", "ConvRuntime")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.ui_state["convergence_form"] = {
        "name_prefix": "conv",
        "cutoff_energies": [300.0, 360.0],
        "shared_params": {"task": "SINGLEPOINT"},
    }

    job_300 = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job_360 = service.add_job(
        doc,
        label="conv_ecut_360",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job_300.status = JobStatus.COMPLETED
    job_360.status = JobStatus.DEFINED

    run = CalculationRun(job_id=job_300.id, status=JobStatus.COMPLETED, run_index=1)
    result = Result(calculation_run_id=run.id, result_type="castep_singlepoint", data={"total_energy": -12.3456})
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result

    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0, 360.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job_300.id), main_window_module.MainWindow._cutoff_key(360.0): str(job_360.id)},
    )

    window._document = doc
    window._refresh_ui_after_project_change()

    assert window._convergence_view._table.item(0, window._convergence_view._STATUS_COLUMN).text() == "completed"
    assert window._convergence_view._table.item(0, window._convergence_view._TOTAL_ENERGY_COLUMN).text() == "-12.3456"
    assert window._convergence_view._table.item(0, window._convergence_view._MAX_FORCE_COLUMN).text() == ""
    assert window._convergence_view._table.item(1, window._convergence_view._STATUS_COLUMN).text() == "ready"
    window.close()


def test_main_window_syncs_convergence_row_max_force(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_force_runtime.mufinder", "ConvForceRuntime")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.ui_state["convergence_form"] = {
        "name_prefix": "conv",
        "cutoff_energies": [300.0],
        "shared_params": {"task": "SINGLEPOINT"},
    }

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.COMPLETED

    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED, run_index=1)
    result = Result(
        calculation_run_id=run.id,
        result_type="castep_singlepoint",
        data={"total_energy": -12.3456, "max_force": 0.0987},
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result

    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )

    window._document = doc
    window._refresh_ui_after_project_change()

    assert window._convergence_view._table.item(0, window._convergence_view._MAX_FORCE_COLUMN).text() == "0.0987"
    window.close()


def test_main_window_convergence_status_uses_cutoff_energies_when_legacy_sweep_values_are_stale(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_stale_sweep_values.mufinder", "ConvStaleSweepValues")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.ui_state["convergence_form"] = {
        "name_prefix": "conv",
        "cutoff_energies": [400.0, 500.0],
        "shared_params": {"task": "SINGLEPOINT"},
    }

    job_400 = service.add_job(
        doc,
        label="conv_ecut_400",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job_400.status = JobStatus.COMPLETED
    run_400 = CalculationRun(job_id=job_400.id, status=JobStatus.COMPLETED, run_index=1)
    result_400 = Result(calculation_run_id=run_400.id, result_type="castep_singlepoint", data={"total_energy": -10.4})
    run_400.result_id = result_400.id
    doc.runs[run_400.id] = run_400
    doc.results[result_400.id] = result_400

    job_500 = service.add_job(
        doc,
        label="conv_ecut_500",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job_500.status = JobStatus.COMPLETED
    run_500 = CalculationRun(job_id=job_500.id, status=JobStatus.COMPLETED, run_index=1)
    result_500 = Result(calculation_run_id=run_500.id, result_type="castep_singlepoint", data={"total_energy": -10.5})
    run_500.result_id = result_500.id
    doc.runs[run_500.id] = run_500
    doc.results[result_500.id] = result_500

    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "sweep_values": [300.0, 400.0, 500.0],
            "cutoff_energies": [400.0, 500.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={
            main_window_module.MainWindow._cutoff_key(400.0): str(job_400.id),
            main_window_module.MainWindow._cutoff_key(500.0): str(job_500.id),
        },
    )

    window._document = doc
    states = window._convergence_controller.build_states()

    assert set(states) == {400.0, 500.0}
    assert states[400.0].status == "completed"
    assert states[500.0].status == "completed"
    window.close()


def test_main_window_deserializes_convergence_config_with_structured_values(main_window) -> None:
    window = main_window

    config = window._deserialize_convergence_config(
        {
            "name_prefix": " conv ",
            "cutoff_energies": [300.0, 360.0],
            "shared_params": {
                "task": "SINGLEPOINT",
                "elec_energy_tol": {"value": 1e-6, "unit": "eV"},
                "kpoint_mp_grid": [2, 2, 2],
                "species_pot": {"name": "species_pot", "rows": [["Si", "Si_00PBE.usp"]]},
            },
            "overrides": [
                {
                    "cutoffs": [360.0],
                    "params": {"elec_energy_tol": {"value": 2e-6, "unit": "eV"}},
                },
                {"cutoffs": ["bad"], "params": {"task": "SINGLEPOINT"}},
            ],
        }
    )

    assert config is not None
    assert config.name_prefix == "conv"
    assert isinstance(config.shared_params["elec_energy_tol"], ValueWithUnit)
    assert config.shared_params["kpoint_mp_grid"] == (2, 2, 2)
    assert isinstance(config.shared_params["species_pot"], BlockSection)
    assert config.overrides[frozenset({360.0})]["elec_energy_tol"].unit == "eV"
    window.close()


def test_main_window_deserializes_cutoff_config_prefers_cutoff_energies_when_fields_diverge(main_window) -> None:
    window = main_window

    config = window._deserialize_convergence_config(
        {
            "name_prefix": "conv",
            "sweep_parameter": "cut_off_energy",
            "sweep_values": [300.0, 360.0],
            "cutoff_energies": [360.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        }
    )

    assert config is not None
    assert config.sweep_values == (360.0,)
    assert config.cutoff_energies == (360.0,)
    window.close()


def test_main_window_convergence_definition_helpers_support_legacy_labels(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_legacy_labels.mufinder", "ConvLegacyLabels")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )
    definition.parameters["job_labels"] = [job.label]
    definition.parameters.pop("job_ids_by_cutoff", None)
    window._document = doc

    resolved_job = window._convergence_controller.job_for_cutoff(definition, 300.0)

    assert resolved_job is job
    assert definition_output_directory(doc, definition) == doc.project_root / "files" / "convergence_cutoff"
    window.close()


def test_main_window_detach_convergence_job_clears_job_id_mapping(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_detach_job.mufinder", "ConvDetachJob")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )
    window._document = doc

    window._detach_convergence_job(job.id)

    assert definition.parameters["job_ids_by_cutoff"] == {}
    window.close()


def test_main_window_build_states_merges_equivalent_convergence_definition_jobs(
    main_window,
    tmp_path: Path,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_active_definition.mufinder", "ConvActiveDefinition")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    old_job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    old_job.status = JobStatus.COMPLETED
    run = CalculationRun(job_id=old_job.id, status=JobStatus.COMPLETED, run_index=1)
    result = Result(calculation_run_id=run.id, result_type="castep_singlepoint", data={"total_energy": -9.99})
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result

    old_definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(old_job.id)},
    )
    new_definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )
    doc.ui_state[window._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(new_definition.id)
    window._document = doc

    states = window._convergence_controller.build_states()

    assert old_definition.id != new_definition.id
    assert states[300.0].status == "completed"
    assert states[300.0].total_energy == -9.99
    window.close()


def test_convergence_states_merge_matching_cutoffs_from_reshaped_configuration(
    main_window,
    tmp_path: Path,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_merge_reshaped.mufinder", "ConvMergeReshaped")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    old_completed_300 = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    old_completed_300.status = JobStatus.COMPLETED
    completed_400 = service.add_job(
        doc,
        label="conv_ecut_400",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    completed_400.status = JobStatus.COMPLETED

    old_definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0, 400.0],
            "shared_params": {"task": "SINGLEPOINT", "max_scf_cycles": 100},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={
            main_window_module.MainWindow._cutoff_key(300.0): str(old_completed_300.id),
            main_window_module.MainWindow._cutoff_key(400.0): str(completed_400.id),
        },
    )
    active_definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0, 400.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [
                {"cutoffs": [300.0], "params": {"max_scf_cycles": 200}},
                {"cutoffs": [400.0], "params": {"max_scf_cycles": 100}},
            ],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )
    doc.ui_state[window._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(active_definition.id)
    window._document = doc

    states = window._convergence_controller.build_states()

    assert old_definition.id != active_definition.id
    assert states[300.0].status == "ready"
    assert states[400.0].status == "completed"
    assert active_definition.parameters["job_ids_by_cutoff"] == {"400": str(completed_400.id)}
    window.close()


def test_main_window_regenerated_cutoff_list_does_not_inherit_stale_results(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_stale_result_guard.mufinder", "ConvStaleResultGuard")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    old_job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    old_job.status = JobStatus.COMPLETED
    run = CalculationRun(job_id=old_job.id, status=JobStatus.COMPLETED, run_index=1)
    result = Result(calculation_run_id=run.id, result_type="castep_singlepoint", data={"total_energy": -9.99})
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result

    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(old_job.id)},
    )

    # Simulate a regenerated list that diverges from the active definition.
    doc.ui_state["convergence_form"] = {
        "name_prefix": "conv",
        "cutoff_energies": [300.0, 340.0],
        "cutoff_parameter_rows": [
            {"cutoff": 300.0, "params": {"task": "SINGLEPOINT"}, "added_advanced": []},
            {"cutoff": 340.0, "params": {"task": "SINGLEPOINT"}, "added_advanced": []},
        ],
    }
    window._document = doc

    states = window._convergence_controller.build_states()

    assert set(states) == {300.0, 340.0}
    assert states[300.0].status == "ready"
    assert states[300.0].total_energy is None
    assert states[340.0].status == "ready"
    assert states[340.0].total_energy is None
    window.close()


def test_convergence_configured_without_project_shows_info(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._on_convergence_configured(
        ConvergenceConfig(cutoff_energies=(300.0,), shared_params={"task": "SINGLEPOINT"})
    )

    assert infos == ["No project"]
    window.close()


def test_convergence_configured_without_host_structure_warns(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "conv_no_host.mufinder", "ConvNoHost")

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._on_convergence_configured(
        ConvergenceConfig(cutoff_energies=(300.0,), shared_params={"task": "SINGLEPOINT"})
    )

    assert warnings == ["No host structure"]
    window.close()


def test_convergence_configured_handles_expansion_errors(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_expand_err.mufinder", "ConvExpandErr")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )
    monkeypatch.setattr(
        main_window_module,
        "expand_convergence_jobs",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("invalid")),
    )

    window._on_convergence_configured(
        ConvergenceConfig(cutoff_energies=(300.0,), shared_params={"task": "SINGLEPOINT"})
    )

    assert warnings == ["Convergence setup invalid"]
    window.close()


def test_convergence_configured_saves_ready_configuration_without_materializing_jobs(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_ready_only.mufinder", "ConvReadyOnly")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    config = ConvergenceConfig(
        cutoff_energies=(300.0,),
        shared_params={"task": "SINGLEPOINT"},
        overrides={},
        name_prefix="conv",
    )

    window._on_convergence_configured(config)

    assert len(doc.calculations) == 1
    assert len(doc.jobs) == 0
    assert not (doc.project_root / "files" / "convergence_cutoff").exists()
    window.close()


def test_main_window_convergence_failed_handlers_forward_jobs(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_failed_forward.mufinder", "ConvFailedForward")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED

    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )

    window._document = doc

    inspected: list[list[object]] = []
    reset: list[list[object]] = []
    monkeypatch.setattr(window, "_inspect_failed_jobs", lambda jobs: inspected.append(jobs))
    monkeypatch.setattr(window, "_reset_failed_jobs", lambda jobs: reset.append(jobs))

    window._convergence_view.inspect_failed_requested.emit([300.0])
    window._convergence_view.reset_failed_requested.emit([300.0])

    assert inspected == [[job]]
    assert reset == [[job]]
    window.close()


def test_main_window_configured_cutoff_shows_ready_and_stays_off_dashboard(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_ready.mufinder", "ConvReady")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )
    doc.ui_state["convergence_form"] = {
        "name_prefix": "conv",
        "cutoff_energies": [300.0],
        "shared_params": {"task": "SINGLEPOINT"},
    }

    window._document = doc
    window._refresh_ui_after_project_change()

    assert window._jobs_table.rowCount() == 0
    assert window._convergence_view._table.item(0, window._convergence_view._STATUS_COLUMN).text() == "ready"
    window.close()


def test_main_window_form_cutoff_shows_ready_before_submission(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_form_ready.mufinder", "ConvFormReady")

    doc.ui_state["convergence_form"] = {
        "name_prefix": "conv",
        "cutoff_energies": [300.0, 360.0],
        "cutoff_parameter_rows": [
            {
                "cutoff": 300.0,
                "params": {"task": "SINGLEPOINT", "xc_functional": "PBE"},
                "added_advanced": [],
            },
            {
                "cutoff": 360.0,
                "params": {"task": "SINGLEPOINT", "xc_functional": "PBE"},
                "added_advanced": [],
            },
        ],
    }

    window._document = doc
    window._refresh_ui_after_project_change()

    assert window._jobs_table.rowCount() == 0
    assert window._convergence_view._table.item(0, window._convergence_view._STATUS_COLUMN).text() == "ready"
    assert window._convergence_view._table.item(1, window._convergence_view._STATUS_COLUMN).text() == "ready"
    window.close()


def test_main_window_recovers_convergence_form_rows_from_definition_when_ui_state_incomplete(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_restore_from_definition.mufinder", "ConvRestoreFromDefinition")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0, 360.0],
            "shared_params": {"task": "SINGLEPOINT", "max_scf_cycles": 75},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )

    doc.ui_state["convergence_form"] = {"name_prefix": "conv"}

    window._document = doc
    window._refresh_ui_after_project_change()

    assert window._convergence_view._table.rowCount() == 2
    assert window._convergence_view._table.item(0, window._convergence_view._KEY_COLUMN).text() == "300"
    assert window._convergence_view._table.item(1, window._convergence_view._KEY_COLUMN).text() == "360"
    assert doc.ui_state["convergence_form"]["cutoff_energies"] == [300.0, 360.0]
    window.close()


def test_main_window_reset_failed_convergence_job_restores_ready_state(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_reset_ready.mufinder", "ConvResetReady")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )
    doc.ui_state["convergence_form"] = {
        "name_prefix": "conv",
        "cutoff_energies": [300.0],
        "shared_params": {"task": "SINGLEPOINT"},
    }
    window._document = doc
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._reset_failed_jobs([job])

    assert window._jobs_table.rowCount() == 0
    assert window._convergence_controller.build_states()[300.0].status == "ready"
    window.close()


def test_main_window_remove_convergence_cutoff_deletes_files_and_entities(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_remove_delete.mufinder", "ConvRemoveDelete")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.ui_state["convergence_form"] = {
        "name_prefix": "conv",
        "cutoff_energies": [300.0],
        "shared_params": {"task": "SINGLEPOINT"},
    }

    output_root = doc.project_root / "files" / "convergence_cutoff"
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    job_dir = output_root / job.label
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{job.label}.castep").write_text("done", encoding="utf-8")

    run = CalculationRun(job_id=job.id, status=JobStatus.FAILED, run_index=1)
    result = Result(calculation_run_id=run.id, result_type="castep_singlepoint", data={"total_energy": -1.0})
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result

    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=output_root,
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )

    window._document = doc
    window._refresh_ui_after_project_change()
    monkeypatch.setattr(window, "_prompt_cutoff_file_disposition", lambda cutoff, path, **kwargs: "delete")

    window._on_remove_cutoffs_from_convergence([300.0])

    assert not job_dir.exists()
    assert doc.jobs == {}
    assert doc.runs == {}
    assert doc.results == {}
    assert doc.calculations == {}
    assert window._convergence_view._table.rowCount() == 0
    window.close()


def test_main_window_remove_convergence_cutoff_archives_files_when_kept(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_remove_archive.mufinder", "ConvRemoveArchive")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.ui_state["convergence_form"] = {
        "name_prefix": "conv",
        "cutoff_energies": [300.0],
        "shared_params": {"task": "SINGLEPOINT"},
    }

    output_root = doc.project_root / "files" / "convergence_cutoff"
    ready_dir = output_root / "conv_ecut_300"
    ready_dir.mkdir(parents=True, exist_ok=True)
    (ready_dir / "conv_ecut_300.param").write_text("param", encoding="utf-8")

    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=output_root,
    )

    window._document = doc
    window._refresh_ui_after_project_change()
    monkeypatch.setattr(window, "_prompt_cutoff_file_disposition", lambda cutoff, path, **kwargs: "archive")

    window._on_remove_cutoffs_from_convergence([300.0])

    archived = output_root / "removed_from_workflow" / "conv_ecut_300"
    assert archived.exists()
    assert not ready_dir.exists()
    assert doc.calculations == {}
    assert window._convergence_view._table.rowCount() == 0
    window.close()


def test_main_window_remove_convergence_cutoff_cancel_keeps_row_and_files(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_remove_cancel.mufinder", "ConvRemoveCancel")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.ui_state["convergence_form"] = {
        "name_prefix": "conv",
        "cutoff_energies": [300.0],
        "shared_params": {"task": "SINGLEPOINT"},
    }

    output_root = doc.project_root / "files" / "convergence_cutoff"
    ready_dir = output_root / "conv_ecut_300"
    ready_dir.mkdir(parents=True, exist_ok=True)
    (ready_dir / "conv_ecut_300.param").write_text("param", encoding="utf-8")

    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=output_root,
    )

    window._document = doc
    window._refresh_ui_after_project_change()
    monkeypatch.setattr(window, "_prompt_cutoff_file_disposition", lambda cutoff, path, **kwargs: None)

    window._on_remove_cutoffs_from_convergence([300.0])

    assert ready_dir.exists()
    assert len(doc.calculations) == 1
    assert window._convergence_view._table.rowCount() == 1
    window.close()


def test_main_window_remove_cutoffs_without_document_delegates_to_widget(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window

    removed: list[list[float]] = []
    monkeypatch.setattr(window._convergence_view, "remove_cutoffs", lambda cutoffs: removed.append(list(cutoffs)))

    window._on_remove_cutoffs_from_convergence([300.0])

    assert removed == [[300.0]]
    window.close()


def test_main_window_remove_single_convergence_cutoff_warns_on_file_cleanup_error(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_remove_error.mufinder", "ConvRemoveError")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    output_root = doc.project_root / "files" / "convergence_cutoff"
    ready_dir = output_root / "conv_ecut_300"
    ready_dir.mkdir(parents=True, exist_ok=True)
    (ready_dir / "conv_ecut_300.param").write_text("param", encoding="utf-8")
    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=output_root,
    )
    window._document = doc

    warnings: list[str] = []
    monkeypatch.setattr(window, "_prompt_cutoff_file_disposition", lambda cutoff, path, **kwargs: "delete")
    monkeypatch.setattr(window, "_delete_path", lambda path: (_ for _ in ()).throw(OSError("busy")))
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda _parent, title, _message: warnings.append(title))

    removed, _shared, aborted = window._remove_single_convergence_cutoff(300.0)
    assert removed is False
    assert aborted is False
    assert warnings == ["File cleanup failed"]
    window.close()


def test_main_window_remove_cutoff_from_definition_updates_lists_and_labels(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_remove_definition.mufinder", "ConvRemoveDefinition")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "sweep_values": [300.0, 360.0, "bad"],
            "cutoff_energies": [300.0, 360.0, "bad"],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [
                {"cutoffs": [300.0, 360.0], "params": {"max_scf_cycles": 50}},
                {"sweep_values": [300.0, 360.0], "params": {"metals_method": "DM"}},
                {"cutoffs": [300.0], "params": {"xc_functional": "PBE"}},
            ],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): "job-1"},
    )
    definition.parameters["job_labels"] = ["conv_ecut_300", "conv_ecut_360", "conv_ecut_bad"]
    window._document = doc

    window._convergence_controller.remove_cutoff_from_definition(definition, 300.0)

    assert definition.parameters["configuration"]["cutoff_energies"] == [360.0, "bad"]
    assert definition.parameters["configuration"]["sweep_values"] == [360.0, "bad"]
    assert definition.parameters["configuration"]["overrides"] == [
        {"cutoffs": [360.0], "params": {"max_scf_cycles": 50}},
        {"sweep_values": [360.0], "params": {"metals_method": "DM"}},
    ]
    assert definition.parameters["job_ids_by_cutoff"] == {}
    assert definition.parameters["job_labels"] == ["conv_ecut_360", "conv_ecut_bad"]
    window.close()


def test_main_window_remove_convergence_cutoff_clears_equivalent_definition_mappings(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_deleted_stale.mufinder", "ConvDeletedStale")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    old_definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )
    running_job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    running_job.status = JobStatus.RUNNING
    old_definition.parameters["job_ids_by_cutoff"] = {"300": str(running_job.id)}

    active_definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )
    doc.ui_state[window._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(active_definition.id)

    window._on_remove_cutoffs_from_convergence([300.0])

    new_definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )
    doc.ui_state[window._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(new_definition.id)

    states = window._convergence_controller.build_states()

    assert states[300.0].status == "ready"
    assert new_definition.parameters.get("job_ids_by_cutoff", {}) == {}
    assert old_definition.id not in doc.calculations
    window.close()


def test_main_window_remove_convergence_cutoff_preserves_other_contexts(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_preserve_other.mufinder", "ConvPreserveOther")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    active_definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )
    other_definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "other_convergence_cutoff",
    )
    doc.ui_state[window._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(active_definition.id)

    window._on_remove_cutoffs_from_convergence([300.0])

    assert active_definition.id not in doc.calculations
    assert other_definition.id in doc.calculations
    assert other_definition.parameters["configuration"]["cutoff_energies"] == [300.0]
    window.close()


def test_main_window_examine_convergence_launches_dialog(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_examine_launch.mufinder", "ConvExamineLaunch")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    states = {
        300.0: SweepPointRunState(status="completed", total_energy=-10.0, max_force=0.2),
        400.0: SweepPointRunState(status="completed", total_energy=-10.02, max_force=0.15),
    }
    monkeypatch.setattr(window._convergence_controller, "build_states", lambda: states)

    captured: dict[str, object] = {}

    class _FakeDialog:
        def __init__(self, cutoff_data, *, num_atoms, initial_state=None, parent=None) -> None:
            captured["cutoff_data"] = cutoff_data
            captured["num_atoms"] = num_atoms
            captured["initial_state"] = initial_state
            captured["parent"] = parent
            self.cutoff_adopted = SimpleNamespace(connect=lambda _cb: None)
            self.state_changed = SimpleNamespace(connect=lambda _cb: None)

        def exec(self) -> int:
            captured["opened"] = True
            return 0

        def state_snapshot(self) -> dict[str, object]:
            return {"restored": True}

    monkeypatch.setattr(main_window_module, "ConvergenceExaminerDialog", _FakeDialog)

    window._on_examine_convergence()

    assert captured["cutoff_data"] == states
    assert captured["num_atoms"] == 2
    assert captured["initial_state"] == {}
    assert captured["parent"] is window
    assert captured["opened"] is True
    assert doc.ui_state[window._CONVERGENCE_EXAMINER_STATE_KEY] == {"restored": True}
    window.close()


def test_main_window_on_cutoff_adopted_sets_project_default(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_adopt_default.mufinder", "ConvAdoptDefault")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    definition = service.register_convergence_cutoff_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "sweep_parameter": "cut_off_energy",
            "sweep_values": [300.0, 400.0, 550.0],
            "cutoff_energies": [300.0, 400.0, 550.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )
    doc.ui_state[window._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(definition.id)
    doc.ui_state[window._CONVERGENCE_EXAMINER_STATE_KEY] = {
        "plot_data": {
            "cutoffs": [300.0, 400.0, 550.0],
            "energies": [-10.0, -10.1, -10.15],
            "forces": [0.3, 0.18, 0.05],
            "energy_threshold": 0.001,
            "force_threshold": 0.05,
        }
    }
    window._document = doc
    window._refresh_ui_after_project_change()

    select_route(window, "convergence_cutoff")
    assert window._main_stack.currentWidget() is window._convergence_view

    window._on_cutoff_adopted(550.0, "asymptotic", 0.001, 0.05)

    assert doc.project.recommended_cutoff_energy == pytest.approx(550.0)
    assert doc.project.recommended_cutoff_method == "asymptotic"
    assert doc.project.recommended_cutoff_energy_threshold == pytest.approx(0.001)
    assert doc.project.recommended_cutoff_force_threshold == pytest.approx(0.05)
    assert doc.ui_state["default_cut_off_energy"] == pytest.approx(550.0)
    assert doc.ui_state["default_cut_off_energy_method"] == "asymptotic"
    assert doc.ui_state["default_cut_off_energy_threshold"] == pytest.approx(0.001)
    assert doc.ui_state["default_cut_off_force_threshold"] == pytest.approx(0.05)
    snapshot = doc.ui_state[window._CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY]["cutoff"]
    assert snapshot["host_structure_id"] == str(host.id)
    assert snapshot["simulation_cell_structure_id"] == str(host.id)
    assert snapshot["default_value"] == pytest.approx(550.0)
    assert snapshot["plot_data"]["cutoffs"] == [300.0, 400.0, 550.0]
    assert doc.provenance[-1].action == "set_recommended_cutoff"
    assert "550" in window.statusBar().currentMessage()
    assert window._main_stack.currentWidget() is window._convergence_view
    window.close()


def test_main_window_convergence_examiner_state_changed_updates_ui_state_without_saving_project(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "conv_examiner_state_update.mufinder", "ConvExaminerStateUpdate")
    window._document = doc
    initial_state_text = doc.project_root.joinpath("state.json").read_text(encoding="utf-8")

    saves: list[bool] = []
    monkeypatch.setattr(window._project_service, "save_project", lambda _doc: saves.append(True))

    window._on_convergence_examiner_state_changed({"results_text": "cached"})

    assert doc.ui_state[window._CONVERGENCE_EXAMINER_STATE_KEY] == {"results_text": "cached"}
    assert doc.project_root.joinpath("state.json").read_text(encoding="utf-8") == initial_state_text
    assert saves == []
    window.close()


def test_main_window_convergence_examiner_state_persists_across_project_reload_after_explicit_save(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    root = tmp_path / "conv_examiner_state_reload.mufinder"
    doc = service.create_project(root, "ConvExaminerStateReload")
    window._document = doc
    initial_state_text = doc.project_root.joinpath("state.json").read_text(encoding="utf-8")

    window._on_convergence_examiner_state_changed(
        {
            "results_text": "Recommended cutoff: 500 eV",
            "plot_data": {
                "cutoffs": [300.0, 400.0, 500.0],
                "energies": [-10.0, -10.01, -10.0105],
                "forces": [0.3, 0.2, 0.19],
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
            },
        }
    )

    assert doc.project_root.joinpath("state.json").read_text(encoding="utf-8") == initial_state_text

    service.save_project(doc)
    loaded = service.load_project(root)

    assert loaded.ui_state[window._CONVERGENCE_EXAMINER_STATE_KEY]["results_text"] == "Recommended cutoff: 500 eV"
    assert loaded.ui_state[window._CONVERGENCE_EXAMINER_STATE_KEY]["plot_data"]["cutoffs"] == [300.0, 400.0, 500.0]
    window.close()


def test_main_window_convergence_overview_restores_pinned_snapshots_on_reload(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    root = tmp_path / "overview_reload.mufinder"
    doc = service.create_project(root, "OverviewReload")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.project.recommended_cutoff_energy = 550.0
    doc.project.recommended_cutoff_method = "asymptotic"
    doc.project.recommended_cutoff_energy_threshold = 0.001
    doc.project.recommended_cutoff_force_threshold = 0.05
    doc.project.recommended_kpoint_grid = (2, 2, 2)
    doc.project.recommended_kpoint_method = "delta-threshold"
    doc.project.recommended_kpoint_energy_threshold = 0.001
    doc.project.recommended_kpoint_force_threshold = 0.05
    doc.ui_state[window._CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] = {
        "cutoff": {
            "host_structure_id": str(host.id),
            "simulation_cell_structure_id": str(host.id),
            "default_value": 550.0,
            "method": "asymptotic",
            "energy_threshold": 0.001,
            "force_threshold": 0.05,
            "plot_data": {
                "cutoffs": [300.0, 400.0, 550.0],
                "energies": [-10.0, -10.1, -10.15],
                "forces": [0.3, 0.18, 0.05],
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
            },
        },
        "kpoint": {
            "host_structure_id": str(host.id),
            "simulation_cell_structure_id": str(host.id),
            "default_value": [2, 2, 2],
            "method": "delta-threshold",
            "energy_threshold": 0.001,
            "force_threshold": 0.05,
            "point_labels": {"8.0": "Grid: 2 x 2 x 2"},
            "plot_data": {
                "cutoffs": [8.0],
                "energies": [-10.0],
                "forces": [0.05],
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
            },
        },
    }
    service.save_project(doc)

    reopened = MainWindow()
    assert reopened._open_project_from_path(root)

    select_route(reopened, "convergence")

    assert reopened._main_stack.currentWidget() is reopened._convergence_overview_view
    assert reopened._convergence_overview_view._cutoff_card._plot._placeholder.isHidden() is True
    assert reopened._convergence_overview_view._kpoint_card._plot._placeholder.isHidden() is True
    assert len(reopened._convergence_overview_view._cutoff_card._plot._figure.axes) == 2
    assert len(reopened._convergence_overview_view._kpoint_card._plot._figure.axes) == 2
    reopened.close()
    window.close()


def test_main_window_convergence_overview_hides_stale_snapshot_when_host_changes(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "overview_stale.mufinder", "OverviewStale")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.project.recommended_cutoff_energy = 550.0
    doc.project.recommended_cutoff_method = "asymptotic"
    doc.project.recommended_cutoff_energy_threshold = 0.001
    doc.project.recommended_cutoff_force_threshold = 0.05
    doc.ui_state[window._CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] = {
        "cutoff": {
            "host_structure_id": "different-host",
            "simulation_cell_structure_id": str(host.id),
            "default_value": 550.0,
            "method": "asymptotic",
            "energy_threshold": 0.001,
            "force_threshold": 0.05,
            "plot_data": {
                "cutoffs": [300.0, 400.0, 550.0],
                "energies": [-10.0, -10.1, -10.15],
                "forces": [0.3, 0.18, 0.05],
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
            },
        }
    }
    window._document = doc
    window._refresh_ui_after_project_change()

    select_route(window, "convergence")

    assert (
        window._convergence_overview_view._cutoff_card._plot._placeholder.text()
        == "No adopted cutoff analysis is pinned yet.\nOpen the analysis dialog in the child workflow and adopt a project default to pin it here."
    )
    assert window._convergence_overview_view._cutoff_card._plot._canvas.isHidden() is True
    window.close()


def _cutoff_window_with_completed_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run two default cutoff rows through the real flow and mark them completed."""

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "cutoff_rerun.mufinder", "CutoffRerun")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc
    window._refresh_ui_after_project_change()
    view = window._convergence_view
    for cutoff in (300.0, 360.0):
        view._add_row(cutoff)
    sent: list[list[str]] = []
    monkeypatch.setattr(
        window,
        "_submit_jobs_on_cluster",
        lambda jobs, **_kwargs: sent.append(
            [job.label for job in jobs if job.status == JobStatus.DEFINED]
        ),
    )
    view._run_cluster_selected()
    for job in doc.jobs.values():
        job.status = JobStatus.COMPLETED
    sent.clear()
    return window, doc, view, sent


def test_cutoff_rows_keep_their_completed_jobs_across_a_refresh_and_a_new_row(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: completed cutoff rows were submitted again after a refresh.

    A form state whose rows all use the default parameters was restored as if it
    were a legacy state, pinning each row to a task-only parameter set. The rows
    then no longer matched the definition their jobs ran under, so the next run
    registered a new definition, flipped them to ready and created new jobs.
    """

    window, doc, view, sent = _cutoff_window_with_completed_rows(tmp_path, monkeypatch)
    defaults = sorted(view.build_config().shared_params)

    window._refresh_ui_after_project_change()

    assert view._parameter_values == {}
    assert sorted(view.build_config().shared_params) == defaults
    status_column = view._STATUS_COLUMN
    assert [view._table.item(row, status_column).text() for row in range(2)] == [
        "completed",
        "completed",
    ]

    view._add_row(420.0)
    view._table.selectAll()
    view._run_cluster_selected()

    # Only the new row is submitted; the completed rows keep their jobs.
    assert sent == [["cutoff_convergence_ecut_420"]]
    assert sorted((job.label, job.status) for job in doc.jobs.values()) == [
        ("cutoff_convergence_ecut_300", JobStatus.COMPLETED),
        ("cutoff_convergence_ecut_360", JobStatus.COMPLETED),
        ("cutoff_convergence_ecut_420", JobStatus.DEFINED),
    ]
    assert [view._table.item(row, status_column).text() for row in range(2)] == [
        "completed",
        "completed",
    ]
    window.close()


def test_cutoff_rows_saved_with_task_only_parameters_are_read_back_as_defaults(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project saved while the restore bug was live is repaired on load."""

    window, doc, view, sent = _cutoff_window_with_completed_rows(tmp_path, monkeypatch)
    doc.ui_state["convergence_form"] = {
        **doc.ui_state["convergence_form"],
        "cutoff_parameter_rows": [
            {"cutoff": cutoff, "params": {"task": "SINGLEPOINT"}, "added_advanced": []}
            for cutoff in (300.0, 360.0)
        ],
    }

    window._refresh_ui_after_project_change()

    assert view._parameter_values == {}
    view._table.selectAll()
    view._run_cluster_selected()
    assert sent in ([], [[]])
    assert len(doc.jobs) == 2
    window.close()


def test_main_window_batch_row_payload_keeps_a_rows_own_untouched_mixed_values(main_window) -> None:
    from types import SimpleNamespace

    window = main_window
    definition = SimpleNamespace(
        parameters={
            "configuration": {
                "row_settings": {
                    "explicit": {
                        "parameter_values": {"max_scf_cycles": 80, "xc_functional": "LDA"},
                        "added_advanced": ["nextra_bands"],
                    },
                    # "derived" has no stored value for max_scf_cycles: it takes
                    # it from its source defaults and must keep doing so.
                    "derived": {"parameter_values": {"xc_functional": "LDA"}, "added_advanced": []},
                }
            }
        }
    )
    payload = {
        "parameter_values": {"xc_functional": "PBE"},
        "added_advanced": ["nextra_bands"],
        "num_farms": 2,
    }
    untouched = {"max_scf_cycles", "nextra_bands"}

    explicit = window._batch_row_payload(definition, "explicit", payload, untouched)
    derived = window._batch_row_payload(definition, "derived", payload, untouched)

    assert explicit == {
        "parameter_values": {"xc_functional": "PBE", "max_scf_cycles": 80},
        "added_advanced": ["nextra_bands"],
        "num_farms": 2,
    }
    assert derived == {
        "parameter_values": {"xc_functional": "PBE"},
        "added_advanced": [],
        "num_farms": 2,
    }
    # With nothing untouched the dialog's payload is stored as it is.
    assert window._batch_row_payload(definition, "explicit", payload, set()) is payload
    window.close()
