"""k-point convergence actions in the main window."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.app.convergence_types import ConvergenceConfig
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectService
from tests.support.main_window import select_route


def test_main_window_kpoint_examiner_state_changed_updates_ui_state_without_saving_project(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "kpoint_examiner_state_update.mufinder", "KPointExaminerStateUpdate"
    )
    window._document = doc
    initial_state_text = doc.project_root.joinpath("state.json").read_text(encoding="utf-8")

    saves: list[bool] = []
    monkeypatch.setattr(window._project_service, "save_project", lambda _doc: saves.append(True))

    window._on_kpoint_examiner_state_changed({"results_text": "cached"})

    assert doc.ui_state[window._KPOINT_EXAMINER_STATE_KEY] == {"results_text": "cached"}
    assert doc.project_root.joinpath("state.json").read_text(encoding="utf-8") == initial_state_text
    assert saves == []
    window.close()


def test_main_window_kpoint_examiner_state_persists_across_project_reload_after_explicit_save(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    root = tmp_path / "kpoint_examiner_state_reload.mufinder"
    doc = service.create_project(root, "KPointExaminerStateReload")
    window._document = doc
    initial_state_text = doc.project_root.joinpath("state.json").read_text(encoding="utf-8")

    window._on_kpoint_examiner_state_changed(
        {
            "results_text": "Recommended grid: 2 x 2 x 2",
            "plot_data": {
                "cutoffs": [8.0, 27.0],
                "energies": [-10.0, -10.01],
                "forces": [0.10, 0.04],
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
            },
        }
    )

    assert doc.project_root.joinpath("state.json").read_text(encoding="utf-8") == initial_state_text

    service.save_project(doc)
    loaded = service.load_project(root)

    assert (
        loaded.ui_state[window._KPOINT_EXAMINER_STATE_KEY]["results_text"]
        == "Recommended grid: 2 x 2 x 2"
    )
    assert loaded.ui_state[window._KPOINT_EXAMINER_STATE_KEY]["plot_data"]["cutoffs"] == [8.0, 27.0]
    window.close()


def test_kpoint_convergence_configured_registers_snapshot(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "kpoint_ui.mufinder", "KPointUI")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    window._project_service.add_structure(window._document, host)
    window._document.project.host_structure_id = host.id

    config = ConvergenceConfig(
        sweep_parameter="kpoint_mp_grid",
        sweep_values=((2, 2, 2), (3, 3, 2)),
        shared_params={"task": "SINGLEPOINT", "xc_functional": "PBE"},
        overrides={},
        name_prefix="kconv",
    )

    window._on_kpoint_convergence_configured(config)

    assert len(window._document.jobs) == 0
    assert len(window._document.calculations) == 1
    definition = next(iter(window._document.calculations.values()))
    assert definition.workflow_stage == WorkflowStage.CONVERGENCE_KPOINT
    assert definition.parameters["configuration"]["sweep_parameter"] == "kpoint_mp_grid"
    assert definition.parameters["configuration"]["sweep_values"] == [[2, 2, 2], [3, 3, 2]]
    assert definition.parameters["job_ids_by_grid"] == {}
    window.close()


def test_refresh_kpoint_simulation_cell_without_document_disables_selector(main_window) -> None:
    window = main_window

    window._refresh_kpoint_convergence_simulation_cell()

    assert not window._kpoint_convergence_view._simulation_cell_combo.isEnabled()
    assert (
        "Load a host structure" in window._kpoint_convergence_view._simulation_cell_summary.text()
    )
    window.close()


def test_main_window_form_kpoint_shows_ready_before_submission(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_form_ready.mufinder", "KPointFormReady")
    doc.ui_state["kpoint_convergence_form"] = {
        "name_prefix": "kconv",
        "kpoint_grids": [[2, 2, 2], [3, 3, 2]],
    }

    window._document = doc
    window._refresh_ui_after_project_change()

    assert window._jobs_table.rowCount() == 0
    assert (
        window._kpoint_convergence_view._table.item(
            0, window._kpoint_convergence_view._STATUS_COLUMN
        ).text()
        == "ready"
    )
    assert (
        window._kpoint_convergence_view._table.item(
            1, window._kpoint_convergence_view._STATUS_COLUMN
        ).text()
        == "ready"
    )
    window.close()


def test_kpoint_convergence_preserves_running_grid_across_reconfiguration(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_running.mufinder", "KPointRunning")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    window._document = doc

    old_definition = service.register_convergence_kpoint_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2]],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_kpoint",
    )
    running_job = service.add_job(
        doc,
        label="kconv_kgrid_2x2x2",
        workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
        calc_type="castep_convergence_kpoint",
    )
    running_job.status = JobStatus.RUNNING
    old_definition.parameters["job_ids_by_grid"] = {
        main_window_module.MainWindow._grid_key((2, 2, 2)): str(running_job.id)
    }

    new_definition = service.register_convergence_kpoint_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2], [3, 3, 3]],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_kpoint",
    )
    doc.ui_state[window._ACTIVE_KPOINT_DEFINITION_KEY] = str(new_definition.id)

    states = window._kpoint_convergence_controller.build_states()

    assert states[(2, 2, 2)].status == JobStatus.RUNNING.value
    assert states[(2, 2, 2)].job_id == running_job.id
    assert new_definition.parameters["job_ids_by_grid"]["2x2x2"] == str(running_job.id)
    window.close()


def test_kpoint_deleted_grid_does_not_reinherit_stale_status(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_deleted_stale.mufinder", "KPointDeletedStale")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    window._document = doc

    old_definition = service.register_convergence_kpoint_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2]],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_kpoint",
    )
    running_job = service.add_job(
        doc,
        label="kconv_kgrid_2x2x2",
        workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
        calc_type="castep_convergence_kpoint",
    )
    running_job.status = JobStatus.RUNNING
    old_definition.parameters["job_ids_by_grid"] = {"2x2x2": str(running_job.id)}

    active_definition = service.register_convergence_kpoint_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2]],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_kpoint",
    )
    doc.ui_state[window._ACTIVE_KPOINT_DEFINITION_KEY] = str(active_definition.id)

    window._on_remove_grids_from_kpoint_convergence([(2, 2, 2)])

    new_definition = service.register_convergence_kpoint_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2]],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_kpoint",
    )
    doc.ui_state[window._ACTIVE_KPOINT_DEFINITION_KEY] = str(new_definition.id)

    states = window._kpoint_convergence_controller.build_states()

    assert states[(2, 2, 2)].status == "ready"
    assert new_definition.parameters.get("job_ids_by_grid", {}) == {}
    assert old_definition.parameters.get("job_ids_by_grid", {}) == {}
    window.close()


def test_main_window_submit_from_kpoint_dedupes_existing_ready_jobs(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_submit_dedupe.mufinder", "KPointSubmitDedupe")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    label_a = main_window_module.convergence_job_name(
        "kconv", (2, 2, 2), sweep_parameter="kpoint_mp_grid"
    )
    label_b = main_window_module.convergence_job_name(
        "kconv", (3, 3, 2), sweep_parameter="kpoint_mp_grid"
    )
    job_a = service.add_job(
        doc,
        label=label_a,
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_KPOINT,
        calc_type="castep_convergence_kpoint",
    )
    job_b = service.add_job(
        doc,
        label=label_b,
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_KPOINT,
        calc_type="castep_convergence_kpoint",
    )
    job_a.status = JobStatus.DEFINED
    job_b.status = JobStatus.DEFINED

    service.register_convergence_kpoint_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2], [3, 3, 2]],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_kpoint",
        job_ids_by_grid={
            main_window_module.MainWindow._grid_key((2, 2, 2)): str(job_a.id),
            main_window_module.MainWindow._grid_key((3, 3, 2)): str(job_b.id),
        },
    )
    window._document = doc

    submitted_ids: list[list[object]] = []
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda jobs: submitted_ids.append([job.id for job in jobs])
    )

    window._kpoint_convergence_controller.submit([(2, 2, 2), (2, 2, 2), (3, 3, 2)], target="local")

    assert submitted_ids == [[job_a.id, job_b.id]]
    window.close()


def test_main_window_remove_kpoint_grids_without_document_delegates_to_widget(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window

    removed: list[list[tuple[int, int, int]]] = []
    monkeypatch.setattr(
        window._kpoint_convergence_view, "remove_grids", lambda grids: removed.append(list(grids))
    )

    window._on_remove_grids_from_kpoint_convergence([(2, 2, 2)])

    assert removed == [[(2, 2, 2)]]
    window.close()


def test_main_window_remove_kpoint_grids_updates_definition_configuration(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_remove_rows.mufinder", "KPointRemoveRows")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    definition = service.register_convergence_kpoint_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2], [3, 3, 2]],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_kpoint",
        job_ids_by_grid={"2x2x2": "a", "3x3x2": "b"},
    )
    doc.ui_state[window._ACTIVE_KPOINT_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["kpoint_convergence_form"] = {
        "name_prefix": "kconv",
        "kpoint_grids": [[2, 2, 2], [3, 3, 2]],
    }

    window._document = doc
    window._refresh_ui_after_project_change()

    window._on_remove_grids_from_kpoint_convergence([(2, 2, 2)])

    configuration = definition.parameters["configuration"]
    assert configuration["sweep_values"] == [[3, 3, 2]]
    assert definition.parameters["job_ids_by_grid"] == {"3x3x2": "b"}
    assert window._kpoint_convergence_view._table.rowCount() == 1
    assert (
        window._kpoint_convergence_view._table.item(
            0, window._kpoint_convergence_view._KEY_COLUMN
        ).text()
        == "3 x 3 x 2"
    )
    window.close()


def test_main_window_examine_kpoint_convergence_launches_dialog(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_examine_launch.mufinder", "KPointExamineLaunch")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.ui_state[window._KPOINT_EXAMINER_STATE_KEY] = "invalid"
    window._document = doc

    states = {
        (2, 2, 2): main_window_module.SweepPointRunState(
            status="completed", total_energy=-10.0, max_force=0.2
        ),
        (3, 3, 3): main_window_module.SweepPointRunState(
            status="completed", total_energy=-10.02, max_force=0.15
        ),
    }
    monkeypatch.setattr(window._kpoint_convergence_controller, "build_states", lambda: states)

    captured: dict[str, object] = {}

    class _FakeDialog:
        def __init__(
            self,
            cutoff_data,
            *,
            num_atoms,
            sweep_kind,
            point_labels,
            lattice_vectors=None,
            initial_state=None,
            notes=(),
            parent=None,
        ) -> None:
            captured["cutoff_data"] = cutoff_data
            captured["num_atoms"] = num_atoms
            captured["notes"] = list(notes)
            captured["sweep_kind"] = sweep_kind
            captured["point_labels"] = point_labels
            captured["lattice_vectors"] = lattice_vectors
            captured["initial_state"] = initial_state
            captured["parent"] = parent
            self.default_adopted = SimpleNamespace(connect=lambda _cb: None)
            self.state_changed = SimpleNamespace(connect=lambda _cb: None)

        def exec(self) -> int:
            captured["opened"] = True
            return 0

        def state_snapshot(self) -> dict[str, object]:
            return {"cached": True}

    monkeypatch.setattr(main_window_module, "ConvergenceExaminerDialog", _FakeDialog)

    window._on_examine_kpoint_convergence()

    assert captured["cutoff_data"] == {8.0: states[(2, 2, 2)], 27.0: states[(3, 3, 3)]}
    assert captured["num_atoms"] == 2
    assert captured["sweep_kind"] == "kpoint"
    assert captured["point_labels"][8.0] == "Grid: 2 x 2 x 2"
    assert captured["lattice_vectors"] == host.lattice_vectors
    assert captured["initial_state"] == {}
    assert captured["parent"] is window
    assert captured["opened"] is True
    assert doc.ui_state[window._KPOINT_EXAMINER_STATE_KEY] == {"cached": True}
    window.close()


def test_main_window_on_kpoint_default_adopted_handles_unresolvable_label(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_adopt_warn.mufinder", "KPointAdoptWarn")
    window._document = doc

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._on_kpoint_default_adopted(8.0, "converged", 0.001, 0.05, {})

    assert warnings == ["Adopt failed"]
    window.close()


def test_main_window_on_kpoint_default_adopted_sets_project_default(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "kpoint_adopt_default.mufinder", "KPointAdoptDefault")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    definition = service.register_convergence_kpoint_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2]],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_kpoint",
    )
    doc.ui_state[window._ACTIVE_KPOINT_DEFINITION_KEY] = str(definition.id)
    doc.ui_state[window._KPOINT_EXAMINER_STATE_KEY] = {
        "plot_data": {
            "cutoffs": [8.0, 27.0],
            "energies": [-10.0, -10.01],
            "forces": [0.10, 0.04],
            "energy_threshold": 0.001,
            "force_threshold": 0.05,
        }
    }
    window._document = doc
    window._refresh_ui_after_project_change()

    select_route(window, "convergence_kpoint")
    assert window._main_stack.currentWidget() is window._kpoint_convergence_view

    window._on_kpoint_default_adopted(
        8.0,
        "converged",
        0.001,
        0.05,
        {8.0: "Grid: 2 x 2 x 2"},
    )

    assert doc.project.recommended_kpoint_grid == (2, 2, 2)
    assert doc.project.recommended_kpoint_method == "converged"
    assert doc.project.recommended_kpoint_energy_threshold == pytest.approx(0.001)
    assert doc.project.recommended_kpoint_force_threshold == pytest.approx(0.05)
    assert doc.project.recommended_kpoint_cell_id == str(host.id)
    # The grid is stored with the lattice it was converged on, so later
    # workflows can rescale it to their own cell.
    assert doc.project.recommended_kpoint_lattice == host.lattice_vectors
    for strip in window._project_defaults_strips():
        assert strip.summary().kpoint_grid == (2, 2, 2)
        assert strip._kpoint_value.text() == "2 x 2 x 2"
    snapshot = doc.ui_state[window._CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY]["kpoint"]
    assert snapshot["host_structure_id"] == str(host.id)
    assert snapshot["simulation_cell_structure_id"] == str(host.id)
    assert snapshot["default_value"] == [2, 2, 2]
    assert snapshot["point_labels"]["8.0"] == "Grid: 2 x 2 x 2"
    assert doc.provenance[-1].action == "set_recommended_kpoint_grid"
    assert "2x2x2" in window.statusBar().currentMessage()
    assert window._main_stack.currentWidget() is window._kpoint_convergence_view
    window.close()


def test_main_window_kpoint_examiner_and_adoption_use_the_sweep_supercell(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep run on a supercell is normalized and adopted against that supercell.

    The examiner's per-atom energies need the supercell's atom count and its
    spacings the supercell's lattice; the adopted grid must be stored with the
    supercell lattice, or later workflows would apply it unscaled to other cells.
    """

    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "kpoint_supercell_sweep.mufinder", "KPointSupercellSweep"
    )
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    supercell = Structure(
        label="2x1x1",
        lattice_vectors=[[10.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "Si", "Si"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.125, 0.25, 0.25],
            [0.5, 0.0, 0.0],
            [0.625, 0.25, 0.25],
        ],
        parent_structure_id=host.id,
    )
    service.add_structure(doc, supercell)
    definition = service.register_convergence_kpoint_definition(
        doc,
        input_structure_id=supercell.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[1, 2, 2]],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_kpoint",
    )
    doc.ui_state[window._ACTIVE_KPOINT_DEFINITION_KEY] = str(definition.id)
    window._document = doc

    states = {
        (1, 2, 2): main_window_module.SweepPointRunState(
            status="completed", total_energy=-20.0, max_force=0.2
        ),
        (2, 4, 4): main_window_module.SweepPointRunState(
            status="completed", total_energy=-20.02, max_force=0.1
        ),
    }
    monkeypatch.setattr(window._kpoint_convergence_controller, "build_states", lambda: states)
    captured: dict[str, object] = {}

    class _FakeDialog:
        def __init__(
            self,
            cutoff_data,
            *,
            num_atoms,
            sweep_kind,
            point_labels,
            lattice_vectors=None,
            initial_state=None,
            notes=(),
            parent=None,
        ) -> None:
            captured["num_atoms"] = num_atoms
            captured["lattice_vectors"] = lattice_vectors
            self.default_adopted = SimpleNamespace(connect=lambda _cb: None)
            self.state_changed = SimpleNamespace(connect=lambda _cb: None)

        def exec(self) -> int:
            return 0

        def state_snapshot(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(main_window_module, "ConvergenceExaminerDialog", _FakeDialog)
    window._on_examine_kpoint_convergence()

    assert captured["num_atoms"] == 4
    assert captured["lattice_vectors"] == supercell.lattice_vectors

    monkeypatch.setattr(window, "_confirm_adopted_xc_functional", lambda _functional: True)
    window._on_kpoint_default_adopted(16.0, "converged", 0.001, 0.05, {16.0: "Grid: 1 x 4 x 4"})

    assert doc.project.recommended_kpoint_grid == (1, 4, 4)
    assert doc.project.recommended_kpoint_cell_id == str(supercell.id)
    assert doc.project.recommended_kpoint_lattice == supercell.lattice_vectors
    window.close()


def test_kpoint_examiner_points_keep_one_grid_per_count_and_explain_the_rest() -> None:
    """Grids sharing a k-point count cannot share the examiner's axis."""

    lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 6.0]]
    completed = main_window_module.SweepPointRunState(
        status="completed", total_energy=-10.0, max_force=0.1
    )
    pending = main_window_module.SweepPointRunState(status="ready")
    states = {
        (2, 2, 4): completed,  # oversamples the long axis
        (4, 2, 2): completed,  # more uniform for this cell
        (2, 2, 2): completed,
        (2, 2, 1): pending,  # same count as (1, 2, 2) below, but has no result
        (4, 1, 1): completed,
    }

    counts, labels, notes = MainWindow._kpoint_examiner_points(states, lattice)

    assert labels[16.0] == "Grid: 4 x 2 x 2"
    assert counts[16.0] is states[(4, 2, 2)]
    assert labels[8.0] == "Grid: 2 x 2 x 2"
    # A completed grid is kept over a uniform one with nothing to examine.
    assert labels[4.0] == "Grid: 4 x 1 x 1"
    assert len(notes) == 2
    # Each note gives the reason that actually decided which grid was kept.
    assert any(
        note.startswith(
            "2 x 2 x 4 left out: same k-point count (16) as 4 x 2 x 2, which is examined instead "
            "because it samples the cell most uniformly."
        )
        for note in notes
    )
    assert any(
        note.startswith(
            "2 x 2 x 1 left out: same k-point count (4) as 4 x 1 x 1, which is examined instead "
            "because it is the only one with a completed result."
        )
        for note in notes
    )


def test_main_window_kpoint_overview_recomputes_spacings_on_the_sweep_cell(
    main_window, tmp_path: Path
) -> None:
    """The overview refreshes stored spacings from the cell the snapshot was swept on."""

    from mufinder.science.kpoint_grids import spacing_from_grid

    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "kpoint_overview_supercell.mufinder", "KPointOverviewSupercell"
    )
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    supercell = Structure(
        label="2x1x1",
        lattice_vectors=[[10.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        parent_structure_id=host.id,
    )
    service.add_structure(doc, supercell)
    doc.project.recommended_kpoint_grid = (4, 2, 2)
    doc.ui_state[window._CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] = {
        "kpoint": {
            "host_structure_id": str(host.id),
            "simulation_cell_structure_id": str(supercell.id),
            "default_value": [4, 2, 2],
            "point_labels": {"16.0": "Grid: 4 x 2 x 2"},
            "plot_data": {
                "cutoffs": [16.0],
                "energies": [-20.0],
                "point_spacings": {"16.0": 999.0},
            },
        }
    }
    window._document = doc

    snapshot = window._validated_kpoint_overview_snapshot()

    assert snapshot is not None
    assert snapshot["plot_data"]["point_spacings"][16.0] == pytest.approx(
        spacing_from_grid(supercell.lattice_vectors, (4, 2, 2))
    )
    assert snapshot["plot_data"]["point_spacings"][16.0] != pytest.approx(
        spacing_from_grid(host.lattice_vectors, (4, 2, 2))
    )
    window.close()
