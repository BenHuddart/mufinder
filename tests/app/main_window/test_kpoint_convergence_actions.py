"""k-point convergence actions in the main window."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.app.convergence_types import ConvergenceConfig
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Structure
from mufinder.services.convergence_defaults import (
    CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY,
    KPOINT_EXAMINER_STATE_KEY,
)
from mufinder.services.project_service import ProjectService
from tests.support.main_window import select_route


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


def _kpoint_adopt_window(window, tmp_path: Path):
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
    doc.ui_state[KPOINT_EXAMINER_STATE_KEY] = {
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
    return doc, host


def test_main_window_kpoint_examine_button_opens_the_examiner_and_adopts_through_it(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    doc, host = _kpoint_adopt_window(window, tmp_path)
    states = {
        (2, 2, 2): main_window_module.SweepPointRunState(
            status="completed", total_energy=-10.0, max_force=0.2
        ),
    }
    monkeypatch.setattr(window._kpoint_convergence_controller, "build_states", lambda: states)
    dialogs: list[SimpleNamespace] = []

    class _FakeDialog:
        def __init__(self, points, **kwargs) -> None:
            self.record = SimpleNamespace(points=points, kwargs=kwargs, adopted=[], opened=False)
            dialogs.append(self.record)
            self.default_adopted = SimpleNamespace(connect=self.record.adopted.append)
            self.state_changed = SimpleNamespace(connect=lambda _callback: None)

        def exec(self) -> int:
            self.record.opened = True
            return 0

        def state_snapshot(self) -> dict[str, object]:
            return {"cached": True}

    monkeypatch.setattr(main_window_module, "ConvergenceExaminerDialog", _FakeDialog)

    window._kpoint_convergence_view.examine_requested.emit()

    (dialog,) = dialogs
    assert dialog.points == {8.0: states[(2, 2, 2)]}
    initial_state = dialog.kwargs.pop("initial_state")
    assert initial_state["plot_data"]["cutoffs"] == [8.0, 27.0]
    assert dialog.kwargs == {
        "num_atoms": 1,
        "sweep_kind": "kpoint",
        "point_labels": {8.0: "Grid: 2 x 2 x 2"},
        "lattice_vectors": host.lattice_vectors,
        "notes": [],
        "parent": window,
    }
    assert dialog.opened is True
    assert doc.ui_state[KPOINT_EXAMINER_STATE_KEY] == {"cached": True}

    (adopted,) = dialog.adopted
    adopted(8.0, "converged", 0.001, 0.05, None)
    assert doc.project.recommended_kpoint_grid == (2, 2, 2)
    assert window.statusBar().currentMessage() == "Project default k-point grid set to 2x2x2"
    window.close()


def test_main_window_kpoint_adopt_failure_shows_a_warning(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    window._document = ProjectService().create_project(
        tmp_path / "kpoint_adopt_warn.mufinder", "KPointAdoptWarn"
    )
    warnings: list[tuple[object, str]] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda parent, title, _message: warnings.append((parent, title)),
    )

    window._kpoint_convergence_controller.adopt(8.0, "converged", 0.001, 0.05, point_labels={})

    assert warnings == [(window, "Adopt failed")]
    window.close()


def test_main_window_adopted_kpoint_grid_refreshes_the_window(main_window, tmp_path: Path) -> None:
    window = main_window
    doc, _host = _kpoint_adopt_window(window, tmp_path)
    select_route(window, "convergence_kpoint")
    assert window._main_stack.currentWidget() is window._kpoint_convergence_view

    window._kpoint_convergence_controller.adopt(
        8.0, "converged", 0.001, 0.05, point_labels={8.0: "Grid: 2 x 2 x 2"}
    )

    for strip in window._project_defaults_strips():
        assert strip.summary().kpoint_grid == (2, 2, 2)
        assert strip._kpoint_value.text() == "2 x 2 x 2"
    assert window._convergence_overview_view._kpoint_card._plot._placeholder.isHidden() is True
    loaded = ProjectService().load_project(doc.project_root)
    assert loaded.project.recommended_kpoint_grid == (2, 2, 2)
    assert CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY in loaded.ui_state
    assert "2x2x2" in window.statusBar().currentMessage()
    assert window._main_stack.currentWidget() is window._kpoint_convergence_view
    window.close()
