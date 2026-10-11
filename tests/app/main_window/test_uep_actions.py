"""UEP screening actions in the main window."""

from __future__ import annotations

from pathlib import Path


import mufinder.app.main_window as main_window_module
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectService


def test_materialize_uep_rewrites_defined_job_inputs_when_project_defaults_change(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_rewrite.mufinder", "UepRewrite")

    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.project.recommended_kpoint_grid = (4, 4, 4)
    doc.project.recommended_kpoint_lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    doc.ui_state["uep_form"] = {"selected_source_structure_id": str(host.id)}
    window._document = doc

    window._on_uep_configured(
        main_window_module.UepConfig(
            name_prefix="uep",
            rows=(main_window_module.UepSourceRow(label="row_1", parameter_values={}),),
        )
    )

    jobs = window._uep_controller.materialize(["row_1"])
    assert len(jobs) == 1
    job = jobs[0]
    assert job.status == JobStatus.DEFINED

    cell_path = doc.project_root / "files" / "uep" / job.label / f"{job.label}.cell"
    initial_cell = cell_path.read_text(encoding="utf-8")
    assert "KPOINT_MP_GRID 4 4 4" in initial_cell

    doc.project.recommended_kpoint_grid = (5, 5, 5)

    jobs = window._uep_controller.materialize(["row_1"])
    assert len(jobs) == 1
    assert jobs[0].id == job.id

    refreshed_cell = cell_path.read_text(encoding="utf-8")
    assert "KPOINT_MP_GRID 5 5 5" in refreshed_cell
    window.close()


def test_uep_preserves_running_row_across_reconfiguration(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_running.mufinder", "UepRunning")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    window._document = doc

    row = main_window_module.UepSourceRow(label="row_1", parameter_values={})
    old_definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_uep_config(main_window_module.UepConfig(name_prefix="uep", rows=(row,))),
        output_directory=doc.project_root / "files" / "uep",
    )
    running_job = service.add_job(
        doc,
        label="uep_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    running_job.status = JobStatus.RUNNING
    old_definition.parameters["job_ids_by_label"] = {"row_1": str(running_job.id)}

    new_definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_uep_config(main_window_module.UepConfig(name_prefix="uep", rows=(row,))),
        output_directory=doc.project_root / "files" / "uep",
    )
    doc.ui_state[window._ACTIVE_UEP_DEFINITION_KEY] = str(new_definition.id)

    states = window._uep_controller.build_states()

    assert states["row_1"].status == JobStatus.RUNNING.value
    assert states["row_1"].job_id == running_job.id
    assert new_definition.parameters["job_ids_by_label"]["row_1"] == str(running_job.id)
    window.close()


def test_uep_deleted_row_does_not_reinherit_stale_status(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_deleted_stale.mufinder", "UepDeletedStale")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    window._document = doc

    row = main_window_module.UepSourceRow(label="row_1", parameter_values={})
    old_definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_uep_config(main_window_module.UepConfig(name_prefix="uep", rows=(row,))),
        output_directory=doc.project_root / "files" / "uep",
    )
    running_job = service.add_job(
        doc,
        label="uep_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    running_job.status = JobStatus.RUNNING
    old_definition.parameters["job_ids_by_label"] = {"row_1": str(running_job.id)}

    window._on_remove_uep_rows(["row_1"])

    new_definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_uep_config(main_window_module.UepConfig(name_prefix="uep", rows=(row,))),
        output_directory=doc.project_root / "files" / "uep",
    )
    doc.ui_state[window._ACTIVE_UEP_DEFINITION_KEY] = str(new_definition.id)

    states = window._uep_controller.build_states()

    assert states["row_1"].status == "ready"
    assert new_definition.parameters.get("job_ids_by_label", {}) == {}
    assert old_definition.id not in doc.calculations
    window.close()
